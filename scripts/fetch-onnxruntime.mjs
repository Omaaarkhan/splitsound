#!/usr/bin/env node
// Fetches Microsoft's official ONNX Runtime shared library for one target,
// verifies it, keeps it in vendor/onnxruntime/<target>/ and copies the library
// into src-tauri/resources/onnxruntime/ under the name the app loads.
//
//   node scripts/fetch-onnxruntime.mjs                      # this computer
//   node scripts/fetch-onnxruntime.mjs --target linux-x64
//   node scripts/fetch-onnxruntime.mjs --no-copy            # vendor/ only
//
// Targets: macos-arm64, linux-x64, linux-arm64, windows-x64, windows-arm64.
// (Microsoft no longer ships an Intel macOS build, so there is no macos-x64.)
//
// Where it comes from
// -------------------
// The library is taken from the Microsoft.ML.OnnxRuntime package on
// nuget.org, which Microsoft publishes and signs, and which carries the
// native libraries for every platform in one archive.
//
// What is checked
// ---------------
// * One version is pinned. It must satisfy the API level the engine builds
//   `ort` for (the `api-22` feature in crates/splitsound-core/Cargo.toml) and
//   is the version the engine's parity tests were run against.
// * The download must stay on HTTPS and on nuget.org's own hosts.
// * The package must be exactly PACKAGE.bytes long and match PACKAGE.sha512,
//   which is the hash NuGet itself publishes for this package
//   (`packageHash` in the registration entry linked below).
// * Each library taken out of the package must match its own pinned SHA-256.
// * No dependencies: Node built-ins only, including the zip reader.
//
// Updating
// --------
// Change VERSION, download the new package by hand, and replace the pins with
// freshly computed ones. Check PACKAGE.sha512 against the `packageHash` at
//   https://api.nuget.org/v3/registration5-gz-semver2/microsoft.ml.onnxruntime/<version>.json
// The `ort` crate must move with it (see .github/dependabot.yml).

import { createHash } from "node:crypto";
import {
  closeSync, copyFileSync, createWriteStream, existsSync, mkdirSync, openSync, readFileSync,
  readSync, renameSync, rmSync, statSync, writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { finished } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { inflateRawSync } from "node:zlib";

const VERSION = "1.32.0";

const PACKAGE = {
  url: `https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${VERSION}/microsoft.ml.onnxruntime.${VERSION}.nupkg`,
  bytes: 184391411,
  // Base64, exactly as NuGet publishes it.
  sha512: "fj2P9s5o2s5MvwwdDiCF4frwXUJed1gLh1wb0igCUnonbrvfoHMmAFtaj9atgYbBflcQtE3FQ6zGv/s75FQEsQ==",
};

const ALLOWED_HOSTS = new Set(["api.nuget.org", "globalcdn.nuget.org"]);

// entry: path inside the package. out: the name the app loads.
const TARGETS = {
  "macos-arm64": {
    entry: "runtimes/osx-arm64/native/libonnxruntime.dylib",
    out: "libonnxruntime.dylib",
    bytes: 47622768,
    sha256: "21ff304871400e846a30c8a1165d15173add56598afb45f98227e2dfc716a497",
  },
  "linux-x64": {
    entry: "runtimes/linux-x64/native/libonnxruntime.so",
    out: "libonnxruntime.so",
    bytes: 34824680,
    sha256: "4213d6d8c42ac4754865849cb139d31fc39127db6a326babf90bbc81a6eb52fc",
  },
  "linux-arm64": {
    entry: "runtimes/linux-arm64/native/libonnxruntime.so",
    out: "libonnxruntime.so",
    bytes: 27165936,
    sha256: "fd1db07c49cb1ad11370ef4393a405eef959fa9622fb5b2880f85d025757e719",
  },
  "windows-x64": {
    entry: "runtimes/win-x64/native/onnxruntime.dll",
    out: "onnxruntime.dll",
    bytes: 20602168,
    sha256: "348c3d29d6995eee37e3451825eaddb7ea7400be3afda6ca2286b12bc775f042",
  },
  "windows-arm64": {
    entry: "runtimes/win-arm64/native/onnxruntime.dll",
    out: "onnxruntime.dll",
    bytes: 19053920,
    sha256: "d9db6113549a3bd1eae55dbcb96bd95ee1f35c2823f39a42972664abab6517d2",
  },
};

// Licence texts that ship beside the library.
const NOTICES = {
  LICENSE: { bytes: 1094, sha256: "c250d6278f0b47a6439fb7592b08b58a55eb9f535aa49a1db63211c3f982b674" },
  "ThirdPartyNotices.txt": { bytes: 341120, sha256: "bf8073fd6a112ac2a656bb60799652127e881689e51d37ad20df446cd5188b2d" },
};

const root = fileURLToPath(new URL("..", import.meta.url));
const vendorDir = join(root, "vendor", "onnxruntime");
const resourceDir = join(root, "src-tauri", "resources", "onnxruntime");

function fail(message) {
  console.error(`fetch-onnxruntime: ${message}`);
  process.exit(1);
}

function hostTarget() {
  const os = { darwin: "macos", linux: "linux", win32: "windows" }[process.platform];
  const arch = { arm64: "arm64", x64: "x64" }[process.arch];
  return `${os}-${arch}`;
}

function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

/** Downloads the package to `path`, refusing anything that is not the pinned file. */
async function download(path) {
  let url = PACKAGE.url;
  let response;
  for (let hop = 0; ; hop += 1) {
    const { protocol, hostname } = new URL(url);
    if (protocol !== "https:" || !ALLOWED_HOSTS.has(hostname)) fail(`refusing to download from ${url}`);
    response = await fetch(url, { redirect: "manual" });
    if (response.status >= 300 && response.status < 400 && response.headers.has("location")) {
      if (hop >= 5) fail("too many redirects");
      url = new URL(response.headers.get("location"), url).href;
      continue;
    }
    break;
  }
  if (!response.ok || !response.body) fail(`download failed: HTTP ${response.status}`);

  const partial = `${path}.partial`;
  const file = createWriteStream(partial);
  const hash = createHash("sha512");
  let bytes = 0;
  for await (const chunk of response.body) {
    bytes += chunk.length;
    if (bytes > PACKAGE.bytes) {
      file.destroy();
      rmSync(partial, { force: true });
      fail(`the package is larger than the expected ${PACKAGE.bytes} bytes`);
    }
    hash.update(chunk);
    if (!file.write(chunk)) await new Promise((resume) => file.once("drain", resume));
  }
  file.end();
  await finished(file);

  const digest = hash.digest("base64");
  if (bytes !== PACKAGE.bytes || digest !== PACKAGE.sha512) {
    rmSync(partial, { force: true });
    fail(`the package does not match its pin (got ${bytes} bytes, sha512 ${digest})`);
  }
  renameSync(partial, path);
}

/** True if the cached package is the pinned one. */
function cachedPackageIsGood(path) {
  if (!existsSync(path) || statSync(path).size !== PACKAGE.bytes) return false;
  return createHash("sha512").update(readFileSync(path)).digest("base64") === PACKAGE.sha512;
}

/**
 * Reads named entries out of a zip file. Supports what a NuGet package uses:
 * stored and deflated entries, no zip64, no encryption. Sizes are taken from
 * the caller's pins, never trusted from the archive.
 */
function readZipEntries(path, wanted) {
  const fd = openSync(path, "r");
  try {
    const size = statSync(path).size;
    const read = (offset, length) => {
      const buffer = Buffer.alloc(length);
      if (readSync(fd, buffer, 0, length, offset) !== length) fail("unexpected end of package");
      return buffer;
    };

    // End-of-central-directory record: within the last 64 KiB + 22 bytes.
    const tailLength = Math.min(size, 65557);
    const tail = read(size - tailLength, tailLength);
    const eocd = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
    if (eocd < 0) fail("the package is not a zip archive");
    const entryCount = tail.readUInt16LE(eocd + 10);
    const directorySize = tail.readUInt32LE(eocd + 12);
    const directoryOffset = tail.readUInt32LE(eocd + 16);
    if (entryCount === 0xffff || directoryOffset === 0xffffffff) fail("zip64 packages are not supported");
    if (directoryOffset + directorySize > size) fail("the package's directory is out of bounds");

    const directory = read(directoryOffset, directorySize);
    const found = new Map();
    let at = 0;
    for (let i = 0; i < entryCount; i += 1) {
      if (at + 46 > directory.length || directory.readUInt32LE(at) !== 0x02014b50) fail("corrupt zip directory");
      const flags = directory.readUInt16LE(at + 8);
      const method = directory.readUInt16LE(at + 10);
      const compressedSize = directory.readUInt32LE(at + 20);
      const nameLength = directory.readUInt16LE(at + 28);
      const extraLength = directory.readUInt16LE(at + 30);
      const commentLength = directory.readUInt16LE(at + 32);
      const localOffset = directory.readUInt32LE(at + 42);
      const name = directory.toString("utf8", at + 46, at + 46 + nameLength);
      at += 46 + nameLength + extraLength + commentLength;
      if (!wanted.has(name)) continue;
      if (found.has(name)) fail(`the package lists ${name} twice`);
      if (flags & 0x1) fail(`${name} is encrypted`);
      found.set(name, { method, compressedSize, localOffset });
    }

    const out = new Map();
    for (const [name, pin] of wanted) {
      const entry = found.get(name);
      if (!entry) fail(`the package has no ${name}`);
      // Never inflate more than the pinned size, whatever the archive claims.
      if (entry.compressedSize > pin.bytes + 1024) fail(`${name} is larger than expected`);
      const local = read(entry.localOffset, 30);
      if (local.readUInt32LE(0) !== 0x04034b50) fail(`corrupt zip entry for ${name}`);
      const dataOffset = entry.localOffset + 30 + local.readUInt16LE(26) + local.readUInt16LE(28);
      if (dataOffset + entry.compressedSize > size) fail(`${name} runs past the end of the package`);
      const raw = read(dataOffset, entry.compressedSize);
      let data;
      if (entry.method === 0) data = raw;
      else if (entry.method === 8) data = inflateRawSync(raw, { maxOutputLength: pin.bytes });
      else fail(`${name} uses an unsupported compression method (${entry.method})`);
      if (data.length !== pin.bytes || sha256(data) !== pin.sha256) {
        fail(`${name} does not match its pin (got ${data.length} bytes, sha256 ${sha256(data)})`);
      }
      out.set(name, data);
    }
    return out;
  } finally {
    closeSync(fd);
  }
}

/* ------------------------------------------------------------------ */

const argv = process.argv.slice(2);
let target = hostTarget();
let copy = true;
for (let i = 0; i < argv.length; i += 1) {
  const arg = argv[i];
  if (arg === "--target") target = argv[++i];
  else if (arg.startsWith("--target=")) target = arg.slice("--target=".length);
  else if (arg === "--no-copy") copy = false;
  else if (arg === "--help" || arg === "-h") {
    console.log(`usage: fetch-onnxruntime.mjs [--target <${Object.keys(TARGETS).join(" | ")}>] [--no-copy]`);
    process.exit(0);
  } else fail(`unknown argument ${arg}`);
}

const pin = TARGETS[target];
if (!pin) fail(`unknown target "${target}". Targets: ${Object.keys(TARGETS).join(", ")}`);

const targetDir = join(vendorDir, target);
const libraryPath = join(targetDir, pin.out);

const haveLibrary =
  existsSync(libraryPath) && statSync(libraryPath).size === pin.bytes && sha256(readFileSync(libraryPath)) === pin.sha256;

if (haveLibrary) {
  console.log(`ONNX Runtime ${VERSION} for ${target} is already in vendor/ and matches its pin.`);
} else {
  const packagePath = join(vendorDir, "cache", `microsoft.ml.onnxruntime.${VERSION}.nupkg`);
  mkdirSync(dirname(packagePath), { recursive: true });
  if (cachedPackageIsGood(packagePath)) {
    console.log("Using the verified package already in vendor/onnxruntime/cache.");
  } else {
    console.log(`Downloading ONNX Runtime ${VERSION} from nuget.org (${(PACKAGE.bytes / 1_048_576).toFixed(0)} MB)...`);
    await download(packagePath);
    console.log("Package matches the hash NuGet publishes for it.");
  }

  const wanted = new Map([[pin.entry, pin], ...Object.entries(NOTICES)]);
  const entries = readZipEntries(packagePath, wanted);
  mkdirSync(targetDir, { recursive: true });
  // Write under a temporary name, then rename, so a half-written library is never picked up.
  const temporary = `${libraryPath}.partial`;
  writeFileSync(temporary, entries.get(pin.entry), { mode: 0o755 });
  renameSync(temporary, libraryPath);
  for (const name of Object.keys(NOTICES)) writeFileSync(join(targetDir, name), entries.get(name));
  writeFileSync(join(targetDir, "VERSION"), `${VERSION}\n`);
  console.log(`vendor/onnxruntime/${target}/${pin.out}: ${pin.bytes} bytes, sha256 ${pin.sha256}`);
}

if (copy) {
  mkdirSync(resourceDir, { recursive: true });
  copyFileSync(libraryPath, join(resourceDir, pin.out));
  console.log(`Copied to src-tauri/resources/onnxruntime/${pin.out}`);
}
