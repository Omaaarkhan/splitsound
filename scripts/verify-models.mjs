#!/usr/bin/env node
// Checks that every model the app pins can be obtained and is exactly the file
// the manifest describes: the compressed download matches its size and
// SHA-256, and so does the ONNX file inside it.
//
// By default the files are fetched from the models release of this repository
// with the GitHub CLI (works for private repositories too). Pass --dir to check
// files already on disk instead, for example ml/dist after an export.
//
// Usage:
//   node scripts/verify-models.mjs
//   node scripts/verify-models.mjs --dir ml/dist [--manifest ml/dist/manifest.json]
//   node scripts/verify-models.mjs --release models-v1 --repo owner/name

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { createZstdDecompress } from "node:zlib";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), all[index + 1]]);
    return pairs;
  }, []),
);

const manifestPath =
  args.manifest ?? fileURLToPath(new URL("../crates/splitsound-core/models/manifest.json", import.meta.url));
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const release = args.release ?? manifest.release ?? "models-v1";
const repo = args.repo ?? process.env.GITHUB_REPOSITORY ?? "Omaaarkhan/splitsound";

/** Size and SHA-256 of a stream, read to the end. */
async function measure(stream) {
  const hash = createHash("sha256");
  let bytes = 0;
  for await (const chunk of stream) {
    hash.update(chunk);
    bytes += chunk.length;
  }
  return { bytes, sha256: hash.digest("hex") };
}

/** Like `measure`, but for the zstd-decompressed contents, giving up past `limit` bytes. */
async function measureDecompressed(path, limit) {
  const hash = createHash("sha256");
  let bytes = 0;
  await pipeline(createReadStream(path), createZstdDecompress(), async (source) => {
    for await (const chunk of source) {
      bytes += chunk.length;
      if (bytes > limit) throw new Error(`decompresses to more than the expected ${limit} bytes`);
      hash.update(chunk);
    }
  });
  return { bytes, sha256: hash.digest("hex") };
}

const scratch = args.dir ? null : mkdtempSync(join(tmpdir(), "splitsound-models-"));
let failures = 0;

for (const model of manifest.models) {
  const fail = (reason) => {
    failures += 1;
    console.error(`FAIL  ${model.id}: ${reason}`);
  };
  // What we publish for each model is the zstd-compressed graph. (For the
  // restricted Demucs model that graph holds no weights; the weights are
  // fetched from their publisher by the app and are not ours to host.)
  const download = model.acquire;
  if (download?.type !== "download") {
    fail(`unsupported acquire type ${JSON.stringify(download?.type)}`);
    continue;
  }
  const archiveName = download.file;
  let archivePath;

  if (args.dir) {
    archivePath = join(args.dir, archiveName);
    if (!existsSync(archivePath)) {
      fail(`${archivePath} does not exist`);
      continue;
    }
  } else {
    try {
      execFileSync("gh", ["release", "download", release, "--repo", repo, "--pattern", archiveName, "--dir", scratch], {
        stdio: ["ignore", "ignore", "pipe"],
      });
    } catch (error) {
      fail(`could not download ${archiveName} from release ${release} of ${repo}: ${String(error.stderr ?? error).trim()}`);
      continue;
    }
    archivePath = join(scratch, archiveName);
  }

  const archive = await measure(createReadStream(archivePath));
  if (archive.bytes !== download.sizeBytes) {
    fail(`${archiveName} is ${archive.bytes} bytes, manifest says ${download.sizeBytes}`);
  } else if (archive.sha256 !== download.sha256) {
    fail(`${archiveName} has SHA-256 ${archive.sha256}, manifest says ${download.sha256}`);
  } else {
    try {
      const inner = await measureDecompressed(archivePath, model.sizeBytes);
      if (inner.bytes !== model.sizeBytes) {
        fail(`${model.file} is ${inner.bytes} bytes once decompressed, manifest says ${model.sizeBytes}`);
      } else if (inner.sha256 !== model.sha256) {
        fail(`${model.file} has SHA-256 ${inner.sha256}, manifest says ${model.sha256}`);
      } else {
        const mb = (statSync(archivePath).size / 1_048_576).toFixed(1);
        console.log(`ok    ${model.id}  ${mb} MB download, ${(model.sizeBytes / 1_048_576).toFixed(1)} MB installed`);
      }
    } catch (error) {
      fail(`${archiveName}: ${error.message}`);
    }
  }

  if (scratch) rmSync(archivePath, { force: true });
}

if (scratch) rmSync(scratch, { recursive: true, force: true });
if (failures > 0) {
  console.error(`\n${failures} of ${manifest.models.length} models failed verification.`);
  process.exit(1);
}
console.log(`\nAll ${manifest.models.length} models match the manifest.`);
