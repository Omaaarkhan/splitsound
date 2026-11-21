#!/usr/bin/env node
// Turns a folder of collected installers into a publishable release:
//
// 1. signs the files the in-app updater installs (if a signing key is in the
//    environment) and writes latest.json, the file the updater reads;
// 2. writes SHA256SUMS.txt covering every file in the folder.
//
// Used by the release workflow and by the CI rehearsal, so the two cannot drift.
//
// Usage:
//   node scripts/prepare-release.mjs --dir release --version 26.47.0 \
//     --repo owner/name --tag v26.47.0
//
// Environment:
//   TAURI_SIGNING_PRIVATE_KEY            the updater signing key (optional)
//   TAURI_SIGNING_PRIVATE_KEY_PASSWORD   its password
// Without the key nothing is signed and no latest.json is written; the
// installers can still be published, but the updater will not offer them.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), all[index + 1]]);
    return pairs;
  }, []),
);
for (const required of ["dir", "version", "repo", "tag"]) {
  if (!args[required]) {
    console.error(`prepare-release: missing --${required}`);
    process.exit(2);
  }
}

const root = fileURLToPath(new URL("..", import.meta.url));
const dir = args.dir;
const isUpdaterFile = (name) => /\.app\.tar\.gz$|\.AppImage$|-setup\.exe$/.test(name);

const installers = readdirSync(dir).filter((name) => !name.endsWith(".sig"));
if (installers.length === 0) {
  console.error(`prepare-release: ${dir} is empty`);
  process.exit(1);
}

if (process.env.TAURI_SIGNING_PRIVATE_KEY) {
  const cli = join(root, "node_modules", ".bin", process.platform === "win32" ? "tauri.cmd" : "tauri");
  if (!existsSync(cli)) {
    console.error("prepare-release: the Tauri CLI is not installed. Run: npm ci --ignore-scripts");
    process.exit(1);
  }
  for (const name of installers.filter(isUpdaterFile)) {
    // Binding the signature to the version stops an old build being replayed
    // as a new one: the app rejects a signature made for a different version.
    execFileSync(cli, ["signer", "sign", "--app-version", args.version, join(dir, name)], {
      stdio: ["ignore", "ignore", "inherit"],
    });
    if (!existsSync(join(dir, `${name}.sig`))) {
      console.error(`prepare-release: signing ${name} produced no signature`);
      process.exit(1);
    }
    console.log(`signed  ${name}`);
  }
  const manifest = execFileSync(
    process.execPath,
    [join(root, "scripts", "make-update-manifest.mjs"), "--dir", dir, "--version", args.version, "--repo", args.repo, "--tag", args.tag],
    { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] },
  );
  writeFileSync(join(dir, "latest.json"), manifest);
  console.log("wrote   latest.json");
} else {
  console.log("No updater signing key in the environment: nothing signed, no latest.json.");
}

const sums = readdirSync(dir)
  .filter((name) => name !== "SHA256SUMS.txt")
  .sort()
  .map((name) => `${createHash("sha256").update(readFileSync(join(dir, name))).digest("hex")}  ${name}`);
writeFileSync(join(dir, "SHA256SUMS.txt"), `${sums.join("\n")}\n`);
console.log(`wrote   SHA256SUMS.txt (${sums.length} files)`);
