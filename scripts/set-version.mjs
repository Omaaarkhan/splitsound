#!/usr/bin/env node
// Stamps a release version into the app.
//
// The version in the repository is a development placeholder. The release
// workflow calls this before building, so the installers, the About panel and
// the updater all agree on the real version without CI committing to main.
//
// Usage: node scripts/set-version.mjs 26.47.0

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const version = process.argv[2];
if (!/^(0|[1-9]\d{0,2})\.(0|[1-9]\d{0,2})\.(0|[1-9]\d{0,4})$/.test(version ?? "")) {
  console.error("set-version: expected a version like 26.47.0");
  process.exit(2);
}
const [major, minor, patch] = version.split(".").map(Number);
if (major > 255 || minor > 255 || patch > 65535) {
  console.error("set-version: Windows installers need major and minor below 256 and patch below 65536");
  process.exit(2);
}

const path = fileURLToPath(new URL("../src-tauri/tauri.conf.json", import.meta.url));
const text = readFileSync(path, "utf8");
const config = JSON.parse(text);
if (typeof config.version !== "string") {
  console.error("set-version: src-tauri/tauri.conf.json has no version field to replace");
  process.exit(1);
}

// Replace only the top-level version so the file's formatting is untouched.
const pattern = /^(\s{2}"version"\s*:\s*")[^"]*(")/m;
if (!pattern.test(text)) {
  console.error("set-version: could not find the version line in src-tauri/tauri.conf.json");
  process.exit(1);
}
writeFileSync(path, text.replace(pattern, `$1${version}$2`));
console.log(`set-version: ${config.version} -> ${version}`);
