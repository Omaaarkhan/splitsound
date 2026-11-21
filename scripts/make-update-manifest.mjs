#!/usr/bin/env node
// Writes latest.json, the file the in-app updater reads to learn about a release.
//
// It pairs each updater artifact in <dir> with its detached signature
// (<artifact>.sig, made by `tauri signer sign`) and the public download URL
// that artifact will have once attached to the GitHub release.
//
// Usage:
//   node scripts/make-update-manifest.mjs --dir release --version 26.47.0 \
//     --repo owner/name --tag v26.47.0 [--notes "text"] > release/latest.json

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), all[index + 1]]);
    return pairs;
  }, []),
);
for (const required of ["dir", "version", "repo", "tag"]) {
  if (!args[required]) {
    console.error(`make-update-manifest: missing --${required}`);
    process.exit(2);
  }
}
if (!/^[\w.-]+\/[\w.-]+$/.test(args.repo) || !/^v[\d.]+$/.test(args.tag)) {
  console.error("make-update-manifest: --repo must be owner/name and --tag must look like v26.47.0");
  process.exit(2);
}

// Which artifact updates which platform. Keys are the names the updater looks
// up: "<os>-<arch>" and, on newer versions, "<os>-<arch>-<installer>".
const targets = [
  { match: /_macos-arm64\.app\.tar\.gz$/, keys: ["darwin-aarch64", "darwin-aarch64-app"] },
  { match: /_linux-x64\.AppImage$/, keys: ["linux-x86_64", "linux-x86_64-appimage"] },
  { match: /_windows-x64-setup\.exe$/, keys: ["windows-x86_64", "windows-x86_64-nsis"] },
];

const files = readdirSync(args.dir);
const platforms = {};
for (const target of targets) {
  const artifact = files.find((name) => target.match.test(name));
  if (!artifact) {
    console.error(`make-update-manifest: no artifact matches ${target.match}; that platform will not auto-update`);
    continue;
  }
  if (!files.includes(`${artifact}.sig`)) {
    console.error(`make-update-manifest: ${artifact} has no signature; refusing to list it`);
    process.exit(1);
  }
  const entry = {
    signature: readFileSync(join(args.dir, `${artifact}.sig`), "utf8").trim(),
    url: `https://github.com/${args.repo}/releases/download/${args.tag}/${encodeURIComponent(artifact)}`,
  };
  for (const key of target.keys) platforms[key] = entry;
}

if (Object.keys(platforms).length === 0) {
  console.error("make-update-manifest: nothing to list");
  process.exit(1);
}

process.stdout.write(
  `${JSON.stringify(
    {
      version: args.version,
      notes: args.notes ?? `See https://github.com/${args.repo}/releases/tag/${args.tag}`,
      pub_date: new Date().toISOString(),
      platforms,
    },
    null,
    2,
  )}\n`,
);
