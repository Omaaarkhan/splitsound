#!/usr/bin/env node
// Gathers what `tauri build` produced into one folder under stable names:
//
//   SplitSound_<version>_macos-arm64.dmg              installer
//   SplitSound_<version>_macos-arm64.app.tar.gz       what the updater installs
//   SplitSound_<version>_linux-x64.AppImage           installer and updater file
//   SplitSound_<version>_linux-x64.deb / .rpm         installers
//   SplitSound_<version>_windows-x64-setup.exe        installer and updater file
//
// Usage:
//   node scripts/collect-artifacts.mjs --bundle-dir target/release/bundle \
//     --platform linux-x64 --out release-artifacts

import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, value, index, all) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), all[index + 1]]);
    return pairs;
  }, []),
);
const platforms = ["macos-arm64", "linux-x64", "windows-x64"];
if (!args["bundle-dir"] || !args.out || !platforms.includes(args.platform)) {
  console.error(`collect-artifacts: need --bundle-dir, --out and --platform (${platforms.join(" | ")})`);
  process.exit(2);
}

const config = JSON.parse(
  readFileSync(fileURLToPath(new URL("../src-tauri/tauri.conf.json", import.meta.url)), "utf8"),
);
const product = config.productName;
const base = `${product}_${config.version}_${args.platform}`;
const bundleDir = resolve(args["bundle-dir"]);
const outDir = resolve(args.out);
mkdirSync(outDir, { recursive: true });

/** The single file in `dir` whose name ends with `suffix`. */
function only(dir, suffix) {
  const path = join(bundleDir, dir);
  const found = existsSync(path) ? readdirSync(path).filter((name) => name.endsWith(suffix)) : [];
  if (found.length !== 1) {
    console.error(`collect-artifacts: expected exactly one *${suffix} in ${path}, found ${found.length}`);
    process.exit(1);
  }
  return join(path, found[0]);
}

function keep(source, name) {
  const target = join(outDir, name);
  copyFileSync(source, target);
  console.log(`${name}  ${(statSync(target).size / 1_048_576).toFixed(1)} MB`);
}

if (args.platform === "macos-arm64") {
  keep(only("dmg", ".dmg"), `${base}.dmg`);
  const app = only("macos", ".app");
  const archive = join(outDir, `${base}.app.tar.gz`);
  // Plain tar of the .app, without macOS resource-fork entries, which would
  // otherwise land inside the bundle on extraction and break its signature.
  execFileSync("tar", ["--no-xattrs", "-czf", archive, "-C", join(bundleDir, "macos"), app.split("/").pop()], {
    stdio: "inherit",
    env: { ...process.env, COPYFILE_DISABLE: "1" },
  });
  console.log(`${base}.app.tar.gz  ${(statSync(archive).size / 1_048_576).toFixed(1)} MB`);
} else if (args.platform === "linux-x64") {
  keep(only("appimage", ".AppImage"), `${base}.AppImage`);
  keep(only("deb", ".deb"), `${base}.deb`);
  keep(only("rpm", ".rpm"), `${base}.rpm`);
} else {
  keep(only("nsis", "-setup.exe"), `${base}-setup.exe`);
}
