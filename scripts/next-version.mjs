#!/usr/bin/env node
// Prints the version the next release should carry.
//
// Releases are weekly, so versions are calendar based: YY.WW.N
//   YY  ISO week-numbering year, two digits   (26)
//   WW  ISO week, no leading zero             (47)
//   N   build within that week, from 0        (0)
// That is valid semver, sorts correctly across years, and fits the Windows
// installer limits (each part below 256 / 256 / 65536).
//
// Usage: node scripts/next-version.mjs [--date 2026-11-20]

import { execFileSync } from "node:child_process";

function isoWeek(date) {
  // Thursday of the current week decides the ISO year.
  const d = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
  const day = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - day);
  const yearStart = Date.UTC(d.getUTCFullYear(), 0, 1);
  const week = Math.ceil(((d.getTime() - yearStart) / 86_400_000 + 1) / 7);
  return { year: d.getUTCFullYear(), week };
}

const dateArg = process.argv.indexOf("--date");
const now = dateArg > 0 ? new Date(`${process.argv[dateArg + 1]}T12:00:00Z`) : new Date();
if (Number.isNaN(now.getTime())) {
  console.error("next-version: --date must look like 2026-11-20");
  process.exit(2);
}

const { year, week } = isoWeek(now);
const prefix = `${year % 100}.${week}.`;

let tags = [];
try {
  tags = execFileSync("git", ["tag", "--list", `v${prefix}*`], { encoding: "utf8" })
    .split("\n")
    .filter(Boolean);
} catch {
  // Not a git checkout, or no tags yet: start the week at 0.
}

const next = tags
  .map((tag) => tag.slice(1 + prefix.length))
  .filter((rest) => /^\d+$/.test(rest))
  .reduce((highest, rest) => Math.max(highest, Number(rest) + 1), 0);

console.log(`${prefix}${next}`);
