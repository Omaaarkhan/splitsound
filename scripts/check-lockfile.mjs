#!/usr/bin/env node
// Fails if package-lock.json could pull code from anywhere but the npm
// registry, or lists a package without an integrity hash.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const lock = JSON.parse(readFileSync(fileURLToPath(new URL("../package-lock.json", import.meta.url)), "utf8"));
const problems = [];

for (const [path, entry] of Object.entries(lock.packages ?? {})) {
  if (path === "" || entry.link) continue; // the project itself, or a workspace link
  if (!entry.resolved) {
    problems.push(`${path}: no download location recorded`);
    continue;
  }
  if (!entry.resolved.startsWith("https://registry.npmjs.org/")) {
    problems.push(`${path}: comes from ${entry.resolved}, not the npm registry`);
  }
  if (!/^sha512-[A-Za-z0-9+/]+={0,2}$/.test(entry.integrity ?? "")) {
    problems.push(`${path}: missing or weak integrity hash`);
  }
}

if (problems.length > 0) {
  console.error(`package-lock.json has ${problems.length} problem(s):\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log(`package-lock.json: ${Object.keys(lock.packages).length - 1} packages, all from the npm registry with sha512 integrity.`);
