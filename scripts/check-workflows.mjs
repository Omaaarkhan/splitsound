#!/usr/bin/env node
// Guards the GitHub Actions workflows against the common supply-chain mistakes:
//
// - every third-party action must be pinned to a full commit SHA (a tag can be
//   moved to different code; a commit cannot),
// - every workflow must set default token permissions, and may only default to
//   read access,
// - `pull_request_target` is not allowed (it runs with secrets on untrusted code).

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const dir = fileURLToPath(new URL("../.github/workflows", import.meta.url));
const problems = [];

for (const file of readdirSync(dir).filter((name) => /\.ya?ml$/.test(name))) {
  const text = readFileSync(join(dir, file), "utf8");
  const lines = text.split("\n");

  lines.forEach((line, index) => {
    const uses = line.match(/^\s*(?:-\s*)?uses:\s*["']?([^"'\s#]+)/);
    if (!uses) return;
    const ref = uses[1];
    if (ref.startsWith("./")) return; // a workflow or action in this repository
    if (!/@[0-9a-f]{40}$/.test(ref)) {
      problems.push(`${file}:${index + 1}: "${ref}" is not pinned to a full commit SHA`);
    }
  });

  if (/^\s*pull_request_target\s*:/m.test(text) || /^on:\s*\[?[^\n]*pull_request_target/m.test(text)) {
    problems.push(`${file}: uses pull_request_target`);
  }

  const top = text.match(/^permissions:\s*(.*)\n((?:[ \t]+.*\n?)*)/m);
  if (!top) {
    problems.push(`${file}: no top-level "permissions:" block`);
  } else {
    const granted = `${top[1]}\n${top[2]}`;
    if (/write-all/.test(granted) || /:\s*write\b/.test(granted)) {
      problems.push(`${file}: top-level permissions grant write access; grant it per job instead`);
    }
  }
}

if (problems.length > 0) {
  console.error(`Workflow problems:\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log("Workflows: every action is pinned to a commit and default permissions are read-only.");
