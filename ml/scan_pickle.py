"""Static (no-execution) inspection of PyTorch checkpoint files.

    uv run python scan_pickle.py [--allow-package demucs] [--json out.json] FILE...

Nothing in the inspected file is ever unpickled, imported or executed. The file
is only read as bytes: a zip container is walked with `zipfile`, and each pickle
stream is disassembled with the standard library's `pickletools.genops`, which
decodes opcodes without running them.

For every pickle stream it reports
  * each global the pickle would import (GLOBAL / STACK_GLOBAL / INST opcodes),
  * how often the "call something" opcodes occur (REDUCE, BUILD, NEWOBJ, ...),
  * whether every global falls inside the project policy.

Policy (from SECURITY.md): a checkpoint may reference only `torch`, `numpy`,
`collections` and the model's own package (given with --allow-package).
Anything else is listed as OUTSIDE POLICY and the exit status is 2, so that the
caller stops instead of loading the file.

Standard library only, on purpose: this is the gate that runs before any
third-party loader touches a downloaded file.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickletools
import sys
import zipfile
from pathlib import Path

# Top-level modules a checkpoint is allowed to reference without review.
BASE_ALLOWED_MODULES = ("torch", "numpy", "collections")

# Opcodes that make the unpickler call or construct something. Counting them
# shows how much "behaviour" a pickle has; which callables they can reach is
# bounded by the list of globals.
CALL_OPCODES = ("REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX", "INST", "OBJ", "EXT1", "EXT2", "EXT4")

STRING_OPCODES = ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
                  "SHORT_BINSTRING", "BINSTRING", "STRING")
MEMO_PUT = ("MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT")
MEMO_GET = ("GET", "BINGET", "LONG_BINGET")


def scan_pickle_stream(data: bytes) -> list[dict]:
    """Disassemble every pickle found back to back in `data`.

    Returns one dict per pickle stream: {"globals": {...}, "opcodes": {...}}.
    """
    results = []
    stream = io.BytesIO(data)
    while stream.tell() < len(data):
        start = stream.tell()
        ops = []
        try:
            for opcode, arg, pos in pickletools.genops(stream):
                ops.append((opcode.name, arg, pos))
        except Exception as exc:  # trailing raw bytes (legacy format tensors) end the scan
            if not ops:
                break
            results.append(_summarise(ops, error=f"{type(exc).__name__}: {exc}"))
            break
        if not ops:
            break
        results.append(_summarise(ops))
        if stream.tell() == start:
            break
    return results


def _summarise(ops: list[tuple], error: str | None = None) -> dict:
    globals_found: dict[str, int] = {}
    opcode_counts: dict[str, int] = {}
    memo: dict[int, object] = {}
    next_memo = 0

    def add(module: str, name: str) -> None:
        key = f"{module}.{name}"
        globals_found[key] = globals_found.get(key, 0) + 1

    for index, (name, arg, _pos) in enumerate(ops):
        if name in CALL_OPCODES:
            opcode_counts[name] = opcode_counts.get(name, 0) + 1

        # Track the memo so STACK_GLOBAL arguments fetched with BINGET resolve.
        if name == "MEMOIZE":
            memo[next_memo] = ops[index - 1][1] if index else None
            next_memo += 1
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            memo[int(arg)] = ops[index - 1][1] if index else None
            next_memo = max(next_memo, int(arg) + 1)

        if name in ("GLOBAL", "INST"):
            module, _, attr = str(arg).partition(" ")
            add(module, attr)
        elif name == "STACK_GLOBAL":
            # The two most recent values pushed are (module, name). Walk back
            # over memo bookkeeping; anything that is not a plain string is
            # reported as unresolved, which fails the policy.
            values: list[str] = []
            for back in range(index - 1, -1, -1):
                bname, barg, _ = ops[back]
                if bname in MEMO_PUT:
                    continue
                if bname in MEMO_GET:
                    values.append(str(memo.get(int(barg), "<unresolved>")))
                elif bname in STRING_OPCODES:
                    values.append(str(barg))
                else:
                    values.append("<unresolved>")
                if len(values) == 2:
                    break
            while len(values) < 2:
                values.append("<unresolved>")
            add(values[1], values[0])

    out = {"globals": dict(sorted(globals_found.items())),
           "opcodes": dict(sorted(opcode_counts.items())),
           "num_opcodes": len(ops)}
    if error:
        out["error"] = error
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_file(path: Path) -> dict:
    """Scan one checkpoint. Never unpickles."""
    report: dict = {"file": path.name, "sizeBytes": path.stat().st_size,
                    "sha256": sha256_file(path), "pickles": []}
    if zipfile.is_zipfile(path):
        report["container"] = "zip (torch >= 1.6 format)"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            report["zipEntries"] = len(names)
            # Every member that is not a raw tensor storage is listed, so a
            # stray script or second pickle cannot hide in the archive.
            other = [n for n in names if "/data/" not in n]
            report["nonStorageEntries"] = other
            for member in names:
                if member.endswith(".pkl") or member.endswith(".pickle"):
                    for result in scan_pickle_stream(archive.read(member)):
                        result["member"] = member
                        report["pickles"].append(result)
    else:
        report["container"] = "raw pickle stream (legacy torch format or plain pickle)"
        for result in scan_pickle_stream(path.read_bytes()):
            result["member"] = "<file>"
            report["pickles"].append(result)
    return report


def outside_policy(report: dict, allowed_modules: tuple[str, ...]) -> list[str]:
    bad = []
    for pickle_report in report["pickles"]:
        for name in pickle_report["globals"]:
            top = name.split(".", 1)[0]
            if top not in allowed_modules:
                bad.append(name)
        if "error" in pickle_report:
            bad.append(f"<scan error: {pickle_report['error']}>")
    if not report["pickles"]:
        bad.append("<no pickle stream found>")
    return sorted(set(bad))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--allow-package", action="append", default=[],
                        help="the model's own top-level package (repeatable), e.g. demucs")
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()

    allowed = BASE_ALLOWED_MODULES + tuple(args.allow_package)
    reports, failed = [], False
    for path in args.files:
        report = scan_file(path)
        report["allowedModules"] = list(allowed)
        report["outsidePolicy"] = outside_policy(report, allowed)
        reports.append(report)

        print(f"{report['file']}  {report['sizeBytes']} bytes  sha256={report['sha256']}")
        print(f"  container: {report['container']}")
        if "nonStorageEntries" in report:
            print(f"  zip entries: {report['zipEntries']}, non-storage: {report['nonStorageEntries']}")
        for pickle_report in report["pickles"]:
            print(f"  pickle {pickle_report['member']}: {pickle_report['num_opcodes']} opcodes, "
                  f"calls {pickle_report['opcodes']}")
            for name, count in pickle_report["globals"].items():
                flag = "" if name.split(".", 1)[0] in allowed else "   <-- OUTSIDE POLICY"
                print(f"    global {name}  x{count}{flag}")
        if report["outsidePolicy"]:
            failed = True
            print(f"  RESULT: OUTSIDE POLICY -> {report['outsidePolicy']}")
        else:
            print("  RESULT: ok (all globals within policy)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
