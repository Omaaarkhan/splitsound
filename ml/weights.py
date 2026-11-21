"""Pinned download, verification and safe loading of third-party model files.

    uv run --locked python weights.py fetch            # download what is missing, verify everything
    uv run --locked python weights.py verify           # verify what is on disk, download nothing
    uv run --locked python weights.py fetch --only htdemucs_6s

Every file is described in `weights.lock.json` (URL, size, full SHA-256, the
set of globals its pickle references). The order of operations for a file is
always:

  1. download over HTTPS from the publisher's own host (each redirect hop is
     checked against `allowedHosts`; plain HTTP is refused);
  2. compare size and full SHA-256 with the lock, and the publisher's own
     checksum where one exists (Demucs file names embed a SHA-256 prefix);
  3. statically scan the pickle with `scan_pickle.py` (no unpickling) and
     require the globals to equal `expectedGlobals` exactly, with everything
     outside the base policy listed in `approvedExceptions`;
  4. only then may `load_*` deserialise it, and only with
     `torch.load(weights_only=True)` plus an explicit allow-list.

Steps 1-3 never import torch and never execute anything from the file.
A JSON record of what was fetched goes to `dist/security/downloads.json`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
# Sibling modules are imported by explicit path so the script also works under
# `python -I` (isolated mode: no current directory, no user site on sys.path).
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import scan_pickle  # noqa: E402

LOCK_PATH = ML_DIR / "weights.lock.json"
CACHE_DIR = ML_DIR / ".cache" / "weights"
EVIDENCE_PATH = ML_DIR / "dist" / "security" / "downloads.json"


class WeightsError(RuntimeError):
    """A provenance, integrity or policy check failed. Never catch and continue."""


# ----------------------------------------------------------------------------
# lock file
# ----------------------------------------------------------------------------

def read_lock() -> dict:
    return json.loads(LOCK_PATH.read_text())


def lock_entry(model_id: str) -> dict:
    for entry in read_lock()["files"]:
        if model_id in entry["usedBy"] or entry["name"] == model_id:
            return entry
    raise WeightsError(f"{model_id!r} is not in {LOCK_PATH.name}")


def local_path(entry: dict) -> Path:
    if "cachePath" in entry:                       # source files keep their package layout
        return CACHE_DIR.parent / entry["cachePath"]
    return CACHE_DIR / entry["name"]


# ----------------------------------------------------------------------------
# download (HTTPS only, allow-listed hosts, every redirect hop checked)
# ----------------------------------------------------------------------------

def _check_url(url: str, allowed_hosts: dict) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise WeightsError(f"refusing non-HTTPS URL: {url}")
    if parts.hostname not in allowed_hosts:
        raise WeightsError(f"refusing host {parts.hostname!r} (not in allowedHosts): {url}")
    if parts.username or parts.password:
        raise WeightsError(f"refusing URL with credentials: {url}")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: dict, chain: list):
        self.allowed_hosts = allowed_hosts
        self.chain = chain

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl, self.allowed_hosts)
        # Record host and path only: CDN redirect URLs carry short-lived signed
        # query strings that are not worth keeping.
        parts = urllib.parse.urlsplit(newurl)
        self.chain.append(f"{code} -> https://{parts.hostname}{parts.path}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(entry: dict, allowed_hosts: dict) -> dict:
    """Download one file to the cache. Returns transport evidence. Verifies nothing else."""
    url = entry["url"]
    # A file may bring its own extra hosts (with the reason recorded next to it in the lock);
    # they apply to that file only.
    allowed_hosts = {**allowed_hosts, **entry.get("extraAllowedHosts", {})}
    _check_url(url, allowed_hosts)
    chain: list[str] = []
    opener = urllib.request.build_opener(_CheckedRedirects(allowed_hosts, chain))  # TLS verification is on by default
    destination = local_path(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    size = 0
    with opener.open(url, timeout=120) as response:
        headers = {k.lower(): v for k, v in response.headers.items()}
        with open(partial, "wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
                digest.update(block)
                size += len(block)
                if size > entry["sizeBytes"]:
                    out.close()
                    partial.unlink()
                    raise WeightsError(f"{entry['name']}: download exceeds pinned size {entry['sizeBytes']}")
    if size != entry["sizeBytes"] or digest.hexdigest() != entry["sha256"]:
        partial.unlink()
        raise WeightsError(f"{entry['name']}: downloaded {size} bytes sha256={digest.hexdigest()}, "
                           f"lock says {entry['sizeBytes']} bytes sha256={entry['sha256']}")
    os.replace(partial, destination)
    return {
        "downloadedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "redirects": chain,
        "httpLastModified": headers.get("last-modified"),
        "httpEtag": headers.get("etag"),
        "httpContentLength": headers.get("content-length"),
        "httpServer": headers.get("server"),
    }


# ----------------------------------------------------------------------------
# verification (integrity + static pickle policy). No torch, no unpickling.
# ----------------------------------------------------------------------------

def verify(entry: dict) -> dict:
    """Check one cached file against the lock. Raises WeightsError on any mismatch."""
    path = local_path(entry)
    if not path.exists():
        raise WeightsError(f"{entry['name']} is not downloaded; run `weights.py fetch`")
    size = path.stat().st_size
    sha256 = scan_pickle.sha256_file(path)
    if size != entry["sizeBytes"] or sha256 != entry["sha256"]:
        raise WeightsError(f"{entry['name']}: on disk {size} bytes sha256={sha256}, "
                           f"lock says {entry['sizeBytes']} bytes sha256={entry['sha256']}")
    result = {"name": entry["name"], "sizeBytes": size, "sha256": sha256}

    publisher = entry.get("publisherChecksum")
    if publisher is None:
        result["publisherChecksum"] = "none published. " + entry.get("publisherChecksumNote", "")
    elif publisher["type"] == "sha256-prefix":
        if not sha256.startswith(publisher["value"]) or f"-{publisher['value']}." not in entry["name"]:
            raise WeightsError(f"{entry['name']}: publisher checksum prefix {publisher['value']} does not match")
        result["publisherChecksum"] = f"sha256 prefix {publisher['value']} (from the file name) matches"
    else:
        raise WeightsError(f"unknown publisherChecksum type {publisher['type']!r}")

    policy = entry.get("pickle")
    if policy is not None:
        report = scan_pickle.scan_file(path)
        found = sorted({name for p in report["pickles"] for name in p["globals"]})
        errors = [p["error"] for p in report["pickles"] if "error" in p]
        if errors or not report["pickles"]:
            raise WeightsError(f"{entry['name']}: pickle scan failed: {errors or 'no pickle found'}")
        if found != sorted(policy["expectedGlobals"]):
            raise WeightsError(f"{entry['name']}: pickle globals changed.\n  found    {found}\n"
                               f"  expected {sorted(policy['expectedGlobals'])}")
        allowed = scan_pickle.BASE_ALLOWED_MODULES + ((policy["modelPackage"],) if policy.get("modelPackage") else ())
        outside = [name for name in found if name.split(".", 1)[0] not in allowed]
        unapproved = [name for name in outside if name not in policy.get("approvedExceptions", [])]
        result["pickleGlobals"] = found
        result["outsideBasePolicy"] = outside
        result["unapproved"] = unapproved
        result["zipNonStorageEntries"] = report.get("nonStorageEntries")
        result["pickleCallOpcodes"] = [p["opcodes"] for p in report["pickles"]]
    return result


def require_loadable(entry: dict) -> dict:
    """verify() plus the owner-approval gate. Call this immediately before deserialising."""
    result = verify(entry)
    if result.get("unapproved"):
        raise WeightsError(
            f"{entry['name']}: refusing to load. The pickle references globals outside the base policy "
            f"(torch, numpy, collections, model package) that the project owner has not approved: "
            f"{result['unapproved']}. See SECURITY.md; approval is recorded in weights.lock.json "
            f"under pickle.approvedExceptions.")
    return result


# ----------------------------------------------------------------------------
# safe loading (imports torch lazily; weights_only=True, explicit allow-list)
# ----------------------------------------------------------------------------

# Globals of Meta's other Demucs checkpoints (htdemucs, htdemucs_ft): the same five names plus
# three that rebuild numpy float64 training metrics. Used for comparison exports only.
_DEMUCS_COMPARISON_GLOBALS = frozenset({
    ("numpy.core.multiarray", "scalar"), ("numpy", "dtype"), ("_codecs", "encode"),
})


def load_htdemucs(model_id: str):
    """Return (HTDemucs module in eval mode, lock entry) for one single-model id.

    Meta's Demucs checkpoints are never unpickled, by torch or by anything
    else. They are read with the symbolic reader in convert_reference.py: the
    pickle's globals stay inert tokens, REDUCE is recorded and not executed,
    and the float16 tensors are copied straight out of the zip. This function
    then builds the model from plain values:

      kwargs          -> ints / floats / bools / strings / lists as written in the file;
                         `segment` is recorded as "Fraction applied to ('39/5',)"
                         and the Fraction is constructed here from that string
      state           -> name -> float32 tensor made from the raw float16 bytes

    (`torch.load(weights_only=True)` would also work, but only by allow-listing
    `fractions.Fraction`, which is outside the project's base policy.)
    """
    entry = lock_entry(model_id)
    verify(entry)                       # size, SHA-256, and the static scan's exact set of globals

    import fractions
    import inspect
    import warnings

    import numpy as np
    import torch
    from demucs.htdemucs import HTDemucs

    import convert_reference as symbolic

    strict = set(entry["pickle"]["expectedGlobals"]) <= {f"{m}.{n}" for m, n in symbolic.ALLOWED_GLOBALS}
    allowed = symbolic.ALLOWED_GLOBALS if strict else symbolic.ALLOWED_GLOBALS | _DEMUCS_COMPARISON_GLOBALS
    archive, prefix, root = symbolic.open_checkpoint(local_path(entry), allowed, allow_build=not strict)
    if not (isinstance(root.get("klass"), symbolic.Global) and root["klass"].is_("demucs.htdemucs", "HTDemucs")):
        raise WeightsError(f"{entry['name']}: not an HTDemucs checkpoint")
    if root["args"] != ():
        raise WeightsError(f"{entry['name']}: unexpected positional constructor arguments")

    def plain(value):
        """A constructor argument as an ordinary Python value. Anything else is an error."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, symbolic.Call) and value.func.is_("fractions", "Fraction"):
            (text,) = value.args
            numerator, denominator = text.split("/")
            return fractions.Fraction(int(numerator), int(denominator))
        raise WeightsError(f"{entry['name']}: unexpected constructor argument {value!r}")

    kwargs = {key: plain(value) for key, value in root["kwargs"].items()}
    # Same filtering demucs.states.load_model applies: drop constructor
    # arguments this demucs version no longer knows.
    known = inspect.signature(HTDemucs.__init__).parameters
    for key in list(kwargs):
        if key not in known:
            warnings.warn(f"dropping unknown HTDemucs argument {key!r}")
            del kwargs[key]

    state = {}
    for key, value in root["state"].items():
        tensor = symbolic.describe_tensor(key, value, prefix, archive)
        halves = np.frombuffer(symbolic.tensor_float16_bytes(archive, tensor), dtype="<f2").reshape(tensor.shape)
        state[key] = torch.from_numpy(halves.astype(np.float32))        # exact: every float16 is a float32

    model = HTDemucs(**kwargs)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, entry


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "verify"])
    parser.add_argument("--only", action="append", default=[], help="model id or file name (repeatable)")
    args = parser.parse_args()

    lock = read_lock()
    entries = [e for e in lock["files"]
               if not args.only or e["name"] in args.only or set(args.only) & set(e["usedBy"])]
    evidence = json.loads(EVIDENCE_PATH.read_text()) if EVIDENCE_PATH.exists() else {}
    status = 0
    for entry in entries:
        record = evidence.get(entry["name"], {})
        try:
            if args.command == "fetch" and not local_path(entry).exists():
                record.update(download(entry, lock["allowedHosts"]))
                record["url"] = entry["url"]
            result = verify(entry)
            record.update(result)
            record["url"] = entry["url"]
            record["verifiedAt"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            gate = ("torch.load refused (not approved: " + ", ".join(result["unapproved"]) + "); read only by the symbolic reader"
                    if result.get("unapproved") else "ok for torch.load(weights_only=True)")
            print(f"[ok] {entry['name']}  {result['sizeBytes']} bytes  sha256={result['sha256']}")
            print(f"     publisher checksum: {result['publisherChecksum']}")
            if "pickleGlobals" in result:
                print(f"     pickle globals: {result['pickleGlobals']}")
                print(f"     outside base policy: {result['outsideBasePolicy']}  -> {gate}")
        except (WeightsError, urllib.error.URLError) as exc:
            status = 1
            record["error"] = str(exc)
            print(f"[FAIL] {entry['name']}: {exc}")
        evidence[entry["name"]] = record
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
