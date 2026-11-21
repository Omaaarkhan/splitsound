"""Compress the exported graphs and write dist/manifest.json (schemaVersion 2).

    uv run --locked python package.py

For every model in MODELS (in this order):
  * runs verify_onnx on the final .onnx (the weight-free htdemucs_6s graph with its own policy);
  * writes `<file>.zst` beside it (zstd level 19, single frame, content size and checksum in the frame);
  * records the SHA-256 and size of both files, computed from the bytes on disk.

The SHA-256 values in the manifest are a security control: the app pins them
and refuses any file that does not match. They are therefore always computed
here, from the final files, never copied from an export report.

The schema is documented in CONTRACT.md section 7.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import zstandard  # noqa: E402

import onnx_tools  # noqa: E402
import verify_onnx  # noqa: E402

DIST = ML_DIR / "dist"
ASSETS = ML_DIR / "assets"
ZSTD_LEVEL = 19

DEMUCS_TERMS = ("Demucs is published by Meta for research. Its weights are not under an open-source licence and are "
                "likely limited to non-commercial use (CC BY-NC). SplitSound downloads them directly from Meta and "
                "keeps them on this computer.")

# Presentation fields that do not come from the export reports.
MODELS = {
    "scnet_large": {
        "name": "SCNet Large",
        "description": "Best overall quality for vocals, drums, bass and everything else.",
        "source": "GitHub", "restricted": False, "terms": None, "license": "MIT",
        "origin": "https://github.com/starrytong/SCNet",
    },
    "scnet_small": {
        "name": "SCNet",
        "description": "Vocals, drums, bass and everything else, about three times faster than SCNet Large and a quarter of the download.",
        "source": "GitHub", "restricted": False, "terms": None, "license": "MIT",
        "origin": "https://github.com/starrytong/SCNet",
    },
    "melband_roformer_vocals": {
        "name": "Mel-Band RoFormer Vocals",
        "description": "Studio-grade vocal isolation that minimises vocal bleed into other stems; large and slow.",
        "source": "GitHub", "restricted": False, "terms": None, "license": "MIT",
        "origin": "https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model",
    },
    "htdemucs_6s": {
        "name": "Demucs 6-stem",
        "description": "Adds separate guitar and piano stems to vocals, drums, bass and other.",
        "source": "Meta", "restricted": True, "terms": DEMUCS_TERMS,
        "license": "Graph: MIT. Weights: not open source (see terms); not distributed by SplitSound.",
        "origin": "https://github.com/facebookresearch/demucs",
    },
    "efficientat_mn10_as": {
        "name": "Instrument tagger",
        "description": "Recognises which instruments and voices are present.",
        "source": "GitHub", "restricted": False, "terms": None, "license": "MIT",
        "origin": "https://github.com/fschmid56/EfficientAT",
    },
}


def compress(path: Path) -> Path:
    """zstd level 19, one frame, content size and xxhash checksum in the frame header/footer."""
    out_path = path.with_name(path.name + ".zst")
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_content_size=True, write_checksum=True, threads=0)
    data = path.read_bytes()
    out_path.write_bytes(compressor.compress(data))
    # Round trip before the hash is published.
    assert zstandard.ZstdDecompressor().decompress(out_path.read_bytes(), max_output_size=len(data) + 1) == data
    return out_path


def file_record(path: Path) -> dict:
    return {"file": path.name, "sha256": onnx_tools.sha256_file(path), "sizeBytes": path.stat().st_size}


def build_entry(model_id: str) -> dict | None:
    graph_path = DIST / f"{model_id}.onnx"
    report_path = DIST / f"{model_id}.export.json"
    if not graph_path.exists() or not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    weight_free = "weights" in report

    inert = (verify_onnx.verify(graph_path, external_data=report["weights"]["file"], max_inline_bytes=65536)
             if weight_free else verify_onnx.verify(graph_path))
    if inert["violations"]:
        raise SystemExit(f"{graph_path.name} failed verify_onnx: {inert['violations']}")
    graph = file_record(graph_path)
    assert graph["sha256"] == report["sha256"], f"{model_id}: file on disk differs from its export report; re-export"
    compressed = file_record(compress(graph_path))

    entry = {"id": model_id, "kind": inert["metadata"]["kind"], "revision": 1}
    entry.update({key: MODELS[model_id][key] for key in ("name", "description", "source", "restricted", "terms", "license")})
    entry["attribution"] = inert["metadata"]["attribution"]
    entry["origin"] = MODELS[model_id]["origin"]
    entry.update(graph)                                   # file, sha256, sizeBytes of the .onnx as used at run time
    entry["acquire"] = {"type": "download", **compressed}
    entry["opset"] = inert["opset"]

    if entry["kind"] == "tagger":
        entry.update({"sampleRate": report["sampleRate"], "channels": 1, "windowSamples": report["windowSamples"],
                      "hopSamples": report["windowSamples"] // 2, "numClasses": report["numClasses"],
                      "dynamicBatch": False,
                      "labels": file_record(ASSETS / "audioset_labels.json"),
                      "instrumentMap": file_record(ASSETS / "instrument_map.json")})
        return entry

    entry.update({key: report[key] for key in ("sources", "targets", "sampleRate", "channels", "segmentSamples",
                                               "strideSamples", "nfft", "hop", "bins", "frames", "dynamicBatch")})
    entry["window"] = {"htdemucs": "hann", "scnet": "rectangular"}.get(entry["kind"], report.get("window"))
    entry["weightStorage"] = report.get("weightStorage", "float32")
    for key in ("melBands", "winLength", "normalize"):
        if key in report:
            entry[key] = report[key]
    if weight_free:
        weights_path = DIST / report["weights"]["file"]
        table_path = DIST / report["weights"]["tensors"]
        table = json.loads(table_path.read_text())
        if weights_path.exists():                         # present on the build machine, never distributed
            produced = file_record(weights_path)
            assert (produced["sha256"], produced["sizeBytes"]) == (table["weights"]["sha256"], table["weights"]["sizeBytes"])
        entry["weights"] = {
            "file": report["weights"]["file"], "sha256": table["weights"]["sha256"], "sizeBytes": table["weights"]["sizeBytes"],
            "tensors": file_record(table_path),
            "upstream": {key: table["upstream"][key] for key in ("url", "sha256", "sizeBytes", "format")},
        }
    return entry


def main() -> int:
    entries = []
    for model_id in MODELS:
        entry = build_entry(model_id)
        if entry is None:
            print(f"-- {model_id}: not exported, skipped")
            continue
        entries.append(entry)
        raw, packed = entry["sizeBytes"], entry["acquire"]["sizeBytes"]
        print(f"{entry['id']:<26s} {raw / 1e6:8.2f} MB -> {packed / 1e6:8.2f} MB zstd ({packed / raw:5.1%})  sha256 {entry['sha256']}")
        if "weights" in entry:
            print(f"{'':<26s} + weights written on the user's machine: {entry['weights']['sizeBytes'] / 1e6:.2f} MB "
                  f"from {entry['weights']['upstream']['url']}")
    manifest = {"schemaVersion": 2, "models": entries}
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    # The only files that may ever leave the build machine. dist/ also holds
    # htdemucs_6s.weights (Meta's weights, not redistributable), compare/ and eval/.
    publish = ["manifest.json"]
    for entry in entries:
        publish.append(entry["acquire"]["file"])
        if "weights" in entry:
            publish.append(entry["weights"]["tensors"]["file"])
        for key in ("labels", "instrumentMap"):
            if key in entry:
                publish.append(f"../assets/{entry[key]['file']}")
    (DIST / "PUBLISH.txt").write_text(
        "Publish exactly these files and nothing else from this directory.\n"
        "NEVER publish htdemucs_6s.weights, compare/ or eval/: they contain or derive from material\n"
        "that is not redistributable (see ../LICENSES.md).\n\n" + "\n".join(publish) + "\n")
    print(f"wrote {DIST / 'manifest.json'} with {len(entries)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
