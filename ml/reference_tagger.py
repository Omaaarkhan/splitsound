"""Reference instrument tagging over a whole track: numpy + onnxruntime only (no torch).

The executable specification of CONTRACT.md section 2, mirrored by the Rust engine.

CLI (from the `ml/` directory):

    uv run --locked reference_tagger.py tag <tagger.onnx> <in_32k_mono.wav> [--json out.json]

The input must already be 32 kHz. Stereo is averaged to mono. This reference
does not resample; convert first, e.g.

    ffmpeg -i song.wav -ac 1 -ar 32000 song_32k.wav

The whole mel frontend is inside the ONNX graph, so the host only cuts windows.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 32000
WINDOW = 320000             # 10 s
HOP = 160000                # 5 s
NUM_CLASSES = 527

ASSETS = Path(__file__).resolve().parent / "assets"


def window_starts(length: int) -> list[int]:
    """Start sample of every analysis window.

    One window if the track fits in 10 s; otherwise windows every 5 s until the
    last one reaches the end of the track. Windows running past the end are
    zero-padded.
    """
    if length <= WINDOW:
        return [0]
    count = 1 + math.ceil((length - WINDOW) / HOP)
    return [k * HOP for k in range(count)]


def tag(session: ort.InferenceSession, mono: np.ndarray) -> dict[str, np.ndarray]:
    """[L] float32 mono at 32 kHz -> {"windows": [W, 527], "max": [527], "mean": [527]}."""
    mono = np.asarray(mono, dtype=np.float32)
    assert mono.ndim == 1 and mono.shape[0] > 0
    length = mono.shape[0]

    starts = window_starts(length)
    per_window = np.zeros((len(starts), NUM_CLASSES), dtype=np.float32)
    for w, start in enumerate(starts):
        window = np.zeros(WINDOW, dtype=np.float32)
        available = min(WINDOW, length - start)
        window[:available] = mono[start: start + available]        # the rest stays zero
        per_window[w] = session.run(["probs"], {"waveform": window[np.newaxis]})[0][0]

    return {
        "windows": per_window,
        "max": per_window.max(axis=0),
        "mean": per_window.mean(axis=0, dtype=np.float32),
    }


def detect_instruments(aggregated_max: np.ndarray, instrument_map: dict, threshold: float | None = None) -> list[dict]:
    """Apply the curated map: every mapped class whose max-over-windows probability reaches the threshold."""
    if threshold is None:
        threshold = instrument_map["threshold"]
    found = []
    for index, info in instrument_map["classes"].items():
        probability = float(aggregated_max[int(index)])
        limit = info.get("threshold", threshold)
        if probability >= limit:
            found.append({"index": int(index), "label": info["label"], "family": info["family"],
                          "generic": bool(info.get("generic", False)), "probability": probability})
    found.sort(key=lambda item: -item["probability"])
    return found


def open_model(path: str | Path, threads: int = 0) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    assert int(metadata["sampleRate"]) == SAMPLE_RATE and int(metadata["windowSamples"]) == WINDOW
    assert int(metadata["numClasses"]) == NUM_CLASSES
    return session


def read_mono_32k(path: str | Path) -> np.ndarray:
    import soundfile

    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)     # [L, C]
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"{path}: sample rate is {sample_rate}, expected {SAMPLE_RATE}. "
                         f"Convert first: ffmpeg -i in -ac 1 -ar {SAMPLE_RATE} out.wav")
    return data.mean(axis=1, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("tag", help="tag a 32 kHz mono WAV")
    cmd.add_argument("model")
    cmd.add_argument("wav")
    cmd.add_argument("--threshold", type=float, default=None, help="override the threshold in instrument_map.json")
    cmd.add_argument("--top", type=int, default=15, help="how many raw AudioSet classes to print")
    cmd.add_argument("--threads", type=int, default=0)
    cmd.add_argument("--json", type=Path, help="write max/mean/per-window probabilities and detections")
    args = parser.parse_args()

    labels = json.loads((ASSETS / "audioset_labels.json").read_text())
    instrument_map = json.loads((ASSETS / "instrument_map.json").read_text())
    session = open_model(args.model, args.threads)
    mono = read_mono_32k(args.wav)
    result = tag(session, mono)

    print(f"{args.wav}: {mono.shape[0] / SAMPLE_RATE:.1f} s, {result['windows'].shape[0]} window(s)")
    print(f"top {args.top} AudioSet classes (max over windows / mean over windows):")
    for index in np.argsort(-result["max"])[: args.top]:
        print(f"  {result['max'][index]:.3f} / {result['mean'][index]:.3f}  {index:3d}  {labels[index]}")
    detections = detect_instruments(result["max"], instrument_map, args.threshold)
    limit = args.threshold if args.threshold is not None else instrument_map["threshold"]
    print(f"instruments detected (max >= {limit}):")
    for item in detections:
        generic = "  (generic)" if item["generic"] else ""
        print(f"  {item['probability']:.3f}  [{item['family']:<6s}] {item['label']}{generic}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "windowStarts": window_starts(mono.shape[0]),
            "max": result["max"].tolist(), "mean": result["mean"].tolist(),
            "windows": result["windows"].tolist(), "detections": detections,
        }) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
