"""Generate the test vectors for the Rust engine's unit and integration tests into dist/testdata/.

    uv run --locked python make_testdata.py                 # everything
    uv run --locked python make_testdata.py --only transforms converter

Sections (see dist/testdata/README.md, written by this script, for every file):
  clip        the deterministic 12 s synthetic stereo clip
  transforms  STFT / inverse-STFT vectors for each model kind (single channel, .npy)
  stems       expected per-stem output of reference.py for each exported separation model
  tagger      a fixed 10 s window and the tagger's 527 probabilities for it
  converter   a tiny checkpoint in exactly the layout of Meta's file, its tensor table and the
              expected weights file, plus inputs the converter must reject

Only numpy, soundfile and onnxruntime are needed, except for the `converter`
section, which uses torch to WRITE the small fixture (it never loads one).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import numpy as np  # noqa: E402
import soundfile  # noqa: E402

import convert_reference  # noqa: E402
import reference  # noqa: E402
import reference_tagger  # noqa: E402
import synth  # noqa: E402

DIST = ML_DIR / "dist"
OUT = DIST / "testdata"
SEPARATION_MODELS = ["scnet_small", "scnet_large", "htdemucs_6s", "melband_roformer_vocals"]
TAGGER = "efficientat_mn10_as"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_clip(index: dict) -> np.ndarray:
    clip = synth.synth_clip(12.0)
    soundfile.write(str(OUT / "synth_12s.wav"), clip.T, synth.SAMPLE_RATE, subtype="FLOAT")
    index["clip"] = {"file": "synth_12s.wav", "sampleRate": 44100, "channels": 2, "samples": int(clip.shape[1]),
                     "seed": 20240607, "command": "uv run --locked python make_testdata.py --only clip"}
    return clip


def make_transforms(index: dict, clip: np.ndarray) -> None:
    index["transforms"] = {}
    # htdemucs: one channel of the first 7.8 s
    x = np.ascontiguousarray(clip[0, : reference.HT_SEGMENT])
    spec = reference.htdemucs_stft(x[np.newaxis])                       # [2, bins, frames] = real, imag
    complex_spec = (spec[0] + 1j * spec[1]).astype(np.complex64)
    round_trip = reference.htdemucs_istft_channel(spec[0], spec[1])
    np.save(OUT / "stft_htdemucs_input.npy", x)
    np.save(OUT / "stft_htdemucs_spec.npy", complex_spec)
    np.save(OUT / "stft_htdemucs_roundtrip.npy", round_trip)
    index["transforms"]["htdemucs"] = {
        "input": "stft_htdemucs_input.npy", "inputShape": list(x.shape),
        "spec": "stft_htdemucs_spec.npy", "specShape": list(complex_spec.shape), "specDtype": "complex64 [bins, frames]",
        "roundTrip": "stft_htdemucs_roundtrip.npy",
        "note": "roundTrip = ISTFT_htdemucs(spec). It differs from the input near the segment edges by design "
                "(CONTRACT.md 2.2); compare against this file, not against the input."}
    # scnet: one channel of the first 11 s
    x = np.ascontiguousarray(clip[0, : reference.SC_SEGMENT])
    spec = reference.scnet_stft(x[np.newaxis])
    complex_spec = (spec[0] + 1j * spec[1]).astype(np.complex64)
    round_trip = reference.scnet_istft_channel(spec[0], spec[1])
    np.save(OUT / "stft_scnet_input.npy", x)
    np.save(OUT / "stft_scnet_spec.npy", complex_spec)
    np.save(OUT / "stft_scnet_roundtrip.npy", round_trip)
    index["transforms"]["scnet"] = {
        "input": "stft_scnet_input.npy", "inputShape": list(x.shape),
        "spec": "stft_scnet_spec.npy", "specShape": list(complex_spec.shape), "specDtype": "complex64 [bins, frames]",
        "roundTrip": "stft_scnet_roundtrip.npy",
        "note": "roundTrip = ISTFT_scnet(spec); equals the input to float32 precision."}
    if hasattr(reference, "melband_stft"):
        x = np.ascontiguousarray(clip[0, : reference.MB_SEGMENT])
        spec = reference.melband_stft(x[np.newaxis])
        complex_spec = (spec[0] + 1j * spec[1]).astype(np.complex64)
        round_trip = reference.melband_istft_channel(spec[0], spec[1])
        np.save(OUT / "stft_melband_input.npy", x)
        np.save(OUT / "stft_melband_spec.npy", complex_spec)
        np.save(OUT / "stft_melband_roundtrip.npy", round_trip)
        index["transforms"]["melband_roformer"] = {
            "input": "stft_melband_input.npy", "inputShape": list(x.shape),
            "spec": "stft_melband_spec.npy", "specShape": list(complex_spec.shape), "specDtype": "complex64 [bins, frames]",
            "roundTrip": "stft_melband_roundtrip.npy",
            "note": "roundTrip = ISTFT_melband(spec)."}
    weight = {str(n): reference.transition_weight(n)[[0, 1, n // 2 - 1, n // 2, n - 1]].tolist()
              for n in (reference.HT_SEGMENT, reference.SC_SEGMENT, reference.MB_SEGMENT)}
    index["transforms"]["transitionWeightSamples"] = {
        "note": "weight[i] for i in [0, 1, N/2-1, N/2, N-1], keyed by N", "values": weight}
    index["transforms"]["shiftAmounts"] = {str(k): reference.shift_amounts(k) for k in (0, 1, 2, 3, 4)}
    index["transforms"]["command"] = "uv run --locked python make_testdata.py --only transforms"


def make_stems(index: dict, clip: np.ndarray) -> None:
    index["stems"] = {}
    for model_id in SEPARATION_MODELS:
        path = DIST / f"{model_id}.onnx"
        if not path.exists():
            print(f"-- stems: {model_id} is not exported, skipped")
            continue
        model = reference.open_model(path)
        estimate = reference.separate(model, clip, shifts=0, normalize=model.normalize)
        folder = OUT / "stems" / model_id
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for s, name in enumerate(model.sources):
            soundfile.write(str(folder / f"{name}.wav"), estimate[s].T, reference.SAMPLE_RATE, subtype="FLOAT")
            files[name] = {"file": f"stems/{model_id}/{name}.wav", "peak": float(np.abs(estimate[s]).max()),
                           "rms": float(np.sqrt(np.mean(estimate[s].astype(np.float64) ** 2)))}
        norm_flag = "" if model.normalize else " --no-normalize"
        index["stems"][model_id] = {
            "modelSha256": sha256(path), "segmentSamples": model.segment, "strideSamples": model.stride,
            "shifts": 0, "normalize": model.normalize, "files": files,
            "command": f"uv run --locked reference.py separate dist/{model_id}.onnx dist/testdata/synth_12s.wav "
                       f"dist/testdata/stems/{model_id} --all-stems{norm_flag}"}
        print(f"stems: {model_id} -> {list(files)}")


def make_tagger(index: dict) -> None:
    path = DIST / f"{TAGGER}.onnx"
    if not path.exists():
        return
    window = synth.synth_clip(10.0, sample_rate=reference_tagger.SAMPLE_RATE).mean(axis=0).astype(np.float32)
    session = reference_tagger.open_model(path)
    probs = session.run(["probs"], {"waveform": window[np.newaxis]})[0][0]
    np.save(OUT / "tagger_input_32k.npy", window)
    np.save(OUT / "tagger_probs.npy", probs)
    # A 23 s input to pin the windowing: 4 windows (0, 5, 10, 15 s), the last zero-padded.
    long_input = np.concatenate([window, window[::-1], window[: 3 * reference_tagger.SAMPLE_RATE]])
    result = reference_tagger.tag(session, long_input)
    np.save(OUT / "tagger_long_input_32k.npy", long_input)
    np.save(OUT / "tagger_long_max.npy", result["max"])
    np.save(OUT / "tagger_long_mean.npy", result["mean"])
    index["tagger"] = {
        "modelSha256": sha256(path), "input": "tagger_input_32k.npy", "probs": "tagger_probs.npy",
        "top5": [int(i) for i in np.argsort(-probs)[:5]],
        "long": {"input": "tagger_long_input_32k.npy", "samples": int(long_input.shape[0]),
                 "windowStarts": reference_tagger.window_starts(long_input.shape[0]),
                 "max": "tagger_long_max.npy", "mean": "tagger_long_mean.npy"},
        "command": "uv run --locked python make_testdata.py --only tagger"}


def make_converter(index: dict) -> None:
    """A miniature checkpoint in the exact layout of Meta's file, and inputs that must be rejected."""
    import collections
    from fractions import Fraction

    import torch
    from demucs.htdemucs import HTDemucs

    folder = OUT / "converter"
    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(5)
    state = collections.OrderedDict()
    state["encoder.0.conv.weight"] = torch.randn(3, 2, 4, 1, generator=generator).half()
    state["encoder.0.conv.bias"] = torch.randn(3, generator=generator).half()
    # every special float16 class: zeros of both signs, the smallest subnormal, a subnormal,
    # the smallest normal, the largest finite value and its negative
    state["edge.values"] = torch.from_numpy(np.array([0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00],
                                                     dtype=np.uint16).view(np.float16).copy())
    state["decoder.0.scale"] = torch.randn(5, 7, generator=generator).half()
    package = {"klass": HTDemucs, "args": (),
               "kwargs": {"sources": ["drums", "bass"], "audio_channels": 2, "samplerate": 44100,
                          "segment": Fraction(39, 5), "channels": 48, "wiener_residual": False, "t_lr": None},
               "state": state,
               # (every dict has at least two entries: Meta's file only ever uses the SETITEMS opcode)
               "training_args": {"epochs": 3, "lr": 0.0003, "dset": {"test_mapping": {0: "a", 1: "b"}, "musdb": "x"}},
               "metrics": {"nsdr": 9.0, "nsdr_drums": 10.5}}
    buffer = io.BytesIO()
    torch.save(package, buffer, pickle_protocol=2)          # written by us; never loaded
    with zipfile.ZipFile(buffer) as written:
        prefix = [n for n in written.namelist() if n.endswith("/data.pkl")][0][: -len("data.pkl")]
        members = {"data.pkl": written.read(prefix + "data.pkl"), "version": b"3\n"}
        for name in written.namelist():
            rest = name[len(prefix):]
            if rest.startswith("data/"):
                members[rest] = written.read(name)

    def write_zip(path: Path, entries: dict, compression=zipfile.ZIP_STORED) -> None:
        # Same layout as Meta's 2022 file: archive/data.pkl, archive/data/<n>, archive/version; all stored.
        with zipfile.ZipFile(path, "w", compression) as archive:
            for name, data in entries.items():
                info = zipfile.ZipInfo("archive/" + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = compression
                archive.writestr(info, data)

    fixture = folder / "fixture.th"
    write_zip(fixture, members)

    # tensor table and expected weights, computed here with numpy (independent of the converter)
    rows, offset, blob = [], 0, b""
    for key, tensor in state.items():
        data = tensor.numpy().astype("<f4").tobytes()
        rows.append({"key": key, "shape": list(tensor.shape), "sourceDtype": "float16", "offset": offset, "byteLength": len(data)})
        blob += data
        offset += len(data)
    table = {"schemaVersion": 1, "model": "fixture", "weightsFile": "fixture.weights",
             "upstream": {"url": None, "sha256": sha256(fixture), "sizeBytes": fixture.stat().st_size,
                          "format": "torch-zip", "storageDtype": "float16"},
             "weights": {"sha256": hashlib.sha256(blob).hexdigest(), "sizeBytes": len(blob)}, "tensors": rows}
    (folder / "fixture.tensors.json").write_text(json.dumps(table, indent=1) + "\n")
    (folder / "fixture.expected.weights").write_bytes(blob)
    produced = folder / "fixture.produced.weights"
    result = convert_reference.convert(fixture, folder / "fixture.tensors.json", produced)
    assert produced.read_bytes() == blob and result == table["weights"]
    produced.unlink()

    # ---- inputs the converter must refuse -------------------------------------
    rejects = {}

    def reject(name: str, entries: dict, why: str, compression=zipfile.ZIP_STORED) -> None:
        path = folder / name
        write_zip(path, entries, compression)
        bad_table = dict(table)
        bad_table["upstream"] = {**table["upstream"], "sha256": sha256(path), "sizeBytes": path.stat().st_size}
        (folder / (name + ".tensors.json")).write_text(json.dumps(bad_table, indent=1) + "\n")
        try:
            convert_reference.convert(path, folder / (name + ".tensors.json"), folder / "should_not_exist.weights")
        except convert_reference.ConversionError as error:
            rejects[name] = {"why": why, "referenceError": str(error)}
            return
        raise AssertionError(f"{name} was accepted by the converter")

    pickle_bytes = members["data.pkl"]
    # a global that is not on the allow-list (harmless if anyone did unpickle it: it would print a word)
    foreign = b"\x80\x02cbuiltins\nprint\nX\x05\x00\x00\x00hello\x85R."
    reject("reject_foreign_global.th", {**members, "data.pkl": foreign}, "pickle references builtins.print")
    reject("reject_unknown_opcode.th", {**members, "data.pkl": b"\x80\x02\x8c\x05hello."}, "opcode SHORT_BINUNICODE (0x8c) is not in the allowed set")
    reject("reject_protocol.th", {**members, "data.pkl": b"\x80\x04" + pickle_bytes[2:]}, "pickle protocol 4")
    reject("reject_trailing_bytes.th", {**members, "data.pkl": pickle_bytes + b"N."}, "bytes after the STOP opcode")
    reject("reject_truncated_pickle.th", {**members, "data.pkl": pickle_bytes[:-20]}, "pickle ends early")
    reject("reject_extra_member.th", {**members, "run.sh": b"echo no\n"}, "zip member outside data.pkl / version / data/<n>")
    reject("reject_compressed.th", members, "zip members are deflated instead of stored", compression=zipfile.ZIP_DEFLATED)
    nan_members = dict(members)
    storage_name = sorted(n for n in members if n.startswith("data/"))[0]
    nan_members[storage_name] = struct.pack("<H", 0x7E00) + members[storage_name][2:]
    reject("reject_nan_weight.th", nan_members, "a NaN among the weights")
    short_members = dict(members)
    short_members[storage_name] = members[storage_name][:-2]
    reject("reject_short_storage.th", short_members, "a storage shorter than the pickle says")
    reject("reject_version.th", {**members, "version": b"4\n"}, "unexpected format version")
    (folder / "should_not_exist.weights.part").unlink(missing_ok=True)

    # wrong pinned hash: the right file with a table that pins another SHA-256
    wrong = dict(table)
    wrong["upstream"] = {**table["upstream"], "sha256": "0" * 64}
    (folder / "reject_wrong_pin.tensors.json").write_text(json.dumps(wrong, indent=1) + "\n")
    try:
        convert_reference.convert(fixture, folder / "reject_wrong_pin.tensors.json", folder / "should_not_exist.weights")
        raise AssertionError("wrong pin accepted")
    except convert_reference.ConversionError as error:
        rejects["reject_wrong_pin.tensors.json (with fixture.th)"] = {"why": "checkpoint SHA-256 differs from the pin", "referenceError": str(error)}

    index["converter"] = {
        "fixture": "converter/fixture.th", "table": "converter/fixture.tensors.json",
        "expectedWeights": "converter/fixture.expected.weights", "expectedWeightsSha256": table["weights"]["sha256"],
        "tensors": len(rows), "mustReject": rejects,
        "note": "fixture.th has the member layout and pickle structure of Meta's htdemucs_6s checkpoint "
                "(written with pickle protocol 2 and re-packed as archive/data.pkl, archive/data/<n>, archive/version), "
                "with four tiny float16 tensors including every special float16 class. Each reject_* file comes with "
                "its own <name>.tensors.json that pins that file's hash, so the rejection is caused by the content.",
        "command": "uv run --locked python make_testdata.py --only converter"}
    print(f"converter: fixture with {len(rows)} tensors, {len(rejects)} inputs rejected as expected")


README = """# Test vectors

Generated by `ml/make_testdata.py`; `testdata.json` is the machine-readable index (file names, shapes,
SHA-256 of the model each expected output was produced with, and the command that regenerates it).
Nothing here is committed. All `.npy` files are little-endian, version 1.0 headers, C order.

| file | what it is |
|------|------------|
| `synth_12s.wav` | deterministic 12 s stereo 44.1 kHz float32 clip (tones, chirps, noise bursts, amplitude modulation; `synth.py`, seed 20240607) |
| `stft_htdemucs_input.npy` | float32 `[343980]`: left channel of the first 7.8 s of the clip |
| `stft_htdemucs_spec.npy` | complex64 `[2048, 336]`: `STFT_htdemucs` of it (CONTRACT.md 2.1) |
| `stft_htdemucs_roundtrip.npy` | float32 `[343980]`: `ISTFT_htdemucs` of that spectrogram (not equal to the input at the edges, by design) |
| `stft_scnet_input.npy` | float32 `[485100]`: left channel of the first 11 s of the clip |
| `stft_scnet_spec.npy` | complex64 `[2049, 476]`: `STFT_scnet` of it (CONTRACT.md 3.1) |
| `stft_scnet_roundtrip.npy` | float32 `[485100]`: `ISTFT_scnet` of that spectrogram (equals the input to float precision) |
| `stft_melband_*.npy` | the same three files for the `melband_roformer` transform (`STFT_melband` / `ISTFT_melband`, CONTRACT.md 4) |
| `stems/<model>/<stem>.wav` | float32 stereo: what `reference.py` produces for the clip with that model (shifts 0, per-model `normalize`, default stride) |
| `tagger_input_32k.npy` | float32 `[320000]`: one 10 s mono window at 32 kHz |
| `tagger_probs.npy` | float32 `[527]`: the tagger's output for that window |
| `tagger_long_input_32k.npy`, `tagger_long_max.npy`, `tagger_long_mean.npy` | a 23 s input (4 windows, the last zero-padded) and the max / mean aggregates |
| `converter/fixture.th` | a miniature checkpoint in the layout of Meta's `htdemucs_6s` file |
| `converter/fixture.tensors.json`, `converter/fixture.expected.weights` | its tensor table and the exact weights file the converter must produce |
| `converter/reject_*.th` (+ `.tensors.json`) | inputs the converter must refuse; `testdata.json` lists why and the reference's error text |

Regenerate everything: `cd ml && uv run --locked python make_testdata.py`.
One section: `uv run --locked python make_testdata.py --only clip|transforms|stems|tagger|converter`.

Suggested tolerances: transforms 1e-5 absolute (float32 FFT); stems 60 dB SNR against the WAVs when
running the same `.onnx` on another machine or execution provider, 100 dB on the same CPU provider;
tagger probabilities 1e-4 absolute; converter output must be byte-identical.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sections = ["clip", "transforms", "stems", "tagger", "converter"]
    parser.add_argument("--only", nargs="+", choices=sections, default=sections)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "testdata.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    clip = make_clip(index)
    if "transforms" in args.only:
        make_transforms(index, clip)
    if "stems" in args.only:
        make_stems(index, clip)
    if "tagger" in args.only:
        make_tagger(index)
    if "converter" in args.only:
        make_converter(index)
    index["files"] = {str(p.relative_to(OUT)): {"sha256": sha256(p), "sizeBytes": p.stat().st_size}
                      for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("testdata.json", "README.md")}
    index_path.write_text(json.dumps(index, indent=1) + "\n")
    (OUT / "README.md").write_text(README)
    print(f"{len(index['files'])} files in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
