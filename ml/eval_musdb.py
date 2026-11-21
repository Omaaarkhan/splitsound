"""Real-music sanity checks on the MUSDB18 7-second sample (144 excerpts with ground-truth stems).

    uv run --locked python eval_musdb.py prepare        # verify + extract the pinned archive into dist/eval/musdb7/
    uv run --locked python eval_musdb.py tagger         # tagger on every excerpt; threshold study against stem energy
    uv run --locked python eval_musdb.py separation     # SDR per stem, and reference.py vs the PyTorch originals

The archive is the dataset authors' own release asset, pinned by SHA-256 in
weights.lock.json. It is treated strictly as data: the zip is unpacked with
member names checked against a fixed pattern, and the `.stem.mp4` files are
only ever handed to ffmpeg for decoding. Nothing from it is executed, and
nothing under dist/eval/ is committed.

Each `.stem.mp4` holds five stereo AAC streams: mixture, drums, bass, other, vocals.

SDR here is the plain signal-to-distortion ratio over the whole excerpt and
both channels, 10*log10(sum ref^2 / sum (ref - est)^2), the definition used by
the Music Demixing Challenge. It is NOT museval/BSSEval, and the excerpts are
7 s of lossy audio, so the numbers are a regression baseline, not something to
compare with published MUSDB18-HQ scores.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import numpy as np  # noqa: E402

import reference  # noqa: E402
import reference_tagger  # noqa: E402
import weights  # noqa: E402

EVAL_DIR = ML_DIR / "dist" / "eval" / "musdb7"
REPORT_DIR = ML_DIR / "dist" / "reports"
MEMBER_PATTERN = re.compile(r"^(train|test)/[^/\\]+\.stem\.mp4$")
STEMS = ["drums", "bass", "other", "vocals"]          # audio streams 1..4; stream 0 is the mixture


def prepare() -> None:
    entry = weights.lock_entry("eval:musdb7")
    weights.verify(entry)                               # size + SHA-256
    count = 0
    with zipfile.ZipFile(weights.local_path(entry)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if not MEMBER_PATTERN.match(info.filename):
                raise SystemExit(f"unexpected archive member, refusing to extract: {info.filename!r}")
            subset, name = info.filename.split("/", 1)
            destination = EVAL_DIR / subset / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != info.file_size:
                destination.write_bytes(archive.read(info))
            count += 1
    print(f"{count} excerpts in {EVAL_DIR}")


def tracks(subset: str) -> list[Path]:
    found = sorted((EVAL_DIR / subset).glob("*.stem.mp4"))
    if not found:
        raise SystemExit("no excerpts found; run `eval_musdb.py prepare` first")
    return found


def decode(path: Path, stream: int, sample_rate: int = 44100, channels: int = 2) -> np.ndarray:
    """Decode one audio stream of a stem file with ffmpeg. Returns [channels, L] float32."""
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path.resolve()), "-map", f"0:a:{stream}",
               "-ac", str(channels), "-ar", str(sample_rate), "-f", "f32le", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T.copy()


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 1e-10))


def sdr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref64 = ref.astype(np.float64)
    return 10.0 * np.log10((np.sum(ref64 ** 2) + 1e-10) / (np.sum((ref64 - est.astype(np.float64)) ** 2) + 1e-10))


# ----------------------------------------------------------------------------
# tagger threshold study
# ----------------------------------------------------------------------------

def tagger_study(threads: int) -> None:
    labels = json.loads((reference_tagger.ASSETS / "audioset_labels.json").read_text())
    instrument_map = json.loads((reference_tagger.ASSETS / "instrument_map.json").read_text())
    session = reference_tagger.open_model(ML_DIR / "dist" / "efficientat_mn10_as.onnx", threads)
    family_classes: dict[str, list[int]] = {}
    for index, info in instrument_map["classes"].items():
        family_classes.setdefault(info["family"], []).append(int(index))

    rows = []
    for subset in ("train", "test"):
        for path in tracks(subset):
            mono = decode(path, 0, sample_rate=reference_tagger.SAMPLE_RATE, channels=1)[0]
            probs = reference_tagger.tag(session, mono)["max"]
            mix_db = rms_db(decode(path, 0))
            stem_db = {name: rms_db(decode(path, k + 1)) for k, name in enumerate(STEMS)}
            rows.append({"track": f"{subset}/{path.name}", "mixDb": mix_db, "stemDb": stem_db,
                         "probs": probs.tolist(),
                         "family": {fam: float(max(probs[i] for i in idx)) for fam, idx in family_classes.items()}})
    print(f"{len(rows)} excerpts tagged (7 s each, so one zero-padded 10 s window per excerpt)")

    # Ground truth from stem energy: present if the stem is within 30 dB of the
    # mixture level, absent if it is more than 50 dB below it; in between is ignored.
    def truth(row: dict, stem: str) -> bool | None:
        relative = row["stemDb"][stem] - row["mixDb"]
        if relative >= -30.0:
            return True
        if relative <= -50.0:
            return False
        return None

    thresholds = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
    study = {}
    print("\nFamily score = max probability over the family's classes. Recall on excerpts where the stem is")
    print("present, false-positive rate on excerpts where it is silent:")
    for stem in ("vocals", "drums", "bass"):
        present = [r["family"][stem] for r in rows if truth(r, stem) is True]
        absent = [r["family"][stem] for r in rows if truth(r, stem) is False]
        study[stem] = {"present": len(present), "absent": len(absent), "byThreshold": {}}
        print(f"  {stem}: present in {len(present)} excerpts, silent in {len(absent)}; "
              f"median score when present {np.median(present):.3f}"
              + (f", when silent {np.median(absent):.3f}" if absent else ""))
        for t in thresholds:
            recall = float(np.mean([p >= t for p in present])) if present else float("nan")
            false_positive = float(np.mean([p >= t for p in absent])) if absent else float("nan")
            study[stem]["byThreshold"][str(t)] = {"recall": recall, "falsePositiveRate": false_positive}
            print(f"      threshold {t:4.2f}: recall {recall:5.2f}   false positives {false_positive:5.2f}")

    print("\nHow often each mapped class reaches a threshold across all excerpts (specific classes only):")
    counts = []
    for index, info in instrument_map["classes"].items():
        values = np.array([r["probs"][int(index)] for r in rows])
        counts.append((info["family"], info["label"], float(np.mean(values >= 0.1)), float(np.mean(values >= 0.2)),
                       float(np.median(values)), float(values.max())))
    for family, label, at10, at20, median, peak in sorted(counts, key=lambda c: -c[2])[:28]:
        print(f"  [{family:<6s}] {label:<22s} >=0.10 in {at10:5.1%}   >=0.20 in {at20:5.1%}   median {median:.3f}   max {peak:.3f}")

    top = np.argsort(-np.mean([r["probs"] for r in rows], axis=0))[:12]
    print("\nHighest mean probability over all excerpts:", ", ".join(f"{labels[i]} {np.mean([r['probs'][i] for r in rows]):.2f}" for i in top))

    # ---- the same tagger on the isolated ground-truth stems ------------------
    # Positives: a family's score on its own stem. Negatives: its score on the
    # stems that cannot contain it (guitar and piano live inside MUSDB's
    # "other" stem, so "other" is neither a positive nor a negative for them).
    print("\nTagging the isolated ground-truth stems instead of the mix:")
    stem_scores: dict[str, dict[str, list[float]]] = {stem: {fam: [] for fam in family_classes} for stem in STEMS}
    stem_top: dict[str, dict[str, int]] = {stem: {} for stem in STEMS}
    for subset in ("train", "test"):
        for path in tracks(subset):
            for k, stem in enumerate(STEMS):
                mono = decode(path, k + 1, sample_rate=reference_tagger.SAMPLE_RATE, channels=1)[0]
                if rms_db(mono) < -45.0:
                    continue                                    # stem (nearly) silent in this excerpt
                probs = reference_tagger.tag(session, mono)["max"]
                for fam, idx in family_classes.items():
                    stem_scores[stem][fam].append(float(max(probs[i] for i in idx)))
                best = max(instrument_map["classes"], key=lambda i: probs[int(i)])
                name = instrument_map["classes"][best]["label"]
                stem_top[stem][name] = stem_top[stem].get(name, 0) + 1
    families = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    print("  median family score (rows: which stem was tagged, columns: family)")
    print("  " + " " * 9 + "".join(f"{fam:>9s}" for fam in families) + "     clips")
    for stem in STEMS:
        print(f"  {stem:<9s}" + "".join(f"{np.median(stem_scores[stem][fam]):9.3f}" for fam in families)
              + f"   {len(stem_scores[stem]['vocals']):5d}")
    stem_study = {}
    for fam, own, negatives in (("vocals", "vocals", ["drums", "bass", "other"]),
                                ("drums", "drums", ["bass", "other", "vocals"]),
                                ("bass", "bass", ["drums", "other", "vocals"]),
                                ("guitar", None, ["drums", "bass", "vocals"]),
                                ("piano", None, ["drums", "bass", "vocals"])):
        positives = stem_scores[own][fam] if own else []
        negative_scores = [v for stem in negatives for v in stem_scores[stem][fam]]
        stem_study[fam] = {}
        line = f"  {fam:<7s}"
        for t in thresholds:
            recall = float(np.mean([v >= t for v in positives])) if positives else float("nan")
            false_positive = float(np.mean([v >= t for v in negative_scores]))
            stem_study[fam][str(t)] = {"recall": recall, "falsePositiveRate": false_positive}
        print(f"  {fam}: recall on its own stem / false-positive rate on stems that cannot contain it")
        for t in thresholds:
            entry = stem_study[fam][str(t)]
            print(f"      threshold {t:4.2f}: recall {entry['recall']:5.2f}   false positives {entry['falsePositiveRate']:5.2f}")
    for stem in STEMS:
        ranked = sorted(stem_top[stem].items(), key=lambda kv: -kv[1])[:6]
        print(f"  most frequent top instrument class on the {stem} stem: " + ", ".join(f"{k} ({v})" for k, v in ranked))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tagger_musdb7.json").write_text(json.dumps({"mixStudy": study, "stemStudy": stem_study, "rows": [
        {k: v for k, v in r.items() if k != "probs"} for r in rows]}, indent=1) + "\n")


# ----------------------------------------------------------------------------
# separation: SDR against the ground-truth stems, and agreement with the PyTorch originals
# ----------------------------------------------------------------------------

# name -> list of (model id used to load the PyTorch original, path of the ONNX graph under dist/)
MODEL_SETS = {
    "scnet_large": [("scnet_large", "scnet_large.onnx")],
    "scnet_small": [("scnet_small", "scnet_small.onnx")],
    "htdemucs_6s": [("htdemucs_6s", "htdemucs_6s.onnx")],
    # comparison exports, not distributed (dist/compare/)
    "scnet_large_float32": [("scnet_large", "compare/scnet_large_f32.onnx")],
    "scnet_small_float32": [("scnet_small", "compare/scnet_small_f32.onnx")],
    "htdemucs_ft": [(f"htdemucs_ft_{name}", f"compare/htdemucs_ft_{name}.onnx") for name in ("drums", "bass", "other", "vocals")],
    "htdemucs": [("htdemucs", "compare/htdemucs.onnx")],
}


def separate_with_reference(models: list, mix: np.ndarray) -> dict[str, np.ndarray]:
    estimates, targets = [], []
    for model in models:
        estimates.append(reference.separate(model, mix, shifts=0, normalize=True))
        targets.append([model.sources.index(t) for t in model.targets])
    combined = reference.combine_bag(estimates, targets)
    return {models[0].sources[index]: audio for index, audio in combined.items()}


def separate_with_torch(torch_models: list, bag_weights: list, mix: np.ndarray, overlap: float) -> dict[str, np.ndarray]:
    """The originals' own flow: normalise, demucs.apply.apply_model on the PyTorch modules, de-normalise."""
    import torch
    from demucs.apply import BagOfModels, apply_model

    bag = BagOfModels(torch_models, bag_weights)
    wav = torch.from_numpy(mix.copy())
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        out = apply_model(bag, wav[None], shifts=0, split=True, overlap=overlap, progress=False, device="cpu")[0]
    out = out * ref.std() + ref.mean()
    return {name: out[k].numpy() for k, name in enumerate(bag.sources)}


def load_torch_original(model_id: str):
    if model_id.startswith("scnet"):
        import check_reference
        import export_scnet

        return check_reference.SCNetForApply(export_scnet.load_scnet(model_id)[0])
    return weights.load_htdemucs(model_id)[0]


def score(stems: dict[str, np.ndarray], truth: dict[str, np.ndarray]) -> dict[str, float]:
    if "guitar" in stems:                    # MUSDB's "other" contains guitar and piano
        stems = dict(stems)
        stems["other"] = stems["other"] + stems["guitar"] + stems["piano"]
    result = {}
    for name in STEMS:
        length = min(truth[name].shape[1], stems[name].shape[1])
        if rms_db(truth[name]) > -60.0:      # skip silent ground-truth stems
            result[name] = sdr_db(truth[name][:, :length], stems[name][:, :length])
    return result


def separation_study(sets: list[str], torch_clips: int, threads: int, stride: int | None) -> None:
    test_tracks = tracks("test")
    mixes = {p.name: decode(p, 0) for p in test_tracks}
    truths = {p.name: {name: decode(p, k + 1) for k, name in enumerate(STEMS)} for p in test_tracks}

    # A multi-chunk piece of real music with ground truth: the first ten test excerpts back to back (70 s).
    medley_mix = np.concatenate([mixes[p.name] for p in test_tracks[:10]], axis=1)
    medley_truth = {name: np.concatenate([truths[p.name][name] for p in test_tracks[:10]], axis=1) for name in STEMS}

    report = {}
    for set_name in sets:
        models = [reference.open_model(ML_DIR / "dist" / onnx_path, threads, stride=stride) for _, onnx_path in MODEL_SETS[set_name]]
        key = set_name if stride is None else f"{set_name}@stride{stride}"
        print(f"\n== {key}: stride {models[0].stride} of segment {models[0].segment}")

        sdr: dict[str, list[float]] = {name: [] for name in STEMS}
        start = time.time()
        for p in test_tracks:
            for name, value in score(separate_with_reference(models, mixes[p.name]), truths[p.name]).items():
                sdr[name].append(value)
        elapsed = time.time() - start
        summary = {name: {"median": float(np.median(v)), "mean": float(np.mean(v)), "clips": len(v)} for name, v in sdr.items()}
        summary["all"] = {"median": float(np.mean([summary[n]["median"] for n in STEMS])),
                          "mean": float(np.mean([summary[n]["mean"] for n in STEMS]))}
        print(f"  SDR vs ground truth, {len(test_tracks)} test excerpts of 7 s (reference.py, {elapsed:.0f} s):")
        for name in STEMS:
            print(f"    {name:<7s} median {summary[name]['median']:6.2f} dB   mean {summary[name]['mean']:6.2f} dB   ({summary[name]['clips']} non-silent)")
        print(f"    mean of the four medians {summary['all']['median']:.2f} dB   mean of the four means {summary['all']['mean']:.2f} dB")

        start = time.time()
        medley_stems = separate_with_reference(models, medley_mix)
        medley_seconds = time.time() - start
        medley = score(medley_stems, medley_truth)
        medley["all"] = float(np.mean([medley[n] for n in STEMS]))
        print(f"  SDR on the 70 s medley (multi-chunk): " + "  ".join(f"{k} {v:.2f}" for k, v in medley.items())
              + f"   [{medley_seconds:.0f} s, {medley_mix.shape[1] / 44100 / medley_seconds:.2f}x realtime]")
        report[key] = {"sdr": summary, "medleySdr": medley, "stride": models[0].stride}

        if torch_clips > 0:
            torch_models = [load_torch_original(model_id) for model_id, _ in MODEL_SETS[set_name]]
            sources = models[0].sources
            bag_weights = [[1.0 if name in model.targets else 0.0 for name in sources] for model in models]
            overlap = 1.0 - models[0].stride / models[0].segment
            pieces = {p.name: mixes[p.name] for p in test_tracks[:torch_clips]}
            pieces["medley of 10 excerpts (70 s)"] = medley_mix
            agreement = {}
            for label, mix in pieces.items():
                torch_stems = separate_with_torch(torch_models, bag_weights, mix, overlap)
                onnx_stems = medley_stems if label.startswith("medley") else separate_with_reference(models, mix)
                agreement[label] = {name: sdr_db(torch_stems[name], onnx_stems[name]) for name in onnx_stems}
                print(f"  reference.py vs PyTorch original, SNR dB [{label[:40]:<40s}] "
                      + "  ".join(f"{k} {v:.1f}" for k, v in agreement[label].items()))
            worst = min(v for a in agreement.values() for v in a.values())
            print(f"  worst stem SNR vs PyTorch on real music: {worst:.1f} dB")
            report[key]["agreementSnrDb"] = agreement
            report[key]["worstAgreementSnrDb"] = worst
            del torch_models
        del models

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "separation_musdb7.json"
    merged = json.loads(out_path.read_text()) if out_path.exists() else {}
    merged.update(report)
    out_path.write_text(json.dumps(merged, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "tagger", "separation"])
    parser.add_argument("--sets", nargs="+", default=["scnet_large", "scnet_small", "htdemucs_6s"], choices=list(MODEL_SETS))
    parser.add_argument("--torch-clips", type=int, default=2,
                        help="how many excerpts (plus the 70 s medley) to also run through the PyTorch original; 0 = skip")
    parser.add_argument("--stride", type=int, default=None, help="override the chunk stride (samples)")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "tagger":
        tagger_study(args.threads)
    else:
        separation_study(args.sets, args.torch_clips, args.threads, args.stride)
    return 0


if __name__ == "__main__":
    sys.exit(main())
