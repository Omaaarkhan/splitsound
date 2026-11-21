"""Check reference.py (numpy + onnxruntime) against the original PyTorch code.

    uv run --locked python check_reference.py --model scnet_large
    uv run --locked python check_reference.py --model scnet_small
    uv run --locked python check_reference.py --model htdemucs_6s
    uv run --locked python check_reference.py --model melband_roformer_vocals --wav song.wav
    uv run --locked python check_reference.py --random-init        # untrained HTDemucs, no checkpoint (CI)
    ... --wav song.wav      also compare on a 44.1 kHz file
    ... --quick             fewer track lengths and shift settings

Each row reports max |error| and SNR in dB (10*log10(sum ref^2 / sum err^2)).

  stft / istft   the numpy transforms vs the model's own torch.stft / torch.istft code
  track          reference.separate vs demucs.apply.apply_model(shifts=0, split=True) driving the
                 ORIGINAL PyTorch model, for several track lengths (shorter than a segment, a
                 one-sample last chunk, ...)
  normalize      with the track normalisation of demucs.separate / SCNet's inference
  shifts         reference.separate(shifts=K) vs apply_model(shifts=K) with its random draws
                 replaced by MAX_SHIFT - d_k

For SCNet the chunking oracle is also demucs' apply_model: the original SCNet
module is given the three attributes apply_model needs (segment, samplerate,
valid_length) so that it is chunked exactly like HTDemucs. SCNet's own
inference script uses 20 s chunks of varying length and a random shift per
chunk, which a fixed-shape graph cannot reproduce (CONTRACT.md section 3).

This is the only place where torch and the reference meet; reference.py never imports torch.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from fractions import Fraction
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import reference  # noqa: E402
import synth  # noqa: E402


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    ref64 = ref.astype(np.float64)
    noise = float(((ref64 - est.astype(np.float64)) ** 2).sum())
    if noise == 0.0:
        return float("inf")
    return 10.0 * np.log10(float((ref64 ** 2).sum()) / noise)


def compare(name: str, ref: np.ndarray, est: np.ndarray, rows: list, labels: list[str] | None = None) -> None:
    row = {"check": name, "maxAbsErr": float(np.abs(ref - est).max()), "maxAbsRef": float(np.abs(ref).max()),
           "snrDb": snr_db(ref, est)}
    per_stem = ""
    if labels is not None:
        row["snrDbPerStem"] = {label: snr_db(ref[i], est[i]) for i, label in enumerate(labels)}
        per_stem = "  " + "  ".join(f"{k} {v:.1f}" for k, v in row["snrDbPerStem"].items())
    rows.append(row)
    print(f"  {name:<46s} max|err| {row['maxAbsErr']:.3e}  (max|ref| {row['maxAbsRef']:.3f})  SNR {row['snrDb']:6.1f} dB{per_stem}")


class SCNetForApply(torch.nn.Module):
    """The original SCNet module with the attributes demucs.apply.apply_model looks for."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.sources = list(model.sources)
        self.audio_channels = model.audio_channels
        self.samplerate = reference.SAMPLE_RATE
        self.segment = Fraction(reference.SC_SEGMENT, reference.SAMPLE_RATE)       # 11 s

    def valid_length(self, length: int) -> int:
        return reference.SC_SEGMENT

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")         # torch.stft's "no window" warning
            return self.model(mix)


class MelbandForApply(torch.nn.Module):
    """Mel-Band RoFormer Core module wrapped for `demucs.apply.apply_model`."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core
        self.sources = ["vocals"]
        self.audio_channels = reference.CHANNELS
        self.samplerate = reference.SAMPLE_RATE
        self.segment = Fraction(reference.MB_SEGMENT, reference.SAMPLE_RATE)  # 8.0 s

    def valid_length(self, length: int) -> int:
        return reference.MB_SEGMENT

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            window = torch.hann_window(reference.MB_NFFT, periodic=True, device=mix.device)
            b, c, n = mix.shape
            pad_tail = reference.MB_SEGMENT - n
            if pad_tail > 0:
                mix = F.pad(mix, (0, pad_tail))
            flat = mix.reshape(b * c, reference.MB_SEGMENT)
            spec = torch.stft(
                flat, n_fft=reference.MB_NFFT, hop_length=reference.MB_HOP,
                win_length=reference.MB_NFFT, window=window, center=True, normalized=False,
                return_complex=True,
            )  # [B*C, F, T]
            spec_real = torch.view_as_real(spec).permute(0, 3, 1, 2)  # [B*C, 2, F, T]
            spec_4 = spec_real.reshape(b, c * 2, reference.MB_BINS, reference.MB_FRAMES)
            out_spec = self.core(spec_4)[:, 0]  # [B, 4, F, T]
            out_sc = out_spec.reshape(b * c, 2, reference.MB_BINS, reference.MB_FRAMES).permute(0, 2, 3, 1).contiguous()
            wave = torch.istft(
                torch.view_as_complex(out_sc), n_fft=reference.MB_NFFT, hop_length=reference.MB_HOP,
                win_length=reference.MB_NFFT, window=window, center=True, normalized=False,
                length=reference.MB_SEGMENT,
            )
            return wave.reshape(b, 1, c, reference.MB_SEGMENT)[..., :n]


def check_melband_upstream(core: torch.nn.Module, model: reference.Model, extra_wav: str | None, rows: list) -> None:
    """Compare MelBandRoformerCore + ONNX against ZFTurbo's upstream module if optional deps are installed."""
    try:
        import beartype  # noqa: F401
        import einops  # noqa: F401
        import librosa  # noqa: F401
        import rotary_embedding_torch  # noqa: F401
    except ImportError:
        print("  upstream MelBandRoformer check skipped (optional upstream packages beartype/einops/librosa/rotary_embedding_torch are not in uv.lock)")
        return
    import export_melband_roformer
    import weights

    cfg = export_melband_roformer.load_verified_config()
    src_root = weights.local_path(weights.lock_entry("melband_roformer_vocals:source")).parents[2]
    sys.path.insert(0, str(src_root))
    try:
        from models.bs_roformer.mel_band_roformer import MelBandRoformer as UpstreamRoformer  # type: ignore

        ckpt = weights.lock_entry("melband_roformer_vocals")
        weights.require_loadable(ckpt)
        sd = torch.load(weights.local_path(ckpt), map_location="cpu", weights_only=True)
        model_cfg = {k: v for k, v in cfg["model"].items() if not str(k).startswith("multi_stft")}
        model_cfg["flash_attn"] = False
        up = UpstreamRoformer(**model_cfg)
        up.load_state_dict(sd, strict=True)
        up.eval()
        clip = reference.read_wav(extra_wav)[:, :reference.MB_SEGMENT] if extra_wav else synth.synth_clip(12.0)[:, :reference.MB_SEGMENT]
        # All three see the same full segment: a file shorter than one segment is zero-padded, then trimmed back.
        n = clip.shape[1]
        wav = np.zeros((clip.shape[0], reference.MB_SEGMENT), dtype=np.float32)
        wav[:, :n] = clip
        with torch.no_grad():
            up_wave = up(torch.from_numpy(wav)[None])[0].numpy()[..., :n]  # [1, 2, n]
        core_wave = MelbandForApply(core)(torch.from_numpy(wav)[None])[0].numpy()[..., :n]
        onnx_wave = reference.melband_run_segment(model.session, wav)[None][..., :n]
        compare("upstream MelBandRoformer vs Core [segment]", up_wave, core_wave, rows)
        compare("upstream MelBandRoformer vs ONNX [segment]", up_wave, onnx_wave, rows)
    finally:
        if sys.path and sys.path[0] == str(src_root):
            sys.path.pop(0)


def torch_apply(model, mix: np.ndarray, overlap: float, shifts: int = 0) -> np.ndarray:
    from demucs.apply import apply_model

    with torch.no_grad():
        out = apply_model(model, torch.from_numpy(mix)[None], shifts=shifts, split=True, overlap=overlap,
                          transition_power=1.0, progress=False, device="cpu")
    return out[0].numpy()


def transform_checks(kind: str, torch_model, rows: list) -> None:
    rng = np.random.default_rng(7)
    clip = synth.synth_clip(12.0)
    print("STFT / ISTFT against the model's own torch code")
    if kind == "htdemucs":
        n = reference.HT_SEGMENT
        for label, segment in (("synthetic clip", clip[:, :n]),
                               ("white noise", (0.3 * rng.standard_normal((2, n))).astype(np.float32))):
            with torch.no_grad():
                torch_spec = torch_model._magnitude(torch_model._spec(torch.from_numpy(segment)[None]))[0].numpy()
            compare(f"stft [{label}]", torch_spec, reference.htdemucs_stft(segment), rows)
        spec = rng.standard_normal((4, 4, reference.HT_BINS, reference.HT_FRAMES)).astype(np.float32)
        with torch.no_grad():
            torch_wave = torch_model._ispec(torch_model._mask(None, torch.from_numpy(spec)[None]), n)[0].numpy()
        compare("istft [random complex input, DC imag != 0]", torch_wave, reference.htdemucs_istft(spec), rows)
        round_trip = reference.htdemucs_istft(reference.htdemucs_stft(clip[:, :n])[None])[0]
        compare("istft(stft(x)) vs x, whole segment (not exact by design)", clip[:, :n], round_trip, rows)
        compare("istft(stft(x)) vs x, interior only", clip[:, 4096:n - 4096], round_trip[:, 4096:n - 4096], rows)
    elif kind == "melband_roformer":
        n = reference.MB_SEGMENT
        window = torch.hann_window(reference.MB_NFFT, periodic=True)
        config = dict(n_fft=reference.MB_NFFT, hop_length=reference.MB_HOP, win_length=reference.MB_NFFT,
                      window=window, center=True, normalized=False)
        for label, segment in (("synthetic clip", clip[:, :n]),
                               ("white noise", (0.3 * rng.standard_normal((2, n))).astype(np.float32))):
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z = torch.view_as_real(torch.stft(torch.from_numpy(segment), **config, return_complex=True))  # [2, F, T, 2]
                torch_spec = z.permute(0, 3, 1, 2).reshape(4, reference.MB_BINS, reference.MB_FRAMES).numpy()
                torch_back = torch.istft(torch.view_as_complex(z.contiguous()), **config, length=n).numpy()
            mine = reference.melband_stft(segment)
            compare(f"stft [{label}]", torch_spec, mine, rows)
            back = np.stack([reference.melband_istft_channel(mine[2 * c], mine[2 * c + 1]) for c in range(2)])
            compare(f"istft [{label}] vs torch.istft", torch_back, back, rows)
            compare(f"istft(stft(x)) vs x [{label}]", segment, back, rows)
        spec = rng.standard_normal((2, reference.MB_BINS, reference.MB_FRAMES)).astype(np.float32)
        spec_i = rng.standard_normal((2, reference.MB_BINS, reference.MB_FRAMES)).astype(np.float32)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_wave = torch.istft(torch.complex(torch.from_numpy(spec), torch.from_numpy(spec_i)), **config, length=n).numpy()
        mine = np.stack([reference.melband_istft_channel(spec[c], spec_i[c]) for c in range(2)])
        compare("istft [random complex input, DC/Nyquist imag != 0]", torch_wave, mine, rows)
    else:
        n = reference.SC_SEGMENT
        config = dict(n_fft=reference.NFFT, hop_length=reference.HOP, win_length=reference.NFFT, center=True, normalized=True)
        for label, segment in (("synthetic clip", clip[:, :n]),
                               ("white noise", (0.3 * rng.standard_normal((2, n))).astype(np.float32))):
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                padded = F.pad(torch.from_numpy(segment), (0, reference.SC_TAIL))
                z = torch.view_as_real(torch.stft(padded, **config, return_complex=True))      # [2, F, T, 2]
                torch_spec = z.permute(0, 3, 1, 2).reshape(4, reference.SC_BINS, reference.SC_FRAMES).numpy()
                torch_back = torch.istft(torch.view_as_complex(z.contiguous()), **config)[:, :n].numpy()
            mine = reference.scnet_stft(segment)
            compare(f"stft [{label}]", torch_spec, mine, rows)
            back = np.stack([reference.scnet_istft_channel(mine[2 * c], mine[2 * c + 1]) for c in range(2)])
            compare(f"istft [{label}] vs torch.istft", torch_back, back, rows)
            compare(f"istft(stft(x)) vs x [{label}]", segment, back, rows)
        spec = rng.standard_normal((2, reference.SC_BINS, reference.SC_FRAMES)).astype(np.float32)
        spec_i = rng.standard_normal((2, reference.SC_BINS, reference.SC_FRAMES)).astype(np.float32)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_wave = torch.istft(torch.complex(torch.from_numpy(spec), torch.from_numpy(spec_i)), **config)[:, :n].numpy()
        mine = np.stack([reference.scnet_istft_channel(spec[c], spec_i[c]) for c in range(2)])
        compare("istft [random complex input, DC/Nyquist imag != 0]", torch_wave, mine, rows)


def track_checks(apply_target, model: reference.Model, extra_wav: str | None, quick: bool, rows: list) -> None:
    clip = synth.synth_clip(12.0)
    n, sources = model.segment, model.sources
    overlap = 1.0 - model.stride / n
    assert int((1 - overlap) * n) == model.stride, "stride is not expressible as demucs' int((1 - overlap) * N)"

    print(f"Full track vs demucs.apply.apply_model on the PyTorch model (overlap {overlap:g}), no track normalisation")
    long_clip = np.concatenate([clip, clip[:, ::-1], clip], axis=1)             # 36 s
    pieces = {"12 s synthetic": clip}
    if not quick:
        pieces["3 s (shorter than one segment)"] = clip[:, : 3 * reference.SAMPLE_RATE]
        pieces["one segment + 1 sample (1-sample last chunk)"] = long_clip[:, : n + 1]
        pieces["one stride + 1 sample"] = long_clip[:, : model.stride + 1]
        pieces["36 s synthetic (several chunks)"] = long_clip
    if extra_wav:
        pieces[f"file {Path(extra_wav).name}"] = reference.read_wav(extra_wav)
    for label, mix in pieces.items():
        ref = torch_apply(apply_target, mix, overlap)
        est = reference.separate(model, mix, shifts=0, normalize=False)
        compare(f"track [{label}]", ref, est, rows, sources)

    if model.normalize:
        print("Track normalisation (demucs.separate / SCNet inference)")
        wav = torch.from_numpy(clip.copy())
        mono = wav.mean(0)
        ref = torch_apply(apply_target, ((wav - mono.mean()) / mono.std()).numpy(), overlap)
        ref = ref * float(mono.std()) + float(mono.mean())
        compare("track normalize=True [12 s synthetic]", ref, reference.separate(model, clip, shifts=0, normalize=True), rows, sources)

    print("Deterministic shifts vs demucs' shift trick with its random draws pinned")
    import demucs.apply as demucs_apply

    for shifts in ([2] if quick else [2, 3]):
        amounts = reference.shift_amounts(shifts)
        draws = iter([reference.MAX_SHIFT - amount for amount in amounts])
        original = demucs_apply.random.randint
        demucs_apply.random.randint = lambda low, high: next(draws)     # demucs draws `offset`; d_k = MAX_SHIFT - offset
        try:
            ref = torch_apply(apply_target, clip, overlap, shifts=shifts)
        finally:
            demucs_apply.random.randint = original
        compare(f"track shifts={shifts} (d_k = {amounts})", ref, reference.separate(model, clip, shifts=shifts, normalize=False), rows, sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--random-init", action="store_true", help="untrained HTDemucs exported to a temp file")
    group.add_argument("--model", help="scnet_small | scnet_large | melband_roformer_vocals | htdemucs_6s (needs dist/<id>.onnx and the checkpoint)")
    parser.add_argument("--onnx", type=Path, help="graph to test instead of dist/<model>.onnx "
                        "(e.g. the float32-weights comparison export of an SCNet model)")
    parser.add_argument("--wav", help="additional 44.1 kHz WAV to compare on")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path, help="write the results as JSON")
    args = parser.parse_args()
    rows: list[dict] = []

    if args.random_init:
        import export_htdemucs

        torch_model = export_htdemucs.random_init_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "random.onnx"
            export_htdemucs.export_single_file(torch_model, "random", list(torch_model.sources), path)
            model = reference.open_model(path)
            transform_checks("htdemucs", torch_model, rows)
            track_checks(torch_model, model, args.wav, args.quick, rows)
        name = "random-init"
    else:
        name = args.model
        model = reference.open_model(args.onnx or ML_DIR / "dist" / f"{args.model}.onnx")
        if model.kind == "scnet":
            import export_scnet

            original, _ = export_scnet.load_scnet(args.model)
            torch_model = SCNetForApply(original)
            transform_checks(model.kind, torch_model, rows)
            track_checks(torch_model, model, args.wav, args.quick, rows)
        elif model.kind == "melband_roformer":
            import export_melband_roformer

            core, _, _ = export_melband_roformer.load_core_and_entry()
            torch_model = MelbandForApply(core)
            assert model.sources == list(torch_model.sources)
            transform_checks(model.kind, torch_model, rows)
            check_melband_upstream(core, model, args.wav, rows)
            track_checks(torch_model, model, args.wav, args.quick, rows)
        else:
            import weights

            torch_model, _ = weights.load_htdemucs(args.model)
            assert model.sources == list(torch_model.sources)
            transform_checks(model.kind, torch_model, rows)
            track_checks(torch_model, model, args.wav, args.quick, rows)

    scored = [row for row in rows if "vs x" not in row["check"]]
    worst = min(scored, key=lambda row: row["snrDb"])
    print(f"worst SNR vs PyTorch (excluding the round-trip rows): {worst['snrDb']:.1f} dB  [{worst['check']}]")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"model": name, "torch": torch.__version__, "checks": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
