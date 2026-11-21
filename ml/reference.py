"""Reference full-track stem separation: numpy + onnxruntime only (no torch).

This file is the executable specification the Rust engine mirrors. It is
written to be read: explicit loops, no cleverness. CONTRACT.md describes the
same rules in prose; the test vectors were generated with this file.

Layout:
    1. chunked separation of a whole track        (CONTRACT.md section 1, every model kind)
    2. the `htdemucs` segment transform           (section 2)
    3. the `scnet` segment transform              (section 3)
    4. the `melband_roformer` segment transform   (section 4)
    5. model loading and the command line

CLI (from the `ml/` directory):

    uv run --locked reference.py separate <model.onnx> <in.wav> <outdir>
    uv run --locked reference.py separate <a.onnx>,<b.onnx>,... <in.wav> <outdir>     # a bag of models

Options: --shifts K (deterministic test-time augmentation, default 0),
--no-normalize, --threads N, --all-stems, --stride SAMPLES (override the
model's default chunk stride). The input must already be 44.1 kHz (mono is
duplicated to stereo); this reference does not resample.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 44100
CHANNELS = 2
MAX_SHIFT = SAMPLE_RATE // 2            # 22050, range of the deterministic shifts
NFFT = 4096                             # HTDemucs and SCNet use a 4096-point FFT ...
HOP = 1024                              # ... with a hop of 1024 (Mel-Band RoFormer uses MB_NFFT / MB_HOP)


class Model:
    """An opened separation model: the ONNX session plus the constants the host needs."""

    def __init__(self, session: ort.InferenceSession, kind: str, sources: list[str], targets: list[str],
                 segment: int, stride: int, normalize: bool = True):
        self.session = session
        self.kind = kind                # "htdemucs" | "scnet" | "melband_roformer"
        self.sources = sources          # output order of the graph
        self.targets = targets          # the sources this model is trusted for
        self.segment = segment          # N: samples per model call
        self.stride = stride            # samples between chunk starts
        self.normalize = normalize      # default track normalisation flag


# ============================================================================
# 1. Chunked separation of a whole track (CONTRACT.md section 1)
# ============================================================================

def transition_weight(segment: int) -> np.ndarray:
    """Triangle of length `segment`: 1/h, 2/h, ..., 1, 1, ..., 2/h, 1/h with h = segment / 2."""
    half = segment // 2
    rising = np.arange(1, half + 1, dtype=np.float32)
    falling = np.arange(segment - half, 0, -1, dtype=np.float32)
    weight = np.concatenate([rising, falling])
    return weight / weight.max()


def separate_split(model: Model, mix: np.ndarray) -> np.ndarray:
    """[C, L] -> [S, C, L]. Overlapping segments, weighted overlap-add. No shifts, no normalisation."""
    channels, length = mix.shape
    assert length > 0
    segment_length = model.segment
    out = np.zeros((len(model.sources), channels, length), dtype=np.float32)
    sum_weight = np.zeros(length, dtype=np.float32)
    weight = transition_weight(segment_length)

    for offset in range(0, length, model.stride):
        chunk_length = min(segment_length, length - offset)

        # The model always sees `segment_length` samples. A short (last) chunk is
        # centred in the segment; the rest of the segment is filled with the real
        # audio around it where the track has any, and with zeros outside the track.
        delta = segment_length - chunk_length
        start = offset - delta // 2
        segment = np.zeros((channels, segment_length), dtype=np.float32)
        copy_from = max(0, start)
        copy_to = min(length, start + segment_length)
        segment[:, copy_from - start: copy_to - start] = mix[:, copy_from:copy_to]

        estimate = run_segment(model, segment)                          # [S, C, N]
        chunk = estimate[:, :, delta // 2: delta // 2 + chunk_length]  # centre trim

        # Note: a short chunk uses the FIRST chunk_length weights, not a re-centred triangle.
        out[:, :, offset: offset + chunk_length] += weight[:chunk_length] * chunk
        sum_weight[offset: offset + chunk_length] += weight[:chunk_length]

    return out / sum_weight


def shift_amounts(shifts: int) -> list[int]:
    """Deterministic replacement for demucs' random shift trick.

    Pass k prepends d_k = floor(k * MAX_SHIFT / passes) zeros. shifts = 0 and
    shifts = 1 both mean a single unshifted pass.
    """
    passes = max(1, shifts)
    return [(k * MAX_SHIFT) // passes for k in range(passes)]


def separate(model: Model, mix: np.ndarray, shifts: int = 0, normalize: bool | None = None) -> np.ndarray:
    """Full pipeline for one model: [2, L] float32 at 44.1 kHz -> [S, 2, L] float32."""
    if normalize is None:
        normalize = model.normalize
    mix = mix.astype(np.float32)
    channels, length = mix.shape
    assert channels == CHANNELS, "duplicate mono to stereo before calling"

    # Track-level normalisation (demucs.separate and SCNet's inference both do this).
    mean, std = np.float32(0.0), np.float32(1.0)
    if normalize:
        mono = mix.mean(axis=0)
        mean = mono.mean(dtype=np.float32)
        std = mono.std(ddof=1, dtype=np.float32) if length > 1 else np.float32(0.0)
        if not std >= 1e-8:              # digital silence (or NaN): the originals would divide by zero
            std = np.float32(1.0)
        mix = (mix - mean) / std

    amounts = shift_amounts(shifts)
    total = np.zeros((len(model.sources), channels, length), dtype=np.float32)
    for amount in amounts:
        shifted = np.concatenate([np.zeros((channels, amount), dtype=np.float32), mix], axis=1)
        estimate = separate_split(model, shifted)
        total += estimate[:, :, amount:]
    out = total / np.float32(len(amounts))

    return out * std + mean


def combine_bag(estimates: list[np.ndarray], targets: list[list[int]]) -> dict[int, np.ndarray]:
    """Combine models that share a source list (CONTRACT.md 1.4).

    estimates[m] is model m's [S, C, L]; targets[m] lists the source indices
    model m is trusted for. Each source is the mean over the models that
    target it; a source no model targets is not produced.
    Returns {source index: [C, L]}.
    """
    num_sources = estimates[0].shape[0]
    stems: dict[int, np.ndarray] = {}
    for s in range(num_sources):
        voters = [m for m in range(len(estimates)) if s in targets[m]]
        if not voters:
            continue
        total = np.zeros_like(estimates[0][s])
        for m in voters:
            total += estimates[m][s]
        stems[s] = total / np.float32(len(voters))
    return stems


def run_segment(model: Model, segment: np.ndarray) -> np.ndarray:
    """[C, N] -> [S, C, N] for one segment, according to the model kind."""
    if model.kind == "htdemucs":
        return htdemucs_run_segment(model.session, segment)
    if model.kind == "scnet":
        return scnet_run_segment(model.session, segment)
    if model.kind == "melband_roformer":
        return melband_run_segment(model.session, segment)
    raise ValueError(f"unknown model kind {model.kind!r}")


# ============================================================================
# 2. htdemucs segment transform (CONTRACT.md section 2)
# ============================================================================

HT_SEGMENT = 343980                     # 39/5 s
HT_BINS = 2048                          # bins 0..2047; the Nyquist bin is dropped
HT_FRAMES = 336                         # ceil(HT_SEGMENT / HOP)
HT_PAD = 1536                           # 3 * HOP / 2, reflect padding on the left
HT_PAD_RIGHT = HT_PAD + HT_FRAMES * HOP - HT_SEGMENT        # 1620
HT_WINDOW_SQUARE_SUM = 1.5              # sum over the 4 overlapping frames of hann^2

assert HT_FRAMES == math.ceil(HT_SEGMENT / HOP) and HT_PAD_RIGHT == 1620


def hann_window() -> np.ndarray:
    """Periodic Hann window of length NFFT (what torch.hann_window returns)."""
    n = np.arange(NFFT, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / NFFT)).astype(np.float32)


HANN = hann_window()


def htdemucs_stft(segment: np.ndarray) -> np.ndarray:
    """[C, HT_SEGMENT] float32 -> [2*C, HT_BINS, HT_FRAMES] float32, complex-as-channels."""
    channels, length = segment.shape
    assert length == HT_SEGMENT
    spec = np.zeros((2 * channels, HT_BINS, HT_FRAMES), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(NFFT))
    for c in range(channels):
        # Reflect padding without repeating the edge sample:
        #   padded = x[PAD], x[PAD-1], ..., x[1], x[0], ..., x[N-1], x[N-2], ..., x[N-1-PAD_RIGHT]
        padded = np.pad(segment[c], (HT_PAD, HT_PAD_RIGHT), mode="reflect")
        assert padded.shape[0] == (HT_FRAMES + 3) * HOP
        for j in range(HT_FRAMES):
            frame = padded[j * HOP: j * HOP + NFFT] * HANN
            bins = np.fft.rfft(frame) * scale               # 2049 complex bins
            spec[2 * c, :, j] = bins.real[:HT_BINS]         # drop the Nyquist bin
            spec[2 * c + 1, :, j] = bins.imag[:HT_BINS]
    return spec


def htdemucs_istft_channel(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """[HT_BINS, HT_FRAMES] real and imaginary parts -> [HT_SEGMENT] float32."""
    assert real.shape == (HT_BINS, HT_FRAMES) and imag.shape == (HT_BINS, HT_FRAMES)
    accumulator = np.zeros((HT_FRAMES + 3) * HOP, dtype=np.float32)
    scale = np.float32(math.sqrt(NFFT))
    bins = np.zeros(NFFT // 2 + 1, dtype=np.complex64)
    for j in range(HT_FRAMES):
        bins[:HT_BINS] = real[:, j] + 1j * imag[:, j]
        bins[HT_BINS] = 0.0                 # the Nyquist bin the model never sees is zero
        bins[0] = bins[0].real              # a real inverse FFT ignores the imaginary part of DC;
        #                                     made explicit because some FFT libraries reject it
        frame = np.fft.irfft(bins, n=NFFT).astype(np.float32) * scale * HANN
        accumulator[j * HOP: j * HOP + NFFT] += frame
    return accumulator[HT_PAD: HT_PAD + HT_SEGMENT] / np.float32(HT_WINDOW_SQUARE_SUM)


def htdemucs_istft(spec_out: np.ndarray) -> np.ndarray:
    """[S, 2*C, HT_BINS, HT_FRAMES] -> [S, C, HT_SEGMENT]."""
    sources, twice_channels = spec_out.shape[0], spec_out.shape[1]
    out = np.zeros((sources, twice_channels // 2, HT_SEGMENT), dtype=np.float32)
    for s in range(sources):
        for c in range(twice_channels // 2):
            out[s, c] = htdemucs_istft_channel(spec_out[s, 2 * c], spec_out[s, 2 * c + 1])
    return out


def htdemucs_run_segment(session: ort.InferenceSession, segment: np.ndarray) -> np.ndarray:
    """STFT, graph (spectrogram branch + waveform branch), inverse STFT, add the waveform branch."""
    spec = htdemucs_stft(segment)
    spec_out, wave_out = session.run(
        ["spec_out", "wave_out"],
        {"mix": segment[np.newaxis].astype(np.float32), "spec": spec[np.newaxis]},
    )
    return htdemucs_istft(spec_out[0]) + wave_out[0]


# ============================================================================
# 3. scnet segment transform (CONTRACT.md section 3)
# ============================================================================

SC_SEGMENT = 485100                     # 11 s
SC_BINS = 2049                          # all bins, 0..2048
SC_FRAMES = 476
SC_TAIL = 1300                          # zeros appended: (SC_SEGMENT + SC_TAIL) / HOP = 475, an odd number
SC_CENTER = NFFT // 2                   # 2048, reflect padding on both sides

assert (SC_SEGMENT + SC_TAIL) % HOP == 0 and (SC_SEGMENT + SC_TAIL) // HOP + 1 == SC_FRAMES and SC_FRAMES % 2 == 0


def scnet_stft(segment: np.ndarray) -> np.ndarray:
    """[C, SC_SEGMENT] float32 -> [2*C, SC_BINS, SC_FRAMES] float32. Rectangular window."""
    channels, length = segment.shape
    assert length == SC_SEGMENT
    spec = np.zeros((2 * channels, SC_BINS, SC_FRAMES), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(NFFT))
    for c in range(channels):
        tailed = np.concatenate([segment[c], np.zeros(SC_TAIL, dtype=np.float32)])
        padded = np.pad(tailed, (SC_CENTER, SC_CENTER), mode="reflect")
        assert padded.shape[0] == NFFT + (SC_FRAMES - 1) * HOP
        for j in range(SC_FRAMES):
            bins = np.fft.rfft(padded[j * HOP: j * HOP + NFFT]) * scale     # no window
            spec[2 * c, :, j] = bins.real
            spec[2 * c + 1, :, j] = bins.imag
    return spec


def scnet_istft_channel(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """[SC_BINS, SC_FRAMES] real and imaginary parts -> [SC_SEGMENT] float32."""
    assert real.shape == (SC_BINS, SC_FRAMES) and imag.shape == (SC_BINS, SC_FRAMES)
    accumulator = np.zeros(NFFT + (SC_FRAMES - 1) * HOP, dtype=np.float32)
    scale = np.float32(math.sqrt(NFFT))
    bins = np.zeros(SC_BINS, dtype=np.complex64)
    for j in range(SC_FRAMES):
        bins[:] = real[:, j] + 1j * imag[:, j]
        bins[0] = bins[0].real              # a real inverse FFT ignores the imaginary parts of
        bins[SC_BINS - 1] = bins[SC_BINS - 1].real      # DC and Nyquist; made explicit
        accumulator[j * HOP: j * HOP + NFFT] += np.fft.irfft(bins, n=NFFT).astype(np.float32) * scale
    out = accumulator[SC_CENTER: SC_CENTER + SC_SEGMENT].copy()
    # Divide by the number of frames overlapping each sample: 3 for the first hop, 4 afterwards.
    out[:HOP] /= np.float32(3.0)
    out[HOP:] /= np.float32(4.0)
    return out


def scnet_run_segment(session: ort.InferenceSession, segment: np.ndarray) -> np.ndarray:
    spec = scnet_stft(segment)
    spec_out = session.run(["spec_out"], {"spec": spec[np.newaxis]})[0][0]      # [S, 2*C, bins, frames]
    sources, twice_channels = spec_out.shape[0], spec_out.shape[1]
    out = np.zeros((sources, twice_channels // 2, SC_SEGMENT), dtype=np.float32)
    for s in range(sources):
        for c in range(twice_channels // 2):
            out[s, c] = scnet_istft_channel(spec_out[s, 2 * c], spec_out[s, 2 * c + 1])
    return out


# ============================================================================
# 4. melband_roformer segment transform (CONTRACT.md section 4)
# ============================================================================

MB_NFFT = 2048
MB_HOP = 441
MB_SEGMENT = 352800                     # 8 s (800 hops)
MB_BINS = MB_NFFT // 2 + 1              # 1025, bins 0..1024 (DC to Nyquist)
MB_FRAMES = 801                         # MB_SEGMENT // MB_HOP + 1
MB_CENTER = MB_NFFT // 2                # 1024, reflect padding on both sides

assert MB_SEGMENT % MB_HOP == 0 and MB_SEGMENT // MB_HOP + 1 == MB_FRAMES


def _mb_hann_window() -> np.ndarray:
    n = np.arange(MB_NFFT, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / MB_NFFT)).astype(np.float32)


def _mb_istft_envelope() -> np.ndarray:
    win_sq = MB_HANN.astype(np.float64) ** 2
    env = np.zeros(MB_NFFT + (MB_FRAMES - 1) * MB_HOP, dtype=np.float64)
    for j in range(MB_FRAMES):
        env[j * MB_HOP: j * MB_HOP + MB_NFFT] += win_sq
    return env[MB_CENTER: MB_CENTER + MB_SEGMENT].astype(np.float32)


MB_HANN = _mb_hann_window()
MB_ENV = _mb_istft_envelope()


def melband_stft(segment: np.ndarray) -> np.ndarray:
    """[C, MB_SEGMENT] float32 -> [2*C, MB_BINS, MB_FRAMES] float32. Periodic Hann, unnormalized."""
    channels, length = segment.shape
    assert length == MB_SEGMENT
    spec = np.zeros((2 * channels, MB_BINS, MB_FRAMES), dtype=np.float32)
    for c in range(channels):
        padded = np.pad(segment[c], (MB_CENTER, MB_CENTER), mode="reflect")
        assert padded.shape[0] == MB_NFFT + (MB_FRAMES - 1) * MB_HOP
        for j in range(MB_FRAMES):
            bins = np.fft.rfft(padded[j * MB_HOP: j * MB_HOP + MB_NFFT] * MB_HANN)
            spec[2 * c, :, j] = bins.real
            spec[2 * c + 1, :, j] = bins.imag
    return spec


def melband_istft_channel(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """[MB_BINS, MB_FRAMES] real and imaginary parts -> [MB_SEGMENT] float32."""
    assert real.shape == (MB_BINS, MB_FRAMES) and imag.shape == (MB_BINS, MB_FRAMES)
    accumulator = np.zeros(MB_NFFT + (MB_FRAMES - 1) * MB_HOP, dtype=np.float32)
    bins = np.zeros(MB_BINS, dtype=np.complex64)
    for j in range(MB_FRAMES):
        bins[:] = real[:, j] + 1j * imag[:, j]
        bins[0] = bins[0].real
        bins[MB_BINS - 1] = bins[MB_BINS - 1].real
        accumulator[j * MB_HOP: j * MB_HOP + MB_NFFT] += np.fft.irfft(bins, n=MB_NFFT).astype(np.float32) * MB_HANN
    return accumulator[MB_CENTER: MB_CENTER + MB_SEGMENT] / MB_ENV


def melband_istft(spec_out: np.ndarray) -> np.ndarray:
    """[S, 2*C, MB_BINS, MB_FRAMES] -> [S, C, MB_SEGMENT]."""
    sources, twice_channels = spec_out.shape[0], spec_out.shape[1]
    out = np.zeros((sources, twice_channels // 2, MB_SEGMENT), dtype=np.float32)
    for s in range(sources):
        for c in range(twice_channels // 2):
            out[s, c] = melband_istft_channel(spec_out[s, 2 * c], spec_out[s, 2 * c + 1])
    return out


def melband_run_segment(session: ort.InferenceSession, segment: np.ndarray) -> np.ndarray:
    spec = melband_stft(segment)
    spec_out = session.run(["spec_out"], {"spec": spec[np.newaxis]})[0][0]
    return melband_istft(spec_out)


# ============================================================================
# 5. Model loading and CLI
# ============================================================================

def open_model(path: str | Path, threads: int = 0, providers: list | None = None,
               stride: int | None = None) -> Model:
    """Open an exported separation graph.

    kind, sources, targets, segmentSamples and strideSamples are read from the
    metadata the exporter embeds in the ONNX file (the same values are in
    dist/manifest.json). For a graph whose weights live in a sibling file
    (htdemucs_6s), onnxruntime finds that file next to the .onnx by itself.
    """
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    if "melband" in Path(path).stem.lower():
        options.enable_mem_pattern = False
    # The graph has one symbolic dimension, "batch". On the CPU provider it is
    # left symbolic: fixing it makes onnxruntime plan one static arena for the
    # whole graph, which costs 2 to 4 GB more peak memory for no speed gain
    # (BENCHMARKS.md). Providers that need static shapes (CoreML) should fix it
    # with add_free_dimension_override_by_name("batch", 1).
    session = ort.InferenceSession(str(path), options, providers=providers or ["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    assert int(metadata["sampleRate"]) == SAMPLE_RATE
    normalize = metadata.get("normalize", "true").lower() != "false"
    return Model(session, metadata["kind"], json.loads(metadata["sources"]), json.loads(metadata["targets"]),
                 int(metadata["segmentSamples"]), stride or int(metadata["strideSamples"]), normalize=normalize)


def read_wav(path: str | Path) -> np.ndarray:
    """Read a WAV as [2, L] float32 at 44.1 kHz. Mono is duplicated; nothing is resampled."""
    import soundfile

    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)   # [L, C]
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"{path}: sample rate is {sample_rate}, expected {SAMPLE_RATE}. "
                         f"Convert first, e.g. ffmpeg -i in -ar {SAMPLE_RATE} out.wav")
    audio = np.ascontiguousarray(data.T)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if audio.shape[0] != CHANNELS:
        raise SystemExit(f"{path}: {audio.shape[0]} channels, expected 1 or 2")
    return audio


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    """Write [C, L] float32 as a 32-bit float WAV (no clipping, no dither)."""
    import soundfile

    soundfile.write(str(path), audio.T, SAMPLE_RATE, subtype="FLOAT")


def separate_file(model_paths: list[str], wav_path: str, out_dir: str, shifts: int = 0,
                  normalize: bool | None = None, threads: int = 0, all_stems: bool = False,
                  stride: int | None = None) -> dict[str, np.ndarray]:
    mix = read_wav(wav_path)
    estimates, target_indices, sources = [], [], None
    for model_path in model_paths:
        model = open_model(model_path, threads, stride=stride)
        if sources is None:
            sources = model.sources
        assert model.sources == sources, "all models of a bag must share the same source list"
        estimates.append(separate(model, mix, shifts=shifts, normalize=normalize))
        target_indices.append([sources.index(name) for name in model.targets])
        del model

    if all_stems and len(model_paths) == 1:
        target_indices = [list(range(len(sources)))]
    combined = combine_bag(estimates, target_indices)
    stems = {sources[index]: audio for index, audio in combined.items()}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        write_wav(Path(out_dir) / f"{name}.wav", audio)
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sep = commands.add_parser("separate", help="split a 44.1 kHz WAV into stems")
    sep.add_argument("model", help="path to an exported .onnx, or several separated by commas (a bag)")
    sep.add_argument("wav")
    sep.add_argument("outdir")
    sep.add_argument("--shifts", type=int, default=0)
    sep.add_argument("--no-normalize", action="store_true")
    sep.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (0 = default)")
    sep.add_argument("--stride", type=int, default=None, help="samples between chunk starts (default: the model's)")
    sep.add_argument("--all-stems", action="store_true",
                     help="write every source of a single model, not only the ones it is trusted for")
    args = parser.parse_args()

    stems = separate_file(args.model.split(","), args.wav, args.outdir, shifts=args.shifts,
                          normalize=(False if args.no_normalize else None), threads=args.threads,
                          all_stems=args.all_stems, stride=args.stride)
    for name, audio in stems.items():
        print(f"{name}: {audio.shape[1]} samples, peak {np.abs(audio).max():.4f} -> {Path(args.outdir) / (name + '.wav')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
