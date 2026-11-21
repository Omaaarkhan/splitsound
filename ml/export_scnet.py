"""Export SCNet and SCNet-large (starrytong, MIT code and MIT weights) to ONNX.

    uv run --locked python export_scnet.py                       # scnet_small and scnet_large into dist/
    uv run --locked python export_scnet.py --models scnet_small
    uv run --locked python export_scnet.py --float32-storage     # keep weights as float32 (twice the size)

Graph contract (CONTRACT.md section 3); the STFT and inverse STFT are done by the host:
    input   spec      float32 [batch, 4, 2049, 476]        rectangular-window STFT of one 11 s segment
    output  spec_out  float32 [batch, 4, 4, 2049, 476]     per-source spectrogram, de-normalised

The graph is `SCNet.forward` between its `torch.stft` and its `torch.istft`,
with two substitutions so that only ordinary ONNX operators are needed and
float32 accuracy is kept:

  * the real FFT / inverse real FFT that SCNet takes along the frame axis
    between dual-path layers (`FeatureConversion`) becomes a MatMul with a
    constant DFT matrix (the axis length is fixed at 476);
  * `GroupNorm(1, C)` computes its mean and variance one axis at a time
    instead of over C*F*T elements at once. The exporter would otherwise turn
    it into InstanceNormalization over one very long vector (up to 14 million
    elements), whose single-precision running sum loses accuracy.

Weights are stored as float16 with a Cast to float32 at load (see
BENCHMARKS.md for the measured cost) unless --float32-storage is given.
All arithmetic is float32 either way.

Checkpoints are fetched and verified by weights.py (pinned SHA-256, static
pickle scan) and loaded with `torch.load(weights_only=True)`; the model code
is imported from the pinned commit of starrytong/SCNet.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

import onnx_tools  # noqa: E402
import reference  # noqa: E402
import synth  # noqa: E402
import verify_onnx  # noqa: E402
import weights  # noqa: E402

OPSET = 17
SOURCE_COMMIT = "5d95bf96b19c3eede63248d171efeca8e3abb948"
ATTRIBUTION = ("SCNet: Sparse Compression Network for Music Source Separation, by Weinan Tong, Jiaxu Zhu, Jun Chen, "
               "Shiyin Kang, Tao Jiang, Yang Li, Zhiyong Wu and Helen Meng (ICASSP 2024). Code and pretrained weights "
               "MIT License, (c) 2024 starrytong. https://github.com/starrytong/SCNet")
MODELS = {
    # id: (config file inside the pinned source tree, default overlap)
    "scnet_small": "conf/config.yaml",
    "scnet_large": "large_config.yaml",
}
DEFAULT_STRIDE = 363825         # int(0.75 * 485100): overlap 0.25, the same rule as demucs


def source_root() -> Path:
    """Directory of the pinned SCNet source files, after verifying each one against the lock."""
    for entry in weights.read_lock()["files"]:
        if any(used.startswith("scnet") and (used.endswith(":source") or used.endswith(":config"))
               for used in entry["usedBy"]):
            weights.verify(entry)
    return weights.CACHE_DIR.parent / "scnet" / SOURCE_COMMIT


def load_scnet(model_id: str) -> tuple[nn.Module, dict]:
    """Return (original SCNet module with weights, lock entry)."""
    root = source_root()
    sys.path.insert(0, str(root))
    try:
        from scnet.SCNet import SCNet
    finally:
        sys.path.remove(str(root))
    config = yaml.safe_load((root / MODELS[model_id]).read_text())["model"]       # data only
    model = SCNet(**config)

    entry = weights.lock_entry(model_id)
    weights.require_loadable(entry)
    package = torch.load(weights.local_path(entry), map_location="cpu", weights_only=True)
    model.load_state_dict(package["best_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, entry


class AxisGroupNorm(nn.Module):
    """`GroupNorm(1, C)` with the statistics reduced one axis at a time."""

    def __init__(self, norm: nn.GroupNorm):
        super().__init__()
        assert norm.num_groups == 1 and norm.affine
        self.eps = norm.eps
        self.weight = norm.weight
        self.bias = norm.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axes = list(range(x.dim() - 1, 0, -1))              # last axis first, channel axis last
        mean = x
        for axis in axes:
            mean = mean.mean(dim=axis, keepdim=True)
        centered = x - mean
        variance = centered * centered
        for axis in axes:
            variance = variance.mean(dim=axis, keepdim=True)
        shape = [1, -1] + [1] * (x.dim() - 2)
        return centered / torch.sqrt(variance + self.eps) * self.weight.view(shape) + self.bias.view(shape)


class RealFftAsMatMul(nn.Module):
    """`torch.fft.rfft(x, dim=3, norm="ortho")` followed by cat([real, imag], dim=1), for a fixed length."""

    def __init__(self, frames: int):
        super().__init__()
        t = torch.arange(frames, dtype=torch.float64)[:, None]
        k = torch.arange(frames // 2 + 1, dtype=torch.float64)[None, :]
        angle = 2.0 * math.pi * t * k / frames
        self.register_buffer("cos", (torch.cos(angle) / math.sqrt(frames)).float())       # [T, T/2+1]
        self.register_buffer("minus_sin", (-torch.sin(angle) / math.sqrt(frames)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.matmul(x, self.cos), torch.matmul(x, self.minus_sin)], dim=1)


class InverseRealFftAsMatMul(nn.Module):
    """x[:, :C/2] + i x[:, C/2:] -> `torch.fft.irfft(., dim=3, norm="ortho")`, for a fixed output length."""

    def __init__(self, frames: int, channels: int):
        super().__init__()
        self.half = channels // 2
        k = torch.arange(frames // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(frames, dtype=torch.float64)[None, :]
        angle = 2.0 * math.pi * k * t / frames
        fold = torch.full((frames // 2 + 1, 1), 2.0, dtype=torch.float64)      # bins 1..T/2-1 appear twice
        fold[0] = 1.0
        fold[-1] = 1.0
        self.register_buffer("cos", (fold * torch.cos(angle) / math.sqrt(frames)).float())            # [T/2+1, T]
        self.register_buffer("minus_sin", (-fold * torch.sin(angle) / math.sqrt(frames)).float())     # rows 0 and T/2 are zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x[:, : self.half], self.cos) + torch.matmul(x[:, self.half:], self.minus_sin)


class SCNetCore(nn.Module):
    """`SCNet.forward` without the STFT in front and the inverse STFT behind."""

    def __init__(self, model: nn.Module, axis_group_norm: bool = True):
        super().__init__()
        self.sources = list(model.sources)
        model = copy.deepcopy(model)
        if axis_group_norm:
            for parent in list(model.modules()):
                for name, child in list(parent.named_children()):
                    if isinstance(child, nn.GroupNorm):
                        setattr(parent, name, AxisGroupNorm(child))
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.dual_path = model.separation_net.dp_modules
        conversions = []
        for index, conversion in enumerate(model.separation_net.feature_conversion):
            if conversion.inverse:
                conversions.append(InverseRealFftAsMatMul(reference.SC_FRAMES, conversion.channels))
            else:
                conversions.append(RealFftAsMatMul(reference.SC_FRAMES))
        self.conversions = nn.ModuleList(conversions)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = spec
        B, C, Fr, T = x.shape
        # Mean and (unbiased) standard deviation over channels, bins and frames,
        # reduced one axis at a time for the same reason as in AxisGroupNorm:
        # 3.9 million elements in one single-precision reduction is not accurate.
        count = reference.SC_FRAMES * reference.SC_BINS * 2 * reference.CHANNELS
        mean = x
        for axis in (3, 2, 1):
            mean = mean.mean(dim=axis, keepdim=True)
        variance = (x - mean) * (x - mean)
        for axis in (3, 2, 1):
            variance = variance.mean(dim=axis, keepdim=True)
        std = torch.sqrt(variance * (count / (count - 1.0)))
        x = (x - mean) / (1e-5 + std)

        skips, lengths, original_lengths = [], [], []
        for block in self.encoder:
            x, skip, block_lengths, block_original = block(x)
            skips.append(skip)
            lengths.append(block_lengths)
            original_lengths.append(block_original)

        for dual_path, conversion in zip(self.dual_path, self.conversions):
            x = dual_path(x)
            x = conversion(x)

        for fusion, upsample in self.decoder:
            x = fusion(x, skips.pop())
            x = upsample(x, lengths.pop(), original_lengths.pop())

        x = x.view(B, len(self.sources), -1, Fr, T)        # channel = source*4 + audio_channel*2 + (real | imag)
        return x * std[:, None] + mean[:, None]


def segment_inputs(batch: int) -> torch.Tensor:
    """Deterministic test segments [batch, 2, N]: the synthetic clip first, then noise at other levels."""
    length = reference.SC_SEGMENT
    items = [torch.from_numpy(synth.synth_clip(12.0)[:, :length].copy())]
    generator = torch.Generator().manual_seed(1234)
    for index in range(1, batch):
        items.append((0.05 * (index + 1)) * torch.randn(2, length, generator=generator))
    return torch.stack(items)


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    noise = float(((ref.astype(np.float64) - est) ** 2).sum())
    return 10.0 * np.log10(float((ref.astype(np.float64) ** 2).sum()) / max(noise, 1e-300))


def export_model(model: nn.Module, model_id: str, out_path: Path, entry: dict | None, float16_storage: bool = True,
                 dynamic_batch: bool = True, axis_group_norm: bool = True, stride: int = DEFAULT_STRIDE) -> dict:
    core = SCNetCore(model, axis_group_norm=axis_group_norm).eval()
    sources = list(model.sources)
    assert model.stft_config == {"n_fft": reference.NFFT, "hop_length": reference.HOP, "win_length": reference.NFFT,
                                 "center": True, "normalized": True}
    assert model.audio_channels == reference.CHANNELS

    mix = segment_inputs(1)
    spec = torch.from_numpy(reference.scnet_stft(mix[0].numpy()))[None]
    dynamic_axes = {"spec": {0: "batch"}, "spec_out": {0: "batch"}} if dynamic_batch else None
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.onnx"
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            torch.onnx.export(core, (spec,), str(raw_path), dynamo=False, opset_version=OPSET,
                              input_names=["spec"], output_names=["spec_out"], dynamic_axes=dynamic_axes,
                              do_constant_folding=True)
        graph_model = onnx.load(str(raw_path))

    batch_dim = "batch" if dynamic_batch else 1
    onnx_tools.declare_shapes(graph_model, {
        "spec_out": [batch_dim, len(sources), 2 * reference.CHANNELS, reference.SC_BINS, reference.SC_FRAMES]})
    storage = None
    if float16_storage:
        storage = onnx_tools.store_weights_as_float16(graph_model)
    metadata = {
        "id": model_id, "kind": "scnet", "sources": json.dumps(sources), "targets": json.dumps(sources),
        "sampleRate": reference.SAMPLE_RATE, "channels": reference.CHANNELS,
        "segmentSamples": reference.SC_SEGMENT, "strideSamples": stride,
        "nfft": reference.NFFT, "hop": reference.HOP, "license": "MIT", "attribution": ATTRIBUTION,
    }
    if entry is not None:
        metadata["origin"] = "https://github.com/starrytong/SCNet"
        metadata["originSha256"] = entry["sha256"]
    onnx_tools.set_metadata(graph_model, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph_model, str(out_path))
    weight_bytes = onnx_tools.initializer_stats(graph_model)
    del graph_model

    inert = verify_onnx.verify(out_path)
    if inert["violations"]:
        raise RuntimeError(f"{out_path.name} failed verify_onnx: {inert['violations']}")

    # ---- parity ----------------------------------------------------------------
    # torch reference = the ORIGINAL module (its own torch.stft / istft, its own
    # GroupNorm and rfft). ONNX side = numpy STFT -> graph -> numpy ISTFT.
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    parity = {}
    for batch in ([1, 2] if dynamic_batch else [1]):
        mix = segment_inputs(batch)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_full = model(mix).numpy()                                         # [B, S, 2, N]
        spec = np.stack([reference.scnet_stft(mix[b].numpy()) for b in range(batch)])
        onnx_spec_out = session.run(["spec_out"], {"spec": spec})[0]
        onnx_full = np.zeros_like(torch_full)
        for b in range(batch):
            for s in range(len(sources)):
                for c in range(reference.CHANNELS):
                    onnx_full[b, s, c] = reference.scnet_istft_channel(onnx_spec_out[b, s, 2 * c], onnx_spec_out[b, s, 2 * c + 1])
        parity[f"batch{batch}"] = {
            "endToEndMaxAbsErr": float(np.abs(onnx_full - torch_full).max()),
            "endToEndMaxAbs": float(np.abs(torch_full).max()),
            "endToEndSnrDb": snr_db(torch_full, onnx_full),
            "snrDbPerStem": {name: snr_db(torch_full[:, s], onnx_full[:, s]) for s, name in enumerate(sources)},
        }
    del session

    return {
        "id": model_id, "kind": "scnet", "file": out_path.name, "sha256": inert["sha256"], "sizeBytes": inert["sizeBytes"],
        "sources": sources, "targets": sources, "sampleRate": reference.SAMPLE_RATE, "channels": reference.CHANNELS,
        "segmentSamples": reference.SC_SEGMENT, "strideSamples": stride, "nfft": reference.NFFT, "hop": reference.HOP,
        "bins": reference.SC_BINS, "frames": reference.SC_FRAMES, "dynamicBatch": dynamic_batch,
        "weightStorage": "float16" if float16_storage else "float32", "float16Storage": storage,
        "initializerBytesByType": weight_bytes, "axisGroupNorm": axis_group_norm,
        "opset": OPSET, "exporter": "torch.onnx.export(dynamo=False) [TorchScript]",
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "parameters": sum(p.numel() for p in model.parameters()),
        "origin": entry["url"] if entry else None, "originSha256": entry["sha256"] if entry else None,
        "sourceCommit": SOURCE_COMMIT, "parity": parity,
        "onnxVerify": {k: inert[k] for k in ("opset", "domainsUsed", "nodeCount", "opTypes", "initializerCount",
                                             "externalData", "subgraphs", "localFunctions", "checker", "inputs",
                                             "outputs")},
    }


def print_report(report: dict) -> None:
    print(f"{report['file']}: {report['sizeBytes'] / 1e6:.1f} MB ({report['weightStorage']} weights)  "
          f"sha256={report['sha256']}  params {report['parameters'] / 1e6:.2f} M  axisGroupNorm {report['axisGroupNorm']}")
    if report["float16Storage"]:
        s = report["float16Storage"]
        print(f"  float16 storage: {s['tensors']} tensors, max rounding error {s['maxAbsRoundingError']:.2e} "
              f"(relative to tensor peak {s['maxRelRoundingError']:.2e}), overflowed {s['overflowed']}, "
              f"underflowed to zero {s['underflowedToZero']}")
    for name, p in report["parity"].items():
        stems = "  ".join(f"{k} {v:.1f}" for k, v in p["snrDbPerStem"].items())
        print(f"  {name}: end-to-end vs original SCNet.forward max|err| {p['endToEndMaxAbsErr']:.3e} "
              f"(max|x| {p['endToEndMaxAbs']:.3f}) = {p['endToEndSnrDb']:.1f} dB SNR   [{stems}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--out", type=Path, default=ML_DIR / "dist")
    parser.add_argument("--float32-storage", action="store_true", help="store weights as float32 instead of float16")
    parser.add_argument("--native-group-norm", action="store_true",
                        help="export GroupNorm as the exporter does it (for comparison only)")
    parser.add_argument("--suffix", default="", help="appended to the file stem, for comparison exports")
    args = parser.parse_args()

    for model_id in args.models:
        model, entry = load_scnet(model_id)
        report = export_model(model, model_id, args.out / f"{model_id}{args.suffix}.onnx", entry,
                              float16_storage=not args.float32_storage,
                              axis_group_norm=not args.native_group_norm)
        if not args.suffix:
            (args.out / f"{model_id}.export.json").write_text(json.dumps(report, indent=2) + "\n")
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
