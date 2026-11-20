"""Export HTDemucs v4 (Meta) to ONNX with the STFT / inverse STFT outside the graph.

    uv run --locked python export_htdemucs.py                         # htdemucs_6s: weight-free graph + tensor table
    uv run --locked python export_htdemucs.py --single-file htdemucs_6s htdemucs htdemucs_ft
                                                                      # self-contained graphs into dist/compare/ (never shipped)
    uv run --locked python export_htdemucs.py --self-test             # random weights, no checkpoint, for CI

Licensing drives the shape of this script. Demucs' CODE is MIT, its WEIGHTS
are not (LICENSES.md), so SplitSound does not distribute them in any form.
What is distributed for `htdemucs_6s` is:

    dist/htdemucs_6s.onnx          the graph only. Every weight is an initializer whose data lives in
                                   an external file `htdemucs_6s.weights` that is NOT shipped.
    dist/htdemucs_6s.tensors.json  for each weight: state-dict key, shape, offset and length in that file,
                                   plus the pinned SHA-256 of Meta's checkpoint and of the weights file.

The end user's machine downloads Meta's checkpoint from Meta and writes the
weights file itself (convert_reference.py is the reference for that step).

To make that possible the graph is exported with constant folding off, so
each initializer is exactly one state-dict tensor (as float32) and nothing
derived from weights is baked into the graph. This script proves it: every
initializer must equal its state-dict tensor bit for bit, and no other tensor
in the graph may hold more than a few hundred values.

Graph contract (CONTRACT.md section 2):
    inputs   mix       float32 [batch, 2, 343980]
             spec      float32 [batch, 4, 2048, 336]
    outputs  spec_out  float32 [batch, S, 4, 2048, 336]
             wave_out  float32 [batch, S, 2, 343980]

The graph is `HTDemucs.forward` from just after `_spec`/`_magnitude` to just
before `_mask`/`_ispec`, with mean/variance reductions done one axis at a
time (single-precision reductions over millions of elements are inaccurate in
onnxruntime; see BENCHMARKS.md).

Checkpoints are read only through the symbolic reader (weights.load_htdemucs);
no Demucs file is ever unpickled.
"""
from __future__ import annotations

import argparse
import hashlib
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
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from onnx import TensorProto, numpy_helper  # noqa: E402
from torch import nn  # noqa: E402

import convert_reference  # noqa: E402
import onnx_tools  # noqa: E402
import reference  # noqa: E402
import synth  # noqa: E402
import verify_onnx  # noqa: E402
import weights  # noqa: E402

OPSET = 17
STRIDE = 257985                      # int(0.75 * 343980): demucs' default overlap of 0.25
WEIGHTS_FILE = "htdemucs_6s.weights"
MAX_INLINE_ELEMENTS = 256            # no tensor left inside the weight-free graph may be larger than this
ATTRIBUTION = ("HTDemucs v4 (Hybrid Transformer Demucs) by Simon Rouard, Francisco Massa and Alexandre Defossez, "
               "Meta Platforms, Inc. Code: MIT License, https://github.com/facebookresearch/demucs . "
               "The pretrained weights are NOT under the MIT License; see LICENSES.md.")

# bag name -> exported graph ids, in the order of the bag's `models` list
BAGS = {
    "htdemucs_6s": ["htdemucs_6s"],
    "htdemucs": ["htdemucs"],
    "htdemucs_ft": ["htdemucs_ft_drums", "htdemucs_ft_bass", "htdemucs_ft_other", "htdemucs_ft_vocals"],
}


class AxisGroupNorm(nn.Module):
    """`GroupNorm(1, C)` with mean and variance reduced one axis at a time.

    `channels_last=True` reproduces demucs' `MyGroupNorm`, which normalises a
    [B, T, C] tensor by moving C to axis 1 and back.
    """

    def __init__(self, norm: nn.GroupNorm, channels_last: bool):
        super().__init__()
        assert norm.num_groups == 1 and norm.affine
        self.eps = norm.eps
        self.weight = norm.weight           # same attribute names, so state-dict keys are unchanged
        self.bias = norm.bias
        self.channels_last = channels_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            x = x.transpose(1, 2)
        axes = list(range(x.dim() - 1, 0, -1))
        mean = x
        for axis in axes:
            mean = mean.mean(dim=axis, keepdim=True)
        centered = x - mean
        variance = centered * centered
        for axis in axes:
            variance = variance.mean(dim=axis, keepdim=True)
        shape = [1, -1] + [1] * (x.dim() - 2)
        x = centered / torch.sqrt(variance + self.eps) * self.weight.view(shape) + self.bias.view(shape)
        if self.channels_last:
            x = x.transpose(1, 2)
        return x


def axis_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean over every axis but the batch, one axis at a time, keeping dimensions."""
    for axis in range(x.dim() - 1, 0, -1):
        x = x.mean(dim=axis, keepdim=True)
    return x


def axis_mean_std(x: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """What `x.mean(dims)` and the unbiased `x.std(dims)` compute, reduced one axis at a time."""
    mean = axis_mean(x)
    variance = axis_mean((x - mean) * (x - mean)) * (count / (count - 1.0))
    return mean, torch.sqrt(variance)


class HTDemucsCore(nn.Module):
    """`HTDemucs.forward` without the STFT at the front and the mask + ISTFT at the back.

    The body is the evaluation path of demucs 4.0.1 `HTDemucs.forward`
    (`use_train_segment=True`, input already of training length), line for
    line, with `spec` supplied by the caller instead of computed.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        assert model.cac, "only complex-as-channels models are supported"
        assert model.use_train_segment
        for parent in list(model.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.GroupNorm) and child.num_groups == 1:
                    setattr(parent, name, AxisGroupNorm(child, channels_last=type(child).__name__ == "MyGroupNorm"))
        self.model = model

    def forward(self, mix: torch.Tensor, spec: torch.Tensor):
        m = self.model
        length = mix.shape[-1]
        x = spec
        B, C, Fq, T = x.shape

        # Normalisation of the frequency branch input (per batch item).
        mean, std = axis_mean_std(x, 2 * reference.CHANNELS * reference.HT_BINS * reference.HT_FRAMES)
        x = (x - mean) / (1e-5 + std)

        # Normalisation of the time branch input.
        xt = mix
        meant, stdt = axis_mean_std(xt, reference.CHANNELS * reference.HT_SEGMENT)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)

        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_downsampler_t(xt)

        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        assert len(saved) == 0 and len(lengths_t) == 0 and len(saved_t) == 0

        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]          # spec_out, de-normalised
        xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]      # wave_out, de-normalised
        return x, xt


def random_init_model(sources: list[str] | None = None) -> nn.Module:
    """An untrained HTDemucs with the architecture of the released `htdemucs`. No files involved."""
    from demucs.htdemucs import HTDemucs

    torch.manual_seed(0)
    model = HTDemucs(sources=sources or ["drums", "bass", "other", "vocals"], segment=Fraction(39, 5),
                     bottom_channels=512, dconv_mode=3, t_layers=5)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def bag_targets(bag: str) -> list[list[int] | None]:
    """Per sub-model source indices with a non-zero bag weight (None = all sources).

    Read from the bag definition shipped inside the hash-locked demucs package
    (`demucs/remote/<bag>.yaml`), together with the checkpoint signatures.
    """
    import demucs
    import yaml

    definition = yaml.safe_load((Path(demucs.__file__).parent / "remote" / f"{bag}.yaml").read_text())
    signatures = definition["models"]
    bag_weights = definition.get("weights")
    result = []
    for index, model_id in enumerate(BAGS[bag]):
        entry = weights.lock_entry(model_id)
        assert entry["name"].startswith(signatures[index] + "-"), (entry["name"], signatures[index])
        result.append(None if bag_weights is None else [k for k, w in enumerate(bag_weights[index]) if w > 0])
    return result


def verification_inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic segments [batch, 2, N] and their STFT (computed by the numpy reference)."""
    length = reference.HT_SEGMENT
    items = [torch.from_numpy(synth.synth_clip(12.0)[:, :length].copy())]
    generator = torch.Generator().manual_seed(1234)
    for index in range(1, batch):
        items.append((0.05 * (index + 1)) * torch.randn(2, length, generator=generator))
    mix = torch.stack(items)
    spec = torch.from_numpy(np.stack([reference.htdemucs_stft(item.numpy()) for item in items]))
    return mix, spec


def trace(core: nn.Module, raw_path: Path, constant_folding: bool, dynamic_batch: bool) -> None:
    mix, spec = verification_inputs(1)
    dynamic_axes = {name: {0: "batch"} for name in ("mix", "spec", "spec_out", "wave_out")} if dynamic_batch else None
    # nn.MultiheadAttention has an inference "fast path" that calls one fused
    # native op (aten::_native_multi_head_attention) which has no ONNX mapping.
    # Turning it off for the trace makes the layers run their ordinary
    # MatMul / Softmax implementation; the parity check compares against the
    # fast path anyway.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            torch.onnx.export(core, (mix, spec), str(raw_path), dynamo=False, opset_version=OPSET,
                              input_names=["mix", "spec"], output_names=["spec_out", "wave_out"],
                              dynamic_axes=dynamic_axes, do_constant_folding=constant_folding)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def graph_metadata(model: nn.Module, model_id: str, targets: list[str], extra: dict | None) -> dict:
    metadata = {
        "id": model_id, "kind": "htdemucs", "sources": json.dumps(list(model.sources)), "targets": json.dumps(targets),
        "sampleRate": model.samplerate, "channels": model.audio_channels,
        "segmentSamples": reference.HT_SEGMENT, "strideSamples": STRIDE,
        "nfft": model.nfft, "hop": model.hop_length, "license": "MIT (graph only; weights excluded)",
        "attribution": ATTRIBUTION,
    }
    metadata.update(extra or {})
    return metadata


def parity(model: nn.Module, session: ort.InferenceSession, dynamic_batch: bool) -> dict:
    """ONNX (numpy STFT -> graph -> numpy ISTFT) against the untouched `HTDemucs.forward`."""
    results = {}
    sources = list(model.sources)
    for batch in ([1, 2] if dynamic_batch else [1]):
        mix, spec = verification_inputs(batch)
        with torch.no_grad():
            torch_full = model(mix).numpy()
        spec_out, wave_out = session.run(["spec_out", "wave_out"], {"mix": mix.numpy(), "spec": spec.numpy()})
        onnx_full = np.stack([reference.htdemucs_istft(spec_out[b]) + wave_out[b] for b in range(batch)])

        def snr(ref: np.ndarray, est: np.ndarray) -> float:
            noise = float(((ref.astype(np.float64) - est) ** 2).sum())
            return 10.0 * np.log10(float((ref.astype(np.float64) ** 2).sum()) / max(noise, 1e-300))

        results[f"batch{batch}"] = {
            "endToEndMaxAbsErr": float(np.abs(onnx_full - torch_full).max()),
            "endToEndMaxAbs": float(np.abs(torch_full).max()),
            "endToEndSnrDb": snr(torch_full, onnx_full),
            "snrDbPerStem": {name: snr(torch_full[:, s], onnx_full[:, s]) for s, name in enumerate(sources)},
        }
    return results


def base_report(model: nn.Module, model_id: str, targets: list[str], inert: dict, dynamic_batch: bool) -> dict:
    return {
        "id": model_id, "kind": "htdemucs", "file": inert["file"], "sha256": inert["sha256"], "sizeBytes": inert["sizeBytes"],
        "sources": list(model.sources), "targets": targets, "sampleRate": model.samplerate,
        "channels": model.audio_channels, "segmentSamples": reference.HT_SEGMENT, "strideSamples": STRIDE,
        "nfft": model.nfft, "hop": model.hop_length, "bins": reference.HT_BINS, "frames": reference.HT_FRAMES,
        "dynamicBatch": dynamic_batch, "opset": OPSET, "exporter": "torch.onnx.export(dynamo=False) [TorchScript]",
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                     "numpy": np.__version__},
        "parameters": sum(p.numel() for p in model.parameters()),
        "onnxVerify": {k: inert[k] for k in ("opset", "domainsUsed", "nodeCount", "opTypes", "initializerCount",
                                             "externalData", "subgraphs", "localFunctions", "checker", "inputs",
                                             "outputs")},
    }


def output_shapes(model: nn.Module, dynamic_batch: bool) -> dict:
    batch_dim = "batch" if dynamic_batch else 1
    count = len(model.sources)
    return {"spec_out": [batch_dim, count, 2 * reference.CHANNELS, reference.HT_BINS, reference.HT_FRAMES],
            "wave_out": [batch_dim, count, reference.CHANNELS, reference.HT_SEGMENT]}


# ----------------------------------------------------------------------------
# A. self-contained graph (weights inside). Comparison only; never distributed.
# ----------------------------------------------------------------------------

def export_single_file(model: nn.Module, model_id: str, targets: list[str], out_path: Path,
                       dynamic_batch: bool = True, extra_metadata: dict | None = None) -> dict:
    original = model
    core = HTDemucsCore(_copy(model)).eval()
    with tempfile.TemporaryDirectory() as tmp:
        trace(core, Path(tmp) / "raw.onnx", constant_folding=True, dynamic_batch=dynamic_batch)
        graph_model = onnx.load(str(Path(tmp) / "raw.onnx"))
    onnx_tools.declare_shapes(graph_model, output_shapes(original, dynamic_batch))
    onnx_tools.set_metadata(graph_model, graph_metadata(original, model_id, targets, extra_metadata))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph_model, str(out_path))
    del graph_model

    inert = verify_onnx.verify(out_path)
    if inert["violations"]:
        raise RuntimeError(f"{out_path.name} failed verify_onnx: {inert['violations']}")
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    report = base_report(original, model_id, targets, inert, dynamic_batch)
    report["parity"] = parity(original, session, dynamic_batch)
    return report


def _copy(model: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(model)


# ----------------------------------------------------------------------------
# B. weight-free graph + tensor table (what is distributed for htdemucs_6s)
# ----------------------------------------------------------------------------

def export_skeleton(model: nn.Module, model_id: str, targets: list[str], out_dir: Path, entry: dict,
                    dynamic_batch: bool = True) -> dict:
    original = model
    core = HTDemucsCore(_copy(model)).eval()
    state = {key: value.detach().numpy() for key, value in original.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        trace(core, Path(tmp) / "raw.onnx", constant_folding=False, dynamic_batch=dynamic_batch)
        graph_model = onnx.load(str(Path(tmp) / "raw.onnx"))
    graph = graph_model.graph

    # 1. Every initializer must be exactly one state-dict tensor. Rename it to the state-dict key.
    renames = {}
    for init in graph.initializer:
        assert init.name.startswith("model."), f"initializer {init.name!r} is not a state-dict tensor"
        key = init.name[len("model."):]
        assert key in state, f"initializer {init.name!r} has no state-dict entry"
        assert init.data_type == TensorProto.FLOAT
        assert np.array_equal(numpy_helper.to_array(init), state[key]), f"{key}: graph data differs from the checkpoint"
        renames[init.name] = key
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name in renames:
                node.input[index] = renames[name]
    present = set(renames.values())

    # 2. Nothing else in the graph may carry data of any size.
    largest_inline = 0
    for node in graph.node:
        for attribute in node.attribute:
            tensors = [attribute.t] if attribute.type == onnx.AttributeProto.TENSOR else list(attribute.tensors)
            for tensor in tensors:
                elements = int(np.prod(tensor.dims)) if tensor.dims else 1
                largest_inline = max(largest_inline, elements)
                assert elements <= MAX_INLINE_ELEMENTS, f"{node.name}: constant with {elements} elements in the graph"

    # 3. Lay the weights out in checkpoint order and point the initializers at the external file.
    archive, _root, checkpoint_tensors = convert_reference.read_checkpoint(weights.local_path(entry))
    table_rows, offset = [], 0
    layout = {}
    digest = hashlib.sha256()
    for tensor in checkpoint_tensors:
        if tensor.key not in present:
            continue                                    # a parameter the inference graph does not use
        data = np.ascontiguousarray(state[tensor.key].astype("<f4")).tobytes()
        layout[tensor.key] = (offset, len(data))
        table_rows.append({"key": tensor.key, "shape": list(tensor.shape), "sourceDtype": "float16",
                           "offset": offset, "byteLength": len(data)})
        digest.update(data)
        offset += len(data)
    assert set(layout) == present
    unused = [tensor.key for tensor in checkpoint_tensors if tensor.key not in present]

    for init in graph.initializer:
        key = renames[init.name]
        init.name = key
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for field, value in (("location", WEIGHTS_FILE), ("offset", layout[key][0]), ("length", layout[key][1])):
            item = init.external_data.add()
            item.key, item.value = field, str(value)

    onnx_tools.declare_shapes(graph_model, output_shapes(original, dynamic_batch))
    onnx_tools.set_metadata(graph_model, graph_metadata(original, model_id, targets, {
        "weightsFile": WEIGHTS_FILE, "weightsOrigin": entry["url"], "weightsOriginSha256": entry["sha256"]}))
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / f"{model_id}.onnx"
    graph_path.write_bytes(graph_model.SerializeToString())     # weights are external: nothing to embed
    del graph_model

    table = {
        "schemaVersion": 1,
        "model": model_id,
        "weightsFile": WEIGHTS_FILE,
        "layout": "tensors in this order, each little-endian float32 in row-major order at `offset`; no header, no padding",
        "upstream": {"url": entry["url"], "sha256": entry["sha256"], "sizeBytes": entry["sizeBytes"],
                     "format": "torch-zip", "storageDtype": "float16"},
        "weights": {"sha256": digest.hexdigest(), "sizeBytes": offset},
        "tensors": table_rows,
    }
    table_path = out_dir / f"{model_id}.tensors.json"
    table_path.write_text(json.dumps(table, indent=1) + "\n")

    # 4. Produce the weights file the way an end user's machine does: from Meta's
    #    file with the torch-free converter. It must reproduce the pinned hash.
    weights_path = out_dir / WEIGHTS_FILE
    converted = convert_reference.convert(weights.local_path(entry), table_path, weights_path)
    assert converted == table["weights"], (converted, table["weights"])

    inert = verify_onnx.verify(graph_path, external_data=WEIGHTS_FILE, max_inline_bytes=65536)
    if inert["violations"]:
        raise RuntimeError(f"{graph_path.name} failed verify_onnx: {inert['violations']}")

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    report = base_report(original, model_id, targets, inert, dynamic_batch)
    report["parity"] = parity(original, session, dynamic_batch)
    report["weights"] = {"file": WEIGHTS_FILE, **table["weights"], "tensors": table_path.name,
                         "tensorCount": len(table_rows), "unusedCheckpointTensors": unused,
                         "largestInlineConstantElements": largest_inline,
                         "upstream": table["upstream"]}
    report["tensorTableSha256"] = onnx_tools.sha256_file(table_path)
    return report


def print_report(report: dict) -> None:
    print(f"{report['file']}: {report['sizeBytes'] / 1e6:.2f} MB  sha256={report['sha256']}")
    print(f"  sources {report['sources']}  targets {report['targets']}  opset {report['opset']}  "
          f"dynamicBatch {report['dynamicBatch']}  nodes {report['onnxVerify']['nodeCount']}")
    if "weights" in report:
        w = report["weights"]
        print(f"  weights file {w['file']}: {w['sizeBytes']} bytes  sha256={w['sha256']}  "
              f"({w['tensorCount']} tensors, largest constant left in the graph: {w['largestInlineConstantElements']} elements)")
    for name, p in report["parity"].items():
        stems = "  ".join(f"{k} {v:.1f}" for k, v in p["snrDbPerStem"].items())
        print(f"  {name}: end-to-end vs HTDemucs.forward max|err| {p['endToEndMaxAbsErr']:.3e} "
              f"(max|x| {p['endToEndMaxAbs']:.3f}) = {p['endToEndSnrDb']:.1f} dB SNR   [{stems}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ML_DIR / "dist")
    parser.add_argument("--single-file", nargs="+", choices=list(BAGS), metavar="BAG",
                        help="export self-contained comparison graphs into <out>/compare instead")
    parser.add_argument("--self-test", action="store_true",
                        help="export a randomly initialised model to a temporary directory and verify it")
    args = parser.parse_args()

    if args.self_test:
        with tempfile.TemporaryDirectory() as tmp:
            report = export_single_file(random_init_model(), "selftest", ["drums", "bass", "other", "vocals"],
                                        Path(tmp) / "selftest.onnx")
            print_report(report)
        return 0

    if args.single_file:
        for bag in args.single_file:
            for model_id, target_indices in zip(BAGS[bag], bag_targets(bag)):
                model, entry = weights.load_htdemucs(model_id)
                sources = list(model.sources)
                targets = sources if target_indices is None else [sources[k] for k in target_indices]
                report = export_single_file(model, model_id, targets, args.out / "compare" / f"{model_id}.onnx",
                                            extra_metadata={"weightsOrigin": entry["url"]})
                (args.out / "compare" / f"{model_id}.export.json").write_text(json.dumps(report, indent=2) + "\n")
                print_report(report)
                del model
        return 0

    model, entry = weights.load_htdemucs("htdemucs_6s")
    report = export_skeleton(model, "htdemucs_6s", list(model.sources), args.out, entry)
    (args.out / "htdemucs_6s.export.json").write_text(json.dumps(report, indent=2) + "\n")
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
