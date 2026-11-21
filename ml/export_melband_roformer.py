"""Export Mel-Band RoFormer Vocals (KimberleyJensen, MIT weights and ZFTurbo/lucidrains MIT code) to ONNX.

    uv run --locked python export_melband_roformer.py                 # float16-storage into dist/
    uv run --locked python export_melband_roformer.py --float32-storage --out dist/compare --suffix _f32

Graph contract (CONTRACT.md section 4); the STFT and inverse STFT are done by the host:
    input   spec      float32 [batch, 4, 1025, 801]     Hann-window STFT (unnormalized) of one 8 s segment
    output  spec_out  float32 [batch, 1, 4, 1025, 801]  vocal spectrogram with DC bin zeroed

Checkpoints and upstream config are verified by weights.py (pinned SHA-256,
static pickle scan) and loaded with `torch.load(weights_only=True)`.

Upstream MIT License Notices (architecture adapted from):
- ZFTurbo/Music-Source-Separation-Training (commit 1f700c386e0dd763fe0a57242bb9420dce987354)
  MIT License — Copyright (c) 2024 Roman Solovyev (ZFTurbo)
- lucidrains/BS-RoFormer
  MIT License — Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import argparse
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
from torch.nn import functional as F  # noqa: E402

import onnx_tools  # noqa: E402
import reference  # noqa: E402
import synth  # noqa: E402
import verify_onnx  # noqa: E402
import weights  # noqa: E402

OPSET = 17
MODEL_ID = "melband_roformer_vocals"
SOURCE_COMMIT = "1f700c386e0dd763fe0a57242bb9420dce987354"
ATTRIBUTION = ("Mel-Band RoFormer vocal separation model by KimberleyJensen (MIT License) "
               "and ZFTurbo/Music-Source-Separation-Training (MIT License). "
               "https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model")
DEFAULT_STRIDE = 176400  # 352800 // 2 (config num_overlap: 2 -> 4 s hop)


class _ConfigSafeLoader(yaml.SafeLoader):
    """Restricted YAML SafeLoader allowing only !!python/tuple used in upstream loss resolution lists."""


_ConfigSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def load_verified_config() -> dict:
    """Verify pinned upstream config/source entries and parse the config YAML safely."""
    for entry in weights.read_lock()["files"]:
        if any(used.startswith(f"{MODEL_ID}:") for used in entry["usedBy"]):
            weights.verify(entry)
    config_entry = weights.lock_entry(f"{MODEL_ID}:config")
    cfg = yaml.load(weights.local_path(config_entry).read_text(), Loader=_ConfigSafeLoader)
    model_cfg = cfg["model"]
    audio_cfg = cfg["audio"]
    assert audio_cfg["chunk_size"] == reference.MB_SEGMENT
    assert audio_cfg["sample_rate"] == reference.SAMPLE_RATE
    assert model_cfg["stft_n_fft"] == reference.MB_NFFT
    assert model_cfg["stft_hop_length"] == reference.MB_HOP
    assert model_cfg["stft_win_length"] == reference.MB_NFFT
    assert model_cfg["stft_normalized"] is False
    assert model_cfg["dim_freqs_in"] == reference.MB_BINS
    assert model_cfg["stereo"] is True
    assert model_cfg["num_stems"] == 1
    assert model_cfg.get("linear_transformer_depth", 0) == 0
    return cfg


def slaney_mel_freqs_per_band(sample_rate: int = 44100, n_fft: int = 2048, num_bands: int = 60) -> np.ndarray:
    """Exact Slaney-scale triangular Mel filterbank boolean mask [num_bands, n_fft//2 + 1].

    Matches librosa.filters.mel(sr=44100, n_fft=2048, n_mels=60) with [0,0]=1 and [-1,-1]=1.
    """
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
        arr = np.asanyarray(hz, dtype=np.float64)
        mels = (arr - f_min) / f_sp
        return np.where(arr >= min_log_hz, min_log_mel + np.log(np.maximum(arr, min_log_hz) / min_log_hz) / logstep, mels)

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        hz = f_min + f_sp * mels
        return np.where(mels >= min_log_mel, min_log_hz * np.exp(logstep * (mels - min_log_mel)), hz)

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), num_bands + 2)
    mel_f = mel_to_hz(mel_points)

    fb = np.zeros((num_bands, len(fftfreqs)), dtype=np.float64)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(num_bands):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2: num_bands + 2] - mel_f[:num_bands])
    fb *= enorm[:, np.newaxis]
    fb = fb.astype(np.float32)
    fb[0, 0] = 1.0
    fb[-1, -1] = 1.0
    mask = fb > 0.0
    assert mask.any(axis=0).all(), "every frequency bin must belong to at least one band"
    return mask


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True)).clamp(min=1e-12)
        return (x / norm) * self.scale * self.gamma


class RotaryStatic(nn.Module):
    """Static-sequence interleaved-pair RotaryEmbedding matching rotary_embedding_torch."""

    def __init__(self, dim_head: int, seq_len: int):
        super().__init__()
        self.dim_head = dim_head
        self.seq_len = seq_len
        self.register_buffer("freqs", torch.zeros(dim_head // 2))
        self.register_buffer("cos_buf", torch.ones(1, 1, seq_len, dim_head), persistent=False)
        self.register_buffer("sin_buf", torch.zeros(1, 1, seq_len, dim_head), persistent=False)

    def bake(self) -> None:
        t = torch.arange(self.seq_len, dtype=self.freqs.dtype, device=self.freqs.device)
        angles = torch.outer(t, self.freqs)  # [seq_len, dim_head // 2]
        angles = angles.unsqueeze(-1).expand(-1, -1, 2).reshape(self.seq_len, self.dim_head)
        self.cos_buf.copy_(torch.cos(angles)[None, None])
        self.sin_buf.copy_(torch.sin(angles)[None, None])

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_eff, heads, seq_len, dim_head]
        pairs = x.reshape(-1, x.shape[1], self.seq_len, self.dim_head // 2, 2)
        x0 = pairs[..., 0]
        x1 = pairs[..., 1]
        rotated = torch.stack((-x1, x0), dim=-1).reshape(-1, x.shape[1], self.seq_len, self.dim_head)
        return x * self.cos_buf + rotated * self.sin_buf


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_embed: RotaryStatic):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.seq_len = rotary_embed.seq_len
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_eff, seq_len, dim]
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(-1, self.seq_len, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rotary_embed.rotate(q)
        k = self.rotary_embed.rotate(k)

        if self.seq_len > 128:
            head_outs = []
            for h in range(self.heads):
                q_h = q[:, h]
                k_h = k[:, h]
                v_h = v[:, h]
                attn_h = torch.softmax(torch.matmul(q_h * self.scale, k_h.transpose(-1, -2)), dim=-1)
                head_outs.append(torch.matmul(attn_h, v_h))
            out = torch.stack(head_outs, dim=1)
        else:
            attn = torch.matmul(q * self.scale, k.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        gates = torch.sigmoid(self.to_gates(x).permute(0, 2, 1).unsqueeze(-1))
        out = (out * gates).permute(0, 2, 1, 3).reshape(-1, self.seq_len, self.heads * self.dim_head)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        dim_inner = int(dim * mult)
        # Keep exact layer indices of upstream nn.Sequential (Linear at 1, GELU at 2, Linear at 4)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(dim_inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, dim_head: int, heads: int, seq_len: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, rotary_embed=RotaryStatic(dim_head, seq_len)),
                FeedForward(dim=dim, mult=4),
            ]))
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class BandSplit(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([
            nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            for dim_in in dim_inputs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, sum(dim_inputs)] -> [B, T, num_bands, dim]
        parts = torch.split(x, self.dim_inputs, dim=-1)
        outs = [layer(part) for part, layer in zip(parts, self.to_features)]
        return torch.stack(outs, dim=-2)


def make_mlp(dim_in: int, dim_out: int, dim_hidden: int, depth: int) -> nn.Sequential:
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)
    layers: list[nn.Module] = []
    for ind, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_d, out_d))
        if ind < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    def __init__(self, dim: int, dim_inputs: tuple[int, ...], depth: int, mlp_expansion_factor: int):
        super().__init__()
        dim_hidden = dim * mlp_expansion_factor
        self.num_bands = len(dim_inputs)
        self.to_freqs = nn.ModuleList([
            nn.Sequential(
                make_mlp(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )
            for dim_in in dim_inputs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, num_bands, dim] -> [B, T, sum(dim_inputs)]
        band_tensors = torch.unbind(x, dim=-2)
        outs = [mlp(feat) for feat, mlp in zip(band_tensors, self.to_freqs)]
        return torch.cat(outs, dim=-1)


class MelBandRoformerCore(nn.Module):
    """Pure tensor core of MelBandRoformer between STFT and ISTFT."""

    def __init__(self, model_cfg: dict):
        super().__init__()
        dim = model_cfg["dim"]
        depth = model_cfg["depth"]
        time_depth = model_cfg["time_transformer_depth"]
        freq_depth = model_cfg["freq_transformer_depth"]
        num_bands = model_cfg["num_bands"]
        dim_head = model_cfg["dim_head"]
        heads = model_cfg["heads"]
        mlp_factor = model_cfg.get("mlp_expansion_factor", 4)
        mask_depth = model_cfg["mask_estimator_depth"]
        zero_dc = model_cfg.get("zero_dc", True)

        self.dim = dim
        self.num_bands = num_bands
        self.bins = reference.MB_BINS
        self.frames = reference.MB_FRAMES
        self.channels = reference.CHANNELS

        self.layers = nn.ModuleList([
            nn.ModuleList([
                Transformer(dim=dim, depth=time_depth, dim_head=dim_head, heads=heads, seq_len=self.frames),
                Transformer(dim=dim, depth=freq_depth, dim_head=dim_head, heads=heads, seq_len=num_bands),
            ])
            for _ in range(depth)
        ])

        mask = slaney_mel_freqs_per_band(reference.SAMPLE_RATE, reference.MB_NFFT, num_bands)
        num_freqs_per_band = mask.sum(axis=1).tolist()
        dims_in = tuple(2 * int(nf) * self.channels for nf in num_freqs_per_band)
        self.total_band_scalars = sum(dims_in)

        self.band_split = BandSplit(dim=dim, dim_inputs=dims_in)
        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dims_in, depth=mask_depth, mlp_expansion_factor=mlp_factor)
        ])

        # Build gather/scatter-average tables operating over flat slot layout (f*4 + s*2 + c).
        in_gather_idx = []
        scatter_buckets: list[list[int]] = [[] for _ in range(self.bins * 4)]
        out_cursor = 0
        for band_idx in range(num_bands):
            for f_bin in np.flatnonzero(mask[band_idx]).tolist():
                for ch in range(self.channels):
                    for re_im in range(2):
                        flat_slot = f_bin * 4 + ch * 2 + re_im
                        in_gather_idx.append(flat_slot)
                        scatter_buckets[flat_slot].append(out_cursor)
                        out_cursor += 1
        assert out_cursor == self.total_band_scalars
        assert all(1 <= len(b) <= 2 for b in scatter_buckets), "expected 1..2 overlapping bands per bin"

        sentinel = self.total_band_scalars
        max_overlap = max(len(b) for b in scatter_buckets)
        out_gather = np.full((self.bins * 4, max_overlap), sentinel, dtype=np.int64)
        inv_denom = np.zeros(self.bins * 4, dtype=np.float32)
        for flat_slot, srcs in enumerate(scatter_buckets):
            for k, src_pos in enumerate(srcs):
                out_gather[flat_slot, k] = src_pos
            f_bin = flat_slot // 4
            inv_denom[flat_slot] = 0.0 if (zero_dc and f_bin == 0) else (1.0 / len(srcs))

        self.max_overlap = max_overlap
        self.register_buffer("in_gather_idx", torch.tensor(in_gather_idx, dtype=torch.long), persistent=False)
        self.register_buffer("out_gather_idx", torch.tensor(out_gather.reshape(-1), dtype=torch.long), persistent=False)
        self.register_buffer("inv_denom", torch.tensor(inv_denom, dtype=torch.float32), persistent=False)

    def bake_rotary(self) -> None:
        for mod in self.modules():
            if isinstance(mod, RotaryStatic):
                mod.bake()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [B, 4, 1025, 801] where channel order is L_r, L_i, R_r, R_i
        spec_btf = spec.permute(0, 3, 2, 1)  # [B, T, F, 4]
        spec_flat = spec_btf.reshape(-1, self.frames, self.bins * 4)  # [B, T, 4100]

        x = torch.index_select(spec_flat, 2, self.in_gather_idx)  # [B, T, 7916]
        x = self.band_split(x)  # [B, T, 60, dim]

        for time_tf, freq_tf in self.layers:
            x = x.permute(0, 2, 1, 3).reshape(-1, self.frames, self.dim)
            x = time_tf(x)
            x = x.reshape(-1, self.num_bands, self.frames, self.dim).permute(0, 2, 1, 3)
            x = x.reshape(-1, self.num_bands, self.dim)
            x = freq_tf(x)
            x = x.reshape(-1, self.frames, self.num_bands, self.dim)

        masks = self.mask_estimators[0](x)  # [B, T, 7916]
        zero_pad = torch.zeros_like(masks[:, :, :1])
        masks_padded = torch.cat((masks, zero_pad), dim=2)  # [B, T, 7917]
        gathered = torch.index_select(masks_padded, 2, self.out_gather_idx)  # [B, T, 4100*2]
        masks_avg = gathered.reshape(-1, self.frames, self.bins * 4, self.max_overlap).sum(dim=-1) * self.inv_denom

        spec_sc = spec_btf.reshape(-1, self.frames, self.bins, self.channels, 2)
        mask_sc = masks_avg.reshape(-1, self.frames, self.bins, self.channels, 2)

        xr, xi = spec_sc[..., 0], spec_sc[..., 1]
        mr, mi = mask_sc[..., 0], mask_sc[..., 1]
        out_r = xr * mr - xi * mi
        out_i = xr * mi + xi * mr
        out_flat = torch.stack((out_r, out_i), dim=-1).reshape(-1, self.frames, self.bins, 4)
        return out_flat.permute(0, 3, 2, 1).unsqueeze(1)  # [B, 1, 4, 1025, 801]


def load_core_and_entry() -> tuple[MelBandRoformerCore, dict, dict]:
    cfg = load_verified_config()
    entry = weights.lock_entry(MODEL_ID)
    weights.require_loadable(entry)
    sd = torch.load(weights.local_path(entry), map_location="cpu", weights_only=True)
    core = MelBandRoformerCore(cfg["model"])
    core.load_state_dict(sd, strict=True)
    core.bake_rotary()
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)
    return core, entry, cfg


def segment_inputs(batch: int) -> torch.Tensor:
    length = reference.MB_SEGMENT
    items = [torch.from_numpy(synth.synth_clip(12.0)[:, :length].copy())]
    generator = torch.Generator().manual_seed(5678)
    for index in range(1, batch):
        items.append((0.05 * (index + 1)) * torch.randn(2, length, generator=generator))
    return torch.stack(items)


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    noise = float(((ref.astype(np.float64) - est) ** 2).sum())
    return 10.0 * np.log10(float((ref.astype(np.float64) ** 2).sum()) / max(noise, 1e-300))


def export_model(core: MelBandRoformerCore, entry: dict, out_path: Path, float16_storage: bool = True,
                 dynamic_batch: bool = True, stride: int = DEFAULT_STRIDE) -> dict:
    mix = segment_inputs(1)
    spec = torch.from_numpy(reference.melband_stft(mix[0].numpy()))[None]
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
        "spec_out": [batch_dim, 1, 2 * reference.CHANNELS, reference.MB_BINS, reference.MB_FRAMES],
    })
    storage = onnx_tools.store_weights_as_float16(graph_model) if float16_storage else None
    metadata = {
        "id": MODEL_ID,
        "kind": "melband_roformer",
        "sources": json.dumps(["vocals"]),
        "targets": json.dumps(["vocals"]),
        "sampleRate": reference.SAMPLE_RATE,
        "channels": reference.CHANNELS,
        "segmentSamples": reference.MB_SEGMENT,
        "strideSamples": stride,
        "nfft": reference.MB_NFFT,
        "hop": reference.MB_HOP,
        "bins": reference.MB_BINS,
        "frames": reference.MB_FRAMES,
        "winLength": reference.MB_NFFT,
        "melBands": core.num_bands,
        "normalize": "false",
        "license": "MIT",
        "attribution": ATTRIBUTION,
        "origin": "https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model",
        "originSha256": entry["sha256"],
    }
    onnx_tools.set_metadata(graph_model, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph_model, str(out_path))
    weight_bytes = onnx_tools.initializer_stats(graph_model)
    del graph_model

    inert = verify_onnx.verify(out_path)
    if inert["violations"]:
        raise RuntimeError(f"{out_path.name} failed verify_onnx: {inert['violations']}")

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    parity = {}
    for batch in ([1, 2] if dynamic_batch else [1]):
        batch_mix = segment_inputs(batch)
        batch_spec = np.stack([reference.melband_stft(batch_mix[b].numpy()) for b in range(batch)])
        with torch.no_grad():
            torch_spec_out = core(torch.from_numpy(batch_spec)).numpy()
        onnx_spec_out = session.run(["spec_out"], {"spec": batch_spec})[0]
        torch_wave = np.stack([reference.melband_istft(torch_spec_out[b]) for b in range(batch)])
        onnx_wave = np.stack([reference.melband_istft(onnx_spec_out[b]) for b in range(batch)])
        parity[f"batch{batch}"] = {
            "specMaxAbsErr": float(np.abs(onnx_spec_out - torch_spec_out).max()),
            "endToEndMaxAbsErr": float(np.abs(onnx_wave - torch_wave).max()),
            "endToEndMaxAbs": float(np.abs(torch_wave).max()),
            "endToEndSnrDb": snr_db(torch_wave, onnx_wave),
            "snrDbPerStem": {"vocals": snr_db(torch_wave[:, 0], onnx_wave[:, 0])},
        }
    del session

    return {
        "id": MODEL_ID,
        "kind": "melband_roformer",
        "file": out_path.name,
        "sha256": inert["sha256"],
        "sizeBytes": inert["sizeBytes"],
        "sources": ["vocals"],
        "targets": ["vocals"],
        "sampleRate": reference.SAMPLE_RATE,
        "channels": reference.CHANNELS,
        "segmentSamples": reference.MB_SEGMENT,
        "strideSamples": stride,
        "nfft": reference.MB_NFFT,
        "hop": reference.MB_HOP,
        "bins": reference.MB_BINS,
        "frames": reference.MB_FRAMES,
        "window": "hann",
        "melBands": core.num_bands,
        "winLength": reference.MB_NFFT,
        "normalize": False,
        "dynamicBatch": dynamic_batch,
        "weightStorage": "float16" if float16_storage else "float32",
        "float16Storage": storage,
        "initializerBytesByType": weight_bytes,
        "opset": OPSET,
        "exporter": "torch.onnx.export(dynamo=False) [TorchScript]",
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "parameters": sum(p.numel() for p in core.parameters()),
        "origin": entry["url"],
        "originSha256": entry["sha256"],
        "sourceCommit": SOURCE_COMMIT,
        "parity": parity,
        "onnxVerify": {k: inert[k] for k in ("opset", "domainsUsed", "nodeCount", "opTypes", "initializerCount",
                                             "externalData", "subgraphs", "localFunctions", "checker", "inputs",
                                             "outputs")},
    }


def print_report(report: dict) -> None:
    print(f"{report['file']}: {report['sizeBytes'] / 1e6:.1f} MB ({report['weightStorage']} weights)  "
          f"sha256={report['sha256']}  params {report['parameters'] / 1e6:.2f} M")
    if report["float16Storage"]:
        s = report["float16Storage"]
        print(f"  float16 storage: {s['tensors']} tensors, max rounding error {s['maxAbsRoundingError']:.2e} "
              f"(relative {s['maxRelRoundingError']:.2e}), overflowed {s['overflowed']}, "
              f"underflowed {s['underflowedToZero']}")
    for name, p in report["parity"].items():
        print(f"  {name}: vs MelBandRoformerCore max|err| {p['endToEndMaxAbsErr']:.3e} "
              f"(max|x| {p['endToEndMaxAbs']:.3f}) = {p['endToEndSnrDb']:.1f} dB SNR")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ML_DIR / "dist")
    parser.add_argument("--float32-storage", action="store_true")
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    core, entry, _cfg = load_core_and_entry()
    report = export_model(core, entry, args.out / f"{MODEL_ID}{args.suffix}.onnx",
                          float16_storage=not args.float32_storage,
                          dynamic_batch=not args.fixed_batch,
                          stride=args.stride)
    if not args.suffix:
        (args.out / f"{MODEL_ID}.export.json").write_text(json.dumps(report, indent=2) + "\n")
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
