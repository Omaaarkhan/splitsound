"""Export the instrument tagger: EfficientAT mn10_as (MobileNetV3, AudioSet) with its mel frontend in the graph.

    uv run --locked --group tagger python export_tagger.py

Model:  fschmid56/EfficientAT `mn10_as` (MIT), 4.88 M parameters, mAP 0.471 on AudioSet.
Graph contract (CONTRACT.md section 2):
    input   waveform  float32 [1, 320000]   10 s of mono audio at 32 kHz, raw samples in [-1, 1]
    output  probs     float32 [1, 527]      sigmoid class probabilities, AudioSet order

Everything the original does between the waveform and the network is inside
the graph, built only from Conv / Pad / MatMul / Log style operators so that
no STFT operator is needed:

    pre-emphasis   y[n] = x[n+1] - 0.97 x[n]                       (Conv, kernel [-0.97, 1])
    reflect pad    512 samples each side                           (Pad, mode reflect)
    STFT           n_fft 1024, hop 320, Hann(800, symmetric) window centred in the 1024 frame,
                   as one Conv1d whose kernels are window * cos / -window * sin
    power          re^2 + im^2
    mel            128 Kaldi-style triangular filters, 0 .. 15000 Hz  (MatMul)
    log            log(mel + 1e-5), then (x + 4.5) / 5

The upstream model code is imported from a pinned commit (each file's SHA-256
is in weights.lock.json and was read before first use); the weights are loaded
with `torch.load(weights_only=True)`. Parity is checked against the unmodified
upstream model fed by the unmodified upstream frontend (`AugmentMelSTFT`).

Also writes assets/audioset_labels.json (527 display names in output order).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import tempfile
import types
import warnings
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

import synth  # noqa: E402
import verify_onnx  # noqa: E402
import weights  # noqa: E402

MODEL_ID = "efficientat_mn10_as"
OPSET = 17
SAMPLE_RATE = 32000
WINDOW_SAMPLES = 320000
NUM_CLASSES = 527
N_FFT, HOP, WIN_LENGTH, N_MELS = 1024, 320, 800, 128
FMIN, FMAX = 0.0, 15000.0
FRAMES = 1000
ATTRIBUTION = ("EfficientAT mn10_as by Florian Schmid, Khaled Koutini and Gerhard Widmer "
               "(\"Efficient Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation\", ICASSP 2023). "
               "MIT License. https://github.com/fschmid56/EfficientAT . Trained on AudioSet (Google, CC BY 4.0).")


def source_root() -> Path:
    """Directory holding the pinned upstream source files, after verifying each one."""
    root = None
    for entry in weights.read_lock()["files"]:
        if f"{MODEL_ID}:source" in entry["usedBy"]:
            weights.verify(entry)                               # size + SHA-256, raises on mismatch
            path = weights.local_path(entry)
            relative = entry["name"].split("/", 1)[1]
            root = Path(str(path)[: -len(relative)])
    assert root is not None
    return root


def load_original() -> tuple[nn.Module, nn.Module, list[str]]:
    """Return (upstream MobileNet with weights, upstream mel frontend, 527 labels)."""
    root = source_root()

    # Upstream `models/mn/model.py` does `from helpers.utils import NAME_TO_WIDTH`.
    # `helpers/utils.py` reads a CSV relative to the working directory when
    # imported and is otherwise training code, so it is neither downloaded nor
    # run; the one name the import needs is provided here.
    helpers = types.ModuleType("helpers")
    helpers_utils = types.ModuleType("helpers.utils")
    helpers_utils.NAME_TO_WIDTH = lambda name: 1.0
    helpers.utils = helpers_utils
    sys.modules.setdefault("helpers", helpers)
    sys.modules.setdefault("helpers.utils", helpers_utils)
    sys.path.insert(0, str(root))
    try:
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
    finally:
        sys.path.remove(str(root))

    # Same construction as upstream inference.py for `mn10_as`, minus the
    # download: pretrained_name=None means upstream fetches and loads nothing.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = get_model(width_mult=1.0, pretrained_name=None, strides=(2, 2, 2, 2), head_type="mlp")
        frontend = AugmentMelSTFT(n_mels=N_MELS, sr=SAMPLE_RATE, win_length=WIN_LENGTH, hopsize=HOP)

    entry = weights.lock_entry(MODEL_ID)
    weights.require_loadable(entry)
    state = torch.load(weights.local_path(entry), map_location="cpu", weights_only=True)   # plain state dict
    model.load_state_dict(state, strict=True)
    model.eval()
    frontend.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with open(root / "metadata" / "class_labels_indices.csv", newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    assert [int(row[0]) for row in rows] == list(range(NUM_CLASSES))
    labels = [row[2] for row in rows]
    return model, frontend, labels


class ConvMelFrontend(nn.Module):
    """The upstream mel frontend re-expressed with convolutions (no STFT operator, no randomness)."""

    def __init__(self):
        super().__init__()
        import torchaudio

        # Window: symmetric Hann of 800 samples, centred in the 1024-sample frame
        # (what torch.stft does when win_length < n_fft).
        window = torch.zeros(N_FFT, dtype=torch.float64)
        left = (N_FFT - WIN_LENGTH) // 2
        window[left: left + WIN_LENGTH] = torch.hann_window(WIN_LENGTH, periodic=False, dtype=torch.float64)

        # Mel filterbank exactly as upstream builds it in eval mode.
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0)
        assert tuple(mel_basis.shape) == (N_MELS, N_FFT // 2)
        # Only bins with a non-zero mel weight are computed: bins above FMAX
        # (and the Nyquist bin, which upstream pads with a zero weight) are dropped.
        used = int(torch.nonzero(mel_basis.abs().sum(dim=0) > 0).max()) + 1
        self.bins = used

        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(used, dtype=torch.float64)
        angle = 2.0 * math.pi * k[:, None] * n[None, :] / N_FFT
        real = window[None, :] * torch.cos(angle)
        imag = -window[None, :] * torch.sin(angle)
        kernel = torch.cat([real, imag], dim=0)[:, None, :].to(torch.float32)      # [2*bins, 1, 1024]

        self.register_buffer("preemphasis", torch.tensor([[[-0.97, 1.0]]], dtype=torch.float32))
        self.register_buffer("stft_kernel", kernel)
        self.register_buffer("mel_basis", mel_basis[:, :used].to(torch.float32).contiguous())

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)                                   # [1, 1, 320000]
        x = F.conv1d(x, self.preemphasis)                           # [1, 1, 319999]
        x = F.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")      # [1, 1, 321023]
        z = F.conv1d(x, self.stft_kernel, stride=HOP)               # [1, 2*bins, 1000]
        real, imag = z[:, : self.bins], z[:, self.bins:]
        power = real * real + imag * imag                           # [1, bins, 1000]
        mel = torch.matmul(self.mel_basis, power)                   # [1, 128, 1000]
        mel = torch.log(mel + 1e-5)
        mel = (mel + 4.5) / 5.0
        return mel.unsqueeze(1)                                     # [1, 1, 128, 1000]


class TaggerGraph(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.frontend = ConvMelFrontend()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        logits, _features = self.model(self.frontend(waveform))
        return torch.sigmoid(logits).reshape(1, NUM_CLASSES)


def original_probs(model: nn.Module, frontend: nn.Module, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream inference.py on CPU: mel(waveform) -> model(spec.unsqueeze(0)) -> sigmoid."""
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = frontend(waveform)
        logits, _ = model(spec.unsqueeze(0))
    return torch.sigmoid(logits.float()).reshape(1, NUM_CLASSES), spec


def verification_waveforms() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    clip = synth.synth_clip(10.0, sample_rate=SAMPLE_RATE).mean(axis=0)
    quiet_tail = clip.copy()
    quiet_tail[SAMPLE_RATE * 3:] = 0.0                              # a zero-padded last window
    return {
        "synthetic clip (mono, 32 kHz)": clip,
        "synthetic clip, 3 s then zeros": quiet_tail,
        "white noise 0.1": (0.1 * rng.standard_normal(WINDOW_SAMPLES)).astype(np.float32),
        "silence": np.zeros(WINDOW_SAMPLES, dtype=np.float32),
        "full-scale 440 Hz sine": np.sin(2 * np.pi * 440.0 * np.arange(WINDOW_SAMPLES) / SAMPLE_RATE).astype(np.float32),
    }


def export(out_path: Path, extra_wavs: list[str]) -> dict:
    model, frontend, labels = load_original()
    graph_module = TaggerGraph(model).eval()
    example = torch.from_numpy(verification_waveforms()["synthetic clip (mono, 32 kHz)"])[None]

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.onnx"
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            torch.onnx.export(graph_module, (example,), str(raw_path), dynamo=False, opset_version=OPSET,
                              input_names=["waveform"], output_names=["probs"], do_constant_folding=True)
        graph_model = onnx.load(str(raw_path))

    entry = weights.lock_entry(MODEL_ID)
    metadata = {
        "id": MODEL_ID, "kind": "tagger", "sampleRate": str(SAMPLE_RATE), "windowSamples": str(WINDOW_SAMPLES),
        "numClasses": str(NUM_CLASSES), "license": "MIT", "attribution": ATTRIBUTION,
        "origin": entry["url"], "originSha256": entry["sha256"],
    }
    del graph_model.metadata_props[:]
    for key in sorted(metadata):
        prop = graph_model.metadata_props.add()
        prop.key, prop.value = key, metadata[key]
    graph_model.doc_string = ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph_model, str(out_path))

    inert = verify_onnx.verify(out_path)
    if inert["violations"]:
        raise RuntimeError(f"{out_path.name} failed verify_onnx: {inert['violations']}")

    # ---- parity against the unmodified upstream model + frontend ----------------
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    waveforms = verification_waveforms()
    for wav in extra_wavs:
        import soundfile

        data, rate = soundfile.read(wav, dtype="float32", always_2d=True)
        assert rate == SAMPLE_RATE, f"{wav}: expected 32 kHz"
        mono = data.mean(axis=1)[:WINDOW_SAMPLES]
        waveforms[f"file {Path(wav).name}"] = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)))
    parity = {}
    for name, waveform in waveforms.items():
        tensor = torch.from_numpy(waveform.astype(np.float32))[None]
        reference_probs, reference_mel = original_probs(model, frontend, tensor)
        with torch.no_grad():
            own_mel = graph_module.frontend(tensor)[0]
        onnx_probs = session.run(["probs"], {"waveform": tensor.numpy()})[0]
        top = int(reference_probs[0].argmax())
        parity[name] = {
            "probsMaxAbsErr": float(np.abs(onnx_probs - reference_probs.numpy()).max()),
            "melMaxAbsErr": float((own_mel - reference_mel).abs().max()),
            "topClass": labels[top],
            "topProb": float(reference_probs[0, top]),
            "sameTop5": bool(np.array_equal(np.argsort(-onnx_probs[0])[:5], np.argsort(-reference_probs[0].numpy())[:5])),
        }

    labels_path = ML_DIR / "assets" / "audioset_labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, indent=0, ensure_ascii=False) + "\n")

    return {
        "id": MODEL_ID, "file": out_path.name, "sha256": inert["sha256"], "sizeBytes": inert["sizeBytes"],
        "sampleRate": SAMPLE_RATE, "windowSamples": WINDOW_SAMPLES, "numClasses": NUM_CLASSES,
        "frontend": {"preemphasis": 0.97, "nfft": N_FFT, "hop": HOP, "winLength": WIN_LENGTH, "nMels": N_MELS,
                     "fmin": FMIN, "fmax": FMAX, "frames": FRAMES, "stftBinsComputed": graph_module.frontend.bins},
        "opset": OPSET, "exporter": "torch.onnx.export(dynamo=False) [TorchScript]",
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "parameters": sum(p.numel() for p in model.parameters()),
        "origin": entry["url"], "originSha256": entry["sha256"],
        "parity": parity,
        "onnxVerify": {k: inert[k] for k in ("opset", "domainsUsed", "nodeCount", "opTypes", "initializerCount",
                                             "externalData", "subgraphs", "localFunctions", "checker", "inputs",
                                             "outputs")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ML_DIR / "dist")
    parser.add_argument("--wav", action="append", default=[], help="extra 32 kHz WAV to include in the parity check")
    args = parser.parse_args()

    report = export(args.out / f"{MODEL_ID}.onnx", args.wav)
    (args.out / f"{MODEL_ID}.export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['file']}: {report['sizeBytes'] / 1e6:.2f} MB  sha256={report['sha256']}  "
          f"params {report['parameters'] / 1e6:.2f} M  stft bins computed {report['frontend']['stftBinsComputed']}")
    print(f"  ops {report['onnxVerify']['opTypes']}")
    for name, p in report["parity"].items():
        print(f"  {name:<34s} probs max|err| {p['probsMaxAbsErr']:.2e}  mel max|err| {p['melMaxAbsErr']:.2e}  "
              f"top: {p['topClass']} {p['topProb']:.3f}  same top-5: {p['sameTop5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
