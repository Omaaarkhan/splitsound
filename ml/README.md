# SplitSound ML

Everything needed to turn the upstream models into the ONNX files the app
runs, plus the numpy reference implementations the Rust engine mirrors.
Python is used here only; the app has no Python at run time.

| read this | for |
|-----------|-----|
| `CONTRACT.md` | the authoritative spec: tensors, STFT maths, chunking, conversion of the Demucs weights, tagger, manifest |
| `LICENSES.md` | licence and provenance of every model, with sources |
| `SECURITY.md` | what was vetted and how; what could not be verified |
| `BENCHMARKS.md` | speed, memory, execution-provider recommendation, measured quality |
| `dist/testdata/README.md` | the test vectors |

## What gets built

| id | kind | shipped as | notes |
|----|------|-----------|-------|
| `scnet_large` | `scnet` | `scnet_large.onnx.zst` | default engine, 4 stems |
| `scnet_small` | `scnet` | `scnet_small.onnx.zst` | faster, smaller, 4 stems |
| `melband_roformer_vocals` | `melband_roformer` | `melband_roformer_vocals.onnx.zst` | Studio-tier vocal isolation, vocals stem |
| `htdemucs_6s` | `htdemucs` | `htdemucs_6s.onnx.zst` (graph only) + `htdemucs_6s.tensors.json` | restricted: the weights are fetched from Meta and converted on the user's machine |
| `efficientat_mn10_as` | `tagger` | `efficientat_mn10_as.onnx.zst` | instrument tagger, 527 AudioSet classes |

`dist/` and `.cache/` are gitignored. `dist/htdemucs_6s.weights`,
`dist/compare/` and `dist/eval/` exist on the build machine only and must
never be published.

## Requirements

macOS or Linux, [`uv`](https://docs.astral.sh/uv/) 0.10+, `ffmpeg` (only for
the MUSDB evaluation). Python 3.12 is fetched by `uv`. About 4 GB of disk.
Every command below is run from this `ml/` directory and uses the hash-pinned
lockfile (`--locked`); nothing is installed from anywhere but PyPI.

## Reproduce everything from scratch

```bash
cd ml

# 1. Download the pinned upstream files (HTTPS, publisher hosts only) and verify
#    size, SHA-256 and the static pickle scan of each. Never unpickles anything.
uv run --locked python -I weights.py fetch

# 2. Export. Each script checks its graph against the original PyTorch model
#    and runs verify_onnx on the result.
uv run --locked python export_scnet.py                          # scnet_small, scnet_large
uv run --locked python export_melband_roformer.py               # melband_roformer_vocals
uv run --locked python export_htdemucs.py                       # htdemucs_6s graph, tensor table, local weights file
uv run --locked --group tagger python export_tagger.py          # tagger + assets/audioset_labels.json

# 3. Package: verify, compress (zstd 19), hash, write dist/manifest.json.
uv run --locked python package.py

# 4. Test vectors for the Rust tests.
uv run --locked python make_testdata.py
```

## Checks

```bash
# Inert-data check on the exported graphs (exit status 1 on any violation); meant for CI.
uv run --locked python verify_onnx.py dist/scnet_large.onnx dist/scnet_small.onnx dist/melband_roformer_vocals.onnx dist/efficientat_mn10_as.onnx
uv run --locked python verify_onnx.py --external-data htdemucs_6s.weights --max-inline-bytes 65536 dist/htdemucs_6s.onnx

# Reference implementation against the original PyTorch code.
uv run --locked python check_reference.py --model scnet_large --onnx dist/compare/scnet_large_f32.onnx
uv run --locked python check_reference.py --model scnet_small
uv run --locked python check_reference.py --model htdemucs_6s
uv run --locked python check_reference.py --random-init --quick      # no checkpoint needed

# The torch-free checkpoint converter (reference for the Rust port).
uv run --locked python -I convert_reference.py selftest
uv run --locked python -I convert_reference.py convert .cache/weights/5c90dfd2-34c22ccb.th dist/htdemucs_6s.tensors.json /tmp/htdemucs_6s.weights

# Re-verify every downloaded file against the lock without downloading.
uv run --locked python -I weights.py verify

# Static scan of any checkpoint (stdlib only, never unpickles).
uv run --locked python -I scan_pickle.py --allow-package demucs .cache/weights/5c90dfd2-34c22ccb.th

# Dependency audit of the locked set.
uv export --locked --no-emit-project --format requirements-txt --all-groups -o dist/security/requirements.locked.txt
uv run --locked --group security pip-audit -r dist/security/requirements.locked.txt --require-hashes --disable-pip
```

## Using the references

```bash
# Separation: 44.1 kHz WAV in, one float32 WAV per stem out.
uv run --locked reference.py separate dist/scnet_large.onnx song.wav out/
uv run --locked reference.py separate dist/htdemucs_6s.onnx song.wav out/ --shifts 2

# Tagging: 32 kHz mono WAV in (ffmpeg -i song.wav -ac 1 -ar 32000 song_32k.wav).
uv run --locked reference_tagger.py tag dist/efficientat_mn10_as.onnx song_32k.wav
```

From the repository root the same commands work with
`uv run --locked --project ml ml/reference.py separate ...`.

## Evaluation and benchmarks

```bash
uv run --locked python -I weights.py fetch --only eval:musdb7   # MUSDB18 7-second sample, pinned
uv run --locked python eval_musdb.py prepare
uv run --locked python eval_musdb.py separation                 # SDR per stem, agreement with PyTorch
uv run --locked python eval_musdb.py tagger                     # tagger threshold study
uv run --locked python bench.py                                 # writes dist/reports/bench.json
```

Comparison exports that are never distributed (float32-weight SCNet graphs,
self-contained Demucs graphs) go to `dist/compare/`:

```bash
uv run --locked python export_scnet.py --float32-storage --out dist/compare --suffix _f32
uv run --locked python export_htdemucs.py --single-file htdemucs_6s
```

## Files

| file | purpose |
|------|---------|
| `weights.lock.json` | pins (URL, size, SHA-256, expected pickle globals) for every downloaded file |
| `weights.py` | pinned download, verification, and the only code that loads checkpoints |
| `scan_pickle.py` | static pickle scanner (stdlib only) |
| `convert_reference.py` | torch-free, pickle-free converter for Meta's `htdemucs_6s` checkpoint |
| `export_scnet.py`, `export_htdemucs.py`, `export_tagger.py` | exports with parity checks |
| `onnx_tools.py`, `verify_onnx.py` | ONNX post-processing; inert-data verification |
| `reference.py`, `reference_tagger.py` | numpy + onnxruntime references (no torch) |
| `check_reference.py`, `eval_musdb.py`, `bench.py` | parity, real-music evaluation, benchmarks |
| `package.py`, `make_testdata.py`, `synth.py` | manifest and compression; test vectors; synthetic clip |
| `assets/` | `audioset_labels.json`, `instrument_map.json` |

## Updating a pinned file

A changed upstream file is a security event, not a routine update: `weights.py`
refuses it. To accept a new file deliberately, download it into an empty
directory, run `scan_pickle.py` on it, review the globals, and only then
change the size, SHA-256 and `expectedGlobals` in `weights.lock.json`, bump
the model's `revision` in the manifest, and record the change in SECURITY.md.
