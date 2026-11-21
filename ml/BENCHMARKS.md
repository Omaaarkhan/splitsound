# Benchmarks and measured quality

Machine: Mac mini (Mac16,10), Apple M4, 10 cores (4 performance + 6
efficiency), 16 GB, macOS 27.2. onnxruntime 1.32.0 (CPU and CoreML execution
providers). Raw results: `dist/reports/bench.json`, `bench_memory.json`,
`separation_musdb7.json`, `reference_parity_*.json`. Reproduce with
`bench.py` and `eval_musdb.py`.

> **Read the timings as upper bounds.** The machine was shared while these
> ran: another process was using three to four cores throughout (load
> average 11 to 30). Times are the best of two runs per configuration and are
> still pessimistic and noisy; in particular the comparison between thread
> counts is not reliable for the SCNet models. Peak memory and the
> CPU-versus-CoreML comparison are much less affected. This should be re-run
> on an idle machine before any number is quoted publicly.

"Segment time" is one run of the ONNX graph on one segment. "Realtime" is
seconds of audio in the segment divided by that time. A whole track costs
`N / STRIDE = 1.33` graph runs per segment of audio for SCNet and HTDemucs
(`2.0` for `melband_roformer_vocals` at 50 % overlap; each 7 s test excerpt
takes two 8 s RoFormer calls because `stride = 4 s < 7 s`), plus the
host-side STFT and inverse STFT (about 0.3 s per segment in the numpy
reference; a compiled FFT is several times faster).

## 1. Recommendation

**Apple Silicon default: CPU execution provider, intra-op threads = number of
performance cores (4 on an M4), `batch` left symbolic, default graph
optimisation, one segment per call.**

* Do not use all cores. With 10 threads (performance + efficiency cores)
  every SCNet measurement was slower than with 4.
* Do not fix the batch dimension on CPU. It costs 2 to 4 GB of peak memory
  and is not faster (section 3).
* CoreML is worth offering only for `htdemucs_6s`: about 1.8 times faster
  per segment and half the memory, at the price of a 24 s model compilation
  when the session is created (set `ModelCacheDirectory` so that it happens
  once) and a 10 s first run. For SCNet it gives nothing, because CoreML does
  not implement LSTM and those nodes, which dominate the run time, stay on the
  CPU. For the tagger it is slower than the CPU.
* Budget about 2 GB of memory for `scnet_large` and `htdemucs_6s`, 0.8 GB for
  `scnet_small`.
* `scnet_large` is slow on CPU: roughly 4.4 s per 11 s segment here, i.e. a
  four-minute song in about two minutes at best on this machine. `scnet_small`
  is three to four times faster for 0.8 dB less SDR. Consider making the
  small model the default on machines with fewer than four performance cores.
* `melband_roformer_vocals` must run CPU-only (`allow_gpu = false`,
  `memory_pattern = false`, symbolic batch, 4 performance cores): CoreML
  (`MLProgram, ALL`) fails to compile/execute the 228M-parameter transformer and
  inflates swap/temp storage. Unrolling time attention per head (`seq_len >
  128`) plus `memory_pattern = false` brings peak RSS down from ~6.85–7.0 GiB
  to **3.58 GiB** (`3671 MiB`) at **5.28 s** per 8 s segment (1.52× realtime).

## 2. Segment time by provider and thread count

| model (segment) | configuration | segment time | realtime | session create | first run | peak RSS |
|---|---|---|---|---|---|---|
| `scnet_small` (11 s) | CPU, 1 thread | 1.47 s | 7.5x | 0.13 s | 1.78 s | 2133 MiB |
| | CPU, 4 threads | 1.52 s | 7.2x | 0.11 s | 1.83 s | 2257 MiB |
| | CPU, 10 threads | 2.76 s | 4.0x | 0.11 s | 2.83 s | 2265 MiB |
| | CoreML MLProgram, all units | 1.45 s | 7.6x | 2.70 s | 5.42 s | 1874 MiB |
| `scnet_large` (11 s) | CPU, 1 thread | 6.07 s | 1.8x | 0.20 s | 5.76 s | 6334 MiB |
| | CPU, 4 threads | 8.03 s | 1.4x | 0.19 s | 5.98 s | 3849 MiB |
| | CPU, 10 threads | 10.46 s | 1.1x | 0.19 s | 8.70 s | 5137 MiB |
| | CoreML MLProgram, all units | 5.83 s | 1.9x | 6.62 s | 6.74 s | 4523 MiB |
| `htdemucs_6s` (7.8 s) | CPU, 1 thread | 1.37 s | 5.7x | 0.43 s | 1.92 s | 4185 MiB |
| | CPU, 4 threads | 0.87 s | 8.9x | 0.23 s | 0.98 s | 4191 MiB |
| | CPU, 10 threads | 0.84 s | 9.3x | 0.24 s | 1.07 s | 4201 MiB |
| | CoreML MLProgram, all units | 0.47 s | 16.7x | 24.30 s | 9.77 s | 1987 MiB |
| tagger (10 s window) | CPU, 1 thread | 0.036 s | 280x | 0.03 s | 0.04 s | 156 MiB |
| | CPU, 4 threads | 0.020 s | 500x | 0.02 s | 0.02 s | 158 MiB |
| | CoreML MLProgram, all units | 0.038 s | 264x | 2.25 s | 1.28 s | 303 MiB |

All rows above had the batch fixed to 1, which is what inflates the CPU
memory figures; see section 3 for the setting that is recommended instead.
The `scnet_large` rows ran under the heaviest load; the 4-thread figure in
section 3 (4.4 to 4.6 s), measured a few minutes later, is more
representative.

CoreML details (one attempt per model: `ModelFormat=MLProgram`,
`MLComputeUnits=ALL`, batch fixed to 1): CoreML accepted 1061 of 1073 nodes
of each SCNet graph in 13 partitions (the 12 it left are the LSTMs), 1900 of
1925 nodes of `htdemucs_6s` in 26 partitions, and 42 of 88 nodes of the
tagger in 21 partitions. The mean absolute output agreed with the CPU
provider to nine digits for every model; a full per-sample parity check of
CoreML output was **not** done. The NeuralNetwork format and the
`CPUAndGPU` unit selection were not run (`bench.py` without `--quick` does).

## 3. Batch dimension, memory pattern and arena (CPU, 4 threads)

| model | batch fixed | fixed, memory pattern off | fixed, pattern and arena off | **batch symbolic** | symbolic, arena off |
|---|---|---|---|---|---|
| `scnet_small` | 1.60 s / 2258 MiB | 1.62 s / 1579 MiB | 1.92 s / 1748 MiB | **1.19 s / 789 MiB** | 1.47 s / 1131 MiB |
| `scnet_large` | 4.62 s / 6294 MiB | 6.20 s / 3858 MiB | 5.72 s / 4548 MiB | **4.42 s / 1900 MiB** | 4.91 s / 3095 MiB |
| `htdemucs_6s` | 1.23 s / 4198 MiB | 1.27 s / 2658 MiB | 1.30 s / 2438 MiB | **1.12 s / 1861 MiB** | 1.13 s / 2153 MiB |

For `melband_roformer_vocals` (8 s segment, per-head unrolled time attention,
symbolic batch, CPU 4 threads): default (`mem_pattern=true, arena=true`) runs
`5.37 s / 3961 MiB` (3.87 GiB), **`mem_pattern=false, arena=true`** runs
**`5.28 s / 3671 MiB` (3.58 GiB)** (1.52× realtime), and `both=false` runs
`5.59 s / 4120 MiB`.

(segment time / peak resident memory of the whole process.) With every shape
static, onnxruntime plans one arena for all intermediate tensors and peak
memory triples. Leaving `batch` symbolic is the cheapest fix and is what
`reference.py` does.

## 4. Graph optimisation level (CPU, 4 threads, batch fixed)

| model | disabled | basic | extended | all |
|---|---|---|---|---|
| `scnet_small` | 2.00 s, 879 MiB | 1.52 s | 1.65 s | 1.52 s |
| `htdemucs_6s` | 1.26 s, 1901 MiB | 1.17 s | 1.07 s | 0.87 s |

Optimisation is worth 15 to 30 % and costs 0.1 to 0.25 s at session creation.
Use the default (`all`). Session creation is fast in every case (0.1 to 0.4 s
on CPU), including the float16-to-float32 casts of the SCNet weights and the
110 MB external weights file of `htdemucs_6s`, so there is no need to cache
an optimised model. Outputs are identical across levels and thread counts to
the precision checked.

## 5. Separation quality (MUSDB18 7-second excerpts)

50 test excerpts, each one segment long; `reference.py` with shifts 0. SDR is
the plain definition `10*log10(sum ref^2 / sum (ref - est)^2)` over the whole
excerpt and both channels, per stem; the table gives the median over
excerpts. The audio is AAC-compressed and 7 s long, so these are **not**
comparable with published MUSDB18-HQ scores (museval, full tracks); they are
a baseline for regressions and for comparing the models with each other.

| model | drums | bass | other | vocals | mean of the four |
|---|---|---|---|---|---|
| `Studio` (`melband_roformer_vocals` + `scnet_large` on residual) | **10.48** | **9.08** | **7.42** | **11.59** | **9.64 dB** |
| `scnet_large` | 10.33 | 8.64 | 7.08 | 10.69 | **9.19 dB** |
| `scnet_small` | 9.72 | 8.50 | 5.99 | 9.51 | **8.43 dB** |
| `htdemucs_6s` | 9.21 | 8.19 | 4.97 | 8.54 | **7.73 dB** |

For `htdemucs_6s`, "other" is its other + guitar + piano outputs, because
MUSDB's "other" stem contains those instruments. `htdemucs_ft` and
`htdemucs` were not exported or measured.

* float16 weight storage (shipped) versus float32 weights, `scnet_small`:
  8.43 dB in both cases, every stem within 0.01 dB.
* Chunk overlap, `scnet_small`, on a 70 s medley of ten excerpts: overlap 0.25
  gives 7.40 dB, overlap 0.5 gives 7.36 dB for 1.5 times the work. The default
  stride stays at overlap 0.25.
* Studio cascade (`melband_roformer_vocals` followed by `scnet_large`): running
  Pass 2 (`scnet_large`) on the vocal-free **residual** (`9.64 dB` mean of
  medians) outperformed running Pass 2 on the raw mix (`9.43 dB` mean of
  medians; `10.33 / 8.64 / 7.15 / 11.59`). Note two caveats for the Studio
  row: (1) KimberleyJensen's checkpoint training data is undocumented, so
  whether test tracks were seen in training is unknown; (2) the 7 s MUSDB18
  excerpts are shorter than RoFormer's 8 s segment, so both model calls
  (chunks at 0 s and 4 s) see zero-padded audio and overlap-add is exercised
  only over the last 3 s of each excerpt. Separately, stride
  `176400` (50% overlap) vs `88200` (75% overlap) on `melband_roformer_vocals`
  differed by only ~0.1 dB vocal bleed on a 20 s vocal-heavy excerpt while
  costing 2.08× runtime, so default stride stays `176400`.

## 6. Numerical agreement with the PyTorch originals

`check_reference.py` and the export scripts; SNR of the whole pipeline
(numpy STFT, ONNX graph, numpy inverse STFT, chunking) against the original
PyTorch module driven by `demucs.apply.apply_model`.

| what | SNR |
|---|---|
| STFT / inverse STFT, both kinds, against `torch.stft` / `torch.istft` | 136 to 139 dB |
| `htdemucs_6s`, synthetic tracks of five lengths, normalisation, shifts 2 and 3 | 121 to 130 dB |
| `scnet_small`, float32 weights, same set | 119 to 126 dB |
| `scnet_small`, float32 weights, real music (2 excerpts + 70 s medley) | 114 to 127 dB per stem |
| `scnet_large`, float32 weights (quick set) | 119 to 122 dB |
| `scnet_large` **as shipped** (float16 storage) | 65 dB (63 to 66 on the non-silent stems) |
| `scnet_small` **as shipped**, real music | 55 to 73 dB per stem |
| `melband_roformer_vocals`, PyTorch Core vs upstream (`MelBandRoformer.forward`), real vocals (Clairo 8 s)* | 128.54 dB |
| `melband_roformer_vocals`, float32 ONNX vs PyTorch Core, real vocals (Clairo 8 s) | 127.72 dB wave (122.66 dB spec) |
| `melband_roformer_vocals` **as shipped** (float16 storage), real vocals (Clairo 8 s)* | 75.22 dB vs PyTorch Core (75.05 dB vs upstream) |
| tagger probabilities against the original model and frontend | max abs error 7e-7 |

*\* The `melband_roformer_vocals` upstream (`MelBandRoformer.forward`) numbers were
recorded in a one-off manual parity check on the Clairo 8 s excerpt with the
optional upstream packages (`beartype`, `einops`, `rotary_embedding_torch`,
`librosa`) excluded from `uv.lock`. `check_reference.py --model
melband_roformer_vocals --wav <clip.wav>` reproduces the PyTorch
`MelBandRoformerCore` / STFT checks with `uv run --locked` (and also runs the
upstream comparison automatically when those optional packages are present).*

A finding that matters for any future export: onnxruntime's single-precision
reductions over millions of elements are inaccurate. Exported naively, SCNet
agreed with PyTorch to only 64 dB; computing `GroupNorm(1, C)` statistics one
axis at a time raised that to 74 dB, and doing the same for the model's
input mean / standard deviation raised it to 125 dB. HTDemucs improved from
about 80 dB to 126 dB the same way. PyTorch itself is within 125 dB of a
float64 run.

## 7. Windows and Linux (not tested)

Nothing below was measured; treat it as a checklist.

* **Linux and Windows on CPU.** The same recommendation should hold: CPU
  provider, threads equal to the number of physical performance cores, batch
  symbolic. SCNet's cost is dominated by 12 bidirectional LSTMs, which scale
  poorly with threads, so single-core speed matters more than core count.
* **DirectML.** The onnxruntime documentation requires
  `enable_mem_pattern = false` and sequential execution mode for this
  provider, and it prefers static shapes, so fix `batch` to 1 there. Whether
  its LSTM kernel is fast enough to make SCNet worthwhile on a GPU, and
  whether a float16-arithmetic variant would help, is unknown.
* **CUDA / other GPU providers.** Batching several segments per call (the
  graphs allow it) is the first thing to try; not evaluated.
* **In-memory model loading** (common on Windows packaging) needs the session
  option `session.model_external_initializers_file_folder_path` for
  `htdemucs_6s` (CONTRACT.md 5.8).
* The exports use opset 17 and only default-domain operators, which every
  onnxruntime release since 1.13 accepts.
