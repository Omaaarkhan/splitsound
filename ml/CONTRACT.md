# SplitSound ML contract

Status: **v2.2, final for everything that is built.** Sections 1, 2, 3, 5, 6, 7
and 8 describe the files in `dist/` exactly. Section 4 (`melband_roformer`) is
reserved: that model was deferred and is not exported. The changelog at the end lists every edit.

All indices are 0-based, ranges are half-open. Audio is float32 in [-1, 1],
planar `[channels, samples]`. A "segment" is one fixed-size model invocation.
`reference.py` (separation) and `reference_tagger.py` (tagging) are the
executable versions of this document; the test vectors were produced by them.

Model kinds:

| kind               | models                    | host-side transform        | section |
|--------------------|---------------------------|----------------------------|---------|
| `scnet`            | `scnet_large`, `scnet_small` (default engine) | rectangular-window STFT | 3 |
| `htdemucs`         | `htdemucs_6s` (restricted, converted on device) | Hann STFT + waveform branch | 2, 5 |
| `melband_roformer` | none yet (deferred)       | not specified              | 4       |
| `tagger`           | `efficientat_mn10_as`     | none (frontend in graph)   | 6       |

---------------------------------------------------------------------------

## 1. Full-track separation (common to every separation kind)

Everything here is identical for all separation models. Only the constants
`N` (segment samples) and `STRIDE` (samples between chunk starts) differ; both
are integers given per model in the manifest (`segmentSamples`,
`strideSamples`). `run_segment` is the per-kind transform of sections 2-4: it
maps `[2, N]` to `[S, 2, N]`.

Input `mix[2, L]` at 44100 Hz (mono must be duplicated to two channels,
anything else resampled first). Output `out[S, 2, L]`.

### 1.1 Track normalisation (on by default)

    ref  = 0.5 * (mix[0] + mix[1])              # mono
    mu   = mean(ref)
    sd   = std(ref)                             # unbiased: divides by L - 1
    mix  = (mix - mu) / sd
    ...chunked separation, shifts...
    out  = out * sd + mu

If `sd < 1e-8` or `L < 2` (digital silence) use `sd = 1`. (The originals
divide by zero there; this is the one deliberate deviation.)

### 1.2 Chunk schedule and weighted overlap-add

    offsets = 0, STRIDE, 2*STRIDE, ...  while offset < L
    out = zeros[S, 2, L];  sum_w = zeros[L]
    for each offset o:
        len_o = min(N, L - o)                   # only the last chunk(s) can be short
        delta = N - len_o
        start = o - delta // 2                  # may be negative
        seg[c, i] = mix[c, start + i] if 0 <= start + i < L else 0      for i in 0..N
        y = run_segment(seg)                    # [S, 2, N]
        chunk = y[:, :, delta//2 : delta//2 + len_o]
        out[:, :, o : o+len_o] += weight[0..len_o] * chunk
        sum_w[o : o+len_o]     += weight[0..len_o]
    out /= sum_w

Two details that are easy to get wrong:

* a short last chunk is **centred** in its segment and the rest of the
  segment is filled with the **real audio around it** where the track has any
  (zeros only outside the track);
* a short chunk uses the **first** `len_o` entries of `weight`, not a
  re-centred window.

Transition weight, a triangle of length `N` with `h = N / 2` (`N` is even for
every model):

    weight[i] = (i + 1) / h          for i in 0..h
    weight[i] = (N - i) / h          for i in h..N

### 1.3 Deterministic shifts (test-time augmentation)

    MAX_SHIFT = 22050
    passes = max(1, shifts)
    for k in 0..passes:
        d_k = floor(k * MAX_SHIFT / passes)             # 0 for k = 0
        y_k = chunked_separation(concat(zeros[2, d_k], mix))    # section 1.2
        acc += y_k[:, :, d_k:]
    out = acc / passes

`shifts = 0` and `shifts = 1` are the same single unshifted pass. This is
demucs' "shift trick" with its random offset replaced by `MAX_SHIFT - d_k`.

### 1.4 Bags

When several models share one `sources` list and each lists `targets`, every
model is run through 1.1-1.3 independently and each source is the mean of the
estimates from the models that target it. (Shipped models are all single
models; the rule exists for `htdemucs_ft`, which is not distributed.)

### 1.5 Complex-as-channels layout

Wherever a spectrogram crosses the graph boundary it is real-valued with the
complex parts as channels:

    c = 0: left  real      c = 1: left  imag
    c = 2: right real      c = 3: right imag

Graph outputs put the source axis first: `[B, S, 4, bins, frames]`.

The only symbolic dimension of any separation graph is the batch, named
`batch`. Feed one segment at a time (`B = 1`) unless a provider benefits from
more. On the CPU execution provider leave the dimension symbolic: fixing it
(`AddFreeDimensionOverrideByName("batch", 1)`) makes onnxruntime plan a static
arena and raises peak memory from about 1.9 GB to 4.2 to 6.3 GB with no speed
gain. Fix it only for providers that require static shapes, such as CoreML
(BENCHMARKS.md).

---------------------------------------------------------------------------

## 2. `htdemucs` segment transform

Constants:

| name     | value  | meaning                                             |
|----------|--------|-----------------------------------------------------|
| `N`      | 343980 | segment samples (39/5 s)                            |
| `STRIDE` | 257985 | `int(0.75 * N)` (overlap 0.25)                      |
| `NFFT`   | 4096   | FFT size = window length                            |
| `HOP`    | 1024   |                                                     |
| `F`      | 2048   | bins kept: 0..2047 (the Nyquist bin 2048 is dropped)|
| `T`      | 336    | frames = ceil(N / HOP)                              |
| `PAD`    | 1536   | reflect padding on the left (3 * HOP / 2)           |
| `PAD_R`  | 1620   | reflect padding on the right (PAD + T*HOP - N)      |

Graph I/O:

| name       | dir | dtype   | shape                   |
|------------|-----|---------|-------------------------|
| `mix`      | in  | float32 | `[B, 2, 343980]`        |
| `spec`     | in  | float32 | `[B, 4, 2048, 336]`     |
| `spec_out` | out | float32 | `[B, S, 4, 2048, 336]`  |
| `wave_out` | out | float32 | `[B, S, 2, 343980]`     |

The graph contains the model's own normalisation of both inputs and the
de-normalisation of both outputs. `run_segment(seg)`:

    spec           = STFT_htdemucs(seg)
    spec_out, wave = graph(mix = seg, spec = spec)
    y[s, ch]       = ISTFT_htdemucs(spec_out[s, 2*ch], spec_out[s, 2*ch+1]) + wave[s, ch]

### 2.1 `STFT_htdemucs` (per channel `x[0..N)`)

1. Reflect-pad without repeating the edge sample (numpy `mode="reflect"`):
   `y = x[PAD], x[PAD-1], ..., x[1], x[0], ..., x[N-1], x[N-2], ..., x[N-1-PAD_R]`,
   `len(y) = 347136 = (T + 3) * HOP`.
2. Window: periodic Hann, `w[n] = 0.5 - 0.5*cos(2*pi*n/NFFT)`.
3. `Z[k, j] = (1/sqrt(NFFT)) * sum_{n=0}^{NFFT-1} w[n] * y[j*HOP + n] * exp(-2i*pi*k*n/NFFT)`
   for `j in 0..T`, `k in 0..2048`.

### 2.2 `ISTFT_htdemucs` (per source and channel)

1. Per frame: 2049 bins with bin 2048 = 0; set the imaginary part of bin 0 to
   zero (a real inverse FFT ignores it; some libraries reject it instead).
2. `f_j[n] = w[n] * sqrt(NFFT) * irfft(Z[:, j])[n]` (`irfft` with `1/NFFT` scaling).
3. `acc[j*HOP + n] += f_j[n]`, `len(acc) = (T + 3) * HOP`.
4. `x[m] = acc[m + PAD] / 1.5` for `m in 0..N`.

`1.5` is the sum of the four overlapping squared Hann windows. demucs adds
two zero frames on each side before its inverse, which makes that sum complete
everywhere in the kept range, hence a constant divisor.

`ISTFT(STFT(x))` is **not** `x` near the segment edges (first 1536 and last
~1452 samples are attenuated, 34-37 dB SNR over a whole segment). That is how
demucs works and the network was trained with it; do not "fix" it.

---------------------------------------------------------------------------

## 3. `scnet` segment transform

SCNet has no waveform branch: one spectrogram in, one per source out. Its STFT
uses a **rectangular** window (the original calls `torch.stft` without a
window) and keeps **all 2049 bins**.

Constants:

| name       | value  | meaning                                                  |
|------------|--------|----------------------------------------------------------|
| `N`        | 485100 | segment samples (11 s, the training segment)             |
| `STRIDE`   | 363825 | `int(0.75 * N)` (overlap 0.25); manifest `strideSamples`  |
| `NFFT`     | 4096   | FFT size; window = all ones                              |
| `HOP`      | 1024   |                                                          |
| `F`        | 2049   | bins kept: 0..2048 (all of them)                         |
| `T`        | 476    | frames                                                   |
| `TAIL`     | 1300   | zeros appended so that `(N + TAIL) / HOP = 475` is odd   |
| `CENTER`   | 2048   | reflect padding on both sides (NFFT / 2)                 |

(`T` must be even: the network takes a real FFT along the frame axis.)

Graph I/O:

| name       | dir | dtype   | shape                   |
|------------|-----|---------|-------------------------|
| `spec`     | in  | float32 | `[B, 4, 2049, 476]`     |
| `spec_out` | out | float32 | `[B, 4, 4, 2049, 476]`  |

Sources are `[drums, bass, other, vocals]`. The graph contains the model's
normalisation and de-normalisation. `run_segment(seg)`:

    spec     = STFT_scnet(seg)
    spec_out = graph(spec = spec)
    y[s, ch] = ISTFT_scnet(spec_out[s, 2*ch], spec_out[s, 2*ch+1])

### 3.1 `STFT_scnet` (per channel `x[0..N)`)

1. `xp = x ++ zeros[TAIL]`, `len(xp) = 486400 = 475 * HOP`.
2. Reflect-pad `CENTER` samples on both sides (edge sample not repeated):
   `y = xp[2048], ..., xp[1], xp[0], ..., xp[486399], xp[486398], ..., xp[484351]`,
   `len(y) = 490496`.
3. `Z[k, j] = (1/sqrt(NFFT)) * sum_{n=0}^{NFFT-1} y[j*HOP + n] * exp(-2i*pi*k*n/NFFT)`
   for `j in 0..476`, `k in 0..2049`. No window.

### 3.2 `ISTFT_scnet` (per source and channel)

1. Per frame, 2049 bins as given; set the imaginary parts of bin 0 and bin
   2048 to zero (a real inverse FFT ignores them).
2. `f_j[n] = sqrt(NFFT) * irfft(Z[:, j])[n]` (no window).
3. `acc[j*HOP + n] += f_j[n]`, `len(acc) = NFFT + 475*HOP = 490496`.
4. `x[m] = acc[m + CENTER] / env[m]` for `m in 0..N`, where `env[m]` is the
   number of frames overlapping that sample:

       env[m] = 3    for m in 0..1024
       env[m] = 4    for m in 1024..485100

This is `torch.istft` with no window: it divides by the overlap-added squared
window, which for a rectangular window is the frame count.

---------------------------------------------------------------------------

## 4. `melband_roformer` segment transform  (reserved)

Not built. The kind name is reserved in the manifest schema; no model of this
kind is in `dist/manifest.json` and `reference.py` raises for it.

---------------------------------------------------------------------------

## 5. Converted model: `htdemucs_6s` weights

Demucs' weights are not redistributable (LICENSES.md), so they are never part
of a SplitSound download. What ships is the graph without weights; the app
builds the weights file on the user's machine from Meta's own checkpoint.
`convert_reference.py` is the reference implementation of this section and
`dist/testdata/converter/` holds a miniature fixture and eleven inputs that
must be rejected.

### 5.1 Files

| file | origin | size | SHA-256 |
|------|--------|------|---------|
| `htdemucs_6s.onnx` | SplitSound download (`.onnx.zst`) | ~0.9 MB | manifest `sha256` |
| `htdemucs_6s.tensors.json` | SplitSound download | 86 kB | manifest `weights.tensors.sha256` |
| `5c90dfd2-34c22ccb.th` | **Meta**, `https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th` | 54996327 | `34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd` |
| `htdemucs_6s.weights` | written by the app | 109659984 | `42687f50d153e31e7b2702e603ce4c0ffe3b152c6a5ad684c2da54a702cff3d6` |

The conversion is a pure function of the checkpoint bytes, so the weights file
has one possible content; the app verifies its size and SHA-256 (manifest
`weights.sha256`) after writing it and before every use. Meta's checkpoint can
be deleted afterwards.

### 5.2 Procedure

1. Download the checkpoint from the `upstream.url` in the manifest (HTTPS,
   that host only). Check `sizeBytes`, then SHA-256, against `upstream`.
   Stop on any mismatch. Meta's own check is weaker: the 8 hex digits in the
   file name are the first 32 bits of the same SHA-256.
2. Open it as a zip and validate the container (5.3).
3. Interpret `archive/data.pkl` symbolically (5.4) and validate the result (5.5).
4. For each row of the tensor table, in order: locate the tensor, read its
   float16 data, widen to float32 (5.6), append to the output.
5. Verify the output's size and SHA-256, then move it into place next to the
   graph under the name `htdemucs_6s.weights`.

### 5.3 Container

A zip archive in which every member is **stored** (compression method 0), not
encrypted, and named exactly one of

    archive/data.pkl          the pickle, at most 1 MiB        (actual: 70079 bytes)
    archive/version           the bytes "3\n"
    archive/data/<n>          storage n, n a decimal number    (actual: 525 of them)

Anything else (another member, a compressed member, more than 4096 members, a
file over 128 MiB) is an error. Storages are raw little-endian float16.

### 5.4 Symbolic pickle interpreter

The pickle is protocol 2 and uses 28 opcodes. The interpreter is a stack
machine whose values are: none, bool, int, float, string, tuple, list, dict
(insertion-ordered, keys are strings or ints) and three **inert** kinds:

* `Global(module, name)`: two strings. Never looked up.
* `Call(global, args)`: a recorded REDUCE. Never executed.
* `Persistent(tuple)`: a recorded storage reference.

Only these five globals may appear; any other name is an error:

    collections.OrderedDict      torch._utils._rebuild_tensor_v2      torch.HalfStorage
    demucs.htdemucs.HTDemucs     fractions.Fraction

| byte | opcode | argument | effect |
|------|--------|----------|--------|
| `80` | PROTO | 1 byte | must be 2 |
| `2E` | STOP | | stack must hold exactly one value and the input must be exhausted; that value is the result |
| `28` | MARK | | push a mark |
| `4E` | NONE | | push none |
| `88` / `89` | NEWTRUE / NEWFALSE | | push true / false |
| `4B` | BININT1 | 1 byte unsigned | push int |
| `4D` | BININT2 | 2 bytes LE unsigned | push int |
| `4A` | BININT | 4 bytes LE signed | push int |
| `47` | BINFLOAT | 8 bytes **big**-endian IEEE double | push float |
| `58` | BINUNICODE | 4 bytes LE length, then UTF-8 | push string (length at most 1024) |
| `7D` / `5D` / `29` | EMPTY_DICT / EMPTY_LIST / EMPTY_TUPLE | | push an empty container |
| `74` | TUPLE | | pop to the mark, push those values as a tuple |
| `85` / `86` / `87` | TUPLE1 / TUPLE2 / TUPLE3 | | pop 1 / 2 / 3 values, push a tuple |
| `61` | APPEND | | pop a value, append it to the list on top |
| `65` | APPENDS | | pop to the mark, extend the list below with those values |
| `75` | SETITEMS | | pop to the mark (key, value, key, value, ...), insert into the dict below |
| `71` / `72` | BINPUT / LONG_BINPUT | 1 byte / 4 bytes LE index | memo[index] = top of stack (not popped) |
| `68` / `6A` | BINGET / LONG_BINGET | 1 byte / 4 bytes LE index | push memo[index]; error if unset or if it is a list or dict |
| `63` | GLOBAL | two `\n`-terminated ASCII lines | push `Global(module, name)` if allowed, else error |
| `52` | REDUCE | | pop args (a tuple) and a Global; if it is `collections.OrderedDict` with `()` push an empty dict, otherwise push `Call(global, args)` |
| `51` | BINPERSID | | pop a tuple, push `Persistent(tuple)` |

Every other byte is an error. Limits: 200000 opcodes, stack depth 4096,
65536 memo entries. In this file the memo is only ever read back for strings,
globals and tuples, so a port needs no shared mutable values; reading a list
or dict from the memo is rejected to keep that true.

### 5.5 Result

The root is a dict with keys `klass`, `args`, `kwargs`, `state` and optionally
`training_args`, `metrics` (both ignored). Require `klass ==
Global("demucs.htdemucs", "HTDemucs")` and `args == ()`. `state` maps each
parameter name to

    Call(Global("torch._utils", "_rebuild_tensor_v2"),
         (Persistent(("storage", Global("torch", "HalfStorage"), key, "cpu", storage_elements)),
          offset, shape, stride, requires_grad, backward_hooks))

For each: `key` is a decimal string naming `archive/data/<key>`; that member's
size must be `2 * storage_elements`; `shape` and `stride` are tuples of
positive ints of equal length (at most 8); `stride` must be the row-major
stride of `shape` (contiguous tensor); `offset + product(shape) <=
storage_elements`. The tensor's data is the `2 * product(shape)` bytes at byte
offset `2 * offset` of the member. Limits: 16 Mi elements per tensor, 64 Mi in
total (actual: largest 2654208, total 27414996).

### 5.6 float16 to float32

Per element, with `s` the sign bit, `e` the 5 exponent bits and `f` the 10
fraction bits of the little-endian uint16:

    e == 31            -> error (infinity or NaN; none occur)
    e == 0 and f == 0  -> s << 31                                   (signed zero)
    e == 0             -> normalise: shift f left until bit 10 is set, k shifts; f &= 0x3FF; e = 1 - k
    then               -> (s << 31) | ((e + 112) << 23) | (f << 13)

written as little-endian uint32. Every float16 value is exactly representable,
so there is no rounding.

### 5.7 Weights file and tensor table

`htdemucs_6s.tensors.json`:

    { "schemaVersion": 1, "model": "htdemucs_6s", "weightsFile": "htdemucs_6s.weights",
      "upstream": { "url", "sha256", "sizeBytes", "format": "torch-zip", "storageDtype": "float16" },
      "weights":  { "sha256", "sizeBytes" },
      "tensors":  [ { "key", "shape", "sourceDtype": "float16", "offset", "byteLength" }, ... ] }

The weights file is the listed tensors in table order, each as little-endian
float32 in row-major order starting at its `offset`. No header, no padding:
`offset` of a row is the sum of the `byteLength` of the rows before it, and
`byteLength == 4 * product(shape)`. The table has 525 rows, in checkpoint order.

### 5.8 How the graph finds the weights

Each of the graph's 525 initializers is named exactly like its state-dict key
and is declared as ONNX external data with `location = "htdemucs_6s.weights"`,
`offset` and `length` equal to the table's. The graph file contains no weight
data at all (its largest embedded constant has 96 elements). Observed with
onnxruntime 1.32.0:

* Session created **from a file path**: the location is resolved relative to
  the directory of the `.onnx`. Put `htdemucs_6s.weights` in the same
  directory. If it is missing, session creation fails ("External data path
  does not exist").
* Session created **from bytes in memory**: fails unless the session option
  `session.model_external_initializers_file_folder_path` is set to the
  directory holding the weights file.
* A location that is absolute or escapes the model directory (`../`) is
  rejected by onnxruntime itself. `verify_onnx.py --external-data
  htdemucs_6s.weights --max-inline-bytes 65536` additionally requires every
  reference to be that one plain file name with consistent, non-overlapping
  ranges.

onnxruntime does not check the content of the weights file; the app must
verify its SHA-256.

---------------------------------------------------------------------------

## 6. Instrument tagger (`kind: "tagger"`)

Model `efficientat_mn10_as`: EfficientAT MobileNetV3 (mn10_as), 527 AudioSet
classes. The whole mel frontend is inside the graph; the host only cuts
windows.

| name       | dir | dtype   | shape         | content                                  |
|------------|-----|---------|---------------|------------------------------------------|
| `waveform` | in  | float32 | `[1, 320000]` | 10 s mono at 32000 Hz, raw samples       |
| `probs`    | out | float32 | `[1, 527]`    | class probabilities (sigmoid applied)    |

Batch is fixed at 1.

### 6.1 Windowing and aggregation

Input: mono `x[0..L)` at 32000 Hz (stereo is averaged; resampling from the
track rate is the host's job and any band-limited resampler will do).

    WINDOW = 320000;  HOP = 160000
    starts = [0]                                           if L <= WINDOW
    starts = [k * HOP for k in 0..(1 + ceil((L - WINDOW) / HOP))]   otherwise
    for each start s:
        w[i] = x[s + i] if s + i < L else 0     for i in 0..WINDOW
        p[window] = graph(waveform = w)                    # [527]
    max[c]  = max  over windows of p[window][c]
    mean[c] = mean over windows of p[window][c]

### 6.2 What is inside the graph (for reference only)

pre-emphasis `y[n] = x[n+1] - 0.97*x[n]`; reflect pad 512; STFT `n_fft` 1024,
hop 320, symmetric Hann(800) centred in the frame, as a Conv1d with fixed
cosine/sine kernels (481 bins, those below 15 kHz); power; 128 Kaldi mel
filters 0-15000 Hz; `log(mel + 1e-5)`; `(x + 4.5) / 5`; MobileNetV3; sigmoid.
1000 frames per window.

### 6.3 Assets and thresholds

* `assets/audioset_labels.json`: JSON array of 527 display names, index =
  output position.
* `assets/instrument_map.json`: `classes` maps a class index (as a string) to
  `{ "label", "family", "audioset", "generic"? }`, `family` one of
  `vocals | drums | bass | guitar | piano | other`. 90 classes are mapped;
  every other class is not an instrument and is ignored. `generic: true` marks
  umbrella classes (for example "Guitar") that have more specific classes in
  the map.
* Detection rule: a mapped class is detected when its `max` probability is at
  least the threshold.

Measured on the 144 MUSDB18 7-second excerpts (`eval_musdb.py tagger`):

* **On the full mix the tagger is weak.** "Music" takes 0.91 on average and
  instrument classes are pushed down: with every excerpt containing vocals,
  drums and bass, the family scores have medians 0.065 (vocals), 0.010 (drums),
  0.005 (bass). At 0.05 that is 61 % recall for vocals, 7 % for drums, 2 % for
  bass. `mixThreshold` = 0.05 is provided for a quick pre-separation hint, but
  it must not be relied on for drums or bass.
* **On separated stems it works.** On the isolated stems, recall / false
  positive rate at `threshold` = 0.10: vocals 81 % / 0 %, drums 97 % / 2 %,
  bass 79 % / 9 % (1 % at 0.20). Typical top classes: "Drum kit", "Bass
  guitar", "Male vocals", and "Guitar" / "Synthesizer" / "Piano" on the
  `other` stem.

Recommended use: decide vocals / drums / bass presence from the energy of the
separated stems; run the tagger on each separated stem and only consider the
classes whose family can live in that stem (`other` stem: `guitar`, `piano`
and `other` families) to name what is in it. A bass guitar scores highly as
"Guitar" (0.50 median on bass stems), which is why the family restriction
matters.

---------------------------------------------------------------------------

## 7. Manifest (`dist/manifest.json`, schemaVersion 2)

    { "schemaVersion": 2, "models": [ <model>, ... ] }

Models are listed in the order the app should offer them. Every SHA-256 is of
the exact bytes of the named file; the app refuses a file that does not match.

Fields of every model:

| field | type | meaning |
|-------|------|---------|
| `id` | string | stable identifier, also the file stem |
| `kind` | string | `scnet`, `htdemucs`, `melband_roformer` or `tagger` |
| `revision` | int | bumped whenever the files of this id change |
| `name` | string | display name |
| `description` | string | one plain sentence |
| `source` | string | where the weights come from: `"GitHub"` or `"Meta"` |
| `restricted` | bool | true when the weights are not open source |
| `terms` | string or null | text to show before a restricted model is fetched |
| `license`, `attribution`, `origin` | string | licence summary, credit line, upstream project URL |
| `file`, `sha256`, `sizeBytes` | | the `.onnx` as used at run time (after decompression) |
| `acquire` | object | `{ "type": "download", "file": "<id>.onnx.zst", "sha256", "sizeBytes" }`: a single zstd frame (level 19, content size and checksum present) that decompresses to `file` |
| `opset` | int | ONNX opset of the graph (17) |

Separation models add:

| field | meaning |
|-------|---------|
| `sources` | output order of the graph |
| `targets` | the sources this model should be used for |
| `sampleRate`, `channels` | 44100, 2 |
| `segmentSamples`, `strideSamples` | `N` and `STRIDE` of section 1 |
| `nfft`, `hop`, `bins`, `frames`, `window` | transform constants (`window`: `"hann"` or `"rectangular"`); informative, the transform is fixed per `kind` |
| `dynamicBatch` | true: the graph has a symbolic `batch` dimension |
| `weightStorage` | `"float32"` or `"float16"` (stored as float16, cast to float32 inside the graph at load; all arithmetic is float32) |

A model whose weights are built on the user's machine (only `htdemucs_6s`)
adds `weights`, and its `file` / `acquire` describe the weight-free graph:

    "weights": {
      "file": "htdemucs_6s.weights", "sha256": "...", "sizeBytes": 109659984,
      "tensors": { "file": "htdemucs_6s.tensors.json", "sha256": "...", "sizeBytes": ... },
      "upstream": { "url": "https://dl.fbaipublicfiles.com/...", "sha256": "...", "sizeBytes": 54996327, "format": "torch-zip" }
    }

The tagger adds `sampleRate` (32000), `channels` (1), `windowSamples`,
`hopSamples`, `numClasses`, `dynamicBatch` (false) and two pinned assets:

    "labels":        { "file": "audioset_labels.json", "sha256", "sizeBytes" },
    "instrumentMap": { "file": "instrument_map.json",  "sha256", "sizeBytes" }

---------------------------------------------------------------------------

## 8. Weight storage and compression

Measured on the shipped files (`package.py`, zstd level 19, one frame):

| file | raw | `.zst` | ratio |
|------|-----|--------|-------|
| `scnet_large.onnx` (float16 storage) | 85.75 MB | 78.00 MB | 91.0 % |
| `scnet_small.onnx` (float16 storage) | 22.52 MB | 19.75 MB | 87.7 % |
| `efficientat_mn10_as.onnx` (float32) | 23.74 MB | 18.99 MB | 80.0 % |
| `htdemucs_6s.onnx` (graph only) | 0.92 MB | 0.04 MB | 4.3 % |

For comparison, not shipped: `scnet_small` with float32 weights is 44.54 MB
and compresses only to 39.43 MB (88.5 %), because trained float32 weights have
random low mantissa bits. Demucs is the opposite case: Meta stores its
checkpoints in float16, so the widened float32 weights file (109.66 MB) would
compress to 54.91 MB (50.1 %); it is built locally and never downloaded.

**float16 storage (SCNet).** The SCNet checkpoints are float32. The shipped
graphs keep every large weight tensor as a float16 initializer followed by a
`Cast` to float32, so all arithmetic is still float32 and onnxruntime folds
the casts once when the session is created. This halves the download (19.75
instead of 39.43 MB; 78.00 instead of 156.75 MB). The cost is the rounding
of the weights: the output differs from the float32-weight graph at 55 to 73
dB SNR, and the separation quality is unchanged (`scnet_small`, 50 MUSDB test
excerpts: mean of the per-stem median SDR 8.43 dB with either storage, every
stem within 0.01 dB). About 6400 of the weights per model are smaller than
the smallest float16 and become zero; none overflow.

**float16 arithmetic** (running the graph itself in half precision) was not
pursued: onnxruntime's CPU kernels for Conv, LSTM and MatMul are float32, so a
float16 graph on CPU runs slower or not at all, and it would only pay off on
GPU providers that were not available to measure here.

The tagger is left in float32: it is small and its input passes through a
logarithm, where weight rounding is not worth the 9 MB.

---------------------------------------------------------------------------

## Changelog

- v1: first draft: htdemucs I/O, STFT/ISTFT, chunking, shifts, ft bag.
- v2: reorganised around a common chunking section (1). Added `scnet` (3) and
  the tagger (6) as built. `N` and `STRIDE` now come from the manifest per
  model. htdemucs maths unchanged from v1. Only `htdemucs_6s` remains of the
  Demucs family, as a converted model.
- v2.1: added the conversion spec (5) and the manifest schema (7).
- v2.2: SCNet stride fixed at 363825; section 4 reserved (model deferred);
  added storage and compression measurements (8).
