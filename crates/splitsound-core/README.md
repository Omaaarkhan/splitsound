# splitsound-core

The SplitSound engine: decode a song, split it into stems, say which
instruments are in it, and export the result. A plain Rust library with no UI,
no async runtime and no Tauri dependency. Everything is blocking and
thread-safe; long calls take a `Progress` for reporting and cancellation.

This file is for whoever wires the app shell to the engine. It follows the
order a session goes through. Type names in `code` are in this crate; the
"contract" is `src/ipc/contract.ts`.

## Wiring in five minutes

```rust
use splitsound_core::{
    Engine, EngineConfig, FnProgress, CancelFlag, SplitOptions, SplitMode, Quality,
    SplitRequest, ErrorCode, DetectOutcome,
};

// Once, at start-up. Cheap: nothing is loaded until first use.
let mut config = EngineConfig::new(app_data_dir.join("models"));
config.onnxruntime_path = Some(resource_dir.join("onnxruntime"));   // the bundled library
config.acceleration = settings.acceleration;                         // Acceleration::{Auto,Cpu,Gpu}
config.threads = settings.threads;                                   // 0 = automatic
let engine = Engine::new(config)?;                                   // keep it for the app's lifetime (Send + Sync)

// import_track
let track = engine.import(&path, &cancel)?;          // ImportedTrack { audio, info, artwork }
splitsound_core::wav::write_wav_f32(&session_dir.join("mix.wav"), &track.audio, 44_100)?;

// background instrument scan: a few seconds; needs scnet_small and the tagger
match engine.detect(&track.audio, &cancel)? {
    DetectOutcome::Done(report) => { /* analysis: "done"; report.detections, report.suggested_mode */ }
    DetectOutcome::Unavailable { missing } => { /* analysis: "unavailable"; offer to fetch `missing` */ }
}

// estimate_split
let options = SplitOptions { mode: SplitMode::All, quality: Quality::Best };
let plan = engine.plan(options)?;                    // models, download_bytes, needs_consent, steps
let secs = engine.estimate_seconds(options, track.info.duration_sec)?;

// start_split (on the job thread)
let cancel = CancelFlag::new();
let progress = FnProgress::new(|u| emit("split:progress", u), cancel.clone());
let mut request = SplitRequest::new(options);
request.download_missing = true;
request.accept_restricted = args.accept_restricted;  // only after the user saw the terms
match engine.split_to_dir(&track.audio, request, &session_dir, &progress) {
    Ok(result) => { /* result.stems: Vec<StemResult>, result.model_ids, result.elapsed_sec */ }
    Err(e) if e.code() == ErrorCode::Cancelled => { /* back to `imported` */ }
    Err(e) => { /* IpcError { code: e.code().as_str(), message: e.message() } ; log e.detail() */ }
}
engine.unload();   // optional: frees the model sessions (hundreds of MB) between jobs
```

`cancel.cancel()` from any thread stops the job at the next model segment
(at most a few seconds), network read or packet.

## How the engine's types map onto the contract

| contract (`contract.ts`) | engine |
| --- | --- |
| `IpcError.code` / `.message` | `Error::code().as_str()` / `Error::message()`. `Error::detail()` is technical text for logs, never for users. The codes are the contract's list one to one, including `terms_required`. |
| `StemId`, `SplitMode`, `Quality`, `SplitOptions`, `ExportFormat` | same names, serde gives the contract's strings (`"vocals"`, `"all"`, `"best"`, `"studio"`, `"wav32f"`). `SplitMode::stems()` is `STEMS_BY_MODE`; `StemId::ORDER` is `STEM_ORDER`. |
| `Detection` | `Detection { label, family, confidence }` |
| `AnalysisStatus` | `DetectOutcome::Done(report)` is `done`, `DetectOutcome::Unavailable { missing }` is `unavailable`, an `Err` is `failed`. `report.suggested_mode` is `Session.suggestedMode`. |
| `TrackInfo` | `SourceInfo` (title, artist, album, duration_sec, sample_rate, channels, codec, bitrate_kbps, file_size_bytes, has_artwork). The shell adds `path`, `fileName` and the title fallback. `ImportedTrack::artwork` holds the cover bytes and media type for `get_artwork`. |
| `StemInfo` | `StemResult { id, path, stats: StemStats { present, active_fraction, peak_db, rms_db, activity }, contains }`; serialises flat with the contract's field names. |
| `SplitProgress` | `ProgressUpdate { stage, fraction, label, step, step_count, eta_sec }`; add `sessionId`. `fraction` never decreases within one call. |
| `SplitEstimate` | `Engine::plan()` gives `models` and `download_bytes`; `Engine::estimate_seconds()` gives `estimatedSec`. |
| `ModelStatus` | `ModelStatus` from `engine.models().list()` (the shell fills `downloadProgress` from progress events). |
| `ModelDownloadProgress` | `DownloadObserver::progress(model_id, received, total)` |
| `Peaks` | `engine::peaks_for_file(path, buckets)` then `Peaks::to_base64()` |
| `Settings.acceleration` / `.threads` | `Engine::configure(acceleration, threads)` |
| `AppInfo.gpu` / `.cpuThreads` | `Engine::gpu_info()` / `Engine::threads()` |

## The API, in the order a session uses it

**Import.** `Engine::import(path, progress)` decodes MP3, WAV, FLAC, Ogg
Vorbis, Opus, AAC (M4A and ADTS), ALAC (M4A and CAF) and AIFF to stereo
44.1 kHz float. Mono becomes dual mono; surround is downmixed (centre and
surrounds at -3 dB, LFE dropped, normalised so it cannot clip). Files over
two hours, with more than 8 channels, or damaged are errors (`decode_failed`),
not-audio is `unsupported_format`. The whole decode runs behind a panic
boundary. `decode::decode_file` is the same thing with explicit limits and
target rate.

**Listen.** `Engine::detect(mix, progress)` is the scan to run after import.
The tagger alone is poor on a full mix (ML measured 61 % recall for vocals,
7 % for drums and 2 % for bass, because "Music" swamps everything) and good
on separated stems, so `detect` does a **preview separation**: up to three
11 second segments around 25 %, 50 % and 75 % of the track (moved to the
loudest spot nearby, near-silent spots skipped, fewer for short tracks) go
through `scnet_small` once each. Then:

* vocals, drums and bass are *heard* when the separated stem is louder than
  -30 dB relative to the mix in some segment; the energy confidence rises
  linearly to 1 at -12 dB. These two thresholds are judgement, not fitted to
  the evaluation set;
* every stem loud enough is given to the tagger, restricted to the families
  that stem can hold. That is what names guitar, piano, synthesizer, strings
  and so on in "other", and what turns "drums" into "Drum kit";
* where the tagger also names the vocals, drums or bass, the strongest class
  gets `1 - (1 - p_tagger)(1 - c_energy)`; if it names nothing, the family is
  reported under its plain name with the energy confidence.

The result is `DetectOutcome::Done(DetectionReport { detections,
suggested_mode, vocals_heard, drums_heard, bass_heard, segments })`.
`suggested_mode` is `all` when a guitar or piano was heard in "other",
otherwise `band`; "no vocals heard" is reported but does not change it. If
`scnet_small` or the tagger is not installed the result is
`DetectOutcome::Unavailable { missing }`, not an error. **There is no
mix-only fallback**: its answers were not good enough to show. It takes
about 3.7 s for a 3.4 minute song on the (loaded) test machine, and returns
`busy` while a split is running.

**Plan.** `Engine::plan(options)` says which models the split uses, which are
installed, how much is left to download and, in `needs_consent`, which
missing models are *restricted*. Show each such model's `terms` and only then
call `split_to_dir` with `accept_restricted = true`; without it the call fails
with `terms_required` before anything is fetched.

**Split.** `Engine::split_to_dir(mix, request, out_dir, progress)` does the
whole job: download missing models (if `download_missing`), separate, measure
each stem, name what is inside each stem that has something in it, and write `<out_dir>/<stem id>.wav` as
32-bit float 44.1 kHz stereo (temporary name, then rename). One split runs at
a time; a second call returns `busy`. `Engine::split` is the in-memory core
without downloads, analysis or files.

**Play / draw.** `wav::MappedWav::open(path)` memory-maps a working file and
gives the samples as one interleaved `&[f32]` for the mixer.
`engine::peaks_for_file(path, buckets)` computes the waveform overview from the
map (5 ms for 3.4 minutes at 2000 buckets once the file is in the page
cache, 117 ms cold), so peaks need no cache.

**Export.** `engine::export_file(src_wav, format, dst, progress)` and
`engine::export_mixdown(&[(path, gain_db)], format, dst, progress)` write WAV
16 (TPDF dither) / 24 / 32-float, FLAC (24-bit) and MP3 (320 kbit/s).
Check `ExportFormat::is_available()` for MP3 (see "MP3 and the LGPL").

**Models.** `engine.models()` is the `ModelStore`:
`list()`, `status(id)`, `is_installed(id)`,
`install(id, accept_restricted, observer)`, `ensure(ids, accept, observer)`,
`remove(id)`, `verify(id)`, `import_file(path, accept_restricted)` (returns
the id it recognised, or `None`), `clean_temp_files()`.

## What each split runs

| mode | quality | passes |
| --- | --- | --- |
| vocals | fast | `scnet_small`: vocals |
| vocals | best | `scnet_large`: vocals |
| vocals | studio | `melband_roformer_vocals`: vocals |
| band | fast | `scnet_small`: vocals, drums, bass |
| band | best | `scnet_large`: vocals, drums, bass |
| band | studio | `melband_roformer_vocals`: vocals; `scnet_large` on the residual: drums, bass |
| all | fast | `htdemucs_6s` on the mix: vocals, drums, bass, guitar, piano |
| all | best | `scnet_large`: vocals, drums, bass; then `htdemucs_6s` on what is left: guitar, piano |
| all | studio | `melband_roformer_vocals`: vocals; `scnet_large` on the residual: drums, bass; then `htdemucs_6s` on what is left: guitar, piano |

Fast and best differ in the model, not in the chunk overlap: every pass uses
the model's own stride (25 % overlap for SCNet/HTDemucs; 50 % overlap for
Mel-Band RoFormer). ML measured tighter overlap as no better for the extra work.

"Other" (and "instrumental" in the two-stem mode) is always the mix minus the
named stems, so stems sum to the mix exactly. `separate::pipeline` holds this
table as code. In `studio`, vocals are isolated first with
`melband_roformer_vocals` (no track normalisation) before passing the vocal-free
residual to `scnet_large` for drums and bass (and `htdemucs_6s` for guitar and
piano in `all`). This was measured to improve all four MUSDB18 stems (`9.64 dB`
vs `9.19 dB` mean median SDR for `scnet_large`) and sharply reduces vocal bleed
(note: KimberleyJensen's checkpoint training data is undocumented and the 7 s
MUSDB18 excerpts are zero-padded to 8 s, so see `ml/BENCHMARKS.md` for
caveats).

### Guitar and piano: mix or residual

In `all` / `best`, `htdemucs_6s` listens to the **residual** (the mix minus
SCNet's vocals, drums and bass). This was **not measured**: MUSDB18, the only
ground truth available, has no guitar or piano stems, and the sum
guitar + piano + other is the same under both strategies, so its four-stem
scores cannot tell them apart. The residual cascade is the brief's stated
default for that case. `separate::pipeline::GuitarPianoSource` switches it in
one line once there is material to measure on.

## Models and security

The rules of `docs/architecture.md` ("Models are pinned") are implemented in
`store.rs`, `manifest.rs` and `checkpoint.rs`:

* The manifest (`models/manifest.json`) is compiled in; unknown fields, unsafe
  file names, non-HTTPS or non-allow-listed upstream URLs are rejected when it
  is parsed.
* Project downloads: HTTPS only, fixed base URL
  `https://github.com/Omaaarkhan/splitsound/releases/download/models-v1/`,
  redirects followed by hand and restricted to GitHub hosts. The download is
  cut off when it exceeds the pinned size and must match the pinned SHA-256
  before it is decompressed; zstd decompression (pure Rust) stops one buffer
  past the pinned output size; the result must match its own pinned SHA-256;
  then temp file, fsync, rename.
* Every load re-reads the file, checks size and SHA-256, and gives the
  **verified bytes** to ONNX Runtime from memory. A self-contained graph is
  checked to reference no external file; a graph with external weights is
  checked to reference exactly its own weights file, and those bytes are also
  passed from memory, so the runtime never resolves a path next to the model.
* Hand-imported files are recognised only by content (size and SHA-256 of an
  installed file, a compressed download or an upstream checkpoint).
* **Restricted models** (`htdemucs_6s`): the project ships only a weight-free
  graph. With `accept_restricted`, the engine downloads Meta's original
  checkpoint from `dl.fbaipublicfiles.com` (pinned size and SHA-256), reads it
  with a strict symbolic unpickler that cannot execute anything (28 opcodes,
  5 allow-listed names as inert tokens, bounded sizes, stored-only zip
  members), widens float16 to float32, writes the weights file to a temporary
  name, checks its pinned SHA-256 and renames it. The checkpoint is deleted
  afterwards. Without the flag: `terms_required`, nothing fetched. The same
  gate applies to importing those files by hand.

## ONNX Runtime

Loaded at run time (`ort` in `load-dynamic` mode) from one absolute path:
`EngineConfig::onnxruntime_path`, else `ORT_DYLIB_PATH`, else
`vendor/onnxruntime/<host>/`. The system search path is never used. One
library per process (the first `Engine` wins).

`scripts/fetch-onnxruntime.mjs` fetches Microsoft's official build (version
1.32.0 pinned), verifies it and copies it to `src-tauri/resources/onnxruntime/`.
**Its SHA-256 pins are not filled in yet**: when it was written,
`github.com/microsoft/onnxruntime` answered 404 from this machine, so the
official archives could not be hashed. Until someone runs
`node scripts/fetch-onnxruntime.mjs --print-hashes` and pastes the result in,
the script refuses every target. All measurements here used the ONNX Runtime
1.32.0 library from Microsoft's PyPI wheel through `ORT_DYLIB_PATH`.

Execution providers: CPU by default and for `auto` everywhere, as
`ml/BENCHMARKS.md` recommends (CoreML does nothing for SCNet and only helps
`htdemucs_6s`, after a long compilation). `gpu` tries CoreML on macOS and
falls back to CPU if the provider fails to register or the session fails to
build; DirectML is not wired (untested here). Two things that are easy to get
wrong and are handled in `runtime.rs`:

* `ort` enables ONNX Runtime's automatic device selection by default, which
  silently picks CoreML on macOS (and then spends minutes compiling). The
  engine sets the policy to "prefer CPU" so only its own choice applies.
* The graphs' `batch` dimension is left symbolic on the CPU provider. Fixing
  it triples peak memory.

Threads default to the number of performance cores on Apple Silicon and to
the available parallelism (capped at 16) elsewhere; one session runs at a
time, with a single inter-op thread.

## Performance

Measured with `splitsound bench` on a 3.4 minute track of real music (30
MUSDB18 excerpts joined; 204 s), Apple M4, 4 threads (the performance cores),
CPU provider, ONNX Runtime 1.32.0, release build, one process per row. The
machine was **not idle** (load average about 7 from other jobs), so these are
pessimistic; `ml/BENCHMARKS.md` has per-segment times.

| mode | quality | models | split time | speed | peak memory |
| --- | --- | --- | --- | --- | --- |
| vocals | fast | `scnet_small` | 27 s | 7.5x realtime | 0.94 GB |
| band | fast | `scnet_small` | 28 s | 7.4x | 1.08 GB |
| all | fast | `htdemucs_6s` | 30 s | 6.8x | 2.44 GB |
| vocals | best | `scnet_large` | 112 s | 1.8x | 2.08 GB |
| band | best | `scnet_large` | 112 s | 1.8x | 2.21 GB |
| all | best | `scnet_large` + `htdemucs_6s` | 142 s | 1.4x | 3.08 GB |

"Split time" is `Engine::split` (model loading included). `split_to_dir` adds
the stem measurements, tagging and file writing: for `all` / `best` on this track the whole job (tagging every stem included)
took 144 s and peaked at 2.86 GB. Decoding the
track took 0.2 s; the preview detection 3.7 s; peaks for a 3.4 minute working
file at 2000 buckets 5 ms warm, 117 ms cold.

**Memory.** ONNX Runtime needs about 0.8 GB to run a segment of
`scnet_small`, about 1.9 GB for `scnet_large` or `htdemucs_6s`, and about
3.6 GB peak RSS for `melband_roformer_vocals` (CPU, `memory_pattern = false`,
per-head time attention; `ml/BENCHMARKS.md`). On top of that the engine holds the mix, the residual
and one stereo accumulator per stem of the pass that is running (about 21 MB
per minute each). `split_to_dir` measures, writes and releases each stem as
soon as its pass is done, and on tracks over five minutes releases the model
before doing so. Peak resident memory of the whole process on a **ten
minute** track (`splitsound separate`, the app path):

| mode / quality | peak memory |
| --- | --- |
| band / fast | 1.79 GB (measured through `Engine::split`, which holds more) |
| band / best | 2.93 GB |
| all / fast | 3.32 GB |
| all / best | 3.51 and 3.68 GB (two runs) |

So the target of about 3 GB for a ten minute song is met by the two- and
four-stem modes and **missed by the six-stem modes by 0.3 to 0.7 GB**. The
cause is the per-stem accumulators (five of them in `all` / `fast`) next to a
1.9 GB model. The fix is to finalise finished chunks to disk as the pass
advances instead of accumulating whole tracks; that was not done.
`Engine::unload()` frees the sessions between jobs.

## Known limitations

* AAC in MP4/M4A keeps the encoder's priming samples (1024 to 2112 frames of
  near-silence at the start): Symphonia does not apply MP4 edit lists.
* Truncation is an error where the container records an exact length (WAV,
  AIFF, FLAC, MP4, MP3 with a LAME tag). Ogg files, raw ADTS and MP3 without
  a LAME tag carry no reliable length; a cut-off file of those kinds decodes
  to what is there.
* `Acceleration::Gpu` exists only on macOS (CoreML) and is not recommended.
* MP3 export has no ID3 tags.
* Downloads from the project's own release could not be tested against
  GitHub: `models-v1` is not published yet (the request fails cleanly with
  HTTP 404). The same code path was exercised from a local folder holding the
  real `.onnx.zst` files, and the upstream download from Meta was tested live.

## MP3 and the LGPL

MP3 export uses LAME through `mp3lame-encoder` / `mp3lame-sys`, which compile
LAME from vendored source with `cc` alone. LAME is LGPL; the binding crates
declare LGPL-3.0. It is behind the default-on cargo feature `mp3`; build with
`--no-default-features` for an LGPL-free engine (MP3 export then returns
`unsupported_format`). Shipping it statically linked in an MIT application is
compatible with the LGPL as long as users can rebuild the application with a
modified LAME, which an open-source project satisfies; flag for a licence
review all the same. Built and tested on macOS arm64 only.

## Tests

```sh
cargo test -p splitsound-core                    # unit + integration; model tests skip without models
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
SPLITSOUND_MODELS_DIR=/path/to/models \
  cargo test -p splitsound-core --test ml_parity -- --nocapture
cargo test -p splitsound-core --test opus_parity -- --nocapture   # Opus against libopus
```

`tests/fixtures/` holds the decoder fixtures (one small file per format, made
by `make_fixtures.sh` from synthetic signals) and the Opus parity set with
libopus reference decodes.

## Dependencies

All from crates.io, versions locked in `Cargo.lock`. "Unsafe" counts lines
mentioning `unsafe` in the crate's own source; "build" says whether it has a
build script.

| crate | licence | first published | downloads | unsafe | build | notes |
| --- | --- | --- | --- | --- | --- | --- |
| symphonia 0.6.1 | MPL-2.0 | 2020-09 | 22.8 M | no (`forbid`) | no | Decoders and demuxers. MPL is file-level copyleft; used unmodified. |
| opuscule =0.2.2 | MPL-2.0 | **2026-06** | **36 k** | no (`forbid`) | no | **Young, little used, machine-translated from libopus by its own account.** Pinned exactly; checked against the RFC 8251 vectors and against libopus on 17 files (see below); every packet decode is behind an error-and-panic boundary. |
| rubato 5.1 | MIT/Apache-2.0 | 2020-05 | 16 M | 52 lines | no | Resampler (FFT, synchronous). |
| realfft 3.5 (rustfft) | MIT | 2020-06 | 21 M | few; rustfft has SIMD intrinsics | no | STFT. |
| ort =2.0.0-rc.15 (+ort-sys) | MIT/Apache-2.0 | 2022-11 | 27 M | yes (FFI) | ort-sys has one | ONNX Runtime bindings. **Release candidate**, pinned exactly. `download-binaries` is off: the build script fetches nothing. |
| ureq 3.5 (rustls, ring, webpki-roots) | MIT/Apache-2.0 (ring: Apache-2.0 AND ISC) | 2018-06 | 276 M | 1 line; ring has C/asm | ring has one (compiles vendored C with `cc`) | Blocking HTTPS client. |
| ruzstd 0.9 | MIT | 2019-11 | 83 M | 39 lines | no | Pure-Rust zstd decoder. |
| zip =9.1.3 | MIT | 2014-11 | 355 M | 41 lines | no | Reads the upstream checkpoint; built without any compression feature. Pinned to a release a few weeks old: this crate publishes almost daily (four releases on the day this was written). |
| sha2 0.11 | MIT/Apache-2.0 | 2016-05 | 1.27 B | intrinsics | no | SHA-256. |
| serde 1, serde_json 1 | MIT/Apache-2.0 | 2014 / 2015 | 1.7 B | few | yes (version probes, no I/O beyond that) | Manifest, tables, result types. |
| memmap2 0.9 | MIT/Apache-2.0 | 2020-01 | 440 M | yes (mmap) | no | Memory-mapped working files. |
| flacenc 0.5 | Apache-2.0 | 2022-06 | **0.95 M** | 7 lines | yes (`built`, records build info) | Pure-Rust FLAC encoder. Modest usage; output is verified by decoding it back with Symphonia in the tests. |
| mp3lame-encoder 0.3, mp3lame-sys 0.2 | **LGPL-3.0** (LAME: LGPL) | 2022-12 | 1.2 M | yes (FFI) | mp3lame-sys compiles vendored LAME with `cc` | Optional (`mp3` feature). The vendored LAME source was not compared against upstream. |
| thiserror 2 | MIT/Apache-2.0 | 2019-10 | 1.9 B | no | yes (version probe) | Error derive. |

The CLI adds only `serde_json`.

### Opus decoder evidence

`opuscule` against the official RFC 8251 test vectors (12 vectors, decoded to
48 kHz stereo): every packet's final range matched; output within 1 LSB of
the reference decodes. Against libopus (through ffmpeg) on the 17 committed
files in `tests/fixtures/opus/`: lengths identical on every file (pre-skip
and end trimming included), SILK files bit-identical at 16 bits, hybrid and
CELT within 1 LSB (79 to 85 dB SNR, the limit of the 16-bit reference).
A 400 000 packet mutation fuzz produced 9 arithmetic-overflow panics in a
debug build and none otherwise, which is what the panic boundary is for.
