# Architecture

SplitSound is a desktop app that splits a song into stems (vocals, drums, bass,
guitar, piano, everything else) and tells you which instruments it hears.
Everything runs on the user's computer. There is no server and no Python at
runtime.

## Layout

| Path | What lives there |
| --- | --- |
| `crates/splitsound-core/` | The engine, a plain Rust library with no UI or Tauri dependency: decoding, resampling, STFT, model store, separation, instrument detection, stem analysis, peaks, export. |
| `crates/splitsound-cli/` | `splitsound` command-line tool over the engine. Used for testing and scripting. |
| `src-tauri/` | The desktop shell (Tauri 2): commands and events, the split queue, the session library, the playback mixer, settings, updater. |
| `src/` | The interface (React + TypeScript + Vite). Talks to the shell only through `src/ipc/contract.ts`. Runs in a plain browser against a mock backend for development and screenshot tests. |
| `ml/` | Python scripts that export the upstream models to ONNX, plus a numpy reference of the separation algorithm. Run by maintainers, never by the app. |
| `scripts/` | Build helpers (fetching ONNX Runtime, setting the version). |
| `vendor/` | Downloaded third-party binaries (ONNX Runtime). Not committed. |
| `.github/` | CI, weekly release, dependency updates. |

## How a song becomes stems

1. **Import.** The shell asks the engine to decode the file (MP3, WAV, FLAC,
   Opus, Ogg Vorbis, AAC/M4A, AIFF) to 44.1 kHz stereo float. The result is
   written to the session folder as `mix.wav` so the original can always be
   played and re-split.
2. **Listen.** An AudioSet tagger runs over the mix in 10-second windows and
   reports which instruments it hears. Each instrument maps to the stem it
   would land in, which is how the app suggests a split.
3. **Split.** The engine runs the separation models through ONNX Runtime in
   overlapping fixed-length segments. STFT and inverse STFT are done in Rust;
   the networks only ever see and return plain tensors.
   - Vocals, drums and bass come from **SCNet** (small for Fast, large for
     Best).
   - Guitar and piano come from **Demucs 6-stem**, the only practical model
     for them. Its licence is restricted, so it is optional and the app asks
     first (see Models).
   - "Other" is always defined as the mix minus every named stem, so the stems
     sum back to the original exactly.
4. **Check.** Each stem is measured (level, how much of the song it is active
   in). A stem with next to nothing in it is marked absent. The tagger runs
   again on the "other" stem to name what is left in it.
5. **Keep.** Stems are stored as 32-bit float WAV in the session folder. The
   mixer memory-maps them for playback; export converts them to the format the
   user picks.

## Processes and threads

One process. The webview draws the interface. Rust owns everything else:

- a job thread that runs one split at a time from a queue,
- a mixer thread that reads the memory-mapped stems, applies gain, mute and
  solo, resamples to the output device rate and feeds a lock-free ring buffer,
- the audio callback, which only copies out of that ring buffer.

## Models

Models are not shipped inside the installer. They are fetched the first time
they are needed, into the app data folder. The list of models, their sizes,
licences and SHA-256 digests is `crates/splitsound-core/models/manifest.json`,
compiled into the binary. `ml/LICENSES.md` records where each licence claim
comes from.

| Model | Used for | Weights licence | How the app gets it |
| --- | --- | --- | --- |
| SCNet Small, SCNet Large | vocals, drums, bass, other | MIT (stated by the author) | zstd-compressed ONNX from this repository's `models-v1` release |
| Instrument detector (EfficientAT) | naming the instruments | MIT | same |
| Demucs 6-stem | guitar, piano | **Not open source.** Published by Meta for research; likely CC BY-NC | downloaded from Meta by the user's own copy of the app, after they agree to the terms |

**Why Demucs is handled differently.** The Demucs code is MIT but its author
has said the weights are not. So this project does not redistribute them in
any form. What it ships is the network's *graph* with no weights in it
(derived from the MIT code) and a table of where each tensor goes. When a
user agrees to the terms, the app downloads Meta's original checkpoint from
Meta's server, checks it against a pinned digest, reads the tensors out of it
with a strict parser that cannot execute anything, writes them to a weights
file next to the graph, and checks that file against a second pinned digest.
A model marked `restricted` is never fetched as a side effect of anything
else.

## Security model

The app opens files from the internet (songs) and downloads large binaries
(models), so these rules are requirements, not suggestions.

**Models are pinned.** The manifest is compiled in, never fetched. A download
is accepted only if it comes over HTTPS, is no larger than the manifest says,
and its SHA-256 matches, both for the compressed file and for the decompressed
ONNX file. Decompression is bounded by the expected size. Installation is
write-to-temp then rename. The digest is checked again when a model is loaded.
Files the user imports by hand are accepted only if they match a pinned
digest. There is no way to load an arbitrary model.

**Models are inert.** Exported ONNX files use only standard ONNX operators
and no custom ones, and either embed all their weights or refer to exactly
one sibling weights file by a fixed name (`ml/verify_onnx.py` enforces this).
`ml/SECURITY.md` records the evidence for every file.

**Pickles are never executed.** PyTorch checkpoints are pickles and can run
code when loaded the usual way. Maintainers only open them in the export
environment after fetching them from the original publisher, checksumming
and statically scanning them. The one checkpoint the app itself reads
(Demucs) is parsed by a purpose-built reader that understands only the few
record types that file contains, treats every name in it as inert data,
bounds every size before allocating, and rejects anything unexpected.

**ONNX Runtime is the official build.** The shared library is taken from
Microsoft's own NuGet package by `scripts/fetch-onnxruntime.mjs`, which pins
the version, the package hash NuGet publishes, and the SHA-256 of each library
inside it. The app loads it from an absolute path inside its own bundle.

**Audio files are untrusted.** Decoding uses memory-safe Rust decoders
wherever one exists. Durations, channel counts and sizes read from headers are
bounded before anything is allocated. A file that cannot be decoded produces
an error, never a crash: jobs run behind a panic boundary.

**The webview has the least privilege that works.** A strict Content Security
Policy (no remote content, no inline script), no shell access, only the
Tauri capabilities the app uses. Text from tags is rendered as text. Every
command validates its arguments: session ids are checked against a fixed
pattern and resolved inside the library folder, file names are sanitised, and
`reveal_path` only accepts paths the app itself produced.

**Updates are signed.** The updater only installs releases signed with the
project's key, and each signature is bound to its version so an old build
cannot be offered as a new one. The interface can ask "is there an update?"
and "install it", nothing more: where updates come from is fixed in the app.
Releases are built by CI from `main`, with checksums and build provenance
attached. The signing key lives in a GitHub environment that only `main` can
use; build jobs and pull-request checks never see it.

**Dependencies are locked and audited.** Lockfiles are committed and CI
installs from them. `cargo deny` and `npm audit` run on every change. GitHub
Actions are pinned to commit SHAs with minimal token permissions. Dependency
updates wait out a cooldown period before they are proposed.

**Nothing leaves the machine.** No telemetry, no accounts. The only network
requests are model downloads and the update check, both to GitHub, and the
optional Demucs download from Meta's server.