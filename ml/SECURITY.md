# Security record for the ML pipeline

What was downloaded, from where, how it was checked, how it is loaded, and
what could not be verified. Raw evidence (scanner output, download headers,
audit output, verifier reports) is in `dist/security/` on the build machine.
All times are UTC. Build machine: macOS arm64; torch 2.15.0, onnx 1.23.2,
onnxruntime 1.32.0, demucs 4.0.1.

## 1. Rules the pipeline enforces

| rule | where it is enforced |
|------|----------------------|
| Files come only from the publisher's own host, over HTTPS; every redirect hop is checked against an allow-list | `weights.py` (`_check_url`, `_CheckedRedirects`), hosts in `weights.lock.json` |
| Every downloaded file is pinned by size and full SHA-256 and refused on mismatch | `weights.lock.json`, `weights.verify` |
| A checkpoint is scanned statically before anything loads it, and its set of pickle globals must equal the recorded set exactly | `scan_pickle.py` (stdlib only), `weights.verify` |
| Checkpoints are loaded only with `torch.load(weights_only=True)`, or not unpickled at all | `weights.py`, `export_scnet.py`, `export_tagger.py`, `convert_reference.py` |
| A checkpoint referencing globals outside torch / numpy / collections / its own package cannot be passed to `torch.load` without owner approval | `weights.require_loadable`, `approvedExceptions` in the lock (empty) |
| Python dependencies: exact versions and hashes, PyPI only | `uv.lock`, every command uses `--locked` |
| Exported ONNX files are inert data | `verify_onnx.py` (exit status 1 on violation), called by every export and by `package.py` |
| Manifest digests are computed from the final bytes on disk | `package.py` |
| Downloaded audio is data only | `eval_musdb.py` (member-name pattern, ffmpeg decode, nothing executed) |

There is no `weights_only=False`, `pickle.load`, `eval` or `exec` anywhere in `ml/`.

## 2. Provenance of every downloaded file

Downloaded by `weights.py fetch` on 2026-11-20 between 03:38 and 03:59
(headers and redirect chains in `dist/security/downloads.json`).

### Model weights

| file | URL | bytes | SHA-256 | publisher's own checksum |
|------|-----|-------|---------|--------------------------|
| `5c90dfd2-34c22ccb.th` (htdemucs_6s) | `https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th` | 54996327 | `34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd` | SHA-256 prefix `34c22ccb` in the file name: **matches** |
| `955717e8-8726e21a.th` (htdemucs) | same directory | 84141911 | `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4` | prefix `8726e21a`: matches |
| `f7e0c4bc-ba3fe64a.th` (ft drums) | same directory | 84141271 | `ba3fe64ae8ef66ac9a4857222ce48efbdc5eb3ad375cb79dd13debee5aaa4066` | prefix `ba3fe64a`: matches |
| `d12395a8-e57c48e6.th` (ft bass) | same directory | 84141271 | `e57c48e6b0e38af4f7118d7bd08c49f0a0c0edf7d09143bdd902ea0d237303e6` | prefix `e57c48e6`: matches |
| `92cfc3b6-ef3bcb9c.th` (ft other) | same directory | 84141271 | `ef3bcb9c8b40d14ae5d51b6db2587339cc12c6b77c0be151ce6d69002e087bf2` | prefix `ef3bcb9c`: matches |
| `04573f0d-f3cf25b2.th` (ft vocals) | same directory | 84141271 | `f3cf25b222c4eed7cd49dd8b2c9597d50c18bd154090f7b919cfa5f93cf22c49` | prefix `f3cf25b2`: matches |
| `scnet_small.th` (Drive name `checkpoint.th`) | `https://drive.usercontent.google.com/download?id=1CdEIIqsoRfHn1SJ7rccPfyYioW3BlXcW&export=download&confirm=t` | 42434545 | `f8aff3bc8dd816aad392800d9d69519add719a2fbd22550af6f8cb2138d25ff5` | **none published** |
| `scnet_large.th` (Drive name `SCNet-large.th`) | `...download?id=1s7QvQwn8ag9oVstGDBQ6KZvacJkvyK7t&export=download&confirm=t` | 168848417 | `719e5abb8ed920305dad546ac3cd6fb0b1e9c3092d14ce21827bfc0423af3070` | **none published** |
| `mn10_as_mAP_471.pt` (tagger) | `https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt` | 19708753 | `0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e` | **none published** (GitHub API `digest: null`; its stated size matches) |

**Demucs.** Host and path are the ones hard-coded in the hash-locked
`demucs==4.0.1` package (`ROOT_URL` in `pretrained.py`, names in
`remote/files.txt`). Meta's check is the 8 hex digits in each file name, the
first 32 bits of the file's SHA-256; demucs verifies it (`check_hash=True`).
That check **ran and passed** when demucs first fetched the files, and
`weights.py` repeats it on every verify. Thirty-two bits do not resist a
deliberate substitution, so the full digests above were computed here and
pinned; they are trust-on-first-use. A second, independent download through
`weights.py` was byte-identical to the first. Only `5c90dfd2-34c22ccb.th`
matters for what ships; the other five are pinned for comparison exports and
were not used after the events in section 4.

**SCNet.** The author publishes the checkpoints only as Google Drive links, in
`README.md` of `starrytong/SCNet` at commit
`5d95bf96b19c3eede63248d171efeca8e3abb948` (read at that commit). Google Drive
is therefore the canonical source, and `drive.usercontent.google.com` is
allowed **for these two files and the large model's `config.yaml` only**
(`extraAllowedHosts` on those lock entries). The author states no checksum,
so the digests are trust-on-first-use, strengthened by a cross-check against
the copies mirrored in ZFTurbo/Music-Source-Separation-Training:

| mirror asset | bytes | SHA-256 | result |
|--------------|-------|---------|--------|
| `v.1.0.6/scnet_checkpoint_musdb18.ckpt` | 42434986 | `1bc0d1abb20bfdf966dcd07637bafd03e4bc13653d09ef18bc9b3e342eafe2aa` | all 372 tensors **bit-identical** to the author's small checkpoint |
| `v1.0.9/SCNet-large_starrytong_fixed.ckpt` | 168852258 | `65900dfa07d6b6e5d784c0f143920200a4bd281d6e78a806c549d0b912d5885e` | all 372 tensors **bit-identical** to the author's large checkpoint |

The mirror files are not byte-identical (re-saved without the `best_state`
wrapper); the comparison is of the state dicts
(`dist/security/scnet_mirror_crosscheck.json`). Google Drive reported
`x-goog-hash: crc32c=Q3CNNA==` (small) and `crc32c=dhF5yA==` (large). The
mirror copies were used for this comparison only and then deleted; nothing
that ships comes from them.

**Tagger.** Asset of the author's own GitHub release (`v0.0.1`, published and
uploaded by `fschmid56`), redirected by GitHub to
`release-assets.githubusercontent.com`.

### Source code fetched at a pinned commit (not installed as packages)

No package is installed from a git URL. Two upstream projects are not on
PyPI; the files needed from them are fetched from
`raw.githubusercontent.com/<owner>/<repo>/<full commit SHA>/<path>`, pinned by
SHA-256 in the lock, and were read in full before first use.

| project | commit | files | review notes |
|---------|--------|-------|--------------|
| `starrytong/SCNet` | `5d95bf96b19c3eede63248d171efeca8e3abb948` | `scnet/SCNet.py`, `scnet/separation.py`, `scnet/__init__.py`, `conf/config.yaml` | model definition only: no I/O, network, subprocess or dynamic code. `apply.py` / `inference.py` (which import `accelerate`) are not fetched. Configs are parsed with `yaml.safe_load` |
| `fschmid56/EfficientAT` | `a425fdce92572e602a1d5634799bd9f1f2efa806` | `models/mn/model.py`, `utils.py`, `block_types.py`, `attention_pooling.py`, `models/preprocess.py`, `metadata/class_labels_indices.csv` | `model.py` can download weights by itself when given a `pretrained_name`; it is always called with `pretrained_name=None`, so that path never runs. `helpers/utils.py` (reads a CSV from the working directory at import) is not fetched; the one name imported from it is supplied locally |

The large-model `config.yaml` comes from the author's Drive link
(1080 bytes, `629a4901184bf1d3a75b0b13904f35974785aa042cad3c010fd576248cdce3f0`).

### Evaluation audio

| file | URL | bytes | SHA-256 | publisher checksum |
|------|-----|-------|---------|--------------------|
| `MUSDB18-7-STEMS.zip` | `https://github.com/sigsep/sigsep-mus-db/releases/download/v0.4.0/MUSDB18-7-STEMS.zip` | 147209385 | `365f66acffc57282e474a4c24b32afec3678258c50b8d45646ac849b0a287df3` | none published (size matches the GitHub API) |

This is the dataset authors' own release asset (uploaded by `faroit`); the
URL is `sample-url` in `musdb/configs/mus.yaml` of sigsep/sigsep-mus-db at
commit `c0181671b9955530f7b03654a322df3f225263a6`. The archive holds 144
`.stem.mp4` files and two directory entries, nothing else. It is unpacked with
each member name matched against `^(train|test)/[^/\\]+\.stem\.mp4$`, and the
files are only decoded by ffmpeg (argument list, no shell). Nothing from it is
executed and nothing is committed.

## 3. Static inspection of the pickles

Each checkpoint was disassembled, never unpickled, by three tools:
`scan_pickle.py` (this repository; `pickletools.genops`), `picklescan` 1.0.5
and `fickling` 0.1.13 (both hash-locked in the `security` dependency group).
All three report the same globals for every file.

| checkpoint | globals referenced | outside the base policy |
|------------|--------------------|-------------------------|
| SCNet small, SCNet large | `collections.OrderedDict`, `torch.FloatStorage`, `torch._utils._rebuild_tensor_v2` | none |
| EfficientAT `mn10_as` | the same three plus `torch.LongStorage` | none |
| Demucs `5c90dfd2` (htdemucs_6s) | `collections.OrderedDict`, `torch.HalfStorage`, `torch._utils._rebuild_tensor_v2`, `demucs.htdemucs.HTDemucs`, **`fractions.Fraction`** | `fractions.Fraction` |
| Demucs, the other five | the same plus `numpy.core.multiarray.scalar`, `numpy.dtype`, **`_codecs.encode`** | `fractions.Fraction`, `_codecs.encode` |

Every zip holds only `data.pkl`, `version` and raw tensor storages.
picklescan: 0 dangerous globals, 0 infected files, everywhere. fickling rates
every file `LIKELY_UNSAFE`, which is its verdict for any pickle that imports a
non-standard-library module, i.e. for every PyTorch checkpoint; it found no
`os`, `subprocess`, `builtins`, `eval` or similar.

What the two out-of-policy names do, read from the opcodes: `fractions.Fraction`
is applied once to the string `'39/5'` (the 7.8 s segment length);
`_codecs.encode` is applied to 8 raw bytes and `'latin1'` to rebuild numpy
float64 training metrics. Both are standard library and look harmless. They
are nevertheless outside the project's rule (torch, numpy, collections, the
model's own package), so this was reported and **no approval was given**;
`approvedExceptions` is empty for all six files and `weights.require_loadable`
refuses to hand any of them to `torch.load`.

## 4. How each checkpoint is loaded

| checkpoint | method |
|------------|--------|
| SCNet small / large | `torch.load(weights_only=True)`, no extra allow-list; the `best_state` dict goes to `load_state_dict(strict=True)` |
| EfficientAT | `torch.load(weights_only=True)`, no extra allow-list; `load_state_dict(strict=True)` |
| SCNet mirror copies | `torch.load(weights_only=True)` after the same scan, for the tensor comparison only |
| Demucs `htdemucs_6s` | **not unpickled at all.** `convert_reference.py` interprets the pickle symbolically: 28 known opcodes, 5 allowed names kept as inert tokens, REDUCE recorded and never executed, all sizes bounded. The float16 tensors are copied out of the zip. The export builds `HTDemucs(**kwargs)` from the plain values and constructs `Fraction(39, 5)` itself from the recorded string |
| Demucs, the other five | not loaded since the event below; the same symbolic reader would be used |

**Disclosure.** Before the security requirements were issued, all six Demucs
checkpoints were loaded once through demucs' stock loader (`torch.hub`, which
calls `torch.load` with `weights_only=False`) in order to read their
configurations. That was on this build machine, at about 03:24 on 2026-11-20.
The static scans done afterwards show nothing in those files that could have
done harm, but the load happened before the scan, not after it. It has not
been repeated; those copies were deleted and the files re-downloaded through
`weights.py`.

The same symbolic reader is the reference for what runs on end users'
machines (CONTRACT.md section 5). `dist/testdata/converter/` contains eleven
inputs it must reject (foreign global, unknown opcode, wrong protocol,
trailing or truncated pickle, extra zip member, compressed member, NaN
weight, short storage, wrong version, wrong pinned hash).

## 5. Python dependencies

* `uv.lock`: 88 packages, every one with `source = { registry =
  "https://pypi.org/simple" }` (plus the project itself), 518 SHA-256 hashes.
  No git, URL or path dependency. `[tool.uv] index-url` pins PyPI.
* Direct dependencies: `torch`, `demucs==4.0.1`, `onnx`, `onnxruntime`,
  `numpy`, `soundfile`, `zstandard`. Groups, not installed by default:
  `tagger` (`torchvision`, needed only to import the upstream MobileNet
  code), `security` (`fickling==0.1.13`, `picklescan==1.0.5`,
  `pip-audit==2.10.1`). `musdb`, `museval`, `onnxscript`, `psutil`, `librosa`
  were considered and left out.
* `demucs` is published as a source distribution only, so installing it runs
  its `setup.py`; its hash is locked like the others.
* Audit: `pip-audit 2.10.1 -r requirements.locked.txt --require-hashes
  --disable-pip` over the full locked set including all groups, last run
  2026-11-20: **"No known vulnerabilities found"**
  (`dist/security/pip-audit.txt`). `uv` 0.10.3 has no `audit` subcommand.

## 6. The exported ONNX files are inert

`verify_onnx.py` asserts, for each file: only the default operator domain;
every operator exists in the standard set at the declared opset; no `If`,
`Loop`, `Scan` or any subgraph attribute; no model-local functions; no sparse
initializers; no external data (with the one exception below); and
`onnx.checker.check_model(path, full_check=True)`. Results on the final files
(`dist/security/onnx_verify.json`, `onnx_verify_htdemucs_6s.json`):

| file | opset | domains | nodes | subgraphs | functions | external data | checker |
|------|-------|---------|-------|-----------|-----------|---------------|---------|
| `scnet_large.onnx` | 17 | default only | 2592 | 0 | 0 | none (316 initializers embedded) | ok |
| `scnet_small.onnx` | 17 | default only | 2585 | 0 | 0 | none (316 embedded) | ok |
| `efficientat_mn10_as.onnx` | 17 | default only | 429 | 0 | 0 | none (131 embedded) | ok |
| `htdemucs_6s.onnx` | 17 | default only | 4959 | 0 | 0 | **525 initializers in `htdemucs_6s.weights`**, by design | ok |

`htdemucs_6s.onnx` is the deliberate exception: it is distributed without
weights. It is checked with `--external-data htdemucs_6s.weights
--max-inline-bytes 65536`, which requires every external reference to be that
one plain file name (no directory, no `..`, not absolute) with offsets and
lengths that match each tensor's shape and do not overlap, and requires the
graph file itself to hold almost no tensor data (it holds 13648 bytes of
shape constants; the largest has 96 elements). Under the default policy the
same file is rejected. onnxruntime 1.32.0 independently refuses an external
location that is absolute or leaves the model directory (tested).

onnxruntime does not check what is inside the weights file. The app must
verify its SHA-256 (`42687f50d153e31e7b2702e603ce4c0ffe3b152c6a5ad684c2da54a702cff3d6`,
109659984 bytes) after conversion and before use.

## 7. Manifest digests and reproducibility

`package.py` computes every SHA-256 in `dist/manifest.json` from the files on
disk at packaging time and cross-checks each graph against its export report.

**Reproducibility: yes, on this machine.** After the manifest was written,
all four exports were run again into an empty directory and every output was
compared with the manifest:

| output | reproduced bit for bit |
|--------|------------------------|
| `scnet_large.onnx`, `scnet_small.onnx`, `efficientat_mn10_as.onnx` | yes |
| `htdemucs_6s.onnx` (weight-free graph) and `htdemucs_6s.tensors.json` | yes |
| `htdemucs_6s.weights` (written by `convert_reference.py` from Meta's file) | yes |
| every `.onnx.zst` (re-compressed from the `.onnx`) | yes |

This holds for one machine and one set of locked versions. The graph bytes
embed the exporter's version string and depend on the torch, onnx and
zstandard versions, so a different lockfile is expected to change the
digests; a different operating system or CPU has not been tried. The weights
file of `htdemucs_6s` is different in kind: it is a pure function of Meta's
checkpoint (lossless float16 to float32 widening in a fixed order), so its
digest must be the same on every platform, and the app relies on that.

Final digests (also in `dist/manifest.json`):

| file | bytes | SHA-256 |
|------|-------|---------|
| `scnet_large.onnx` | 85750454 | `6f21c2bd71ee38dd49b92d3c04a4a0505f9f876cb4a061e919574bc6ecf02cab` |
| `scnet_large.onnx.zst` | 77996104 | `05b28dee5fe5da3456c66d477d2250af70dac0d4d1683666d99a8f81ffc4f338` |
| `scnet_small.onnx` | 22516980 | `358d85f1f89727fa4a64553ab29493d32d70c9be7a888d20ca0fbc2fc7e4a440` |
| `scnet_small.onnx.zst` | 19752783 | `1aed68ce5f7454df1a0942e9ffd3b5846d682a0772bed27576b8376a3aeb455c` |
| `htdemucs_6s.onnx` | 921793 | `dabfbabffda5c64c8e7645cf8720028189f27825f3b6fb00c19a21d6f622cde2` |
| `htdemucs_6s.onnx.zst` | 39985 | `6b650670a4c6dcb0366ef823a84e5f404abea6af59e7de07dc7455d413ae064b` |
| `htdemucs_6s.tensors.json` | 85865 | `37e74ecc3e0983a03bd73a5b957eb29982c3fd5682f9a3bd5513f6d8ba5d1728` |
| `htdemucs_6s.weights` (built locally, never shipped) | 109659984 | `42687f50d153e31e7b2702e603ce4c0ffe3b152c6a5ad684c2da54a702cff3d6` |
| `efficientat_mn10_as.onnx` | 23741689 | `7cc428d1869758cc3bfb3b3774abec63d525fb48b96d8a08b0eadb9427e5a435` |
| `efficientat_mn10_as.onnx.zst` | 18989845 | `42eeeee67ab47511925826ffbb76a7d906303990eaad4170a53be962b38832b7` |
| `assets/audioset_labels.json` | 7921 | `810eedff382eaaa456c1151eceec75e1dfbbbc8949a118752350dbf22648a874` |
| `assets/instrument_map.json` | 10710 | `07ef7c8905c042baed4257f1c4ade30d6766b23c44078b7d983cd5615e1d6d1e` |

The manifest itself is not signed. Whoever publishes it must protect it by
other means (shipping it inside the signed application, or signing it):
the digests only help if the list of digests is trustworthy.

## 8. What could not be verified

* **Publisher checksums.** SCNet (both), the tagger and the MUSDB sample have
  none; their SHA-256 pins are trust-on-first-use. For SCNet that is backed
  by the bit-identical mirror tensors; for the tagger and the MUSDB sample
  only by the size GitHub reports. Demucs has a 32-bit prefix only.
* **Google Drive as a host.** A Drive link can be replaced by its owner at
  any time and Drive offers no signed release. The pin detects a change; it
  cannot tell a legitimate update from a compromise.
* **The first Demucs load** (section 4) happened before the files were
  scanned.
* **Code review depth.** The upstream model files were read by one reviewer in
  one pass. PyPI packages were not reviewed at all beyond the hash lock and
  the vulnerability audit; `torch`, `onnx` and `onnxruntime` are large native
  dependencies taken on trust.
* **Weights as data.** The checks show that the checkpoints and ONNX files
  cannot execute code through the loaders used. They say nothing about
  whether the weights themselves were tampered with at the publisher
  (a backdoored model would pass every check here).
* **Licence statements** rest on issue comments read through the GitHub API on
  2026-11-19; they could later be edited (LICENSES.md).
* **Other platforms.** Everything was built and run on one macOS arm64
  machine. The exports were not rebuilt on Linux or Windows, so
  cross-platform reproducibility of the digests is untested.
* **The Rust converter** does not exist yet; only its Python reference and
  test fixtures do.
