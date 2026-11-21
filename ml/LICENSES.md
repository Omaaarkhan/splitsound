# Model licences and provenance

One entry per model: what licence the **weights** are under, the primary
source for that (URL, quote, date), the code licence, the training data as
the author states it, and what remains uncertain. Statements quoted from
GitHub were read through the GitHub REST API on 2026-11-19 and are quoted
verbatim; the author association is the one GitHub reports.

None of this is legal advice. Where a licence rests on one sentence written
by an author in an issue thread, that is said plainly.

---------------------------------------------------------------------------

## Summary

| model | shipped? | code | weights | basis for the weights licence |
|-------|----------|------|---------|-------------------------------|
| `scnet_large`, `scnet_small` | yes, converted to ONNX | MIT | **MIT** | author's written confirmation (issue comment) |
| `melband_roformer_vocals` | yes, converted to ONNX | MIT | **MIT** | author's written statement + HF model card (below) |
| `efficientat_mn10_as` (tagger) | yes, converted to ONNX | MIT | **MIT by implication** | weights published in the MIT-licensed repository's own releases; no separate statement |
| `htdemucs_6s` | **graph only**; weights fetched from Meta by the user's machine | MIT | **not open source**, likely CC BY-NC | author statements below |
| `htdemucs`, `htdemucs_ft` | **no** (comparison exports only, never distributed) | MIT | not open source | same |

---------------------------------------------------------------------------

## SCNet and SCNet-large (`scnet_small`, `scnet_large`)

* **Project:** https://github.com/starrytong/SCNet , "SCNet: Sparse
  Compression Network for Music Source Separation" (Tong et al., ICASSP 2024,
  arXiv 2401.13276).
* **Code licence:** MIT, `LICENSE` at commit
  `5d95bf96b19c3eede63248d171efeca8e3abb948`: "MIT License. Copyright (c) 2024
  starrytong".
* **Weights licence:** MIT. Source: https://github.com/starrytong/SCNet/issues/35
  ("License clarification for the released SCNet / SCNet-large pretrained
  weights"), comment by **starrytong** (author association: OWNER),
  2026-07-17T06:52:01Z:

  > Hi, thank you for your kind words and for checking! I confirm that the
  > released SCNet and SCNet-large pretrained weights are distributed under the
  > MIT License, consistent with the source code. You are welcome to
  > redistribute the original checkpoints and format-converted versions,
  > including ONNX exports, as part of your MIT-licensed tool, with appropriate
  > attribution. Thank you for supporting SCNet!

* **Where the weights come from:** the author's Google Drive links in
  `README.md` at the commit above ("Download Model Checkpoint" for SCNet,
  "SCNet-large" and its `config.yaml`). URLs, sizes and SHA-256 are pinned in
  `weights.lock.json`; see SECURITY.md for how they were cross-checked.
* **Training data, as stated by the author:** README: "The model checkpoint
  was trained on the MUSDB dataset." Both published configs point `data.wav`
  at a `musdbhq` directory, i.e. MUSDB18-HQ. No extra data is mentioned.
* **Attribution used in the app and embedded in the ONNX metadata:** "SCNet:
  Sparse Compression Network for Music Source Separation, by Weinan Tong,
  Jiaxu Zhu, Jun Chen, Shiyin Kang, Tao Jiang, Yang Li, Zhiyong Wu and Helen
  Meng (ICASSP 2024). Code and pretrained weights MIT License, (c) 2024
  starrytong. https://github.com/starrytong/SCNet"
* **Caveats:**
  * The weights licence rests on one issue comment by the repository owner;
    the repository's `LICENSE` and README do not mention weights. The issue
    is open and the comment could be edited or deleted; a dated copy of the
    API response should be kept with the project's records.
  * The statement grants redistribution "as part of your MIT-licensed tool"
    in reply to another project's request. It reads as a general confirmation
    ("are distributed under the MIT License"), and MIT itself has no field of
    use restriction, but it was not addressed to SplitSound.
  * **MUSDB18-HQ is a research / educational dataset.** Its tracks carry mixed
    licences, many of them non-commercial (CC BY-NC-SA), and the dataset's own
    terms speak of academic use. Whether a licence on training data restricts
    weights trained on it is legally unsettled; the author's MIT grant covers
    what the author can grant and does not settle that question. This applies
    to every MUSDB-trained model, including this one.

---------------------------------------------------------------------------

## Mel-Band RoFormer vocals (`melband_roformer_vocals`)

* **Project:** https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model ;
  weights on https://huggingface.co/KimberleyJSN/melbandroformer (commit
  `ac9b0614ab3cd7f77219e18ba494dfd93956c348`, file `MelBandRoformer.ckpt`,
  913106900 bytes, Hugging Face LFS SHA-256
  `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`, model
  card `license: mit`, verified locally on download).
* **Code licence:** MIT. Model definition in
  `ZFTurbo/Music-Source-Separation-Training` at commit
  `1f700c386e0dd763fe0a57242bb9420dce987354` (`LICENSE`: "MIT License.
  Copyright (c) 2024 Roman Solovyev (ZFTurbo)", based on
  `lucidrains/BS-RoFormer`, MIT).
* **Weights licence:** MIT. Source read through the GitHub API:
  https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/issues/18
  ("License clarification for Kim_Vocal_2 checkpoint"), comment by
  **KimberleyJensen** (author association: OWNER), 2026-04-22T12:36:21Z:

  > @xocialize Sorry for the late reply, I have changed the license to MIT on
  > huggingface, that can apply to the Kim_Vocal_2 too, you are free to use
  > these models and redistribute them in any way you like. I hope this is ok
  > for you.

* **Attribution used in the app and embedded in the ONNX metadata:**
  "Mel-Band RoFormer vocal separation model by KimberleyJensen (MIT License)
  and ZFTurbo/Music-Source-Separation-Training (MIT License).
  https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model"
* **Training data:** not stated in the sources read. Treat as undocumented
  until the author's model card or repository says otherwise.
* **Caveats:** the same single-statement basis as SCNet; training data
  undocumented.

---------------------------------------------------------------------------

## EfficientAT mn10_as (`efficientat_mn10_as`, the instrument tagger)

* **Project:** https://github.com/fschmid56/EfficientAT , "Efficient
  Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation"
  (Schmid, Koutini, Widmer, ICASSP 2023).
* **Code licence:** MIT, `LICENSE` at commit
  `a425fdce92572e602a1d5634799bd9f1f2efa806`: "MIT License. Copyright (c) 2022
  Florian Schmid". GitHub's licence detection reports `MIT` for the repository.
* **Weights licence:** MIT **by implication**. The weights are assets of the
  repository's own release `v0.0.1` ("Pre-Trained Models and PaSST ensemble
  predictions", published 2022-11-17 by `fschmid56`; asset
  `mn10_as_mAP_471.pt` uploaded by `fschmid56`). README, line 3: "In this
  repository, we publish the pre-trained models and the code described in the
  papers", and "Pre-trained models are available in the Github Releases and
  are automatically downloaded from there." There is **no separate licence
  statement for the weights**.
* **Training data, as stated by the author:** README: "All models available
  are pre-trained on ImageNet [9] by default ..., followed by training on
  AudioSet [4]", by knowledge distillation from PaSST transformer ensembles.
* **Caveats:**
  * No explicit weights licence, unlike the three models above. If an
    explicit one is required, the fallback is PANNs CNN14 (weights published
    on Zenodo under CC BY 4.0, about 16 times larger) or YAMNet (Apache-2.0).
  * AudioSet's annotations are CC BY 4.0 (Google); the underlying audio is
    YouTube video with assorted terms. ImageNet images are for
    non-commercial research. The same open question about training data
    applies as for MUSDB.
  * The label list (`assets/audioset_labels.json`) is AudioSet's
    `class_labels_indices.csv` (CC BY 4.0, Google), taken from the EfficientAT
    repository at the commit above.

---------------------------------------------------------------------------

## HTDemucs v4 (`htdemucs_6s`; also `htdemucs`, `htdemucs_ft`)

* **Project:** https://github.com/facebookresearch/demucs (archived) and its
  maintained fork https://github.com/adefossez/demucs .
* **Code licence:** MIT ("Demucs code is released under the MIT license as
  found in the LICENSE file").
* **Weights licence: not MIT, not open source.**
  * https://github.com/facebookresearch/demucs/issues/327 ("License of
    pre-trained models"), comment by **adefossez** (Alexandre Defossez, the
    author; association: CONTRIBUTOR), 2022-05-23T15:36:04Z:

    > The model weights are not covered by the MIT license, and are provided
    > only for scientific purposes.

  * `README.md` of adefossez/demucs at commit
    `829d21890023800713e5dd231c67e43ecc06069b`, section "License":

    > Demucs code is released under the MIT license as found in the LICENSE
    > file. The license of the weights is not clear: MusDB contains some data
    > that were only usable for research purpose. I sadly cannot help granting
    > a license, both coming from Meta custom data, or from the original MusDB
    > data. The 6 source variant of htdemucs was trained only on Meta internal
    > dataset, which likely means CC-NC-BY license is applicable, as it was the
    > case for all models released by FAIR. Do not write me to ask about it, I
    > won't be able to clarify the situation further, either for MusDB or Meta
    > internal dataset.

* **Training data, as stated by the author:** `htdemucs` / `htdemucs_ft`:
  MUSDB18-HQ plus 800 additional songs (Demucs README); `htdemucs_6s`: "trained
  only on Meta internal dataset" (quote above).
* **What SplitSound does about it:**
  * No Demucs weight, in any format, is distributed by SplitSound.
  * `htdemucs_6s.onnx` as shipped contains the computation graph only (derived
    from the MIT code) and **no weight data**; `verify_onnx.py` enforces that
    and the export script checks that every weight initializer is external.
  * `htdemucs_6s.tensors.json` lists parameter names, shapes and offsets. It
    contains no weight values.
  * The user's machine downloads Meta's original file from Meta's host and
    converts it locally (CONTRACT.md section 5). The manifest marks the model
    `restricted: true` and carries the terms text shown to the user.
  * `htdemucs` and the four `htdemucs_ft` models can still be exported for
    comparison (`export_htdemucs.py --single-file`); those files stay on the
    build machine under `dist/compare/`.
* **Caveats:** whether an end user may use the weights at all outside
  "scientific purposes" is exactly what the author says is unclear. The
  restricted flag and terms text exist so that the user makes that decision
  knowingly. A weight-free graph plus a parameter table is, as far as we can
  tell, not a derivative of the weights, but that view has not been tested.

---------------------------------------------------------------------------

## Evaluation data (never shipped)

* **MUSDB18 7-second sample** (`MUSDB18-7-STEMS.zip`, release asset of
  sigsep/sigsep-mus-db `v0.4.0`, uploaded by `faroit`): 144 excerpts used for
  the SDR baseline and the tagger threshold study. MUSDB18 is for
  research / educational use with mixed track licences (largely CC BY-NC-SA).
  It is downloaded to `dist/eval/` on the build machine only, is gitignored,
  and no audio from it is part of any deliverable; only aggregate numbers are
  reported.
* The synthetic clip in `dist/testdata/` is generated by `synth.py` and is
  ours (MIT, like the rest of the repository).
