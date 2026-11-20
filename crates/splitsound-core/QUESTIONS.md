# Questions for the ML contract

Written by the engine side. Each item says what the engine does today, so
nothing here blocks anything.

1. **Tensor table.** The engine compiles `htdemucs_6s.tensors.json` in and
   checks its SHA-256 against the manifest (`weights.tensors.sha256`),
   rather than downloading it. Is that the intent? (The manifest's `labels`
   and `instrumentMap` digests are checked the same way, by a unit test.)

2. **Guitar and piano ground truth.** Deciding between running
   `htdemucs_6s` on the mix or on the residual needs material with guitar
   and piano stems. MUSDB has none, and guitar + piano + other is identical
   under both strategies. The engine uses the residual cascade unmeasured.

3. **Instrument map hierarchy.** To collapse umbrella classes ("Brass" when
   "Trumpet" was heard) the engine would like an optional `parent` field
   (class index of the umbrella) on entries of `instrument_map.json`. It is
   already read if present. Without it the engine uses the family for the
   single-instrument families and a word match on labels inside `other`.

4. **Preview thresholds.** `Engine::detect` counts vocals, drums or bass as
   heard when the `scnet_small` stem is above -30 dB relative to the mix in
   an 11 s segment, with full energy confidence from -12 dB. These were
   chosen by judgement. If the evaluation set can give a fitted threshold
   (level of each estimated stem relative to the mix, for excerpts where the
   true stem is silent against where it is not), it is two constants in
   `engine.rs`.

Resolved since the first version of this file: the manifest shape (the
engine now parses `ml/dist/manifest.json` as written, section 7), peak
memory (the batch dimension is left symbolic, as BENCHMARKS.md recommends)
and the "best" overlap (kept at the default stride).
