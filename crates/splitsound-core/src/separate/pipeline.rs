//! What each split mode and quality runs: which models, on what, and which
//! stems are taken from each.
//!
//! | mode   | quality | passes |
//! |--------|---------|--------|
//! | vocals | fast    | `scnet_small` on the mix: vocals |
//! | vocals | best    | `scnet_large` on the mix: vocals |
//! | vocals | studio  | `melband_roformer_vocals` on the mix: vocals |
//! | band   | fast    | `scnet_small` on the mix: vocals, drums, bass |
//! | band   | best    | `scnet_large` on the mix: vocals, drums, bass |
//! | band   | studio  | `melband_roformer_vocals`: vocals; then `scnet_large` on the residual: drums, bass |
//! | all    | fast    | `htdemucs_6s` on the mix: vocals, drums, bass, guitar, piano |
//! | all    | best    | `scnet_large`: vocals, drums, bass; then `htdemucs_6s` on what is left: guitar, piano |
//! | all    | studio  | `melband_roformer_vocals`: vocals; `scnet_large` on the residual: drums, bass; `htdemucs_6s` on what is left: guitar, piano |
//!
//! Each pass uses the model's own stride from the manifest (25 % overlap for
//! SCNet and HTDemucs; 50 % overlap for Mel-Band RoFormer).
//! More overlap was measured to be no better for the extra work
//! (`ml/BENCHMARKS.md`).
//!
//! "Other" (and "instrumental" in the two-stem mode) is never taken from a
//! model: it is the mix minus every named stem, so the stems always sum
//! back to the mix exactly.
//!
//! `htdemucs_6s` is a restricted model (see the manifest); every plan that
//! uses it says so, so the app can ask for consent first.

use crate::tagger::{SplitMode, StemId};

/// `fast`, `best` or `studio`; `Quality` in the UI contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    /// The small model: one quick pass.
    Fast,
    /// The large model; several times slower.
    Best,
    /// Mel-Band RoFormer vocal isolation before remaining separation; slowest
    /// and largest download, with the least vocal bleed.
    Studio,
}

/// What a split should produce; `SplitOptions` in the UI contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitOptions {
    /// How finely to split.
    pub mode: SplitMode,
    /// Speed against quality.
    pub quality: Quality,
}

/// What a pass listens to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassInput {
    /// The original mix.
    Mix,
    /// The mix minus every stem produced by earlier passes.
    Residual,
}

/// One model run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pass {
    /// Model id in the manifest.
    pub model_id: &'static str,
    /// The stems taken from this pass.
    pub stems: &'static [StemId],
    /// What the model listens to.
    pub input: PassInput,
}

impl Pass {
    /// User-facing label: "Separating vocals, drums and bass".
    pub fn label(&self) -> String {
        let names: Vec<&str> = self.stems.iter().map(|s| s.as_str()).collect();
        let list = match names.as_slice() {
            [] => String::new(),
            [one] => (*one).to_string(),
            [init @ .., last] => format!("{} and {last}", init.join(", ")),
        };
        format!("Separating {list}")
    }
}

/// Where guitar and piano come from in `all` / `best`.
///
/// The choice could not be measured (see the crate README, "Guitar and
/// piano"); [`GuitarPianoSource::CHOSEN`] is the default the brief names
/// for that case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuitarPianoSource {
    /// `htdemucs_6s` listens to the original mix.
    Mix,
    /// `htdemucs_6s` listens to the mix minus vocals, drums and bass.
    Residual,
}

impl GuitarPianoSource {
    /// The strategy in use.
    pub const CHOSEN: GuitarPianoSource = GuitarPianoSource::Residual;
}

const VDB: &[StemId] = &[StemId::Vocals, StemId::Drums, StemId::Bass];
const DB: &[StemId] = &[StemId::Drums, StemId::Bass];
const VOCALS: &[StemId] = &[StemId::Vocals];
const GP: &[StemId] = &[StemId::Guitar, StemId::Piano];
const VDBGP: &[StemId] = &[
    StemId::Vocals,
    StemId::Drums,
    StemId::Bass,
    StemId::Guitar,
    StemId::Piano,
];

/// The model passes for `options`, in the order they run.
pub fn passes(options: SplitOptions) -> Vec<Pass> {
    passes_with(options, GuitarPianoSource::CHOSEN)
}

/// [`passes`] with an explicit guitar/piano strategy (for measuring).
pub fn passes_with(options: SplitOptions, guitar_piano: GuitarPianoSource) -> Vec<Pass> {
    let one = |model_id, stems| Pass {
        model_id,
        stems,
        input: PassInput::Mix,
    };
    let residual = |model_id, stems| Pass {
        model_id,
        stems,
        input: PassInput::Residual,
    };
    let gp_pass = Pass {
        model_id: "htdemucs_6s",
        stems: GP,
        input: match guitar_piano {
            GuitarPianoSource::Mix => PassInput::Mix,
            GuitarPianoSource::Residual => PassInput::Residual,
        },
    };
    match (options.mode, options.quality) {
        (SplitMode::Vocals, Quality::Fast) => vec![one("scnet_small", VOCALS)],
        (SplitMode::Vocals, Quality::Best) => vec![one("scnet_large", VOCALS)],
        (SplitMode::Vocals, Quality::Studio) => vec![one("melband_roformer_vocals", VOCALS)],
        (SplitMode::Band, Quality::Fast) => vec![one("scnet_small", VDB)],
        (SplitMode::Band, Quality::Best) => vec![one("scnet_large", VDB)],
        (SplitMode::Band, Quality::Studio) => vec![
            one("melband_roformer_vocals", VOCALS),
            residual("scnet_large", DB),
        ],
        (SplitMode::All, Quality::Fast) => vec![one("htdemucs_6s", VDBGP)],
        (SplitMode::All, Quality::Best) => vec![one("scnet_large", VDB), gp_pass],
        (SplitMode::All, Quality::Studio) => vec![
            one("melband_roformer_vocals", VOCALS),
            residual("scnet_large", DB),
            gp_pass,
        ],
    }
}

/// The stem that holds "everything else" in `mode`.
pub fn remainder_stem(mode: SplitMode) -> StemId {
    match mode {
        SplitMode::Vocals => StemId::Instrumental,
        SplitMode::Band | SplitMode::All => StemId::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_plan_covers_its_mode_exactly() {
        for mode in [SplitMode::Vocals, SplitMode::Band, SplitMode::All] {
            for quality in [Quality::Fast, Quality::Best, Quality::Studio] {
                let plan = passes(SplitOptions { mode, quality });
                assert!(!plan.is_empty());
                let mut produced: Vec<StemId> =
                    plan.iter().flat_map(|p| p.stems.iter().copied()).collect();
                produced.push(remainder_stem(mode));
                produced.sort();
                let mut wanted = mode.stems().to_vec();
                wanted.sort();
                assert_eq!(produced, wanted, "{mode:?} {quality:?}");
                // The first pass always listens to the mix.
                assert_eq!(plan[0].input, PassInput::Mix);
                // Nothing models "other": it is always the remainder.
                assert!(plan.iter().all(|p| !p.stems.contains(&StemId::Other)));
            }
        }
    }

    #[test]
    fn plans_match_the_design() {
        let p = |mode, quality| passes(SplitOptions { mode, quality });
        assert_eq!(
            p(SplitMode::Vocals, Quality::Fast)[0].model_id,
            "scnet_small"
        );
        assert_eq!(
            p(SplitMode::Vocals, Quality::Best)[0].model_id,
            "scnet_large"
        );
        assert_eq!(p(SplitMode::Band, Quality::Fast)[0].model_id, "scnet_small");
        assert_eq!(p(SplitMode::Band, Quality::Best)[0].model_id, "scnet_large");
        assert_eq!(p(SplitMode::All, Quality::Fast)[0].model_id, "htdemucs_6s");
        let best = p(SplitMode::All, Quality::Best);
        assert_eq!(best.len(), 2);
        assert_eq!(best[1].model_id, "htdemucs_6s");
        assert_eq!(best[1].stems, GP);
        assert_eq!(best[1].input, PassInput::Residual);
        assert_eq!(
            passes_with(
                SplitOptions {
                    mode: SplitMode::All,
                    quality: Quality::Best
                },
                GuitarPianoSource::Mix
            )[1]
            .input,
            PassInput::Mix
        );
        // Only the 6-stem plans touch the restricted model.
        for mode in [SplitMode::Vocals, SplitMode::Band] {
            for q in [Quality::Fast, Quality::Best, Quality::Studio] {
                assert!(p(mode, q).iter().all(|x| x.model_id != "htdemucs_6s"));
            }
        }
        let studio_band = p(SplitMode::Band, Quality::Studio);
        assert_eq!(studio_band.len(), 2);
        assert_eq!(studio_band[0].model_id, "melband_roformer_vocals");
        assert_eq!(studio_band[1].model_id, "scnet_large");
        assert_eq!(studio_band[1].input, PassInput::Residual);

        let studio_all = p(SplitMode::All, Quality::Studio);
        assert_eq!(studio_all.len(), 3);
        assert_eq!(studio_all[0].model_id, "melband_roformer_vocals");
        assert_eq!(studio_all[1].model_id, "scnet_large");
        assert_eq!(studio_all[1].input, PassInput::Residual);
        assert_eq!(studio_all[2].model_id, "htdemucs_6s");
        assert_eq!(studio_all[2].input, PassInput::Residual);
    }

    #[test]
    fn labels_and_serde() {
        let best = passes(SplitOptions {
            mode: SplitMode::All,
            quality: Quality::Best,
        });
        assert_eq!(best[0].label(), "Separating vocals, drums and bass");
        assert_eq!(best[1].label(), "Separating guitar and piano");
        let fast = passes(SplitOptions {
            mode: SplitMode::Vocals,
            quality: Quality::Fast,
        });
        assert_eq!(fast[0].label(), "Separating vocals");
        assert_eq!(serde_json::to_string(&Quality::Best).unwrap(), "\"best\"");
        assert_eq!(
            serde_json::to_string(&Quality::Studio).unwrap(),
            "\"studio\""
        );
        let o: SplitOptions = serde_json::from_str(r#"{"mode":"all","quality":"fast"}"#).unwrap();
        assert_eq!(o.mode, SplitMode::All);
    }
}
