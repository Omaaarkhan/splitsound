//! Instrument detection with the AudioSet tagger (`ml/CONTRACT.md`
//! section 6, ported from `ml/reference_tagger.py`).
//!
//! The graph takes 10 s of 32 kHz mono and returns 527 class
//! probabilities; the whole mel frontend is inside it. This module cuts the
//! windows, aggregates, and turns raw probabilities into the short list of
//! [`Detection`]s the app shows, using the curated `instrument_map.json`.

use std::collections::HashMap;

use ort::session::Session;
use ort::value::TensorRef;

use crate::error::{Error, Result};
use crate::runtime::run_error;

/// Samples per analysis window: 10 s at 32 kHz.
pub const WINDOW: usize = 320_000;
/// Samples between window starts: 5 s.
pub const HOP: usize = 160_000;
/// Number of AudioSet classes.
pub const NUM_CLASSES: usize = 527;

/// A stem the engine can produce; `StemId` in the UI contract.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum StemId {
    /// Lead and backing vocals.
    Vocals,
    /// Drums and percussion.
    Drums,
    /// Bass.
    Bass,
    /// Guitars.
    Guitar,
    /// Piano and keys the model treats as piano.
    Piano,
    /// Everything that is not a named stem.
    Other,
    /// Everything that is not vocals (two-stem mode only).
    Instrumental,
}

impl StemId {
    /// Canonical lane order, top to bottom (`STEM_ORDER` in the contract).
    pub const ORDER: [StemId; 7] = [
        StemId::Vocals,
        StemId::Drums,
        StemId::Bass,
        StemId::Guitar,
        StemId::Piano,
        StemId::Other,
        StemId::Instrumental,
    ];

    /// The lowercase id used in JSON and file names.
    pub fn as_str(self) -> &'static str {
        match self {
            StemId::Vocals => "vocals",
            StemId::Drums => "drums",
            StemId::Bass => "bass",
            StemId::Guitar => "guitar",
            StemId::Piano => "piano",
            StemId::Other => "other",
            StemId::Instrumental => "instrumental",
        }
    }

    /// Parses the lowercase id.
    pub fn parse(s: &str) -> Option<StemId> {
        StemId::ORDER.into_iter().find(|id| id.as_str() == s)
    }

    /// Capitalised name for labels: "Vocals".
    pub fn display_name(self) -> &'static str {
        match self {
            StemId::Vocals => "Vocals",
            StemId::Drums => "Drums",
            StemId::Bass => "Bass",
            StemId::Guitar => "Guitar",
            StemId::Piano => "Piano",
            StemId::Other => "Other",
            StemId::Instrumental => "Instrumental",
        }
    }
}

/// How finely to split; `SplitMode` in the UI contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    /// Vocals + instrumental.
    Vocals,
    /// Vocals, drums, bass, other.
    Band,
    /// Vocals, drums, bass, guitar, piano, other.
    All,
}

impl SplitMode {
    /// The stems this mode produces, in lane order (`STEMS_BY_MODE`).
    pub fn stems(self) -> &'static [StemId] {
        match self {
            SplitMode::Vocals => &[StemId::Vocals, StemId::Instrumental],
            SplitMode::Band => &[StemId::Vocals, StemId::Drums, StemId::Bass, StemId::Other],
            SplitMode::All => &[
                StemId::Vocals,
                StemId::Drums,
                StemId::Bass,
                StemId::Guitar,
                StemId::Piano,
                StemId::Other,
            ],
        }
    }
}

/// An instrument the tagger heard; `Detection` in the UI contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Detection {
    /// Short human name: "Electric guitar", "Singing", "Synthesizer".
    pub label: String,
    /// The stem this instrument ends up in.
    pub family: StemId,
    /// Highest probability over the analysis windows, `0..=1`.
    pub confidence: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct MapFile {
    threshold: f32,
    mix_threshold: f32,
    classes: HashMap<String, MapClass>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct MapClass {
    label: String,
    family: StemId,
    #[serde(default)]
    generic: bool,
    /// Optional per-class threshold override.
    #[serde(default)]
    threshold: Option<f32>,
    /// Optional index of the umbrella class this one belongs under.
    #[serde(default)]
    parent: Option<usize>,
}

#[derive(Debug, Clone)]
struct MappedClass {
    index: usize,
    label: String,
    family: StemId,
    generic: bool,
    threshold: Option<f32>,
    parent: Option<usize>,
}

/// The curated map from AudioSet classes to instruments and stem families.
#[derive(Debug, Clone)]
pub struct InstrumentMap {
    classes: Vec<MappedClass>,
    /// Threshold for a separated stem.
    pub threshold: f32,
    /// Threshold for the full mix, where instrument scores are much lower.
    pub mix_threshold: f32,
}

impl InstrumentMap {
    /// Parses `instrument_map.json`.
    pub fn from_json(json: &str) -> Result<Self> {
        let file: MapFile = serde_json::from_str(json)
            .map_err(|e| Error::internal(format!("instrument map: {e}")))?;
        let mut classes = Vec::with_capacity(file.classes.len());
        for (key, c) in file.classes {
            let index: usize = key
                .parse()
                .map_err(|_| Error::internal(format!("instrument map: bad class index {key:?}")))?;
            if index >= NUM_CLASSES {
                return Err(Error::internal(format!(
                    "instrument map: class index {index} out of range"
                )));
            }
            if c.family == StemId::Instrumental {
                return Err(Error::internal(
                    "instrument map: `instrumental` is not a family",
                ));
            }
            classes.push(MappedClass {
                index,
                label: c.label,
                family: c.family,
                generic: c.generic,
                threshold: c.threshold,
                parent: c.parent,
            });
        }
        classes.sort_by_key(|c| c.index);
        if !(0.0..=1.0).contains(&file.threshold) || !(0.0..=1.0).contains(&file.mix_threshold) {
            return Err(Error::internal("instrument map: threshold out of range"));
        }
        Ok(InstrumentMap {
            classes,
            threshold: file.threshold,
            mix_threshold: file.mix_threshold,
        })
    }

    /// The map compiled into the engine.
    pub fn builtin() -> Result<Self> {
        Self::from_json(include_str!("../models/instrument_map.json"))
    }

    /// Number of mapped classes.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// True when no class is mapped.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Turns per-class probabilities (maximum over windows) into
    /// detections, strongest first.
    ///
    /// * `threshold`: minimum probability (a class's own `threshold` in the
    ///   map overrides it).
    /// * `families`: when given, only classes of these families count. Use
    ///   it when tagging a separated stem: only the families that can live
    ///   in that stem (a bass guitar scores highly as "Guitar").
    ///
    /// Near-duplicates are collapsed: an umbrella class ("Guitar",
    /// "Singing", "Drum") is dropped when a more specific class under it
    /// was also detected ("Electric guitar", "Male vocals", "Snare drum"),
    /// and when two classes share a label only the stronger is kept.
    pub fn detect(
        &self,
        max_probs: &[f32],
        threshold: f32,
        families: Option<&[StemId]>,
    ) -> Vec<Detection> {
        let mut found: Vec<(&MappedClass, f32)> = self
            .classes
            .iter()
            .filter(|c| families.is_none_or(|f| f.contains(&c.family)))
            .filter_map(|c| {
                let p = max_probs.get(c.index).copied().unwrap_or(0.0);
                let limit = c.threshold.unwrap_or(threshold);
                (p.is_finite() && p >= limit).then_some((c, p))
            })
            .collect();

        // Drop umbrella classes that a more specific detection covers.
        let specific: Vec<&MappedClass> = found
            .iter()
            .map(|(c, _)| *c)
            .filter(|c| !c.generic)
            .collect();
        found.retain(|(c, _)| !c.generic || !specific.iter().any(|s| self.covers(c, s)));

        // Strongest first; ties by class index for a stable order.
        found.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.index.cmp(&b.0.index)));
        let mut out: Vec<Detection> = Vec::with_capacity(found.len());
        for (c, p) in found {
            if out
                .iter()
                .any(|d| d.label == c.label && d.family == c.family)
            {
                continue;
            }
            out.push(Detection {
                label: c.label.clone(),
                family: c.family,
                confidence: p.clamp(0.0, 1.0),
            });
        }
        out
    }

    /// Whether the umbrella class `generic` is made redundant by the
    /// specific class `specific`.
    fn covers(&self, generic: &MappedClass, specific: &MappedClass) -> bool {
        // 1. The map says so explicitly (walk up the parent chain).
        let mut parent = specific.parent;
        let mut hops = 0;
        while let Some(p) = parent {
            if p == generic.index {
                return true;
            }
            hops += 1;
            if hops > 8 {
                break;
            }
            parent = self
                .classes
                .iter()
                .find(|c| c.index == p)
                .and_then(|c| c.parent);
        }
        if specific.parent.is_some() {
            return false;
        }
        // 2. No parent information. The families with a single instrument
        //    (vocals, drums, bass, guitar, piano) are their own group: any
        //    specific class of the family replaces the family's umbrella.
        if generic.family != specific.family {
            return false;
        }
        if generic.family != StemId::Other {
            return true;
        }
        // 3. In the catch-all family, go by name: "Organ" is covered by
        //    "Hammond organ", "Bell" by "Church bell", but "Brass" is not
        //    covered by "Violin".
        let g = generic.label.to_lowercase();
        let s = specific.label.to_lowercase();
        s.split(|ch: char| !ch.is_alphanumeric())
            .any(|word| word == g)
    }

    /// Detections for the full mix (uses [`InstrumentMap::mix_threshold`]).
    pub fn detect_in_mix(&self, max_probs: &[f32]) -> Vec<Detection> {
        self.detect(max_probs, self.mix_threshold, None)
    }

    /// Detections inside a separated stem (uses
    /// [`InstrumentMap::threshold`] and only the families that can live in
    /// that stem given the stems that were split out).
    pub fn detect_in_stem(
        &self,
        max_probs: &[f32],
        stem: StemId,
        mode: SplitMode,
    ) -> Vec<Detection> {
        let families = families_in_stem(stem, mode);
        self.detect(max_probs, self.threshold, Some(&families))
    }
}

/// The instrument families that can be present in `stem` when the song was
/// split with `mode`: a named stem holds its own family; `other` and
/// `instrumental` hold every family that did not get a stem of its own.
pub fn families_in_stem(stem: StemId, mode: SplitMode) -> Vec<StemId> {
    const FAMILIES: [StemId; 6] = [
        StemId::Vocals,
        StemId::Drums,
        StemId::Bass,
        StemId::Guitar,
        StemId::Piano,
        StemId::Other,
    ];
    match stem {
        StemId::Other | StemId::Instrumental => FAMILIES
            .into_iter()
            .filter(|f| *f == StemId::Other || !mode.stems().contains(f))
            .collect(),
        named => vec![named],
    }
}

/// Suggests a split from what was heard in the mix.
///
/// * a guitar or piano was heard: [`SplitMode::All`], so they get stems of
///   their own;
/// * otherwise: [`SplitMode::Band`].
///
/// The two-stem mode is never suggested: it is a choice about what the
/// user wants to do (karaoke, an acapella), not about what is in the song.
/// With no detections at all (analysis unavailable) the answer is `Band`.
pub fn suggest_mode(detections: &[Detection]) -> SplitMode {
    if detections
        .iter()
        .any(|d| matches!(d.family, StemId::Guitar | StemId::Piano))
    {
        SplitMode::All
    } else {
        SplitMode::Band
    }
}

/// Start sample of every analysis window for `len` samples at 32 kHz: one
/// window if the audio fits in 10 s, otherwise one every 5 s until the last
/// reaches the end.
pub fn window_starts(len: usize) -> Vec<usize> {
    window_starts_with_hop(len, HOP)
}

/// [`window_starts`] with a custom hop (clamped to `1..=WINDOW`). A hop of
/// [`WINDOW`] gives back-to-back windows: half the work of the reference
/// schedule, used where a second opinion on every moment is not needed.
pub fn window_starts_with_hop(len: usize, hop: usize) -> Vec<usize> {
    let hop = hop.clamp(1, WINDOW);
    if len <= WINDOW {
        return vec![0];
    }
    let count = 1 + (len - WINDOW).div_ceil(hop);
    (0..count).map(|k| k * hop).collect()
}

/// Raw tagger output for one piece of audio.
#[derive(Debug, Clone, PartialEq)]
pub struct TagScores {
    /// Per class, the maximum probability over the windows.
    pub max: Vec<f32>,
    /// Per class, the mean probability over the windows.
    pub mean: Vec<f32>,
    /// Number of windows analysed.
    pub windows: usize,
}

/// The tagger: a session plus its window buffer.
pub struct Tagger {
    session: Session,
    window: Vec<f32>,
}

impl Tagger {
    /// Wraps a session of the tagger graph.
    pub fn new(session: Session) -> Self {
        Tagger {
            session,
            window: vec![0.0; WINDOW],
        }
    }

    /// Scores mono audio at 32 kHz.
    ///
    /// `on_window(done, total)` is called after each window; return an
    /// error to stop (cancellation). Empty audio is scored as one window of
    /// silence.
    pub fn score(
        &mut self,
        mono_32k: &[f32],
        on_window: &mut dyn FnMut(usize, usize) -> Result<()>,
    ) -> Result<TagScores> {
        self.score_with_hop(mono_32k, HOP, on_window)
    }

    /// [`Tagger::score`] with a custom hop between windows (see
    /// [`window_starts_with_hop`]).
    pub fn score_with_hop(
        &mut self,
        mono_32k: &[f32],
        hop: usize,
        on_window: &mut dyn FnMut(usize, usize) -> Result<()>,
    ) -> Result<TagScores> {
        let starts = window_starts_with_hop(mono_32k.len(), hop);
        let mut max = vec![0f32; NUM_CLASSES];
        let mut sum = vec![0f64; NUM_CLASSES];
        for (i, &start) in starts.iter().enumerate() {
            let available = WINDOW.min(mono_32k.len().saturating_sub(start));
            self.window[..available].copy_from_slice(&mono_32k[start..start + available]);
            self.window[available..].fill(0.0);
            let input = TensorRef::from_array_view(([1usize, WINDOW], &self.window[..]))
                .map_err(run_error)?;
            let outputs = self
                .session
                .run(ort::inputs!["waveform" => input])
                .map_err(run_error)?;
            let value = outputs
                .get("probs")
                .ok_or_else(|| Error::internal("tagger: graph has no `probs` output"))?;
            let (_, probs) = value.try_extract_tensor::<f32>().map_err(run_error)?;
            if probs.len() != NUM_CLASSES {
                return Err(Error::internal(format!(
                    "tagger: expected {NUM_CLASSES} classes, got {}",
                    probs.len()
                )));
            }
            for (c, p) in probs.iter().enumerate() {
                let p = if p.is_finite() { *p } else { 0.0 };
                if i == 0 || p > max[c] {
                    max[c] = p;
                }
                sum[c] += f64::from(p);
            }
            on_window(i + 1, starts.len())?;
        }
        let n = starts.len() as f64;
        Ok(TagScores {
            max,
            mean: sum.iter().map(|s| (s / n) as f32).collect(),
            windows: starts.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAP: &str = r#"{
        "schemaVersion": 1, "families": [], "threshold": 0.1, "mixThreshold": 0.05,
        "classes": {
            "27":  {"label": "Singing", "family": "vocals", "generic": true},
            "32":  {"label": "Male vocals", "family": "vocals"},
            "140": {"label": "Guitar", "family": "guitar", "generic": true},
            "141": {"label": "Electric guitar", "family": "guitar"},
            "142": {"label": "Bass guitar", "family": "bass"},
            "153": {"label": "Piano", "family": "piano"},
            "155": {"label": "Organ", "family": "other", "generic": true},
            "157": {"label": "Hammond organ", "family": "other"},
            "158": {"label": "Synthesizer", "family": "other"},
            "185": {"label": "Brass", "family": "other", "generic": true},
            "187": {"label": "Trumpet", "family": "other", "parent": 185},
            "191": {"label": "Violin", "family": "other"},
            "162": {"label": "Drum kit", "family": "drums"},
            "300": {"label": "Synthesizer", "family": "other"},
            "400": {"label": "Loud thing", "family": "other", "threshold": 0.9}
        }
    }"#;

    fn probs(set: &[(usize, f32)]) -> Vec<f32> {
        let mut p = vec![0f32; NUM_CLASSES];
        for (i, v) in set {
            p[*i] = *v;
        }
        p
    }

    fn labels(d: &[Detection]) -> Vec<&str> {
        d.iter().map(|x| x.label.as_str()).collect()
    }

    #[test]
    fn window_schedule_matches_the_reference() {
        assert_eq!(window_starts(0), vec![0]);
        assert_eq!(window_starts(1), vec![0]);
        assert_eq!(window_starts(WINDOW), vec![0]);
        assert_eq!(window_starts(WINDOW + 1), vec![0, HOP]);
        assert_eq!(window_starts(WINDOW + HOP), vec![0, HOP]);
        assert_eq!(window_starts(WINDOW + HOP + 1), vec![0, HOP, 2 * HOP]);
        assert_eq!(
            window_starts_with_hop(3 * WINDOW, WINDOW),
            vec![0, WINDOW, 2 * WINDOW]
        );
        assert_eq!(window_starts_with_hop(3 * WINDOW + 1, WINDOW).len(), 4);
        assert_eq!(window_starts_with_hop(WINDOW, 0), vec![0]);
        // 3 minutes 30 s: last window must reach the end.
        let len = 210 * 32_000;
        let s = window_starts(len);
        assert_eq!(s.len(), 41);
        assert!(s.last().unwrap() + WINDOW >= len);
        assert!(s[s.len() - 2] + WINDOW < len);
    }

    #[test]
    fn detections_are_sorted_thresholded_and_collapsed() {
        let map = InstrumentMap::from_json(MAP).unwrap();
        let p = probs(&[
            (27, 0.5),   // Singing (umbrella) ...
            (32, 0.3),   // ... covered by Male vocals
            (140, 0.6),  // Guitar (umbrella) ...
            (141, 0.2),  // ... covered by Electric guitar
            (153, 0.07), // Piano: below the stem threshold, above the mix one
            (158, 0.7),  // Synthesizer
            (300, 0.9),  // same label, stronger: this one is kept
            (400, 0.5),  // has its own threshold of 0.9
            (5, 0.99),   // not in the map: ignored
        ]);
        let d = map.detect(&p, map.threshold, None);
        assert_eq!(
            labels(&d),
            vec!["Synthesizer", "Male vocals", "Electric guitar"]
        );
        assert_eq!(d[0].confidence, 0.9);
        assert_eq!(d[1].family, StemId::Vocals);
        assert!(d.windows(2).all(|w| w[0].confidence >= w[1].confidence));
        // On the mix the lower threshold lets the piano in.
        assert!(labels(&map.detect_in_mix(&p)).contains(&"Piano"));
    }

    #[test]
    fn umbrella_classes_survive_without_a_specific_one() {
        let map = InstrumentMap::from_json(MAP).unwrap();
        // Guitar alone stays.
        assert_eq!(
            labels(&map.detect(&probs(&[(140, 0.4)]), 0.1, None)),
            vec!["Guitar"]
        );
        // "Organ" is covered by "Hammond organ" (by name)...
        let d = map.detect(&probs(&[(155, 0.5), (157, 0.3)]), 0.1, None);
        assert_eq!(labels(&d), vec!["Hammond organ"]);
        // ...but "Brass" is not covered by "Violin", only by its child.
        let d = map.detect(&probs(&[(185, 0.5), (191, 0.3)]), 0.1, None);
        assert_eq!(labels(&d), vec!["Brass", "Violin"]);
        let d = map.detect(&probs(&[(185, 0.5), (187, 0.3)]), 0.1, None);
        assert_eq!(labels(&d), vec!["Trumpet"]);
        // A bass guitar does not hide the guitar umbrella (other family).
        let d = map.detect(&probs(&[(140, 0.5), (142, 0.3)]), 0.1, None);
        assert_eq!(labels(&d), vec!["Guitar", "Bass guitar"]);
    }

    #[test]
    fn stem_restriction_by_family() {
        let map = InstrumentMap::from_json(MAP).unwrap();
        let p = probs(&[(140, 0.5), (142, 0.6), (158, 0.4), (162, 0.3), (153, 0.2)]);
        // A bass stem: the high "Guitar" score must not leak in.
        assert_eq!(
            labels(&map.detect_in_stem(&p, StemId::Bass, SplitMode::Band)),
            vec!["Bass guitar"]
        );
        // "other" in a 4-stem split still holds guitar and piano...
        assert_eq!(
            labels(&map.detect_in_stem(&p, StemId::Other, SplitMode::Band)),
            vec!["Guitar", "Synthesizer", "Piano"]
        );
        // ...but not in a 6-stem split, where they have their own stems.
        assert_eq!(
            labels(&map.detect_in_stem(&p, StemId::Other, SplitMode::All)),
            vec!["Synthesizer"]
        );
        // "instrumental" holds everything but vocals.
        assert_eq!(
            families_in_stem(StemId::Instrumental, SplitMode::Vocals),
            vec![
                StemId::Drums,
                StemId::Bass,
                StemId::Guitar,
                StemId::Piano,
                StemId::Other
            ]
        );
    }

    #[test]
    fn suggestions() {
        let det = |family| Detection {
            label: "x".into(),
            family,
            confidence: 0.5,
        };
        assert_eq!(suggest_mode(&[]), SplitMode::Band);
        assert_eq!(
            suggest_mode(&[det(StemId::Vocals), det(StemId::Drums)]),
            SplitMode::Band
        );
        assert_eq!(suggest_mode(&[det(StemId::Other)]), SplitMode::Band);
        assert_eq!(
            suggest_mode(&[det(StemId::Vocals), det(StemId::Guitar)]),
            SplitMode::All
        );
        assert_eq!(suggest_mode(&[det(StemId::Piano)]), SplitMode::All);
    }

    #[test]
    fn bad_maps_are_rejected() {
        assert!(InstrumentMap::from_json("{}").is_err());
        assert!(InstrumentMap::from_json(r#"{"threshold":0.1,"mixThreshold":0.05,"classes":{"900":{"label":"x","family":"other"}}}"#).is_err());
        assert!(InstrumentMap::from_json(r#"{"threshold":0.1,"mixThreshold":0.05,"classes":{"abc":{"label":"x","family":"other"}}}"#).is_err());
        assert!(InstrumentMap::from_json(r#"{"threshold":0.1,"mixThreshold":0.05,"classes":{"1":{"label":"x","family":"kazoo"}}}"#).is_err());
        assert!(
            InstrumentMap::from_json(r#"{"threshold":7,"mixThreshold":0.05,"classes":{}}"#)
                .is_err()
        );
    }

    #[test]
    fn stem_ids_and_modes_match_the_contract() {
        let ids: Vec<&str> = StemId::ORDER.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            ids,
            [
                "vocals",
                "drums",
                "bass",
                "guitar",
                "piano",
                "other",
                "instrumental"
            ]
        );
        assert_eq!(
            serde_json::to_string(&StemId::Instrumental).unwrap(),
            "\"instrumental\""
        );
        assert_eq!(StemId::parse("piano"), Some(StemId::Piano));
        assert_eq!(StemId::parse("kazoo"), None);
        assert_eq!(SplitMode::Vocals.stems().len(), 2);
        assert_eq!(SplitMode::Band.stems().len(), 4);
        assert_eq!(SplitMode::All.stems().len(), 6);
        assert_eq!(serde_json::to_string(&SplitMode::All).unwrap(), "\"all\"");
    }
}
