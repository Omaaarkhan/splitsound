//! The engine: one object the app shell (and the CLI) talks to.
//!
//! An [`Engine`] is cheap to create and to keep alive: it owns the model
//! store and the runtime settings, and loads ONNX Runtime and model
//! sessions lazily, on first use. Every method is blocking and
//! thread-safe; long ones take a [`Progress`] for reporting and
//! cancellation. See the crate README for a tour.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Instant;

use crate::analyze::{StemStats, analyze_stem, analyze_stereo};
use crate::audio::{SAMPLE_RATE, Stereo, TAGGER_SAMPLE_RATE};
use crate::decode::{Artwork, DecodeLimits, SourceInfo, decode_file};
use crate::error::{Error, ErrorCode, Result};
use crate::export::{ExportFormat, export_audio, mixdown};
use crate::peaks::{Peaks, compute_peaks};
use crate::progress::{PlannedStep, Progress, Stage, Tracker, check_cancelled};
use crate::runtime::{Acceleration, Provider, Runtime, RuntimeConfig};
use crate::separate::chunk::{SegmentModel, count_segments, separate_track};
use crate::separate::models::open_segment_model;
use crate::separate::pipeline::{Pass, PassInput, SplitOptions, passes, remainder_stem};
use crate::store::{DownloadObserver, ModelSource, ModelStatus, ModelStore};
use crate::tagger::{Detection, InstrumentMap, SplitMode, StemId, Tagger, suggest_mode};
use crate::wav::{MappedWav, write_wav_f32};

/// Id of the tagger model in the manifest.
pub const TAGGER_MODEL: &str = "efficientat_mn10_as";

/// How to build an [`Engine`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Folder models are installed in (the app's data folder). Created if
    /// missing.
    pub models_dir: PathBuf,
    /// Absolute path of the ONNX Runtime library (or the folder holding
    /// it). The app passes the copy inside its bundle. `None` falls back to
    /// `ORT_DYLIB_PATH`, then `vendor/onnxruntime/<host>/` (tests, CLI).
    pub onnxruntime_path: Option<PathBuf>,
    /// `Settings.acceleration`.
    pub acceleration: Acceleration,
    /// `Settings.threads`; 0 picks automatically.
    pub threads: usize,
    /// Where models are downloaded from. Leave at the default.
    pub model_source: ModelSource,
    /// Optional folder for the GPU provider's compiled-model cache.
    pub cache_dir: Option<PathBuf>,
}

impl EngineConfig {
    /// Defaults: CPU/auto acceleration, automatic threads, the project's
    /// model release.
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        EngineConfig {
            models_dir: models_dir.into(),
            onnxruntime_path: None,
            acceleration: Acceleration::Auto,
            threads: 0,
            model_source: ModelSource::default(),
            cache_dir: None,
        }
    }
}

/// A decoded file ready to become a session.
#[derive(Debug, Clone)]
pub struct ImportedTrack {
    /// The mix: stereo, 44.1 kHz.
    pub audio: Stereo,
    /// Facts about the source file.
    pub info: SourceInfo,
    /// Cover art, if any.
    pub artwork: Option<Artwork>,
}

/// What the preview heard.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionReport {
    /// Instruments heard, strongest first.
    pub detections: Vec<Detection>,
    /// The split the engine suggests: `all` when a guitar or piano was
    /// heard in what is left after vocals, drums and bass, else `band`.
    pub suggested_mode: SplitMode,
    /// Whether vocals were heard. Worth showing ("this sounds
    /// instrumental"); it does not change the suggestion.
    pub vocals_heard: bool,
    /// Whether drums were heard.
    pub drums_heard: bool,
    /// Whether bass was heard.
    pub bass_heard: bool,
    /// How many 11 second segments were analysed (0 for a silent track).
    pub segments: usize,
}

/// Outcome of [`Engine::detect`]; `AnalysisStatus` in the UI contract is
/// `done` for [`DetectOutcome::Done`] and `unavailable` for
/// [`DetectOutcome::Unavailable`].
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum DetectOutcome {
    /// The preview ran.
    Done(DetectionReport),
    /// A model the preview needs is not installed. Nothing was analysed;
    /// this is not an error. Fetch `missing` and call again.
    Unavailable {
        /// Ids of the missing models.
        missing: Vec<String>,
    },
}

/// What a split needs before it can start; feeds `SplitEstimate` and the
/// consent prompt for restricted models.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitPlan {
    /// The options this plan is for.
    pub options: SplitOptions,
    /// Models the split uses, in the order they run, with install state,
    /// sizes, licence and (for restricted ones) terms.
    pub models: Vec<ModelStatus>,
    /// Bytes still to download. 0 when everything is installed.
    pub download_bytes: u64,
    /// Ids of models that are restricted **and not installed yet**: the
    /// split must be started with `accept_restricted` (after showing their
    /// `terms`) or it fails with `terms_required`.
    pub needs_consent: Vec<String>,
    /// User-facing labels of the separation steps, in order.
    pub steps: Vec<String>,
}

/// A request to split and store the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitRequest {
    /// Mode and quality.
    pub options: SplitOptions,
    /// Download missing models as the first stage instead of failing with
    /// `model_missing`.
    pub download_missing: bool,
    /// The user agreed to the terms of the restricted models this split
    /// needs. Without it a missing restricted model is `terms_required`.
    pub accept_restricted: bool,
    /// Run the tagger on every stem that has something in it, to name what
    /// is inside (`StemResult::contains`). Skipped quietly if the tagger is
    /// not installed.
    pub tag_stems: bool,
}

impl SplitRequest {
    /// A request with downloads off, no consent, and stem tagging on.
    pub fn new(options: SplitOptions) -> Self {
        SplitRequest {
            options,
            download_missing: false,
            accept_restricted: false,
            tag_stems: true,
        }
    }
}

/// Stems in memory.
#[derive(Debug, Clone)]
pub struct Split {
    /// One entry per stem of the mode, in lane order. They sum to the mix.
    pub stems: Vec<(StemId, Stereo)>,
    /// Ids of the models that ran, in order.
    pub model_ids: Vec<String>,
}

/// One stored stem; the engine half of `StemInfo` in the UI contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StemResult {
    /// Which stem.
    pub id: StemId,
    /// The working file: 32-bit float WAV, 44.1 kHz stereo.
    pub path: PathBuf,
    /// Level, activity and presence.
    #[serde(flatten)]
    pub stats: StemStats,
    /// Instruments heard inside this stem, among the families that can be
    /// in it. Most interesting for "other".
    pub contains: Vec<Detection>,
}

/// The result of [`Engine::split_to_dir`].
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitResult {
    /// The stems, in lane order.
    pub stems: Vec<StemResult>,
    /// Ids of the models that ran.
    pub model_ids: Vec<String>,
    /// Wall-clock seconds, downloads included.
    pub elapsed_sec: f64,
}

/// GPU path of this machine; `AppInfo.gpu` in the UI contract.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct GpuInfo {
    /// Whether a GPU execution path exists.
    pub available: bool,
    /// Its display name.
    pub name: Option<String>,
}

struct LoadedSeparator {
    id: String,
    model: Box<dyn SegmentModel + Send>,
}

/// The engine. See the module docs.
pub struct Engine {
    store: ModelStore,
    runtime_config: Mutex<RuntimeConfig>,
    instrument_map: InstrumentMap,
    /// The most recently used separator. Holding this lock is also what
    /// makes "one split at a time" true.
    separator: Mutex<Option<LoadedSeparator>>,
    tagger: Mutex<Option<Tagger>>,
}

/// Compute seconds per second of audio on an Apple M4 with four threads
/// (segment times from `ml/BENCHMARKS.md` divided by the stride). Only used
/// for the estimate and for weighting progress.
fn cost_per_audio_sec(model_id: &str) -> f64 {
    match model_id {
        "scnet_small" => 0.145,
        "scnet_large" => 0.54,
        "htdemucs_6s" => 0.19,
        _ => 0.3,
    }
}

fn pass_cost(pass: &Pass, audio_sec: f64) -> f64 {
    audio_sec * cost_per_audio_sec(pass.model_id)
}

/// The model the preview separation uses.
pub const PREVIEW_MODEL: &str = "scnet_small";
/// Most segments the preview looks at.
const PREVIEW_SEGMENTS: usize = 3;
/// A candidate segment quieter than this is not worth analysing.
const PREVIEW_MIN_RMS_DB: f32 = -50.0;
/// A stem this far below the mix (energy, in dB) or further counts as not
/// heard: separation models leave bleed around this level for sources that
/// are absent.
const HEARD_FLOOR_DB: f32 = -30.0;
/// At this level relative to the mix, or above, the energy evidence alone
/// is full confidence.
const HEARD_FULL_DB: f32 = -12.0;

fn energy(audio: &Stereo) -> f64 {
    audio
        .left
        .iter()
        .chain(&audio.right)
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum()
}

/// Where the preview listens: up to three segments of `n` samples around
/// 25 %, 50 % and 75 % of the track (fewer for short tracks), each moved by
/// up to one segment either way to the loudest spot nearby, skipping spots
/// that are near-silent and spots that overlap one already chosen.
pub(crate) fn preview_starts(mix: &Stereo, n: usize) -> Vec<usize> {
    let len = mix.len();
    if len == 0 || n == 0 {
        return Vec::new();
    }
    // Loudness on every 16th sample is plenty for choosing a spot.
    let rms_db = |start: usize, end: usize| {
        let mut sum = 0f64;
        let mut count = 0usize;
        for i in (start..end).step_by(16) {
            let m = 0.5 * (f64::from(mix.left[i]) + f64::from(mix.right[i]));
            sum += m * m;
            count += 1;
        }
        crate::audio::amplitude_to_db((sum / count.max(1) as f64).sqrt() as f32)
    };
    if len <= n {
        return if rms_db(0, len) > PREVIEW_MIN_RMS_DB {
            vec![0]
        } else {
            Vec::new()
        };
    }
    let fractions: &[f64] = match (len / n).min(PREVIEW_SEGMENTS) {
        0 | 1 => &[0.5],
        2 => &[0.33, 0.67],
        _ => &[0.25, 0.5, 0.75],
    };
    let last = len - n;
    let mut chosen: Vec<usize> = Vec::new();
    for f in fractions {
        let centre = (*f * len as f64) as isize - (n / 2) as isize;
        let mut best: Option<(f32, usize)> = None;
        for k in -2isize..=2 {
            let start = (centre + k * (n / 2) as isize).clamp(0, last as isize) as usize;
            if chosen.iter().any(|c| c.abs_diff(start) < n / 2) {
                continue;
            }
            let level = rms_db(start, start + n);
            // Prefer the planned spot unless another is clearly louder.
            let score = level - 1.5 * k.unsigned_abs() as f32;
            if level > PREVIEW_MIN_RMS_DB && best.is_none_or(|(b, _)| score > b) {
                best = Some((score, start));
            }
        }
        if let Some((_, start)) = best {
            chosen.push(start);
        }
    }
    chosen.sort_unstable();
    chosen
}

/// Confidence in `0..=1` from a stem's level relative to the mix.
pub(crate) fn energy_confidence(ratio_db: f32) -> f32 {
    ((ratio_db - HEARD_FLOOR_DB) / (HEARD_FULL_DB - HEARD_FLOOR_DB)).clamp(0.0, 1.0)
}

/// Builds the report from the preview's evidence: for each of vocals,
/// drums, bass and other, the best level relative to the mix over the
/// segments and the tagger's per-class maxima on that stem (`None` when
/// the stem was never loud enough to tag).
pub(crate) fn report_from_evidence(
    map: &InstrumentMap,
    evidence: &[(StemId, f32, Option<Vec<f32>>)],
    segments: usize,
) -> DetectionReport {
    let mut detections: Vec<Detection> = Vec::new();
    let mut heard = [false; 3];
    for (stem, ratio_db, scores) in evidence {
        let tagged = scores
            .as_ref()
            .map(|s| map.detect_in_stem(s, *stem, SplitMode::Band))
            .unwrap_or_default();
        if *stem == StemId::Other {
            detections.extend(tagged);
            continue;
        }
        // Vocals, drums and bass: presence is decided by energy alone.
        let from_energy = energy_confidence(*ratio_db);
        if from_energy <= 0.0 {
            continue;
        }
        match stem {
            StemId::Vocals => heard[0] = true,
            StemId::Drums => heard[1] = true,
            StemId::Bass => heard[2] = true,
            _ => {}
        }
        if tagged.is_empty() {
            // Heard, but the tagger could not name it: report the family.
            detections.push(Detection {
                label: stem.display_name().to_string(),
                family: *stem,
                confidence: from_energy,
            });
        } else {
            for (i, mut d) in tagged.into_iter().enumerate() {
                if i == 0 {
                    // Either kind of evidence is enough on its own; both
                    // together are stronger than each.
                    d.confidence = 1.0 - (1.0 - d.confidence) * (1.0 - from_energy);
                }
                detections.push(d);
            }
        }
    }
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    DetectionReport {
        suggested_mode: suggest_mode(&detections),
        detections,
        vocals_heard: heard[0],
        drums_heard: heard[1],
        bass_heard: heard[2],
        segments,
    }
}

impl Engine {
    /// Creates an engine. Nothing heavy happens here: ONNX Runtime and the
    /// models are loaded on first use.
    pub fn new(config: EngineConfig) -> Result<Engine> {
        let store = ModelStore::with_source(
            &config.models_dir,
            crate::manifest::Manifest::builtin().clone(),
            config.model_source.clone(),
        )?;
        Ok(Engine {
            store,
            runtime_config: Mutex::new(RuntimeConfig {
                library_path: config.onnxruntime_path,
                acceleration: config.acceleration,
                threads: config.threads,
                cache_dir: config.cache_dir,
            }),
            instrument_map: InstrumentMap::builtin()?,
            separator: Mutex::new(None),
            tagger: Mutex::new(None),
        })
    }

    /// The model store: list, install, remove, verify and import models.
    pub fn models(&self) -> &ModelStore {
        &self.store
    }

    /// The curated instrument map the detections come from.
    pub fn instrument_map(&self) -> &InstrumentMap {
        &self.instrument_map
    }

    fn runtime(&self) -> Result<Arc<Runtime>> {
        let config = self
            .runtime_config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        Runtime::load(config)
    }

    /// Applies new `acceleration` / `threads` settings. Loaded models are
    /// dropped so the next use picks the settings up.
    pub fn configure(&self, acceleration: Acceleration, threads: usize) {
        {
            let mut cfg = self
                .runtime_config
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cfg.acceleration = acceleration;
            cfg.threads = threads;
        }
        self.unload();
    }

    /// Frees the loaded model sessions (several hundred MB). They are
    /// loaded again on next use. Does nothing while a split is running.
    pub fn unload(&self) {
        if let Ok(mut s) = self.separator.try_lock() {
            *s = None;
        }
        if let Ok(mut t) = self.tagger.try_lock() {
            *t = None;
        }
    }

    /// The GPU path on this machine. Loads ONNX Runtime to find out; if
    /// that fails the answer is "none".
    pub fn gpu_info(&self) -> GpuInfo {
        let provider = self.runtime().ok().and_then(|rt| rt.gpu_provider());
        GpuInfo {
            available: provider.is_some(),
            name: provider.map(|p| p.display_name().to_string()),
        }
    }

    /// Threads inference will use with the current settings.
    pub fn threads(&self) -> usize {
        let cfg = self
            .runtime_config
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cfg.threads == 0 {
            crate::runtime::auto_threads()
        } else {
            cfg.threads
        }
    }

    /// Decodes a file to the stereo 44.1 kHz mix plus its tags and cover.
    ///
    /// # Errors
    /// `file_not_found`, `unsupported_format`, `decode_failed`, `cancelled`.
    pub fn import(&self, path: &Path, progress: &dyn Progress) -> Result<ImportedTrack> {
        let decoded = decode_file(path, SAMPLE_RATE, &DecodeLimits::default(), progress)?;
        Ok(ImportedTrack {
            audio: decoded.audio,
            info: decoded.info,
            artwork: decoded.artwork,
        })
    }

    fn with_tagger<T>(&self, work: impl FnOnce(&mut Tagger) -> Result<T>) -> Result<T> {
        let mut slot = self.tagger.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let verified = self.store.load(TAGGER_MODEL)?;
            let (session, _) = self
                .runtime()?
                .session_from_verified(&verified.graph, None)?;
            *slot = Some(Tagger::new(session));
        }
        match slot.as_mut() {
            Some(t) => work(t),
            None => Err(Error::internal("tagger slot empty")),
        }
    }

    /// Tagger maxima for `audio`, with `hop` samples (at 32 kHz) between
    /// windows.
    fn tag_scores(&self, audio: &Stereo, hop: usize, progress: &dyn Progress) -> Result<Vec<f32>> {
        let mono = audio.to_mono();
        let mut resampled = crate::resample::resample(&[mono], SAMPLE_RATE, TAGGER_SAMPLE_RATE)?;
        let mono_32k = resampled.pop().unwrap_or_default();
        self.with_tagger(|tagger| {
            let scores =
                tagger.score_with_hop(&mono_32k, hop, &mut |_, _| check_cancelled(progress))?;
            Ok(scores.max)
        })
    }

    /// The scan run after import: says which instruments are in the song
    /// and suggests a split, in a few seconds, without splitting it.
    ///
    /// The tagger is unreliable on a full mix (it hears "Music" and little
    /// else: 7 % recall for drums), so this does a **preview separation**
    /// instead. Up to three 11 second segments spread through the track
    /// are run through `scnet_small` once each. Vocals, drums and bass are
    /// then judged by how loud each separated stem is relative to the mix
    /// (heard above -30 dB, full confidence from -12 dB), and every stem
    /// loud enough is given to the tagger, restricted to the instrument
    /// families that stem can hold; that is what names guitars, pianos,
    /// synthesizers, strings and so on in "other". Where the tagger also
    /// names the vocals, drums or bass, the two confidences are combined.
    ///
    /// Needs `scnet_small` and the tagger. If either is not installed the
    /// result is [`DetectOutcome::Unavailable`], not an error. There is no
    /// mix-only fallback: its answers were not good enough to show.
    ///
    /// # Errors
    /// `busy` (a split is running; try again afterwards), `cancelled`.
    pub fn detect(&self, mix: &Stereo, progress: &dyn Progress) -> Result<DetectOutcome> {
        let missing: Vec<String> = [PREVIEW_MODEL, TAGGER_MODEL]
            .into_iter()
            .filter(|id| !self.store.is_installed(id))
            .map(str::to_string)
            .collect();
        if !missing.is_empty() {
            return Ok(DetectOutcome::Unavailable { missing });
        }
        let entry = self.store.manifest().get(PREVIEW_MODEL)?.clone();
        let n = entry.segment_samples;
        let named = [StemId::Vocals, StemId::Drums, StemId::Bass];
        let mut wanted = Vec::with_capacity(named.len());
        for stem in named {
            wanted.push(
                entry
                    .source_index(stem.as_str())
                    .ok_or_else(|| Error::internal("preview model lacks a source"))?,
            );
        }
        let starts = preview_starts(mix, n);
        // Per stem: best level relative to the mix, tagger maxima.
        let mut evidence: Vec<(StemId, f32, Option<Vec<f32>>)> = named
            .into_iter()
            .chain([StemId::Other])
            .map(|s| (s, crate::audio::SILENCE_DB, None))
            .collect();

        let mut slot = self.lock_separator()?;
        if !starts.is_empty() {
            self.load_separator(&mut slot, PREVIEW_MODEL)?;
        }
        for &start in &starts {
            check_cancelled(progress)?;
            let end = (start + n).min(mix.len());
            let excerpt = Stereo::from_channels(
                mix.left[start..end].to_vec(),
                mix.right[start..end].to_vec(),
            );
            let Some(loaded) = slot.as_mut() else {
                return Err(Error::internal("separator slot empty"));
            };
            // One model call: the excerpt is at most one segment long.
            let mut stems = separate_track(
                loaded.model.as_mut(),
                &excerpt,
                n,
                &wanted,
                0,
                &mut |_, _| check_cancelled(progress),
            )?;
            let mut other = excerpt.clone();
            for s in &stems {
                other.subtract(s);
            }
            stems.push(other);
            let mix_energy = energy(&excerpt).max(1e-20);
            for (slot_ev, audio) in evidence.iter_mut().zip(&stems) {
                let ratio_db = (10.0 * (energy(audio) / mix_energy).max(1e-20).log10()) as f32;
                slot_ev.1 = slot_ev.1.max(ratio_db);
                if ratio_db <= HEARD_FLOOR_DB {
                    continue;
                }
                // One 10 s tagger window per stem per segment.
                let scores = self.tag_scores(audio, crate::tagger::WINDOW, progress)?;
                match &mut slot_ev.2 {
                    Some(best) => {
                        for (b, s) in best.iter_mut().zip(&scores) {
                            *b = b.max(*s);
                        }
                    }
                    none => *none = Some(scores),
                }
            }
        }
        drop(slot);
        Ok(DetectOutcome::Done(report_from_evidence(
            &self.instrument_map,
            &evidence,
            starts.len(),
        )))
    }

    /// Names the instruments inside a separated stem. Only instrument
    /// families that can be in that stem, given `mode`, are considered (a
    /// bass guitar scores highly as "Guitar"). Needs the tagger.
    pub fn detect_in_stem(
        &self,
        stem_audio: &Stereo,
        stem: StemId,
        mode: SplitMode,
        progress: &dyn Progress,
    ) -> Result<Vec<Detection>> {
        // Back-to-back windows: half the work of the reference schedule.
        let scores = self.tag_scores(stem_audio, crate::tagger::WINDOW, progress)?;
        Ok(self.instrument_map.detect_in_stem(&scores, stem, mode))
    }

    /// What `options` needs: the models (with install state and terms),
    /// the bytes still to download and which restricted models need the
    /// user's consent first. Cheap; touches no network.
    pub fn plan(&self, options: SplitOptions) -> Result<SplitPlan> {
        let plan = passes(options);
        let mut models: Vec<ModelStatus> = Vec::new();
        for pass in &plan {
            if !models.iter().any(|m| m.id == pass.model_id) {
                models.push(self.store.status(pass.model_id)?);
            }
        }
        let download_bytes = models
            .iter()
            .filter(|m| !m.installed)
            .map(|m| m.download_bytes)
            .sum();
        let needs_consent = models
            .iter()
            .filter(|m| m.restricted && !m.installed)
            .map(|m| m.id.clone())
            .collect();
        Ok(SplitPlan {
            options,
            models,
            download_bytes,
            needs_consent,
            steps: plan.iter().map(Pass::label).collect(),
        })
    }

    /// Rough processing time in seconds for a track of `duration_sec`,
    /// model loading included, downloads excluded.
    pub fn estimate_seconds(&self, options: SplitOptions, duration_sec: f64) -> Result<f64> {
        let mut total = 0.0;
        for pass in passes(options) {
            self.store.manifest().get(pass.model_id)?;
            total += pass_cost(&pass, duration_sec) + 2.0;
        }
        // The estimate was measured with four fast cores.
        let threads = self.threads().max(1) as f64;
        Ok(total * (4.0 / threads).clamp(0.8, 4.0) + 0.04 * duration_sec)
    }

    fn lock_separator(&self) -> Result<std::sync::MutexGuard<'_, Option<LoadedSeparator>>> {
        match self.separator.try_lock() {
            Ok(g) => Ok(g),
            Err(TryLockError::Poisoned(p)) => Ok(p.into_inner()),
            Err(TryLockError::WouldBlock) => Err(Error::new(
                ErrorCode::Busy,
                "Another split is already running",
            )),
        }
    }

    fn load_separator(&self, slot: &mut Option<LoadedSeparator>, id: &str) -> Result<()> {
        if slot.as_ref().is_some_and(|s| s.id == id) {
            return Ok(());
        }
        // Drop the previous model first: two at once is the difference
        // between fitting in memory and not.
        *slot = None;
        let entry = self.store.manifest().get(id)?;
        let kind = entry
            .kind
            .separator()
            .ok_or_else(|| Error::internal(format!("{id} is not a separation model")))?;
        let verified = self.store.load(id)?;
        let external = verified
            .external
            .as_ref()
            .map(|(n, b)| (n.as_str(), b.clone()));
        let (session, _provider) = self
            .runtime()?
            .session_from_verified(&verified.graph, external)?;
        *slot = Some(LoadedSeparator {
            id: id.to_string(),
            model: open_segment_model(kind, session, entry.sources.len())?,
        });
        Ok(())
    }

    /// Splits `mix` (stereo, 44.1 kHz) into stems, in memory.
    ///
    /// The models must be installed (`model_missing` otherwise; see
    /// [`Engine::plan`] and [`ModelStore::install`]). Progress covers
    /// loading and separating. This holds every stem at once; the app path,
    /// [`Engine::split_to_dir`], needs less memory because it writes each
    /// stem out as soon as it exists.
    ///
    /// # Errors
    /// `model_missing`, `busy` (another split is running), `cancelled`,
    /// `out_of_memory`.
    pub fn split(
        &self,
        mix: &Stereo,
        options: SplitOptions,
        progress: &dyn Progress,
    ) -> Result<Split> {
        let plan = passes(options);
        let steps = self.separation_steps(&plan, mix)?;
        let tracker = Tracker::new(progress, steps);
        let mut produced: Vec<(StemId, Stereo)> = Vec::new();
        let model_ids = self.run_passes(mix, options, &plan, &tracker, 0, &mut |id, audio| {
            produced.push((id, audio));
            Ok(())
        })?;
        // Lane order.
        let mut stems = Vec::with_capacity(produced.len());
        for id in options.mode.stems() {
            if let Some(at) = produced.iter().position(|(s, _)| s == id) {
                stems.push(produced.swap_remove(at));
            }
        }
        Ok(Split { stems, model_ids })
    }

    fn separation_steps(&self, plan: &[Pass], mix: &Stereo) -> Result<Vec<PlannedStep>> {
        let audio_sec = mix.duration_sec(SAMPLE_RATE);
        let mut steps = Vec::with_capacity(plan.len());
        for pass in plan {
            self.store.manifest().get(pass.model_id)?;
            steps.push(PlannedStep {
                stage: Stage::Separating,
                label: pass.label(),
                // Loading the model is part of the step's weight.
                weight: pass_cost(pass, audio_sec) + 1.0,
            });
        }
        Ok(steps)
    }

    /// Runs the passes of `plan`. Every stem is handed to `on_stem` as soon
    /// as it is final (the remainder stem last), so the caller decides how
    /// long it stays in memory. Returns the ids of the models that ran.
    fn run_passes(
        &self,
        mix: &Stereo,
        options: SplitOptions,
        plan: &[Pass],
        tracker: &Tracker<'_>,
        first_step: usize,
        on_stem: &mut dyn FnMut(StemId, Stereo) -> Result<()>,
    ) -> Result<Vec<String>> {
        // Fail early, before any work, if a model is missing.
        for pass in plan {
            if !self.store.is_installed(pass.model_id) {
                let entry = self.store.manifest().get(pass.model_id)?;
                return Err(Error::new(
                    ErrorCode::ModelMissing,
                    format!("The model \u{201c}{}\u{201d} is not installed", entry.name),
                ));
            }
        }
        let mut slot = self.lock_separator()?;
        // The mix minus everything produced so far. It ends up as "other".
        let mut residual = mix.clone();
        let mut model_ids = Vec::new();
        for (i, pass) in plan.iter().enumerate() {
            tracker.begin(first_step + i)?;
            let entry = self.store.manifest().get(pass.model_id)?.clone();
            self.load_separator(&mut slot, pass.model_id)?;
            let Some(loaded) = slot.as_mut() else {
                return Err(Error::internal("separator slot empty"));
            };
            let mut wanted = Vec::with_capacity(pass.stems.len());
            for stem in pass.stems {
                wanted.push(entry.source_index(stem.as_str()).ok_or_else(|| {
                    Error::internal(format!("{} has no source {}", entry.id, stem.as_str()))
                })?);
            }
            let stride = entry.stride_samples;
            let total = count_segments(mix.len(), stride, 0).max(1);
            // Load time counts as the first few percent of the step.
            tracker.update(0.03)?;
            let input = match pass.input {
                PassInput::Mix => mix,
                PassInput::Residual => &residual,
            };
            let stems = separate_track(
                loaded.model.as_mut(),
                input,
                stride,
                &wanted,
                0,
                &mut |done, _| tracker.update(0.03 + 0.97 * done as f64 / total as f64),
            )?;
            model_ids.push(entry.id.clone());
            // On a long track the model is released before the stems are
            // measured and written (and before the next model loads): there,
            // keeping it is the difference between fitting in memory and
            // not. A short track keeps it loaded for the next job.
            if mix.duration_sec(SAMPLE_RATE) > 300.0 {
                *slot = None;
            }
            for (stem, audio) in pass.stems.iter().zip(stems) {
                residual.subtract(&audio);
                on_stem(*stem, audio)?;
            }
        }
        drop(slot);
        on_stem(remainder_stem(options.mode), residual)?;
        Ok(model_ids)
    }

    /// The whole job behind `start_split`: optionally download missing
    /// models, split, measure every stem, name what is inside it, and write
    /// each stem to `out_dir` as `<stem id>.wav` (32-bit float, 44.1 kHz
    /// stereo; written to a temporary name and renamed, so a file under
    /// its final name is always complete).
    ///
    /// Each stem is measured, written and released as soon as the model
    /// pass that makes it has finished, so at most one pass's stems are in
    /// memory at a time.
    ///
    /// Progress is one monotonic fraction across all stages, with a label
    /// and step counts per stage and an ETA once there is history.
    ///
    /// # Errors
    /// `terms_required` (a needed restricted model is missing and
    /// `accept_restricted` is false; nothing is downloaded), `model_missing`
    /// (missing and `download_missing` is false), `offline`,
    /// `model_download_failed`, `busy`, `cancelled`, `disk_full`,
    /// `out_of_memory`.
    pub fn split_to_dir(
        &self,
        mix: &Stereo,
        request: SplitRequest,
        out_dir: &Path,
        progress: &dyn Progress,
    ) -> Result<SplitResult> {
        let started = Instant::now();
        let options = request.options;
        let plan = passes(options);
        let split_plan = self.plan(options)?;
        let missing: Vec<&ModelStatus> =
            split_plan.models.iter().filter(|m| !m.installed).collect();
        if !missing.is_empty() {
            if let Some(m) = missing.iter().find(|m| m.restricted)
                && !request.accept_restricted
            {
                return Err(Error::new(
                    ErrorCode::TermsRequired,
                    format!(
                        "\u{201c}{}\u{201d} has licence terms that need to be accepted first",
                        m.name
                    ),
                ));
            }
            if !request.download_missing {
                return Err(Error::new(
                    ErrorCode::ModelMissing,
                    format!(
                        "The model \u{201c}{}\u{201d} is not installed",
                        missing[0].name
                    ),
                ));
            }
        }

        let audio_sec = mix.duration_sec(SAMPLE_RATE);
        let mut steps = Vec::new();
        for m in &missing {
            steps.push(PlannedStep {
                stage: Stage::Downloading,
                label: format!("Downloading {}", m.name),
                // Assume a few MB per second; only the ratio matters.
                weight: m.download_bytes as f64 / 4.0e6 + 1.0,
            });
        }
        let downloads = steps.len();
        steps.extend(self.separation_steps(&plan, mix)?);
        // Measuring and saving the stems of a pass happens at the end of
        // that pass's step; these two steps cover the remainder stem.
        let analyze_step = steps.len();
        steps.push(PlannedStep {
            stage: Stage::Analyzing,
            label: "Checking the stems".into(),
            weight: 0.3 + 0.006 * audio_sec,
        });
        steps.push(PlannedStep {
            stage: Stage::Writing,
            label: "Saving stems".into(),
            weight: 0.2 + 0.002 * audio_sec,
        });
        let tracker = Tracker::new(progress, steps);

        // Stage 1: downloads.
        for (i, m) in missing.iter().enumerate() {
            tracker.begin(i)?;
            struct Bridge<'a, 'b>(&'a Tracker<'b>);
            impl DownloadObserver for Bridge<'_, '_> {
                fn progress(&self, _id: &str, received: u64, total: u64) {
                    let _ = self.0.update(received as f64 / total.max(1) as f64);
                }
                fn is_cancelled(&self) -> bool {
                    self.0.sink().is_cancelled()
                }
            }
            self.store
                .install(&m.id, request.accept_restricted, &Bridge(&tracker))?;
        }

        // Stages 2 to 4: separate; measure, name and save each stem as it
        // arrives, then let go of it.
        std::fs::create_dir_all(out_dir).map_err(|e| Error::from_write(&e, out_dir))?;
        let remainder = remainder_stem(options.mode);
        let tag = request.tag_stems && self.store.is_installed(TAGGER_MODEL);
        let mut results: Vec<StemResult> = Vec::new();
        let model_ids = self.run_passes(
            mix,
            options,
            &plan,
            &tracker,
            downloads,
            &mut |id, audio| {
                if id == remainder {
                    tracker.begin(analyze_step)?;
                }
                let stats = analyze_stereo(&audio, Some(mix), SAMPLE_RATE);
                let contains = if tag && stats.present {
                    self.detect_in_stem(&audio, id, options.mode, progress)?
                } else {
                    Vec::new()
                };
                if id == remainder {
                    tracker.begin(analyze_step + 1)?;
                }
                let path = out_dir.join(format!("{}.wav", id.as_str()));
                write_wav_f32(&path, &audio, SAMPLE_RATE)?;
                results.push(StemResult {
                    id,
                    path,
                    stats,
                    contains,
                });
                Ok(())
            },
        )?;
        tracker.update(1.0)?;
        // Lane order.
        results.sort_by_key(|r| options.mode.stems().iter().position(|s| *s == r.id));
        Ok(SplitResult {
            stems: results,
            model_ids,
            elapsed_sec: started.elapsed().as_secs_f64(),
        })
    }

    /// Measures one stem (see [`analyze_stem`]); `mix` sharpens the
    /// "audible" decision and should be passed when available.
    pub fn analyze(&self, stem: &Stereo, mix: Option<&Stereo>) -> StemStats {
        let s = stem.to_interleaved();
        let m = mix.map(Stereo::to_interleaved);
        analyze_stem(&s, m.as_deref(), 2, SAMPLE_RATE)
    }
}

/// Waveform overview of a working file, straight from a memory map.
pub fn peaks_for_file(path: &Path, buckets: usize) -> Result<Peaks> {
    let wav = MappedWav::open(path)?;
    Ok(compute_peaks(
        wav.samples(),
        usize::from(wav.channels()),
        buckets,
    ))
}

/// Exports one working file to `dst` in `format`.
pub fn export_file(
    src: &Path,
    format: ExportFormat,
    dst: &Path,
    progress: &dyn Progress,
) -> Result<()> {
    let wav = MappedWav::open(src)?;
    export_audio(
        wav.samples(),
        wav.channels(),
        wav.sample_rate(),
        format,
        dst,
        progress,
    )
}

/// Exports several working files mixed together, each with a gain in dB
/// (the mixer's current levels), to `dst` in `format`. All sources must be
/// stereo at the same rate.
pub fn export_mixdown(
    sources: &[(&Path, f32)],
    format: ExportFormat,
    dst: &Path,
    progress: &dyn Progress,
) -> Result<()> {
    let mut maps = Vec::with_capacity(sources.len());
    for (path, _) in sources {
        maps.push(MappedWav::open(path)?);
    }
    let Some(first) = maps.first() else {
        return Err(Error::new(
            ErrorCode::NotFound,
            "There is nothing to mix down",
        ));
    };
    let rate = first.sample_rate();
    if maps
        .iter()
        .any(|m| m.channels() != 2 || m.sample_rate() != rate)
    {
        return Err(Error::internal("mixdown sources differ in format"));
    }
    check_cancelled(progress)?;
    let inputs: Vec<(&[f32], f32)> = maps
        .iter()
        .zip(sources)
        .map(|(m, (_, g))| (m.samples(), *g))
        .collect();
    let mixed = mixdown(&inputs)?;
    export_audio(&mixed, 2, rate, format, dst, progress)
}

/// Which provider a session would use is an implementation detail of the
/// runtime; re-exported here for `AppInfo`.
pub type ExecutionProvider = Provider;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::separate::pipeline::Quality;
    use crate::testutil::TempDir;

    fn engine(dir: &TempDir) -> Engine {
        Engine::new(EngineConfig::new(dir.path().join("models"))).unwrap()
    }

    #[test]
    fn engine_is_send_and_sync_and_cheap() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
        let dir = TempDir::new("engine-new");
        let t0 = Instant::now();
        let e = engine(&dir);
        assert!(t0.elapsed().as_millis() < 500);
        assert!(e.models().dir().is_dir());
        assert!(e.threads() >= 1);
        e.configure(Acceleration::Cpu, 3);
        assert_eq!(e.threads(), 3);
        e.unload();
    }

    #[test]
    fn plan_reports_models_downloads_and_consent() {
        let dir = TempDir::new("engine-plan");
        let e = engine(&dir);
        let band = e
            .plan(SplitOptions {
                mode: SplitMode::Band,
                quality: Quality::Fast,
            })
            .unwrap();
        assert_eq!(band.models.len(), 1);
        assert_eq!(band.models[0].id, "scnet_small");
        assert!(!band.models[0].installed && !band.models[0].restricted);
        assert!(band.needs_consent.is_empty());
        assert_eq!(band.steps, vec!["Separating vocals, drums and bass"]);

        let all = e
            .plan(SplitOptions {
                mode: SplitMode::All,
                quality: Quality::Best,
            })
            .unwrap();
        let ids: Vec<&str> = all.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["scnet_large", "htdemucs_6s"]);
        assert_eq!(all.needs_consent, vec!["htdemucs_6s"]);
        let six = &all.models[1];
        assert!(six.restricted && six.terms.as_deref().is_some_and(|t| !t.is_empty()));
        assert_eq!(six.source, "Meta");
        assert_eq!(
            all.download_bytes,
            all.models.iter().map(|m| m.download_bytes).sum::<u64>()
        );
        // Serialises with the contract's field names.
        let v = serde_json::to_value(&all).unwrap();
        assert!(v["models"][1]["restricted"].as_bool().unwrap());
        assert!(v.get("downloadBytes").is_some() && v.get("needsConsent").is_some());

        let secs = e
            .estimate_seconds(
                SplitOptions {
                    mode: SplitMode::Band,
                    quality: Quality::Best,
                },
                240.0,
            )
            .unwrap();
        let fast = e
            .estimate_seconds(
                SplitOptions {
                    mode: SplitMode::Band,
                    quality: Quality::Fast,
                },
                240.0,
            )
            .unwrap();
        assert!(secs > fast && fast > 1.0, "{secs} {fast}");
    }

    #[test]
    fn missing_models_and_terms_are_reported_before_any_work() {
        let dir = TempDir::new("engine-missing");
        let e = engine(&dir);
        let mix = Stereo::silence(44_100);
        let out = dir.path().join("out");
        let fast = SplitOptions {
            mode: SplitMode::Band,
            quality: Quality::Fast,
        };
        let err = e
            .split(&mix, fast, &crate::progress::NoProgress)
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::ModelMissing);
        let err = e
            .split_to_dir(
                &mix,
                SplitRequest::new(fast),
                &out,
                &crate::progress::NoProgress,
            )
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::ModelMissing);
        // A plan with the restricted model: consent comes first, even with
        // downloads enabled, and nothing is fetched.
        let all = SplitOptions {
            mode: SplitMode::All,
            quality: Quality::Fast,
        };
        let mut req = SplitRequest::new(all);
        req.download_missing = true;
        let err = e
            .split_to_dir(&mix, req, &out, &crate::progress::NoProgress)
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::TermsRequired);
        assert!(!out.exists());
        assert_eq!(std::fs::read_dir(e.models().dir()).unwrap().count(), 0);
        // Detection without the tagger.
        match e.detect(&mix, &crate::progress::NoProgress).unwrap() {
            DetectOutcome::Unavailable { missing } => {
                assert_eq!(
                    missing,
                    vec![PREVIEW_MODEL.to_string(), TAGGER_MODEL.to_string()]
                );
            }
            other => panic!("expected unavailable, got {other:?}"),
        }
        let v = serde_json::to_value(DetectOutcome::Unavailable {
            missing: vec!["x".into()],
        })
        .unwrap();
        assert_eq!(v["status"], "unavailable");
    }

    #[test]
    fn preview_segments_are_spread_loud_and_distinct() {
        let n = 485_100usize;
        let tone = |frames: usize, amp: f32| -> Vec<f32> {
            (0..frames).map(|i| amp * (i as f32 * 0.05).sin()).collect()
        };
        // Silence and empty tracks: nothing to analyse.
        assert!(preview_starts(&Stereo::new(), n).is_empty());
        assert!(preview_starts(&Stereo::silence(n * 5), n).is_empty());
        assert!(preview_starts(&Stereo::silence(1000), n).is_empty());
        // Shorter than one segment: the whole thing, once.
        assert_eq!(
            preview_starts(&Stereo::from_mono(tone(n / 2, 0.3)), n),
            vec![0]
        );
        // Between one and two segments: one, in the middle.
        let s = preview_starts(&Stereo::from_mono(tone(n * 3 / 2, 0.3)), n);
        assert_eq!(s.len(), 1);
        // A four minute track: three, in order, inside the track, apart.
        let len = 44_100 * 240;
        let starts = preview_starts(&Stereo::from_mono(tone(len, 0.3)), n);
        assert_eq!(starts.len(), 3);
        assert!(starts.windows(2).all(|w| w[1] >= w[0] + n / 2));
        assert!(starts.iter().all(|s| s + n <= len));
        for (s, f) in starts.iter().zip([0.25, 0.5, 0.75]) {
            let centre = (*s + n / 2) as f64 / len as f64;
            assert!((centre - f).abs() < 0.06, "{centre} vs {f}");
        }
        // Loud only in the last third: quiet spots are skipped, the loud
        // part is found.
        let mut x = vec![0.0f32; len];
        for (i, v) in x.iter_mut().enumerate().skip(len * 2 / 3) {
            *v = 0.3 * (i as f32 * 0.05).sin();
        }
        let starts = preview_starts(&Stereo::from_mono(x), n);
        assert!(!starts.is_empty() && starts.len() <= 2, "{starts:?}");
        assert!(starts.iter().all(|s| s + n > len * 2 / 3));
    }

    #[test]
    fn evidence_becomes_detections() {
        use crate::tagger::NUM_CLASSES;
        let map = InstrumentMap::builtin().unwrap();
        assert_eq!(energy_confidence(-40.0), 0.0);
        assert_eq!(energy_confidence(-30.0), 0.0);
        assert!((energy_confidence(-21.0) - 0.5).abs() < 1e-6);
        assert_eq!(energy_confidence(-3.0), 1.0);

        let probs = |set: &[(usize, f32)]| {
            let mut p = vec![0f32; NUM_CLASSES];
            for (i, v) in set {
                p[*i] = *v;
            }
            Some(p)
        };
        // Class indices from the built-in map: 32 Male vocals, 140 Guitar,
        // 141 Electric guitar, 153 Piano, 162 Drum kit.
        let evidence = vec![
            // Vocals loud and named by the tagger: confidences combine.
            (StemId::Vocals, -8.0, probs(&[(32, 0.6)])),
            // Drums clearly there, but the tagger is unsure: energy speaks.
            (StemId::Drums, -15.0, probs(&[(162, 0.02)])),
            // Bass stem is only bleed; even a confident tagger is ignored.
            (StemId::Bass, -38.0, probs(&[(142, 0.9)])),
            // "Other" holds a guitar; the vocal class there is not its family.
            (
                StemId::Other,
                -10.0,
                probs(&[(141, 0.45), (140, 0.5), (32, 0.9)]),
            ),
        ];
        let r = report_from_evidence(&map, &evidence, 3);
        assert!(r.vocals_heard && r.drums_heard && !r.bass_heard);
        assert_eq!(r.segments, 3);
        assert_eq!(r.suggested_mode, SplitMode::All, "a guitar was heard");
        let find = |family| {
            r.detections
                .iter()
                .filter(|d| d.family == family)
                .collect::<Vec<_>>()
        };
        let vocals = find(StemId::Vocals);
        assert_eq!(vocals.len(), 1);
        assert_eq!(vocals[0].label, "Male vocals");
        assert!(
            (vocals[0].confidence - 1.0).abs() < 1e-6,
            "energy alone is already full"
        );
        let drums = find(StemId::Drums);
        assert_eq!(drums.len(), 1);
        assert_eq!(drums[0].label, "Drums");
        assert!((drums[0].confidence - energy_confidence(-15.0)).abs() < 1e-6);
        assert!(find(StemId::Bass).is_empty());
        let guitar = find(StemId::Guitar);
        assert_eq!(guitar.len(), 1, "the umbrella class is collapsed");
        assert_eq!(guitar[0].label, "Electric guitar");
        assert!(
            r.detections
                .windows(2)
                .all(|w| w[0].confidence >= w[1].confidence)
        );
        assert!(
            r.detections
                .iter()
                .all(|d| (0.0..=1.0).contains(&d.confidence))
        );

        // Nothing but drums and bass: band, and "no vocals" is reported.
        let evidence = vec![
            (StemId::Vocals, -45.0, None),
            (StemId::Drums, -5.0, probs(&[(162, 0.8)])),
            (StemId::Bass, -9.0, None),
            (StemId::Other, -50.0, None),
        ];
        let r = report_from_evidence(&map, &evidence, 2);
        assert!(!r.vocals_heard && r.drums_heard && r.bass_heard);
        assert_eq!(r.suggested_mode, SplitMode::Band);
        // A silent track.
        let r = report_from_evidence(&map, &[], 0);
        assert!(r.detections.is_empty() && r.suggested_mode == SplitMode::Band);
    }

    #[test]
    fn file_helpers_work_on_working_files() {
        let dir = TempDir::new("engine-files");
        let a = dir.path().join("a.wav");
        let b = dir.path().join("b.wav");
        let tone = |amp: f32| Stereo::from_channels(vec![amp; 4_410], vec![-amp; 4_410]);
        write_wav_f32(&a, &tone(0.5), SAMPLE_RATE).unwrap();
        write_wav_f32(&b, &tone(0.25), SAMPLE_RATE).unwrap();
        let p = peaks_for_file(&a, 10).unwrap();
        assert_eq!(p.buckets, 10);
        assert_eq!(&p.data[..2], &[-64, 64]);
        let out = dir.path().join("mix.wav");
        export_mixdown(
            &[(&a, 0.0), (&b, 0.0)],
            ExportFormat::Wav32f,
            &out,
            &crate::progress::NoProgress,
        )
        .unwrap();
        let mixed = crate::wav::read_wav(&out, u64::MAX).unwrap().samples;
        assert!((mixed[0] - 0.75).abs() < 1e-6 && (mixed[1] + 0.75).abs() < 1e-6);
        let single = dir.path().join("a16.wav");
        export_file(
            &a,
            ExportFormat::Wav16,
            &single,
            &crate::progress::NoProgress,
        )
        .unwrap();
        assert_eq!(
            crate::wav::read_wav(&single, u64::MAX).unwrap().spec.frames,
            4_410
        );
        assert!(
            export_mixdown(&[], ExportFormat::Wav16, &out, &crate::progress::NoProgress).is_err()
        );
    }
}
