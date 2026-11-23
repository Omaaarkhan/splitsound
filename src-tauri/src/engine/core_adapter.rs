//! Adapts the desktop shell's engine trait to splitsound-core.
//!
//! Debug builds can set `SPLITSOUND_MODELS_DIR` to an absolute path to an
//! existing model store (for example `vendor/models` in this checkout).
//! `SPLITSOUND_MODELS_BASE_URL` overrides the HTTPS project download base
//! URL; the core still enforces its host allowlist and manifest digests.
//! Release builds ignore both variables and use the app's own model store.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use splitsound_core as core;

use super::{
    CancelToken, DecodedTrack, Engine, EngineConfig, EngineError, EngineProgress, EngineResult,
    ExportItem, ExportJob, HardwareInfo, ScanOutcome, ScanResult, SplitOutput, SplitPlan,
    SplitRequest, StemAnalysis,
};
use crate::ipc::{
    Detection, ErrorCode, ExportFormat, GpuInfo, ModelDownloadProgress, ModelStatus, Peaks,
    Quality, SplitMode, SplitOptions, SplitStage, StemId, TrackInfo,
};
use crate::util::lock;

struct InferenceState {
    busy: bool,
    applied_threads: u32,
}

struct InferenceGuard<'a> {
    lock: &'a Mutex<InferenceState>,
    cv: &'a Condvar,
}

impl Drop for InferenceGuard<'_> {
    fn drop(&mut self) {
        let mut st = lock(self.lock);
        st.busy = false;
        self.cv.notify_one();
    }
}

pub struct CoreEngine {
    engine: core::Engine,
    requested_threads: AtomicU32,
    // The core's preview and split both use the separator. Let either wait
    // for the other without holding up import, playback, or downloads.
    inference_state: Mutex<InferenceState>,
    inference_cv: Condvar,
}

#[cfg(debug_assertions)]
pub(crate) fn models_dir_override() -> Option<PathBuf> {
    std::env::var_os("SPLITSOUND_MODELS_DIR")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
}

impl CoreEngine {
    pub fn new(config: EngineConfig) -> EngineResult<Self> {
        let runtime_path = config
            .onnxruntime
            .filter(|p| p.is_absolute())
            .ok_or_else(|| {
                EngineError::new(
                    ErrorCode::Internal,
                    "The bundled ONNX Runtime library is missing.",
                )
            })?;
        // Validate presence/non-placeholder size synchronously, deferring actual
        // dynamic library loading and session initialization until first inference.
        let meta = std::fs::metadata(&runtime_path).map_err(|_| {
            EngineError::new(
                ErrorCode::Internal,
                "The bundled ONNX Runtime library is missing.",
            )
        })?;
        if !meta.is_file() || meta.len() < 64 * 1024 {
            return Err(EngineError::new(
                ErrorCode::Internal,
                "The bundled ONNX Runtime library is missing.",
            ));
        }
        let mut engine_config = core::EngineConfig::new(config.models_dir);
        engine_config.onnxruntime_path = Some(runtime_path);
        engine_config.acceleration = core::Acceleration::Cpu;
        engine_config.cache_dir = Some(config.cache_dir);
        #[cfg(debug_assertions)]
        if let Ok(url) = std::env::var("SPLITSOUND_MODELS_BASE_URL")
            && !url.is_empty()
        {
            engine_config.model_source = core::ModelSource::Release {
                base_url: format!("{}/", url.trim_end_matches('/')),
            };
        }
        let engine = core::Engine::new(engine_config)?;
        engine.models().clean_temp_files();
        Ok(Self {
            engine,
            requested_threads: AtomicU32::new(0),
            inference_state: Mutex::new(InferenceState {
                busy: false,
                applied_threads: 0,
            }),
            inference_cv: Condvar::new(),
        })
    }

    fn apply_threads(&self, applied: &mut u32) {
        let requested = self.requested_threads.load(Ordering::Relaxed);
        if *applied != requested {
            self.engine
                .configure(core::Acceleration::Cpu, requested as usize);
            *applied = requested;
        }
    }

    fn inference(&self, cancel: &CancelToken) -> EngineResult<InferenceGuard<'_>> {
        let mut st = lock(&self.inference_state);
        loop {
            if cancel.is_cancelled() {
                return Err(EngineError::cancelled());
            }
            if !st.busy {
                st.busy = true;
                self.apply_threads(&mut st.applied_threads);
                return Ok(InferenceGuard {
                    lock: &self.inference_state,
                    cv: &self.inference_cv,
                });
            }
            let (guard, _) = self
                .inference_cv
                .wait_timeout(st, Duration::from_millis(25))
                .unwrap_or_else(|e| e.into_inner());
            st = guard;
        }
    }
}

impl From<core::Error> for EngineError {
    fn from(error: core::Error) -> Self {
        use core::ErrorCode as E;
        let code = match error.code() {
            E::FileNotFound => ErrorCode::FileNotFound,
            E::UnsupportedFormat => ErrorCode::UnsupportedFormat,
            E::DecodeFailed => ErrorCode::DecodeFailed,
            E::ModelMissing => ErrorCode::ModelMissing,
            E::TermsRequired => ErrorCode::TermsRequired,
            E::ModelDownloadFailed => ErrorCode::ModelDownloadFailed,
            E::Offline => ErrorCode::Offline,
            E::Cancelled => ErrorCode::Cancelled,
            E::Busy => ErrorCode::Busy,
            E::OutOfMemory => ErrorCode::OutOfMemory,
            E::DiskFull => ErrorCode::DiskFull,
            E::AudioDevice => ErrorCode::AudioDevice,
            E::NotFound => ErrorCode::NotFound,
            E::Internal => ErrorCode::Internal,
        };
        if let Some(detail) = error.detail() {
            log::warn!("engine {code:?}: {detail}");
        }
        Self::new(code, error.message())
    }
}

fn split_mode(mode: core::SplitMode) -> SplitMode {
    match mode {
        core::SplitMode::Vocals => SplitMode::Vocals,
        core::SplitMode::Band => SplitMode::Band,
        core::SplitMode::All => SplitMode::All,
    }
}

fn options(options: &SplitOptions) -> core::SplitOptions {
    core::SplitOptions {
        mode: match options.mode {
            SplitMode::Vocals => core::SplitMode::Vocals,
            SplitMode::Band => core::SplitMode::Band,
            SplitMode::All => core::SplitMode::All,
        },
        quality: match options.quality {
            Quality::Fast => core::Quality::Fast,
            Quality::Best => core::Quality::Best,
            Quality::Studio => core::Quality::Studio,
        },
    }
}

fn stem_id(stem: core::StemId) -> StemId {
    match stem {
        core::StemId::Vocals => StemId::Vocals,
        core::StemId::Drums => StemId::Drums,
        core::StemId::Bass => StemId::Bass,
        core::StemId::Guitar => StemId::Guitar,
        core::StemId::Piano => StemId::Piano,
        core::StemId::Other => StemId::Other,
        core::StemId::Instrumental => StemId::Instrumental,
    }
}

fn detection(d: core::Detection) -> Detection {
    Detection {
        label: d.label,
        family: stem_id(d.family),
        confidence: f64::from(d.confidence),
    }
}

fn model(m: core::ModelStatus) -> ModelStatus {
    ModelStatus {
        id: m.id,
        name: m.name,
        description: m.description,
        size_bytes: m.size_bytes,
        download_bytes: m.download_bytes,
        installed: m.installed,
        download_progress: m.download_progress.map(f64::from),
        license: m.license,
        attribution: m.attribution,
        source: m.source,
        restricted: m.restricted,
        terms: m.terms,
    }
}

type Callback<'a, T> = Mutex<&'a mut (dyn FnMut(T) + Send)>;

struct ProgressBridge<'a> {
    cancel: &'a CancelToken,
    callback: Option<Callback<'a, EngineProgress>>,
}

impl<'a> ProgressBridge<'a> {
    fn silent(cancel: &'a CancelToken) -> Self {
        Self {
            cancel,
            callback: None,
        }
    }
}

impl core::Progress for ProgressBridge<'_> {
    fn report(&self, update: &core::ProgressUpdate) {
        let Some(callback) = &self.callback else {
            return;
        };
        lock(callback)(EngineProgress {
            stage: match update.stage {
                core::Stage::Downloading => SplitStage::Downloading,
                core::Stage::Preparing => SplitStage::Preparing,
                core::Stage::Separating => SplitStage::Separating,
                core::Stage::Analyzing => SplitStage::Analyzing,
                core::Stage::Writing => SplitStage::Writing,
            },
            fraction: f64::from(update.fraction),
            label: update.label.clone(),
            step: update.step,
            step_count: update.step_count,
        });
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

struct DownloadBridge<'a> {
    cancel: &'a CancelToken,
    callback: Callback<'a, ModelDownloadProgress>,
}

impl core::DownloadObserver for DownloadBridge<'_> {
    fn progress(&self, id: &str, received_bytes: u64, total_bytes: u64) {
        lock(&self.callback)(ModelDownloadProgress {
            model_id: id.to_string(),
            received_bytes,
            total_bytes,
        });
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }
}

impl Engine for CoreEngine {
    fn name(&self) -> &'static str {
        "core"
    }

    fn decode(
        &self,
        input: &Path,
        mix_out: &Path,
        cancel: &CancelToken,
    ) -> EngineResult<DecodedTrack> {
        let wav = core::decode::decode_to_wav(
            input,
            mix_out,
            core::audio::SAMPLE_RATE,
            &core::decode::DecodeLimits::default(),
            false,
            &ProgressBridge::silent(cancel),
        )?;
        let info = wav.info;
        Ok(DecodedTrack {
            info: TrackInfo {
                path: String::new(),
                file_name: String::new(),
                title: info.title.unwrap_or_default(),
                artist: info.artist,
                album: info.album,
                duration_sec: info.duration_sec,
                sample_rate: info.sample_rate,
                channels: u32::from(info.channels),
                codec: info.codec,
                bitrate_kbps: info.bitrate_kbps,
                file_size_bytes: info.file_size_bytes,
                has_artwork: info.has_artwork,
            },
            artwork: wav.artwork.map(|art| art.data),
        })
    }

    fn scan_instruments(&self, wav: &Path, cancel: &CancelToken) -> EngineResult<ScanOutcome> {
        let _guard = self.inference(cancel)?;
        let mix = core::wav::MappedWav::open(wav)?;
        mix.advise_sequential();
        Ok(
            match self
                .engine
                .detect_mapped(&mix, &ProgressBridge::silent(cancel))?
            {
                core::DetectOutcome::Done(report) => ScanOutcome::Done(ScanResult {
                    detections: report.detections.into_iter().map(detection).collect(),
                    suggested_mode: split_mode(report.suggested_mode),
                }),
                core::DetectOutcome::Unavailable { missing } => ScanOutcome::MissingModels(missing),
            },
        )
    }

    fn plan_split(&self, duration_sec: f64, settings: &SplitOptions) -> EngineResult<SplitPlan> {
        let options = options(settings);
        let plan = self.engine.plan(options)?;
        Ok(SplitPlan {
            models: plan.models.into_iter().map(model).collect(),
            estimated_sec: self.engine.estimate_seconds(options, duration_sec)?,
            // The core plan lists separation passes. The job adds analysis
            // and writing after the last pass.
            step_count: plan.steps.len() as u32 + 2,
        })
    }

    fn split(
        &self,
        req: &SplitRequest,
        progress: &mut (dyn FnMut(EngineProgress) + Send),
        cancel: &CancelToken,
    ) -> EngineResult<SplitOutput> {
        let _guard = self.inference(cancel)?;
        let mix = core::wav::MappedWav::open(&req.mix)?.to_stereo()?;
        let mut request = core::SplitRequest::new(options(&req.options));
        request.accept_restricted = req.accept_restricted;
        // The shell owns downloads so it can also emit per-model progress.
        let bridge = ProgressBridge {
            cancel,
            callback: Some(Mutex::new(progress)),
        };
        let result =
            self.engine
                .split_to_dir_with_sync(&mix, request, &req.out_dir, false, &bridge)?;
        let stems = result
            .stems
            .into_iter()
            .map(|stem| {
                let stats = stem.stats;
                (
                    stem_id(stem.id),
                    StemAnalysis {
                        present: stats.present,
                        active_fraction: f64::from(stats.active_fraction),
                        peak_db: f64::from(stats.peak_db),
                        rms_db: f64::from(stats.rms_db),
                        activity: stats
                            .activity
                            .into_iter()
                            .map(|v| (f64::from(v) * 100.0).round() / 100.0)
                            .collect(),
                        contains: stem.contains.into_iter().map(detection).collect(),
                    },
                )
            })
            .collect();
        Ok(SplitOutput {
            stems,
            model_ids: result.model_ids,
        })
    }

    fn peaks(&self, wav: &Path, buckets: u32) -> EngineResult<Peaks> {
        let peaks = core::engine::peaks_for_file(wav, buckets as usize)?;
        Ok(Peaks {
            buckets: peaks.buckets as u32,
            data: peaks.to_base64(),
        })
    }

    fn export(&self, job: &ExportJob, cancel: &CancelToken) -> EngineResult<Vec<PathBuf>> {
        let format = match job.format {
            ExportFormat::Wav16 => core::ExportFormat::Wav16,
            ExportFormat::Wav24 => core::ExportFormat::Wav24,
            ExportFormat::Wav32f => core::ExportFormat::Wav32f,
            ExportFormat::Flac => core::ExportFormat::Flac,
            ExportFormat::Mp3 => core::ExportFormat::Mp3,
        };
        let export_one = |item: &ExportItem| -> EngineResult<PathBuf> {
            let progress = ProgressBridge::silent(cancel);
            if let [(path, 1.0)] = item.sources.as_slice() {
                core::engine::export_file(path, format, &item.dest, &progress)?;
            } else {
                let sources: Vec<_> = item
                    .sources
                    .iter()
                    .map(|(path, gain)| {
                        let db = if *gain > 0.0 {
                            20.0 * gain.log10()
                        } else {
                            core::audio::SILENCE_DB
                        };
                        (path.as_path(), db)
                    })
                    .collect();
                core::engine::export_mixdown(&sources, format, &item.dest, &progress)?;
            }
            Ok(item.dest.clone())
        };
        if job.items.len() <= 1 {
            let mut files = Vec::with_capacity(job.items.len());
            for item in &job.items {
                files.push(export_one(item)?);
            }
            return Ok(files);
        }
        let workers = self.engine.threads().clamp(2, 4).min(job.items.len());
        let next = AtomicUsize::new(0);
        let results: Vec<Mutex<Option<EngineResult<PathBuf>>>> =
            (0..job.items.len()).map(|_| Mutex::new(None)).collect();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= job.items.len() {
                            break;
                        }
                        let res = export_one(&job.items[i]);
                        *lock(&results[i]) = Some(res);
                    }
                });
            }
        });
        let mut files = Vec::with_capacity(job.items.len());
        for slot in results {
            let res = lock(&slot).take().unwrap_or_else(|| {
                Err(EngineError::new(
                    ErrorCode::Internal,
                    "Export task did not finish.",
                ))
            })?;
            files.push(res);
        }
        Ok(files)
    }

    fn list_models(&self) -> Vec<ModelStatus> {
        self.engine.models().list().into_iter().map(model).collect()
    }

    fn download_models(
        &self,
        ids: &[String],
        accept_restricted: bool,
        progress: &mut (dyn FnMut(ModelDownloadProgress) + Send),
        cancel: &CancelToken,
    ) -> EngineResult<()> {
        let store = self.engine.models();
        // Preflight the whole request so an unaccepted model cannot cause a
        // partial install, even when it was installed at the time of the call.
        for id in ids {
            let model = store.status(id)?;
            if model.restricted && !accept_restricted {
                return Err(EngineError::new(
                    ErrorCode::TermsRequired,
                    format!(
                        "{} needs your agreement to its terms before downloading.",
                        model.name
                    ),
                ));
            }
        }
        let ids: Vec<_> = ids.iter().map(String::as_str).collect();
        let bridge = DownloadBridge {
            cancel,
            callback: Mutex::new(progress),
        };
        store.ensure(&ids, accept_restricted, &bridge)?;
        Ok(())
    }

    fn remove_model(&self, id: &str) -> EngineResult<()> {
        self.engine.remove_model(id)?;
        Ok(())
    }

    fn import_model_files(&self, paths: &[PathBuf]) -> EngineResult<Vec<String>> {
        let mut imported = Vec::new();
        for path in paths {
            // The frozen import command has no consent argument.
            if let Some(id) = self.engine.models().import_file(path, false)?
                && !imported.contains(&id)
            {
                imported.push(id);
            }
        }
        Ok(imported)
    }

    fn configure_threads(&self, threads: u32) {
        self.requested_threads.store(threads, Ordering::Relaxed);
        if let Ok(mut st) = self.inference_state.try_lock()
            && !st.busy
        {
            self.apply_threads(&mut st.applied_threads);
        }
    }

    fn hardware(&self) -> HardwareInfo {
        let threads = self.requested_threads.load(Ordering::Relaxed);
        HardwareInfo {
            gpu: GpuInfo {
                available: false,
                name: None,
            },
            cpu_threads: if threads == 0 {
                core::runtime::auto_threads() as u32
            } else {
                threads
            },
        }
    }
}
