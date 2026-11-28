//! The boundary between the desktop shell and the separation engine.
//!
//! Everything the shell needs from an engine is the [`Engine`] trait. Two
//! implementations exist:
//!
//! - [`core_adapter::CoreEngine`] adapts the production engine.
//! - The debug and test builds can opt into a WAV-only mock with
//!   `SPLITSOUND_MOCK_ENGINE=1`.
//!
//! Swapping implementations touches nothing outside this module: the shell
//! only ever holds an `Arc<dyn Engine>` from [`select`].
//!
//! # Contract for implementors
//!
//! - Every method may be called from any thread, concurrently (a split, a
//!   scan, an export and a model download can overlap).
//! - Long operations poll the [`CancelToken`] and return
//!   [`EngineError::cancelled`] promptly once it is set.
//! - Files are written to a temporary name and renamed into place.
//! - Errors carry a plain-language message that is shown to the user as is.
//! - **Restricted models**: a model whose [`ModelStatus::restricted`] is
//!   true must never be downloaded unless the caller passed
//!   `accept_restricted = true`. Nothing may download a model as a side
//!   effect except [`Engine::download_models`]. The shell enforces this too.
//! - Panics are caught by the shell, but please don't rely on that.

pub mod core_adapter;
#[cfg(any(debug_assertions, test))]
pub mod mock;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ipc::{
    Detection, ErrorCode, ExportFormat, GpuInfo, IpcError, ModelDownloadProgress, ModelStatus,
    Peaks, SplitMode, SplitOptions, SplitStage, StemId, TrackInfo,
};
pub use crate::util::CancelToken;

/// An engine failure, ready to show. Maps 1:1 onto [`IpcError`].
#[derive(Debug, Clone, PartialEq)]
pub struct EngineError {
    pub code: ErrorCode,
    pub message: String,
}

impl EngineError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        EngineError {
            code,
            message: message.into(),
        }
    }

    pub fn cancelled() -> Self {
        Self::new(ErrorCode::Cancelled, "Cancelled.")
    }

    pub fn is_cancelled(&self) -> bool {
        self.code == ErrorCode::Cancelled
    }
}

impl From<EngineError> for IpcError {
    fn from(e: EngineError) -> Self {
        IpcError::new(e.code, e.message)
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

pub type EngineResult<T> = Result<T, EngineError>;

/// Where an engine keeps things and how it finds ONNX Runtime.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Installed models live here (`<app data>/models`).
    pub models_dir: PathBuf,
    /// Scratch space the engine may use (`<app cache>/engine`).
    pub cache_dir: PathBuf,
    /// Absolute path of the bundled ONNX Runtime library, if found and
    /// valid. See [`crate::onnxruntime`]. Always load it by this path.
    pub onnxruntime: Option<PathBuf>,
}

/// Result of [`Engine::decode`].
#[derive(Debug, Clone)]
pub struct DecodedTrack {
    /// Track facts. The shell overwrites `path`, `file_name`,
    /// `file_size_bytes` and `has_artwork`; tag text is passed through as
    /// plain strings and never interpreted.
    pub info: TrackInfo,
    /// Raw embedded cover art, if any. The shell keeps it only if the bytes
    /// are a PNG or JPEG.
    pub artwork: Option<Vec<u8>>,
}

/// Result of [`Engine::scan_instruments`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScanResult {
    /// Strongest first.
    pub detections: Vec<Detection>,
    pub suggested_mode: SplitMode,
}

pub enum ScanOutcome {
    Done(ScanResult),
    MissingModels(Vec<String>),
}

/// What a split needs, from [`Engine::plan_split`].
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPlan {
    /// Every model the split uses, with `installed`, `restricted` and
    /// `terms` filled in.
    pub models: Vec<ModelStatus>,
    /// Rough processing time on this machine, excluding downloads.
    pub estimated_sec: f64,
    /// How many steps [`Engine::split`] will report (`EngineProgress::step_count`),
    /// so the shell can number the whole job before the engine starts.
    pub step_count: u32,
}

impl SplitPlan {
    /// Models that still have to be downloaded.
    pub fn missing(&self) -> impl Iterator<Item = &ModelStatus> {
        self.models.iter().filter(|m| !m.installed)
    }

    /// Missing models that need the user's agreement before download.
    pub fn missing_restricted(&self) -> impl Iterator<Item = &ModelStatus> {
        self.missing().filter(|m| m.restricted)
    }
}

/// Input of [`Engine::split`].
#[derive(Debug, Clone)]
pub struct SplitRequest {
    /// The session's `mix.wav` (32-bit float, 44.1 kHz stereo).
    pub mix: PathBuf,
    /// Empty folder to write `<stem>.wav` files into (same format as the mix).
    pub out_dir: PathBuf,
    pub options: SplitOptions,
    /// The user agreed to the terms of restricted models. The shell has
    /// already downloaded every missing model; an engine that fetches
    /// models itself must still honour this flag.
    pub accept_restricted: bool,
}

/// Progress of the engine's part of a split.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineProgress {
    /// Stage of the engine's separation, analysis or file writing.
    pub stage: SplitStage,
    /// 0..1 over the whole call. Need not be monotonic; the shell smooths it.
    pub fraction: f64,
    /// Sentence case, no trailing period: "Separating drums".
    pub label: String,
    /// 1-based step and the number of steps of the engine's part.
    pub step: u32,
    pub step_count: u32,
}

/// Result of [`Engine::split`].
#[derive(Debug, Clone, PartialEq)]
pub struct SplitOutput {
    /// Measurements of the written stems, in lane order.
    pub stems: Vec<(StemId, StemAnalysis)>,
    pub model_ids: Vec<String>,
}

/// Measurements of one stem. See `StemInfo` in the contract.
#[derive(Debug, Clone, PartialEq)]
pub struct StemAnalysis {
    pub present: bool,
    pub active_fraction: f64,
    pub peak_db: f64,
    pub rms_db: f64,
    /// [`crate::ipc::ACTIVITY_BINS`] values in 0..1.
    pub activity: Vec<f64>,
    pub contains: Vec<Detection>,
    /// Optional precomputed 960-bucket raw peak bytes (`2 * 960` bytes) to seed
    /// the session's peak cache without re-reading the written WAV from disk.
    pub peaks_960: Option<Vec<u8>>,
}

/// One file of an export.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportItem {
    /// Working-format WAV(s) to read, each with a linear gain. One entry for
    /// a stem; several for a mixdown.
    pub sources: Vec<(PathBuf, f32)>,
    /// Final path (the shell picked a free, sanitised name).
    pub dest: PathBuf,
}

/// Input of [`Engine::export`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExportJob {
    pub format: ExportFormat,
    pub items: Vec<ExportItem>,
}

/// What [`Engine::hardware`] reports.
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareInfo {
    pub gpu: GpuInfo,
    pub cpu_threads: u32,
}

/// Everything the shell needs from a separation engine.
pub trait Engine: Send + Sync + 'static {
    /// Short name for logs ("mock", "core").
    fn name(&self) -> &'static str;

    /// Decodes `input` (any supported format) to `mix_out` as 32-bit float
    /// 44.1 kHz stereo WAV and reads its tags and cover art. Errors:
    /// `unsupported_format`, `decode_failed`, `file_not_found`, `disk_full`.
    fn decode(
        &self,
        input: &Path,
        mix_out: &Path,
        cancel: &CancelToken,
    ) -> EngineResult<DecodedTrack>;

    /// Lists instruments heard in a working-format WAV, or reports which
    /// models the shell needs to install before retrying the scan.
    fn scan_instruments(&self, wav: &Path, cancel: &CancelToken) -> EngineResult<ScanOutcome>;

    /// What a split of a song of `duration_sec` with `options` needs.
    fn plan_split(&self, duration_sec: f64, options: &SplitOptions) -> EngineResult<SplitPlan>;

    /// Separates `req.mix` into the stems of `req.options.mode`. All models
    /// must be installed (`model_missing` otherwise). `progress` may be
    /// called as often as convenient; the shell throttles it.
    fn split(
        &self,
        req: &SplitRequest,
        progress: &mut (dyn FnMut(EngineProgress) + Send),
        cancel: &CancelToken,
    ) -> EngineResult<SplitOutput>;

    /// Waveform overview of a working-format WAV. The default reads it
    /// memory-mapped; override only if the engine has something better.
    fn peaks(&self, wav: &Path, buckets: u32) -> EngineResult<Peaks> {
        crate::audio::peaks::peaks_of_file(wav, buckets).map_err(|e| {
            log::warn!("peaks of {}: {e}", wav.display());
            EngineError::new(
                ErrorCode::FileNotFound,
                "This audio file is missing or damaged.",
            )
        })
    }

    /// Writes every item of `job` in `job.format`. Returns the files written.
    /// `unsupported_format` if the format can't be produced.
    fn export(&self, job: &ExportJob, cancel: &CancelToken) -> EngineResult<Vec<PathBuf>>;

    /// Every model the engine knows, with install state. `download_progress`
    /// may be left `None`; the shell overlays it.
    fn list_models(&self) -> Vec<ModelStatus>;

    /// Downloads and installs models, reporting progress per model. Must
    /// refuse (`terms_required`) a restricted model unless
    /// `accept_restricted` is true. Unknown ids are `not_found`.
    fn download_models(
        &self,
        ids: &[String],
        accept_restricted: bool,
        progress: &mut (dyn FnMut(ModelDownloadProgress) + Send),
        cancel: &CancelToken,
    ) -> EngineResult<()>;

    fn remove_model(&self, id: &str) -> EngineResult<()>;

    /// Installs model files the user downloaded by hand, accepting only
    /// files that match a pinned digest. Returns the ids recognised.
    fn import_model_files(&self, paths: &[PathBuf]) -> EngineResult<Vec<String>>;

    fn configure_threads(&self, _threads: u32) {}

    /// Releases cached model sessions when idle or shutting down.
    fn unload_models(&self) {}

    fn hardware(&self) -> HardwareInfo;
}

/// Environment variable that forces the mock engine.
#[cfg(any(debug_assertions, test))]
pub const MOCK_ENV: &str = "SPLITSOUND_MOCK_ENGINE";

/// Starts the production engine unless a debug/test build explicitly requests the mock.
pub fn select(config: EngineConfig) -> EngineResult<Arc<dyn Engine>> {
    #[cfg(any(debug_assertions, test))]
    if std::env::var(MOCK_ENV).is_ok_and(|v| v == "1") {
        log::info!("engine: mock ({MOCK_ENV}=1)");
        return Ok(Arc::new(mock::MockEngine::new(config)));
    }
    let startup = std::panic::catch_unwind(|| core_adapter::CoreEngine::new(config));
    match startup {
        Ok(Ok(engine)) => {
            log::info!("engine: splitsound-core");
            return Ok(Arc::new(engine));
        }
        Ok(Err(e)) => log::error!("couldn't start the audio engine: {e}"),
        Err(panic) => log::error!(
            "audio engine startup panicked: {}",
            crate::worker::panic_message(panic.as_ref())
        ),
    }
    Err(EngineError::new(
        ErrorCode::Internal,
        "SplitSound's audio engine could not start. Try reinstalling the app.",
    ))
}
