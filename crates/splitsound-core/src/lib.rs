//! SplitSound engine.

pub mod analyze;
pub mod audio;
pub mod checkpoint;
pub mod decode;
pub mod engine;
pub mod error;
pub mod export;
pub mod manifest;
pub mod opus;
pub mod peaks;
pub mod playback;
pub mod progress;
pub mod resample;
pub mod runtime;
pub mod separate;
pub mod stft;
pub mod store;
pub mod tagger;
pub mod wav;

#[cfg(test)]
pub(crate) mod testutil;

pub use engine::{
    DetectOutcome, DetectionReport, Engine, EngineConfig, GpuInfo, ImportedTrack, Split, SplitPlan,
    SplitRequest, SplitResult, StemResult,
};
pub use error::{Error, ErrorCode, Result};
pub use export::ExportFormat;
pub use progress::{CancelFlag, FnProgress, NoProgress, Progress, ProgressUpdate, Stage};
pub use runtime::Acceleration;
pub use separate::pipeline::{Quality, SplitOptions};
pub use store::{DownloadObserver, ModelSource, ModelStatus, ModelStore, NoObserver};
pub use tagger::{Detection, SplitMode, StemId};
