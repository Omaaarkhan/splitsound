//! Everything that crosses the webview boundary: payload types, the error
//! type and the events Rust sends to the main window.
//!
//! The source of truth is `src/ipc/contract.ts`; [`types`] mirrors it.

pub mod types;

use std::fmt;
use std::sync::Mutex;

use serde::Serialize;

pub use types::*;

/// Result type of every command.
pub type IpcResult<T> = Result<T, IpcError>;

impl IpcError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        IpcError {
            code,
            message: message.into(),
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }

    pub fn busy(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Busy, message)
    }

    pub fn cancelled() -> Self {
        Self::new(ErrorCode::Cancelled, "Cancelled.")
    }

    /// Maps an I/O error to a plain-language error. `what` describes the
    /// operation, e.g. "save the session".
    pub fn io(what: &str, err: &std::io::Error) -> Self {
        log::warn!("I/O error while trying to {what}: {err}");
        match err.kind() {
            std::io::ErrorKind::NotFound => Self::new(
                ErrorCode::FileNotFound,
                format!("Couldn't {what}: the file is missing."),
            ),
            std::io::ErrorKind::StorageFull => Self::new(
                ErrorCode::DiskFull,
                format!("Couldn't {what}: the disk is full. Free up some space and try again."),
            ),
            std::io::ErrorKind::OutOfMemory => Self::new(
                ErrorCode::OutOfMemory,
                format!("Couldn't {what}: the computer ran out of memory."),
            ),
            std::io::ErrorKind::PermissionDenied => Self::new(
                ErrorCode::Internal,
                format!("Couldn't {what}: permission was denied."),
            ),
            _ => Self::new(ErrorCode::Internal, format!("Couldn't {what}.")),
        }
    }
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for IpcError {}

/// An event sent to the main window. Serialises to its payload only.
// Events are built and serialised at once, never stored in bulk, so the
// size difference between variants doesn't matter.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum AppEvent {
    SessionUpdated(Session),
    LibraryChanged(Vec<SessionSummary>),
    SplitProgress(SplitProgress),
    PlaybackState(PlaybackState),
    PlaybackTick(PlaybackTick),
    ModelsProgress(ModelDownloadProgress),
    ModelsChanged(Vec<ModelStatus>),
    OpenFiles(OpenFiles),
}

impl AppEvent {
    /// Event name as listed in `Events` of the contract.
    pub fn name(&self) -> &'static str {
        match self {
            AppEvent::SessionUpdated(_) => "session:updated",
            AppEvent::LibraryChanged(_) => "library:changed",
            AppEvent::SplitProgress(_) => "split:progress",
            AppEvent::PlaybackState(_) => "playback:state",
            AppEvent::PlaybackTick(_) => "playback:tick",
            AppEvent::ModelsProgress(_) => "models:progress",
            AppEvent::ModelsChanged(_) => "models:changed",
            AppEvent::OpenFiles(_) => "app:open-files",
        }
    }
}

/// Where events go. The app uses a Tauri-backed sink that emits to the
/// main window; tests use [`RecordingSink`].
pub trait EventSink: Send + Sync + 'static {
    fn emit(&self, event: AppEvent);
}

/// Drops every event. Useful before the window exists.
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&self, _event: AppEvent) {}
}

/// Keeps every event in memory, for tests.
#[derive(Default)]
pub struct RecordingSink {
    events: Mutex<Vec<AppEvent>>,
}

impl RecordingSink {
    pub fn take(&self) -> Vec<AppEvent> {
        std::mem::take(&mut *self.events.lock().unwrap_or_else(|e| e.into_inner()))
    }

    pub fn snapshot(&self) -> Vec<AppEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl EventSink for RecordingSink {
    fn emit(&self, event: AppEvent) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_serialises_as_bare_payload() {
        let e = AppEvent::OpenFiles(OpenFiles {
            paths: vec!["/a.wav".into()],
        });
        assert_eq!(e.name(), "app:open-files");
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({"paths": ["/a.wav"]})
        );
        let e = AppEvent::PlaybackTick(PlaybackTick {
            position_sec: 1.0,
            levels: vec![],
        });
        assert_eq!(e.name(), "playback:tick");
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({"positionSec": 1.0, "levels": []})
        );
    }
}
