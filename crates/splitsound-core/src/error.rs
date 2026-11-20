//! The engine's error type.
//!
//! Every error carries a stable machine [`ErrorCode`] that maps one to one
//! onto `IpcError.code` in `src/ipc/contract.ts`, a plain-language message
//! that can be shown to a user as is, and optionally a technical detail string
//! meant for logs only.

use std::fmt;
use std::io;
use std::path::Path;

/// Stable machine code for an error. The string form (see
/// [`ErrorCode::as_str`]) is exactly the `IpcError.code` value of the UI
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The input file does not exist or cannot be opened.
    FileNotFound,
    /// The file is not a format (or uses a codec) the engine can read or write.
    UnsupportedFormat,
    /// The file is a supported format but is corrupt, truncated or too large.
    DecodeFailed,
    /// A model needed for the operation is not installed (or failed its
    /// integrity check).
    ModelMissing,
    /// A model with a restricted licence is needed and the caller has not
    /// passed the explicit accept flag.
    TermsRequired,
    /// A model download did not complete or did not verify.
    ModelDownloadFailed,
    /// The network could not be reached.
    Offline,
    /// The operation was cancelled by the caller.
    Cancelled,
    /// The engine is already doing something that excludes this request.
    Busy,
    /// An allocation or the inference runtime ran out of memory.
    OutOfMemory,
    /// The disk is full.
    DiskFull,
    /// Audio output device problem. Never produced by the engine itself; it
    /// exists so the code list matches the UI contract.
    AudioDevice,
    /// A named thing (model id, stem) does not exist.
    NotFound,
    /// Anything else: a bug, an unexpected runtime failure.
    Internal,
}

impl ErrorCode {
    /// The `IpcError.code` string.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::FileNotFound => "file_not_found",
            ErrorCode::UnsupportedFormat => "unsupported_format",
            ErrorCode::DecodeFailed => "decode_failed",
            ErrorCode::ModelMissing => "model_missing",
            ErrorCode::TermsRequired => "terms_required",
            ErrorCode::ModelDownloadFailed => "model_download_failed",
            ErrorCode::Offline => "offline",
            ErrorCode::Cancelled => "cancelled",
            ErrorCode::Busy => "busy",
            ErrorCode::OutOfMemory => "out_of_memory",
            ErrorCode::DiskFull => "disk_full",
            ErrorCode::AudioDevice => "audio_device",
            ErrorCode::NotFound => "not_found",
            ErrorCode::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An engine error: a machine code, a message for people and an optional
/// technical detail.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    code: ErrorCode,
    message: String,
    detail: Option<String>,
}

/// Shorthand for `std::result::Result<T, splitsound_core::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Builds an error from a code and a user-facing message (sentence case,
    /// no jargon).
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Error {
            code,
            message: message.into(),
            detail: None,
        }
    }

    /// Attaches a technical detail (the underlying error text) for logs.
    #[must_use]
    pub fn with_detail(mut self, detail: impl fmt::Display) -> Self {
        self.detail = Some(detail.to_string());
        self
    }

    /// The stable machine code.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// The plain-language message, ready to show.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Technical detail for logs, if any. Not meant for end users.
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    /// True when the error is a cancellation rather than a failure.
    pub fn is_cancelled(&self) -> bool {
        self.code == ErrorCode::Cancelled
    }

    /// The standard cancellation error.
    pub fn cancelled() -> Self {
        Error::new(ErrorCode::Cancelled, "Cancelled")
    }

    /// An internal error with a generic user message and a technical detail.
    pub fn internal(detail: impl fmt::Display) -> Self {
        Error::new(
            ErrorCode::Internal,
            "Something went wrong inside SplitSound",
        )
        .with_detail(detail)
    }

    pub(crate) fn decode(message: impl Into<String>) -> Self {
        Error::new(ErrorCode::DecodeFailed, message)
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Error::new(ErrorCode::UnsupportedFormat, message)
    }

    /// Maps an I/O error that happened while reading `path`.
    pub(crate) fn from_read(err: &io::Error, path: &Path) -> Self {
        let name = display_name(path);
        match err.kind() {
            io::ErrorKind::NotFound => Error::new(
                ErrorCode::FileNotFound,
                format!("The file \u{201c}{name}\u{201d} could not be found"),
            ),
            io::ErrorKind::PermissionDenied => Error::new(
                ErrorCode::FileNotFound,
                format!("SplitSound is not allowed to open \u{201c}{name}\u{201d}"),
            ),
            io::ErrorKind::OutOfMemory => Error::out_of_memory(),
            _ => Error::new(
                ErrorCode::Internal,
                format!("The file \u{201c}{name}\u{201d} could not be read"),
            ),
        }
        .with_detail(format!("{}: {err}", path.display()))
    }

    /// Maps an I/O error that happened while writing `path`.
    pub(crate) fn from_write(err: &io::Error, path: &Path) -> Self {
        let name = display_name(path);
        match err.kind() {
            io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => {
                Error::new(ErrorCode::DiskFull, "There is not enough free disk space")
            }
            io::ErrorKind::NotFound => Error::new(
                ErrorCode::FileNotFound,
                format!("The folder for \u{201c}{name}\u{201d} does not exist"),
            ),
            io::ErrorKind::PermissionDenied => Error::new(
                ErrorCode::Internal,
                format!("SplitSound is not allowed to write \u{201c}{name}\u{201d}"),
            ),
            io::ErrorKind::OutOfMemory => Error::out_of_memory(),
            _ => Error::new(
                ErrorCode::Internal,
                format!("The file \u{201c}{name}\u{201d} could not be written"),
            ),
        }
        .with_detail(format!("{}: {err}", path.display()))
    }

    pub(crate) fn out_of_memory() -> Self {
        Error::new(
            ErrorCode::OutOfMemory,
            "There is not enough memory to finish this",
        )
    }
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_the_ipc_contract() {
        // The exact list from `IpcError.code` in src/ipc/contract.ts.
        let expected = [
            "file_not_found",
            "unsupported_format",
            "decode_failed",
            "model_missing",
            "terms_required",
            "model_download_failed",
            "offline",
            "cancelled",
            "busy",
            "out_of_memory",
            "disk_full",
            "audio_device",
            "not_found",
            "internal",
        ];
        let all = [
            ErrorCode::FileNotFound,
            ErrorCode::UnsupportedFormat,
            ErrorCode::DecodeFailed,
            ErrorCode::ModelMissing,
            ErrorCode::TermsRequired,
            ErrorCode::ModelDownloadFailed,
            ErrorCode::Offline,
            ErrorCode::Cancelled,
            ErrorCode::Busy,
            ErrorCode::OutOfMemory,
            ErrorCode::DiskFull,
            ErrorCode::AudioDevice,
            ErrorCode::NotFound,
            ErrorCode::Internal,
        ];
        for (code, text) in all.iter().zip(expected) {
            assert_eq!(code.as_str(), text);
            assert_eq!(
                serde_json::to_string(code).unwrap(),
                format!("\"{text}\""),
                "serde form must equal as_str"
            );
        }
    }

    #[test]
    fn io_errors_map_to_codes() {
        let p = Path::new("/music/song.mp3");
        let e = Error::from_read(&io::Error::from(io::ErrorKind::NotFound), p);
        assert_eq!(e.code(), ErrorCode::FileNotFound);
        assert!(e.message().contains("song.mp3"));
        assert!(!e.message().contains("/music"), "no full paths in messages");
        let e = Error::from_write(&io::Error::from(io::ErrorKind::StorageFull), p);
        assert_eq!(e.code(), ErrorCode::DiskFull);
    }
}
