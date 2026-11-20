//! Rust-owned update state. The webview can only check or install the
//! last candidate selected by the configured updater.

use std::sync::Mutex as StdMutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, async_runtime::Mutex};
use tauri_plugin_updater::{Error, Update, UpdaterExt};

use crate::TauriSink;
use crate::ipc::{AppEvent, ErrorCode, EventSink, IpcError, IpcResult, UpdateInfo, UpdateProgress};
use crate::util::lock;

pub(crate) struct Updates {
    available: bool,
    pending: Mutex<Option<Update>>,
}

struct ThrottleState {
    downloaded: u64,
    total: Option<u64>,
    last_emit: Option<Instant>,
    last_emitted_bytes: u64,
}

struct ProgressThrottle {
    every: Duration,
    state: StdMutex<ThrottleState>,
}

impl ProgressThrottle {
    const EVERY: Duration = Duration::from_millis(100);

    fn new(every: Duration) -> Self {
        Self {
            every,
            state: StdMutex::new(ThrottleState {
                downloaded: 0,
                total: None,
                last_emit: None,
                last_emitted_bytes: 0,
            }),
        }
    }

    fn on_chunk(
        &self,
        chunk: usize,
        total_bytes: Option<u64>,
        now: Instant,
    ) -> Option<UpdateProgress> {
        let mut st = lock(&self.state);
        st.downloaded = st.downloaded.saturating_add(chunk as u64);
        if total_bytes.is_some() {
            st.total = total_bytes;
        }
        let finished = st.total.is_some_and(|t| st.downloaded >= t);
        let due = st
            .last_emit
            .is_none_or(|last| now.saturating_duration_since(last) >= self.every);
        if due || finished {
            st.last_emit = Some(now);
            st.last_emitted_bytes = st.downloaded;
            Some(UpdateProgress {
                downloaded_bytes: st.downloaded,
                total_bytes: st.total,
            })
        } else {
            None
        }
    }

    fn on_finish(&self) -> Option<UpdateProgress> {
        let mut st = lock(&self.state);
        if st.last_emit.is_none() || st.last_emitted_bytes != st.downloaded {
            st.last_emitted_bytes = st.downloaded;
            Some(UpdateProgress {
                downloaded_bytes: st.downloaded,
                total_bytes: st.total,
            })
        } else {
            None
        }
    }
}

impl Updates {
    pub fn new(app: &AppHandle) -> Self {
        let configured = app.plugin(tauri_plugin_updater::Builder::new().build());
        if let Err(error) = &configured {
            log::warn!("could not initialize the updater: {error}");
        }
        Self {
            available: configured.is_ok(),
            pending: Mutex::new(None),
        }
    }

    pub async fn check(&self, app: &AppHandle) -> IpcResult<Option<UpdateInfo>> {
        let mut pending = self.pending.try_lock().map_err(|_| in_progress())?;
        *pending = None;
        if !self.available {
            return Err(IpcError::internal("The updater is unavailable."));
        }
        *pending = app
            .updater()
            .map_err(update_error)?
            .check()
            .await
            .map_err(update_error)?;
        Ok(pending.as_ref().map(|update| UpdateInfo {
            version: update.version.clone(),
            notes: update.body.clone(),
        }))
    }

    pub async fn install(&self, app: &AppHandle) -> IpcResult<()> {
        let mut pending = self.pending.try_lock().map_err(|_| in_progress())?;
        if !self.available {
            *pending = None;
            return Err(IpcError::internal("The updater is unavailable."));
        }
        let update = pending
            .as_ref()
            .ok_or_else(|| IpcError::internal("Check for an update before installing."))?;
        let sink = TauriSink(app.clone());
        let throttle = ProgressThrottle::new(ProgressThrottle::EVERY);
        let result = update
            .download_and_install(
                |chunk, total_bytes| {
                    if let Some(progress) = throttle.on_chunk(chunk, total_bytes, Instant::now()) {
                        sink.emit(AppEvent::UpdateProgress(progress));
                    }
                },
                || {
                    if let Some(progress) = throttle.on_finish() {
                        sink.emit(AppEvent::UpdateProgress(progress));
                    }
                },
            )
            .await;
        if let Err(error) = result {
            let mapped = update_error(error);
            if mapped.code != ErrorCode::Offline {
                *pending = None;
            }
            return Err(mapped);
        }
        *pending = None;
        #[cfg(not(windows))]
        app.request_restart();
        Ok(())
    }
}

fn in_progress() -> IpcError {
    IpcError::internal("An update operation is already in progress.")
}

fn update_error(error: Error) -> IpcError {
    log::warn!("updater: {error}");
    if matches!(&error, Error::Reqwest(e) if e.is_connect() || e.is_timeout() || e.is_body()) {
        IpcError::new(
            ErrorCode::Offline,
            "Couldn't reach the update server. Check your connection and try again.",
        )
    } else {
        IpcError::internal("Couldn't complete the update. Please try again later.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_throttle_limits_mid_download_updates_and_flushes_on_finish() {
        let t0 = Instant::now();
        let throttle = ProgressThrottle::new(Duration::from_millis(100));
        // First chunk always emits.
        assert_eq!(
            throttle.on_chunk(100, Some(500), t0),
            Some(UpdateProgress {
                downloaded_bytes: 100,
                total_bytes: Some(500),
            })
        );
        // Immediate follow-up chunk is suppressed.
        assert_eq!(
            throttle.on_chunk(100, Some(500), t0 + Duration::from_millis(20)),
            None
        );
        // After the throttle window, progress emits with accumulated bytes.
        assert_eq!(
            throttle.on_chunk(100, Some(500), t0 + Duration::from_millis(120)),
            Some(UpdateProgress {
                downloaded_bytes: 300,
                total_bytes: Some(500),
            })
        );
        // Suppressed tail chunk is flushed by on_finish.
        assert_eq!(
            throttle.on_chunk(50, Some(500), t0 + Duration::from_millis(130)),
            None
        );
        assert_eq!(
            throttle.on_finish(),
            Some(UpdateProgress {
                downloaded_bytes: 350,
                total_bytes: Some(500),
            })
        );
        // Calling on_finish again does not duplicate the last event.
        assert_eq!(throttle.on_finish(), None);
    }

    #[test]
    fn update_error_maps_io_without_leaking_details() {
        let err = update_error(Error::Io(std::io::Error::other(
            "secret internal filesystem path",
        )));
        assert_eq!(err.code, ErrorCode::Internal);
        assert!(!err.message.contains("secret"));
    }

    #[test]
    fn updater_config_enforces_signed_https_and_frozen_prototypes() {
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        assert_eq!(conf["app"]["security"]["freezePrototype"], true);
        let updater = &conf["plugins"]["updater"];
        assert_eq!(updater["requireSignedVersion"], true);
        assert!(
            !updater["pubkey"]
                .as_str()
                .unwrap_or_default()
                .trim()
                .is_empty()
        );
        let endpoints = updater["endpoints"].as_array().unwrap();
        assert!(!endpoints.is_empty());
        for ep in endpoints {
            assert!(
                ep.as_str().unwrap().starts_with("https://"),
                "updater endpoint must use HTTPS: {ep}"
            );
        }
    }
}
