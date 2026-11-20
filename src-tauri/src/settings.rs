//! User settings, stored as `settings.json` in the app config folder.
//!
//! Loading is tolerant: a missing or corrupt file gives the defaults, and
//! each key is read on its own, so an unknown key, a missing key or one
//! value of the wrong type never breaks startup or loses the other values.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::ipc::{IpcError, IpcResult, Settings, SettingsPatch};
use crate::util::{lock, read_bounded, write_atomic};

/// Most CPU threads a user may ask for.
pub const MAX_THREADS: u32 = 256;

pub struct SettingsStore {
    path: PathBuf,
    current: Mutex<Settings>,
}

fn field<T: DeserializeOwned>(obj: &serde_json::Map<String, Value>, key: &str) -> Option<T> {
    obj.get(key)
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn valid_dir(s: &str) -> bool {
    !s.is_empty() && !s.contains('\0') && Path::new(s).is_absolute()
}

/// Parses settings JSON, keeping defaults for anything missing or invalid.
pub fn parse_tolerant(bytes: &[u8]) -> Settings {
    let mut s = Settings::default();
    let Ok(Value::Object(obj)) = serde_json::from_slice::<Value>(bytes) else {
        return s;
    };
    if let Some(v) = field(&obj, "defaultQuality") {
        s.default_quality = v;
    }
    if let Some(v) = field(&obj, "exportFormat") {
        s.export_format = v;
    }
    if let Some(v) = field::<Option<String>>(&obj, "exportDir") {
        s.export_dir = v.filter(|d| valid_dir(d));
    }
    if let Some(v) = field(&obj, "acceleration") {
        s.acceleration = v;
    }
    if let Some(v) = field::<u32>(&obj, "threads") {
        s.threads = v.min(MAX_THREADS);
    }
    if let Some(v) = field(&obj, "checkForUpdates") {
        s.check_for_updates = v;
    }
    s
}

impl SettingsStore {
    /// Loads `path`, falling back to defaults.
    pub fn load(path: PathBuf) -> Self {
        let current = match read_bounded(&path, 1024 * 1024) {
            Ok(bytes) => parse_tolerant(&bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Settings::default(),
            Err(e) => {
                log::warn!(
                    "settings: can't read {}: {e}; using defaults",
                    path.display()
                );
                Settings::default()
            }
        };
        SettingsStore {
            path,
            current: Mutex::new(current),
        }
    }

    pub fn get(&self) -> Settings {
        lock(&self.current).clone()
    }

    /// Applies a partial update, validates it and saves.
    pub fn update(&self, patch: SettingsPatch) -> IpcResult<Settings> {
        let mut cur = lock(&self.current);
        let mut next = cur.clone();
        if let Some(v) = patch.default_quality {
            next.default_quality = v;
        }
        if let Some(v) = patch.export_format {
            next.export_format = v;
        }
        if let Some(v) = patch.export_dir {
            if let Some(d) = &v
                && !valid_dir(d)
            {
                return Err(IpcError::internal("The export folder must be a full path."));
            }
            next.export_dir = v;
        }
        if let Some(v) = patch.acceleration {
            next.acceleration = v;
        }
        if let Some(v) = patch.threads {
            if v > MAX_THREADS {
                return Err(IpcError::internal(format!(
                    "The number of threads must be between 0 and {MAX_THREADS}."
                )));
            }
            next.threads = v;
        }
        if let Some(v) = patch.check_for_updates {
            next.check_for_updates = v;
        }
        let bytes =
            serde_json::to_vec_pretty(&next).map_err(|e| IpcError::internal(e.to_string()))?;
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        write_atomic(&self.path, &bytes).map_err(|e| IpcError::io("save the settings", &e))?;
        *cur = next.clone();
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::{Acceleration, ExportFormat, Quality};

    #[test]
    fn tolerant_loading() {
        assert_eq!(parse_tolerant(b""), Settings::default());
        assert_eq!(parse_tolerant(b"[1,2]"), Settings::default());
        let s = parse_tolerant(
            br#"{"defaultQuality":"best","threads":"lots","exportFormat":"ogg",
                "acceleration":"gpu","exportDir":"relative/dir","someFutureKey":{"a":1},
                "checkForUpdates":false}"#,
        );
        assert_eq!(s.default_quality, Quality::Best);
        assert_eq!(s.threads, 0, "wrong type keeps the default");
        assert_eq!(
            s.export_format,
            ExportFormat::Wav24,
            "unknown value keeps the default"
        );
        assert_eq!(s.acceleration, Acceleration::Gpu);
        assert_eq!(s.export_dir, None, "relative paths are dropped");
        assert!(!s.check_for_updates);
        assert_eq!(
            parse_tolerant(br#"{"threads": 100000}"#).threads,
            MAX_THREADS
        );
    }

    #[test]
    fn update_validates_and_persists() {
        let dir =
            std::env::temp_dir().join(format!("ss-settings-{}", crate::util::new_session_id()));
        let path = dir.join("settings.json");
        let store = SettingsStore::load(path.clone());
        assert_eq!(store.get(), Settings::default());
        let abs = if cfg!(windows) { "C:\\Music" } else { "/Music" };
        let s = store
            .update(SettingsPatch {
                threads: Some(4),
                export_dir: Some(Some(abs.into())),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(s.threads, 4);
        assert!(
            store
                .update(SettingsPatch {
                    export_dir: Some(Some("rel".into())),
                    ..Default::default()
                })
                .is_err()
        );
        assert!(
            store
                .update(SettingsPatch {
                    threads: Some(10_000),
                    ..Default::default()
                })
                .is_err()
        );
        let again = SettingsStore::load(path);
        assert_eq!(again.get().threads, 4);
        assert_eq!(again.get().export_dir.as_deref(), Some(abs));
        let cleared = again
            .update(SettingsPatch {
                export_dir: Some(None),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(cleared.export_dir, None);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
