//! Records the exact paths selected through the native picker, OS open events,
//! or native file drops. The asset protocol stays disabled.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::Deserialize;
use serde_json::{Value, json};
use tauri::plugin::{Builder, Plugin, TauriPlugin};
use tauri::{Runtime, State, Window};
use tauri_plugin_dialog::{DialogExt, FilePath};

use crate::app::dunce_canonicalize;
use crate::ipc::{ErrorCode, IpcError, IpcResult};
use crate::util::lock;

#[derive(Default)]
pub(crate) struct PickedPaths(Mutex<HashSet<PathBuf>>);

impl PickedPaths {
    pub fn grant(&self, path: &Path, directory: bool) -> Option<PathBuf> {
        let path = dunce_canonicalize(path).ok()?;
        if !(if directory {
            path.is_dir()
        } else {
            path.is_file()
        }) {
            return None;
        }
        lock(&self.0).insert(path.clone());
        Some(path)
    }

    pub fn check(&self, path: &str, directory: bool) -> IpcResult<PathBuf> {
        let deny = || {
            let message = if directory {
                "Choose the export folder in SplitSound first."
            } else {
                "Choose the file in SplitSound, drop it in the window, or open it from your file manager."
            };
            IpcError::new(ErrorCode::FileNotFound, message)
        };
        let path = Path::new(path);
        if !path.is_absolute() {
            return Err(deny());
        }
        let path = dunce_canonicalize(path).map_err(|_| deny())?;
        if lock(&self.0).contains(&path)
            && (if directory {
                path.is_dir()
            } else {
                path.is_file()
            })
        {
            Ok(path)
        } else {
            Err(deny())
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
struct Options {
    title: Option<String>,
    directory: bool,
    multiple: bool,
    default_path: Option<String>,
    can_create_directories: Option<bool>,
    filters: Vec<Filter>,
}

#[derive(Deserialize)]
struct Filter {
    name: String,
    extensions: Vec<String>,
}

#[tauri::command]
async fn open<R: Runtime>(
    window: Window<R>,
    picked: State<'_, PickedPaths>,
    options: Options,
) -> IpcResult<Value> {
    let mut dialog = window.dialog().file();
    #[cfg(any(target_os = "macos", windows))]
    {
        dialog = dialog.set_parent(&window);
    }
    if let Some(title) = options.title {
        dialog = dialog.set_title(title);
    }
    if let Some(path) = options.default_path
        && let Ok(dir) = picked.check(&path, true)
    {
        dialog = dialog.set_directory(dir);
    }
    if let Some(can_create) = options.can_create_directories {
        dialog = dialog.set_can_create_directories(can_create);
    }
    for filter in options.filters {
        let extensions: Vec<_> = filter.extensions.iter().map(String::as_str).collect();
        dialog = dialog.add_filter(filter.name, &extensions);
    }
    let directory = options.directory;
    let multiple = options.multiple;
    let selection = tauri::async_runtime::spawn_blocking(move || match (directory, multiple) {
        (true, true) => dialog.blocking_pick_folders(),
        (true, false) => dialog.blocking_pick_folder().map(|p| vec![p]),
        (false, true) => dialog.blocking_pick_files(),
        (false, false) => dialog.blocking_pick_file().map(|p| vec![p]),
    })
    .await
    .map_err(|_| IpcError::internal("Couldn't open the file picker."))?;
    let Some(selection) = selection else {
        return Ok(Value::Null);
    };
    let paths: Vec<_> = selection
        .into_iter()
        .filter_map(|p: FilePath| picked.grant(&p.into_path().ok()?, directory))
        .collect();
    Ok(if multiple {
        json!(paths)
    } else {
        json!(paths.first())
    })
}

/// Preserves the frontend's `plugin:dialog|open` API while granting only the
/// paths returned by the native picker. No other dialog IPC is exposed.
pub(crate) fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("dialog")
        .setup(|app, _| tauri_plugin_dialog::init().initialize(app, Value::Null))
        .invoke_handler(tauri::generate_handler![open])
        .build()
}
