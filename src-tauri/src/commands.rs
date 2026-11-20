//! Tauri commands: one per entry of `Commands` in `src/ipc/contract.ts`.
//!
//! Each is a thin wrapper over [`AppCore`]. Anything that can touch the
//! disk for long runs on the blocking thread pool, so the main thread and
//! the async runtime never stall. Arguments arrive camelCase and are
//! validated by `AppCore` before any path is built from them.

use std::sync::Arc;

use tauri::{AppHandle, State};

use crate::app::AppCore;
use crate::ipc::*;
use crate::open_files::OpenFilesGate;
use crate::picker::PickedPaths;

type Core<'a> = State<'a, Arc<AppCore>>;

/// Runs `f` on the blocking pool. A panic becomes an `internal` error.
async fn blocking<T, F>(core: &Arc<AppCore>, f: F) -> IpcResult<T>
where
    T: Send + 'static,
    F: FnOnce(&AppCore) -> IpcResult<T> + Send + 'static,
{
    let c = core.clone();
    tauri::async_runtime::spawn_blocking(move || f(&c))
        .await
        .map_err(|e| {
            log::error!("command panicked: {e}");
            IpcError::internal("Something went wrong. Please try again.")
        })?
}

/* Library */

#[tauri::command]
pub async fn list_sessions(
    core: Core<'_>,
    gate: State<'_, Arc<OpenFilesGate>>,
    app: AppHandle,
) -> IpcResult<Vec<SessionSummary>> {
    // The frontend subscribes to app:open-files before its startup refresh.
    if gate.set_ready(&crate::TauriSink(app)) {
        log::info!("main window connected to IPC");
    }
    Ok(core.list_sessions())
}

#[tauri::command]
pub(crate) async fn import_track(
    core: Core<'_>,
    picked: State<'_, PickedPaths>,
    path: String,
) -> IpcResult<Session> {
    let path = picked.check(&path, false)?;
    blocking(&core, move |c| c.import_track(&path.to_string_lossy())).await
}

#[tauri::command]
pub async fn get_session(core: Core<'_>, session_id: String) -> IpcResult<Session> {
    core.get_session(&session_id)
}

#[tauri::command]
pub async fn delete_session(core: Core<'_>, session_id: String) -> IpcResult<()> {
    blocking(&core, move |c| c.delete_session(&session_id)).await
}

#[tauri::command]
pub async fn clear_library(core: Core<'_>) -> IpcResult<()> {
    blocking(&core, |c| c.clear_library()).await
}

#[tauri::command]
pub async fn get_artwork(core: Core<'_>, session_id: String) -> IpcResult<Option<String>> {
    blocking(&core, move |c| c.get_artwork(&session_id)).await
}

#[tauri::command]
pub async fn get_peaks(
    core: Core<'_>,
    session_id: String,
    source: SourceId,
    buckets: u32,
) -> IpcResult<Peaks> {
    blocking(&core, move |c| c.get_peaks(&session_id, source, buckets)).await
}

/* Splitting */

#[tauri::command]
pub async fn estimate_split(
    core: Core<'_>,
    session_id: String,
    options: SplitOptions,
) -> IpcResult<SplitEstimate> {
    blocking(&core, move |c| c.estimate_split(&session_id, options)).await
}

#[tauri::command]
pub async fn start_split(
    core: Core<'_>,
    session_id: String,
    options: SplitOptions,
    accept_restricted: Option<bool>,
) -> IpcResult<()> {
    blocking(&core, move |c| {
        c.start_split(&session_id, options, accept_restricted.unwrap_or(false))
    })
    .await
}

#[tauri::command]
pub async fn cancel_split(core: Core<'_>, session_id: String) -> IpcResult<()> {
    blocking(&core, move |c| c.cancel_split(&session_id)).await
}

/* Playback */

#[tauri::command]
pub async fn playback_load(core: Core<'_>, session_id: String) -> IpcResult<PlaybackState> {
    blocking(&core, move |c| c.playback_load(&session_id)).await
}

#[tauri::command]
pub async fn playback_unload(core: Core<'_>) -> IpcResult<()> {
    blocking(&core, |c| {
        c.player.unload();
        Ok(())
    })
    .await
}

#[tauri::command]
pub async fn playback_play(core: Core<'_>) -> IpcResult<()> {
    blocking(&core, |c| c.player.play()).await
}

#[tauri::command]
pub async fn playback_pause(core: Core<'_>) -> IpcResult<()> {
    core.player.pause();
    Ok(())
}

#[tauri::command]
pub async fn playback_seek(core: Core<'_>, position_sec: f64) -> IpcResult<()> {
    core.player.seek(position_sec)
}

#[tauri::command]
pub async fn playback_set_channel(
    core: Core<'_>,
    id: SourceId,
    gain_db: Option<f64>,
    muted: Option<bool>,
    solo: Option<bool>,
) -> IpcResult<()> {
    core.player.set_channel(id, gain_db, muted, solo)
}

#[tauri::command]
pub async fn playback_set_master(core: Core<'_>, gain_db: f64) -> IpcResult<()> {
    core.player.set_master(gain_db);
    Ok(())
}

#[tauri::command]
pub async fn playback_set_source(core: Core<'_>, source: PlaybackSource) -> IpcResult<()> {
    core.player.set_source(source)
}

#[tauri::command]
pub async fn playback_set_loop(core: Core<'_>, r#loop: Option<LoopRange>) -> IpcResult<()> {
    core.player.set_loop(r#loop)
}

/* Export */

#[tauri::command]
pub(crate) async fn export_stems(
    core: Core<'_>,
    picked: State<'_, PickedPaths>,
    session_id: String,
    mut options: ExportOptions,
) -> IpcResult<ExportResult> {
    let dir = picked.check(&options.destination_dir, true)?;
    options.destination_dir = dir.to_string_lossy().into_owned();
    blocking(&core, move |c| c.export_stems(&session_id, options)).await
}

#[tauri::command]
pub async fn prepare_drag(
    core: Core<'_>,
    session_id: String,
    stem_ids: Vec<StemId>,
) -> IpcResult<Vec<String>> {
    blocking(&core, move |c| c.prepare_drag(&session_id, &stem_ids)).await
}

/* Models */

#[tauri::command]
pub async fn list_models(core: Core<'_>) -> IpcResult<Vec<ModelStatus>> {
    blocking(&core, AppCore::list_models).await
}

#[tauri::command]
pub async fn download_models(
    core: Core<'_>,
    model_ids: Vec<String>,
    accept_restricted: Option<bool>,
) -> IpcResult<()> {
    let c: Arc<AppCore> = core.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        c.download_models(model_ids, accept_restricted.unwrap_or(false))
    })
    .await
    .map_err(|_| IpcError::internal("Something went wrong. Please try again."))?
}

#[tauri::command]
pub async fn cancel_model_downloads(core: Core<'_>) -> IpcResult<()> {
    core.cancel_model_downloads();
    Ok(())
}

#[tauri::command]
pub async fn remove_model(core: Core<'_>, model_id: String) -> IpcResult<()> {
    blocking(&core, move |c| c.remove_model(&model_id)).await
}

#[tauri::command]
pub(crate) async fn import_model_files(
    core: Core<'_>,
    picked: State<'_, PickedPaths>,
    paths: Vec<String>,
) -> IpcResult<Vec<String>> {
    let paths = paths
        .iter()
        .map(|p| {
            picked
                .check(p, false)
                .map(|p| p.to_string_lossy().into_owned())
        })
        .collect::<IpcResult<Vec<_>>>()?;
    blocking(&core, move |c| c.import_model_files(&paths)).await
}

/* Settings and system */

#[tauri::command]
pub async fn get_settings(core: Core<'_>) -> IpcResult<Settings> {
    Ok(core.get_settings())
}

#[tauri::command]
pub(crate) async fn update_settings(
    core: Core<'_>,
    picked: State<'_, PickedPaths>,
    mut patch: SettingsPatch,
) -> IpcResult<Settings> {
    if let Some(Some(dir)) = &mut patch.export_dir {
        *dir = picked.check(dir, true)?.to_string_lossy().into_owned();
    }
    blocking(&core, move |c| c.update_settings(patch)).await
}

#[tauri::command]
pub async fn get_app_info(core: Core<'_>) -> IpcResult<AppInfo> {
    blocking(&core, |c| Ok(c.get_app_info())).await
}

#[tauri::command]
pub async fn reveal_path(core: Core<'_>, path: String) -> IpcResult<()> {
    blocking(&core, move |c| {
        let p = c.check_reveal(&path)?;
        tauri_plugin_opener::reveal_item_in_dir(&p).map_err(|e| {
            log::warn!("reveal {}: {e}", p.display());
            IpcError::internal("Couldn't show the file.")
        })
    })
    .await
}

/// Names of every command, for the build script's permission manifest and
/// a test that keeps it in sync with `generate_handler!`.
pub const NAMES: &[&str] = &[
    "list_sessions",
    "import_track",
    "get_session",
    "delete_session",
    "clear_library",
    "get_artwork",
    "get_peaks",
    "estimate_split",
    "start_split",
    "cancel_split",
    "playback_load",
    "playback_unload",
    "playback_play",
    "playback_pause",
    "playback_seek",
    "playback_set_channel",
    "playback_set_master",
    "playback_set_source",
    "playback_set_loop",
    "export_stems",
    "prepare_drag",
    "list_models",
    "download_models",
    "cancel_model_downloads",
    "remove_model",
    "import_model_files",
    "get_settings",
    "update_settings",
    "get_app_info",
    "reveal_path",
];

/// Builds the invoke handler with every command.
#[macro_export]
macro_rules! splitsound_handler {
    () => {
        tauri::generate_handler![
            $crate::commands::list_sessions,
            $crate::commands::import_track,
            $crate::commands::get_session,
            $crate::commands::delete_session,
            $crate::commands::clear_library,
            $crate::commands::get_artwork,
            $crate::commands::get_peaks,
            $crate::commands::estimate_split,
            $crate::commands::start_split,
            $crate::commands::cancel_split,
            $crate::commands::playback_load,
            $crate::commands::playback_unload,
            $crate::commands::playback_play,
            $crate::commands::playback_pause,
            $crate::commands::playback_seek,
            $crate::commands::playback_set_channel,
            $crate::commands::playback_set_master,
            $crate::commands::playback_set_source,
            $crate::commands::playback_set_loop,
            $crate::commands::export_stems,
            $crate::commands::prepare_drag,
            $crate::commands::list_models,
            $crate::commands::download_models,
            $crate::commands::cancel_model_downloads,
            $crate::commands::remove_model,
            $crate::commands::import_model_files,
            $crate::commands::get_settings,
            $crate::commands::update_settings,
            $crate::commands::get_app_info,
            $crate::commands::reveal_path,
        ]
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn command_list_matches_the_contract() {
        let contract = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../src/ipc/contract.ts"
        ))
        .expect("contract.ts");
        let body = contract
            .split("export interface Commands {")
            .nth(1)
            .and_then(|s| s.split("\n}").next())
            .expect("Commands block");
        let mut in_contract: Vec<&str> = body
            .lines()
            .filter_map(|l| {
                // Command names are the keys indented by exactly two spaces.
                let l = l.strip_prefix("  ").filter(|r| !r.starts_with(' '))?;
                let (name, rest) = l.split_once(':')?;
                (rest.trim_start().starts_with('{')
                    && name.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
                .then_some(name)
            })
            .collect();
        in_contract.sort();
        let mut ours = super::NAMES.to_vec();
        ours.sort();
        assert_eq!(ours, in_contract);
        let build =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs")).unwrap();
        for n in super::NAMES {
            assert!(
                build.contains(&format!("\"{n}\"")),
                "build.rs is missing {n}"
            );
        }
        let caps = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/capabilities/main.json"
        ))
        .unwrap();
        for n in super::NAMES {
            assert!(
                caps.contains(&format!("\"allow-{}\"", n.replace('_', "-"))),
                "capability is missing {n}"
            );
        }
    }
}
