//! SplitSound desktop shell (Tauri 2).
//!
//! Module map:
//!
//! - [`ipc`]: the contract types, `IpcError` and events.
//! - [`commands`]: the Tauri commands (thin wrappers over [`app::AppCore`]).
//! - [`app`]: the services behind the commands; Tauri-independent and tested.
//! - [`library`]: sessions on disk. [`worker`]: the split and scan queues.
//! - [`audio`]: WAV I/O, memory-mapped stems, the mixer, resampler, output.
//! - [`engine`]: the engine trait, the mock, and the real adapter stub.
//! - [`settings`], [`artwork`], [`open_files`], [`onnxruntime`], [`util`].
//!
//! [`run`] wires it all into a Tauri app.

pub mod app;
pub mod artwork;
pub mod audio;
pub mod commands;
mod drag;
pub mod engine;
pub mod ipc;
pub mod library;
pub mod onnxruntime;
pub mod open_files;
mod picker;
pub mod settings;
#[cfg(debug_assertions)]
mod smoke;
mod updates;
pub mod util;
pub mod worker;

use std::path::Path;
use std::sync::Arc;

use tauri::webview::PageLoadEvent;
use tauri::{
    AppHandle, DragDropEvent, Emitter, Manager, RunEvent, WebviewEvent, WebviewWindowBuilder,
    WindowEvent,
};

use crate::app::{AppCore, AppPaths, CoreOptions};
use crate::audio::output::CpalBackend;
use crate::engine::EngineConfig;
use crate::ipc::{AppEvent, EventSink};
use crate::open_files::{OpenFilesGate, paths_from_args};
use crate::picker::PickedPaths;

/// Label of the only window.
pub const MAIN_WINDOW: &str = "main";

/// Sends events to the main window.
struct TauriSink(AppHandle);

impl EventSink for TauriSink {
    fn emit(&self, event: AppEvent) {
        if let AppEvent::OpenFiles(files) = &event {
            let picked = self.0.state::<PickedPaths>();
            for path in &files.paths {
                picked.grant(Path::new(path), false);
            }
        }
        if let Err(e) = self.0.emit_to(MAIN_WINDOW, event.name(), &event) {
            log::debug!("couldn't emit {}: {e}", event.name());
        }
    }
}

fn focus_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window(MAIN_WINDOW) {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

/// Builds and runs the app.
pub fn run() {
    let gate = Arc::new(OpenFilesGate::default());
    let gate_for_instance = gate.clone();
    let gate_for_setup = gate.clone();
    let gate_for_load = gate.clone();

    let app = tauri::Builder::default()
        .manage(gate.clone())
        .manage(PickedPaths::default())
        // Must be first: a second launch exits before anything else starts.
        .plugin(tauri_plugin_single_instance::init(move |app, argv, cwd| {
            focus_main(app);
            let paths = paths_from_args(&argv, Path::new(&cwd));
            log::info!(
                "second launch: focused the window, forwarding {} file(s)",
                paths.len()
            );
            gate_for_instance.push(paths, &TauriSink(app.clone()));
        }))
        .plugin(
            tauri_plugin_log::Builder::new()
                .clear_targets()
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::Stdout,
                ))
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::LogDir {
                        file_name: Some("splitsound".into()),
                    },
                ))
                .level(log::LevelFilter::Info)
                .max_file_size(5_000_000)
                .rotation_strategy(tauri_plugin_log::RotationStrategy::KeepOne)
                .build(),
        )
        .plugin(
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(
                    tauri_plugin_window_state::StateFlags::SIZE
                        | tauri_plugin_window_state::StateFlags::POSITION
                        | tauri_plugin_window_state::StateFlags::MAXIMIZED,
                )
                .build(),
        )
        .plugin(picker::init())
        .plugin(drag::init())
        .setup(move |app| {
            let handle = app.handle().clone();
            app.manage(updates::Updates::new(&handle));
            let resolver = app.path();
            let paths = AppPaths::new(
                resolver.app_data_dir()?,
                resolver.app_config_dir()?,
                resolver.app_cache_dir()?,
                resolver.app_log_dir()?,
            );
            #[cfg(debug_assertions)]
            let smoke = smoke::Smoke::from_env()?;
            #[cfg(debug_assertions)]
            let mut paths = smoke.as_ref().map_or(paths, smoke::Smoke::paths);
            #[cfg(debug_assertions)]
            if let Some(models) = engine::core_adapter::models_dir_override() {
                paths.models_dir = models;
            }
            let ort = onnxruntime::resolve(
                std::env::current_exe().ok().as_deref(),
                resolver.resource_dir().ok().as_deref(),
            );
            if ort.is_none() {
                log::warn!("ONNX Runtime library not found (run scripts/fetch-onnxruntime.mjs)");
            }
            let engine = engine::select(EngineConfig {
                models_dir: paths.models_dir.clone(),
                cache_dir: paths.engine_cache_dir.clone(),
                onnxruntime: ort,
            });
            let sink: Arc<dyn EventSink> = Arc::new(TauriSink(handle.clone()));
            let core = AppCore::start_deferred(CoreOptions {
                paths,
                engine,
                events: sink.clone(),
                audio: CpalBackend::factory(),
                version: app.package_info().version.to_string(),
            })?;
            if let Some(dir) = core.get_settings().export_dir {
                app.state::<PickedPaths>().grant(Path::new(&dir), true);
            }
            app.manage(core);

            // Create the window only now, so every command finds its state.
            let config = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == MAIN_WINDOW)
                .cloned()
                .ok_or("the main window is missing from tauri.conf.json")?;
            let builder = WebviewWindowBuilder::from_config(&handle, &config)?;
            // macOS: the UI draws its own top bar (marked with
            // `data-tauri-drag-region`) under inset traffic lights. Windows
            // and Linux keep their native decorations.
            #[cfg(target_os = "macos")]
            let builder = builder
                .title_bar_style(tauri::TitleBarStyle::Overlay)
                .hidden_title(true)
                .traffic_light_position(tauri::LogicalPosition::new(18.0, 22.0));
            #[cfg(debug_assertions)]
            let builder = if let Some(smoke) = smoke {
                smoke.configure(
                    builder,
                    &app.state::<PickedPaths>(),
                    &gate_for_setup,
                    sink.as_ref(),
                )
            } else {
                builder
            };
            let window = builder.build()?;
            // Hidden until the saved size and position have been applied.
            window.show()?;
            log::info!("main window created");

            // Files passed on the command line of this first launch.
            let cwd = std::env::current_dir().unwrap_or_default();
            let args: Vec<String> = std::env::args().collect();
            gate_for_setup.push(paths_from_args(&args, &cwd), sink.as_ref());
            Ok(())
        })
        .on_page_load(move |webview, payload| {
            if webview.label() == MAIN_WINDOW && payload.event() == PageLoadEvent::Started {
                gate_for_load.reset();
            }
        })
        .on_window_event(|window, event| {
            if window.label() == MAIN_WINDOW
                && let WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) = event
            {
                grant_drop(&window.state::<PickedPaths>(), paths);
            }
        })
        .on_webview_event(|webview, event| {
            if webview.label() == MAIN_WINDOW
                && let WebviewEvent::DragDrop(DragDropEvent::Drop { paths, .. }) = event
            {
                grant_drop(&webview.state::<PickedPaths>(), paths);
            }
        })
        .invoke_handler(splitsound_handler!())
        .build(tauri::generate_context!())
        .expect("error while building SplitSound");

    app.run(move |handle, event| match event {
        #[cfg(target_os = "macos")]
        RunEvent::Opened { urls } => {
            let paths = urls
                .into_iter()
                .filter_map(|u| u.to_file_path().ok())
                .collect();
            gate.push(paths, &TauriSink(handle.clone()));
            focus_main(handle);
        }
        RunEvent::Exit => {
            if let Some(core) = handle.try_state::<Arc<AppCore>>() {
                core.shutdown();
            }
        }
        _ => {}
    });
}

fn grant_drop(picked: &PickedPaths, paths: &[std::path::PathBuf]) {
    for path in paths {
        picked.grant_dropped(path);
    }
}
