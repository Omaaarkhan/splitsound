//! Build script: the ONNX Runtime check and Tauri's code generation.
//!
//! ONNX Runtime: the bundler config ships
//! `resources/onnxruntime/<lib>` (see `tauri.<os>.conf.json`), which
//! `scripts/fetch-onnxruntime.mjs` puts there.
//!
//! - Release builds fail with a clear message if the library is missing or
//!   is not a real library (set `SPLITSOUND_ALLOW_MISSING_ONNXRUNTIME=1` to
//!   build a release without it, e.g. for a clippy run).
//! - Other builds create an empty placeholder when it is missing, so `cargo
//!   build`, `tauri dev` and debug bundles work before fetching. The app
//!   ignores an empty file at runtime and uses the mock engine.
//!
//! Permissions: every app command is declared here so it gets an
//! `allow-<command>` permission; `capabilities/main.json` grants them
//! one by one. The in-app `drag` plugin is declared the same way.

use std::path::PathBuf;

const COMMANDS: &[&str] = &[
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
    "check_for_update",
    "install_update",
];

fn onnxruntime() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let name = match os.as_str() {
        "macos" => "libonnxruntime.dylib",
        "windows" => "onnxruntime.dll",
        _ => "libonnxruntime.so",
    };
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("resources")
        .join("onnxruntime");
    let lib = dir.join(name);
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rerun-if-env-changed=SPLITSOUND_ALLOW_MISSING_ONNXRUNTIME");
    let len = std::fs::metadata(&lib).map(|m| m.len()).unwrap_or(0);
    let release = std::env::var("PROFILE").as_deref() == Ok("release");
    let allow_missing = std::env::var("SPLITSOUND_ALLOW_MISSING_ONNXRUNTIME").as_deref() == Ok("1");
    if len < 64 * 1024 {
        if release && !allow_missing {
            panic!(
                "\n\nONNX Runtime is missing: {} is {}.\n\
                 Run `node scripts/fetch-onnxruntime.mjs` before a release build \
                 (or set SPLITSOUND_ALLOW_MISSING_ONNXRUNTIME=1 to build without it).\n",
                lib.display(),
                if lib.exists() {
                    "an empty placeholder"
                } else {
                    "not there"
                }
            );
        }
        if !lib.exists() {
            std::fs::create_dir_all(&dir).expect("create resources/onnxruntime");
            std::fs::write(&lib, b"").expect("write the ONNX Runtime placeholder");
            println!(
                "cargo:warning=ONNX Runtime not fetched; created an empty placeholder at {} (the app will use the mock engine)",
                lib.display()
            );
        }
    }
}

fn main() {
    onnxruntime();
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(COMMANDS))
            .plugin(
                "drag",
                tauri_build::InlinedPlugin::new()
                    .commands(&["start_drag"])
                    .default_permission(tauri_build::DefaultPermissionRule::AllowAllCommands),
            ),
    )
    .expect("failed to run the Tauri build script");
}
