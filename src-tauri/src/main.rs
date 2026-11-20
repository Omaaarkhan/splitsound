// Prevents an extra console window on Windows in release. Do not remove.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// Removes the current working directory from the Windows DLL search order
/// before Tauri, WebView2, or ONNX Runtime loads any dynamic libraries.
#[cfg(windows)]
fn harden_dll_search_path() {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDllDirectoryW(lp_path_name: *const u16) -> i32;
    }
    const EMPTY: [u16; 1] = [0];
    // SAFETY: `EMPTY` is a valid NUL-terminated UTF-16 string; passing `""`
    // removes the current working directory from the default DLL search order.
    unsafe {
        let _ = SetDllDirectoryW(EMPTY.as_ptr());
    }
}

fn main() {
    #[cfg(windows)]
    harden_dll_search_path();
    splitsound_desktop_lib::run()
}
