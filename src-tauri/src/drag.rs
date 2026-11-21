//! Dragging stems out of the window.
//!
//! This is an in-app plugin named `drag` with the same command and argument
//! shape as CrabNebula's `tauri-plugin-drag`, so the UI uses the unmodified
//! `@crabnebula/tauri-plugin-drag` JS package. It wraps the same native
//! crate (`drag`, by CrabNebula) but, unlike the upstream plugin, it:
//!
//! - accepts only files that `prepare_drag` produced (no arbitrary paths,
//!   no data items);
//! - always offers a copy, never a move;
//! - takes the preview image only as a `data:image/png` URL (size-capped),
//!   never as a file path, and falls back to the app icon.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::plugin::{Builder, TauriPlugin};
use tauri::{AppHandle, Manager, Runtime, Window};

use crate::app::AppCore;
use crate::ipc::{IpcError, IpcResult};

/// Preview used when the UI does not pass a PNG.
const DEFAULT_ICON: &[u8] = include_bytes!("../icons/Square71x71Logo.png");
const MAX_ICON_BYTES: usize = 2 * 1024 * 1024;

/// `item` from the JS package: a list of absolute paths (data items are
/// accepted by the type but refused).
#[derive(Deserialize)]
#[serde(untagged)]
pub enum DragItem {
    Files(Vec<PathBuf>),
    Other(#[allow(dead_code)] serde::de::IgnoredAny),
}

#[derive(Serialize, Clone)]
pub struct CallbackResult {
    result: drag::DragResult,
    #[serde(rename = "cursorPos")]
    cursor_pos: drag::CursorPosition,
}

fn icon_bytes(image: Option<serde_json::Value>) -> Vec<u8> {
    if let Some(serde_json::Value::String(s)) = image
        && let Some(b64) = s.strip_prefix("data:image/png;base64,")
        && b64.len() <= MAX_ICON_BYTES * 4 / 3 + 4
        && let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64)
        && crate::artwork::sniff(&bytes) == Some(crate::artwork::ImageKind::Png)
        && png_size(&bytes).is_some_and(|(w, h)| w >= MIN_ICON_PX && h >= MIN_ICON_PX)
    {
        return bytes;
    }
    DEFAULT_ICON.to_vec()
}

/// Smaller previews (like a 1x1 placeholder) would make the drag invisible.
const MIN_ICON_PX: u32 = 16;

/// Width and height from a PNG's IHDR chunk.
fn png_size(png: &[u8]) -> Option<(u32, u32)> {
    if png.get(12..16)? != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(png.get(16..20)?.try_into().ok()?);
    let h = u32::from_be_bytes(png.get(20..24)?.try_into().ok()?);
    Some((w, h))
}

#[tauri::command]
async fn start_drag<R: Runtime>(
    app: AppHandle<R>,
    window: Window<R>,
    item: DragItem,
    image: Option<serde_json::Value>,
    options: Option<serde_json::Value>,
    on_event: Channel<CallbackResult>,
) -> IpcResult<()> {
    let _ = options; // mode is ignored: always copy
    let core = app.state::<Arc<AppCore>>().inner().clone();
    let files = match item {
        DragItem::Files(f) if !f.is_empty() => core.check_drag_paths(&f)?,
        _ => {
            return Err(IpcError::not_found(
                "Only stems prepared by SplitSound can be dragged out.",
            ));
        }
    };
    let icon = icon_bytes(image);
    let (tx, rx) = mpsc::channel();
    app.run_on_main_thread(move || {
        #[cfg(target_os = "linux")]
        let target = window.gtk_window();
        #[cfg(not(target_os = "linux"))]
        let target: tauri::Result<Window<R>> = Ok(window.clone());
        let r = match target {
            Ok(w) => drag::start_drag(
                &w,
                drag::DragItem::Files(files),
                drag::Image::Raw(icon),
                move |result, cursor_pos| {
                    let _ = on_event.send(CallbackResult { result, cursor_pos });
                },
                drag::Options {
                    skip_animatation_on_cancel_or_failure: false,
                    mode: drag::DragMode::Copy,
                },
            )
            .map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        let _ = tx.send(r);
    })
    .map_err(|e| IpcError::internal(format!("Couldn't start dragging ({e}).")))?;
    match rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(IpcError::internal(format!(
            "Couldn't start dragging ({e})."
        ))),
        Err(_) => Err(IpcError::internal("Couldn't start dragging.")),
    }
}

/// The `drag` plugin. Its only command is `start_drag`, allowed by the
/// `drag:allow-start-drag` permission.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("drag")
        .invoke_handler(tauri::generate_handler![start_drag])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icon_falls_back_to_the_app_icon() {
        assert_eq!(icon_bytes(None), DEFAULT_ICON);
        assert_eq!(
            icon_bytes(Some(serde_json::json!("/etc/passwd"))),
            DEFAULT_ICON
        );
        assert_eq!(
            icon_bytes(Some(serde_json::json!("data:image/png;base64,PHN2Zz4="))),
            DEFAULT_ICON,
            "an SVG labelled as PNG is refused"
        );
        let png = base64::engine::general_purpose::STANDARD.encode(DEFAULT_ICON);
        assert_eq!(
            icon_bytes(Some(serde_json::json!(format!(
                "data:image/png;base64,{png}"
            )))),
            DEFAULT_ICON
        );
        assert_eq!(&DEFAULT_ICON[1..4], b"PNG");
        assert_eq!(png_size(DEFAULT_ICON), Some((71, 71)));
        // The UI's transparent 1x1 placeholder is replaced by the app icon.
        let one_px = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        assert_eq!(icon_bytes(Some(serde_json::json!(one_px))), DEFAULT_ICON);
    }
}
