//! Files the operating system asks the app to open: "Open with" (macOS
//! `RunEvent::Opened`), command-line arguments (Windows and Linux file
//! associations launch the app with the path), and paths forwarded from a
//! second launch by the single-instance plugin.
//!
//! They are delivered to the UI as `app:open-files`. Paths that arrive
//! before the UI is ready are held and sent, in one event, once it is.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::ipc::{AppEvent, EventSink, OpenFiles};
use crate::util::lock;

#[derive(Default)]
struct Gate {
    ready: bool,
    pending: Vec<String>,
}

/// Holds paths until the UI can receive them.
#[derive(Default)]
pub struct OpenFilesGate(Mutex<Gate>);

impl OpenFilesGate {
    /// Queues or sends `paths`. Only existing regular files are forwarded.
    pub fn push(&self, paths: Vec<PathBuf>, sink: &dyn EventSink) {
        let mut clean: Vec<String> = Vec::new();
        for p in paths {
            if p.is_absolute()
                && p.is_file()
                && let Ok(c) = crate::app::dunce_canonicalize(&p)
            {
                let s = c.to_string_lossy().into_owned();
                if !clean.contains(&s) {
                    clean.push(s);
                }
            }
        }
        if clean.is_empty() {
            return;
        }
        let mut g = lock(&self.0);
        if g.ready {
            drop(g);
            sink.emit(AppEvent::OpenFiles(OpenFiles { paths: clean }));
        } else {
            for s in clean {
                if !g.pending.contains(&s) {
                    g.pending.push(s);
                }
            }
        }
    }

    /// Marks the UI ready and sends anything held.
    pub fn set_ready(&self, sink: &dyn EventSink) -> bool {
        let mut g = lock(&self.0);
        let first = !g.ready;
        g.ready = true;
        let pending = std::mem::take(&mut g.pending);
        if !pending.is_empty() {
            sink.emit(AppEvent::OpenFiles(OpenFiles { paths: pending }));
        }
        first
    }

    pub(crate) fn reset(&self) {
        lock(&self.0).ready = false;
    }
}

/// File paths among command-line arguments (the first one, the program,
/// is skipped; flags are ignored). Relative paths resolve against `cwd`.
pub fn paths_from_args(args: &[String], cwd: &Path) -> Vec<PathBuf> {
    args.iter()
        .skip(1)
        .filter(|a| !a.is_empty() && !a.starts_with('-'))
        .map(|a| {
            let p = PathBuf::from(a);
            if p.is_absolute() { p } else { cwd.join(p) }
        })
        .filter(|p| p.is_file())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::RecordingSink;

    #[test]
    fn holds_until_ready_then_sends_once() {
        let dir = std::env::temp_dir().join(format!("ss-open-{}", crate::util::new_session_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.mp3");
        let b = dir.join("b.wav");
        std::fs::write(&a, b"x").unwrap();
        std::fs::write(&b, b"x").unwrap();
        let sink = RecordingSink::default();
        let gate = OpenFilesGate::default();
        gate.push(vec![a.clone(), dir.join("missing.wav"), dir.clone()], &sink);
        gate.push(vec![a.clone(), b.clone()], &sink);
        assert!(sink.snapshot().is_empty());
        gate.set_ready(&sink);
        let ev = sink.take();
        assert_eq!(ev.len(), 1);
        let AppEvent::OpenFiles(OpenFiles { paths }) = &ev[0] else {
            panic!()
        };
        assert_eq!(paths.len(), 2, "deduplicated, missing and folders dropped");
        gate.push(vec![b.clone()], &sink);
        assert_eq!(sink.take().len(), 1, "sent at once when ready");
        gate.set_ready(&sink);
        assert!(sink.take().is_empty());

        let args = vec![
            "splitsound".to_string(),
            "--flag".into(),
            "a.mp3".into(),
            b.to_string_lossy().into_owned(),
            "nope.wav".into(),
        ];
        assert_eq!(paths_from_args(&args, &dir), vec![dir.join("a.mp3"), b]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
