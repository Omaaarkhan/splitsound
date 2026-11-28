//! The session library: one folder per imported song.
//!
//! ```text
//! <app data>/library/
//!   <id>/                 id = 32 lowercase hex digits
//!     session.json        written atomically
//!     mix.wav             the decoded original (32-bit float, 44.1 kHz stereo)
//!     artwork.png|jpg     only if the file had PNG or JPEG cover art
//!     stems/<stem>.wav    after a split
//!     stems.partial/      while a split is writing (removed on start)
//!   .staging-<id>/        an import in progress (removed on start)
//!   .trash-<id>/          a delete in progress (removed on start)
//! ```
//!
//! Session ids are generated here and validated before any path is built
//! from one. Stem paths are always recomputed from the id; the paths stored
//! in `session.json` are informational only.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

use serde::{Deserialize, Serialize};

use crate::ipc::{
    AnalysisStatus, IpcError, IpcResult, Session, SessionStatus, SessionSummary, SplitSummary,
    StemId,
};
use crate::util::{self, is_valid_session_id, lock};

const SESSION_FILE: &str = "session.json";
const STORE_FORMAT: u32 = 1;
/// A session.json larger than this is not ours.
const MAX_SESSION_JSON: u64 = 8 * 1024 * 1024;

/// What is persisted in `session.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Stored {
    format: u32,
    session: Session,
    /// Summary of the split whose stems are on disk while a new split is
    /// queued or running (restored on cancel or failure).
    #[serde(default)]
    previous_split: Option<SplitSummary>,
    #[serde(default)]
    size_bytes: Option<u64>,
}

struct Entry {
    session: Session,
    previous_split: Option<SplitSummary>,
    size: u64,
}

/// What [`Library::open`] found and fixed.
#[derive(Debug, Default)]
pub struct LoadReport {
    /// Sessions left `queued` or `splitting` by a crash, now reset.
    pub repaired: Vec<String>,
    /// Sessions whose instrument scan never finished.
    pub pending_scans: Vec<String>,
    /// Folders that could not be read (left untouched).
    pub skipped: Vec<PathBuf>,
}

pub struct Library {
    root: PathBuf,
    entries: Mutex<BTreeMap<String, Entry>>,
    ready_flag: AtomicBool,
    ready_gate: (Mutex<bool>, Condvar),
}

fn move_to_gc(path: &Path, gc_batch: &mut Option<PathBuf>, gc_parent: Option<&Path>) {
    if let Some(parent) = gc_parent {
        let batch = gc_batch.get_or_insert_with(|| {
            let dir = parent.join(format!(".lib-gc-{}", &util::new_session_id()[..8]));
            let _ = fs::create_dir_all(&dir);
            dir
        });
        let dst = batch.join(util::new_session_id());
        if util::rename_retry(path, &dst).is_ok() {
            return;
        }
    }
    let _ = fs::remove_dir_all(path);
}

fn spawn_gc(gc_batch: Option<PathBuf>) {
    if let Some(dir) = gc_batch {
        let _ = std::thread::Builder::new()
            .name("splitsound-lib-gc".into())
            .spawn(move || {
                let _ = util::remove_dir_all_retry(&dir);
            });
    }
}

fn invalid_id() -> IpcError {
    IpcError::not_found("That song doesn't exist.")
}

/// Canonical path of a stem file inside a session folder.
pub fn stem_path(dir: &Path, stem: StemId) -> PathBuf {
    dir.join("stems").join(format!("{}.wav", stem.as_str()))
}

/// Restores the files referenced by the previous session metadata when a
/// crash or failed metadata write interrupts the swap.
pub(crate) fn rollback_stems(dir: &Path, had_stems: bool) -> io::Result<()> {
    let old = dir.join("stems.old");
    let current = dir.join("stems");
    if old.is_dir() || !had_stems {
        match util::remove_dir_all_retry(&current) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        if old.is_dir() {
            util::rename_retry(&old, &current)?;
        }
    }
    Ok(())
}

impl Library {
    /// Loads every session under `root`, cleaning up after interrupted
    /// imports, deletes and splits, and resetting sessions a crash left in
    /// `queued` or `splitting`.
    pub fn open(root: PathBuf) -> io::Result<(Self, LoadReport)> {
        Self::open_with_gc_dir(root, None)
    }

    /// [`Library::open`] that offloads removal of `.staging-*`, `.trash-*`,
    /// `stems.partial`, and `stems.old` directories to a background thread via
    /// O(1) rename into `gc_dir` when provided on the same volume.
    pub fn open_with_gc_dir(root: PathBuf, gc_dir: Option<&Path>) -> io::Result<(Self, LoadReport)> {
        let (entries, report) = Self::scan_root(&root, gc_dir)?;
        Ok((
            Library {
                root,
                entries: Mutex::new(entries),
                ready_flag: AtomicBool::new(true),
                ready_gate: (Mutex::new(true), Condvar::new()),
            },
            report,
        ))
    }

    /// Creates an unhydrated `Library` handle immediately so window creation
    /// can overlap with scanning the library directory on a background thread.
    /// Any query waits cleanly until [`Library::hydrate`] completes.
    pub fn new_deferred(root: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(Library {
            root,
            entries: Mutex::new(BTreeMap::new()),
            ready_flag: AtomicBool::new(false),
            ready_gate: (Mutex::new(false), Condvar::new()),
        })
    }

    /// Hydrates a deferred library in the background and unblocks waiting
    /// callers even if scanning encountered an error.
    pub fn hydrate(&self, gc_dir: Option<&Path>) -> io::Result<LoadReport> {
        let res = Self::scan_root(&self.root, gc_dir);
        match res {
            Ok((loaded, report)) => {
                *lock(&self.entries) = loaded;
                self.mark_ready();
                Ok(report)
            }
            Err(e) => {
                self.mark_ready();
                Err(e)
            }
        }
    }

    fn mark_ready(&self) {
        let mut ready = lock(&self.ready_gate.0);
        *ready = true;
        self.ready_flag.store(true, Ordering::Release);
        self.ready_gate.1.notify_all();
    }

    #[inline]
    fn wait_ready(&self) {
        if self.ready_flag.load(Ordering::Acquire) {
            return;
        }
        let mut ready = lock(&self.ready_gate.0);
        while !*ready {
            ready = self
                .ready_gate
                .1
                .wait(ready)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    fn scan_root(
        root: &Path,
        gc_dir: Option<&Path>,
    ) -> io::Result<(BTreeMap<String, Entry>, LoadReport)> {
        fs::create_dir_all(root)?;
        let mut report = LoadReport::default();
        let mut entries = BTreeMap::new();
        let mut gc_batch: Option<PathBuf> = None;
        for item in fs::read_dir(root)?.flatten() {
            let name = item.file_name().to_string_lossy().into_owned();
            let path = item.path();
            let Ok(ft) = item.file_type() else { continue };
            if name.starts_with(".staging-") || name.starts_with(".trash-") {
                if ft.is_dir() {
                    move_to_gc(&path, &mut gc_batch, gc_dir);
                }
                continue;
            }
            if !ft.is_dir() || !is_valid_session_id(&name) {
                continue;
            }
            match Self::load_one(&path, &name, &mut gc_batch, gc_dir) {
                Ok((mut entry, repaired)) => {
                    if repaired {
                        report.repaired.push(name.clone());
                        entry.size = util::dir_size(&path);
                        let stored = Stored {
                            format: STORE_FORMAT,
                            session: entry.session.clone(),
                            previous_split: entry.previous_split.clone(),
                            size_bytes: Some(entry.size),
                        };
                        if let Err(e) = write_stored(&path, &stored) {
                            log::warn!("library: can't save repaired session {name}: {e}");
                        }
                    }
                    if entry.session.analysis == AnalysisStatus::Pending {
                        report.pending_scans.push(name.clone());
                    }
                    entries.insert(name, entry);
                }
                Err(e) => {
                    log::warn!("library: skipping {}: {e}", path.display());
                    report.skipped.push(path);
                }
            }
        }
        spawn_gc(gc_batch);
        Ok((entries, report))
    }

    fn load_one(
        dir: &Path,
        id: &str,
        gc_batch: &mut Option<PathBuf>,
        gc_dir: Option<&Path>,
    ) -> io::Result<(Entry, bool)> {
        let bytes = util::read_bounded(&dir.join(SESSION_FILE), MAX_SESSION_JSON)?;
        let stored: Stored = serde_json::from_slice(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut s = stored.session;
        let mut prev = stored.previous_split;
        if s.id != id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "session id doesn't match its folder",
            ));
        }
        if s.status == SessionStatus::Splitting {
            rollback_stems(dir, !s.stems.is_empty())?;
        }
        // Single directory pass to clean leftovers and check for mix.wav.
        let mut has_mix = false;
        for f in fs::read_dir(dir)?.flatten() {
            let fname = f.file_name();
            let Some(name) = fname.to_str() else {
                continue;
            };
            if name == "mix.wav" {
                has_mix = f.file_type().is_ok_and(|t| t.is_file());
            } else if name == "stems.partial" || name == "stems.old" {
                move_to_gc(&f.path(), gc_batch, gc_dir);
            } else if util::is_atomic_temp(name) {
                let _ = fs::remove_file(f.path());
            }
        }
        for st in &mut s.stems {
            st.path = stem_path(dir, st.id).to_string_lossy().into_owned();
        }
        let mut repaired = false;
        if matches!(s.status, SessionStatus::Queued | SessionStatus::Splitting) {
            s.status = if s.stems.is_empty() {
                SessionStatus::Imported
            } else {
                SessionStatus::Ready
            };
            s.split = prev.take();
            s.error = None;
            repaired = true;
        }
        if !s.stems.is_empty() && s.stems.iter().any(|st| !Path::new(&st.path).is_file()) {
            s.status = SessionStatus::Failed;
            s.stems.clear();
            s.split = None;
            s.error = Some(
                "Some of this song's stems are missing. Split it again to recreate them.".into(),
            );
            repaired = true;
        }
        if !has_mix {
            s.status = SessionStatus::Failed;
            s.error =
                Some("This song's audio is missing. Delete it and import the file again.".into());
            repaired = true;
        }
        let size = stored.size_bytes.unwrap_or_else(|| util::dir_size(dir));
        Ok((
            Entry {
                session: s,
                previous_split: prev,
                size,
            },
            repaired,
        ))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Folder of session `id`, after validating the id. Does not check that
    /// the session exists.
    pub fn dir(&self, id: &str) -> IpcResult<PathBuf> {
        if !is_valid_session_id(id) {
            return Err(invalid_id());
        }
        Ok(self.root.join(id))
    }

    /// A fresh staging folder for an import.
    pub fn staging_dir(&self, id: &str) -> IpcResult<PathBuf> {
        if !is_valid_session_id(id) {
            return Err(invalid_id());
        }
        Ok(self.root.join(format!(".staging-{id}")))
    }

    pub fn get(&self, id: &str) -> IpcResult<Session> {
        if !is_valid_session_id(id) {
            return Err(invalid_id());
        }
        self.wait_ready();
        lock(&self.entries)
            .get(id)
            .map(|e| e.session.clone())
            .ok_or_else(invalid_id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.wait_ready();
        lock(&self.entries).contains_key(id)
    }

    pub fn ids(&self) -> Vec<String> {
        self.wait_ready();
        lock(&self.entries).keys().cloned().collect()
    }

    /// All sessions, newest first.
    pub fn summaries(&self) -> Vec<SessionSummary> {
        self.wait_ready();
        let entries = lock(&self.entries);
        let mut v: Vec<SessionSummary> = entries
            .values()
            .map(|e| {
                let s = &e.session;
                SessionSummary {
                    id: s.id.clone(),
                    created_at: s.created_at.clone(),
                    title: s.track.title.clone(),
                    artist: s.track.artist.clone(),
                    duration_sec: s.track.duration_sec,
                    status: s.status,
                    stem_ids: s.stems.iter().map(|st| st.id).collect(),
                    has_artwork: s.track.has_artwork,
                    size_bytes: e.size,
                }
            })
            .collect();
        v.sort_by(|a, b| {
            b.created_at
                .cmp(&a.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        v
    }

    pub fn total_bytes(&self) -> u64 {
        self.wait_ready();
        lock(&self.entries).values().map(|e| e.size).sum()
    }

    /// Moves a finished import from its staging folder into the library.
    pub fn insert(&self, session: Session, staging: &Path) -> IpcResult<()> {
        self.wait_ready();
        let dir = self.dir(&session.id)?;
        let size = util::dir_size(staging);
        let stored = Stored {
            format: STORE_FORMAT,
            session: session.clone(),
            previous_split: None,
            size_bytes: Some(size),
        };
        write_stored(staging, &stored).map_err(|e| IpcError::io("save the song", &e))?;
        util::rename_retry(staging, &dir).map_err(|e| IpcError::io("save the song", &e))?;
        lock(&self.entries).insert(
            session.id.clone(),
            Entry {
                session,
                previous_split: None,
                size,
            },
        );
        Ok(())
    }

    /// Changes a session and saves it. `f` gets the session and the summary
    /// of the split whose stems are on disk while another one is pending.
    pub fn update<R>(
        &self,
        id: &str,
        f: impl FnOnce(&mut Session, &mut Option<SplitSummary>) -> R,
    ) -> IpcResult<(Session, R)> {
        self.update_inner(id, None, f)
    }

    /// Like [`Library::update`], also recording a refreshed `size_bytes` in the
    /// same atomic metadata write (avoiding a second `fsync`).
    pub fn update_with_size<R>(
        &self,
        id: &str,
        size: u64,
        f: impl FnOnce(&mut Session, &mut Option<SplitSummary>) -> R,
    ) -> IpcResult<(Session, R)> {
        self.update_inner(id, Some(size), f)
    }

    fn update_inner<R>(
        &self,
        id: &str,
        new_size: Option<u64>,
        f: impl FnOnce(&mut Session, &mut Option<SplitSummary>) -> R,
    ) -> IpcResult<(Session, R)> {
        self.wait_ready();
        let dir = self.dir(id)?;
        let mut entries = lock(&self.entries);
        let entry = entries.get_mut(id).ok_or_else(invalid_id)?;
        let mut session = entry.session.clone();
        let mut prev = entry.previous_split.clone();
        let r = f(&mut session, &mut prev);
        let size = new_size.unwrap_or(entry.size);
        let stored = Stored {
            format: STORE_FORMAT,
            session: session.clone(),
            previous_split: prev.clone(),
            size_bytes: Some(size),
        };
        write_stored(&dir, &stored).map_err(|e| IpcError::io("save the song", &e))?;
        entry.session = session.clone();
        entry.previous_split = prev;
        entry.size = size;
        Ok((session, r))
    }

    /// Recomputes the disk usage of a session.
    pub fn refresh_size(&self, id: &str) {
        self.wait_ready();
        let Ok(dir) = self.dir(id) else { return };
        let size = util::dir_size(&dir);
        let stored = {
            let mut entries = lock(&self.entries);
            let Some(e) = entries.get_mut(id) else {
                return;
            };
            e.size = size;
            Stored {
                format: STORE_FORMAT,
                session: e.session.clone(),
                previous_split: e.previous_split.clone(),
                size_bytes: Some(size),
            }
        };
        let _ = write_stored(&dir, &stored);
    }

    /// Deletes a session folder. The folder is first renamed out of the way
    /// so a failure half-way never leaves a broken session behind.
    pub fn remove(&self, id: &str) -> IpcResult<()> {
        self.wait_ready();
        let dir = self.dir(id)?;
        let mut entries = lock(&self.entries);
        if !entries.contains_key(id) {
            return Err(invalid_id());
        }
        // Use a unique trash name so a leftover `.trash-*` directory from an
        // earlier cleanup never collides on Windows `MoveFileExW`.
        let trash = self
            .root
            .join(format!(".trash-{id}-{}", &util::new_session_id()[..8]));
        match util::rename_retry(&dir, &trash) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(IpcError::io("delete the song", &e)),
        }
        entries.remove(id);
        drop(entries);
        if let Err(e) = util::remove_dir_all_retry(&trash)
            && e.kind() != io::ErrorKind::NotFound
        {
            log::warn!(
                "library: will retry removing {} on startup: {e}",
                trash.display()
            );
        }
        Ok(())
    }
}

fn write_stored(dir: &Path, stored: &Stored) -> io::Result<()> {
    let bytes = serde_json::to_vec(stored).map_err(io::Error::other)?;
    util::write_atomic(&dir.join(SESSION_FILE), &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::{SplitMode, StemInfo, TrackInfo};

    fn sample_session(id: &str) -> Session {
        Session {
            id: id.into(),
            created_at: util::now_iso8601(),
            track: TrackInfo {
                path: "/music/a.wav".into(),
                file_name: "a.wav".into(),
                title: "A".into(),
                artist: Some("B".into()),
                album: None,
                duration_sec: 1.0,
                sample_rate: 44_100,
                channels: 2,
                codec: "WAV".into(),
                bitrate_kbps: None,
                file_size_bytes: 1,
                has_artwork: false,
            },
            status: SessionStatus::Imported,
            analysis: AnalysisStatus::Pending,
            detections: vec![],
            suggested_mode: SplitMode::Band,
            split: None,
            stems: vec![],
            error: None,
        }
    }

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!("ss-lib-{}", util::new_session_id()))
    }

    fn import(lib: &Library, s: &Session) {
        let staging = lib.staging_dir(&s.id).unwrap();
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("mix.wav"), b"x").unwrap();
        lib.insert(s.clone(), &staging).unwrap();
    }

    #[test]
    fn insert_update_reload_remove() {
        let root = temp_root();
        let (lib, _) = Library::open(root.clone()).unwrap();
        let id = util::new_session_id();
        import(&lib, &sample_session(&id));
        assert_eq!(lib.summaries().len(), 1);
        assert!(lib.summaries()[0].size_bytes > 0);
        lib.update(&id, |s, _| s.analysis = AnalysisStatus::Done)
            .unwrap();
        drop(lib);
        let (lib, report) = Library::open(root.clone()).unwrap();
        assert_eq!(lib.get(&id).unwrap().analysis, AnalysisStatus::Done);
        assert!(report.pending_scans.is_empty());
        lib.remove(&id).unwrap();
        assert!(lib.get(&id).is_err());
        assert!(!root.join(&id).exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_ids_never_touch_the_filesystem() {
        let root = temp_root();
        let (lib, _) = Library::open(root.clone()).unwrap();
        for bad in ["..", "../x", "ABCDEF00000000000000000000000000", "", "a/b"] {
            assert!(lib.dir(bad).is_err());
            assert!(lib.get(bad).is_err());
            assert!(lib.remove(bad).is_err());
            assert!(lib.update(bad, |_, _| ()).is_err());
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repairs_sessions_left_mid_split_by_a_crash() {
        let root = temp_root();
        let (lib, _) = Library::open(root.clone()).unwrap();
        let fresh = util::new_session_id();
        let resplit = util::new_session_id();
        import(&lib, &sample_session(&fresh));
        import(&lib, &sample_session(&resplit));
        let summary = SplitSummary {
            mode: SplitMode::Vocals,
            quality: crate::ipc::Quality::Fast,
            finished_at: util::now_iso8601(),
            elapsed_sec: 1.0,
            model_ids: vec![],
        };
        lib.update(&fresh, |s, _| s.status = SessionStatus::Splitting)
            .unwrap();
        let dir = lib.dir(&resplit).unwrap();
        fs::create_dir_all(dir.join("stems")).unwrap();
        fs::create_dir_all(dir.join("stems.partial")).unwrap();
        fs::write(dir.join(".session.json.tmp-abc"), b"junk").unwrap();
        for st in [StemId::Vocals, StemId::Instrumental] {
            fs::write(stem_path(&dir, st), b"x").unwrap();
        }
        lib.update(&resplit, |s, prev| {
            s.status = SessionStatus::Queued;
            s.stems = [StemId::Vocals, StemId::Instrumental]
                .iter()
                .map(|id| StemInfo {
                    id: *id,
                    path: "/somewhere/else.wav".into(),
                    present: true,
                    active_fraction: 1.0,
                    peak_db: 0.0,
                    rms_db: 0.0,
                    activity: vec![],
                    contains: vec![],
                })
                .collect();
            *prev = Some(summary.clone());
            s.split = None;
        })
        .unwrap();
        fs::create_dir_all(root.join(".staging-whatever")).unwrap();
        drop(lib);

        let (lib, report) = Library::open(root.clone()).unwrap();
        assert_eq!(report.repaired.len(), 2);
        assert_eq!(report.pending_scans.len(), 2);
        assert_eq!(lib.get(&fresh).unwrap().status, SessionStatus::Imported);
        let r = lib.get(&resplit).unwrap();
        assert_eq!(r.status, SessionStatus::Ready);
        assert_eq!(r.split, Some(summary));
        assert_eq!(
            r.stems[0].path,
            stem_path(&dir, StemId::Vocals).to_string_lossy()
        );
        assert!(!dir.join("stems.partial").exists());
        assert!(!dir.join(".session.json.tmp-abc").exists());
        assert!(!root.join(".staging-whatever").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unreadable_sessions_are_skipped_not_deleted() {
        let root = temp_root();
        let id = util::new_session_id();
        fs::create_dir_all(root.join(&id)).unwrap();
        fs::write(root.join(&id).join(SESSION_FILE), b"{not json").unwrap();
        let (lib, report) = Library::open(root.clone()).unwrap();
        assert!(lib.ids().is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert!(root.join(&id).join(SESSION_FILE).exists());
        fs::remove_dir_all(root).unwrap();
    }
}
