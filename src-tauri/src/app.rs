//! The shell's services, independent of Tauri: every command is a method
//! here, the Tauri commands in `commands.rs` are thin wrappers. This is what
//! the integration tests drive.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::artwork;
use crate::audio::output::BackendFactory;
use crate::audio::peaks::MAX_BUCKETS;
use crate::audio::player::Player;
use crate::engine::{
    CancelToken, Engine, EngineProgress, EngineResult, ExportItem, ExportJob, ScanOutcome,
    SplitRequest,
};
use crate::ipc::*;
use crate::library::{Library, stem_path};
use crate::settings::SettingsStore;
use crate::util::{self, lock, now_iso8601, sanitize_file_stem, unique_path};
use crate::worker::{Cancelled, Job, Worker};

/// Largest audio file accepted for import (4 GiB).
pub const MAX_IMPORT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Largest model file accepted for manual import (8 GiB).
pub const MAX_MODEL_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_TAG_CHARS: usize = 1024;

/// Folders the app owns.
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub log_dir: PathBuf,
    /// `<data>/library`
    pub library_dir: PathBuf,
    /// `<data>/models`
    pub models_dir: PathBuf,
    /// `<cache>/drag`, emptied on every start.
    pub drag_dir: PathBuf,
    /// `<cache>/engine`
    pub engine_cache_dir: PathBuf,
    /// `<config>/settings.json`
    pub settings_file: PathBuf,
}

impl AppPaths {
    pub fn new(
        data_dir: PathBuf,
        config_dir: PathBuf,
        cache_dir: PathBuf,
        log_dir: PathBuf,
    ) -> Self {
        AppPaths {
            library_dir: data_dir.join("library"),
            models_dir: data_dir.join("models"),
            drag_dir: cache_dir.join("drag"),
            engine_cache_dir: cache_dir.join("engine"),
            settings_file: config_dir.join("settings.json"),
            data_dir,
            config_dir,
            cache_dir,
            log_dir,
        }
    }

    /// Everything under one folder (tests).
    pub fn under(root: &Path) -> Self {
        Self::new(
            root.join("data"),
            root.join("config"),
            root.join("cache"),
            root.join("logs"),
        )
    }

    /// Creates the folders and empties the drag folder.
    pub fn prepare(&self) -> io::Result<()> {
        if self.drag_dir.exists() {
            fs::remove_dir_all(&self.drag_dir)?;
        }
        for d in [
            &self.data_dir,
            &self.config_dir,
            &self.cache_dir,
            &self.log_dir,
            &self.library_dir,
            &self.drag_dir,
        ] {
            fs::create_dir_all(d)?;
        }
        Ok(())
    }
}

/// A queued split.
#[derive(Debug, Clone)]
pub struct SplitJob {
    pub session_id: String,
    pub options: SplitOptions,
    pub accept_restricted: bool,
}

impl Job for SplitJob {
    fn key(&self) -> &str {
        &self.session_id
    }
}

/// Everything [`AppCore::start`] needs.
pub struct CoreOptions {
    pub paths: AppPaths,
    pub engine: EngineResult<Arc<dyn Engine>>,
    pub events: Arc<dyn EventSink>,
    pub audio: BackendFactory,
    pub version: String,
}

pub struct AppCore {
    pub paths: AppPaths,
    engine: EngineResult<Arc<dyn Engine>>,
    events: Arc<dyn EventSink>,
    pub library: Library,
    pub player: Player,
    pub settings: SettingsStore,
    splits: Worker<SplitJob>,
    split_actions: Mutex<()>,
    scans: Worker<String>,
    exports: Mutex<()>,
    drag_copies: Mutex<()>,
    downloads: Mutex<Vec<CancelToken>>,
    download_progress: Mutex<HashMap<String, (f64, Instant)>>,
    /// Canonical paths handed to the UI (exports, drag files, export folders).
    handed_out: Mutex<HashSet<PathBuf>>,
    version: String,
}

fn terms_required(models: &[&ModelStatus]) -> IpcError {
    let names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
    let (verb, pron) = if names.len() == 1 {
        ("needs", "its")
    } else {
        ("need", "their")
    };
    IpcError::new(
        ErrorCode::TermsRequired,
        format!(
            "{} {verb} your agreement to {pron} terms before downloading.",
            names.join(" and ")
        ),
    )
}

fn file_not_found() -> IpcError {
    IpcError::new(ErrorCode::FileNotFound, "That file couldn't be found.")
}

/// Validates a path the user picked (dialog, drop, "Open with"): absolute,
/// existing regular file (not a folder, pipe or device) of sane size.
fn validate_input_file(path: &str, max: u64) -> IpcResult<PathBuf> {
    if path.is_empty() || path.contains('\0') || !Path::new(path).is_absolute() {
        return Err(file_not_found());
    }
    let c = dunce_canonicalize(Path::new(path)).map_err(|_| file_not_found())?;
    let m = fs::metadata(&c).map_err(|_| file_not_found())?;
    if !m.is_file() {
        return Err(IpcError::new(
            ErrorCode::UnsupportedFormat,
            "That isn't a file SplitSound can open.",
        ));
    }
    if m.len() == 0 {
        return Err(IpcError::new(
            ErrorCode::DecodeFailed,
            "That file is empty.",
        ));
    }
    if m.len() > max {
        return Err(IpcError::new(
            ErrorCode::UnsupportedFormat,
            "That file is too large.",
        ));
    }
    Ok(c)
}

/// Canonicalises without the `\\?\` prefix Windows adds, so paths handed
/// to the UI look normal.
pub fn dunce_canonicalize(p: &Path) -> io::Result<PathBuf> {
    let c = fs::canonicalize(p)?;
    #[cfg(windows)]
    {
        let s = c.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\")
            && !rest.starts_with("UNC\\")
        {
            return Ok(PathBuf::from(rest));
        }
    }
    Ok(c)
}

fn clean_tag(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control())
        .take(MAX_TAG_CHARS)
        .collect::<String>()
        .trim()
        .to_string()
}

fn path_string(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Throttled, monotonic `split:progress` reporting with an ETA.
struct Reporter<'a> {
    core: &'a AppCore,
    session_id: String,
    started: Instant,
    last_emit: Option<Instant>,
    last_stage: Option<SplitStage>,
    fraction: f64,
    eta: Option<f64>,
}

impl<'a> Reporter<'a> {
    const EVERY: Duration = Duration::from_millis(100);

    fn new(core: &'a AppCore, session_id: &str) -> Self {
        Reporter {
            core,
            session_id: session_id.into(),
            started: Instant::now(),
            last_emit: None,
            last_stage: None,
            fraction: 0.0,
            eta: None,
        }
    }

    fn report(
        &mut self,
        stage: SplitStage,
        fraction: f64,
        label: &str,
        step: u32,
        step_count: u32,
        force: bool,
    ) {
        let f = if fraction.is_finite() {
            fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.fraction = self.fraction.max(f);
        let now = Instant::now();
        let stage_changed = self.last_stage != Some(stage);
        if !force && !stage_changed && self.last_emit.is_some_and(|t| now - t < Self::EVERY) {
            return;
        }
        let elapsed = self.started.elapsed().as_secs_f64();
        if self.fraction >= 0.03 && elapsed >= 1.0 {
            let raw = elapsed * (1.0 - self.fraction) / self.fraction;
            self.eta = Some(self.eta.map_or(raw, |prev| prev * 0.7 + raw * 0.3));
        }
        self.core
            .events
            .emit(AppEvent::SplitProgress(SplitProgress {
                session_id: self.session_id.clone(),
                stage,
                fraction: self.fraction,
                label: label.to_string(),
                step: step.clamp(1, step_count.max(1)),
                step_count: step_count.max(1),
                eta_sec: if self.fraction >= 1.0 {
                    Some(0.0)
                } else {
                    self.eta.map(|e| e.max(0.0).round())
                },
            }));
        self.last_emit = Some(now);
        self.last_stage = Some(stage);
    }
}

impl AppCore {
    /// Opens the library (repairing what a crash left behind), loads the
    /// settings and starts the split and scan workers.
    pub fn start(opts: CoreOptions) -> io::Result<Arc<AppCore>> {
        opts.paths.prepare()?;
        let (library, report) = Library::open(opts.paths.library_dir.clone())?;
        for id in &report.repaired {
            log::info!("library: repaired session {id} interrupted by a previous exit");
        }
        let settings = SettingsStore::load(opts.paths.settings_file.clone());
        if let Ok(engine) = &opts.engine {
            engine.configure_threads(settings.get().threads);
        }
        let player = Player::new(opts.events.clone(), opts.audio.clone());
        let core = Arc::new_cyclic(|weak: &Weak<AppCore>| {
            let (w1, w2, w3, w4) = (weak.clone(), weak.clone(), weak.clone(), weak.clone());
            let splits = Worker::start(
                "splitsound-split",
                move |job: SplitJob, token: &CancelToken| {
                    if let Some(c) = w1.upgrade() {
                        c.run_split(job, token);
                    }
                },
                move |key, _msg| {
                    if let Some(c) = w2.upgrade() {
                        if let Ok(dir) = c.library.dir(&key) {
                            let _ = fs::remove_dir_all(dir.join("stems.partial"));
                        }
                        c.fail_split(
                            &key,
                            "Something went wrong while splitting this song. Please try again.",
                        );
                    }
                },
            );
            let scans = Worker::start(
                "splitsound-scan",
                move |id: String, token: &CancelToken| {
                    if let Some(c) = w3.upgrade() {
                        c.run_scan(&id, token);
                    }
                },
                move |key, _msg| {
                    if let Some(c) = w4.upgrade() {
                        c.set_analysis(&key, AnalysisStatus::Failed, None);
                    }
                },
            );
            AppCore {
                paths: opts.paths,
                engine: opts.engine,
                events: opts.events,
                library,
                player,
                settings,
                splits,
                split_actions: Mutex::new(()),
                scans,
                exports: Mutex::new(()),
                drag_copies: Mutex::new(()),
                downloads: Mutex::new(Vec::new()),
                download_progress: Mutex::new(HashMap::new()),
                handed_out: Mutex::new(HashSet::new()),
                version: opts.version,
            }
        });
        for id in report.pending_scans {
            core.scans.push(id);
        }
        Ok(core)
    }

    /// Cancels background work. Called on exit.
    pub fn shutdown(&self) {
        for t in lock(&self.downloads).iter() {
            t.cancel();
        }
        self.splits.shutdown();
        self.scans.shutdown();
        self.player.unload();
    }

    fn engine(&self) -> IpcResult<&dyn Engine> {
        self.engine.as_deref().map_err(|error| error.clone().into())
    }

    fn emit(&self, e: AppEvent) {
        self.events.emit(e);
    }

    fn session_changed(&self, s: &Session) {
        self.emit(AppEvent::SessionUpdated(s.clone()));
    }

    fn library_changed(&self) {
        self.emit(AppEvent::LibraryChanged(self.library.summaries()));
    }

    fn remember(&self, p: &Path) {
        if let Ok(c) = dunce_canonicalize(p) {
            lock(&self.handed_out).insert(c);
        }
    }

    /* -------------------------------------------------------------- */
    /* Library                                                         */
    /* -------------------------------------------------------------- */

    pub fn list_sessions(&self) -> Vec<SessionSummary> {
        self.library.summaries()
    }

    pub fn get_session(&self, id: &str) -> IpcResult<Session> {
        self.library.get(id)
    }

    /// Decodes a file into a new session and queues its instrument scan.
    pub fn import_track(&self, path: &str) -> IpcResult<Session> {
        let engine = self.engine()?;
        let input = validate_input_file(path, MAX_IMPORT_BYTES)?;
        let size = fs::metadata(&input).map(|m| m.len()).unwrap_or(0);
        let id = util::new_session_id();
        let staging = self.library.staging_dir(&id)?;
        fs::create_dir_all(&staging).map_err(|e| IpcError::io("import the song", &e))?;
        let result = (|| {
            let decoded = engine.decode(&input, &staging.join("mix.wav"), &CancelToken::new())?;
            let mut has_artwork = false;
            if let Some(bytes) = decoded.artwork.as_deref()
                && let Some(kind) = artwork::sniff(bytes)
            {
                let p = staging.join(format!("artwork.{}", kind.extension()));
                has_artwork = util::write_atomic(&p, bytes).is_ok();
            }
            let file_name = input
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut track = decoded.info;
            track.path = path_string(&input);
            track.title = clean_tag(&track.title);
            if track.title.is_empty() {
                track.title = input
                    .file_stem()
                    .map(|n| clean_tag(&n.to_string_lossy()))
                    .unwrap_or_default();
            }
            if track.title.is_empty() {
                track.title = "Untitled".into();
            }
            track.artist = track
                .artist
                .as_deref()
                .map(clean_tag)
                .filter(|s| !s.is_empty());
            track.album = track
                .album
                .as_deref()
                .map(clean_tag)
                .filter(|s| !s.is_empty());
            track.codec = clean_tag(&track.codec);
            track.file_name = file_name;
            track.file_size_bytes = size;
            track.has_artwork = has_artwork;
            let session = Session {
                id: id.clone(),
                created_at: now_iso8601(),
                track,
                status: SessionStatus::Imported,
                analysis: AnalysisStatus::Pending,
                detections: Vec::new(),
                suggested_mode: SplitMode::Band,
                split: None,
                stems: Vec::new(),
                error: None,
            };
            self.library.insert(session.clone(), &staging)?;
            Ok(session)
        })();
        match result {
            Ok(session) => {
                self.session_changed(&session);
                self.library_changed();
                self.scans.push(session.id.clone());
                Ok(session)
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&staging);
                Err(e)
            }
        }
    }

    pub fn delete_session(&self, id: &str) -> IpcResult<()> {
        self.library.get(id)?;
        self.remove_session(id)?;
        self.library_changed();
        Ok(())
    }

    fn remove_session(&self, id: &str) -> IpcResult<()> {
        let _action = lock(&self.split_actions);
        self.scans.cancel(id);
        if let Cancelled::Queued(_) = self.splits.cancel(id) {
            self.restore_after_cancel(id);
        }
        if !self.splits.wait_until_stopped(id, Duration::from_secs(15))
            || !self.scans.wait_until_stopped(id, Duration::from_secs(15))
        {
            return Err(IpcError::busy(
                "This song is still being processed. Try again shortly.",
            ));
        }
        let _drag = lock(&self.drag_copies);
        self.player.unload_if(id);
        self.library.remove(id)?;
        let drag = self.paths.drag_dir.join(id);
        let _ = fs::remove_dir_all(&drag);
        lock(&self.handed_out).retain(|p| !p.starts_with(&drag));
        Ok(())
    }

    pub fn clear_library(&self) -> IpcResult<()> {
        let mut first_err = None;
        for id in self.library.ids() {
            if let Err(e) = self.remove_session(&id) {
                first_err.get_or_insert(e);
            }
        }
        self.library_changed();
        first_err.map_or(Ok(()), Err)
    }

    pub fn get_artwork(&self, id: &str) -> IpcResult<Option<String>> {
        let s = self.library.get(id)?;
        if !s.track.has_artwork {
            return Ok(None);
        }
        let dir = self.library.dir(id)?;
        for ext in ["png", "jpg"] {
            let p = dir.join(format!("artwork.{ext}"));
            if p.is_file() {
                let bytes = util::read_bounded(&p, artwork::MAX_ARTWORK_BYTES as u64)
                    .map_err(|e| IpcError::io("read the cover art", &e))?;
                return Ok(artwork::data_url(&bytes));
            }
        }
        Ok(None)
    }

    pub fn get_peaks(&self, id: &str, source: SourceId, buckets: u32) -> IpcResult<Peaks> {
        if buckets == 0 || buckets > MAX_BUCKETS {
            return Err(IpcError::internal(format!(
                "The waveform needs between 1 and {MAX_BUCKETS} buckets."
            )));
        }
        let s = self.library.get(id)?;
        let dir = self.library.dir(id)?;
        let path = match source.stem() {
            None => dir.join("mix.wav"),
            Some(stem) => {
                if !s.stems.iter().any(|x| x.id == stem) {
                    return Err(IpcError::not_found("This song doesn't have that stem."));
                }
                stem_path(&dir, stem)
            }
        };
        Ok(self.engine()?.peaks(&path, buckets)?)
    }

    /* -------------------------------------------------------------- */
    /* Instrument scan                                                 */
    /* -------------------------------------------------------------- */

    fn set_analysis(
        &self,
        id: &str,
        status: AnalysisStatus,
        result: Option<crate::engine::ScanResult>,
    ) {
        let r = self.library.update(id, |s, _| {
            s.analysis = status;
            if let Some(r) = result {
                s.detections = r.detections;
                s.suggested_mode = r.suggested_mode;
            }
        });
        if let Ok((s, _)) = r {
            self.session_changed(&s);
            self.library_changed();
        }
    }

    fn run_scan(&self, id: &str, cancel: &CancelToken) {
        let Ok(dir) = self.library.dir(id) else {
            return;
        };
        if !self.library.contains(id) {
            return;
        }
        let scan = (|| -> IpcResult<Option<crate::engine::ScanResult>> {
            let engine = self.engine()?;
            let mix = dir.join("mix.wav");
            let started = Instant::now();
            let outcome = match engine.scan_instruments(&mix, cancel)? {
                ScanOutcome::Done(report) => ScanOutcome::Done(report),
                ScanOutcome::MissingModels(ids) => {
                    let models = engine.list_models();
                    // Automatic previews can only request known unrestricted models.
                    if ids.is_empty()
                        || !ids.iter().all(|id| {
                            models
                                .iter()
                                .any(|model| model.id == *id && !model.restricted)
                        })
                    {
                        return Ok(None);
                    }
                    let downloaded = engine.download_models(
                        &ids,
                        false,
                        &mut |p| self.note_download_progress(&p),
                        cancel,
                    );
                    self.clear_download_progress(&ids);
                    self.models_changed();
                    if let Err(error) = downloaded {
                        log::info!("instrument preview models for {id} are unavailable: {error}");
                        return Ok(None);
                    }
                    self.rescan_unavailable();
                    engine.scan_instruments(&mix, cancel)?
                }
            };
            match outcome {
                ScanOutcome::Done(report) => {
                    log::info!(
                        "instrument scan for {id}: {} detections in {:.2}s",
                        report.detections.len(),
                        started.elapsed().as_secs_f64()
                    );
                    Ok(Some(report))
                }
                ScanOutcome::MissingModels(_) => Ok(None),
            }
        })();
        match scan {
            Ok(Some(report)) => self.set_analysis(id, AnalysisStatus::Done, Some(report)),
            Ok(None) => self.set_analysis(id, AnalysisStatus::Unavailable, None),
            Err(e) if e.code == ErrorCode::Cancelled => {}
            Err(e) if e.code == ErrorCode::ModelMissing => {
                self.set_analysis(id, AnalysisStatus::Unavailable, None);
            }
            Err(e) => {
                log::warn!("instrument scan of {id} failed: {e}");
                self.set_analysis(id, AnalysisStatus::Failed, None);
            }
        }
    }

    /// After models were installed, retry scans that had no tagger.
    fn rescan_unavailable(&self) {
        for id in self.library.ids() {
            if let Ok(s) = self.library.get(&id)
                && s.analysis == AnalysisStatus::Unavailable
                && !self.scans.contains(&id)
            {
                self.set_analysis(&id, AnalysisStatus::Pending, None);
                self.scans.push(id);
            }
        }
    }

    /* -------------------------------------------------------------- */
    /* Splitting                                                       */
    /* -------------------------------------------------------------- */

    fn models_with_progress(&self, mut models: Vec<ModelStatus>) -> Vec<ModelStatus> {
        let p = lock(&self.download_progress);
        for m in &mut models {
            m.download_progress = if m.installed {
                None
            } else {
                p.get(&m.id).map(|v| v.0)
            };
        }
        models
    }

    pub fn estimate_split(&self, id: &str, options: SplitOptions) -> IpcResult<SplitEstimate> {
        let s = self.library.get(id)?;
        let plan = self.engine()?.plan_split(s.track.duration_sec, &options)?;
        let download_bytes = plan.missing().map(|m| m.download_bytes).sum();
        Ok(SplitEstimate {
            estimated_sec: plan.estimated_sec,
            models: self.models_with_progress(plan.models),
            download_bytes,
        })
    }

    /// Queues a split. Refuses (`terms_required`) if a restricted model would
    /// have to be downloaded and `accept_restricted` is false.
    pub fn start_split(
        &self,
        id: &str,
        options: SplitOptions,
        accept_restricted: bool,
    ) -> IpcResult<()> {
        let _action = lock(&self.split_actions);
        let s = self.library.get(id)?;
        // The session status is the source of truth: a finished job may
        // still be winding down in the worker after reporting `ready`.
        if matches!(s.status, SessionStatus::Queued | SessionStatus::Splitting) {
            return Err(IpcError::busy("This song is already being split."));
        }
        let plan = self.engine()?.plan_split(s.track.duration_sec, &options)?;
        let restricted: Vec<&ModelStatus> = plan.missing_restricted().collect();
        if !restricted.is_empty() && !accept_restricted {
            return Err(terms_required(&restricted));
        }
        let model_ids: Vec<String> = plan.models.iter().map(|m| m.id.clone()).collect();
        let (session, accepted) = self.library.update(id, |s, prev| {
            if matches!(s.status, SessionStatus::Queued | SessionStatus::Splitting) {
                return false;
            }
            *prev = if s.stems.is_empty() {
                None
            } else {
                s.split.clone()
            };
            s.status = SessionStatus::Queued;
            s.error = None;
            s.split = Some(SplitSummary {
                mode: options.mode,
                quality: options.quality,
                finished_at: now_iso8601(),
                elapsed_sec: 0.0,
                model_ids,
            });
            true
        })?;
        if !accepted {
            return Err(IpcError::busy("This song is already being split."));
        }
        self.session_changed(&session);
        self.library_changed();
        self.splits.push(SplitJob {
            session_id: id.to_string(),
            options,
            accept_restricted,
        });
        Ok(())
    }

    pub fn cancel_split(&self, id: &str) -> IpcResult<()> {
        let _action = lock(&self.split_actions);
        self.library.get(id)?;
        if let Cancelled::Queued(_) = self.splits.cancel(id) {
            self.restore_after_cancel(id);
        }
        Ok(())
    }

    fn restore_after_cancel(&self, id: &str) {
        if let Ok((s, _)) = self.library.update(id, |s, prev| {
            s.status = if s.stems.is_empty() {
                SessionStatus::Imported
            } else {
                SessionStatus::Ready
            };
            s.split = prev.take();
            s.error = None;
        }) {
            self.session_changed(&s);
            self.library_changed();
        }
    }

    fn fail_split(&self, id: &str, message: &str) {
        if let Ok((s, _)) = self.library.update(id, |s, prev| {
            s.status = SessionStatus::Failed;
            s.split = prev.take();
            s.error = Some(message.to_string());
        }) {
            self.session_changed(&s);
            self.library_changed();
        }
    }

    fn run_split(&self, job: SplitJob, cancel: &CancelToken) {
        let id = job.session_id.clone();
        let result = self.split_inner(&job, cancel);
        if let Ok(dir) = self.library.dir(&id) {
            let _ = fs::remove_dir_all(dir.join("stems.partial"));
        }
        match result {
            Ok(()) => {}
            Err(e) if e.code == ErrorCode::Cancelled || cancel.is_cancelled() => {
                self.restore_after_cancel(&id)
            }
            Err(e) => {
                log::warn!("split of {id} failed: {e}");
                self.fail_split(&id, &e.message);
            }
        }
    }

    fn split_inner(&self, job: &SplitJob, cancel: &CancelToken) -> IpcResult<()> {
        let id = job.session_id.as_str();
        let started = Instant::now();
        let (session, _) = self
            .library
            .update(id, |s, _| s.status = SessionStatus::Splitting)?;
        self.session_changed(&session);
        self.library_changed();
        let dir = self.library.dir(id)?;
        let engine = self.engine()?;
        let plan = engine.plan_split(session.track.duration_sec, &job.options)?;
        let missing: Vec<ModelStatus> = plan.missing().cloned().collect();
        let restricted: Vec<&ModelStatus> = missing.iter().filter(|m| m.restricted).collect();
        if !restricted.is_empty() && !job.accept_restricted {
            return Err(terms_required(&restricted));
        }

        let mut rep = Reporter::new(self, id);
        let pre = u32::from(!missing.is_empty());
        let engine_steps = plan.step_count.max(1);
        let total_steps = pre + engine_steps + 1;
        let dl_bytes: u64 = missing.iter().map(|m| m.download_bytes).sum();
        let dl_share = if missing.is_empty() {
            0.0
        } else {
            let t = dl_bytes as f64 / 20e6;
            (t / (t + plan.estimated_sec.max(1.0))).clamp(0.05, 0.8)
        };
        let write_start = 0.98;

        // 1. Missing models.
        if !missing.is_empty() {
            rep.report(
                SplitStage::Downloading,
                0.0,
                "Downloading models",
                1,
                total_steps,
                true,
            );
            let ids: Vec<String> = missing.iter().map(|m| m.id.clone()).collect();
            let names: HashMap<&str, &str> = missing
                .iter()
                .map(|m| (m.id.as_str(), m.name.as_str()))
                .collect();
            let mut got: HashMap<String, u64> = HashMap::new();
            let res = engine.download_models(
                &ids,
                job.accept_restricted,
                &mut |p| {
                    self.note_download_progress(&p);
                    got.insert(p.model_id.clone(), p.received_bytes);
                    let sum: u64 = got.values().sum();
                    let frac = dl_share * (sum as f64 / dl_bytes.max(1) as f64).min(1.0);
                    let name = names.get(p.model_id.as_str()).copied().unwrap_or("models");
                    rep.report(
                        SplitStage::Downloading,
                        frac,
                        &format!("Downloading {name}"),
                        1,
                        total_steps,
                        false,
                    );
                },
                cancel,
            );
            self.clear_download_progress(&ids);
            self.models_changed();
            res?;
            self.rescan_unavailable();
        }

        // 2. Separation into a scratch folder.
        let partial = dir.join("stems.partial");
        let _ = fs::remove_dir_all(&partial);
        fs::create_dir_all(&partial).map_err(|e| IpcError::io("prepare the stems folder", &e))?;
        let req = SplitRequest {
            mix: dir.join("mix.wav"),
            out_dir: partial.clone(),
            options: job.options,
            accept_restricted: job.accept_restricted,
        };
        rep.report(
            SplitStage::Preparing,
            dl_share,
            "Preparing",
            pre + 1,
            total_steps,
            true,
        );
        let output = engine.split(
            &req,
            &mut |p: EngineProgress| {
                let f = dl_share + (write_start - dl_share) * p.fraction.clamp(0.0, 1.0);
                rep.report(
                    p.stage,
                    f,
                    &p.label,
                    pre + p.step.clamp(1, engine_steps),
                    total_steps,
                    false,
                );
            },
            cancel,
        )?;
        let stems = job.options.mode.stems();
        if output.stems.len() != stems.len()
            || !output.stems.iter().zip(stems).all(|((stem, _), expected)| {
                stem == expected && partial.join(format!("{}.wav", stem.as_str())).is_file()
            })
        {
            return Err(IpcError::internal(
                "The separation finished without producing every stem. Please try again.",
            ));
        }
        if cancel.is_cancelled() {
            return Err(IpcError::cancelled());
        }

        // 3. Swap the measured stems in.
        rep.report(
            SplitStage::Writing,
            write_start,
            "Saving",
            total_steps,
            total_steps,
            true,
        );
        // A first split has no existing stem mappings to release. Keep the
        // original playing while the new files and metadata are installed.
        let detached = !session.stems.is_empty() && self.player.detach(id);
        let swapped = swap_stems(&dir);
        if let Err(e) = swapped {
            if detached {
                self.player.reattach(&session, &dir);
            }
            return Err(IpcError::io("save the stems", &e));
        }
        let infos: Vec<StemInfo> = output
            .stems
            .into_iter()
            .map(|(st, a)| StemInfo {
                id: st,
                path: path_string(&stem_path(&dir, st)),
                present: a.present,
                active_fraction: a.active_fraction,
                peak_db: a.peak_db,
                rms_db: a.rms_db,
                activity: a.activity,
                contains: a.contains,
            })
            .collect();
        let elapsed = started.elapsed().as_secs_f64();
        let committed = self.library.update(id, |s, prev| {
            s.status = SessionStatus::Ready;
            s.stems = infos;
            s.split = Some(SplitSummary {
                mode: job.options.mode,
                quality: job.options.quality,
                finished_at: now_iso8601(),
                elapsed_sec: (elapsed * 10.0).round() / 10.0,
                model_ids: output.model_ids.clone(),
            });
            *prev = None;
            s.error = None;
        });
        let (session, _) = match committed {
            Ok(committed) => committed,
            Err(e) => {
                let restored = crate::library::rollback_stems(&dir, !session.stems.is_empty());
                if let Err(rollback) = restored {
                    log::error!("couldn't restore stems for {id}: {rollback}");
                }
                if detached {
                    self.player.reattach(&session, &dir);
                }
                return Err(e);
            }
        };
        let old = dir.join("stems.old");
        if let Err(e) = fs::remove_dir_all(old)
            && e.kind() != io::ErrorKind::NotFound
        {
            log::warn!("couldn't remove old stems in {}: {e}", dir.display());
        }
        self.library.refresh_size(id);
        rep.report(
            SplitStage::Writing,
            1.0,
            "Done",
            total_steps,
            total_steps,
            true,
        );
        self.session_changed(&session);
        self.library_changed();
        self.player.reattach(&session, &dir);
        Ok(())
    }

    /* -------------------------------------------------------------- */
    /* Playback                                                        */
    /* -------------------------------------------------------------- */

    pub fn playback_load(&self, id: &str) -> IpcResult<PlaybackState> {
        let s = self.library.get(id)?;
        let dir = self.library.dir(id)?;
        self.player.load(&s, &dir)
    }

    /* -------------------------------------------------------------- */
    /* Export and drag-out                                             */
    /* -------------------------------------------------------------- */

    fn checked_stems(s: &Session, ids: &[StemId]) -> IpcResult<Vec<StemId>> {
        if s.stems.is_empty() {
            return Err(IpcError::not_found("This song hasn't been split yet."));
        }
        let mut out = Vec::new();
        for id in ids {
            if !s.stems.iter().any(|x| x.id == *id) {
                return Err(IpcError::not_found("This song doesn't have that stem."));
            }
            if !out.contains(id) {
                out.push(*id);
            }
        }
        Ok(out)
    }

    pub fn export_stems(&self, id: &str, options: ExportOptions) -> IpcResult<ExportResult> {
        let engine = self.engine()?;
        let _export = lock(&self.exports);
        let s = self.library.get(id)?;
        let stems = Self::checked_stems(&s, &options.stem_ids)?;
        if stems.is_empty() {
            return Err(IpcError::internal("Choose at least one stem to export."));
        }
        let d = &options.destination_dir;
        if d.is_empty() || d.contains('\0') || !Path::new(d).is_absolute() {
            return Err(IpcError::new(
                ErrorCode::FileNotFound,
                "That folder doesn't exist.",
            ));
        }
        let dest = dunce_canonicalize(Path::new(d))
            .ok()
            .filter(|p| p.is_dir())
            .ok_or_else(|| IpcError::new(ErrorCode::FileNotFound, "That folder doesn't exist."))?;
        let dir = self.library.dir(id)?;
        let title = sanitize_file_stem(&s.track.title);
        let ext = options.format.extension();
        let mut taken: Vec<PathBuf> = Vec::new();
        let mut items = Vec::new();
        for st in &stems {
            let p = unique_path(
                &dest,
                &format!("{title} ({})", st.display_name()),
                ext,
                &taken,
            )
            .map_err(|e| IpcError::io("choose an export file name", &e))?;
            taken.push(p.clone());
            items.push(ExportItem {
                sources: vec![(stem_path(&dir, *st), 1.0)],
                dest: p,
            });
        }
        if options.include_mixdown {
            let (gains, master) = self
                .player
                .mixdown_gains(id)
                .unwrap_or_else(|| (stems.iter().map(|s| (*s, 1.0)).collect(), 1.0));
            let sources = stems
                .iter()
                .map(|st| {
                    let g = gains.iter().find(|(s, _)| s == st).map_or(1.0, |(_, g)| *g);
                    (stem_path(&dir, *st), g * master)
                })
                .collect();
            let p = unique_path(&dest, &format!("{title} (Mix)"), ext, &taken)
                .map_err(|e| IpcError::io("choose an export file name", &e))?;
            items.push(ExportItem { sources, dest: p });
        }
        let files = engine.export(
            &ExportJob {
                format: options.format,
                items,
            },
            &CancelToken::new(),
        )?;
        self.remember(&dest);
        for f in &files {
            self.remember(f);
        }
        let dest_s = path_string(&dest);
        if let Err(e) = self.settings.update(SettingsPatch {
            export_dir: Some(Some(dest_s.clone())),
            ..Default::default()
        }) {
            log::warn!("couldn't remember the export folder: {}", e.message);
        }
        Ok(ExportResult {
            destination_dir: dest_s,
            files: files.iter().map(|f| path_string(f)).collect(),
        })
    }

    /// Makes "Song title (Vocals).wav" copies in the drag folder.
    pub fn prepare_drag(&self, id: &str, stem_ids: &[StemId]) -> IpcResult<Vec<String>> {
        let _drag = lock(&self.drag_copies);
        let s = self.library.get(id)?;
        let dir = self.library.dir(id)?;
        if s.stems.is_empty() {
            return Err(IpcError::not_found("This song hasn't been split yet."));
        }
        for st in stem_ids {
            if !s.stems.iter().any(|x| x.id == *st) {
                return Err(IpcError::not_found("This song doesn't have that stem."));
            }
        }
        let drag = self.paths.drag_dir.join(id);
        fs::create_dir_all(&drag).map_err(|e| IpcError::io("prepare the files", &e))?;
        let title = sanitize_file_stem(&s.track.title);
        let mut out = Vec::with_capacity(stem_ids.len());
        for st in stem_ids {
            let src = stem_path(&dir, *st);
            let name = format!("{title} ({}).wav", st.display_name());
            let target = drag.join(&name);
            let src_meta = fs::metadata(&src)
                .map_err(|_| IpcError::not_found("This stem's file is missing."))?;
            let fresh = fs::metadata(&target).is_ok_and(|m| {
                m.len() == src_meta.len()
                    && match (m.modified(), src_meta.modified()) {
                        (Ok(a), Ok(b)) => a >= b,
                        _ => false,
                    }
            });
            if !fresh {
                let tmp = drag.join(format!(".{name}.part"));
                fs::copy(&src, &tmp)
                    .and_then(|_| fs::rename(&tmp, &target))
                    .map_err(|e| {
                        let _ = fs::remove_file(&tmp);
                        IpcError::io("prepare the files", &e)
                    })?;
            }
            let c =
                dunce_canonicalize(&target).map_err(|e| IpcError::io("prepare the files", &e))?;
            lock(&self.handed_out).insert(c.clone());
            out.push(path_string(&c));
        }
        Ok(out)
    }

    /// Validates paths the webview asks to drag out: only files made by
    /// [`AppCore::prepare_drag`] are allowed.
    pub fn check_drag_paths(&self, paths: &[PathBuf]) -> IpcResult<Vec<PathBuf>> {
        let root = dunce_canonicalize(&self.paths.drag_dir)
            .map_err(|e| IpcError::io("drag the files", &e))?;
        let handed = lock(&self.handed_out);
        paths
            .iter()
            .map(|p| {
                let c = dunce_canonicalize(p).map_err(|_| file_not_found())?;
                if c.is_file() && c.starts_with(&root) && handed.contains(&c) {
                    Ok(c)
                } else {
                    Err(IpcError::not_found(
                        "Only stems prepared by SplitSound can be dragged out.",
                    ))
                }
            })
            .collect()
    }

    /// Validates a path for `reveal_path`: inside a folder the app owns, or
    /// a path the app handed out.
    pub fn check_reveal(&self, path: &str) -> IpcResult<PathBuf> {
        let deny = || IpcError::not_found("SplitSound can only show files it made.");
        if path.is_empty() || path.contains('\0') || !Path::new(path).is_absolute() {
            return Err(deny());
        }
        let c = dunce_canonicalize(Path::new(path)).map_err(|_| deny())?;
        if lock(&self.handed_out).contains(&c) {
            return Ok(c);
        }
        for root in [
            &self.paths.library_dir,
            &self.paths.models_dir,
            &self.paths.drag_dir,
            &self.paths.log_dir,
        ] {
            if let Ok(r) = dunce_canonicalize(root)
                && c.starts_with(&r)
            {
                return Ok(c);
            }
        }
        Err(deny())
    }

    /* -------------------------------------------------------------- */
    /* Models                                                          */
    /* -------------------------------------------------------------- */

    pub fn list_models(&self) -> IpcResult<Vec<ModelStatus>> {
        Ok(self.models_with_progress(self.engine()?.list_models()))
    }

    fn models_changed(&self) {
        if let Ok(models) = self.list_models() {
            self.emit(AppEvent::ModelsChanged(models));
        }
    }

    fn note_download_progress(&self, p: &ModelDownloadProgress) {
        let frac = if p.total_bytes > 0 {
            (p.received_bytes as f64 / p.total_bytes as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let emit = {
            let mut map = lock(&self.download_progress);
            let now = Instant::now();
            let due = map
                .get(&p.model_id)
                .is_none_or(|(_, t)| now - *t >= Duration::from_millis(100));
            let last = p.received_bytes >= p.total_bytes;
            let t = if due || last {
                now
            } else {
                map.get(&p.model_id).map_or(now, |v| v.1)
            };
            map.insert(p.model_id.clone(), (frac, t));
            due || last
        };
        if emit {
            self.emit(AppEvent::ModelsProgress(p.clone()));
        }
    }

    fn clear_download_progress(&self, ids: &[String]) {
        let mut map = lock(&self.download_progress);
        for id in ids {
            map.remove(id);
        }
    }

    /// Starts downloading models in the background. Refuses
    /// (`terms_required`) restricted models unless `accept_restricted`.
    pub fn download_models(
        self: &Arc<Self>,
        ids: Vec<String>,
        accept_restricted: bool,
    ) -> IpcResult<()> {
        let models = self.engine()?.list_models();
        let mut restricted = Vec::new();
        for id in &ids {
            let m = models
                .iter()
                .find(|m| &m.id == id)
                .ok_or_else(|| IpcError::not_found("That model doesn't exist."))?;
            if m.restricted {
                restricted.push(m);
            }
        }
        if !restricted.is_empty() && !accept_restricted {
            return Err(terms_required(&restricted));
        }
        if ids.is_empty() {
            return Ok(());
        }
        let token = CancelToken::new();
        lock(&self.downloads).push(token.clone());
        let core = self.clone();
        std::thread::Builder::new()
            .name("splitsound-download".into())
            .spawn(move || {
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    core.engine()?
                        .download_models(
                            &ids,
                            accept_restricted,
                            &mut |p| core.note_download_progress(&p),
                            &token,
                        )
                        .map_err(IpcError::from)
                }));
                core.clear_download_progress(&ids);
                lock(&core.downloads).retain(|t| !t.same(&token));
                match res {
                    Ok(Ok(())) => core.rescan_unavailable(),
                    Ok(Err(e)) if e.code == ErrorCode::Cancelled => {
                        log::info!("model download cancelled")
                    }
                    Ok(Err(e)) => log::warn!("model download failed: {e}"),
                    Err(p) => log::error!(
                        "model download panicked: {}",
                        crate::worker::panic_message(p.as_ref())
                    ),
                }
                core.models_changed();
            })
            .map_err(|e| IpcError::internal(format!("Couldn't start the download ({e}).")))?;
        Ok(())
    }

    pub fn cancel_model_downloads(&self) {
        for t in lock(&self.downloads).drain(..) {
            t.cancel();
        }
    }

    pub fn remove_model(&self, id: &str) -> IpcResult<()> {
        self.engine()?.remove_model(id)?;
        self.models_changed();
        Ok(())
    }

    pub fn import_model_files(&self, paths: &[String]) -> IpcResult<Vec<String>> {
        let engine = self.engine()?;
        let files = paths
            .iter()
            .map(|p| validate_input_file(p, MAX_MODEL_FILE_BYTES))
            .collect::<IpcResult<Vec<_>>>()?;
        let ids = engine.import_model_files(&files)?;
        if !ids.is_empty() {
            self.models_changed();
            self.rescan_unavailable();
        }
        Ok(ids)
    }

    /* -------------------------------------------------------------- */
    /* Settings and system                                             */
    /* -------------------------------------------------------------- */

    pub fn get_settings(&self) -> Settings {
        self.settings.get()
    }

    pub fn update_settings(&self, patch: SettingsPatch) -> IpcResult<Settings> {
        let settings = self.settings.update(patch)?;
        if let Ok(engine) = self.engine() {
            engine.configure_threads(settings.threads);
        }
        Ok(settings)
    }

    pub fn get_app_info(&self) -> AppInfo {
        let hw = self
            .engine()
            .map(|engine| engine.hardware())
            .unwrap_or_else(|_| crate::engine::HardwareInfo {
                gpu: GpuInfo {
                    available: false,
                    name: None,
                },
                cpu_threads: std::thread::available_parallelism().map_or(1, |n| n.get() as u32),
            });
        AppInfo {
            version: self.version.clone(),
            platform: Platform::current(),
            arch: std::env::consts::ARCH.to_string(),
            gpu: hw.gpu,
            cpu_threads: hw.cpu_threads,
            models_dir: path_string(&self.paths.models_dir),
            library_dir: path_string(&self.paths.library_dir),
            library_bytes: self.library.total_bytes(),
        }
    }
}

impl Drop for AppCore {
    fn drop(&mut self) {
        for t in lock(&self.downloads).iter() {
            t.cancel();
        }
    }
}

/// Keep the previous files until the new `session.json` commits the swap.
fn swap_stems(dir: &Path) -> io::Result<()> {
    let stems = dir.join("stems");
    let partial = dir.join("stems.partial");
    let old = dir.join("stems.old");
    if old.exists() {
        fs::remove_dir_all(&old)?;
    }
    let had_old = stems.exists();
    if had_old {
        fs::rename(&stems, &old)?;
    }
    if let Err(e) = fs::rename(&partial, &stems) {
        if had_old {
            let _ = fs::rename(&old, &stems);
        }
        return Err(e);
    }
    Ok(())
}
