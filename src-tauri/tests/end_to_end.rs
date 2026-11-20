#![cfg(debug_assertions)]

//! Drives the shell end to end through `AppCore` with the mock engine and
//! no audio device, the way the commands do.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use splitsound_desktop_lib::app::{AppCore, AppPaths, CoreOptions};
use splitsound_desktop_lib::audio::mapped::MappedWav;
use splitsound_desktop_lib::audio::output::{AudioBackend, NoDeviceBackend};
use splitsound_desktop_lib::audio::wav::{WavWriter, WriteFormat};
use splitsound_desktop_lib::engine::mock::MockEngine;
use splitsound_desktop_lib::engine::*;
use splitsound_desktop_lib::ipc::*;
use splitsound_desktop_lib::util::new_session_id;

struct Fixture {
    root: PathBuf,
    core: Arc<AppCore>,
    sink: Arc<RecordingSink>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.core.shutdown();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn engine_config(paths: &AppPaths) -> EngineConfig {
    EngineConfig {
        models_dir: paths.models_dir.clone(),
        cache_dir: paths.engine_cache_dir.clone(),
        onnxruntime: None,
    }
}

fn start_with(root: PathBuf, engine: impl FnOnce(&AppPaths) -> Arc<dyn Engine>) -> Fixture {
    let paths = AppPaths::under(&root);
    let sink = Arc::new(RecordingSink::default());
    let core = AppCore::start(CoreOptions {
        engine: Ok(engine(&paths)),
        paths,
        events: sink.clone(),
        audio: Arc::new(|| Box::new(NoDeviceBackend) as Box<dyn AudioBackend>),
        version: "0.0.0-test".into(),
    })
    .unwrap();
    Fixture { root, core, sink }
}

fn start() -> Fixture {
    let root = std::env::temp_dir().join(format!("ss-e2e-{}", new_session_id()));
    start_with(root, |p| {
        Arc::new(MockEngine::with_time_scale(engine_config(p), 0.02))
    })
}

/// Appends a `LIST/INFO` chunk with a title (`INAM`) to a WAV file.
fn tag_title(path: &Path, title: &str) {
    let mut text = title.as_bytes().to_vec();
    text.push(0);
    if text.len() % 2 == 1 {
        text.push(0);
    }
    let mut chunk = b"LIST".to_vec();
    chunk.extend_from_slice(&((4 + 8 + text.len()) as u32).to_le_bytes());
    chunk.extend_from_slice(b"INFOINAM");
    chunk.extend_from_slice(&(text.len() as u32).to_le_bytes());
    chunk.extend_from_slice(&text);
    let mut bytes = std::fs::read(path).unwrap();
    bytes.extend_from_slice(&chunk);
    let riff = (bytes.len() - 8) as u32;
    bytes[4..8].copy_from_slice(&riff.to_le_bytes());
    std::fs::write(path, bytes).unwrap();
}

/// A 3-second stereo 48 kHz WAV with bass, mids and highs.
fn song(dir: &Path, name: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let p = dir.join(name);
    let rate = 48_000;
    let mut w = WavWriter::create(&p, WriteFormat::Pcm24, 2, rate).unwrap();
    let s: Vec<f32> = (0..rate * 3)
        .flat_map(|i| {
            let t = i as f64 / rate as f64;
            let x = 0.3 * (2.0 * std::f64::consts::PI * 80.0 * t).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 8000.0 * t).sin();
            [x as f32, x as f32 * 0.9]
        })
        .collect();
    w.write(&s).unwrap();
    w.finish().unwrap();
    p
}

fn wait_until(what: &str, timeout: Duration, mut f: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < timeout, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn progress_of(events: &[AppEvent], id: &str) -> Vec<SplitProgress> {
    events
        .iter()
        .filter_map(|e| match e {
            AppEvent::SplitProgress(p) if p.session_id == id => Some(p.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn import_scan_split_export_drag_delete() {
    let f = start();
    let input = song(&f.root.join("music"), "song.wav");
    // A title with characters no file system likes, from the tags.
    tag_title(&input, "My Song: Live?");
    let s = f.core.import_track(&input.to_string_lossy()).unwrap();
    assert_eq!(s.status, SessionStatus::Imported);
    assert_eq!(s.track.title, "My Song: Live?");
    assert_eq!(s.track.sample_rate, 48_000);
    assert!((s.track.duration_sec - 3.0).abs() < 1e-6);
    assert_eq!(s.track.file_name, "song.wav");
    assert!(s.track.file_size_bytes > 0);
    assert_eq!(f.core.list_sessions().len(), 1);

    // The background scan finishes and suggests a split.
    wait_until("the instrument scan", Duration::from_secs(10), || {
        f.core.get_session(&s.id).unwrap().analysis == AnalysisStatus::Done
    });
    let scanned = f.core.get_session(&s.id).unwrap();
    assert!(!scanned.detections.is_empty());
    assert!(
        scanned
            .detections
            .windows(2)
            .all(|w| w[0].confidence >= w[1].confidence)
    );

    // Peaks of the original.
    let p = f.core.get_peaks(&s.id, SourceId::Mix, 100).unwrap();
    assert_eq!(p.buckets, 100);
    assert!(
        f.core.get_peaks(&s.id, SourceId::Vocals, 100).is_err(),
        "no stems yet"
    );
    assert!(f.core.get_peaks(&s.id, SourceId::Mix, 0).is_err());

    // Split into four stems; the SCNet model has to be downloaded first.
    let opts = SplitOptions {
        mode: SplitMode::Band,
        quality: Quality::Fast,
    };
    let est = f.core.estimate_split(&s.id, opts).unwrap();
    assert_eq!(est.models.len(), 1);
    assert!(est.download_bytes > 0);
    f.sink.take();
    f.core.start_split(&s.id, opts, false).unwrap();
    assert_eq!(
        f.core.start_split(&s.id, opts, false).unwrap_err().code,
        ErrorCode::Busy
    );
    wait_until("the split", Duration::from_secs(30), || {
        f.core.get_session(&s.id).unwrap().status == SessionStatus::Ready
    });
    let ready = f.core.get_session(&s.id).unwrap();
    let ids: Vec<StemId> = ready.stems.iter().map(|x| x.id).collect();
    assert_eq!(
        ids,
        vec![StemId::Vocals, StemId::Drums, StemId::Bass, StemId::Other]
    );
    for st in &ready.stems {
        assert_eq!(st.activity.len(), ACTIVITY_BINS);
        assert!(Path::new(&st.path).is_file());
    }
    let summary = ready.split.clone().unwrap();
    assert_eq!(summary.mode, SplitMode::Band);
    assert_eq!(summary.model_ids, vec!["scnet_small".to_string()]);

    // The committed session is readable just before the worker publishes
    // its final progress and ready event. Wait for that event before reading
    // the event log, just as the UI does.
    wait_until("the finished split event", Duration::from_secs(5), || {
        f.sink.snapshot().iter().any(|event| {
            matches!(event, AppEvent::SessionUpdated(session) if session.id == s.id && session.status == SessionStatus::Ready)
        })
    });
    // Progress: downloading first, monotonic, throttled, ends at 1.
    let events = f.sink.take();
    let prog = progress_of(&events, &s.id);
    assert!(prog.len() >= 5, "{} progress events", prog.len());
    assert_eq!(prog[0].stage, SplitStage::Downloading);
    assert!(prog.windows(2).all(|w| w[1].fraction >= w[0].fraction));
    assert_eq!(prog.last().unwrap().fraction, 1.0);
    assert!(prog.iter().all(|p| p.step >= 1 && p.step <= p.step_count));
    assert!(prog.iter().any(|p| p.stage == SplitStage::Analyzing));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AppEvent::ModelsChanged(_)))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AppEvent::ModelsProgress(_)))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AppEvent::SessionUpdated(x) if x.status == SessionStatus::Ready))
    );

    // Stems sum back to the mix.
    let dir = f.core.library.dir(&s.id).unwrap();
    let mix = MappedWav::open(&dir.join("mix.wav")).unwrap();
    let stems: Vec<MappedWav> = ready
        .stems
        .iter()
        .map(|x| MappedWav::open(Path::new(&x.path)).unwrap())
        .collect();
    for i in (0..mix.samples().len()).step_by(997) {
        let sum: f32 = stems.iter().map(|m| m.samples()[i]).sum();
        assert!((sum - mix.samples()[i]).abs() < 1e-4);
    }

    // Playback loads the stems; without a device, play is `audio_device`.
    let st = f.core.playback_load(&s.id).unwrap();
    assert_eq!(st.channels.len(), 4);
    assert_eq!(st.source, PlaybackSource::Stems);
    assert_eq!(
        f.core.player.play().unwrap_err().code,
        ErrorCode::AudioDevice
    );
    f.core
        .player
        .set_channel(SourceId::Drums, None, Some(true), None)
        .unwrap();

    // Export two stems and a mixdown, with sanitised, unique names.
    let out = f.root.join("exports");
    std::fs::create_dir_all(&out).unwrap();
    let res = f
        .core
        .export_stems(
            &s.id,
            ExportOptions {
                stem_ids: vec![StemId::Vocals, StemId::Drums, StemId::Vocals],
                format: ExportFormat::Wav16,
                destination_dir: out.to_string_lossy().into(),
                include_mixdown: true,
            },
        )
        .unwrap();
    let names: Vec<String> = res
        .files
        .iter()
        .map(|p| {
            Path::new(p)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "My Song- Live (Vocals).wav",
            "My Song- Live (Drums).wav",
            "My Song- Live (Mix).wav"
        ]
    );
    assert_eq!(
        f.core.get_settings().export_dir.as_deref(),
        Some(res.destination_dir.as_str())
    );
    // Exporting again never overwrites.
    let again = f
        .core
        .export_stems(
            &s.id,
            ExportOptions {
                stem_ids: vec![StemId::Bass],
                format: ExportFormat::Wav24,
                destination_dir: out.to_string_lossy().into(),
                include_mixdown: false,
            },
        )
        .unwrap();
    assert!(again.files[0].ends_with("My Song- Live (Bass).wav"));
    let flac = f.core.export_stems(
        &s.id,
        ExportOptions {
            stem_ids: vec![StemId::Bass],
            format: ExportFormat::Flac,
            destination_dir: out.to_string_lossy().into(),
            include_mixdown: false,
        },
    );
    assert_eq!(flac.unwrap_err().code, ErrorCode::UnsupportedFormat);
    assert!(
        f.core
            .export_stems(
                &s.id,
                ExportOptions {
                    stem_ids: vec![StemId::Guitar],
                    format: ExportFormat::Wav16,
                    destination_dir: out.to_string_lossy().into(),
                    include_mixdown: false,
                },
            )
            .is_err(),
        "stem not in this split"
    );

    // Drag-out files.
    let drag = f
        .core
        .prepare_drag(&s.id, &[StemId::Bass, StemId::Vocals])
        .unwrap();
    assert!(drag[0].ends_with("My Song- Live (Bass).wav"));
    assert!(drag[1].ends_with("My Song- Live (Vocals).wav"));
    let drag_paths: Vec<PathBuf> = drag.iter().map(PathBuf::from).collect();
    assert_eq!(f.core.check_drag_paths(&drag_paths).unwrap().len(), 2);
    assert!(
        f.core
            .check_drag_paths(std::slice::from_ref(&input))
            .is_err()
    );
    assert!(f.core.check_drag_paths(&[dir.join("mix.wav")]).is_err());

    // Reveal: our folders and handed-out paths only.
    assert!(f.core.check_reveal(&res.files[0]).is_ok());
    assert!(f.core.check_reveal(&res.destination_dir).is_ok());
    assert!(f.core.check_reveal(&drag[0]).is_ok());
    assert!(f.core.check_reveal(&dir.to_string_lossy()).is_ok());
    assert!(f.core.check_reveal(&input.to_string_lossy()).is_err());
    assert!(f.core.check_reveal("relative/path").is_err());
    let sneaky = dir.join("..").join("..").join("..").join("music");
    assert!(f.core.check_reveal(&sneaky.to_string_lossy()).is_err());

    // App info.
    let info = f.core.get_app_info();
    assert!(info.library_bytes > 0);
    assert_eq!(info.version, "0.0.0-test");

    // Delete unloads playback and removes everything.
    f.core.delete_session(&s.id).unwrap();
    assert!(f.core.player.state().session_id.is_none());
    assert!(!dir.exists());
    assert!(!Path::new(&drag[0]).exists());
    assert!(f.core.list_sessions().is_empty());
    assert_eq!(
        f.core.get_session(&s.id).unwrap_err().code,
        ErrorCode::NotFound
    );
}

#[test]
fn restricted_models_need_agreement_everywhere() {
    let f = start();
    let input = song(&f.root.join("music"), "a.wav");
    let s = f.core.import_track(&input.to_string_lossy()).unwrap();
    let all = SplitOptions {
        mode: SplitMode::All,
        quality: Quality::Best,
    };
    let est = f.core.estimate_split(&s.id, all).unwrap();
    let demucs = est
        .models
        .iter()
        .find(|m| m.restricted)
        .expect("a restricted model");
    assert_eq!(demucs.id, "htdemucs_6s");
    assert_eq!(demucs.source, "Meta");
    assert!(demucs.terms.as_deref().unwrap().contains("non-commercial"));

    let err = f.core.start_split(&s.id, all, false).unwrap_err();
    assert_eq!(err.code, ErrorCode::TermsRequired);
    assert_eq!(
        f.core.get_session(&s.id).unwrap().status,
        SessionStatus::Imported
    );
    let err = f
        .core
        .download_models(vec!["htdemucs_6s".into()], false)
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::TermsRequired);
    assert!(f.core.download_models(vec!["nope".into()], true).is_err());
    // Nothing was fetched as a side effect.
    assert!(
        !f.core
            .list_models()
            .unwrap()
            .iter()
            .any(|m| m.id == "htdemucs_6s" && m.installed)
    );

    f.core.start_split(&s.id, all, true).unwrap();
    wait_until("the six-stem split", Duration::from_secs(60), || {
        f.core.get_session(&s.id).unwrap().status == SessionStatus::Ready
    });
    assert_eq!(f.core.get_session(&s.id).unwrap().stems.len(), 6);
    assert!(
        f.core
            .list_models()
            .unwrap()
            .iter()
            .any(|m| m.id == "htdemucs_6s" && m.installed)
    );

    // Once installed, no agreement is needed any more.
    f.core.start_split(&s.id, all, false).unwrap();
    wait_until("the re-split", Duration::from_secs(60), || {
        f.core.get_session(&s.id).unwrap().status == SessionStatus::Ready
    });

    // Removing the model brings the requirement back.
    f.core.remove_model("htdemucs_6s").unwrap();
    assert_eq!(
        f.core.start_split(&s.id, all, false).unwrap_err().code,
        ErrorCode::TermsRequired
    );
}

#[test]
fn cancel_queued_and_running_splits() {
    let root = std::env::temp_dir().join(format!("ss-e2e-{}", new_session_id()));
    // Slower mock so there is time to cancel.
    let f = start_with(root, |p| {
        Arc::new(MockEngine::with_time_scale(engine_config(p), 0.5))
    });
    let a = f
        .core
        .import_track(&song(&f.root.join("m"), "a.wav").to_string_lossy())
        .unwrap();
    let b = f
        .core
        .import_track(&song(&f.root.join("m"), "b.wav").to_string_lossy())
        .unwrap();
    let opts = SplitOptions {
        mode: SplitMode::Vocals,
        quality: Quality::Fast,
    };
    f.core.start_split(&a.id, opts, false).unwrap();
    f.core.start_split(&b.id, opts, false).unwrap();
    assert_eq!(
        f.core.get_session(&b.id).unwrap().status,
        SessionStatus::Queued
    );
    // Cancel the queued one: back to imported at once.
    f.core.cancel_split(&b.id).unwrap();
    let bs = f.core.get_session(&b.id).unwrap();
    assert_eq!(bs.status, SessionStatus::Imported);
    assert_eq!(bs.split, None);
    // Cancel the running one once it is going.
    wait_until("the split to start", Duration::from_secs(10), || {
        f.core.get_session(&a.id).unwrap().status == SessionStatus::Splitting
    });
    std::thread::sleep(Duration::from_millis(300));
    f.core.cancel_split(&a.id).unwrap();
    wait_until("the cancel", Duration::from_secs(10), || {
        f.core.get_session(&a.id).unwrap().status == SessionStatus::Imported
    });
    let dir = f.core.library.dir(&a.id).unwrap();
    assert!(!dir.join("stems.partial").exists());
    assert!(!dir.join("stems").exists());
    // Cancelling something that isn't splitting is fine.
    f.core.cancel_split(&a.id).unwrap();
}

/// An engine that panics while splitting, to check the panic boundary.
struct Panicky(MockEngine);

impl Engine for Panicky {
    fn name(&self) -> &'static str {
        "panicky"
    }
    fn decode(&self, i: &Path, o: &Path, c: &CancelToken) -> EngineResult<DecodedTrack> {
        self.0.decode(i, o, c)
    }
    fn scan_instruments(&self, w: &Path, _c: &CancelToken) -> EngineResult<ScanOutcome> {
        panic!("scan exploded {}", w.display())
    }
    fn plan_split(&self, d: f64, o: &SplitOptions) -> EngineResult<SplitPlan> {
        self.0.plan_split(d, o)
    }
    fn split(
        &self,
        _r: &SplitRequest,
        _p: &mut (dyn FnMut(EngineProgress) + Send),
        _c: &CancelToken,
    ) -> EngineResult<SplitOutput> {
        panic!("split exploded")
    }
    fn export(&self, j: &ExportJob, c: &CancelToken) -> EngineResult<Vec<PathBuf>> {
        self.0.export(j, c)
    }
    fn list_models(&self) -> Vec<ModelStatus> {
        self.0.list_models()
    }
    fn download_models(
        &self,
        ids: &[String],
        a: bool,
        p: &mut (dyn FnMut(ModelDownloadProgress) + Send),
        c: &CancelToken,
    ) -> EngineResult<()> {
        self.0.download_models(ids, a, p, c)
    }
    fn remove_model(&self, id: &str) -> EngineResult<()> {
        self.0.remove_model(id)
    }
    fn import_model_files(&self, p: &[PathBuf]) -> EngineResult<Vec<String>> {
        self.0.import_model_files(p)
    }
    fn hardware(&self) -> HardwareInfo {
        self.0.hardware()
    }
}

#[test]
fn a_panicking_job_fails_its_session_and_the_queue_keeps_going() {
    let root = std::env::temp_dir().join(format!("ss-e2e-{}", new_session_id()));
    let f = start_with(root, |p| {
        Arc::new(Panicky(MockEngine::with_time_scale(engine_config(p), 0.01)))
    });
    let opts = SplitOptions {
        mode: SplitMode::Vocals,
        quality: Quality::Fast,
    };
    let a = f
        .core
        .import_track(&song(&f.root.join("m"), "a.wav").to_string_lossy())
        .unwrap();
    let b = f
        .core
        .import_track(&song(&f.root.join("m"), "b.wav").to_string_lossy())
        .unwrap();
    f.core.start_split(&a.id, opts, false).unwrap();
    f.core.start_split(&b.id, opts, false).unwrap();
    for id in [&a.id, &b.id] {
        wait_until("the failure", Duration::from_secs(20), || {
            f.core.get_session(id).unwrap().status == SessionStatus::Failed
        });
        let s = f.core.get_session(id).unwrap();
        assert!(s.error.unwrap().starts_with("Something went wrong"));
        assert!(s.stems.is_empty());
    }
    // The scan panicked too: marked failed, not stuck pending.
    wait_until("the scans", Duration::from_secs(10), || {
        f.core.get_session(&b.id).unwrap().analysis == AnalysisStatus::Failed
    });
}

#[test]
fn bad_inputs_are_plain_errors() {
    let f = start();
    let missing = f.root.join("nope.wav");
    assert_eq!(
        f.core
            .import_track(&missing.to_string_lossy())
            .unwrap_err()
            .code,
        ErrorCode::FileNotFound
    );
    assert_eq!(
        f.core.import_track("relative.wav").unwrap_err().code,
        ErrorCode::FileNotFound
    );
    assert_eq!(
        f.core
            .import_track(&f.root.to_string_lossy())
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedFormat
    );
    let mp3 = f.root.join("x.mp3");
    std::fs::write(&mp3, b"ID3 not really").unwrap();
    assert_eq!(
        f.core
            .import_track(&mp3.to_string_lossy())
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedFormat
    );
    let junk = f.root.join("junk.wav");
    std::fs::write(&junk, b"RIFF\0\0\0\0WAVEjunk").unwrap();
    assert_eq!(
        f.core
            .import_track(&junk.to_string_lossy())
            .unwrap_err()
            .code,
        ErrorCode::DecodeFailed
    );
    // Failed imports leave nothing behind.
    assert!(f.core.list_sessions().is_empty());
    let leftovers: Vec<_> = std::fs::read_dir(&f.core.paths.library_dir)
        .unwrap()
        .collect();
    assert!(leftovers.is_empty());
    for bad in ["../../etc", "", "0123456789abcdef0123456789ABCDEF"] {
        assert_eq!(
            f.core.get_session(bad).unwrap_err().code,
            ErrorCode::NotFound
        );
        assert_eq!(
            f.core.delete_session(bad).unwrap_err().code,
            ErrorCode::NotFound
        );
        assert!(f.core.prepare_drag(bad, &[StemId::Vocals]).is_err());
    }
}

#[test]
fn crash_mid_split_is_repaired_on_the_next_start() {
    let root = std::env::temp_dir().join(format!("ss-e2e-{}", new_session_id()));
    let id;
    {
        let f = start_with(root.clone(), |p| {
            Arc::new(MockEngine::with_time_scale(engine_config(p), 0.01))
        });
        let s = f
            .core
            .import_track(&song(&root.join("m"), "a.wav").to_string_lossy())
            .unwrap();
        id = s.id.clone();
        // Simulate a crash: mark it splitting behind the app's back.
        f.core
            .library
            .update(&id, |s, _| s.status = SessionStatus::Splitting)
            .unwrap();
        std::fs::create_dir_all(f.core.library.dir(&id).unwrap().join("stems.partial")).unwrap();
        f.core.shutdown();
        std::mem::forget(f); // keep the folder
    }
    let f = start_with(root, |p| {
        Arc::new(MockEngine::with_time_scale(engine_config(p), 0.01))
    });
    let s = f.core.get_session(&id).unwrap();
    assert_eq!(s.status, SessionStatus::Imported);
    assert!(
        !f.core
            .library
            .dir(&id)
            .unwrap()
            .join("stems.partial")
            .exists()
    );
}

struct TrackingEngine {
    inner: MockEngine,
    in_flight_decodes: std::sync::atomic::AtomicUsize,
    max_in_flight_decodes: std::sync::atomic::AtomicUsize,
    in_flight_downloads: std::sync::atomic::AtomicUsize,
    max_in_flight_downloads: std::sync::atomic::AtomicUsize,
    download_calls: std::sync::atomic::AtomicUsize,
    seen_download_ids: std::sync::Mutex<Vec<Vec<String>>>,
    hold_downloads: std::sync::atomic::AtomicBool,
}

impl TrackingEngine {
    fn new(config: EngineConfig) -> Self {
        Self {
            inner: MockEngine::with_time_scale(config, 0.01),
            in_flight_decodes: std::sync::atomic::AtomicUsize::new(0),
            max_in_flight_decodes: std::sync::atomic::AtomicUsize::new(0),
            in_flight_downloads: std::sync::atomic::AtomicUsize::new(0),
            max_in_flight_downloads: std::sync::atomic::AtomicUsize::new(0),
            download_calls: std::sync::atomic::AtomicUsize::new(0),
            seen_download_ids: std::sync::Mutex::new(Vec::new()),
            hold_downloads: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn bump_max(current: &std::sync::atomic::AtomicUsize, max: &std::sync::atomic::AtomicUsize) {
        let n = current.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        max.fetch_max(n, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Engine for TrackingEngine {
    fn name(&self) -> &'static str {
        "tracking"
    }
    fn decode(&self, i: &Path, o: &Path, c: &CancelToken) -> EngineResult<DecodedTrack> {
        Self::bump_max(&self.in_flight_decodes, &self.max_in_flight_decodes);
        std::thread::sleep(Duration::from_millis(30));
        let res = self.inner.decode(i, o, c);
        self.in_flight_decodes
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        res
    }
    fn scan_instruments(&self, _w: &Path, _c: &CancelToken) -> EngineResult<ScanOutcome> {
        Ok(ScanOutcome::Done(ScanResult {
            detections: Vec::new(),
            suggested_mode: SplitMode::Band,
        }))
    }
    fn plan_split(&self, d: f64, o: &SplitOptions) -> EngineResult<SplitPlan> {
        self.inner.plan_split(d, o)
    }
    fn split(
        &self,
        r: &SplitRequest,
        p: &mut (dyn FnMut(EngineProgress) + Send),
        c: &CancelToken,
    ) -> EngineResult<SplitOutput> {
        self.inner.split(r, p, c)
    }
    fn export(&self, j: &ExportJob, c: &CancelToken) -> EngineResult<Vec<PathBuf>> {
        self.inner.export(j, c)
    }
    fn list_models(&self) -> Vec<ModelStatus> {
        self.inner.list_models()
    }
    fn download_models(
        &self,
        ids: &[String],
        a: bool,
        p: &mut (dyn FnMut(ModelDownloadProgress) + Send),
        c: &CancelToken,
    ) -> EngineResult<()> {
        Self::bump_max(&self.in_flight_downloads, &self.max_in_flight_downloads);
        self.download_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.seen_download_ids.lock().unwrap().push(ids.to_vec());
        if self
            .hold_downloads
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            while !c.is_cancelled() {
                std::thread::sleep(Duration::from_millis(5));
            }
            self.in_flight_downloads
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            return Err(EngineError::cancelled());
        }
        let res = self.inner.download_models(ids, a, p, c);
        self.in_flight_downloads
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        res
    }
    fn remove_model(&self, id: &str) -> EngineResult<()> {
        self.inner.remove_model(id)
    }
    fn import_model_files(&self, p: &[PathBuf]) -> EngineResult<Vec<String>> {
        self.inner.import_model_files(p)
    }
    fn hardware(&self) -> HardwareInfo {
        self.inner.hardware()
    }
}

#[test]
fn imports_serialize_and_downloads_use_a_single_bounded_worker() {
    use std::sync::atomic::Ordering;

    let root = std::env::temp_dir().join(format!("ss-e2e-{}", new_session_id()));
    let tracker: Arc<std::sync::OnceLock<Arc<TrackingEngine>>> =
        Arc::new(std::sync::OnceLock::new());
    let t_init = tracker.clone();
    let f = start_with(root, move |p| {
        let engine = Arc::new(TrackingEngine::new(engine_config(p)));
        t_init.set(engine.clone()).ok();
        engine
    });
    let tracker = tracker.get().unwrap().clone();

    // Concurrent import_track calls serialize rather than running parallel decodes.
    let input = song(&f.root.join("music"), "burst.wav");
    let path_str = input.to_string_lossy().into_owned();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let core = f.core.clone();
            let p = path_str.clone();
            std::thread::spawn(move || core.import_track(&p).unwrap())
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(tracker.max_in_flight_decodes.load(Ordering::SeqCst), 1);

    // Burst of download_models calls stays on one worker, deduplicates ids,
    // bounds the queue, and cancels both active and queued requests.
    tracker.hold_downloads.store(true, Ordering::SeqCst);
    f.core
        .download_models(vec!["scnet_small".into(), "scnet_small".into()], false)
        .unwrap();
    wait_until(
        "first download to enter worker",
        Duration::from_secs(5),
        || tracker.in_flight_downloads.load(Ordering::SeqCst) == 1,
    );
    for _ in 0..32 {
        f.core
            .download_models(vec!["scnet_small".into()], false)
            .unwrap();
    }
    let overflow = f
        .core
        .download_models(vec!["scnet_small".into()], false)
        .unwrap_err();
    assert_eq!(overflow.code, ErrorCode::Busy);
    assert_eq!(tracker.max_in_flight_downloads.load(Ordering::SeqCst), 1);

    f.core.cancel_model_downloads();
    wait_until("active download to stop", Duration::from_secs(5), || {
        tracker.in_flight_downloads.load(Ordering::SeqCst) == 0
    });
    std::thread::sleep(Duration::from_millis(50));
    // Queued requests were drained on cancel and never started.
    assert_eq!(tracker.download_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *tracker.seen_download_ids.lock().unwrap(),
        vec![vec!["scnet_small".to_string()]]
    );
}

#[test]
fn prepare_drag_deduplicates_stems_and_ignores_temp_symlinks() {
    let f = start();
    let input = song(&f.root.join("music"), "song.wav");
    let s = f.core.import_track(&input.to_string_lossy()).unwrap();
    let opts = SplitOptions {
        mode: SplitMode::Band,
        quality: Quality::Fast,
    };
    f.core.start_split(&s.id, opts, false).unwrap();
    wait_until("the split", Duration::from_secs(30), || {
        f.core.get_session(&s.id).unwrap().status == SessionStatus::Ready
    });

    #[cfg(unix)]
    let (victim, symlink) = {
        let drag = f.core.paths.drag_dir.join(&s.id);
        std::fs::create_dir_all(&drag).unwrap();
        let victim = f.root.join("victim.txt");
        std::fs::write(&victim, b"untouched").unwrap();
        let symlink = drag.join(".song (Vocals).wav.part");
        std::os::unix::fs::symlink(&victim, &symlink).unwrap();
        (victim, symlink)
    };

    let repeated = vec![
        StemId::Vocals,
        StemId::Bass,
        StemId::Vocals,
        StemId::Vocals,
        StemId::Bass,
    ];
    let drag = f.core.prepare_drag(&s.id, &repeated).unwrap();
    assert_eq!(drag.len(), 2);
    assert!(drag[0].ends_with("song (Vocals).wav"));
    assert!(drag[1].ends_with("song (Bass).wav"));

    #[cfg(unix)]
    {
        assert_eq!(std::fs::read(&victim).unwrap(), b"untouched");
        assert!(symlink.symlink_metadata().is_ok());
    }
}
