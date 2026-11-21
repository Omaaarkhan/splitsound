//! Playback: the mixer thread and the API the `playback_*` commands call.
//!
//! ```text
//!  commands ──MixerCmd──▶ mixer thread ──ring buffer──▶ audio callback
//!     │                    │  MixerCore (44.1 kHz)        (output.rs)
//!     │                    │  Resampler → device rate
//!     ▼                    ▼
//!  PlaybackState      playback:tick (≈30/s), playback:state at the end
//! ```
//!
//! The mixer thread renders blocks ahead of the device into the ring and
//! remembers, per block, where in the song it ends and how loud each
//! channel was. The callback counts the frames it plays, so the position
//! and meters reported in ticks are those of what is actually being heard,
//! not of what was rendered ~50 ms earlier.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use rtrb::Producer;

use super::mapped::MappedWav;
use super::mixer::{self, ChannelParams, MixerCore, RATE, RenderOutcome, Sources};
use super::output::{ActivityFlag, BackendFactory, DeviceController, OutputLink, RING_FRAMES};
use super::resample::Resampler;
use crate::ipc::{
    AppEvent, ChannelState, ErrorCode, EventSink, IpcError, IpcResult, Level, LoopRange,
    PlaybackSource, PlaybackState, PlaybackTick, Session, SourceId, StemId,
};
use crate::util::lock;

/// Frames (at the device rate) rendered per block.
const BLOCK: usize = 256;
/// How far ahead of the device the mixer renders.
const AHEAD_MS: usize = 50;
const TICK_EVERY: Duration = Duration::from_millis(33);
const MAX_METERS: usize = mixer::MAX_CHANNELS + 1;
/// How long synchronous unload/detach waits for the mixer thread to drop its
/// memory-mapped files before surfacing a `busy` error.
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(5);

/// Audio of one loaded session.
pub struct SessionAudio {
    pub session_id: String,
    pub mix: MappedWav,
    /// Stems in lane order.
    pub stems: Vec<(StemId, MappedWav)>,
}

/// Settings applied when audio is (re)loaded.
#[derive(Clone)]
struct MixerInit {
    channels: Vec<ChannelParams>,
    master_db: f64,
    original: bool,
    loop_frames: Option<(u64, u64)>,
    position: u64,
}

enum MixerCmd {
    Load(Box<SessionAudio>, MixerInit),
    Unload(Option<Sender<()>>),
    /// Fence: acknowledges once every earlier command (including any prior
    /// Load/Unload replacing `self.audio`) has completed and dropped old files.
    Sync(Sender<()>),
    Play,
    Pause,
    Seek(u64),
    Channel(usize, ChannelParams),
    Master(f64),
    Original(bool),
    Loop(Option<(u64, u64)>),
    Output(OutputLink),
    DeviceLost(String),
    #[cfg(test)]
    HoldForTest(Receiver<()>),
    Shutdown,
}

struct Shared {
    state: Mutex<PlaybackState>,
    /// Song position (44.1 kHz frames) of what the listener hears now.
    heard: AtomicU64,
    events: Arc<dyn EventSink>,
    activity: Arc<ActivityFlag>,
}

impl Shared {
    fn snapshot(&self, st: &PlaybackState) -> PlaybackState {
        let mut s = st.clone();
        if s.session_id.is_some() {
            s.position_sec =
                (self.heard.load(Ordering::Relaxed) as f64 / RATE as f64).min(s.duration_sec);
        }
        s
    }

    fn emit_state(&self, st: &PlaybackState) {
        self.events.emit(AppEvent::PlaybackState(self.snapshot(st)));
    }
}

fn empty_state(master_gain_db: f64) -> PlaybackState {
    PlaybackState {
        session_id: None,
        playing: false,
        position_sec: 0.0,
        duration_sec: 0.0,
        source: PlaybackSource::Original,
        loop_range: None,
        master_gain_db,
        channels: Vec::new(),
    }
}

fn sec_to_frames(sec: f64) -> u64 {
    if sec.is_finite() && sec > 0.0 {
        (sec * RATE as f64).round() as u64
    } else {
        0
    }
}

fn params_of(c: &ChannelState) -> ChannelParams {
    ChannelParams {
        gain_db: c.gain_db,
        muted: c.muted,
        solo: c.solo,
    }
}

/// Normalises a loop request: finite, ordered, inside the track and at
/// least 10 ms long; anything else means "no loop".
fn normalise_loop(l: Option<LoopRange>, duration: f64) -> Option<LoopRange> {
    let l = l?;
    if !l.start_sec.is_finite() || !l.end_sec.is_finite() {
        return None;
    }
    let start = l.start_sec.clamp(0.0, duration);
    let end = l.end_sec.clamp(0.0, duration);
    (end - start >= 0.01).then_some(LoopRange {
        start_sec: start,
        end_sec: end,
    })
}

fn loop_frames(l: Option<LoopRange>) -> Option<(u64, u64)> {
    l.map(|l| (sec_to_frames(l.start_sec), sec_to_frames(l.end_sec)))
}

/// The playback engine. One session is loaded at a time.
pub struct Player {
    tx: Sender<MixerCmd>,
    shared: Arc<Shared>,
    factory: BackendFactory,
    device: OnceLock<DeviceController>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Player {
    /// Starts the mixer thread. The audio device is opened lazily, on the
    /// first `play`.
    pub fn new(events: Arc<dyn EventSink>, factory: BackendFactory) -> Self {
        let (tx, rx) = mpsc::channel();
        let shared = Arc::new(Shared {
            state: Mutex::new(empty_state(0.0)),
            heard: AtomicU64::new(0),
            events,
            activity: Arc::new(ActivityFlag::default()),
        });
        let sh = shared.clone();
        let thread = thread::Builder::new()
            .name("splitsound-mixer".into())
            .spawn(move || MixerThread::new(rx, sh).run())
            .expect("failed to spawn the mixer thread");
        Player {
            tx,
            shared,
            factory,
            device: OnceLock::new(),
            thread: Some(thread),
        }
    }

    fn send(&self, cmd: MixerCmd) {
        let _ = self.tx.send(cmd);
    }

    fn device(&self) -> &DeviceController {
        self.device.get_or_init(|| {
            let tx = self.tx.clone();
            let tx2 = self.tx.clone();
            DeviceController::spawn(
                self.factory.clone(),
                self.shared.activity.clone(),
                move |link| {
                    let _ = tx.send(MixerCmd::Output(link));
                },
                move |msg| {
                    let _ = tx2.send(MixerCmd::DeviceLost(msg));
                },
            )
        })
    }

    /// Current state, with the position of what is being heard.
    pub fn state(&self) -> PlaybackState {
        let st = lock(&self.shared.state);
        self.shared.snapshot(&st)
    }

    /// Id of the loaded session, if any.
    pub fn loaded_session(&self) -> Option<String> {
        lock(&self.shared.state).session_id.clone()
    }

    fn open_audio(session: &Session, dir: &Path) -> IpcResult<SessionAudio> {
        let mix = MappedWav::open(&dir.join("mix.wav")).map_err(|e| {
            log::warn!("can't open mix for {}: {e}", session.id);
            IpcError::new(
                ErrorCode::FileNotFound,
                "This song's audio file is missing or damaged. Try importing the song again.",
            )
        })?;
        let mut stems = Vec::with_capacity(session.stems.len());
        for s in &session.stems {
            // Resolve inside the session folder; never trust a stored path.
            let p = dir.join("stems").join(format!("{}.wav", s.id.as_str()));
            let w = MappedWav::open(&p).map_err(|e| {
                log::warn!("can't open stem {:?} for {}: {e}", s.id, session.id);
                IpcError::new(
                    ErrorCode::FileNotFound,
                    "Some of this song's stems are missing or damaged. Split it again to recreate them.",
                )
            })?;
            stems.push((s.id, w));
        }
        stems.sort_by_key(|(id, _)| id.lane_index());
        Ok(SessionAudio {
            session_id: session.id.clone(),
            mix,
            stems,
        })
    }

    /// Loads a session (its folder is `dir`). Channels start at 0 dB,
    /// unmuted; the master gain is kept. Plays stems when the session has
    /// them, the original otherwise.
    pub fn load(&self, session: &Session, dir: &Path) -> IpcResult<PlaybackState> {
        let audio = Self::open_audio(session, dir)?;
        let frames = audio.mix.frames();
        let mut st = lock(&self.shared.state);
        let has_stems = !audio.stems.is_empty();
        let next = PlaybackState {
            session_id: Some(session.id.clone()),
            playing: false,
            position_sec: 0.0,
            duration_sec: frames as f64 / RATE as f64,
            source: if has_stems {
                PlaybackSource::Stems
            } else {
                PlaybackSource::Original
            },
            loop_range: None,
            master_gain_db: st.master_gain_db,
            channels: audio
                .stems
                .iter()
                .map(|(id, _)| ChannelState {
                    id: SourceId::from(*id),
                    gain_db: 0.0,
                    muted: false,
                    solo: false,
                })
                .collect(),
        };
        let init = MixerInit {
            channels: next.channels.iter().map(params_of).collect(),
            master_db: next.master_gain_db,
            original: !has_stems,
            loop_frames: None,
            position: 0,
        };
        *st = next;
        self.shared.heard.store(0, Ordering::Relaxed);
        self.send(MixerCmd::Load(Box::new(audio), init));
        self.shared.emit_state(&st);
        Ok(self.shared.snapshot(&st))
    }

    fn wait_ack(rx: Receiver<()>, timeout: Duration) -> IpcResult<()> {
        match rx.recv_timeout(timeout) {
            Ok(()) => Ok(()),
            // If the mixer thread exited, it already dropped `self.audio`
            // before dropping its receiver.
            Err(RecvTimeoutError::Disconnected) => Ok(()),
            Err(RecvTimeoutError::Timeout) => Err(IpcError::busy(
                "The audio player is still closing this song. Try again shortly.",
            )),
        }
    }

    fn unload_with_timeout(&self, timeout: Duration) -> IpcResult<()> {
        let (tx, rx) = mpsc::channel();
        {
            let mut st = lock(&self.shared.state);
            if st.session_id.is_some() {
                *st = empty_state(st.master_gain_db);
                self.shared.heard.store(0, Ordering::Relaxed);
                self.send(MixerCmd::Unload(Some(tx)));
                self.shared.emit_state(&st);
            } else {
                // Fence behind any earlier Load/Unload so no mapping is still
                // being dropped on the mixer thread when we return.
                self.send(MixerCmd::Sync(tx));
            }
        }
        // Wait (without holding the state lock) until the files are unmapped.
        Self::wait_ack(rx, timeout)
    }

    /// Unloads whatever is loaded and waits until the mixer thread has
    /// dropped every memory-mapped file.
    pub fn unload(&self) -> IpcResult<()> {
        self.unload_with_timeout(UNLOAD_TIMEOUT)
    }

    /// Unloads `session_id` if it is the loaded session, and always fences
    /// with the mixer thread so any recently replaced mappings of `session_id`
    /// are guaranteed to be dropped before this returns.
    pub fn unload_if(&self, session_id: &str) -> IpcResult<()> {
        let (tx, rx) = mpsc::channel();
        {
            let mut st = lock(&self.shared.state);
            if st.session_id.as_deref() == Some(session_id) {
                *st = empty_state(st.master_gain_db);
                self.shared.heard.store(0, Ordering::Relaxed);
                self.send(MixerCmd::Unload(Some(tx)));
                self.shared.emit_state(&st);
            } else {
                // Another session may have just replaced `session_id` via
                // `load`; wait for the mixer thread to apply that Load and
                // drop `session_id`'s old `SessionAudio`.
                self.send(MixerCmd::Sync(tx));
            }
        }
        Self::wait_ack(rx, UNLOAD_TIMEOUT)
    }

    /// Releases the files of `session_id` (so its stems folder can be
    /// replaced) while keeping it loaded in the UI. Always fences with the
    /// mixer thread so no mapping of `session_id`'s files remains alive when
    /// this returns. Returns whether it was loaded. Follow with
    /// [`Player::reattach`].
    pub fn detach(&self, session_id: &str) -> IpcResult<bool> {
        let (tx, rx) = mpsc::channel();
        let was_loaded = {
            let mut st = lock(&self.shared.state);
            if st.session_id.as_deref() == Some(session_id) {
                self.send(MixerCmd::Unload(Some(tx)));
                if st.playing {
                    st.playing = false;
                    self.shared.emit_state(&st);
                }
                true
            } else {
                self.send(MixerCmd::Sync(tx));
                false
            }
        };
        Self::wait_ack(rx, UNLOAD_TIMEOUT)?;
        Ok(was_loaded)
    }

    /// Refreshes the working files of the loaded session while keeping its
    /// transport and mixer settings.
    pub fn reattach(&self, session: &Session, dir: &Path) {
        if self.loaded_session().as_deref() != Some(&session.id) {
            return;
        }
        let audio = match Self::open_audio(session, dir) {
            Ok(a) => a,
            Err(_) => {
                let _ = self.unload_if(&session.id);
                return;
            }
        };
        let mut st = lock(&self.shared.state);
        if st.session_id.as_deref() != Some(session.id.as_str()) {
            return;
        }
        let old = std::mem::take(&mut st.channels);
        st.channels = audio
            .stems
            .iter()
            .map(|(id, _)| {
                let sid = SourceId::from(*id);
                old.iter()
                    .find(|c| c.id == sid)
                    .cloned()
                    .unwrap_or(ChannelState {
                        id: sid,
                        gain_db: 0.0,
                        muted: false,
                        solo: false,
                    })
            })
            .collect();
        if st.channels.is_empty() {
            st.source = PlaybackSource::Original;
        } else if old.is_empty() && !st.playing {
            // First split of a song that was loaded: switch to the stems.
            st.source = PlaybackSource::Stems;
        }
        st.duration_sec = audio.mix.frames() as f64 / RATE as f64;
        let init = MixerInit {
            channels: st.channels.iter().map(params_of).collect(),
            master_db: st.master_gain_db,
            original: st.source == PlaybackSource::Original,
            loop_frames: loop_frames(st.loop_range),
            position: self.shared.heard.load(Ordering::Relaxed),
        };
        self.send(MixerCmd::Load(Box::new(audio), init));
        if st.playing {
            self.send(MixerCmd::Play);
        }
        self.shared.emit_state(&st);
    }

    fn require_loaded(st: &PlaybackState) -> IpcResult<()> {
        if st.session_id.is_none() {
            Err(IpcError::not_found("No song is loaded."))
        } else {
            Ok(())
        }
    }

    /// Starts playback. Opens the output device if needed; rejects with
    /// `audio_device` if there is none.
    pub fn play(&self) -> IpcResult<()> {
        Self::require_loaded(&lock(&self.shared.state))?;
        self.device()
            .ensure_open(Duration::from_secs(4))
            .map_err(|m| IpcError::new(ErrorCode::AudioDevice, m))?;
        let mut st = lock(&self.shared.state);
        Self::require_loaded(&st)?;
        if st.playing {
            return Ok(());
        }
        st.playing = true;
        // Playing from the very end restarts (the mixer does the same).
        let heard = self.shared.heard.load(Ordering::Relaxed);
        if heard >= sec_to_frames(st.duration_sec) {
            let start = st.loop_range.map_or(0, |l| sec_to_frames(l.start_sec));
            self.shared.heard.store(start, Ordering::Relaxed);
        }
        self.send(MixerCmd::Play);
        self.shared.emit_state(&st);
        Ok(())
    }

    pub fn pause(&self) {
        let mut st = lock(&self.shared.state);
        if st.playing {
            st.playing = false;
            self.send(MixerCmd::Pause);
            self.shared.emit_state(&st);
        }
    }

    pub fn seek(&self, position_sec: f64) -> IpcResult<()> {
        let st = lock(&self.shared.state);
        Self::require_loaded(&st)?;
        let sec = if position_sec.is_finite() {
            position_sec.clamp(0.0, st.duration_sec)
        } else {
            0.0
        };
        let frames = sec_to_frames(sec);
        self.send(MixerCmd::Seek(frames));
        if !st.playing {
            self.shared.heard.store(frames, Ordering::Relaxed);
            self.shared
                .events
                .emit(AppEvent::PlaybackTick(PlaybackTick {
                    position_sec: sec,
                    levels: zero_levels(&st),
                }));
        }
        Ok(())
    }

    pub fn set_channel(
        &self,
        id: SourceId,
        gain_db: Option<f64>,
        muted: Option<bool>,
        solo: Option<bool>,
    ) -> IpcResult<()> {
        let mut st = lock(&self.shared.state);
        Self::require_loaded(&st)?;
        let Some(index) = st.channels.iter().position(|c| c.id == id) else {
            return Err(IpcError::not_found(
                "That stem isn't part of the loaded song.",
            ));
        };
        let ch = &mut st.channels[index];
        if let Some(g) = gain_db {
            ch.gain_db = mixer::clamp_db(g);
        }
        if let Some(m) = muted {
            ch.muted = m;
        }
        if let Some(s) = solo {
            ch.solo = s;
        }
        let p = params_of(ch);
        self.send(MixerCmd::Channel(index, p));
        self.shared.emit_state(&st);
        Ok(())
    }

    pub fn set_master(&self, gain_db: f64) {
        let mut st = lock(&self.shared.state);
        st.master_gain_db = mixer::clamp_db(gain_db);
        self.send(MixerCmd::Master(st.master_gain_db));
        self.shared.emit_state(&st);
    }

    pub fn set_source(&self, source: PlaybackSource) -> IpcResult<()> {
        let mut st = lock(&self.shared.state);
        Self::require_loaded(&st)?;
        if source == PlaybackSource::Stems && st.channels.is_empty() {
            return Err(IpcError::not_found("This song hasn't been split yet."));
        }
        if st.source != source {
            st.source = source;
            self.send(MixerCmd::Original(source == PlaybackSource::Original));
            self.shared.emit_state(&st);
        }
        Ok(())
    }

    /// Sets or clears the loop. Requests that are not a valid range inside
    /// the song (non-finite, reversed, shorter than 10 ms) clear it.
    pub fn set_loop(&self, range: Option<LoopRange>) -> IpcResult<()> {
        let mut st = lock(&self.shared.state);
        Self::require_loaded(&st)?;
        let l = normalise_loop(range, st.duration_sec);
        st.loop_range = l;
        if let Some(l) = l {
            let heard = self.shared.heard.load(Ordering::Relaxed);
            if heard >= sec_to_frames(l.end_sec) && !st.playing {
                self.shared
                    .heard
                    .store(sec_to_frames(l.start_sec), Ordering::Relaxed);
            }
        }
        self.send(MixerCmd::Loop(loop_frames(l)));
        self.shared.emit_state(&st);
        Ok(())
    }

    /// Linear gains for a mixdown of `session_id` at the current mixer
    /// settings (mute/solo applied), plus the master gain. `None` if that
    /// session is not loaded.
    pub fn mixdown_gains(&self, session_id: &str) -> Option<(Vec<(StemId, f32)>, f32)> {
        let st = lock(&self.shared.state);
        if st.session_id.as_deref() != Some(session_id) {
            return None;
        }
        let params: Vec<ChannelParams> = st.channels.iter().map(params_of).collect();
        let mut gains = vec![0f32; params.len()];
        mixer::effective_gains(&params, &mut gains);
        let stems = st
            .channels
            .iter()
            .zip(gains)
            .filter_map(|(c, g)| c.id.stem().map(|s| (s, g)))
            .collect();
        Some((stems, mixer::db_to_gain(st.master_gain_db)))
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        let _ = self.tx.send(MixerCmd::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn zero_levels(st: &PlaybackState) -> Vec<Level> {
    st.channels
        .iter()
        .map(|c| Level {
            id: c.id,
            peak: 0.0,
        })
        .chain(std::iter::once(Level {
            id: SourceId::Mix,
            peak: 0.0,
        }))
        .collect()
}

/* ------------------------------------------------------------------ */
/* Mixer thread                                                        */
/* ------------------------------------------------------------------ */

struct Mark {
    /// Cumulative device frames pushed when this block ends.
    end: u64,
    /// Song position after this block.
    pos: u64,
    peaks: [f32; MAX_METERS],
    ended: bool,
    stopped: bool,
}

struct Out {
    producer: Producer<f32>,
    played: Arc<AtomicU64>,
    rate: u32,
    resampler: Resampler,
    pushed: u64,
}

struct MixerThread {
    // Declared before `rx` so mapped files always drop before `rx` disconnects.
    audio: Option<Box<SessionAudio>>,
    rx: Receiver<MixerCmd>,
    shared: Arc<Shared>,
    core: MixerCore,
    out: Option<Out>,
    marks: VecDeque<Mark>,
    block: Vec<f32>,
    tick_peaks: [f32; MAX_METERS],
    last_tick: Instant,
    ids: Vec<SourceId>,
}

impl MixerThread {
    fn new(rx: Receiver<MixerCmd>, shared: Arc<Shared>) -> Self {
        MixerThread {
            audio: None,
            rx,
            shared,
            core: MixerCore::new(0, 0),
            out: None,
            marks: VecDeque::with_capacity(RING_FRAMES / BLOCK + 8),
            block: vec![0.0; BLOCK * 2],
            tick_peaks: [0.0; MAX_METERS],
            last_tick: Instant::now(),
            ids: Vec::new(),
        }
    }

    fn run(mut self) {
        loop {
            let busy = self.core.is_active() || !self.marks.is_empty();
            let wait = if busy {
                Duration::from_millis(3)
            } else {
                Duration::from_millis(100)
            };
            match self.rx.recv_timeout(wait) {
                Ok(cmd) => {
                    if !self.handle(cmd) {
                        drop(self.audio.take());
                        return;
                    }
                    while let Ok(cmd) = self.rx.try_recv() {
                        if !self.handle(cmd) {
                            drop(self.audio.take());
                            return;
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    drop(self.audio.take());
                    return;
                }
            }
            self.render_ahead();
            self.follow_device();
        }
    }

    /// Returns false on shutdown.
    fn handle(&mut self, cmd: MixerCmd) -> bool {
        match cmd {
            MixerCmd::Load(audio, init) => {
                // Drop any previous session's mappings before installing the new one.
                drop(self.audio.take());
                self.core = MixerCore::new(audio.stems.len(), audio.mix.frames());
                for (i, p) in init.channels.iter().enumerate() {
                    self.core.set_channel(i, *p);
                }
                self.core.set_master_db(init.master_db);
                self.core.set_original(init.original);
                self.core.set_loop(init.loop_frames);
                self.core.seek(init.position);
                self.ids = audio
                    .stems
                    .iter()
                    .map(|(id, _)| SourceId::from(*id))
                    .collect();
                self.audio = Some(audio);
                self.marks.clear();
                self.tick_peaks = [0.0; MAX_METERS];
            }
            MixerCmd::Unload(ack) => {
                drop(self.audio.take());
                self.core = MixerCore::new(0, 0);
                self.ids.clear();
                self.marks.clear();
                if let Some(ack) = ack {
                    let _ = ack.send(());
                }
            }
            MixerCmd::Sync(ack) => {
                let _ = ack.send(());
            }
            MixerCmd::Play => self.core.play(),
            MixerCmd::Pause => self.core.pause(),
            MixerCmd::Seek(f) => self.core.seek(f),
            MixerCmd::Channel(i, p) => self.core.set_channel(i, p),
            MixerCmd::Master(db) => self.core.set_master_db(db),
            MixerCmd::Original(o) => self.core.set_original(o),
            MixerCmd::Loop(l) => self.core.set_loop(l),
            MixerCmd::Output(link) => {
                // Anything rendered for the old stream is gone: resume from
                // what was last heard.
                self.marks.clear();
                if self.audio.is_some() {
                    self.core.seek(self.shared.heard.load(Ordering::Relaxed));
                }
                self.out = Some(Out {
                    producer: link.producer,
                    played: link.played,
                    rate: link.sample_rate,
                    resampler: Resampler::new(RATE, link.sample_rate),
                    pushed: 0,
                });
            }
            MixerCmd::DeviceLost(msg) => {
                log::warn!("playback stopped: {msg}");
                self.out = None;
                self.marks.clear();
                self.core.stop_now();
                self.core.seek(self.shared.heard.load(Ordering::Relaxed));
                let mut st = lock(&self.shared.state);
                if st.playing {
                    st.playing = false;
                    self.shared.emit_state(&st);
                }
            }
            #[cfg(test)]
            MixerCmd::HoldForTest(release) => {
                let _ = release.recv();
            }
            MixerCmd::Shutdown => {
                drop(self.audio.take());
                return false;
            }
        }
        true
    }

    fn render_ahead(&mut self) {
        let (Some(out), Some(audio)) = (&mut self.out, &self.audio) else {
            self.shared.activity.set(!self.marks.is_empty());
            return;
        };
        self.shared
            .activity
            .set(self.core.is_active() || !self.marks.is_empty());
        if out.producer.is_abandoned() {
            return;
        }
        let target = (out.rate as usize * AHEAD_MS / 1000).max(BLOCK * 2);
        let mut arr: [&[f32]; mixer::MAX_CHANNELS] = [&[]; mixer::MAX_CHANNELS];
        for (i, (_, w)) in audio.stems.iter().enumerate().take(mixer::MAX_CHANNELS) {
            arr[i] = w.samples();
        }
        let n = audio.stems.len().min(mixer::MAX_CHANNELS);
        let src = Sources {
            stems: &arr[..n],
            mix: audio.mix.samples(),
        };
        while self.core.is_active() {
            let free = out.producer.slots() / 2;
            let filled = RING_FRAMES - free;
            if filled >= target || free < BLOCK {
                break;
            }
            let mut outcome = RenderOutcome::default();
            let core = &mut self.core;
            out.resampler.process(&mut self.block, |buf| {
                let o = core.render(&src, buf);
                outcome.reached_end |= o.reached_end;
                outcome.stopped |= o.stopped;
            });
            if let Ok(chunk) = out.producer.write_chunk_uninit(BLOCK * 2) {
                let _ = chunk.fill_from_iter(self.block.iter().copied());
            }
            out.pushed += BLOCK as u64;
            let mut peaks = [0f32; MAX_METERS];
            self.core.take_meters(&mut peaks[..n + 1]);
            // Keep the master meter in the last slot regardless of channel count.
            if n + 1 < MAX_METERS {
                peaks[MAX_METERS - 1] = peaks[n];
                peaks[n] = 0.0;
            }
            self.marks.push_back(Mark {
                end: out.pushed,
                pos: self.core.position(),
                peaks,
                ended: outcome.reached_end,
                stopped: outcome.stopped,
            });
            if outcome.reached_end || outcome.stopped {
                break;
            }
        }
    }

    /// Retires blocks the device has played: updates the heard position,
    /// sends ticks and handles end-of-track.
    fn follow_device(&mut self) {
        let Some(out) = &self.out else { return };
        let played = out.played.load(Ordering::Relaxed);
        let mut consumed = false;
        let mut ended = false;
        let mut stopped = false;
        while self.marks.front().is_some_and(|m| m.end <= played) {
            let m = self.marks.pop_front().expect("checked");
            for (a, b) in self.tick_peaks.iter_mut().zip(m.peaks) {
                *a = a.max(b);
            }
            self.shared.heard.store(m.pos, Ordering::Relaxed);
            ended |= m.ended;
            stopped |= m.stopped;
            consumed = true;
        }
        if ended {
            let mut st = lock(&self.shared.state);
            if st.session_id.is_some() {
                st.playing = false;
                self.shared.emit_state(&st);
            }
        }
        let due = self.last_tick.elapsed() >= TICK_EVERY;
        if (consumed && due) || ended || stopped {
            self.emit_tick(ended || stopped);
        }
    }

    fn emit_tick(&mut self, silent: bool) {
        self.last_tick = Instant::now();
        let peaks = std::mem::replace(&mut self.tick_peaks, [0.0; MAX_METERS]);
        let level = |p: f32| if silent { 0.0 } else { p as f64 };
        let mut levels: Vec<Level> = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, id)| Level {
                id: *id,
                peak: level(peaks[i]),
            })
            .collect();
        levels.push(Level {
            id: SourceId::Mix,
            peak: level(peaks[MAX_METERS - 1]),
        });
        let position_sec = self.shared.heard.load(Ordering::Relaxed) as f64 / RATE as f64;
        self.shared
            .events
            .emit(AppEvent::PlaybackTick(PlaybackTick {
                position_sec,
                levels,
            }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::output::{AudioBackend, FakeBackend, NoDeviceBackend};
    use crate::audio::wav;
    use crate::ipc::{
        AnalysisStatus, RecordingSink, SessionStatus, SplitMode, StemInfo, TrackInfo,
    };
    use std::path::PathBuf;

    fn session(id: &str, dir: &Path, stems: &[StemId]) -> Session {
        Session {
            id: id.into(),
            created_at: "2026-01-01T00:00:00.000Z".into(),
            track: TrackInfo {
                path: "/x.wav".into(),
                file_name: "x.wav".into(),
                title: "X".into(),
                artist: None,
                album: None,
                duration_sec: 0.5,
                sample_rate: 44_100,
                channels: 2,
                codec: "WAV".into(),
                bitrate_kbps: None,
                file_size_bytes: 0,
                has_artwork: false,
            },
            status: if stems.is_empty() {
                SessionStatus::Imported
            } else {
                SessionStatus::Ready
            },
            analysis: AnalysisStatus::Done,
            detections: vec![],
            suggested_mode: SplitMode::Band,
            split: None,
            stems: stems
                .iter()
                .map(|s| StemInfo {
                    id: *s,
                    path: dir
                        .join("stems")
                        .join(format!("{}.wav", s.as_str()))
                        .to_string_lossy()
                        .into(),
                    present: true,
                    active_fraction: 1.0,
                    peak_db: 0.0,
                    rms_db: 0.0,
                    activity: vec![],
                    contains: vec![],
                })
                .collect(),
            error: None,
        }
    }

    /// Writes a 0.5 s session with a quiet mix and a loud vocals stem.
    fn fixture(stems: &[StemId]) -> (PathBuf, Session) {
        let dir = std::env::temp_dir().join(format!("ss-player-{}", crate::util::new_session_id()));
        std::fs::create_dir_all(dir.join("stems")).unwrap();
        let frames = 22_050;
        wav::write_working(&dir.join("mix.wav"), &vec![0.1; frames * 2]).unwrap();
        for (i, s) in stems.iter().enumerate() {
            let v = if i == 0 { 0.5 } else { 0.05 };
            wav::write_working(
                &dir.join("stems").join(format!("{}.wav", s.as_str())),
                &vec![v; frames * 2],
            )
            .unwrap();
        }
        let id = crate::util::new_session_id();
        let s = session(&id, &dir, stems);
        (dir, s)
    }

    fn fake_factory(speed: f64) -> BackendFactory {
        Arc::new(move || {
            Box::new(FakeBackend {
                sample_rate: 48_000,
                speed,
                captured: Arc::new(Mutex::new(Vec::new())),
            }) as Box<dyn AudioBackend>
        })
    }

    fn wait_for(
        sink: &RecordingSink,
        timeout: Duration,
        pred: impl Fn(&[AppEvent]) -> bool,
    ) -> Vec<AppEvent> {
        let t0 = Instant::now();
        loop {
            let ev = sink.snapshot();
            if pred(&ev) || t0.elapsed() > timeout {
                return ev;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn no_output_device_is_an_audio_device_error() {
        let sink = Arc::new(RecordingSink::default());
        let player = Player::new(
            sink.clone(),
            Arc::new(|| Box::new(NoDeviceBackend) as Box<dyn AudioBackend>),
        );
        let (dir, s) = fixture(&[StemId::Vocals, StemId::Instrumental]);
        let st = player.load(&s, &dir).unwrap();
        assert_eq!(st.channels.len(), 2);
        assert_eq!(st.source, PlaybackSource::Stems);
        let err = player.play().unwrap_err();
        assert_eq!(err.code, ErrorCode::AudioDevice);
        assert!(!player.state().playing);
        // Everything else still works without a device.
        player.seek(0.25).unwrap();
        assert!((player.state().position_sec - 0.25).abs() < 1e-3);
        player
            .set_channel(SourceId::Vocals, Some(-6.0), None, Some(true))
            .unwrap();
        assert_eq!(
            player
                .set_channel(SourceId::Mix, Some(0.0), None, None)
                .unwrap_err()
                .code,
            ErrorCode::NotFound
        );
        player.unload().unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn plays_to_the_end_with_ticks_and_stops() {
        let sink = Arc::new(RecordingSink::default());
        // Twice real time so the 0.5 s file finishes quickly.
        let player = Player::new(sink.clone(), fake_factory(2.0));
        let (dir, s) = fixture(&[StemId::Vocals, StemId::Instrumental]);
        player.load(&s, &dir).unwrap();
        player
            .set_channel(SourceId::Instrumental, None, Some(true), None)
            .unwrap();
        sink.take();
        player.play().unwrap();
        let events = wait_for(&sink, Duration::from_secs(5), |ev| {
            ev.iter()
                .any(|e| matches!(e, AppEvent::PlaybackState(s) if !s.playing))
        });
        let ticks: Vec<&PlaybackTick> = events
            .iter()
            .filter_map(|e| match e {
                AppEvent::PlaybackTick(t) => Some(t),
                _ => None,
            })
            .collect();
        assert!(ticks.len() >= 3, "got {} ticks", ticks.len());
        // Positions never go backwards and end at the end.
        assert!(
            ticks
                .windows(2)
                .all(|w| w[1].position_sec >= w[0].position_sec)
        );
        let last = player.state();
        assert!(!last.playing);
        assert!(
            (last.position_sec - 0.5).abs() < 0.01,
            "{}",
            last.position_sec
        );
        // Levels: vocals at 0.5, muted instrumental silent, master = vocals.
        let loud = ticks
            .iter()
            .find(|t| {
                t.levels
                    .iter()
                    .any(|l| l.id == SourceId::Vocals && l.peak > 0.4)
            })
            .expect("a tick with vocals audible");
        assert_eq!(loud.levels.len(), 3);
        assert_eq!(loud.levels[1].id, SourceId::Instrumental);
        assert!(loud.levels[1].peak < 0.01);
        assert_eq!(loud.levels[2].id, SourceId::Mix);
        assert!((loud.levels[2].peak - 0.5).abs() < 0.05);
        // Play again from the end restarts.
        player.play().unwrap();
        thread::sleep(Duration::from_millis(100));
        assert!(player.state().position_sec < 0.45);
        player.pause();
        player.unload().unwrap();
        assert_eq!(player.state().session_id, None);
        assert_eq!(MappedWav::live_mappings_under(&dir), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Plays half a second of silence on the real default output device.
    /// Ignored by default (CI has no device); run with `--ignored`.
    #[test]
    #[ignore]
    fn real_device_smoke_test() {
        let sink = Arc::new(RecordingSink::default());
        let player = Player::new(sink.clone(), crate::audio::output::CpalBackend::factory());
        let dir = std::env::temp_dir().join(format!("ss-real-{}", crate::util::new_session_id()));
        std::fs::create_dir_all(dir.join("stems")).unwrap();
        wav::write_working(&dir.join("mix.wav"), &vec![0.0; 22_050 * 2]).unwrap();
        let s = session(&crate::util::new_session_id(), &dir, &[]);
        player.load(&s, &dir).unwrap();
        sink.take();
        match player.play() {
            Ok(()) => {
                let ev = wait_for(&sink, Duration::from_secs(5), |ev| {
                    ev.iter()
                        .any(|e| matches!(e, AppEvent::PlaybackState(s) if !s.playing && s.session_id.is_some()) )
                        && ev.iter().filter(|e| matches!(e, AppEvent::PlaybackTick(_))).count() > 3
                });
                let ticks = ev
                    .iter()
                    .filter(|e| matches!(e, AppEvent::PlaybackTick(_)))
                    .count();
                eprintln!("real device: {:?}, {ticks} ticks", player.device().status());
                assert!(ticks > 3);
                assert!((player.state().position_sec - 0.5).abs() < 0.02);
            }
            Err(e) => {
                eprintln!("no usable output device here: {}", e.message);
                assert_eq!(e.code, ErrorCode::AudioDevice);
            }
        }
        player.unload().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn source_and_loop_validation() {
        let sink = Arc::new(RecordingSink::default());
        let player = Player::new(sink.clone(), fake_factory(1.0));
        let (dir, s) = fixture(&[]);
        let st = player.load(&s, &dir).unwrap();
        assert_eq!(st.source, PlaybackSource::Original);
        assert!(st.channels.is_empty());
        assert_eq!(
            player.set_source(PlaybackSource::Stems).unwrap_err().code,
            ErrorCode::NotFound
        );
        player
            .set_loop(Some(LoopRange {
                start_sec: 0.1,
                end_sec: 9.0,
            }))
            .unwrap();
        assert_eq!(
            player.state().loop_range,
            Some(LoopRange {
                start_sec: 0.1,
                end_sec: 0.5
            })
        );
        player
            .set_loop(Some(LoopRange {
                start_sec: 0.3,
                end_sec: 0.2,
            }))
            .unwrap();
        assert_eq!(player.state().loop_range, None);
        player.set_master(42.0);
        assert_eq!(player.state().master_gain_db, mixer::MAX_GAIN_DB);
        player.set_master(-1000.0);
        assert_eq!(player.state().master_gain_db, mixer::MIN_GAIN_DB);
        // Master survives a reload.
        player.load(&s, &dir).unwrap();
        assert_eq!(player.state().master_gain_db, mixer::MIN_GAIN_DB);
        player.unload().unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mixdown_gains_follow_mute_and_solo() {
        let sink = Arc::new(RecordingSink::default());
        let player = Player::new(sink, fake_factory(1.0));
        let (dir, s) = fixture(&[StemId::Vocals, StemId::Drums, StemId::Other]);
        player.load(&s, &dir).unwrap();
        player
            .set_channel(SourceId::Drums, Some(-6.0), None, Some(true))
            .unwrap();
        player
            .set_channel(SourceId::Vocals, None, Some(true), Some(true))
            .unwrap();
        let (g, master) = player.mixdown_gains(&s.id).unwrap();
        assert_eq!(g[0], (StemId::Vocals, 1.0));
        assert_eq!(g[1].0, StemId::Drums);
        assert!((g[1].1 - mixer::db_to_gain(-6.0)).abs() < 1e-6);
        assert_eq!(g[2], (StemId::Other, 0.0));
        assert_eq!(master, 1.0);
        assert!(player.mixdown_gains("someone-else").is_none());
        player.unload().unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unload_if_and_detach_synchronously_release_all_mappings() {
        let sink = Arc::new(RecordingSink::default());
        let player = Player::new(sink, fake_factory(1.0));
        let (dir_a, a) = fixture(&[StemId::Vocals, StemId::Instrumental]);
        let (dir_b, b) = fixture(&[StemId::Vocals, StemId::Instrumental]);

        player.load(&a, &dir_a).unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir_a), 3);

        // Unloading the loaded session drops every mapping before returning.
        player.unload_if(&a.id).unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir_a), 0);

        // Switching from A to B and immediately calling unload_if(A) still
        // fences with the mixer thread so A's old mappings are dropped before
        // unload_if(A) returns (even though `state.session_id` is already B).
        player.load(&a, &dir_a).unwrap();
        player.load(&b, &dir_b).unwrap();
        player.unload_if(&a.id).unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir_a), 0);
        assert_eq!(MappedWav::live_mappings_under(&dir_b), 3);

        // `detach` releases B's mappings synchronously while keeping B selected.
        assert!(player.detach(&b.id).unwrap());
        assert_eq!(MappedWav::live_mappings_under(&dir_b), 0);
        assert_eq!(player.loaded_session().as_deref(), Some(b.id.as_str()));

        // `reattach` maps them again; `unload` clears everything.
        player.reattach(&b, &dir_b);
        assert_eq!(MappedWav::live_mappings_under(&dir_b), 3);
        player.unload().unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir_b), 0);

        std::fs::remove_dir_all(dir_a).unwrap();
        std::fs::remove_dir_all(dir_b).unwrap();
    }

    #[test]
    fn unload_surfaces_busy_error_when_mixer_does_not_ack_in_time() {
        let sink = Arc::new(RecordingSink::default());
        let player = Player::new(sink, fake_factory(1.0));
        let (dir, s) = fixture(&[StemId::Vocals, StemId::Instrumental]);
        player.load(&s, &dir).unwrap();
        let (tx, rx) = mpsc::channel();
        player.send(MixerCmd::HoldForTest(rx));
        let err = player
            .unload_with_timeout(Duration::from_millis(40))
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Busy);
        // Release the mixer thread; a subsequent unload finishes cleanly.
        drop(tx);
        player.unload().unwrap();
        assert_eq!(MappedWav::live_mappings_under(&dir), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
