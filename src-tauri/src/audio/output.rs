//! Audio output: the device thread and the real-time callback.
//!
//! The device thread owns the platform stream (cpal streams are not `Send`
//! on every platform, so they never leave it). It opens the default output
//! device on request, rebuilds the stream when it fails or when the default
//! device changes underneath it, and closes it after a while of silence so
//! an idle app does not keep the audio hardware (and power assertions)
//! busy.
//!
//! Every stream gets a fresh lock-free SPSC ring buffer of interleaved
//! stereo `f32`. The producer goes to the mixer thread; the consumer goes
//! into the callback, which only copies out of it ([`fill_device_buffer`]):
//! no locks, no allocation, no I/O.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::util::lock;

/// Ring capacity in stereo frames (85 ms at 192 kHz; the mixer keeps it
/// filled to about 50 ms).
pub const RING_FRAMES: usize = 16_384;

/// How often the device thread checks for a changed default device.
const DEVICE_POLL: Duration = Duration::from_millis(1500);
/// Close the stream after this long without playback.
const IDLE_CLOSE: Duration = Duration::from_secs(30);

/// Error-callback outcomes, set from whatever thread the backend uses.
#[derive(Default)]
pub struct StreamFlags {
    /// The stream died and must be rebuilt.
    pub failed: AtomicBool,
    /// The backend moved the stream to a new default device by itself.
    pub rerouted: AtomicBool,
}

/// What a backend gets to feed a device.
pub struct Feed {
    pub consumer: Consumer<f32>,
    /// Stereo frames the callback has taken out of the ring so far.
    pub played: Arc<AtomicU64>,
    pub flags: Arc<StreamFlags>,
}

/// A running output stream. Dropping `keep_alive` stops it.
pub struct OpenStream {
    pub sample_rate: u32,
    pub channels: u16,
    pub device_id: Option<String>,
    pub device_name: String,
    pub keep_alive: Box<dyn Any>,
}

/// Something that can play audio. The app uses [`CpalBackend`]; tests use
/// fakes (no device, or a simulated one).
pub trait AudioBackend {
    /// Opens the current default output device and starts draining `feed`.
    /// The error is a plain-language message.
    fn open(&mut self, feed: Feed) -> Result<OpenStream, String>;
    /// Stable id of the current default output device, if there is one.
    fn default_device_id(&mut self) -> Option<String>;
}

/// Creates a backend on the device thread.
pub type BackendFactory = Arc<dyn Fn() -> Box<dyn AudioBackend> + Send + Sync>;

/// Link from the device thread to the mixer: where to write and at what rate.
pub struct OutputLink {
    pub producer: Producer<f32>,
    pub played: Arc<AtomicU64>,
    pub sample_rate: u32,
}

/// Copies stereo frames from the ring into a device buffer of `channels`
/// channels, padding with silence on underrun. Real-time safe.
pub fn fill_device_buffer<T>(
    data: &mut [T],
    channels: usize,
    consumer: &mut Consumer<f32>,
    played: &AtomicU64,
) where
    T: SizedSample + FromSample<f32>,
{
    let channels = channels.max(1);
    let frames = data.len() / channels;
    let n = frames.min(consumer.slots() / 2);
    let mut written = 0;
    if n > 0
        && let Ok(chunk) = consumer.read_chunk(n * 2)
    {
        {
            let (a, b) = chunk.as_slices();
            let mut it = a.iter().chain(b.iter()).copied();
            for frame in data.chunks_exact_mut(channels).take(n) {
                let l = it.next().unwrap_or(0.0);
                let r = it.next().unwrap_or(0.0);
                if channels == 1 {
                    frame[0] = T::from_sample((l + r) * 0.5);
                } else {
                    frame[0] = T::from_sample(l);
                    frame[1] = T::from_sample(r);
                    for s in &mut frame[2..] {
                        *s = T::EQUILIBRIUM;
                    }
                }
            }
        }
        chunk.commit_all();
        played.fetch_add(n as u64, Ordering::Relaxed);
        written = n * channels;
    }
    for s in &mut data[written..] {
        *s = T::EQUILIBRIUM;
    }
}

/* ------------------------------------------------------------------ */
/* cpal                                                                */
/* ------------------------------------------------------------------ */

/// The real backend: the platform's default host through cpal.
pub struct CpalBackend {
    host: cpal::Host,
}

impl CpalBackend {
    pub fn new() -> Self {
        CpalBackend {
            host: cpal::default_host(),
        }
    }

    pub fn factory() -> BackendFactory {
        Arc::new(|| Box::new(CpalBackend::new()) as Box<dyn AudioBackend>)
    }
}

impl Default for CpalBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    feed: Feed,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + FromSample<f32> + Send + 'static,
{
    let channels = config.channels as usize;
    let Feed {
        mut consumer,
        played,
        flags,
    } = feed;
    device.build_output_stream::<T, _, _>(
        config,
        move |data: &mut [T], _| fill_device_buffer(data, channels, &mut consumer, &played),
        move |err: cpal::Error| match err.kind() {
            cpal::ErrorKind::Xrun | cpal::ErrorKind::RealtimeDenied => {}
            cpal::ErrorKind::DeviceChanged => flags.rerouted.store(true, Ordering::SeqCst),
            _ => flags.failed.store(true, Ordering::SeqCst),
        },
        Some(Duration::from_secs(3)),
    )
}

impl AudioBackend for CpalBackend {
    fn open(&mut self, feed: Feed) -> Result<OpenStream, String> {
        let device = self.host.default_output_device().ok_or_else(|| {
            "No sound output device was found. Connect speakers or headphones and try again."
                .to_string()
        })?;
        let supported = device
            .default_output_config()
            .map_err(|e| format!("The sound output device can't be used ({e})."))?;
        let config = supported.config();
        let (rate, channels) = (config.sample_rate, config.channels);
        use cpal::SampleFormat as F;
        let stream = match supported.sample_format() {
            F::F32 => build::<f32>(&device, config, feed),
            F::F64 => build::<f64>(&device, config, feed),
            F::I16 => build::<i16>(&device, config, feed),
            F::I32 => build::<i32>(&device, config, feed),
            F::I8 => build::<i8>(&device, config, feed),
            F::U8 => build::<u8>(&device, config, feed),
            F::U16 => build::<u16>(&device, config, feed),
            F::U32 => build::<u32>(&device, config, feed),
            other => {
                return Err(format!(
                    "The sound output device uses a sample format SplitSound doesn't support ({other:?})."
                ));
            }
        }
        .map_err(|e| format!("Couldn't open the sound output device ({e})."))?;
        stream
            .play()
            .map_err(|e| format!("Couldn't start the sound output device ({e})."))?;
        Ok(OpenStream {
            sample_rate: rate,
            channels,
            device_id: device.id().ok().map(|id| id.to_string()),
            device_name: device.to_string(),
            keep_alive: Box::new(stream),
        })
    }

    fn default_device_id(&mut self) -> Option<String> {
        self.host
            .default_output_device()
            .and_then(|d| d.id().ok())
            .map(|id| id.to_string())
    }
}

/* ------------------------------------------------------------------ */
/* Device thread                                                       */
/* ------------------------------------------------------------------ */

/// Device availability as seen by the rest of the app.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Not opened yet (or closed while idle).
    Closed,
    Open {
        name: String,
        sample_rate: u32,
    },
    /// The last attempt failed; the message is plain language.
    Unavailable(String),
}

enum DeviceCmd {
    /// Open the device if it is not open; reply with the result.
    Ensure(Sender<Result<(), String>>),
    Shutdown,
}

/// Handle to the device thread.
pub struct DeviceController {
    tx: Sender<DeviceCmd>,
    status: Arc<Mutex<DeviceStatus>>,
    thread: Option<thread::JoinHandle<()>>,
}

/// Flag the mixer sets while it is producing audio, so the device thread
/// knows when the stream is idle.
#[derive(Default)]
pub struct ActivityFlag(AtomicU8);

impl ActivityFlag {
    pub fn set(&self, active: bool) {
        self.0.store(active as u8, Ordering::Relaxed);
    }
    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed) != 0
    }
}

impl DeviceController {
    /// Starts the device thread. New streams are announced to the mixer with
    /// `on_link`. Nothing is opened until [`DeviceController::ensure_open`].
    /// `on_lost` is called when an open stream dies and no device can
    /// replace it.
    pub fn spawn(
        factory: BackendFactory,
        activity: Arc<ActivityFlag>,
        on_link: impl Fn(OutputLink) + Send + 'static,
        on_lost: impl Fn(String) + Send + 'static,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        let status = Arc::new(Mutex::new(DeviceStatus::Closed));
        let st = status.clone();
        let thread = thread::Builder::new()
            .name("splitsound-audio-device".into())
            .spawn(move || device_thread(factory, rx, st, activity, on_link, on_lost))
            .expect("failed to spawn the audio device thread");
        DeviceController {
            tx,
            status,
            thread: Some(thread),
        }
    }

    /// Opens the default output device if it is not open, waiting up to
    /// `timeout`. The error is a plain-language message.
    pub fn ensure_open(&self, timeout: Duration) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        self.tx
            .send(DeviceCmd::Ensure(tx))
            .map_err(|_| "The audio system stopped unexpectedly.".to_string())?;
        rx.recv_timeout(timeout)
            .unwrap_or_else(|_| Err("The sound output device didn't respond.".to_string()))
    }

    pub fn status(&self) -> DeviceStatus {
        lock(&self.status).clone()
    }
}

impl Drop for DeviceController {
    fn drop(&mut self) {
        let _ = self.tx.send(DeviceCmd::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Current {
    stream: OpenStream,
    flags: Arc<StreamFlags>,
}

fn device_thread(
    factory: BackendFactory,
    rx: Receiver<DeviceCmd>,
    status: Arc<Mutex<DeviceStatus>>,
    activity: Arc<ActivityFlag>,
    on_link: impl Fn(OutputLink),
    on_lost: impl Fn(String),
) {
    let mut backend = factory();
    let mut current: Option<Current> = None;
    // Whether playback wants a device (set by `Ensure`, cleared when idle-closed).
    let mut wanted = false;
    let mut retry_at: Option<Instant> = None;
    let mut backoff = Duration::from_secs(1);
    let mut last_poll = Instant::now();
    let mut last_active = Instant::now();

    let open = |backend: &mut Box<dyn AudioBackend>,
                current: &mut Option<Current>|
     -> Result<(), String> {
        *current = None; // stop the old stream first
        let (producer, consumer) = RingBuffer::<f32>::new(RING_FRAMES * 2);
        let played = Arc::new(AtomicU64::new(0));
        let flags = Arc::new(StreamFlags::default());
        match backend.open(Feed {
            consumer,
            played: played.clone(),
            flags: flags.clone(),
        }) {
            Ok(stream) => {
                log::info!(
                    "audio output: {} at {} Hz, {} channel(s)",
                    stream.device_name,
                    stream.sample_rate,
                    stream.channels
                );
                *lock(&status) = DeviceStatus::Open {
                    name: stream.device_name.clone(),
                    sample_rate: stream.sample_rate,
                };
                on_link(OutputLink {
                    producer,
                    played,
                    sample_rate: stream.sample_rate,
                });
                *current = Some(Current { stream, flags });
                Ok(())
            }
            Err(msg) => {
                log::warn!("audio output unavailable: {msg}");
                *lock(&status) = DeviceStatus::Unavailable(msg.clone());
                Err(msg)
            }
        }
    };

    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(DeviceCmd::Ensure(reply)) => {
                wanted = true;
                last_active = Instant::now();
                let r = if current.is_some() {
                    Ok(())
                } else {
                    let r = open(&mut backend, &mut current);
                    if r.is_ok() {
                        backoff = Duration::from_secs(1);
                        retry_at = None;
                    } else {
                        retry_at = Some(Instant::now() + backoff);
                    }
                    r
                };
                let _ = reply.send(r);
            }
            Ok(DeviceCmd::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }

        if activity.get() {
            last_active = Instant::now();
        }

        let mut reopen = false;
        if let Some(cur) = &mut current {
            if cur.flags.failed.swap(false, Ordering::SeqCst) {
                log::warn!("audio output stream failed; reopening");
                reopen = true;
            } else if cur.flags.rerouted.swap(false, Ordering::SeqCst) {
                // The backend followed the new default device itself.
                cur.stream.device_id = backend.default_device_id();
                log::info!("audio output moved to the new default device");
            } else if last_poll.elapsed() >= DEVICE_POLL {
                last_poll = Instant::now();
                let now = backend.default_device_id();
                if now.is_some() && now != cur.stream.device_id {
                    log::info!("default audio output changed; switching");
                    reopen = true;
                }
            }
            if !reopen && !activity.get() && last_active.elapsed() >= IDLE_CLOSE {
                log::info!("audio output idle; closing the stream");
                current = None;
                wanted = false;
                *lock(&status) = DeviceStatus::Closed;
            }
        } else if wanted && retry_at.is_some_and(|t| Instant::now() >= t) {
            reopen = true;
        }

        if reopen {
            let was_open = current.is_some();
            match open(&mut backend, &mut current) {
                Ok(()) => {
                    backoff = Duration::from_secs(1);
                    retry_at = None;
                }
                Err(msg) => {
                    if was_open {
                        on_lost(msg);
                    }
                    retry_at = Some(Instant::now() + backoff);
                    backoff = (backoff * 2).min(Duration::from_secs(8));
                }
            }
        }
    }
    drop(current);
}

/* ------------------------------------------------------------------ */
/* Test backends                                                       */
/* ------------------------------------------------------------------ */

/// A backend that never finds a device (like a CI machine).
pub struct NoDeviceBackend;

impl AudioBackend for NoDeviceBackend {
    fn open(&mut self, _feed: Feed) -> Result<OpenStream, String> {
        Err(
            "No sound output device was found. Connect speakers or headphones and try again."
                .into(),
        )
    }
    fn default_device_id(&mut self) -> Option<String> {
        None
    }
}

/// A simulated device: a thread that drains the ring in 10 ms periods, as
/// fast as `speed` times real time. Captures everything it plays.
pub struct FakeBackend {
    pub sample_rate: u32,
    pub speed: f64,
    pub captured: Arc<Mutex<Vec<f32>>>,
}

struct FakeStream {
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for FakeStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl AudioBackend for FakeBackend {
    fn open(&mut self, feed: Feed) -> Result<OpenStream, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let period = (self.sample_rate / 100) as usize;
        let sleep = Duration::from_secs_f64(0.01 / self.speed.max(0.01));
        let captured = self.captured.clone();
        let thread = thread::spawn(move || {
            let Feed {
                mut consumer,
                played,
                ..
            } = feed;
            let mut buf = vec![0f32; period * 2];
            while !s.load(Ordering::SeqCst) {
                fill_device_buffer(&mut buf, 2, &mut consumer, &played);
                lock(&captured).extend_from_slice(&buf);
                thread::sleep(sleep);
            }
        });
        Ok(OpenStream {
            sample_rate: self.sample_rate,
            channels: 2,
            device_id: Some("fake".into()),
            device_name: "Fake output".into(),
            keep_alive: Box::new(FakeStream {
                stop,
                thread: Some(thread),
            }),
        })
    }

    fn default_device_id(&mut self) -> Option<String> {
        Some("fake".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::Sample;

    #[test]
    fn callback_copies_maps_channels_and_pads_silence() {
        let (mut p, mut c) = RingBuffer::<f32>::new(64);
        for x in [0.5, -0.5, 1.0, 0.0] {
            p.push(x).unwrap();
        }
        let played = AtomicU64::new(0);
        // Stereo device, more room than data: rest is silence.
        let mut out = [9f32; 8];
        fill_device_buffer(&mut out, 2, &mut c, &played);
        assert_eq!(out, [0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(played.load(Ordering::Relaxed), 2);
        // Mono device averages, 4-channel device fills the extra channels with silence.
        for x in [0.5, -0.5, 1.0, 0.0] {
            p.push(x).unwrap();
        }
        let mut mono = [9f32; 1];
        fill_device_buffer(&mut mono, 1, &mut c, &played);
        assert_eq!(mono, [0.0]);
        let mut quad = [9i16; 4];
        fill_device_buffer(&mut quad, 4, &mut c, &played);
        assert_eq!(quad[0], i16::MAX);
        assert_eq!(&quad[1..], &[0, 0, 0]);
        // Unsigned formats are centred.
        let mut u = [0u16; 2];
        fill_device_buffer(&mut u, 2, &mut c, &played);
        assert_eq!(u, [u16::EQUILIBRIUM; 2]);
        assert_eq!(played.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn ensure_open_reports_a_missing_device_without_panicking() {
        let factory: BackendFactory =
            Arc::new(|| Box::new(NoDeviceBackend) as Box<dyn AudioBackend>);
        let dc =
            DeviceController::spawn(factory, Arc::new(ActivityFlag::default()), |_| {}, |_| {});
        let err = dc.ensure_open(Duration::from_secs(2)).unwrap_err();
        assert!(err.contains("No sound output device"));
        assert!(matches!(dc.status(), DeviceStatus::Unavailable(_)));
    }

    /// A backend whose streams can be made to fail and whose default device
    /// can be switched, to test recovery.
    struct Flaky {
        opens: Arc<AtomicU64>,
        default_id: Arc<Mutex<String>>,
        fail_first: bool,
    }

    impl AudioBackend for Flaky {
        fn open(&mut self, feed: Feed) -> Result<OpenStream, String> {
            let n = self.opens.fetch_add(1, Ordering::SeqCst);
            if n == 0 && self.fail_first {
                feed.flags.failed.store(true, Ordering::SeqCst);
            }
            Ok(OpenStream {
                sample_rate: 44_100,
                channels: 2,
                device_id: Some(lock(&self.default_id).clone()),
                device_name: "flaky".into(),
                keep_alive: Box::new(feed.consumer),
            })
        }
        fn default_device_id(&mut self) -> Option<String> {
            Some(lock(&self.default_id).clone())
        }
    }

    fn flaky(
        fail_first: bool,
    ) -> (
        DeviceController,
        Arc<AtomicU64>,
        Arc<Mutex<String>>,
        Receiver<OutputLink>,
    ) {
        let opens = Arc::new(AtomicU64::new(0));
        let default_id = Arc::new(Mutex::new("speakers".to_string()));
        let (o, d) = (opens.clone(), default_id.clone());
        let factory: BackendFactory = Arc::new(move || {
            Box::new(Flaky {
                opens: o.clone(),
                default_id: d.clone(),
                fail_first,
            }) as Box<dyn AudioBackend>
        });
        let (tx, rx) = mpsc::channel();
        let dc = DeviceController::spawn(
            factory,
            Arc::new(ActivityFlag::default()),
            move |l| {
                let _ = tx.send(l);
            },
            |_| {},
        );
        (dc, opens, default_id, rx)
    }

    #[test]
    fn a_failed_stream_is_rebuilt() {
        let (dc, opens, _, rx) = flaky(true);
        dc.ensure_open(Duration::from_secs(2)).unwrap();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // The device thread notices the failure and opens a fresh stream.
        rx.recv_timeout(Duration::from_secs(3))
            .expect("a second link after the failure");
        assert_eq!(opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn follows_a_new_default_device() {
        let (dc, opens, default_id, rx) = flaky(false);
        dc.ensure_open(Duration::from_secs(2)).unwrap();
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        *lock(&default_id) = "headphones".into();
        rx.recv_timeout(Duration::from_secs(5))
            .expect("a new link for the new device");
        assert_eq!(opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn fake_device_links_to_the_mixer() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let cap = captured.clone();
        let factory: BackendFactory = Arc::new(move || {
            Box::new(FakeBackend {
                sample_rate: 48_000,
                speed: 1.0,
                captured: cap.clone(),
            }) as Box<dyn AudioBackend>
        });
        let (tx, rx) = mpsc::channel();
        let dc = DeviceController::spawn(
            factory,
            Arc::new(ActivityFlag::default()),
            move |l| {
                let _ = tx.send(l);
            },
            |_| {},
        );
        dc.ensure_open(Duration::from_secs(2)).unwrap();
        let mut link = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(link.sample_rate, 48_000);
        for _ in 0..100 {
            link.producer.push(0.25).unwrap();
        }
        let t0 = Instant::now();
        while link.played.load(Ordering::Relaxed) < 50 && t0.elapsed() < Duration::from_secs(2) {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(link.played.load(Ordering::Relaxed), 50);
        assert!(lock(&captured).contains(&0.25));
        assert!(matches!(
            dc.status(),
            DeviceStatus::Open {
                sample_rate: 48_000,
                ..
            }
        ));
    }
}
