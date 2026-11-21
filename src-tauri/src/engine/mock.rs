//! A stand-in engine with synthetic but believable behaviour, so the app
//! can be driven end to end without models.
//!
//! - Decodes WAV only (any PCM/float layout, any rate), resampled to the
//!   working format.
//! - "Separates" by filtering the mix into frequency bands; `other` (or
//!   `instrumental`) is the mix minus the named stems, so stems still sum
//!   back to the original exactly, as with the real engine.
//! - Takes a few seconds per split with real progress and cancellation.
//! - Simulates model downloads (no network), including the restricted
//!   Demucs model that needs the user's agreement.
//! - Exports WAV (16/24/32-bit float); FLAC and MP3 are refused.
//!
//! Install state is kept in `<models dir>/mock-installed.json`.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{
    CancelToken, DecodedTrack, Engine, EngineConfig, EngineError, EngineProgress, EngineResult,
    ExportJob, HardwareInfo, ScanOutcome, ScanResult, SplitOutput, SplitPlan, SplitRequest,
    StemAnalysis,
};
use crate::audio::filters::Chain;
use crate::audio::mapped::MappedWav;
use crate::audio::resample::Resampler;
use crate::audio::wav::{self, WavReader, WavWriter, WriteFormat};
use crate::ipc::{
    ACTIVITY_BINS, Detection, ErrorCode, ExportFormat, GpuInfo, ModelDownloadProgress, ModelStatus,
    Quality, SplitMode, SplitOptions, SplitStage, StemId, TrackInfo,
};
use crate::util::{lock, write_atomic};

const MB: u64 = 1_000_000;

struct MockModel {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    size: u64,
    download: u64,
    license: &'static str,
    attribution: &'static str,
    source: &'static str,
    restricted: bool,
    terms: Option<&'static str>,
}

/// Terms shown before the restricted Demucs model is downloaded.
pub const DEMUCS_TERMS: &str = "Demucs is published by Meta for research. Its weights are not under an open-source licence and are likely limited to non-commercial use (CC BY-NC). SplitSound downloads them directly from Meta and keeps them on this computer.";

const CATALOG: &[MockModel] = &[
    MockModel {
        id: "scnet_small",
        name: "SCNet small",
        description: "Splits vocals, drums, bass and everything else quickly.",
        size: 42 * MB,
        download: 38 * MB,
        license: "MIT",
        attribution: "SCNet: Sparse Compression Network for Music Source Separation (Tong et al.)",
        source: "GitHub",
        restricted: false,
        terms: None,
    },
    MockModel {
        id: "scnet_large",
        name: "SCNet large",
        description: "Splits vocals, drums, bass and everything else at the best quality.",
        size: 165 * MB,
        download: 151 * MB,
        license: "MIT",
        attribution: "SCNet: Sparse Compression Network for Music Source Separation (Tong et al.)",
        source: "GitHub",
        restricted: false,
        terms: None,
    },
    MockModel {
        id: "melband_roformer_vocals",
        name: "Mel-Band RoFormer Vocals",
        description: "Studio-grade vocal isolation that minimises vocal bleed into other stems; large and slow.",
        size: 463 * MB,
        download: 422 * MB,
        license: "MIT",
        attribution: "Mel-Band RoFormer vocal separation model by KimberleyJensen (MIT License) and ZFTurbo/Music-Source-Separation-Training (MIT License). https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model",
        source: "GitHub",
        restricted: false,
        terms: None,
    },
    MockModel {
        id: "htdemucs_6s",
        name: "Demucs 6-stem",
        description: "Separates guitar and piano from the rest of the song.",
        size: 110 * MB,
        download: 102 * MB,
        license: "Research use (likely CC BY-NC)",
        attribution: "Hybrid Transformers for Music Source Separation (Rouard, Massa, Défossez; Meta AI)",
        source: "Meta",
        restricted: true,
        terms: Some(DEMUCS_TERMS),
    },
    MockModel {
        id: "instrument_tagger",
        name: "Instrument listener",
        description: "Listens to a song and names the instruments it hears.",
        size: 81 * MB,
        download: 75 * MB,
        license: "MIT",
        attribution: "EfficientAT (Schmid et al.)",
        source: "GitHub",
        restricted: false,
        terms: None,
    },
];

const TAGGER: &str = "instrument_tagger";
/// Simulated download speed in bytes per second.
const DOWNLOAD_BPS: f64 = 40.0 * MB as f64;
const CHUNK: usize = 65_536;

fn models_for(options: &SplitOptions) -> Vec<&'static str> {
    match (options.mode, options.quality) {
        (SplitMode::Vocals, Quality::Studio) => vec!["melband_roformer_vocals"],
        (SplitMode::Band, Quality::Studio) => vec!["melband_roformer_vocals", "scnet_large"],
        (SplitMode::All, Quality::Studio) => {
            vec!["melband_roformer_vocals", "scnet_large", "htdemucs_6s"]
        }
        (SplitMode::Vocals | SplitMode::Band, Quality::Fast) => vec!["scnet_small"],
        (SplitMode::Vocals | SplitMode::Band, Quality::Best) => vec!["scnet_large"],
        (SplitMode::All, Quality::Fast) => vec!["scnet_small", "htdemucs_6s"],
        (SplitMode::All, Quality::Best) => vec!["scnet_large", "htdemucs_6s"],
    }
}

/// Frequency band (low cut, high cut) that stands in for a stem.
fn band(mode: SplitMode, stem: StemId) -> (Option<f64>, Option<f64>) {
    match (mode, stem) {
        (SplitMode::All, StemId::Bass) => (None, Some(150.0)),
        (SplitMode::All, StemId::Guitar) => (Some(150.0), Some(600.0)),
        (SplitMode::All, StemId::Vocals) => (Some(600.0), Some(3000.0)),
        (SplitMode::All, StemId::Piano) => (Some(3000.0), Some(5000.0)),
        (SplitMode::All, StemId::Drums) => (Some(5000.0), None),
        (_, StemId::Bass) => (None, Some(150.0)),
        (_, StemId::Drums) => (Some(3000.0), None),
        _ => (Some(300.0), Some(3000.0)),
    }
}

/// See the module docs.
pub struct MockEngine {
    config: EngineConfig,
    installed: Mutex<BTreeSet<String>>,
    download_lock: Mutex<()>,
    /// Multiplies every simulated delay (1.0 in the app; small in tests).
    time_scale: f64,
}

impl MockEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self::with_time_scale(config, 1.0)
    }

    pub fn with_time_scale(config: EngineConfig, time_scale: f64) -> Self {
        let installed = fs::read(config.models_dir.join("mock-installed.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Vec<String>>(&b).ok())
            .map(|v| {
                v.into_iter()
                    .filter(|id| CATALOG.iter().any(|m| m.id == id))
                    .collect()
            })
            .unwrap_or_else(|| BTreeSet::from([TAGGER.to_string()]));
        MockEngine {
            config,
            installed: Mutex::new(installed),
            download_lock: Mutex::new(()),
            time_scale,
        }
    }

    fn save_installed(&self, set: &BTreeSet<String>) {
        let _ = fs::create_dir_all(&self.config.models_dir);
        let v: Vec<&String> = set.iter().collect();
        if let Err(e) = write_atomic(
            &self.config.models_dir.join("mock-installed.json"),
            &serde_json::to_vec(&v).unwrap_or_default(),
        ) {
            log::warn!("mock engine: can't save install state: {e}");
        }
    }

    fn is_installed(&self, id: &str) -> bool {
        lock(&self.installed).contains(id)
    }

    fn status(&self, m: &MockModel) -> ModelStatus {
        ModelStatus {
            id: m.id.into(),
            name: m.name.into(),
            description: m.description.into(),
            size_bytes: m.size,
            download_bytes: m.download,
            installed: self.is_installed(m.id),
            download_progress: None,
            license: m.license.into(),
            attribution: m.attribution.into(),
            source: m.source.into(),
            restricted: m.restricted,
            terms: m.terms.map(Into::into),
        }
    }

    /// Sleeps `secs * time_scale`, waking every 20 ms to check `cancel`.
    fn nap(&self, secs: f64, cancel: &CancelToken) -> EngineResult<()> {
        let end = Instant::now() + Duration::from_secs_f64((secs * self.time_scale).max(0.0));
        loop {
            if cancel.is_cancelled() {
                return Err(EngineError::cancelled());
            }
            let now = Instant::now();
            if now >= end {
                return Ok(());
            }
            std::thread::sleep((end - now).min(Duration::from_millis(20)));
        }
    }

    /// How long a split takes, before scaling.
    fn split_seconds(duration_sec: f64, quality: Quality) -> f64 {
        let base = (duration_sec * 0.02).clamp(3.0, 8.0);
        match quality {
            Quality::Fast => base,
            Quality::Best => base * 2.0,
            Quality::Studio => base * 4.5,
        }
    }
}

fn io_err(what: &str, e: io::Error) -> EngineError {
    log::warn!("mock engine: {what}: {e}");
    match e.kind() {
        io::ErrorKind::NotFound => {
            EngineError::new(ErrorCode::FileNotFound, "The file couldn't be found.")
        }
        io::ErrorKind::StorageFull => EngineError::new(ErrorCode::DiskFull, "The disk is full."),
        io::ErrorKind::Unsupported => EngineError::new(
            ErrorCode::UnsupportedFormat,
            "This WAV file uses an encoding SplitSound can't read.",
        ),
        io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof => EngineError::new(
            ErrorCode::DecodeFailed,
            "This file is damaged or isn't really audio.",
        ),
        _ => EngineError::new(ErrorCode::Internal, format!("Couldn't {what}.")),
    }
}

fn db(x: f64) -> f64 {
    if x > 1e-6 { 20.0 * x.log10() } else { -120.0 }
}

/// Writes interleaved stereo chunks to `dest` through a temporary file.
struct AtomicWav {
    tmp: PathBuf,
    dest: PathBuf,
    w: Option<WavWriter>,
}

impl AtomicWav {
    fn create(dest: &Path, format: WriteFormat) -> io::Result<Self> {
        let tmp = dest.with_file_name(format!(
            ".splitsound-{}.part",
            crate::util::new_session_id()
        ));
        let file = fs::File::create_new(&tmp)?;
        let w = WavWriter::from_file(file, format, 2, wav::WORK_RATE).inspect_err(|_| {
            let _ = fs::remove_file(&tmp);
        })?;
        Ok(AtomicWav {
            tmp,
            dest: dest.to_path_buf(),
            w: Some(w),
        })
    }

    fn write(&mut self, s: &[f32]) -> io::Result<()> {
        match &mut self.w {
            Some(w) => w.write(s),
            None => Err(io::Error::other("writer already finished")),
        }
    }

    fn finish(mut self) -> io::Result<PathBuf> {
        if let Some(w) = self.w.take() {
            w.finish()?;
        }
        crate::util::rename_retry(&self.tmp, &self.dest)?;
        Ok(std::mem::take(&mut self.dest))
    }
}

impl Drop for AtomicWav {
    fn drop(&mut self) {
        // Close the file first (Windows can't delete open files), then
        // remove the temporary: a no-op after a successful rename.
        self.w = None;
        let _ = fs::remove_file(&self.tmp);
    }
}

impl MockEngine {
    fn analyse_stem(
        &self,
        stem: StemId,
        wav_path: &Path,
        cancel: &CancelToken,
    ) -> EngineResult<StemAnalysis> {
        let w = MappedWav::open(wav_path).map_err(|e| io_err("read a stem", e))?;
        let s = w.samples();
        let frames = s.len() / 2;
        let mut peak = 0f64;
        let mut sum = 0f64;
        let mut activity = Vec::with_capacity(ACTIVITY_BINS);
        let mut active = 0usize;
        for b in 0..ACTIVITY_BINS {
            if cancel.is_cancelled() {
                return Err(EngineError::cancelled());
            }
            let start = b * frames / ACTIVITY_BINS;
            let end = ((b + 1) * frames / ACTIVITY_BINS).max(start).min(frames);
            let seg = &s[start * 2..end * 2];
            let mut e = 0f64;
            for &x in seg {
                let x = x as f64;
                e += x * x;
                peak = peak.max(x.abs());
            }
            sum += e;
            let rms_db = if seg.is_empty() {
                -120.0
            } else {
                db((e / seg.len() as f64).sqrt())
            };
            if rms_db > -40.0 {
                active += 1;
            }
            activity.push(((rms_db + 60.0) / 60.0).clamp(0.0, 1.0));
        }
        let rms_db = if s.is_empty() {
            -120.0
        } else {
            db((sum / s.len() as f64).sqrt())
        };
        let peak_db = db(peak);
        let active_fraction = active as f64 / ACTIVITY_BINS as f64;
        let present = active_fraction > 0.02 && peak_db > -50.0;
        let contains = if present
            && matches!(stem, StemId::Other | StemId::Instrumental)
            && self.is_installed(TAGGER)
        {
            vec![
                Detection {
                    label: "Synthesizer".into(),
                    family: StemId::Other,
                    confidence: 0.55,
                },
                Detection {
                    label: "Strings".into(),
                    family: StemId::Other,
                    confidence: 0.31,
                },
            ]
        } else {
            Vec::new()
        };
        Ok(StemAnalysis {
            present,
            active_fraction,
            peak_db,
            rms_db,
            activity,
            contains,
        })
    }
}

impl Engine for MockEngine {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn decode(
        &self,
        input: &Path,
        mix_out: &Path,
        cancel: &CancelToken,
    ) -> EngineResult<DecodedTrack> {
        let ext = input
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if ext != "wav" && ext != "wave" {
            return Err(EngineError::new(
                ErrorCode::UnsupportedFormat,
                "This development build can only open WAV files.",
            ));
        }
        let mut r = WavReader::open(input).map_err(|e| io_err("open the file", e))?;
        let info = r.info.clone();
        let in_ch = info.channels as usize;
        let mut rs = Resampler::new(info.sample_rate, wav::WORK_RATE);
        let out_frames_total = (info.frames() as u128 * wav::WORK_RATE as u128)
            .div_ceil(info.sample_rate as u128) as u64;
        let mut out = AtomicWav::create(mix_out, WriteFormat::Float32)
            .map_err(|e| io_err("write the mix", e))?;

        // Decoded stereo frames waiting to be resampled.
        let mut pending: std::collections::VecDeque<f32> = std::collections::VecDeque::new();
        let mut raw = Vec::new();
        let mut eof = false;
        let mut read_err: Option<io::Error> = None;
        let mut written = 0u64;
        let mut block = vec![0f32; 4096 * 2];
        while written < out_frames_total {
            if cancel.is_cancelled() {
                return Err(EngineError::cancelled());
            }
            let n = ((out_frames_total - written) as usize).min(4096);
            let buf = &mut block[..n * 2];
            rs.process(buf, |dst| {
                while pending.len() < dst.len() && !eof {
                    raw.clear();
                    match r.read_frames(CHUNK, &mut raw) {
                        Ok(0) => eof = true,
                        Ok(_) => {
                            for f in raw.chunks_exact(in_ch) {
                                let (l, rr) = if in_ch == 1 {
                                    (f[0], f[0])
                                } else {
                                    (f[0], f[1])
                                };
                                pending.push_back(l);
                                pending.push_back(rr);
                            }
                        }
                        Err(e) => {
                            read_err.get_or_insert(e);
                            eof = true;
                        }
                    }
                }
                for d in dst.iter_mut() {
                    *d = pending.pop_front().unwrap_or(0.0);
                }
            });
            if let Some(e) = read_err.take() {
                return Err(io_err("read the file", e));
            }
            out.write(buf).map_err(|e| io_err("write the mix", e))?;
            written += n as u64;
        }
        out.finish().map_err(|e| io_err("write the mix", e))?;

        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let bits = (info.block_align as u32 * 8) / info.channels as u32;
        Ok(DecodedTrack {
            info: TrackInfo {
                path: String::new(),
                file_name: String::new(),
                title: info.title.clone().unwrap_or(stem),
                artist: info.artist.clone(),
                album: info.album.clone(),
                duration_sec: info.duration_sec(),
                sample_rate: info.sample_rate,
                channels: info.channels as u32,
                codec: "WAV".into(),
                bitrate_kbps: Some(info.sample_rate * info.channels as u32 * bits / 1000),
                file_size_bytes: 0,
                has_artwork: false,
            },
            artwork: None,
        })
    }

    fn scan_instruments(&self, wav_path: &Path, cancel: &CancelToken) -> EngineResult<ScanOutcome> {
        if !self.is_installed(TAGGER) {
            return Ok(ScanOutcome::MissingModels(vec![TAGGER.into()]));
        }
        let w = MappedWav::open(wav_path).map_err(|e| io_err("read the mix", e))?;
        let bands: [(StemId, &str, Option<f64>, Option<f64>); 5] = [
            (StemId::Bass, "Bass guitar", None, Some(150.0)),
            (StemId::Guitar, "Electric guitar", Some(150.0), Some(600.0)),
            (StemId::Vocals, "Singing", Some(600.0), Some(3000.0)),
            (StemId::Piano, "Piano", Some(3000.0), Some(5000.0)),
            (StemId::Drums, "Drum kit", Some(5000.0), None),
        ];
        let mut energy = [0f64; 5];
        let s = w.samples();
        let mut buf = Vec::with_capacity(CHUNK * 2);
        for (i, (_, _, lo, hi)) in bands.iter().enumerate() {
            let mut chain = Chain::band(*lo, *hi, wav::WORK_RATE as f64);
            for chunk in s.chunks(CHUNK * 2) {
                if cancel.is_cancelled() {
                    return Err(EngineError::cancelled());
                }
                buf.clear();
                buf.extend_from_slice(chunk);
                chain.process(&mut buf);
                energy[i] += buf.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
            }
        }
        self.nap(1.2, cancel)?;
        let total: f64 = energy.iter().sum::<f64>().max(1e-12);
        let mut detections: Vec<Detection> = bands
            .iter()
            .zip(energy)
            .map(|((family, label, _, _), e)| Detection {
                label: (*label).into(),
                family: *family,
                confidence: ((e / total) * 2.5).clamp(0.02, 0.97),
            })
            .chain(std::iter::once(Detection {
                label: "Synthesizer".into(),
                family: StemId::Other,
                confidence: 0.3,
            }))
            .filter(|d| d.confidence >= 0.15)
            .collect();
        detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let conf = |f: StemId| {
            detections
                .iter()
                .find(|d| d.family == f)
                .map_or(0.0, |d| d.confidence)
        };
        let suggested_mode = if conf(StemId::Guitar) >= 0.5 || conf(StemId::Piano) >= 0.5 {
            SplitMode::All
        } else if conf(StemId::Drums) >= 0.3 || conf(StemId::Bass) >= 0.3 {
            SplitMode::Band
        } else {
            SplitMode::Vocals
        };
        Ok(ScanOutcome::Done(ScanResult {
            detections,
            suggested_mode,
        }))
    }

    fn plan_split(&self, duration_sec: f64, options: &SplitOptions) -> EngineResult<SplitPlan> {
        let ids = models_for(options);
        Ok(SplitPlan {
            models: CATALOG
                .iter()
                .filter(|m| ids.contains(&m.id))
                .map(|m| self.status(m))
                .collect(),
            estimated_sec: Self::split_seconds(duration_sec, options.quality),
            // Loading, one per named stem, the residual, saving.
            step_count: options.mode.stems().len() as u32 + 3,
        })
    }

    fn split(
        &self,
        req: &SplitRequest,
        progress: &mut (dyn FnMut(EngineProgress) + Send),
        cancel: &CancelToken,
    ) -> EngineResult<SplitOutput> {
        let ids = models_for(&req.options);
        for id in &ids {
            if !self.is_installed(id) {
                let name = CATALOG.iter().find(|m| m.id == *id).map_or(*id, |m| m.name);
                return Err(EngineError::new(
                    ErrorCode::ModelMissing,
                    format!("The {name} model isn't installed."),
                ));
            }
        }
        let mode = req.options.mode;
        let stems = mode.stems();
        let named: Vec<StemId> = stems
            .iter()
            .copied()
            .filter(|s| !matches!(s, StemId::Other | StemId::Instrumental))
            .collect();
        let residual = *stems.last().expect("every mode has a residual stem");
        let step_count = named.len() as u32 + 4;
        let mix = MappedWav::open(&req.mix).map_err(|e| io_err("read the mix", e))?;
        let src = mix.samples();
        let frames = src.len() / 2;
        let total_secs =
            Self::split_seconds(frames as f64 / wav::WORK_RATE as f64, req.options.quality);
        let passes = named.len() + 2;
        let per_pass = total_secs / passes as f64;

        progress(EngineProgress {
            stage: SplitStage::Preparing,
            fraction: 0.0,
            label: "Loading the song".into(),
            step: 1,
            step_count,
        });
        self.nap(per_pass, cancel)?;

        let chunks = frames.div_ceil(CHUNK).max(1);
        let chunk_nap = per_pass / chunks as f64;
        let mut written = Vec::new();
        let mut buf = Vec::with_capacity(CHUNK * 2);
        for (i, stem) in named.iter().enumerate() {
            let (lo, hi) = band(mode, *stem);
            let mut chain = Chain::band(lo, hi, wav::WORK_RATE as f64);
            let dest = req.out_dir.join(format!("{}.wav", stem.as_str()));
            let mut out = AtomicWav::create(&dest, WriteFormat::Float32)
                .map_err(|e| io_err("write a stem", e))?;
            for (c, chunk) in src.chunks(CHUNK * 2).enumerate() {
                if cancel.is_cancelled() {
                    return Err(EngineError::cancelled());
                }
                buf.clear();
                buf.extend_from_slice(chunk);
                chain.process(&mut buf);
                for x in buf.iter_mut() {
                    *x *= 0.8;
                }
                out.write(&buf).map_err(|e| io_err("write a stem", e))?;
                progress(EngineProgress {
                    stage: SplitStage::Separating,
                    fraction: (1 + i) as f64 / passes as f64
                        + (c + 1) as f64 / chunks as f64 / passes as f64,
                    label: format!("Separating {}", stem.as_str()),
                    step: 2 + i as u32,
                    step_count,
                });
                self.nap(chunk_nap, cancel)?;
            }
            written.push((*stem, out.finish().map_err(|e| io_err("write a stem", e))?));
        }

        // The residual is the mix minus every named stem.
        let maps = written
            .iter()
            .map(|(_, p)| MappedWav::open(p))
            .collect::<io::Result<Vec<_>>>()
            .map_err(|e| io_err("read a stem", e))?;
        let dest = req.out_dir.join(format!("{}.wav", residual.as_str()));
        let mut out = AtomicWav::create(&dest, WriteFormat::Float32)
            .map_err(|e| io_err("write a stem", e))?;
        for (c, chunk) in src.chunks(CHUNK * 2).enumerate() {
            if cancel.is_cancelled() {
                return Err(EngineError::cancelled());
            }
            let off = c * CHUNK * 2;
            buf.clear();
            buf.extend_from_slice(chunk);
            for m in &maps {
                for (k, x) in buf.iter_mut().enumerate() {
                    *x -= m.samples().get(off + k).copied().unwrap_or(0.0);
                }
            }
            out.write(&buf).map_err(|e| io_err("write a stem", e))?;
            progress(EngineProgress {
                stage: SplitStage::Separating,
                fraction: (1 + named.len()) as f64 / passes as f64
                    + (c + 1) as f64 / chunks as f64 / passes as f64,
                label: format!("Separating {}", residual.as_str()),
                step: 2 + named.len() as u32,
                step_count,
            });
            self.nap(chunk_nap, cancel)?;
        }
        written.push((
            residual,
            out.finish().map_err(|e| io_err("write a stem", e))?,
        ));
        progress(EngineProgress {
            stage: SplitStage::Analyzing,
            fraction: 1.0,
            label: "Listening to the stems".into(),
            step: step_count - 1,
            step_count,
        });
        let stems = written
            .into_iter()
            .map(|(id, path)| {
                self.analyse_stem(id, &path, cancel)
                    .map(|analysis| (id, analysis))
            })
            .collect::<EngineResult<Vec<_>>>()?;
        progress(EngineProgress {
            stage: SplitStage::Writing,
            fraction: 1.0,
            label: "Saving stems".into(),
            step: step_count,
            step_count,
        });
        Ok(SplitOutput {
            stems,
            model_ids: ids.iter().map(|s| s.to_string()).collect(),
        })
    }

    fn export(&self, job: &ExportJob, cancel: &CancelToken) -> EngineResult<Vec<PathBuf>> {
        let format = match job.format {
            ExportFormat::Wav16 => WriteFormat::Pcm16,
            ExportFormat::Wav24 => WriteFormat::Pcm24,
            ExportFormat::Wav32f => WriteFormat::Float32,
            ExportFormat::Flac | ExportFormat::Mp3 => {
                return Err(EngineError::new(
                    ErrorCode::UnsupportedFormat,
                    "This development build can only export WAV files.",
                ));
            }
        };
        let mut done = Vec::new();
        for item in &job.items {
            let maps = item
                .sources
                .iter()
                .map(|(p, g)| MappedWav::open(p).map(|m| (m, *g)))
                .collect::<io::Result<Vec<_>>>()
                .map_err(|e| io_err("read a stem", e))?;
            let len = maps
                .iter()
                .map(|(m, _)| m.samples().len())
                .max()
                .unwrap_or(0);
            let mut out =
                AtomicWav::create(&item.dest, format).map_err(|e| io_err("write the export", e))?;
            let mut buf = vec![0f32; CHUNK * 2];
            let mut off = 0;
            while off < len {
                if cancel.is_cancelled() {
                    return Err(EngineError::cancelled());
                }
                let n = (len - off).min(CHUNK * 2);
                let b = &mut buf[..n];
                b.fill(0.0);
                for (m, g) in &maps {
                    let s = m.samples();
                    for (k, x) in b.iter_mut().enumerate() {
                        *x += s.get(off + k).copied().unwrap_or(0.0) * g;
                    }
                }
                out.write(b).map_err(|e| io_err("write the export", e))?;
                off += n;
            }
            done.push(out.finish().map_err(|e| io_err("write the export", e))?);
        }
        Ok(done)
    }

    fn list_models(&self) -> Vec<ModelStatus> {
        CATALOG.iter().map(|m| self.status(m)).collect()
    }

    fn download_models(
        &self,
        ids: &[String],
        accept_restricted: bool,
        progress: &mut (dyn FnMut(ModelDownloadProgress) + Send),
        cancel: &CancelToken,
    ) -> EngineResult<()> {
        let _guard = lock(&self.download_lock);
        let mut models = Vec::new();
        for id in ids {
            let m = CATALOG.iter().find(|m| m.id == id).ok_or_else(|| {
                EngineError::new(ErrorCode::NotFound, "That model doesn't exist.")
            })?;
            if m.restricted && !accept_restricted {
                return Err(EngineError::new(
                    ErrorCode::TermsRequired,
                    format!(
                        "{} can only be downloaded after you agree to its terms.",
                        m.name
                    ),
                ));
            }
            models.push(m);
        }
        for m in models {
            if self.is_installed(m.id) {
                continue;
            }
            let secs = m.download as f64 / DOWNLOAD_BPS * self.time_scale;
            let steps = ((secs / 0.05).ceil() as u64).max(1);
            for i in 0..=steps {
                if cancel.is_cancelled() {
                    return Err(EngineError::cancelled());
                }
                progress(ModelDownloadProgress {
                    model_id: m.id.into(),
                    received_bytes: m.download * i / steps,
                    total_bytes: m.download,
                });
                if i < steps {
                    std::thread::sleep(Duration::from_secs_f64(secs / steps as f64));
                }
            }
            let mut set = lock(&self.installed);
            set.insert(m.id.into());
            self.save_installed(&set);
        }
        Ok(())
    }

    fn remove_model(&self, id: &str) -> EngineResult<()> {
        if !CATALOG.iter().any(|m| m.id == id) {
            return Err(EngineError::new(
                ErrorCode::NotFound,
                "That model doesn't exist.",
            ));
        }
        let mut set = lock(&self.installed);
        set.remove(id);
        self.save_installed(&set);
        Ok(())
    }

    fn import_model_files(&self, paths: &[PathBuf]) -> EngineResult<Vec<String>> {
        // The mock can't check digests; it recognises `<model id>.onnx` or
        // `<model id>.onnx.zst` by name only. The real engine accepts files
        // only if their SHA-256 matches the pinned manifest.
        let mut found = Vec::new();
        for p in paths {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let id = name.trim_end_matches(".zst").trim_end_matches(".onnx");
            if let Some(m) = CATALOG.iter().find(|m| m.id == id)
                && !found.contains(&m.id.to_string())
            {
                if m.restricted {
                    return Err(EngineError::new(
                        ErrorCode::TermsRequired,
                        format!("{} needs your agreement to its terms.", m.name),
                    ));
                }
                found.push(m.id.to_string());
            }
        }
        if !found.is_empty() {
            let mut set = lock(&self.installed);
            set.extend(found.iter().cloned());
            self.save_installed(&set);
        }
        Ok(found)
    }

    fn hardware(&self) -> HardwareInfo {
        HardwareInfo {
            gpu: GpuInfo {
                available: false,
                name: None,
            },
            cpu_threads: std::thread::available_parallelism().map_or(1, |n| n.get() as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> (MockEngine, PathBuf) {
        let dir = std::env::temp_dir().join(format!("ss-mock-{}", crate::util::new_session_id()));
        fs::create_dir_all(&dir).unwrap();
        let e = MockEngine::with_time_scale(
            EngineConfig {
                models_dir: dir.join("models"),
                cache_dir: dir.join("cache"),
                onnxruntime: None,
            },
            0.01,
        );
        (e, dir)
    }

    fn write_input(path: &Path, rate: u32, channels: u16, secs: f64) {
        let frames = (rate as f64 * secs) as usize;
        let mut w = WavWriter::create(path, WriteFormat::Pcm16, channels, rate).unwrap();
        let s: Vec<f32> = (0..frames)
            .flat_map(|i| {
                let t = i as f64 / rate as f64;
                let x = ((2.0 * std::f64::consts::PI * 220.0 * t).sin() * 0.3
                    + (2.0 * std::f64::consts::PI * 1500.0 * t).sin() * 0.2)
                    as f32;
                std::iter::repeat_n(x, channels as usize)
            })
            .collect();
        w.write(&s).unwrap();
        w.finish().unwrap();
    }

    #[test]
    fn decode_resamples_to_the_working_format() {
        let (e, dir) = engine();
        let input = dir.join("Song.wav");
        write_input(&input, 48_000, 1, 1.0);
        let out = dir.join("mix.wav");
        let d = e.decode(&input, &out, &CancelToken::new()).unwrap();
        assert_eq!(d.info.title, "Song");
        assert_eq!(d.info.sample_rate, 48_000);
        assert_eq!(d.info.channels, 1);
        assert!((d.info.duration_sec - 1.0).abs() < 1e-9);
        let m = MappedWav::open(&out).unwrap();
        assert_eq!(m.frames(), 44_100);
        let peak = m.samples().iter().fold(0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.4 && peak < 0.55, "{peak}");
        let err = e
            .decode(&dir.join("x.mp3"), &out, &CancelToken::new())
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::UnsupportedFormat);
        fs::write(dir.join("bad.wav"), b"RIFFnope").unwrap();
        let err = e
            .decode(&dir.join("bad.wav"), &out, &CancelToken::new())
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::DecodeFailed);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn restricted_model_needs_agreement() {
        let (e, dir) = engine();
        let plan = e
            .plan_split(
                100.0,
                &SplitOptions {
                    mode: SplitMode::All,
                    quality: Quality::Fast,
                },
            )
            .unwrap();
        let ids: Vec<&str> = plan.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["scnet_small", "htdemucs_6s"]);
        assert_eq!(plan.missing_restricted().count(), 1);
        let band = e
            .plan_split(
                100.0,
                &SplitOptions {
                    mode: SplitMode::Band,
                    quality: Quality::Best,
                },
            )
            .unwrap();
        assert_eq!(band.models.len(), 1);
        assert_eq!(band.models[0].id, "scnet_large");
        let mut n = 0;
        let err = e
            .download_models(
                &["htdemucs_6s".into()],
                false,
                &mut |_| n += 1,
                &CancelToken::new(),
            )
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::TermsRequired);
        assert_eq!(n, 0, "nothing downloaded");
        e.download_models(
            &["htdemucs_6s".into()],
            true,
            &mut |_| n += 1,
            &CancelToken::new(),
        )
        .unwrap();
        assert!(n > 0);
        assert!(
            e.list_models()
                .iter()
                .any(|m| m.id == "htdemucs_6s" && m.installed)
        );
        // Install state persists.
        let again = MockEngine::with_time_scale(e.config.clone(), 0.01);
        assert!(again.is_installed("htdemucs_6s"));
        fs::remove_dir_all(dir).unwrap();
    }
}
