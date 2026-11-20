//! Decoding audio files to stereo float PCM.
//!
//! Containers and most codecs are handled by Symphonia (pure Rust): MP3,
//! WAV, FLAC, Ogg Vorbis, AAC in MP4/M4A and ADTS, ALAC in MP4/M4A and CAF,
//! AIFF. Opus (in Ogg) is demuxed by Symphonia and decoded by `opuscule`
//! (see [`crate::opus`]).
//!
//! # Untrusted input
//! Files come from anywhere, so:
//! * channel count, sample rate and duration from headers are checked
//!   against [`DecodeLimits`] before anything is sized from them, and the
//!   duration cap is enforced again on what is actually decoded;
//! * tags and cover art are size-limited;
//! * audio is downmixed and resampled packet by packet, so memory is
//!   bounded by the *output* length regardless of the source rate or
//!   channel count;
//! * a file that decodes to nothing, loses too many packets, or comes out
//!   clearly shorter than its container says is an error (`decode_failed`),
//!   not a silently short track;
//! * the whole decode runs behind a panic boundary, so a bug in a decoder
//!   surfaces as `decode_failed` instead of unwinding into the caller.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use symphonia::core::audio::Channels;
use symphonia::core::audio::Position;
use symphonia::core::codecs::audio::well_known::{
    CODEC_ID_AAC, CODEC_ID_ALAC, CODEC_ID_FLAC, CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3,
    CODEC_ID_OPUS, CODEC_ID_VORBIS,
};
use symphonia::core::codecs::audio::{AudioCodecId, AudioDecoder, AudioDecoderOptions};
use symphonia::core::common::Limit;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::well_known::{
    FORMAT_ID_ADTS, FORMAT_ID_AIFF, FORMAT_ID_CAF, FORMAT_ID_FLAC, FORMAT_ID_ISOMP4, FORMAT_ID_MP3,
    FORMAT_ID_WAVE,
};
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, MetadataRevision, StandardTag, StandardVisualKey};

use crate::audio::Stereo;
use crate::error::{Error, ErrorCode, Result};
use crate::opus::OpusStreamDecoder;
use crate::progress::{Progress, check_cancelled};
use crate::resample::{MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, StreamResampler};

/// Bounds applied to everything read from a file.
#[derive(Debug, Clone)]
pub struct DecodeLimits {
    /// Longest track accepted, in seconds. Longer files are rejected with
    /// `decode_failed`. Default: two hours.
    pub max_duration_sec: u32,
    /// Largest channel count accepted. Default: 8 (7.1).
    pub max_channels: usize,
    /// Largest cover image kept, in bytes; bigger pictures are skipped.
    /// Default: 16 MiB.
    pub max_artwork_bytes: usize,
    /// Largest single tag kept, in bytes. Default: 64 KiB.
    pub max_tag_bytes: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        DecodeLimits {
            max_duration_sec: 2 * 60 * 60,
            max_channels: 8,
            max_artwork_bytes: 16 * 1024 * 1024,
            max_tag_bytes: 64 * 1024,
        }
    }
}

/// Embedded cover art.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artwork {
    /// Media type such as `image/jpeg`. Sniffed from the bytes when the tag
    /// does not say; `application/octet-stream` when unknown.
    pub media_type: String,
    /// The image file bytes, exactly as embedded.
    pub data: Vec<u8>,
}

/// Facts about the source file. Maps onto `TrackInfo` in the UI contract
/// (the shell adds `path`, `fileName` and the title fallback).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceInfo {
    /// Title tag, trimmed; `None` when absent or empty.
    pub title: Option<String>,
    /// Artist tag.
    pub artist: Option<String>,
    /// Album tag.
    pub album: Option<String>,
    /// Duration of the decoded audio in seconds.
    pub duration_sec: f64,
    /// Sample rate of the source file in Hz.
    pub sample_rate: u32,
    /// Channel count of the source file.
    pub channels: u16,
    /// Display name: "MP3", "FLAC", "Opus", "Vorbis", "WAV", "AIFF", "AAC",
    /// "ALAC", ...
    pub codec: String,
    /// Average bitrate in kbit/s from file size and duration.
    pub bitrate_kbps: Option<u32>,
    /// Size of the file in bytes.
    pub file_size_bytes: u64,
    /// Whether cover art was found.
    pub has_artwork: bool,
}

/// A decoded file.
#[derive(Debug, Clone)]
pub struct DecodedTrack {
    /// Stereo audio at [`DecodedTrack::sample_rate`]. Mono sources are
    /// duplicated to both channels; more than two channels are downmixed.
    pub audio: Stereo,
    /// The sample rate of `audio` (the rate that was asked for).
    pub sample_rate: u32,
    /// Facts about the source.
    pub info: SourceInfo,
    /// Cover art, if any.
    pub artwork: Option<Artwork>,
}

/// Audio exactly as the decoder produced it: source rate, source channels,
/// interleaved. Used by tests and tools that compare against other decoders;
/// the app itself uses [`decode_file`].
#[derive(Debug, Clone)]
pub struct NativeAudio {
    /// Source sample rate in Hz.
    pub sample_rate: u32,
    /// Source channel count.
    pub channels: usize,
    /// Interleaved samples, `frames * channels` long.
    pub samples: Vec<f32>,
    /// Facts about the source.
    pub info: SourceInfo,
}

/// Where decoded packets go.
trait PcmSink {
    /// Called once, before the first packet, with the stream's format.
    fn begin(
        &mut self,
        rate: u32,
        channels: usize,
        matrix: Vec<(f32, f32)>,
        hint_frames: u64,
    ) -> Result<()>;
    /// One packet of interleaved samples at the source rate and layout.
    fn write(&mut self, interleaved: &[f32]) -> Result<()>;
}

/// Downmixes to stereo and resamples as packets arrive.
struct StereoSink {
    target_rate: u32,
    channels: usize,
    matrix: Vec<(f32, f32)>,
    resampler: Option<StreamResampler>,
    out: [Vec<f32>; 2],
    tmp_l: Vec<f32>,
    tmp_r: Vec<f32>,
}

impl PcmSink for StereoSink {
    fn begin(
        &mut self,
        rate: u32,
        channels: usize,
        matrix: Vec<(f32, f32)>,
        hint_frames: u64,
    ) -> Result<()> {
        self.channels = channels;
        self.matrix = matrix;
        self.resampler = Some(StreamResampler::new(rate, self.target_rate, 2)?);
        let reserve =
            (hint_frames * u64::from(self.target_rate)).div_ceil(u64::from(rate)) as usize;
        for v in self.out.iter_mut() {
            v.try_reserve(reserve).map_err(|_| Error::out_of_memory())?;
        }
        Ok(())
    }

    fn write(&mut self, interleaved: &[f32]) -> Result<()> {
        let frames = interleaved.len() / self.channels.max(1);
        self.tmp_l.clear();
        self.tmp_r.clear();
        self.tmp_l.reserve(frames);
        self.tmp_r.reserve(frames);
        for frame in interleaved.chunks_exact(self.channels.max(1)) {
            let mut l = 0f32;
            let mut r = 0f32;
            for (s, g) in frame.iter().zip(&self.matrix) {
                l += s * g.0;
                r += s * g.1;
            }
            self.tmp_l.push(l);
            self.tmp_r.push(r);
        }
        let Some(rs) = self.resampler.as_mut() else {
            return Err(Error::internal("sink used before begin"));
        };
        let reserve = rs.max_output_for(frames);
        for v in self.out.iter_mut() {
            v.try_reserve(reserve).map_err(|_| Error::out_of_memory())?;
        }
        rs.push(&[&self.tmp_l, &self.tmp_r], &mut self.out)
    }
}

/// Keeps the packets as they are.
#[derive(Default)]
struct NativeSink {
    samples: Vec<f32>,
}

impl PcmSink for NativeSink {
    fn begin(&mut self, _rate: u32, channels: usize, _m: Vec<(f32, f32)>, hint: u64) -> Result<()> {
        self.samples
            .try_reserve(hint as usize * channels)
            .map_err(|_| Error::out_of_memory())
    }

    fn write(&mut self, interleaved: &[f32]) -> Result<()> {
        self.samples
            .try_reserve(interleaved.len())
            .map_err(|_| Error::out_of_memory())?;
        self.samples.extend_from_slice(interleaved);
        Ok(())
    }
}

fn guarded<T>(work: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(r) => r,
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(damaged().with_detail(format!("decoder panicked: {what}")))
        }
    }
}

/// Decodes `path` to stereo float at `target_rate` Hz.
///
/// `progress` is used for cancellation only (decoding takes a second or
/// two); it is polled every few hundred packets.
///
/// # Errors
/// `file_not_found`, `unsupported_format` (not audio, or a codec the engine
/// cannot read), `decode_failed` (corrupt, truncated, too long, too many
/// channels), `cancelled`, `out_of_memory`.
pub fn decode_file(
    path: &Path,
    target_rate: u32,
    limits: &DecodeLimits,
    progress: &dyn Progress,
) -> Result<DecodedTrack> {
    guarded(|| {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&target_rate) {
            return Err(Error::internal("bad target sample rate"));
        }
        let mut sink = StereoSink {
            target_rate,
            channels: 0,
            matrix: Vec::new(),
            resampler: None,
            out: [Vec::new(), Vec::new()],
            tmp_l: Vec::new(),
            tmp_r: Vec::new(),
        };
        let (mut info, artwork) = decode_inner(path, limits, progress, &mut sink)?;
        let Some(mut rs) = sink.resampler.take() else {
            return Err(damaged().with_detail("no audio could be decoded"));
        };
        rs.finish(&mut sink.out)?;
        let [left, right] = sink.out;
        if left.is_empty() {
            return Err(damaged().with_detail("no audio could be decoded"));
        }
        let audio = Stereo::from_channels(left, right);
        info.duration_sec = audio.len() as f64 / f64::from(target_rate);
        info.bitrate_kbps = bitrate_kbps(info.file_size_bytes, info.duration_sec);
        Ok(DecodedTrack {
            audio,
            sample_rate: target_rate,
            info,
            artwork,
        })
    })
}

/// Decodes `path` without downmixing or resampling. Same limits and errors
/// as [`decode_file`].
pub fn decode_native(
    path: &Path,
    limits: &DecodeLimits,
    progress: &dyn Progress,
) -> Result<NativeAudio> {
    guarded(|| {
        let mut sink = NativeSink::default();
        let (info, _) = decode_inner(path, limits, progress, &mut sink)?;
        Ok(NativeAudio {
            sample_rate: info.sample_rate,
            channels: usize::from(info.channels),
            samples: sink.samples,
            info,
        })
    })
}

fn bitrate_kbps(file_size_bytes: u64, duration_sec: f64) -> Option<u32> {
    if duration_sec <= 0.0 {
        return None;
    }
    let kbps = (file_size_bytes as f64 * 8.0 / duration_sec / 1000.0).round();
    if kbps.is_finite() && kbps >= 1.0 {
        Some(kbps.min(f64::from(u32::MAX)) as u32)
    } else {
        None
    }
}

/// True when the first bytes are the signature of a container the engine
/// reads. A file that starts like audio but cannot be opened is damaged or
/// cut short (`decode_failed`); one that does not is simply not audio
/// (`unsupported_format`).
fn looks_like_audio(head: &[u8]) -> bool {
    let at = |o: usize, m: &[u8]| head.len() >= o + m.len() && &head[o..o + m.len()] == m;
    at(0, b"OggS")
        || at(0, b"fLaC")
        || at(0, b"ID3")
        || at(0, b"caff")
        || (at(0, b"RIFF") && at(8, b"WAVE"))
        || (at(0, b"FORM") && (at(8, b"AIFF") || at(8, b"AIFC")))
        || at(4, b"ftyp")
        // MPEG audio / ADTS frame sync.
        || (head.len() >= 2 && head[0] == 0xFF && head[1] & 0xE0 == 0xE0)
}

fn damaged() -> Error {
    Error::decode("This file is damaged and could not be read")
}

fn not_audio() -> Error {
    Error::unsupported("SplitSound cannot read this kind of file")
}

fn map_probe_error(err: SymError, path: &Path, is_audio: bool) -> Error {
    let unreadable = |detail: String| {
        if is_audio {
            Error::decode("This file is damaged or incomplete and could not be read")
                .with_detail(detail)
        } else {
            not_audio().with_detail(detail)
        }
    };
    match err {
        SymError::IoError(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            unreadable("end of file while probing".to_string())
        }
        SymError::IoError(e) => Error::from_read(&e, path),
        SymError::DecodeError(m) => damaged().with_detail(m),
        SymError::LimitError(m) => damaged().with_detail(m),
        other => unreadable(other.to_string()),
    }
}

fn codec_display_name(codec: AudioCodecId, format: &dyn FormatReader) -> String {
    let container = format.format_info().format;
    let name = if codec == CODEC_ID_MP3 {
        "MP3"
    } else if codec == CODEC_ID_MP2 {
        "MP2"
    } else if codec == CODEC_ID_MP1 {
        "MP1"
    } else if codec == CODEC_ID_FLAC {
        "FLAC"
    } else if codec == CODEC_ID_VORBIS {
        "Vorbis"
    } else if codec == CODEC_ID_OPUS {
        "Opus"
    } else if codec == CODEC_ID_AAC {
        "AAC"
    } else if codec == CODEC_ID_ALAC {
        "ALAC"
    } else if container == FORMAT_ID_WAVE {
        "WAV"
    } else if container == FORMAT_ID_AIFF {
        "AIFF"
    } else if container == FORMAT_ID_CAF {
        "CAF"
    } else if container == FORMAT_ID_ADTS {
        "AAC"
    } else {
        "PCM"
    };
    name.to_string()
}

/// Per-source-channel gains into (left, right).
///
/// Front left/right pass through; centre and the surrounds are mixed in at
/// -3 dB; the LFE channel is dropped; everything is then scaled so neither
/// output channel can exceed the sum of full-scale inputs (no clipping).
/// Layouts without speaker positions are treated as: first channel left,
/// second right, any others to both at -3 dB. Mono goes to both channels at
/// unity.
fn downmix_matrix(channels: &Channels, count: usize) -> Vec<(f32, f32)> {
    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
    if count == 1 {
        return vec![(1.0, 1.0)];
    }
    if count == 2 {
        return vec![(1.0, 0.0), (0.0, 1.0)];
    }
    let mut gains: Vec<(f32, f32)> = Vec::with_capacity(count);
    match channels {
        Channels::Positioned(positions) if positions.bits().count_ones() as usize == count => {
            // Planes are in ascending bit order.
            for bit in 0..64u32 {
                let Some(p) = Position::from_bits(1u64 << bit) else {
                    continue;
                };
                if !positions.contains(p) || p.is_empty() {
                    continue;
                }
                let g = if p == Position::FRONT_LEFT {
                    (1.0, 0.0)
                } else if p == Position::FRONT_RIGHT {
                    (0.0, 1.0)
                } else if p == Position::FRONT_CENTER {
                    (H, H)
                } else if p == Position::LFE1 {
                    (0.0, 0.0)
                } else if p == Position::REAR_LEFT
                    || p == Position::SIDE_LEFT
                    || p == Position::FRONT_LEFT_CENTER
                    || p == Position::TOP_FRONT_LEFT
                {
                    (H, 0.0)
                } else if p == Position::REAR_RIGHT
                    || p == Position::SIDE_RIGHT
                    || p == Position::FRONT_RIGHT_CENTER
                    || p == Position::TOP_FRONT_RIGHT
                {
                    (0.0, H)
                } else {
                    // Rear centre, top centre and anything rarer: both sides,
                    // quietly.
                    (0.5, 0.5)
                };
                gains.push(g);
            }
        }
        _ => {}
    }
    if gains.len() != count {
        gains.clear();
        for i in 0..count {
            gains.push(match i {
                0 => (1.0, 0.0),
                1 => (0.0, 1.0),
                _ => (H, H),
            });
        }
    }
    let sum_l: f32 = gains.iter().map(|g| g.0).sum();
    let sum_r: f32 = gains.iter().map(|g| g.1).sum();
    let norm = 1.0 / sum_l.max(sum_r).max(1.0);
    for g in gains.iter_mut() {
        g.0 *= norm;
        g.1 *= norm;
    }
    gains
}

/// Collects title/artist/album and the best cover picture from one
/// metadata revision. Earlier finds are kept (the first source wins).
struct TagCollector {
    title: Option<String>,
    artist: Option<String>,
    album: Option<String>,
    album_artist: Option<String>,
    artwork: Option<(bool, Artwork)>,
    max_artwork: usize,
    max_tag: usize,
}

impl TagCollector {
    fn clean(&self, value: &str) -> Option<String> {
        // Strip control characters (including NULs some taggers leave) and
        // surrounding whitespace; cap the length.
        let cleaned: String = value.chars().filter(|c| !c.is_control()).collect();
        let trimmed = cleaned.trim();
        if trimmed.is_empty() {
            return None;
        }
        let mut end = trimmed.len().min(self.max_tag);
        while !trimmed.is_char_boundary(end) {
            end -= 1;
        }
        Some(trimmed[..end].to_string())
    }

    fn absorb(&mut self, rev: &MetadataRevision) {
        let containers =
            std::iter::once(&rev.media).chain(rev.per_track.iter().map(|p| &p.metadata));
        for container in containers {
            for tag in &container.tags {
                match &tag.std {
                    Some(StandardTag::TrackTitle(v)) if self.title.is_none() => {
                        self.title = self.clean(v);
                    }
                    Some(StandardTag::Artist(v)) if self.artist.is_none() => {
                        self.artist = self.clean(v);
                    }
                    Some(StandardTag::AlbumArtist(v)) if self.album_artist.is_none() => {
                        self.album_artist = self.clean(v);
                    }
                    Some(StandardTag::Album(v)) if self.album.is_none() => {
                        self.album = self.clean(v);
                    }
                    _ => {}
                }
            }
            for visual in &container.visuals {
                if visual.data.is_empty() || visual.data.len() > self.max_artwork {
                    continue;
                }
                let is_front = visual.usage == Some(StandardVisualKey::FrontCover);
                let better = match &self.artwork {
                    None => true,
                    Some((have_front, _)) => is_front && !*have_front,
                };
                if better {
                    let media_type = sniff_image(&visual.data)
                        .map(str::to_string)
                        .or_else(|| {
                            visual
                                .media_type
                                .as_deref()
                                .filter(|m| m.starts_with("image/") && m.len() < 64)
                                .map(str::to_string)
                        })
                        .unwrap_or_else(|| "application/octet-stream".to_string());
                    self.artwork = Some((
                        is_front,
                        Artwork {
                            media_type,
                            data: visual.data.to_vec(),
                        },
                    ));
                }
            }
        }
    }
}

/// Identifies common image formats from their first bytes. The declared
/// media type in a tag is untrusted, so the bytes win when they are
/// recognisable.
fn sniff_image(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if data.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some("image/png")
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some("image/webp")
    } else if data.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

fn drain_metadata(format: &mut dyn FormatReader, tags: &mut TagCollector) {
    let mut meta = format.metadata();
    loop {
        if let Some(rev) = meta.current() {
            tags.absorb(rev);
        }
        if meta.is_latest() {
            break;
        }
        meta.pop();
    }
}

enum AnyDecoder {
    Symphonia(Box<dyn AudioDecoder>),
    Opus(Box<OpusStreamDecoder>),
}

fn decode_inner(
    path: &Path,
    limits: &DecodeLimits,
    progress: &dyn Progress,
    sink: &mut dyn PcmSink,
) -> Result<(SourceInfo, Option<Artwork>)> {
    // Check file type before opening: on Windows `File::open` on a directory
    // fails with ERROR_ACCESS_DENIED (PermissionDenied -> FileNotFound) rather
    // than opening a directory handle.
    let path_meta = std::fs::metadata(path).map_err(|e| Error::from_read(&e, path))?;
    if !path_meta.is_file() {
        let detail = if path_meta.is_dir() {
            "path is a directory"
        } else {
            "path is not a regular file"
        };
        return Err(not_audio().with_detail(detail));
    }
    let mut file = File::open(path).map_err(|e| Error::from_read(&e, path))?;
    let meta = file.metadata().map_err(|e| Error::from_read(&e, path))?;
    if !meta.is_file() {
        let detail = if meta.is_dir() {
            "path is a directory"
        } else {
            "path is not a regular file"
        };
        return Err(not_audio().with_detail(detail));
    }
    let file_size_bytes = meta.len();
    if file_size_bytes == 0 {
        return Err(not_audio().with_detail("empty file"));
    }
    let mut head = [0u8; 16];
    let got = read_up_to(&mut file, &mut head).map_err(|e| Error::from_read(&e, path))?;
    let is_audio = looks_like_audio(&head[..got]);
    file.seek(SeekFrom::Start(0))
        .map_err(|e| Error::from_read(&e, path))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(&ext.to_ascii_lowercase());
    }
    let meta_opts = MetadataOptions::default()
        .limit_tag_bytes(Limit::Maximum(limits.max_tag_bytes))
        .limit_visual_bytes(Limit::Maximum(limits.max_artwork_bytes));
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), meta_opts)
        .map_err(|e| map_probe_error(e, path, is_audio))?;

    let mut tags = TagCollector {
        title: None,
        artist: None,
        album: None,
        album_artist: None,
        artwork: None,
        max_artwork: limits.max_artwork_bytes,
        max_tag: limits.max_tag_bytes,
    };
    drain_metadata(format.as_mut(), &mut tags);

    // Pick the audio track.
    let track = format
        .default_track(TrackType::Audio)
        .or_else(|| format.first_track(TrackType::Audio))
        .ok_or_else(|| not_audio().with_detail("no audio track"))?
        .clone();
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| not_audio().with_detail("no audio codec parameters"))?
        .clone();
    let track_id = track.id;
    let container = format.format_info().format;
    let codec = codec_display_name(params.codec, format.as_ref());

    // Header checks, before anything is sized from the header.
    let bad_rate = |rate: u32| {
        Error::unsupported("This file has a sample rate SplitSound cannot handle")
            .with_detail(format!("{rate} Hz"))
    };
    let bad_channels = |n: usize| {
        Error::unsupported("This file has more channels than SplitSound can handle")
            .with_detail(format!("{n} channels"))
    };
    if let Some(rate) = params.sample_rate
        && !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&rate)
    {
        return Err(bad_rate(rate));
    }
    if let Some(ch) = &params.channels {
        let n = ch.count();
        if n == 0 || n > limits.max_channels {
            return Err(bad_channels(n));
        }
    }
    let too_long = || {
        Error::decode(format!(
            "This file is longer than {} hours, which is more than SplitSound can split",
            limits.max_duration_sec / 3600
        ))
    };
    let declared_frames = track.num_frames;
    if let (Some(frames), Some(rate)) = (declared_frames, params.sample_rate)
        && frames > u64::from(limits.max_duration_sec) * u64::from(rate)
    {
        return Err(too_long());
    }
    // Is the container's frame count exact (so falling short of it means
    // the file is damaged), or only an estimate? Symphonia estimates it for
    // MP3 without a LAME tag and for raw ADTS, Ogg lengths come from
    // wherever the file happens to end, and CAF totals count whole packets
    // (an ALAC stream's last packet is usually partial).
    let lossless_container =
        container == FORMAT_ID_WAVE || container == FORMAT_ID_AIFF || container == FORMAT_ID_FLAC;
    let exact_length = lossless_container
        || container == FORMAT_ID_ISOMP4
        || (container == FORMAT_ID_MP3 && track.delay.is_some());

    let mut decoder = if params.codec == CODEC_ID_OPUS {
        AnyDecoder::Opus(Box::new(OpusStreamDecoder::new(
            params.extra_data.as_deref().unwrap_or(&[]),
            limits.max_channels,
        )?))
    } else {
        let dec = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|e| match e {
                SymError::Unsupported(m) => Error::unsupported(format!(
                    "This file uses an audio format ({codec}) that SplitSound cannot read"
                ))
                .with_detail(m),
                other => damaged().with_detail(other),
            })?;
        AnyDecoder::Symphonia(dec)
    };

    // Established from the first decoded packet.
    let mut src_rate: u32 = 0;
    let mut src_channels: usize = 0;
    let mut interleaved: Vec<f32> = Vec::new();

    const MAX_CONSECUTIVE_BAD_PACKETS: u64 = 64;

    let mut src_frames: u64 = 0;
    let mut max_src_frames: u64 = u64::MAX;
    let mut seen_packets: u64 = 0;
    let mut packets: u64 = 0;
    let mut bad_packets: u64 = 0;
    let mut consecutive_bad_packets: u64 = 0;
    let mut hit_eof_error = false;

    loop {
        if seen_packets.is_multiple_of(256) {
            check_cancelled(progress)?;
        }
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                hit_eof_error = true;
                break;
            }
            Err(SymError::IoError(e)) => return Err(Error::from_read(&e, path)),
            // A chained stream with new parameters: keep what we have.
            Err(SymError::ResetRequired) => break,
            Err(SymError::DecodeError(_)) => {
                seen_packets += 1;
                bad_packets += 1;
                consecutive_bad_packets += 1;
                if consecutive_bad_packets > MAX_CONSECUTIVE_BAD_PACKETS {
                    return Err(damaged().with_detail("too many consecutive demux errors"));
                }
                continue;
            }
            Err(other) => return Err(damaged().with_detail(other)),
        };
        seen_packets += 1;
        if packet.track_id != track_id {
            continue;
        }
        packets += 1;

        // Decode to interleaved f32 plus the packet's rate and layout.
        interleaved.clear();
        let (rate, channels) = match &mut decoder {
            AnyDecoder::Opus(d) => {
                // Opus is strict: any decoder error fails the file.
                d.decode_packet(&packet.data, packet.trim_end.get(), &mut interleaved)?;
                (48_000u32, Channels::Discrete(d.channels() as u16))
            }
            AnyDecoder::Symphonia(d) => match d.decode(&packet) {
                Ok(buf) => {
                    buf.copy_to_vec_interleaved::<f32>(&mut interleaved);
                    (buf.spec().rate(), buf.spec().channels().clone())
                }
                Err(SymError::DecodeError(_)) | Err(SymError::IoError(_)) => {
                    bad_packets += 1;
                    consecutive_bad_packets += 1;
                    if consecutive_bad_packets > MAX_CONSECUTIVE_BAD_PACKETS {
                        return Err(damaged().with_detail("too many consecutive decode errors"));
                    }
                    continue;
                }
                Err(SymError::ResetRequired) => break,
                Err(other) => return Err(damaged().with_detail(other)),
            },
        };
        consecutive_bad_packets = 0;
        if interleaved.is_empty() {
            continue;
        }
        let n_ch = match &decoder {
            AnyDecoder::Opus(d) => d.channels(),
            AnyDecoder::Symphonia(_) => channels.count(),
        };
        if src_rate == 0 {
            if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&rate) {
                return Err(bad_rate(rate));
            }
            if n_ch == 0 || n_ch > limits.max_channels {
                return Err(bad_channels(n_ch));
            }
            src_rate = rate;
            src_channels = n_ch;
            max_src_frames = u64::from(limits.max_duration_sec) * u64::from(rate);
            let matrix = match &decoder {
                AnyDecoder::Opus(d) => d.downmix_matrix(),
                AnyDecoder::Symphonia(_) => downmix_matrix(&channels, n_ch),
            };
            // Reserve from the header only modestly: never more than the
            // cap, never more than one minute up front. Buffers grow as
            // real audio arrives.
            let hint_frames = declared_frames
                .unwrap_or(0)
                .min(u64::from(rate) * 60)
                .min(max_src_frames);
            sink.begin(rate, n_ch, matrix, hint_frames)?;
        } else if rate != src_rate || n_ch != src_channels {
            // Mid-stream format change (chained streams): stop here.
            break;
        }

        src_frames += (interleaved.len() / src_channels) as u64;
        if src_frames > max_src_frames {
            return Err(too_long());
        }
        // Decoders can emit NaN or infinity for damaged frames; never let
        // those into the pipeline.
        for s in interleaved.iter_mut() {
            if !s.is_finite() {
                *s = 0.0;
            }
        }
        sink.write(&interleaved)?;
    }

    if src_frames == 0 {
        return Err(damaged().with_detail("no audio could be decoded"));
    }

    // Too many unreadable packets means the file is damaged, even though
    // something came out. A stray bad frame (common at the edges of real
    // MP3s) is tolerated.
    let tolerated = 3u64.max(packets / 1000);
    if bad_packets > tolerated {
        return Err(damaged().with_detail(format!(
            "{bad_packets} of {packets} packets could not be decoded"
        )));
    }
    // Shorter than the container says: truncated or damaged. Lossless
    // containers get almost no slack (a single dropped FLAC block is a
    // real dropout); lossy ones get a couple of codec frames.
    if exact_length
        && let Some(declared) = declared_frames
        && declared > src_frames
    {
        let missing = declared - src_frames;
        // Some containers (CAF) count encoder priming and padding in their
        // frame total while the decoder trims them, so those never count
        // as missing.
        let trimmed = u64::from(track.delay.unwrap_or(0)) + u64::from(track.padding.unwrap_or(0));
        let allowed = trimmed
            + if lossless_container {
                u64::from(src_rate) / 50
            } else {
                u64::from(src_rate) / 10
            };
        if missing > allowed {
            return Err(
                Error::decode("This file is incomplete: part of the audio is missing").with_detail(
                    format!(
                        "decoded {src_frames} of {declared} frames{}",
                        if hit_eof_error {
                            " (unexpected end of file)"
                        } else {
                            ""
                        }
                    ),
                ),
            );
        }
    }

    drain_metadata(format.as_mut(), &mut tags);

    let duration_sec = src_frames as f64 / f64::from(src_rate);
    let artwork = tags.artwork.take().map(|(_, a)| a);
    let artist = tags.artist.take().or(tags.album_artist.take());
    Ok((
        SourceInfo {
            title: tags.title.take(),
            artist,
            album: tags.album.take(),
            duration_sec,
            sample_rate: src_rate,
            channels: src_channels as u16,
            codec,
            bitrate_kbps: bitrate_kbps(file_size_bytes, duration_sec),
            file_size_bytes,
            has_artwork: artwork.is_some(),
        },
        artwork,
    ))
}

fn read_up_to(file: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match file.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

/// True when `err` means "this is not something the engine can open" rather
/// than "this file is broken".
pub fn is_unsupported(err: &Error) -> bool {
    err.code() == ErrorCode::UnsupportedFormat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_matrices() {
        assert_eq!(downmix_matrix(&Channels::Discrete(1), 1), vec![(1.0, 1.0)]);
        assert_eq!(
            downmix_matrix(&Channels::Discrete(2), 2),
            vec![(1.0, 0.0), (0.0, 1.0)]
        );
        // 5.1: FL FR FC LFE RL RR
        let p = Position::FRONT_LEFT
            | Position::FRONT_RIGHT
            | Position::FRONT_CENTER
            | Position::LFE1
            | Position::REAR_LEFT
            | Position::REAR_RIGHT;
        let m = downmix_matrix(&Channels::Positioned(p), 6);
        assert_eq!(m.len(), 6);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let norm = 1.0 / (1.0 + 2.0 * h);
        assert!((m[0].0 - norm).abs() < 1e-6 && m[0].1 == 0.0);
        assert!((m[2].0 - h * norm).abs() < 1e-6 && (m[2].1 - h * norm).abs() < 1e-6);
        assert_eq!(m[3], (0.0, 0.0), "LFE is dropped");
        assert!(m[4].0 > 0.0 && m[4].1 == 0.0);
        assert!(m[5].1 > 0.0 && m[5].0 == 0.0);
        // Full-scale on every channel cannot clip.
        let l: f32 = m.iter().map(|g| g.0).sum();
        let r: f32 = m.iter().map(|g| g.1).sum();
        assert!(l <= 1.0 + 1e-6 && r <= 1.0 + 1e-6);
        // Unknown layout with 4 channels.
        let m = downmix_matrix(&Channels::Discrete(4), 4);
        assert_eq!(m.len(), 4);
        assert!(m.iter().map(|g| g.0).sum::<f32>() <= 1.0 + 1e-6);
    }

    #[test]
    fn audio_signatures() {
        assert!(looks_like_audio(b"OggS\0\0\0\0"));
        assert!(looks_like_audio(b"fLaC\0\0\0\x22"));
        assert!(looks_like_audio(b"RIFF\x10\0\0\0WAVEfmt "));
        assert!(looks_like_audio(b"FORM\0\0\0\x10AIFFCOMM"));
        assert!(looks_like_audio(b"\0\0\0\x20ftypM4A "));
        assert!(looks_like_audio(&[0xFF, 0xFB, 0x90, 0x00]));
        assert!(looks_like_audio(b"ID3\x03\0"));
        assert!(!looks_like_audio(b"RIFF\x10\0\0\0AVI LIST"));
        assert!(!looks_like_audio(b"<html><body>"));
        assert!(!looks_like_audio(b"PK\x03\x04"));
        assert!(!looks_like_audio(b""));
    }

    #[test]
    fn image_sniffing() {
        assert_eq!(sniff_image(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("image/jpeg"));
        assert_eq!(
            sniff_image(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0]),
            Some("image/png")
        );
        assert_eq!(sniff_image(b"<script>alert(1)</script>"), None);
    }

    #[test]
    fn tag_cleaning() {
        let t = TagCollector {
            title: None,
            artist: None,
            album: None,
            album_artist: None,
            artwork: None,
            max_artwork: 10,
            max_tag: 8,
        };
        assert_eq!(t.clean("  Hello\0\u{7} "), Some("Hello".into()));
        assert_eq!(t.clean(" \0 "), None);
        // Truncation lands on a character boundary.
        assert_eq!(t.clean("ééééé"), Some("éééé".into()));
    }

    fn tiny_pcm16_wav(rate: u32, frames: usize) -> Vec<u8> {
        let channels = 2u16;
        let data_bytes = (frames * usize::from(channels) * 2) as u32;
        let mut v = Vec::with_capacity(44 + data_bytes as usize);
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes()); // PCM
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        let block = channels * 2;
        v.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        v.extend_from_slice(&block.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_bytes.to_le_bytes());
        v.resize(44 + data_bytes as usize, 0);
        v
    }

    #[test]
    fn prime_and_absurd_sample_rates_are_bounded_or_rejected() {
        let dir = crate::testutil::TempDir::new("decode-rates");
        // Absurd rate outside 8..=192 kHz: rejected cleanly before allocating.
        for bad_rate in [4_000u32, 383_987, 1_000_000] {
            let p = dir.path().join(format!("bad-{bad_rate}.wav"));
            std::fs::write(&p, tiny_pcm16_wav(bad_rate, 64)).unwrap();
            let err = decode_file(
                &p,
                crate::audio::SAMPLE_RATE,
                &DecodeLimits::default(),
                &crate::progress::NoProgress,
            )
            .unwrap_err();
            assert_eq!(err.code(), ErrorCode::UnsupportedFormat, "{bad_rate} Hz");
        }
        // Large prime rate inside the supported range: imports quickly with
        // bounded resampler buffers instead of planning a massive FFT.
        let p = dir.path().join("prime-191999.wav");
        std::fs::write(&p, tiny_pcm16_wav(191_999, 256)).unwrap();
        let t0 = std::time::Instant::now();
        let decoded = decode_file(
            &p,
            crate::audio::SAMPLE_RATE,
            &DecodeLimits::default(),
            &crate::progress::NoProgress,
        )
        .unwrap();
        assert!(t0.elapsed() < std::time::Duration::from_secs(2));
        assert_eq!(decoded.info.sample_rate, 191_999);
        assert_eq!(
            decoded.audio.len(),
            (256u64 * u64::from(crate::audio::SAMPLE_RATE)).div_ceil(191_999) as usize
        );
    }

    #[test]
    fn consecutive_corrupt_mp3_frames_bail_out_early() {
        // Build a synthetic MP3 stream whose first frame is valid MPEG-1 Layer III
        // (so probe succeeds) followed by 200 corrupt frames whose main_data
        // fails codec decoding (or invalid sync / bad frame headers).
        // MPEG-1 Layer III, 128 kbps, 44.1 kHz, stereo: frame size = 417 bytes.
        // Header: FF FB 90 00.
        let dir = crate::testutil::TempDir::new("decode-bad-frames");
        let p = dir.path().join("corrupt.mp3");
        let mut bytes = Vec::new();
        // Valid-looking header + zeroed body (valid silence frame).
        let mut frame = vec![0u8; 417];
        frame[0..4].copy_from_slice(&[0xFF, 0xFB, 0x90, 0x00]);
        bytes.extend_from_slice(&frame);
        // 300 frames with MPEG-1 Layer III header and corrupted sideinfo/Huffman
        // payload (all 0xFF bytes, which triggers invalid bitrate/sideinfo or
        // decode errors in Symphonia).
        let mut bad = vec![0xFFu8; 417];
        bad[0..4].copy_from_slice(&[0xFF, 0xFB, 0x90, 0x00]);
        for _ in 0..300 {
            bytes.extend_from_slice(&bad);
        }
        std::fs::write(&p, &bytes).unwrap();
        let err = decode_file(
            &p,
            crate::audio::SAMPLE_RATE,
            &DecodeLimits::default(),
            &crate::progress::NoProgress,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::DecodeFailed);
        assert!(
            err.detail()
                .is_some_and(|d| d.contains("too many consecutive")),
            "expected consecutive-error bailout, got: {err:?}"
        );
    }

    fn ogg_crc(data: &[u8]) -> u32 {
        let mut crc = 0u32;
        for &b in data {
            crc ^= u32::from(b) << 24;
            for _ in 0..8 {
                crc = if (crc & 0x8000_0000) != 0 {
                    (crc << 1) ^ 0x04C1_1DB7
                } else {
                    crc << 1
                };
            }
        }
        crc
    }

    fn ogg_page(header_type: u8, granule: u64, serial: u32, seq: u32, packet: &[u8]) -> Vec<u8> {
        assert!(packet.len() < 255);
        let mut page = Vec::with_capacity(28 + packet.len());
        page.extend_from_slice(b"OggS");
        page.push(0); // version
        page.push(header_type);
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&seq.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes()); // crc placeholder
        page.push(1); // 1 segment
        page.push(packet.len() as u8);
        page.extend_from_slice(packet);
        let crc = ogg_crc(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        page
    }

    #[test]
    fn cancellation_is_checked_while_skipping_other_tracks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CancelOnSecondCheck(AtomicUsize);
        impl Progress for CancelOnSecondCheck {
            fn is_cancelled(&self) -> bool {
                self.0.fetch_add(1, Ordering::SeqCst) >= 1
            }
        }

        let mut opus_head = Vec::with_capacity(19);
        opus_head.extend_from_slice(b"OpusHead");
        opus_head.push(1); // version
        opus_head.push(2); // channels
        opus_head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
        opus_head.extend_from_slice(&48_000u32.to_le_bytes()); // input rate
        opus_head.extend_from_slice(&0i16.to_le_bytes()); // gain
        opus_head.push(0); // mapping family

        let mut opus_tags = Vec::with_capacity(16);
        opus_tags.extend_from_slice(b"OpusTags");
        opus_tags.extend_from_slice(&0u32.to_le_bytes()); // vendor len
        opus_tags.extend_from_slice(&0u32.to_le_bytes()); // comment count

        // Multiplexed Ogg with two Opus streams: stream 1 is chosen as the
        // default audio track, followed by 300 packets belonging only to
        // stream 2. Cancellation must still be polled while skipping them.
        let mut ogg = Vec::new();
        ogg.extend_from_slice(&ogg_page(0x02, 0, 1, 0, &opus_head));
        ogg.extend_from_slice(&ogg_page(0x02, 0, 2, 0, &opus_head));
        ogg.extend_from_slice(&ogg_page(0x00, 0, 1, 1, &opus_tags));
        ogg.extend_from_slice(&ogg_page(0x00, 0, 2, 1, &opus_tags));
        let dummy_packet = [0xFCu8, 0xFF, 0xFE];
        for i in 0..300u32 {
            ogg.extend_from_slice(&ogg_page(
                0x00,
                u64::from(i + 1) * 960,
                2,
                i + 2,
                &dummy_packet,
            ));
        }
        ogg.extend_from_slice(&ogg_page(0x04, 960, 1, 2, &dummy_packet));

        let dir = crate::testutil::TempDir::new("decode-skip-cancel");
        let p = dir.path().join("multi.opus");
        std::fs::write(&p, &ogg).unwrap();
        let progress = CancelOnSecondCheck(AtomicUsize::new(0));
        let err = decode_file(
            &p,
            crate::audio::SAMPLE_RATE,
            &DecodeLimits::default(),
            &progress,
        )
        .unwrap_err();
        assert_eq!(err.code(), ErrorCode::Cancelled);
        assert!(progress.0.load(Ordering::SeqCst) >= 2);
    }
}
