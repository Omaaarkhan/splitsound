//! Working files: 32-bit float 44.1 kHz stereo WAV.
//!
//! * [`write_wav_f32`] / [`WavWriter`] write the engine's working format.
//!   The header is laid out so the sample data starts at byte 56, a multiple
//!   of four, which lets [`MappedWav`] hand the samples out as `&[f32]`
//!   straight from the page cache.
//! * [`read_wav`] reads any plain PCM or float WAV (8/16/24/32-bit integer,
//!   32/64-bit float, any channel count up to [`MAX_WAV_CHANNELS`]) into
//!   interleaved `f32`. It is deliberately small and strict: sizes from the
//!   header are checked against the real file length before anything is
//!   allocated.
//! * [`MappedWav`] memory-maps a working file for the playback mixer and for
//!   peaks.

use std::collections::hash_map::RandomState;
use std::fs::File;
use std::hash::{BuildHasher, Hash, Hasher};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(any(windows, test))]
use std::time::Duration;

use crate::audio::{SAMPLE_RATE, Stereo};
use crate::error::{Error, Result};

/// Byte offset of the first sample in files written by this module.
pub const WORKING_DATA_OFFSET: u64 = 56;

/// Largest channel count [`read_wav`] accepts.
pub const MAX_WAV_CHANNELS: u16 = 8;

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Sample encodings understood by the reader and writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Unsigned 8-bit PCM.
    U8,
    /// Signed 16-bit PCM.
    I16,
    /// Signed 24-bit PCM.
    I24,
    /// Signed 32-bit PCM.
    I32,
    /// 32-bit IEEE float.
    F32,
    /// 64-bit IEEE float.
    F64,
}

impl SampleFormat {
    /// Bytes per sample.
    pub fn bytes(self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::I16 => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 | SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
        }
    }
}

/// What a WAV header says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavSpec {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Sample encoding.
    pub format: SampleFormat,
    /// Byte offset of the sample data.
    pub data_offset: u64,
    /// Number of frames actually present in the file.
    pub frames: u64,
}

/// A decoded WAV file: interleaved float samples plus the header facts.
#[derive(Debug, Clone)]
pub struct WavData {
    /// Header facts.
    pub spec: WavSpec,
    /// Interleaved samples, `frames * channels` long.
    pub samples: Vec<f32>,
}

/// Builds the 56-byte header of a float WAV with `data_bytes` of samples.
/// Sizes saturate at `u32::MAX` (a float stereo 44.1 kHz file reaches 4 GiB
/// after about 3 h 22 min, beyond the engine's 2 hour cap).
fn float_header(channels: u16, sample_rate: u32, data_bytes: u64) -> [u8; 56] {
    let block_align = channels * 4;
    let byte_rate = sample_rate * u32::from(block_align);
    let data_len = u32::try_from(data_bytes).unwrap_or(u32::MAX);
    let riff_len = u32::try_from(data_bytes + 48).unwrap_or(u32::MAX);
    let frames = u32::try_from(data_bytes / u64::from(block_align.max(1))).unwrap_or(u32::MAX);
    let mut h = [0u8; 56];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&riff_len.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&WAVE_FORMAT_IEEE_FLOAT.to_le_bytes());
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&32u16.to_le_bytes());
    h[36..40].copy_from_slice(b"fact");
    h[40..44].copy_from_slice(&4u32.to_le_bytes());
    h[44..48].copy_from_slice(&frames.to_le_bytes());
    h[48..52].copy_from_slice(b"data");
    h[52..56].copy_from_slice(&data_len.to_le_bytes());
    h
}

#[cfg(windows)]
const WINDOWS_RETRY_DELAYS_MS: &[u64] = &[20, 40, 80, 120, 180, 250, 310];

#[cfg(any(windows, test))]
fn is_transient_windows_lock(err: &std::io::Error) -> bool {
    err.kind() == std::io::ErrorKind::PermissionDenied
        || matches!(err.raw_os_error(), Some(5 | 32 | 33))
}

#[cfg(any(windows, test))]
fn retry_transient<T>(
    delays_ms: &[u64],
    mut op: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    let mut attempt = 0;
    loop {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) if attempt < delays_ms.len() && is_transient_windows_lock(&e) => {
                std::thread::sleep(Duration::from_millis(delays_ms[attempt]));
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Retries a file rename or removal briefly on Windows when antivirus or the
/// search indexer holds a transient sharing lock (`ERROR_ACCESS_DENIED`,
/// `ERROR_SHARING_VIOLATION`, `ERROR_LOCK_VIOLATION`). Runs once on Unix.
pub(crate) fn retry_on_windows<T>(op: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    #[cfg(windows)]
    {
        retry_transient(WINDOWS_RETRY_DELAYS_MS, op)
    }
    #[cfg(not(windows))]
    {
        let mut op = op;
        op()
    }
}

pub(crate) fn rename_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    retry_on_windows(|| std::fs::rename(from, to))
}

pub(crate) fn remove_file_retry(path: &Path) -> std::io::Result<()> {
    retry_on_windows(|| std::fs::remove_file(path))
}

/// Opens `path` only if it is a regular file. Checking `metadata` before
/// `File::open` ensures directories on Windows return `UnsupportedFormat`
/// instead of OS error 5 (`ERROR_ACCESS_DENIED` -> `FileNotFound`), while
/// re-checking the open handle prevents a TOCTOU swap.
fn open_regular_file(path: &Path) -> Result<File> {
    let meta = std::fs::metadata(path).map_err(|e| Error::from_read(&e, path))?;
    if !meta.is_file() {
        let detail = if meta.is_dir() {
            "path is a directory"
        } else {
            "path is not a regular file"
        };
        return Err(
            Error::unsupported("SplitSound cannot read this kind of file").with_detail(detail),
        );
    }
    let file = File::open(path).map_err(|e| Error::from_read(&e, path))?;
    let opened_meta = file.metadata().map_err(|e| Error::from_read(&e, path))?;
    if !opened_meta.is_file() {
        return Err(
            Error::unsupported("SplitSound cannot read this kind of file")
                .with_detail("path is not a regular file"),
        );
    }
    Ok(file)
}

/// A temporary sibling file created exclusively by this process.
///
/// Dropping it without calling [`TempFile::persist`] removes the file; a
/// pre-existing path or symlink that was never created by us is never touched.
pub(crate) struct TempFile {
    path: Option<PathBuf>,
}

impl TempFile {
    pub(crate) fn path(&self) -> &Path {
        self.path.as_deref().expect("temp file already persisted")
    }

    /// Renames the temporary file over `dst` and disarms cleanup on drop.
    pub(crate) fn persist(mut self, dst: &Path) -> std::io::Result<()> {
        let Some(tmp) = self.path.as_ref() else {
            return Ok(());
        };
        rename_retry(tmp, dst)?;
        self.path = None;
        Ok(())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if let Some(tmp) = self.path.take() {
            let _ = remove_file_retry(&tmp);
        }
    }
}

fn create_temp_at(path: PathBuf) -> std::io::Result<(TempFile, File)> {
    let file = File::create_new(&path)?;
    Ok((TempFile { path: Some(path) }, file))
}

/// Creates a new sibling temp file with an unpredictable `.part` name using
/// `create_new` semantics, so preexisting files and symlinks are never
/// followed, overwritten or deleted.
///
/// Returns `(TempFile, File)` in that order so `File` is always dropped and
/// closed before `TempFile`'s cleanup removes the path (required on Windows).
pub(crate) fn create_temp_sibling(path: &Path) -> std::io::Result<(TempFile, File)> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let base = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    for _ in 0..32 {
        let mut h = RandomState::new().build_hasher();
        std::process::id().hash(&mut h);
        COUNTER.fetch_add(1, Ordering::Relaxed).hash(&mut h);
        let token = h.finish();
        let mut name = base.clone();
        name.push(format!(".{token:016x}.part"));
        let tmp = path.with_file_name(name);
        match create_temp_at(tmp) {
            Ok(pair) => return Ok(pair),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not create a unique temporary file",
    ))
}

/// Streaming writer for the working format (32-bit float, stereo).
///
/// Samples go to a temporary sibling file; [`WavWriter::finish`] patches the
/// header, flushes, syncs and renames it over the destination, so readers
/// never see a half-written file. Dropping the writer without calling
/// `finish` removes the temporary file.
pub struct WavWriter {
    out: Option<BufWriter<File>>,
    tmp: Option<TempFile>,
    dst: PathBuf,
    channels: u16,
    sample_rate: u32,
    data_bytes: u64,
}

impl WavWriter {
    /// Creates a stereo float writer at `sample_rate` for `path`.
    pub fn create(path: &Path, sample_rate: u32) -> Result<Self> {
        Self::create_with_channels(path, sample_rate, 2)
    }

    /// Creates a float writer with an explicit channel count.
    pub fn create_with_channels(path: &Path, sample_rate: u32, channels: u16) -> Result<Self> {
        if std::fs::metadata(path).is_ok_and(|m| m.is_dir()) {
            return Err(Error::from_write(
                &std::io::Error::from(std::io::ErrorKind::IsADirectory),
                path,
            ));
        }
        let (tmp, file) = create_temp_sibling(path).map_err(|e| Error::from_write(&e, path))?;
        let mut out = BufWriter::with_capacity(1 << 16, file);
        let header = out.write_all(&float_header(channels, sample_rate, 0));
        let writer = WavWriter {
            out: Some(out),
            tmp: Some(tmp),
            dst: path.to_path_buf(),
            channels,
            sample_rate,
            data_bytes: 0,
        };
        header.map_err(|e| Error::from_write(&e, path))?;
        Ok(writer)
    }

    /// Appends interleaved samples.
    pub fn write_interleaved(&mut self, samples: &[f32]) -> Result<()> {
        let Some(out) = self.out.as_mut() else {
            return Err(Error::internal("write after finish"));
        };
        let mut buf = [0u8; 4096];
        for chunk in samples.chunks(1024) {
            for (i, s) in chunk.iter().enumerate() {
                buf[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
            }
            out.write_all(&buf[..chunk.len() * 4])
                .map_err(|e| Error::from_write(&e, &self.dst))?;
        }
        self.data_bytes += samples.len() as u64 * 4;
        Ok(())
    }

    /// Appends planar stereo frames.
    pub fn write_planar(&mut self, left: &[f32], right: &[f32]) -> Result<()> {
        let mut inter = [0f32; 2048];
        let n = left.len().min(right.len());
        let mut pos = 0;
        while pos < n {
            let take = (n - pos).min(1024);
            for i in 0..take {
                inter[2 * i] = left[pos + i];
                inter[2 * i + 1] = right[pos + i];
            }
            self.write_interleaved(&inter[..take * 2])?;
            pos += take;
        }
        Ok(())
    }

    /// Patches the header, syncs and moves the file into place.
    pub fn finish(mut self) -> Result<()> {
        let Some(out) = self.out.take() else {
            return Ok(());
        };
        let dst = self.dst.clone();
        let werr = |e: std::io::Error| Error::from_write(&e, &dst);
        let mut file = out.into_inner().map_err(|e| werr(e.into_error()))?;
        file.seek(SeekFrom::Start(0)).map_err(werr)?;
        file.write_all(&float_header(
            self.channels,
            self.sample_rate,
            self.data_bytes,
        ))
        .map_err(werr)?;
        file.sync_all().map_err(werr)?;
        drop(file);
        if let Some(tmp) = self.tmp.take() {
            tmp.persist(&self.dst).map_err(werr)?;
        }
        Ok(())
    }
}

impl Drop for WavWriter {
    fn drop(&mut self) {
        drop(self.out.take());
        drop(self.tmp.take());
    }
}

/// Writes `audio` as a 32-bit float stereo WAV at `sample_rate`
/// (write-to-temp, then rename).
pub fn write_wav_f32(path: &Path, audio: &Stereo, sample_rate: u32) -> Result<()> {
    let mut w = WavWriter::create(path, sample_rate)?;
    w.write_planar(&audio.left, &audio.right)?;
    w.finish()
}

fn rd_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Parses the RIFF chunk list of `file` and returns the format and where the
/// samples are. Nothing proportional to header-declared sizes is allocated.
pub fn read_wav_spec(file: &mut File, path: &Path) -> Result<WavSpec> {
    let rerr = |e: std::io::Error| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::decode("This WAV file is incomplete")
        } else {
            Error::from_read(&e, path)
        }
    };
    let file_len = file
        .metadata()
        .map_err(|e| Error::from_read(&e, path))?
        .len();
    file.seek(SeekFrom::Start(0)).map_err(rerr)?;
    let mut head = [0u8; 12];
    file.read_exact(&mut head).map_err(rerr)?;
    if &head[0..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return Err(Error::unsupported("This is not a WAV file"));
    }
    let mut pos = 12u64;
    let mut fmt: Option<(u16, u16, u32, u16, u16)> = None;
    // A well-formed file has a handful of chunks; refuse to walk forever.
    for _ in 0..4096 {
        if pos + 8 > file_len {
            break;
        }
        file.seek(SeekFrom::Start(pos)).map_err(rerr)?;
        let mut ch = [0u8; 8];
        file.read_exact(&mut ch).map_err(rerr)?;
        let size = u64::from(rd_u32(&ch, 4));
        let body = pos + 8;
        match &ch[0..4] {
            b"fmt " => {
                if size < 16 {
                    return Err(Error::decode("This WAV file has a damaged header"));
                }
                let mut f = [0u8; 40];
                let take = size.min(40) as usize;
                file.read_exact(&mut f[..take]).map_err(rerr)?;
                let mut tag = rd_u16(&f, 0);
                let channels = rd_u16(&f, 2);
                let rate = rd_u32(&f, 4);
                let block_align = rd_u16(&f, 12);
                let bits = rd_u16(&f, 14);
                if tag == WAVE_FORMAT_EXTENSIBLE && take >= 26 {
                    // The first two bytes of the sub-format GUID are the
                    // real format tag.
                    tag = rd_u16(&f, 24);
                }
                fmt = Some((tag, channels, rate, block_align, bits));
            }
            b"data" => {
                let Some((tag, channels, rate, block_align, bits)) = fmt else {
                    return Err(Error::decode("This WAV file has a damaged header"));
                };
                let format = match (tag, bits) {
                    (WAVE_FORMAT_PCM, 8) => SampleFormat::U8,
                    (WAVE_FORMAT_PCM, 16) => SampleFormat::I16,
                    (WAVE_FORMAT_PCM, 24) => SampleFormat::I24,
                    (WAVE_FORMAT_PCM, 32) => SampleFormat::I32,
                    (WAVE_FORMAT_IEEE_FLOAT, 32) => SampleFormat::F32,
                    (WAVE_FORMAT_IEEE_FLOAT, 64) => SampleFormat::F64,
                    _ => {
                        return Err(Error::unsupported(
                            "This WAV file uses an encoding SplitSound cannot read",
                        ));
                    }
                };
                if channels == 0 || channels > MAX_WAV_CHANNELS {
                    return Err(Error::unsupported(
                        "This WAV file has an unsupported number of channels",
                    ));
                }
                if !(crate::resample::MIN_SAMPLE_RATE..=crate::resample::MAX_SAMPLE_RATE)
                    .contains(&rate)
                {
                    return Err(Error::unsupported(
                        "This WAV file has an unsupported sample rate",
                    ));
                }
                let frame_bytes = format.bytes() as u64 * u64::from(channels);
                if u64::from(block_align) != frame_bytes {
                    return Err(Error::decode("This WAV file has a damaged header"));
                }
                // Trust the file length, not the header: a streamed or
                // truncated file may declare more (or 0 / 0xFFFFFFFF).
                let available = file_len.saturating_sub(body);
                let declared = if size == 0 || size == u64::from(u32::MAX) {
                    available
                } else {
                    size.min(available)
                };
                return Ok(WavSpec {
                    sample_rate: rate,
                    channels,
                    format,
                    data_offset: body,
                    frames: declared / frame_bytes,
                });
            }
            _ => {}
        }
        // Chunks are word aligned.
        pos = body + size + (size & 1);
    }
    Err(Error::decode("This WAV file contains no audio"))
}

/// Reads a whole WAV file into interleaved `f32`.
///
/// `max_frames` bounds the allocation; a longer file is an error rather
/// than a truncated read.
pub fn read_wav(path: &Path, max_frames: u64) -> Result<WavData> {
    let mut file = open_regular_file(path)?;
    let spec = read_wav_spec(&mut file, path)?;
    if spec.frames > max_frames {
        return Err(Error::decode("This audio file is too long"));
    }
    let total = spec.frames as usize * usize::from(spec.channels);
    let mut samples: Vec<f32> = Vec::new();
    samples
        .try_reserve_exact(total)
        .map_err(|_| Error::out_of_memory())?;
    file.seek(SeekFrom::Start(spec.data_offset))
        .map_err(|e| Error::from_read(&e, path))?;
    let bytes_per = spec.format.bytes();
    let mut reader = std::io::BufReader::with_capacity(1 << 16, file);
    let mut buf = vec![0u8; bytes_per * 4096];
    let mut remaining = total;
    while remaining > 0 {
        let take = remaining.min(4096);
        let chunk = &mut buf[..take * bytes_per];
        reader.read_exact(chunk).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::decode("This WAV file is incomplete")
            } else {
                Error::from_read(&e, path)
            }
        })?;
        match spec.format {
            SampleFormat::U8 => {
                samples.extend(chunk.iter().map(|b| (f32::from(*b) - 128.0) / 128.0))
            }
            SampleFormat::I16 => samples.extend(
                chunk
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0),
            ),
            SampleFormat::I24 => samples.extend(chunk.as_chunks::<3>().0.iter().map(|b| {
                let v = i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8;
                v as f32 / 8_388_608.0
            })),
            SampleFormat::I32 => {
                samples.extend(
                    chunk.as_chunks::<4>().0.iter().map(|b| {
                        i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
                    }),
                )
            }
            SampleFormat::F32 => samples.extend(
                chunk
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            ),
            SampleFormat::F64 => samples.extend(chunk.as_chunks::<8>().0.iter().map(|b| {
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
            })),
        }
        remaining -= take;
    }
    Ok(WavData { spec, samples })
}

/// Reads a working file (float or PCM, mono or stereo, any rate) as planar
/// stereo. Returns the audio and its sample rate. Mono becomes dual mono.
pub fn read_wav_stereo(path: &Path, max_frames: u64) -> Result<(Stereo, u32)> {
    let wav = read_wav(path, max_frames)?;
    let stereo = match wav.spec.channels {
        1 => Stereo::from_mono(wav.samples),
        2 => Stereo::from_interleaved(&wav.samples),
        _ => {
            return Err(Error::unsupported(
                "This WAV file has more than two channels",
            ));
        }
    };
    Ok((stereo, wav.spec.sample_rate))
}

enum Backing {
    Mapped(memmap2::Mmap),
    Owned(Vec<f32>),
}

/// A working file opened for random access: the samples as one interleaved
/// `&[f32]`.
///
/// Files in the engine's own layout (32-bit float, little-endian host) are
/// memory-mapped and cost no heap memory; anything else readable by
/// [`read_wav`] is loaded into memory instead, so the type always works.
///
/// # File stability
/// A mapped file must not be truncated or rewritten in place while a
/// `MappedWav` for it is alive (the engine itself only ever replaces
/// working files by rename, which is safe). Violating this can crash the
/// process on some platforms; that is inherent to memory mapping.
pub struct MappedWav {
    backing: Backing,
    offset: usize,
    samples: usize,
    channels: u16,
    sample_rate: u32,
}

impl MappedWav {
    /// Opens `path`.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = open_regular_file(path)?;
        let spec = read_wav_spec(&mut file, path)?;
        let samples = spec.frames as usize * usize::from(spec.channels);
        let mappable = cfg!(target_endian = "little")
            && spec.format == SampleFormat::F32
            && spec.data_offset % 4 == 0
            && samples > 0;
        if mappable {
            // SAFETY: the map is read-only and private to this process. The
            // documented contract of this type is that the file is not
            // modified in place while mapped.
            let map =
                unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Error::from_read(&e, path))?;
            let offset = spec.data_offset as usize;
            let end = offset + samples * 4;
            if map.len() >= end && (map.as_ptr() as usize + offset).is_multiple_of(4) {
                return Ok(MappedWav {
                    backing: Backing::Mapped(map),
                    offset,
                    samples,
                    channels: spec.channels,
                    sample_rate: spec.sample_rate,
                });
            }
        }
        drop(file);
        let wav = read_wav(path, u64::MAX)?;
        Ok(MappedWav {
            samples: wav.samples.len(),
            backing: Backing::Owned(wav.samples),
            offset: 0,
            channels: wav.spec.channels,
            sample_rate: wav.spec.sample_rate,
        })
    }

    /// All samples, interleaved (`frames() * channels()` values).
    pub fn samples(&self) -> &[f32] {
        match &self.backing {
            Backing::Owned(v) => v,
            Backing::Mapped(map) => {
                let bytes = &map[self.offset..self.offset + self.samples * 4];
                // SAFETY: `open` checked that the start is 4-byte aligned,
                // that the range lies inside the map and that the host is
                // little-endian, and every bit pattern is a valid `f32`.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), self.samples) }
            }
        }
    }

    /// Number of frames.
    pub fn frames(&self) -> usize {
        self.samples / usize::from(self.channels.max(1))
    }

    /// Channel count (2 for engine working files).
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Whether the samples come straight from a memory map.
    pub fn is_mapped(&self) -> bool {
        matches!(self.backing, Backing::Mapped(_))
    }

    /// Copies the file into a planar [`Stereo`] (mono is duplicated).
    pub fn to_stereo(&self) -> Result<Stereo> {
        match self.channels {
            1 => Ok(Stereo::from_mono(self.samples().to_vec())),
            2 => Ok(Stereo::from_interleaved(self.samples())),
            _ => Err(Error::unsupported(
                "This WAV file has more than two channels",
            )),
        }
    }
}

/// Number of frames in two hours at the working rate; the default bound for
/// reading working files.
pub fn max_working_frames() -> u64 {
    u64::from(SAMPLE_RATE) * 2 * 3600 + u64::from(SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn ramp(frames: usize) -> Stereo {
        let left: Vec<f32> = (0..frames)
            .map(|i| (i as f32 / frames as f32) - 0.5)
            .collect();
        let right: Vec<f32> = left.iter().map(|v| -v * 0.5).collect();
        Stereo::from_channels(left, right)
    }

    #[test]
    fn float_round_trip_is_bit_exact() {
        let dir = TempDir::new("wav-roundtrip");
        let path = dir.path().join("a.wav");
        let audio = ramp(10_001);
        write_wav_f32(&path, &audio, SAMPLE_RATE).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            WORKING_DATA_OFFSET + 10_001 * 8
        );
        let (back, rate) = read_wav_stereo(&path, u64::MAX).unwrap();
        assert_eq!(rate, SAMPLE_RATE);
        assert_eq!(back, audio);
        // No temp file is left behind.
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names.len(), 1, "{names:?}");
    }

    #[test]
    fn mapped_reader_matches_and_is_zero_copy() {
        let dir = TempDir::new("wav-mmap");
        let path = dir.path().join("a.wav");
        let audio = ramp(4_410);
        write_wav_f32(&path, &audio, SAMPLE_RATE).unwrap();
        let map = MappedWav::open(&path).unwrap();
        assert!(map.is_mapped());
        assert_eq!(map.frames(), 4_410);
        assert_eq!(map.channels(), 2);
        assert_eq!(map.sample_rate(), SAMPLE_RATE);
        assert_eq!(map.samples(), audio.to_interleaved().as_slice());
        assert_eq!(map.to_stereo().unwrap(), audio);
    }

    fn pcm_wav(bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        let block = channels * bits / 8;
        v.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        v.extend_from_slice(&block.to_le_bytes());
        v.extend_from_slice(&bits.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(data.len() as u32).to_le_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn reads_pcm16_and_pcm24() {
        let dir = TempDir::new("wav-pcm");
        let p16 = dir.path().join("p16.wav");
        let mut d = Vec::new();
        for v in [0i16, 16384, -16384, 32767, -32768, 1] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&p16, pcm_wav(16, 2, 48_000, &d)).unwrap();
        let w = read_wav(&p16, u64::MAX).unwrap();
        assert_eq!(w.spec.sample_rate, 48_000);
        assert_eq!(w.spec.frames, 3);
        assert_eq!(w.samples[1], 0.5);
        assert_eq!(w.samples[2], -0.5);
        assert_eq!(w.samples[4], -1.0);

        let p24 = dir.path().join("p24.wav");
        let mut d = Vec::new();
        for v in [0i32, 4_194_304, -4_194_304, -8_388_608] {
            d.extend_from_slice(&v.to_le_bytes()[..3]);
        }
        std::fs::write(&p24, pcm_wav(24, 1, 44_100, &d)).unwrap();
        let w = read_wav(&p24, u64::MAX).unwrap();
        assert_eq!(w.samples, vec![0.0, 0.5, -0.5, -1.0]);
        // A non-float file still opens through MappedWav (loaded, not mapped).
        let m = MappedWav::open(&p24).unwrap();
        assert!(!m.is_mapped());
        assert_eq!(m.samples(), &[0.0, 0.5, -0.5, -1.0]);
    }

    #[test]
    fn header_lies_are_bounded_by_the_file() {
        let dir = TempDir::new("wav-lies");
        // Declares 4 GiB of data but holds 4 frames.
        let mut bytes = pcm_wav(16, 2, 44_100, &[0u8; 16]);
        let n = bytes.len();
        bytes[n - 20..n - 16].copy_from_slice(&u32::MAX.to_le_bytes());
        let p = dir.path().join("lie.wav");
        std::fs::write(&p, &bytes).unwrap();
        let w = read_wav(&p, u64::MAX).unwrap();
        assert_eq!(w.spec.frames, 4);

        // Truncated in the middle of the header.
        let p = dir.path().join("short.wav");
        std::fs::write(&p, &bytes[..30]).unwrap();
        assert_eq!(
            read_wav(&p, u64::MAX).unwrap_err().code().as_str(),
            "decode_failed"
        );

        // Not a WAV at all.
        let p = dir.path().join("junk.wav");
        std::fs::write(&p, b"this is definitely not audio data").unwrap();
        assert_eq!(
            read_wav(&p, u64::MAX).unwrap_err().code().as_str(),
            "unsupported_format"
        );

        // Too long for the caller's bound.
        let p = dir.path().join("ok.wav");
        std::fs::write(&p, pcm_wav(16, 2, 44_100, &[0u8; 400])).unwrap();
        assert_eq!(
            read_wav(&p, 10).unwrap_err().code().as_str(),
            "decode_failed"
        );

        // Zero channels / absurd channel counts are rejected.
        let p = dir.path().join("ch.wav");
        std::fs::write(&p, pcm_wav(16, 0, 44_100, &[0u8; 16])).unwrap();
        assert!(read_wav(&p, u64::MAX).is_err());

        // Missing file.
        let e = read_wav(&dir.path().join("nope.wav"), u64::MAX).unwrap_err();
        assert_eq!(e.code().as_str(), "file_not_found");
    }

    #[test]
    fn dropped_writer_leaves_nothing() {
        let dir = TempDir::new("wav-drop");
        let path = dir.path().join("a.wav");
        {
            let mut w = WavWriter::create(&path, SAMPLE_RATE).unwrap();
            w.write_interleaved(&[0.0; 64]).unwrap();
        }
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn temp_file_is_unpredictable_and_never_follows_or_removes_symlinks() {
        let dir = TempDir::new("wav-temp-symlink");
        let target = dir.path().join("existing.txt");
        std::fs::write(&target, b"keep this").unwrap();

        // Direct collision on a preexisting symlink refuses creation and leaves
        // both the symlink and its target intact.
        let link = dir.path().join("colliding.part");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(create_temp_at(link.clone()).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"keep this");
        assert_eq!(std::fs::read_link(&link).unwrap(), target);

        // Sibling temp names are distinct and do not use the predictable PID name.
        let dst = dir.path().join("out.wav");
        let (a, _fa) = create_temp_sibling(&dst).unwrap();
        let (b, _fb) = create_temp_sibling(&dst).unwrap();
        assert_ne!(a.path(), b.path());
        assert!(a.path().to_string_lossy().ends_with(".part"));
        assert_ne!(
            a.path(),
            dir.path()
                .join(format!("out.wav.{}.part", std::process::id()))
        );
        let pa = a.path().to_path_buf();
        drop(_fa);
        drop(a);
        assert!(!pa.exists());
        assert!(b.path().exists());
    }

    #[test]
    fn directories_are_rejected_as_unsupported_before_open() {
        let dir = TempDir::new("wav-dir");
        assert_eq!(
            read_wav(dir.path(), u64::MAX).unwrap_err().code().as_str(),
            "unsupported_format"
        );
        assert_eq!(
            MappedWav::open(dir.path()).err().unwrap().code().as_str(),
            "unsupported_format"
        );
        assert_eq!(
            WavWriter::create(dir.path(), SAMPLE_RATE)
                .err()
                .unwrap()
                .code()
                .as_str(),
            "internal"
        );
    }

    #[test]
    fn transient_lock_retry_recovers_and_stops_on_non_transient() {
        let mut attempts = 0;
        let ok = retry_transient(&[0, 0, 0], || {
            attempts += 1;
            if attempts < 3 {
                Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
            } else {
                Ok(42)
            }
        });
        assert_eq!(ok.unwrap(), 42);
        assert_eq!(attempts, 3);

        let mut not_found_attempts = 0;
        let err = retry_transient(&[0, 0, 0], || -> std::io::Result<()> {
            not_found_attempts += 1;
            Err(std::io::Error::from(std::io::ErrorKind::NotFound))
        })
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert_eq!(not_found_attempts, 1);
    }
}
