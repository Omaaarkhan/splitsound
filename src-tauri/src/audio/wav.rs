//! Minimal, bounded WAV reading and writing.
//!
//! The shell only needs WAV for two things: memory-mapping the 32-bit float
//! working files the engine writes (`mix.wav`, `stems/*.wav`) and, in the
//! mock engine, decoding and exporting WAV. The parser is strict about
//! sizes: nothing read from a header is trusted before it is checked
//! against the real file length and fixed limits.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Sample rate of every working file.
pub const WORK_RATE: u32 = 44_100;
/// Channel count of every working file.
pub const WORK_CHANNELS: u16 = 2;

/// Upper bounds applied before allocating anything based on a header.
pub const MAX_CHANNELS: u16 = 32;
pub const MIN_RATE: u32 = 4_000;
pub const MAX_RATE: u32 = 768_000;
/// Longest audio accepted: 3 hours.
pub const MAX_DURATION_SEC: u64 = 3 * 3600;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Signed integer PCM (8-bit is unsigned in WAV) with this many bits.
    Int(u16),
    /// IEEE float, 32 or 64 bits.
    Float(u16),
}

/// What a WAV header says, after validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavInfo {
    pub format: SampleFormat,
    pub channels: u16,
    pub sample_rate: u32,
    /// Byte offset of the first sample.
    pub data_offset: u64,
    /// Length of the sample data in bytes (whole frames only).
    pub data_len: u64,
    /// Bytes per frame.
    pub block_align: u16,
    /// Title and artist from a `LIST/INFO` chunk, if present. Plain text.
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
}

impl WavInfo {
    pub fn frames(&self) -> u64 {
        self.data_len / self.block_align as u64
    }

    pub fn duration_sec(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    /// True for the engine's working format: 32-bit float, 44.1 kHz stereo.
    pub fn is_working_format(&self) -> bool {
        self.format == SampleFormat::Float(32)
            && self.channels == WORK_CHANNELS
            && self.sample_rate == WORK_RATE
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// Parses the header of a RIFF/WAVE file (PCM 8/16/24/32-bit, float 32/64,
/// `WAVE_FORMAT_EXTENSIBLE` with either). `RF64` is not supported.
pub fn read_info(file: &mut File) -> io::Result<WavInfo> {
    let file_len = file.metadata()?.len();
    file.seek(SeekFrom::Start(0))?;
    let mut hdr = [0u8; 12];
    file.read_exact(&mut hdr)?;
    if &hdr[0..4] != b"RIFF" || &hdr[8..12] != b"WAVE" {
        return Err(invalid("not a RIFF/WAVE file"));
    }
    let mut pos: u64 = 12;
    let mut fmt: Option<(SampleFormat, u16, u32, u16)> = None;
    let mut data: Option<(u64, u64)> = None;
    let (mut title, mut artist, mut album) = (None, None, None);
    // A WAV file has a handful of chunks; bail out on anything pathological.
    for _ in 0..256 {
        if pos + 8 > file_len {
            break;
        }
        file.seek(SeekFrom::Start(pos))?;
        let mut ch = [0u8; 8];
        file.read_exact(&mut ch)?;
        let id = [ch[0], ch[1], ch[2], ch[3]];
        let size = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]) as u64;
        let body = pos + 8;
        match &id {
            b"fmt " => {
                if !(16..=1024).contains(&size) || body + size > file_len {
                    return Err(invalid("bad fmt chunk"));
                }
                let mut b = vec![0u8; size as usize];
                file.read_exact(&mut b)?;
                fmt = Some(parse_fmt(&b)?);
            }
            b"data" => {
                // Some writers leave the size at 0 or 0xFFFFFFFF when streaming;
                // clamp to what the file really holds.
                let avail = file_len.saturating_sub(body);
                let len = if size == 0 || size == 0xFFFF_FFFF {
                    avail
                } else {
                    size.min(avail)
                };
                data = Some((body, len));
                // Keep going: tags (LIST/INFO) often follow the audio.
                pos = body + len + (len & 1);
                continue;
            }
            b"LIST" if (4..=64 * 1024).contains(&size) && body + size <= file_len => {
                let mut b = vec![0u8; size as usize];
                file.read_exact(&mut b)?;
                if &b[0..4] == b"INFO" {
                    parse_info_list(&b[4..], &mut title, &mut artist, &mut album);
                }
            }
            _ => {}
        }
        pos = body + size + (size & 1);
    }
    let (format, channels, sample_rate, block_align) =
        fmt.ok_or_else(|| invalid("no fmt chunk"))?;
    let (data_offset, data_len) = data.ok_or_else(|| invalid("no data chunk"))?;
    let data_len = data_len - data_len % block_align as u64;
    let frames = data_len / block_align as u64;
    if frames / sample_rate as u64 > MAX_DURATION_SEC {
        return Err(invalid("audio is too long"));
    }
    Ok(WavInfo {
        format,
        channels,
        sample_rate,
        data_offset,
        data_len,
        block_align,
        title,
        artist,
        album,
    })
}

fn parse_fmt(b: &[u8]) -> io::Result<(SampleFormat, u16, u32, u16)> {
    let tag = u16::from_le_bytes([b[0], b[1]]);
    let channels = u16::from_le_bytes([b[2], b[3]]);
    let rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let block_align = u16::from_le_bytes([b[12], b[13]]);
    let bits = u16::from_le_bytes([b[14], b[15]]);
    let tag = if tag == 0xFFFE {
        // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the sub-format GUID.
        if b.len() < 40 {
            return Err(invalid("bad extensible fmt chunk"));
        }
        u16::from_le_bytes([b[24], b[25]])
    } else {
        tag
    };
    let format = match (tag, bits) {
        (1, 8 | 16 | 24 | 32) => SampleFormat::Int(bits),
        (3, 32 | 64) => SampleFormat::Float(bits),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "unsupported WAV encoding",
            ));
        }
    };
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(invalid("unsupported channel count"));
    }
    if !(MIN_RATE..=MAX_RATE).contains(&rate) {
        return Err(invalid("unsupported sample rate"));
    }
    if block_align as u32 != channels as u32 * (bits as u32 / 8) {
        return Err(invalid("inconsistent block alignment"));
    }
    Ok((format, channels, rate, block_align))
}

fn parse_info_list(
    mut b: &[u8],
    title: &mut Option<String>,
    artist: &mut Option<String>,
    album: &mut Option<String>,
) {
    while b.len() >= 8 {
        let id = [b[0], b[1], b[2], b[3]];
        let size = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
        let Some(val) = b.get(8..8 + size) else { break };
        let text = String::from_utf8_lossy(val)
            .trim_end_matches('\0')
            .trim()
            .chars()
            .filter(|c| !c.is_control())
            .take(512)
            .collect::<String>();
        if !text.is_empty() {
            match &id {
                b"INAM" => *title = Some(text),
                b"IART" => *artist = Some(text),
                b"IPRD" => *album = Some(text),
                _ => {}
            }
        }
        let next = 8 + size + (size & 1);
        if next > b.len() {
            break;
        }
        b = &b[next..];
    }
}

/// Streams frames out of a WAV file as interleaved `f32` in -1..1.
pub struct WavReader {
    file: io::BufReader<File>,
    pub info: WavInfo,
    remaining: u64,
    raw: Vec<u8>,
}

impl WavReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let meta = std::fs::metadata(path)?;
        if !meta.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        let mut f = File::open(path)?;
        if !f.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        let info = read_info(&mut f)?;
        f.seek(SeekFrom::Start(info.data_offset))?;
        Ok(WavReader {
            file: io::BufReader::with_capacity(1 << 16, f),
            remaining: info.frames(),
            info,
            raw: Vec::new(),
        })
    }

    /// Reads up to `max_frames` frames, appending `frames * channels`
    /// samples to `out`. Returns the number of frames read (0 at the end).
    pub fn read_frames(&mut self, max_frames: usize, out: &mut Vec<f32>) -> io::Result<usize> {
        let n = (max_frames as u64).min(self.remaining) as usize;
        if n == 0 {
            return Ok(0);
        }
        let ba = self.info.block_align as usize;
        self.raw.resize(n * ba, 0);
        self.file.read_exact(&mut self.raw)?;
        self.remaining -= n as u64;
        match self.info.format {
            SampleFormat::Int(8) => {
                out.extend(self.raw.iter().map(|&b| (b as f32 - 128.0) / 128.0))
            }
            SampleFormat::Int(16) => out.extend(
                self.raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes(*c) as f32 / 32768.0),
            ),
            SampleFormat::Int(24) => {
                out.extend(
                    self.raw.as_chunks::<3>().0.iter().map(|c| {
                        (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0
                    }),
                )
            }
            SampleFormat::Int(32) => out.extend(
                self.raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| (i32::from_le_bytes(*c) as f64 / 2_147_483_648.0) as f32),
            ),
            SampleFormat::Float(32) => out.extend(
                self.raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| sanitize(f32::from_le_bytes(*c))),
            ),
            SampleFormat::Float(64) => out.extend(
                self.raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| sanitize(f64::from_le_bytes(*c) as f32)),
            ),
            _ => return Err(invalid("unsupported sample format")),
        }
        Ok(n)
    }
}

/// Replaces NaN and infinities with silence.
#[inline]
pub fn sanitize(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Output sample encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteFormat {
    Pcm16,
    Pcm24,
    Float32,
}

/// Streams interleaved `f32` samples into a WAV file. Sizes are patched in
/// [`WavWriter::finish`]. The float layout puts the first sample at byte 68
/// (fmt with `cbSize`, a `fact` chunk and a small `JUNK` pad), which keeps
/// the data 4-byte aligned for memory mapping while staying within spec.
pub struct WavWriter {
    out: BufWriter<File>,
    format: WriteFormat,
    channels: u16,
    frames: u64,
    data_size_pos: u64,
    fact_pos: Option<u64>,
    buf: Vec<u8>,
}

impl WavWriter {
    pub fn create(path: &Path, format: WriteFormat, channels: u16, rate: u32) -> io::Result<Self> {
        Self::from_file(File::create(path)?, format, channels, rate)
    }

    pub(crate) fn from_file(
        file: File,
        format: WriteFormat,
        channels: u16,
        rate: u32,
    ) -> io::Result<Self> {
        let mut out = BufWriter::with_capacity(1 << 16, file);
        let bits: u16 = match format {
            WriteFormat::Pcm16 => 16,
            WriteFormat::Pcm24 => 24,
            WriteFormat::Float32 => 32,
        };
        let block_align = channels * bits / 8;
        out.write_all(b"RIFF\0\0\0\0WAVE")?;
        let (fact_pos, data_size_pos) = match format {
            WriteFormat::Float32 => {
                out.write_all(b"fmt ")?;
                out.write_all(&18u32.to_le_bytes())?;
                out.write_all(&3u16.to_le_bytes())?;
                out.write_all(&channels.to_le_bytes())?;
                out.write_all(&rate.to_le_bytes())?;
                out.write_all(&(rate * block_align as u32).to_le_bytes())?;
                out.write_all(&block_align.to_le_bytes())?;
                out.write_all(&bits.to_le_bytes())?;
                out.write_all(&0u16.to_le_bytes())?; // cbSize
                out.write_all(b"fact")?;
                out.write_all(&4u32.to_le_bytes())?;
                out.write_all(&0u32.to_le_bytes())?; // patched: frames
                out.write_all(b"JUNK")?;
                out.write_all(&2u32.to_le_bytes())?;
                out.write_all(&[0, 0])?;
                out.write_all(b"data\0\0\0\0")?;
                (Some(46u64), 64u64)
            }
            _ => {
                out.write_all(b"fmt ")?;
                out.write_all(&16u32.to_le_bytes())?;
                out.write_all(&1u16.to_le_bytes())?;
                out.write_all(&channels.to_le_bytes())?;
                out.write_all(&rate.to_le_bytes())?;
                out.write_all(&(rate * block_align as u32).to_le_bytes())?;
                out.write_all(&block_align.to_le_bytes())?;
                out.write_all(&bits.to_le_bytes())?;
                out.write_all(b"data\0\0\0\0")?;
                (None, 40u64)
            }
        };
        Ok(WavWriter {
            out,
            format,
            channels,
            frames: 0,
            data_size_pos,
            fact_pos,
            buf: Vec::new(),
        })
    }

    /// Appends interleaved samples (a whole number of frames).
    pub fn write(&mut self, samples: &[f32]) -> io::Result<()> {
        debug_assert_eq!(samples.len() % self.channels as usize, 0);
        self.buf.clear();
        match self.format {
            WriteFormat::Float32 => {
                for &s in samples {
                    self.buf.extend_from_slice(&sanitize(s).to_le_bytes());
                }
            }
            WriteFormat::Pcm16 => {
                for &s in samples {
                    let v = (sanitize(s).clamp(-1.0, 1.0) * 32767.0).round() as i16;
                    self.buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            WriteFormat::Pcm24 => {
                for &s in samples {
                    let v = (sanitize(s).clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
                    self.buf.extend_from_slice(&v.to_le_bytes()[0..3]);
                }
            }
        }
        self.out.write_all(&self.buf)?;
        self.frames += (samples.len() / self.channels as usize) as u64;
        Ok(())
    }

    /// Patches the sizes, flushes and syncs the file.
    pub fn finish(mut self) -> io::Result<()> {
        let bytes_per_sample: u64 = match self.format {
            WriteFormat::Pcm16 => 2,
            WriteFormat::Pcm24 => 3,
            WriteFormat::Float32 => 4,
        };
        let data_len = self.frames * self.channels as u64 * bytes_per_sample;
        let pad = data_len & 1;
        if pad == 1 {
            self.out.write_all(&[0])?;
        }
        let riff_len = self.data_size_pos + 4 + data_len + pad - 8;
        if riff_len > u32::MAX as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAV file would exceed 4 GB",
            ));
        }
        let mut f = self.out.into_inner().map_err(|e| e.into_error())?;
        f.seek(SeekFrom::Start(4))?;
        f.write_all(&(riff_len as u32).to_le_bytes())?;
        if let Some(p) = self.fact_pos {
            f.seek(SeekFrom::Start(p))?;
            f.write_all(&(self.frames as u32).to_le_bytes())?;
        }
        f.seek(SeekFrom::Start(self.data_size_pos))?;
        f.write_all(&(data_len as u32).to_le_bytes())?;
        f.sync_all()
    }
}

/// Writes a whole interleaved buffer as a working-format WAV (tests, small files).
pub fn write_working(path: &Path, samples: &[f32]) -> io::Result<()> {
    let mut w = WavWriter::create(path, WriteFormat::Float32, WORK_CHANNELS, WORK_RATE)?;
    w.write(samples)?;
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ss-wav-{}", crate::util::new_session_id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    #[test]
    fn float_round_trip_is_aligned_and_exact() {
        let p = tmp("a.wav");
        let samples: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.01).sin()).collect();
        write_working(&p, &samples).unwrap();
        let mut f = File::open(&p).unwrap();
        let info = read_info(&mut f).unwrap();
        assert!(info.is_working_format());
        assert_eq!(info.data_offset, 68);
        assert_eq!(info.data_offset % 4, 0);
        assert_eq!(info.frames(), 1000);
        let mut r = WavReader::open(&p).unwrap();
        let mut out = vec![];
        while r.read_frames(300, &mut out).unwrap() > 0 {}
        assert_eq!(out, samples);
    }

    #[test]
    fn pcm_round_trip_is_close() {
        for (fmt, tol) in [(WriteFormat::Pcm16, 1e-4), (WriteFormat::Pcm24, 1e-6)] {
            let p = tmp("b.wav");
            let samples: Vec<f32> = (0..200).map(|i| ((i as f32) * 0.05).sin() * 0.9).collect();
            let mut w = WavWriter::create(&p, fmt, 1, 22_050).unwrap();
            w.write(&samples).unwrap();
            w.finish().unwrap();
            let mut r = WavReader::open(&p).unwrap();
            assert_eq!(r.info.sample_rate, 22_050);
            assert_eq!(r.info.channels, 1);
            let mut out = vec![];
            r.read_frames(1000, &mut out).unwrap();
            assert_eq!(out.len(), samples.len());
            for (a, b) in out.iter().zip(&samples) {
                assert!((a - b).abs() < tol, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn rejects_garbage_and_lying_headers() {
        let p = tmp("c.wav");
        std::fs::write(&p, b"RIFF\x10\0\0\0WAVEjunkjunk").unwrap();
        assert!(read_info(&mut File::open(&p).unwrap()).is_err());
        std::fs::write(&p, b"ID3\x03 not a wav").unwrap();
        assert!(read_info(&mut File::open(&p).unwrap()).is_err());
        // A data chunk that claims 4 GB but holds 8 bytes is clamped to 8.
        let mut v = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0".to_vec();
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&44_100u32.to_le_bytes());
        v.extend_from_slice(&(44_100u32 * 4).to_le_bytes());
        v.extend_from_slice(&4u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data\xff\xff\xff\x7f");
        v.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p, &v).unwrap();
        let info = read_info(&mut File::open(&p).unwrap()).unwrap();
        assert_eq!(info.data_len, 8);
        assert_eq!(info.frames(), 2);
        // Absurd channel count.
        v[22] = 0xFF;
        std::fs::write(&p, &v).unwrap();
        assert!(read_info(&mut File::open(&p).unwrap()).is_err());
    }
}
