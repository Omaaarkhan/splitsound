//! Exporting stems and mixdowns.
//!
//! Formats: WAV at 16-bit (TPDF dithered), 24-bit and 32-bit float; FLAC
//! (24-bit, pure-Rust encoder); MP3 (320 kbit/s CBR through LAME) when the
//! crate is built with its default `mp3` feature.
//!
//! Every file is written to a temporary sibling and renamed into place, so
//! a cancelled or failed export never leaves a partial file under the final
//! name.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::audio::db_to_gain;
use crate::error::{Error, ErrorCode, Result};
use crate::progress::{Progress, check_cancelled};

/// Output format; `ExportFormat` in the UI contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// 16-bit PCM WAV, TPDF dithered.
    Wav16,
    /// 24-bit PCM WAV.
    Wav24,
    /// 32-bit float WAV (bit-exact copy of the working data).
    Wav32f,
    /// FLAC, 24-bit.
    Flac,
    /// MP3, 320 kbit/s.
    Mp3,
}

impl ExportFormat {
    /// File extension without the dot.
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Wav16 | ExportFormat::Wav24 | ExportFormat::Wav32f => "wav",
            ExportFormat::Flac => "flac",
            ExportFormat::Mp3 => "mp3",
        }
    }

    /// Parses the contract's id (`wav16`, `wav24`, `wav32f`, `flac`, `mp3`).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "wav16" => Some(ExportFormat::Wav16),
            "wav24" => Some(ExportFormat::Wav24),
            "wav32f" => Some(ExportFormat::Wav32f),
            "flac" => Some(ExportFormat::Flac),
            "mp3" => Some(ExportFormat::Mp3),
            _ => None,
        }
    }

    /// Whether this build can write the format (MP3 needs the `mp3`
    /// cargo feature).
    pub fn is_available(self) -> bool {
        self != ExportFormat::Mp3 || cfg!(feature = "mp3")
    }
}

/// Uniform random numbers for dither: xorshift, fixed seed, so an export
/// is reproducible.
struct Dither(u64);

impl Dither {
    /// Triangular noise in (-1, 1) LSB: the sum of two uniform values.
    fn tpdf(&mut self) -> f32 {
        let mut next = || {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        next() + next()
    }
}

fn pcm_header(bits: u16, channels: u16, sample_rate: u32, data_bytes: u64) -> [u8; 44] {
    let block = channels * bits / 8;
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(
        &u32::try_from(data_bytes + 36)
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    h[8..16].copy_from_slice(b"WAVEfmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&(sample_rate * u32::from(block)).to_le_bytes());
    h[32..34].copy_from_slice(&block.to_le_bytes());
    h[34..36].copy_from_slice(&bits.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&u32::try_from(data_bytes).unwrap_or(u32::MAX).to_le_bytes());
    h
}

fn sanitize(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

fn write_wav_pcm(
    out: &mut impl Write,
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    bits: u16,
    progress: &dyn Progress,
) -> std::io::Result<Result<()>> {
    let bytes_per = usize::from(bits / 8);
    out.write_all(&pcm_header(
        bits,
        channels,
        sample_rate,
        (samples.len() * bytes_per) as u64,
    ))?;
    let mut dither = Dither(0x2545_F491_4F6C_DD1D);
    let mut buf = Vec::with_capacity(8192 * bytes_per);
    for chunk in samples.chunks(8192) {
        if progress.is_cancelled() {
            return Ok(Err(Error::cancelled()));
        }
        buf.clear();
        for &s in chunk {
            let s = sanitize(s);
            if bits == 16 {
                // Dither before rounding so the quantisation error is
                // noise rather than distortion.
                let v = (s * 32768.0 + dither.tpdf())
                    .round()
                    .clamp(-32768.0, 32767.0) as i16;
                buf.extend_from_slice(&v.to_le_bytes());
            } else {
                let v = (f64::from(s) * 8_388_608.0)
                    .round()
                    .clamp(-8_388_608.0, 8_388_607.0) as i32;
                buf.extend_from_slice(&v.to_le_bytes()[..3]);
            }
        }
        out.write_all(&buf)?;
    }
    Ok(Ok(()))
}

/// Feeds `f32` samples to the FLAC encoder as 24-bit integers without
/// first copying the whole track.
struct FlacSource<'a> {
    samples: &'a [f32],
    channels: usize,
    sample_rate: usize,
    pos: usize,
    scratch: Vec<i32>,
}

impl flacenc::source::Source for FlacSource<'_> {
    fn channels(&self) -> usize {
        self.channels
    }
    fn bits_per_sample(&self) -> usize {
        24
    }
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }
    fn read_samples<F: flacenc::source::Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> std::result::Result<usize, flacenc::error::SourceError> {
        let want = block_size * self.channels;
        let end = (self.pos + want).min(self.samples.len());
        self.scratch.clear();
        self.scratch
            .extend(self.samples[self.pos..end].iter().map(|s| {
                (f64::from(sanitize(*s)) * 8_388_608.0)
                    .round()
                    .clamp(-8_388_608.0, 8_388_607.0) as i32
            }));
        dest.fill_interleaved(&self.scratch)?;
        self.pos = end;
        Ok(self.scratch.len() / self.channels)
    }
    fn len_hint(&self) -> Option<usize> {
        Some(self.samples.len() / self.channels)
    }
}

fn encode_flac(samples: &[f32], channels: u16, sample_rate: u32) -> Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, e)| Error::internal(format!("flac config: {e}")))?;
    let source = FlacSource {
        samples,
        channels: usize::from(channels),
        sample_rate: sample_rate as usize,
        pos: 0,
        scratch: Vec::new(),
    };
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| Error::internal(format!("flac encode: {e}")))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| Error::internal(format!("flac write: {e}")))?;
    let mut bytes = sink.into_inner();
    // flacenc records the short final block as the stream's minimum block
    // size. The FLAC format defines that field as excluding the last block
    // (RFC 9639, section 8.2); with min != max a decoder takes the stream
    // for variable-blocksize, and Symphonia then fails to read it. Every
    // block but the last has the fixed size, so set min = max.
    let frames = samples.len() / usize::from(channels);
    let is_streaminfo_first = bytes.len() >= 12 && &bytes[0..4] == b"fLaC" && bytes[4] & 0x7F == 0;
    if is_streaminfo_first && frames > config.block_size {
        let (min, max) = bytes[8..12].split_at_mut(2);
        min.copy_from_slice(max);
    }
    Ok(bytes)
}

#[cfg(feature = "mp3")]
fn encode_mp3(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    progress: &dyn Progress,
) -> Result<Vec<u8>> {
    use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, InterleavedPcm, MonoPcm, Quality};
    let err = |e: &dyn std::fmt::Display| Error::internal(format!("mp3 encoder: {e}"));
    let mut builder = Builder::new().ok_or_else(Error::out_of_memory)?;
    builder
        .set_num_channels(channels as u8)
        .map_err(|e| err(&e))?;
    builder.set_sample_rate(sample_rate).map_err(|e| err(&e))?;
    builder.set_brate(Bitrate::Kbps320).map_err(|e| err(&e))?;
    builder
        .set_quality(Quality::NearBest)
        .map_err(|e| err(&e))?;
    let mut encoder = builder.build().map_err(|e| err(&e))?;
    let mut out: Vec<u8> = Vec::new();
    let mut clipped: Vec<f32> = Vec::new();
    // About one second per call.
    let chunk_len = sample_rate as usize * usize::from(channels);
    for chunk in samples.chunks(chunk_len.max(usize::from(channels))) {
        check_cancelled(progress)?;
        clipped.clear();
        clipped.extend(chunk.iter().map(|s| sanitize(*s).clamp(-1.0, 1.0)));
        let frames = clipped.len() / usize::from(channels);
        out.try_reserve(mp3lame_encoder::max_required_buffer_size(frames))
            .map_err(|_| Error::out_of_memory())?;
        if channels == 1 {
            encoder
                .encode_to_vec(MonoPcm(&clipped[..]), &mut out)
                .map_err(|e| err(&e))?;
        } else {
            encoder
                .encode_to_vec(InterleavedPcm(&clipped[..]), &mut out)
                .map_err(|e| err(&e))?;
        }
    }
    out.try_reserve(mp3lame_encoder::max_required_buffer_size(1152))
        .map_err(|_| Error::out_of_memory())?;
    encoder
        .flush_to_vec::<FlushNoGap>(&mut out)
        .map_err(|e| err(&e))?;
    Ok(out)
}

#[cfg(not(feature = "mp3"))]
fn encode_mp3(_: &[f32], _: u16, _: u32, _: &dyn Progress) -> Result<Vec<u8>> {
    Err(Error::new(
        ErrorCode::UnsupportedFormat,
        "MP3 export is not included in this build of SplitSound",
    ))
}

/// Writes interleaved `samples` (`channels` channels at `sample_rate`) to
/// `dst` in `format`.
///
/// # Errors
/// `unsupported_format` (MP3 in a build without it, or more than two
/// channels), `cancelled`, `disk_full`, `out_of_memory`.
pub fn export_audio(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    format: ExportFormat,
    dst: &Path,
    progress: &dyn Progress,
) -> Result<()> {
    if channels == 0 || channels > 2 || !samples.len().is_multiple_of(usize::from(channels)) {
        return Err(Error::new(
            ErrorCode::UnsupportedFormat,
            "Only mono and stereo audio can be exported",
        ));
    }
    check_cancelled(progress)?;
    let werr = |e: std::io::Error| Error::from_write(&e, dst);
    match format {
        ExportFormat::Wav32f => {
            let mut w = crate::wav::WavWriter::create_with_channels(dst, sample_rate, channels)?;
            for chunk in samples.chunks(1 << 16) {
                check_cancelled(progress)?;
                w.write_interleaved(chunk)?;
            }
            // WavWriter does its own temp-and-rename.
            w.finish()
        }
        ExportFormat::Wav16 | ExportFormat::Wav24 => {
            let bits = if format == ExportFormat::Wav16 {
                16
            } else {
                24
            };
            let (file, tmp) = crate::wav::create_temp_sibling(dst).map_err(werr)?;
            let mut out = BufWriter::with_capacity(1 << 16, file);
            write_wav_pcm(&mut out, samples, channels, sample_rate, bits, progress)
                .map_err(werr)??;
            let file = out.into_inner().map_err(|e| werr(e.into_error()))?;
            file.sync_all().map_err(werr)?;
            drop(file);
            tmp.persist(dst).map_err(werr)
        }
        ExportFormat::Flac | ExportFormat::Mp3 => {
            let bytes = if format == ExportFormat::Flac {
                encode_flac(samples, channels, sample_rate)?
            } else {
                encode_mp3(samples, channels, sample_rate, progress)?
            };
            check_cancelled(progress)?;
            let (mut file, tmp) = crate::wav::create_temp_sibling(dst).map_err(werr)?;
            file.write_all(&bytes).map_err(werr)?;
            file.sync_all().map_err(werr)?;
            drop(file);
            tmp.persist(dst).map_err(werr)
        }
    }
}

/// Mixes several interleaved stereo sources, each with a gain in dB, into
/// one interleaved buffer as long as the longest source. A gain at or
/// below -120 dB mutes the source.
pub fn mixdown(sources: &[(&[f32], f32)]) -> Result<Vec<f32>> {
    let len = sources.iter().map(|(s, _)| s.len()).max().unwrap_or(0);
    let mut out: Vec<f32> = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error::out_of_memory())?;
    out.resize(len, 0.0);
    for (samples, gain_db) in sources {
        let gain = db_to_gain(*gain_db);
        if gain == 0.0 {
            continue;
        }
        for (o, s) in out.iter_mut().zip(samples.iter()) {
            *o += s * gain;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::{CancelFlag, NoProgress};
    use crate::testutil::TempDir;

    fn tone(frames: usize, amp: f32) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let t = i as f32 / 44_100.0;
                [
                    amp * (2.0 * std::f32::consts::PI * 440.0 * t).sin(),
                    amp * (2.0 * std::f32::consts::PI * 660.0 * t).sin(),
                ]
            })
            .collect()
    }

    #[test]
    fn ids_and_extensions() {
        for (id, ext) in [
            ("wav16", "wav"),
            ("wav24", "wav"),
            ("wav32f", "wav"),
            ("flac", "flac"),
            ("mp3", "mp3"),
        ] {
            let f = ExportFormat::parse(id).unwrap();
            assert_eq!(f.extension(), ext);
            assert_eq!(serde_json::to_string(&f).unwrap(), format!("\"{id}\""));
        }
        assert!(ExportFormat::parse("ogg").is_none());
        assert!(ExportFormat::Flac.is_available());
        assert_eq!(ExportFormat::Mp3.is_available(), cfg!(feature = "mp3"));
    }

    #[test]
    fn wav_exports_round_trip() {
        let dir = TempDir::new("export-wav");
        let x = tone(22_050, 0.5);
        for (format, tol) in [
            (ExportFormat::Wav32f, 0.0f32),
            (ExportFormat::Wav24, 1.0 / 8_388_608.0),
            // Dither is up to one LSB each side, plus half an LSB rounding.
            (ExportFormat::Wav16, 1.6 / 32_768.0),
        ] {
            let p = dir.path().join(format!("{format:?}.wav"));
            export_audio(&x, 2, 44_100, format, &p, &NoProgress).unwrap();
            let back = crate::wav::read_wav(&p, u64::MAX).unwrap();
            assert_eq!(back.spec.sample_rate, 44_100);
            assert_eq!(back.spec.channels, 2);
            assert_eq!(back.samples.len(), x.len());
            let worst = x
                .iter()
                .zip(&back.samples)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst <= tol, "{format:?}: max error {worst}");
        }
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            3,
            "no temp files"
        );
    }

    #[cfg(unix)]
    #[test]
    fn export_does_not_follow_or_remove_a_preexisting_temp_symlink() {
        let dir = TempDir::new("export-temp-symlink");
        let target = dir.path().join("existing.txt");
        std::fs::write(&target, b"keep this").unwrap();
        for format in [ExportFormat::Wav16, ExportFormat::Wav32f] {
            let dest = dir.path().join(format!("{format:?}.wav"));
            // A preexisting symlink at the old predictable PID-based temp name
            // must be neither followed nor removed; export uses an unpredictable
            // `create_new` sibling instead.
            let old_tmp = dir
                .path()
                .join(format!("{format:?}.wav.{}.part", std::process::id()));
            std::os::unix::fs::symlink(&target, &old_tmp).unwrap();
            export_audio(&[0.0, 0.0], 2, 44_100, format, &dest, &NoProgress).unwrap();
            assert!(dest.exists());
            assert_eq!(std::fs::read(&target).unwrap(), b"keep this");
            assert_eq!(std::fs::read_link(&old_tmp).unwrap(), target);
        }
    }

    #[test]
    fn dither_is_triangular_and_decorrelates_the_error() {
        let mut d = Dither(1);
        let n = 200_000;
        let v: Vec<f32> = (0..n).map(|_| d.tpdf()).collect();
        let mean = v.iter().sum::<f32>() / n as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(v.iter().all(|x| x.abs() < 1.0));
        assert!(mean.abs() < 0.01, "{mean}");
        // TPDF of 2 LSB peak-to-peak has variance 1/6 LSB^2.
        assert!((var - 1.0 / 6.0).abs() < 0.01, "{var}");

        // A constant a third of an LSB up: without dither it would round to
        // 0 every time; with dither the average comes out right.
        let dir = TempDir::new("export-dither");
        let level = 1.0 / 3.0 / 32_768.0;
        let x = vec![level; 100_000];
        let p = dir.path().join("d.wav");
        export_audio(&x, 2, 44_100, ExportFormat::Wav16, &p, &NoProgress).unwrap();
        let back = crate::wav::read_wav(&p, u64::MAX).unwrap().samples;
        let avg = back.iter().map(|v| f64::from(*v)).sum::<f64>() / back.len() as f64;
        assert!(
            (avg - f64::from(level)).abs() < f64::from(level) * 0.1,
            "{avg}"
        );
        assert!(back.iter().any(|v| *v != 0.0) && back.contains(&0.0));
    }

    #[test]
    fn clipping_and_bad_values_are_contained() {
        let dir = TempDir::new("export-clip");
        let x = [2.0f32, -2.0, f32::NAN, f32::INFINITY, 1.0, -1.0, 0.0, 0.0];
        let p = dir.path().join("c.wav");
        export_audio(&x, 2, 44_100, ExportFormat::Wav24, &p, &NoProgress).unwrap();
        let back = crate::wav::read_wav(&p, u64::MAX).unwrap().samples;
        assert!((back[0] - 1.0).abs() < 1e-6 && back[1] == -1.0);
        assert_eq!((back[2], back[3]), (0.0, 0.0));
        assert!(back.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }

    #[test]
    fn flac_export_decodes_back_losslessly_at_24_bits() {
        let dir = TempDir::new("export-flac");
        let x = tone(30_000, 0.4);
        let p = dir.path().join("a.flac");
        export_audio(&x, 2, 44_100, ExportFormat::Flac, &p, &NoProgress).unwrap();
        let back = crate::decode::decode_native(&p, &Default::default(), &NoProgress).unwrap();
        assert_eq!(back.info.codec, "FLAC");
        assert_eq!((back.sample_rate, back.channels), (44_100, 2));
        assert_eq!(back.samples.len(), x.len());
        let worst = x
            .iter()
            .zip(&back.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 1.0 / 8_388_608.0, "{worst}");
        // It really is compressed.
        assert!(std::fs::metadata(&p).unwrap().len() < (x.len() * 3) as u64 / 2);
    }

    #[cfg(feature = "mp3")]
    #[test]
    fn mp3_export_decodes_back_close_to_the_source() {
        let dir = TempDir::new("export-mp3");
        let x = tone(44_100, 0.4);
        let p = dir.path().join("a.mp3");
        export_audio(&x, 2, 44_100, ExportFormat::Mp3, &p, &NoProgress).unwrap();
        let back = crate::decode::decode_native(&p, &Default::default(), &NoProgress).unwrap();
        assert_eq!(back.info.codec, "MP3");
        assert_eq!((back.sample_rate, back.channels), (44_100, 2));
        // Length within a couple of MP3 frames; level preserved.
        let frames = back.samples.len() / 2;
        assert!((44_100..44_100 + 3_000).contains(&frames), "{frames}");
        let rms = |v: &[f32]| {
            (v.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let ratio = rms(&back.samples) / rms(&x);
        assert!((ratio - 1.0).abs() < 0.1, "{ratio}");
        // 320 kbit/s for one second is about 40 kB.
        let size = std::fs::metadata(&p).unwrap().len();
        assert!((30_000..60_000).contains(&size), "{size}");
    }

    #[test]
    fn mixdown_applies_gains() {
        let a = [1.0f32, 1.0, 1.0, 1.0];
        let b = [0.5f32, -0.5];
        let m = mixdown(&[(&a, -6.0206), (&b, 0.0), (&a, -200.0)]).unwrap();
        assert_eq!(m.len(), 4);
        assert!((m[0] - 1.0).abs() < 1e-4);
        assert!((m[1] - 0.0).abs() < 1e-4);
        assert!((m[2] - 0.5).abs() < 1e-4);
        assert!(mixdown(&[]).unwrap().is_empty());
    }

    #[test]
    fn cancellation_and_bad_input() {
        let dir = TempDir::new("export-cancel");
        let x = tone(1_000, 0.1);
        let flag = CancelFlag::new();
        flag.cancel();
        let p = dir.path().join("a.wav");
        for f in [
            ExportFormat::Wav16,
            ExportFormat::Wav32f,
            ExportFormat::Flac,
        ] {
            assert!(
                export_audio(&x, 2, 44_100, f, &p, &flag)
                    .unwrap_err()
                    .is_cancelled()
            );
        }
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        assert!(export_audio(&x, 3, 44_100, ExportFormat::Wav16, &p, &NoProgress).is_err());
        assert!(export_audio(&x[..3], 2, 44_100, ExportFormat::Wav16, &p, &NoProgress).is_err());
        let missing = dir.path().join("no/such/dir/a.wav");
        assert!(export_audio(&x, 2, 44_100, ExportFormat::Wav16, &missing, &NoProgress).is_err());
    }
}
