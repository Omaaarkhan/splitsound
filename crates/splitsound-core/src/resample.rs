//! High-quality sample-rate conversion (rubato's FFT resampler).
//!
//! [`StreamResampler`] converts planar audio incrementally, so a decoder can
//! feed it packet by packet and never hold the source-rate signal in memory.
//! [`resample`] is the one-shot convenience on top of it.
//!
//! The output is time-aligned with the input (the resampler's start-up
//! delay is trimmed) and exactly `ceil(frames_in * to / from)` frames long.

use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Indexing, Resampler, WindowFunction};

use crate::error::{Error, Result};

/// Lowest sample rate the engine accepts.
pub const MIN_SAMPLE_RATE: u32 = 4_000;
/// Highest sample rate the engine accepts.
pub const MAX_SAMPLE_RATE: u32 = 384_000;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

fn rubato_err(e: impl std::fmt::Display) -> Error {
    Error::internal(format!("resampler: {e}"))
}

/// Incremental resampler for `channels` planar channels.
pub struct StreamResampler {
    inner: Option<Fft<f32>>,
    channels: usize,
    from: u32,
    to: u32,
    /// Input frames not yet consumed (planar).
    pending: Vec<Vec<f32>>,
    /// Scratch output for one chunk.
    chunk_out: Vec<Vec<f32>>,
    /// Output frames still to drop to remove the start-up delay.
    to_trim: usize,
    frames_in: u64,
    frames_out: u64,
}

impl StreamResampler {
    /// Creates a resampler from `from` Hz to `to` Hz. When the rates are
    /// equal the audio passes through untouched.
    pub fn new(from: u32, to: u32, channels: usize) -> Result<Self> {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&from)
            || !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&to)
        {
            return Err(Error::unsupported("This sample rate is not supported"));
        }
        if channels == 0 || channels > 8 {
            return Err(Error::internal("resampler: bad channel count"));
        }
        let (inner, to_trim, chunk_out) = if from == to {
            (None, 0, Vec::new())
        } else {
            // The resampler works on FFT blocks that are a whole number `k`
            // of the smallest rate-matched block (`from/gcd` frames in,
            // `to/gcd` out). Its filter delays the signal by half a block,
            // which is a whole number of output frames only when `k` is
            // even; an odd `k` would shift the audio by a fraction of a
            // sample that trimming cannot remove. So pick the smallest even
            // `k` that gives blocks of at least ~1024 input frames.
            let unit_in = (from / gcd(from, to)) as usize;
            let k = 2 * 512usize.div_ceil(unit_in).max(1);
            let r = Fft::<f32>::new_custom(
                from as usize,
                to as usize,
                k * unit_in,
                1,
                channels,
                WindowFunction::BlackmanHarris2,
                FixedSync::Input,
            )
            .map_err(rubato_err)?;
            let delay = r.output_delay();
            let max_out = r.output_frames_max();
            (Some(r), delay, vec![vec![0f32; max_out]; channels])
        };
        Ok(StreamResampler {
            inner,
            channels,
            from,
            to,
            pending: vec![Vec::new(); channels],
            chunk_out,
            to_trim,
            frames_in: 0,
            frames_out: 0,
        })
    }

    /// Feeds planar input (`input.len() == channels`, equal lengths) and
    /// appends whatever output is ready to `out` (also planar).
    pub fn push(&mut self, input: &[&[f32]], out: &mut [Vec<f32>]) -> Result<()> {
        if input.len() != self.channels || out.len() != self.channels {
            return Err(Error::internal("resampler: channel mismatch"));
        }
        let frames = input[0].len();
        if input.iter().any(|c| c.len() != frames) {
            return Err(Error::internal("resampler: ragged input"));
        }
        self.frames_in += frames as u64;
        if self.inner.is_none() {
            for (o, i) in out.iter_mut().zip(input) {
                o.extend_from_slice(i);
            }
            self.frames_out += frames as u64;
            return Ok(());
        }
        for (p, i) in self.pending.iter_mut().zip(input) {
            p.extend_from_slice(i);
        }
        self.drain(out, false)
    }

    /// Flushes the tail and pads or trims so that the total output is
    /// exactly `ceil(frames_in * to / from)` frames.
    pub fn finish(&mut self, out: &mut [Vec<f32>]) -> Result<()> {
        if out.len() != self.channels {
            return Err(Error::internal("resampler: channel mismatch"));
        }
        if self.inner.is_none() {
            return Ok(());
        }
        let expected = (self.frames_in * u64::from(self.to)).div_ceil(u64::from(self.from));
        self.drain(out, true)?;
        // Keep pumping silence until the delayed tail has come out.
        let mut guard = 0;
        while self.frames_out < expected {
            self.process_chunk(out, Some(0))?;
            guard += 1;
            if guard > 1_000 {
                return Err(Error::internal("resampler: tail did not flush"));
            }
        }
        if self.frames_out > expected {
            let excess = (self.frames_out - expected) as usize;
            for o in out.iter_mut() {
                let keep = o.len().saturating_sub(excess);
                o.truncate(keep);
            }
            self.frames_out = expected;
        }
        Ok(())
    }

    /// Total frames produced so far.
    pub fn frames_out(&self) -> u64 {
        self.frames_out
    }

    fn drain(&mut self, out: &mut [Vec<f32>], flush: bool) -> Result<()> {
        loop {
            let need = match &self.inner {
                Some(r) => r.input_frames_next(),
                None => return Ok(()),
            };
            let have = self.pending[0].len();
            if have >= need {
                self.process_chunk(out, None)?;
            } else if flush && have > 0 {
                self.process_chunk(out, Some(have))?;
            } else {
                return Ok(());
            }
        }
    }

    /// Runs one resampler chunk. `partial` is the number of valid pending
    /// frames when fewer than a full chunk remain (0 pumps silence).
    fn process_chunk(&mut self, out: &mut [Vec<f32>], partial: Option<usize>) -> Result<()> {
        let Some(r) = self.inner.as_mut() else {
            return Ok(());
        };
        let need = r.input_frames_next();
        let valid = partial.unwrap_or(need);
        // The adapter needs `need` frames to exist even for a partial chunk.
        for p in self.pending.iter_mut() {
            if p.len() < need {
                p.resize(need, 0.0);
            }
        }
        let out_cap = self.chunk_out[0].len();
        let input =
            SequentialSliceOfVecs::new(&self.pending, self.channels, need).map_err(rubato_err)?;
        let mut output =
            SequentialSliceOfVecs::new_mut(&mut self.chunk_out, self.channels, out_cap)
                .map_err(rubato_err)?;
        let indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: partial,
            active_channels_mask: None,
        };
        let (used, produced) = r
            .process_into_buffer(&input, &mut output, Some(&indexing))
            .map_err(rubato_err)?;
        let consumed = if partial.is_some() {
            valid.max(used.min(need))
        } else {
            used
        };
        for p in self.pending.iter_mut() {
            let n = consumed.min(p.len());
            p.drain(..n);
            if partial.is_some() {
                p.clear();
            }
        }
        let skip = self.to_trim.min(produced);
        self.to_trim -= skip;
        for (o, c) in out.iter_mut().zip(&self.chunk_out) {
            o.extend_from_slice(&c[skip..produced]);
        }
        self.frames_out += (produced - skip) as u64;
        Ok(())
    }
}

/// Resamples whole planar channels from `from` Hz to `to` Hz.
pub fn resample(input: &[Vec<f32>], from: u32, to: u32) -> Result<Vec<Vec<f32>>> {
    let channels = input.len();
    let mut rs = StreamResampler::new(from, to, channels)?;
    let frames = input.first().map_or(0, Vec::len);
    let expected = (frames as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(channels);
    for _ in 0..channels {
        let mut v = Vec::new();
        v.try_reserve_exact(expected + 8192)
            .map_err(|_| Error::out_of_memory())?;
        out.push(v);
    }
    // Feed in blocks so memory stays flat.
    let mut pos = 0;
    while pos < frames {
        let take = (frames - pos).min(1 << 16);
        let refs: Vec<&[f32]> = input.iter().map(|c| &c[pos..pos + take]).collect();
        rs.push(&refs, &mut out)?;
        pos += take;
    }
    rs.finish(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, rate: u32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    /// SNR of `got` against an ideal sine, ignoring `edge` frames at each end.
    fn sine_snr(got: &[f32], freq: f64, rate: u32, edge: usize) -> f64 {
        let ideal = sine(freq, rate, got.len());
        crate::audio::snr_db(&ideal[edge..got.len() - edge], &got[edge..got.len() - edge])
    }

    #[test]
    fn common_rates_are_accurate_and_aligned() {
        for (from, to) in [
            (48_000u32, 44_100u32),
            (44_100, 32_000),
            (96_000, 44_100),
            (22_050, 44_100),
            (8_000, 44_100),
            (44_100, 48_000),
        ] {
            let frames = from as usize * 2;
            let x = sine(1_000.0, from, frames);
            let y = resample(&[x], from, to).unwrap();
            let expected = (frames as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
            assert_eq!(y[0].len(), expected, "{from}->{to} length");
            // A phase or delay error would wreck this comparison.
            let snr = sine_snr(&y[0], 1_000.0, to, 2_000);
            assert!(snr > 70.0, "{from}->{to}: SNR {snr:.1} dB");
        }
    }

    #[test]
    fn awkward_ratio_still_works() {
        let (from, to) = (44_056u32, 44_100u32);
        let x = sine(440.0, from, from as usize);
        let y = resample(&[x], from, to).unwrap();
        assert_eq!(y[0].len(), 44_100);
        assert!(sine_snr(&y[0], 440.0, to, 4_000) > 60.0);
    }

    #[test]
    fn streaming_equals_one_shot_and_keeps_channels_apart() {
        let from = 48_000u32;
        let to = 44_100u32;
        let l = sine(500.0, from, 50_000);
        let r = sine(2_000.0, from, 50_000);
        let whole = resample(&[l.clone(), r.clone()], from, to).unwrap();
        let mut rs = StreamResampler::new(from, to, 2).unwrap();
        let mut out = vec![Vec::new(), Vec::new()];
        // Irregular packet sizes, including tiny and empty ones.
        let mut pos = 0;
        for size in [1usize, 0, 960, 37, 4096, 11, 20_000, 1 << 20] {
            let take = size.min(l.len() - pos);
            rs.push(&[&l[pos..pos + take], &r[pos..pos + take]], &mut out)
                .unwrap();
            pos += take;
        }
        assert_eq!(pos, l.len());
        rs.finish(&mut out).unwrap();
        assert_eq!(out[0].len(), whole[0].len());
        assert_eq!(out, whole);
        assert!(sine_snr(&out[0], 500.0, to, 2_000) > 70.0);
        assert!(sine_snr(&out[1], 2_000.0, to, 2_000) > 70.0);
    }

    #[test]
    fn same_rate_is_a_passthrough() {
        let x = sine(100.0, 44_100, 1_234);
        let y = resample(std::slice::from_ref(&x), 44_100, 44_100).unwrap();
        assert_eq!(y[0], x);
    }

    #[test]
    fn short_and_empty_inputs() {
        for frames in [0usize, 1, 2, 100] {
            let x = vec![0.25f32; frames];
            let y = resample(&[x], 48_000, 44_100).unwrap();
            let expected = (frames as u64 * 44_100).div_ceil(48_000) as usize;
            assert_eq!(y[0].len(), expected, "{frames} frames in");
        }
    }

    #[test]
    fn absurd_rates_are_rejected() {
        assert!(StreamResampler::new(0, 44_100, 2).is_err());
        assert!(StreamResampler::new(44_100, 10_000_000, 2).is_err());
        assert!(StreamResampler::new(44_100, 48_000, 0).is_err());
    }
}
