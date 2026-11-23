//! High-quality sample-rate conversion (rubato).
//!
//! [`StreamResampler`] converts planar audio incrementally, so a decoder can
//! feed it packet by packet and never hold the source-rate signal in memory.
//! [`resample`] is the one-shot convenience on top of it.
//!
//! Common audio rate pairs whose synchronous FFT block fits within
//! [`MAX_FFT_FRAMES`] use rubato's synchronous FFT resampler; coprime or
//! prime-ratio pairs fall back to rubato's bounded asynchronous sinc
//! resampler so memory and setup time stay small and independent of
//! `gcd(from, to)`.
//!
//! The output is time-aligned with the input (the resampler's start-up
//! delay is trimmed) and exactly `ceil(frames_in * to / from)` frames long.

use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{
    Async, Fft, FixedAsync, FixedSync, Indexing, Resampler, SincInterpolationParameters,
    WindowFunction,
};

use crate::error::{Error, Result};

/// Lowest sample rate the engine accepts.
pub const MIN_SAMPLE_RATE: u32 = 8_000;
/// Highest sample rate the engine accepts.
pub const MAX_SAMPLE_RATE: u32 = 192_000;

/// Largest FFT block (in frames, on either the input or output side) for which
/// the synchronous FFT resampler is used. Standard and broadcast pull-down
/// rate pairs (e.g. `48k -> 44.1k` at 1,280 frames, `44_056 -> 44.1k` at
/// 22,050 frames) stay under this cap; coprime/prime ratios fall back to
/// bounded-chunk sinc interpolation.
const MAX_FFT_FRAMES: usize = 32_768;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

fn rubato_err(e: impl std::fmt::Display) -> Error {
    Error::internal(format!("resampler: {e}"))
}

enum Inner {
    Fft(Box<Fft<f32>>),
    Sinc(Async<f32>),
}

impl Inner {
    fn output_delay(&self) -> usize {
        match self {
            Inner::Fft(r) => r.output_delay(),
            Inner::Sinc(r) => r.output_delay(),
        }
    }

    fn output_frames_max(&self) -> usize {
        match self {
            Inner::Fft(r) => r.output_frames_max(),
            Inner::Sinc(r) => r.output_frames_max(),
        }
    }

    fn input_frames_next(&self) -> usize {
        match self {
            Inner::Fft(r) => r.input_frames_next(),
            Inner::Sinc(r) => r.input_frames_next(),
        }
    }

    fn process_into_buffer(
        &mut self,
        input: &SequentialSliceOfVecs<&[Vec<f32>]>,
        output: &mut SequentialSliceOfVecs<&mut [Vec<f32>]>,
        indexing: &Indexing,
    ) -> std::result::Result<(usize, usize), rubato::ResampleError> {
        match self {
            Inner::Fft(r) => r.process_into_buffer(input, output, Some(indexing)),
            Inner::Sinc(r) => r.process_into_buffer(input, output, Some(indexing)),
        }
    }

    fn reset(&mut self) {
        match self {
            Inner::Fft(r) => r.reset(),
            Inner::Sinc(r) => r.reset(),
        }
    }
}

/// Incremental resampler for `channels` planar channels.
pub struct StreamResampler {
    inner: Option<Inner>,
    channels: usize,
    from: u32,
    to: u32,
    /// Input frames buffered per channel; valid unconsumed frames start at `pending_pos`.
    pending: Vec<Vec<f32>>,
    /// Read cursor (in frames) into each `pending` channel buffer.
    pending_pos: usize,
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
            let g = gcd(from, to);
            let unit_in = (from / g) as usize;
            let unit_out = (to / g) as usize;
            // The FFT resampler works on blocks that are a whole number `k`
            // of the smallest rate-matched block (`from/gcd` frames in,
            // `to/gcd` out). Its filter delays the signal by half a block,
            // which is a whole number of output frames only when `k` is
            // even; an odd `k` would shift the audio by a fraction of a
            // sample that trimming cannot remove. So pick the smallest even
            // `k` that gives blocks of at least ~1024 input frames.
            let k = 2 * 512usize.div_ceil(unit_in).max(1);
            let fft_in = k.saturating_mul(unit_in);
            let fft_out = k.saturating_mul(unit_out);
            let r = if fft_in <= MAX_FFT_FRAMES && fft_out <= MAX_FFT_FRAMES {
                Inner::Fft(Box::new(
                    Fft::<f32>::new_custom(
                        from as usize,
                        to as usize,
                        fft_in,
                        1,
                        channels,
                        WindowFunction::BlackmanHarris2,
                        FixedSync::Input,
                    )
                    .map_err(rubato_err)?,
                ))
            } else {
                Self::new_sinc_inner(from, to, channels)?
            };
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
            pending_pos: 0,
            chunk_out,
            to_trim,
            frames_in: 0,
            frames_out: 0,
        })
    }

    fn new_sinc_inner(from: u32, to: u32, channels: usize) -> Result<Inner> {
        let ratio = f64::from(to) / f64::from(from);
        let chunk_in = if to > from {
            (2048 * from as usize / to as usize).max(128)
        } else {
            1024
        };
        let params = SincInterpolationParameters::default();
        Ok(Inner::Sinc(
            Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_in, channels, FixedAsync::Input)
                .map_err(rubato_err)?,
        ))
    }

    /// Upper bound on how many frames a `push` of `input_frames` can append
    /// to each output channel, for fallible pre-reservation (`try_reserve`).
    pub fn max_output_for(&self, input_frames: usize) -> usize {
        if self.inner.is_none() {
            return input_frames;
        }
        let scaled = (input_frames as u64 * u64::from(self.to)).div_ceil(u64::from(self.from));
        scaled as usize + self.chunk_out.first().map_or(0, Vec::len)
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

    /// Resets internal resampler state so this instance can be reused on a new
    /// clip with the same `from`/`to`/`channels` configuration without
    /// rebuilding the FFT/Sinc filter tables.
    pub fn reset(&mut self) {
        if let Some(r) = self.inner.as_mut() {
            r.reset();
            self.to_trim = r.output_delay();
        } else {
            self.to_trim = 0;
        }
        for p in &mut self.pending {
            p.clear();
        }
        self.pending_pos = 0;
        self.frames_in = 0;
        self.frames_out = 0;
    }

    fn drain(&mut self, out: &mut [Vec<f32>], flush: bool) -> Result<()> {
        loop {
            let need = match &self.inner {
                Some(r) => r.input_frames_next(),
                None => return Ok(()),
            };
            let have = self.pending[0].len() - self.pending_pos;
            if have >= need {
                self.process_chunk(out, None)?;
            } else if flush && have > 0 {
                self.process_chunk(out, Some(have))?;
            } else {
                break;
            }
        }
        if self.pending_pos > 0 {
            let consumed = self.pending_pos;
            for p in self.pending.iter_mut() {
                p.drain(..consumed);
            }
            self.pending_pos = 0;
        }
        Ok(())
    }

    /// Runs one resampler chunk. `partial` is the number of valid pending
    /// frames when fewer than a full chunk remain (0 pumps silence).
    fn process_chunk(&mut self, out: &mut [Vec<f32>], partial: Option<usize>) -> Result<()> {
        let Some(r) = self.inner.as_mut() else {
            return Ok(());
        };
        let need = r.input_frames_next();
        let valid = partial.unwrap_or(need);
        let req_len = self.pending_pos + need;
        // The adapter needs `need` frames from `pending_pos` even for a partial chunk.
        for p in self.pending.iter_mut() {
            if p.len() < req_len {
                p.resize(req_len, 0.0);
            }
        }
        let out_cap = self.chunk_out[0].len();
        let avail = self.pending[0].len();
        let input =
            SequentialSliceOfVecs::new(&self.pending, self.channels, avail).map_err(rubato_err)?;
        let mut output =
            SequentialSliceOfVecs::new_mut(&mut self.chunk_out, self.channels, out_cap)
                .map_err(rubato_err)?;
        let indexing = Indexing {
            input_offset: self.pending_pos,
            output_offset: 0,
            partial_len: partial,
            active_channels_mask: None,
        };
        let (used, produced) = r
            .process_into_buffer(&input, &mut output, &indexing)
            .map_err(rubato_err)?;
        let consumed = if partial.is_some() {
            valid.max(used.min(need))
        } else {
            used
        };
        if partial.is_some() {
            for p in self.pending.iter_mut() {
                p.clear();
            }
            self.pending_pos = 0;
        } else {
            self.pending_pos += consumed.min(self.pending[0].len() - self.pending_pos);
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
        assert!(StreamResampler::new(4_000, 44_100, 2).is_err());
        assert!(StreamResampler::new(383_987, 44_100, 2).is_err());
        assert!(StreamResampler::new(44_100, 10_000_000, 2).is_err());
        assert!(StreamResampler::new(44_100, 48_000, 0).is_err());
    }

    /// SNR of `got` against a unity-amplitude sine at `freq` after allowing
    /// for a constant sub-sample phase offset (returns `(snr_db, delay_samples)`).
    fn sine_snr_phase_aligned(got: &[f32], freq: f64, rate: u32, edge: usize) -> (f64, f64) {
        let slice = &got[edge..got.len() - edge];
        let w = 2.0 * std::f64::consts::PI * freq / f64::from(rate);
        let (mut sum_s, mut sum_c) = (0.0f64, 0.0f64);
        for (k, &v) in slice.iter().enumerate() {
            let t = w * (edge + k) as f64;
            sum_s += f64::from(v) * t.sin();
            sum_c += f64::from(v) * t.cos();
        }
        let phi = sum_c.atan2(sum_s);
        let ideal: Vec<f32> = (edge..got.len() - edge)
            .map(|i| (w * i as f64 + phi).sin() as f32)
            .collect();
        let snr = crate::audio::snr_db(&ideal, slice);
        let delay_samples = phi / w;
        (snr, delay_samples)
    }

    #[test]
    fn prime_ratio_uses_bounded_sinc_and_stays_fast() {
        // Large prime coprime to 44_100 inside the supported range: previously
        // this forced an FFT block of 2 * 191_999 frames (~60 MB).
        let t0 = std::time::Instant::now();
        let rs = StreamResampler::new(191_999, 44_100, 2).unwrap();
        assert!(matches!(rs.inner, Some(Inner::Sinc(_))));
        assert!(
            rs.chunk_out[0].len() <= 4096,
            "chunk_out too large: {}",
            rs.chunk_out[0].len()
        );
        let x = sine(1_000.0, 191_999, 191_999);
        let y = resample(&[x.clone(), x], 191_999, 44_100).unwrap();
        assert!(t0.elapsed() < std::time::Duration::from_secs(2));
        let expected = (191_999u64 * 44_100).div_ceil(191_999) as usize;
        assert_eq!(y[0].len(), expected);
        let (prime_snr, prime_delay) = sine_snr_phase_aligned(&y[0], 1_000.0, 44_100, 2_000);
        eprintln!(
            "191999->44.1k 1kHz sinc SNR (phase-aligned)={prime_snr:.1} dB, residual delay={prime_delay:+.3} samples"
        );
        assert!(prime_snr > 80.0, "prime sinc SNR {prime_snr:.1} dB");
        assert!(prime_delay.abs() < 1.5, "residual delay {prime_delay:.3}");
        // Upsampling from a prime rate also keeps chunk buffers bounded.
        let up = StreamResampler::new(8_009, 44_100, 2).unwrap();
        assert!(matches!(up.inner, Some(Inner::Sinc(_))));
        assert!(
            up.chunk_out[0].len() <= 4096,
            "upsample chunk_out too large: {}",
            up.chunk_out[0].len()
        );
    }

    #[test]
    fn standard_48k_to_44k1_keeps_fft_path_and_measures_quality() {
        let rs = StreamResampler::new(48_000, 44_100, 2).unwrap();
        assert!(matches!(rs.inner, Some(Inner::Fft(_))));
        let frames = 96_000;
        let x = sine(1_000.0, 48_000, frames);
        let y_fft = resample(std::slice::from_ref(&x), 48_000, 44_100).unwrap();
        let snr_fft = sine_snr(&y_fft[0], 1_000.0, 44_100, 2_000);

        // Also run the sinc fallback on the same 48 kHz -> 44.1 kHz signal to
        // verify its waveform quality directly (allowing for Async's constant
        // sub-sample group delay).
        let inner = StreamResampler::new_sinc_inner(48_000, 44_100, 1).unwrap();
        let delay = inner.output_delay();
        let max_out = inner.output_frames_max();
        let mut rs_sinc = StreamResampler {
            inner: Some(inner),
            channels: 1,
            from: 48_000,
            to: 44_100,
            pending: vec![Vec::new()],
            pending_pos: 0,
            chunk_out: vec![vec![0f32; max_out]],
            to_trim: delay,
            frames_in: 0,
            frames_out: 0,
        };
        let mut out_sinc = vec![Vec::new()];
        rs_sinc.push(&[&x], &mut out_sinc).unwrap();
        rs_sinc.finish(&mut out_sinc).unwrap();
        let (snr_sinc, delay_sinc) = sine_snr_phase_aligned(&out_sinc[0], 1_000.0, 44_100, 2_000);
        eprintln!(
            "48k->44.1k 1kHz SNR: fft={snr_fft:.1} dB (zero-phase), sinc={snr_sinc:.1} dB (phase-aligned, residual delay={delay_sinc:+.3} samples)"
        );
        assert!(snr_fft > 100.0, "fft SNR {snr_fft:.1} dB");
        assert!(snr_sinc > 90.0, "sinc SNR {snr_sinc:.1} dB");
        assert!(delay_sinc.abs() < 1.0, "sinc delay {delay_sinc:.3}");
    }
}
