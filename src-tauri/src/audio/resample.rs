//! Streaming stereo sample-rate converter (windowed-sinc, polyphase table).
//!
//! The mixer works at 44.1 kHz; output devices usually want 48 kHz or
//! something else. This converts on the fly with a 64-tap Kaiser-windowed
//! sinc whose coefficients are precomputed for 512 fractional positions
//! and linearly interpolated between them. Passband is flat to about
//! 19 kHz and images/aliases are attenuated by roughly 80 dB, which is
//! inaudible for monitoring. Equal rates bypass the filter entirely.

use std::sync::Arc;

const TAPS: usize = 64;
const HALF: usize = TAPS / 2;
const PHASES: usize = 512;
const KAISER_BETA: f64 = 9.0;
/// Cutoff relative to the lower of the two Nyquist frequencies.
const CUTOFF: f64 = 0.90;
/// Input frames pulled from the source per refill.
const PULL_FRAMES: usize = 256;

/// Converts interleaved stereo audio from `in_rate` to `out_rate`, pulling
/// input from a callback as it needs it.
pub struct Resampler {
    in_rate: u32,
    out_rate: u32,
    /// Input frames advanced per output frame (`in_rate / out_rate`).
    step: f64,
    /// `(PHASES + 1) * TAPS` coefficients, rows normalised to unity gain.
    table: Arc<[f32]>,
    /// Interleaved stereo input history.
    hist: Vec<f32>,
    /// Read position, in frames, into `hist` (centre of the kernel).
    pos: f64,
    pull: Vec<f32>,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let passthrough = in_rate == out_rate;
        let table: Arc<[f32]> = if passthrough {
            Arc::from(Vec::new())
        } else {
            build_table(CUTOFF * (out_rate as f64 / in_rate as f64).min(1.0)).into()
        };
        let mut r = Resampler {
            in_rate,
            out_rate,
            step: in_rate as f64 / out_rate as f64,
            table,
            hist: Vec::with_capacity((PULL_FRAMES + TAPS) * 4),
            pos: 0.0,
            pull: vec![0.0; PULL_FRAMES * 2],
        };
        r.reset();
        r
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    pub fn is_passthrough(&self) -> bool {
        self.in_rate == self.out_rate
    }

    /// Forgets all history (after a device change).
    pub fn reset(&mut self) {
        self.hist.clear();
        // Pre-roll so output frame 0 lines up with input frame 0.
        self.hist.resize((HALF - 1) * 2, 0.0);
        self.pos = (HALF - 1) as f64;
    }

    /// Fills `out` (interleaved stereo) with converted audio. `source` is
    /// called to fill its argument (interleaved stereo, always the same
    /// length) with the next input frames.
    pub fn process(&mut self, out: &mut [f32], mut source: impl FnMut(&mut [f32])) {
        if self.is_passthrough() {
            source(out);
            return;
        }
        let frames = out.len() / 2;
        for f in 0..frames {
            // Make sure the kernel window [i - HALF + 1, i + HALF] is available.
            let i = self.pos as usize;
            while self.hist.len() / 2 < i + HALF + 1 {
                source(&mut self.pull);
                self.hist.extend_from_slice(&self.pull);
            }
            let frac = self.pos - i as f64;
            let ph = frac * PHASES as f64;
            let p0 = (ph as usize).min(PHASES - 1);
            let a = (ph - p0 as f64) as f32;
            let r0 = &self.table[p0 * TAPS..(p0 + 1) * TAPS];
            let r1 = &self.table[(p0 + 1) * TAPS..(p0 + 2) * TAPS];
            let base = (i + 1 - HALF) * 2;
            let win = &self.hist[base..base + TAPS * 2];
            let (mut l, mut r) = (0.0f32, 0.0f32);
            for k in 0..TAPS {
                let c = r0[k] + (r1[k] - r0[k]) * a;
                l += win[2 * k] * c;
                r += win[2 * k + 1] * c;
            }
            out[2 * f] = l;
            out[2 * f + 1] = r;
            self.pos += self.step;
        }
        // Drop history the kernel no longer needs.
        let keep_from = (self.pos as usize).saturating_sub(HALF - 1);
        if keep_from > 0 {
            self.hist.drain(..keep_from * 2);
            self.pos -= keep_from as f64;
        }
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..64 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Builds the polyphase table for a normalised cutoff `fc` (1.0 = input Nyquist).
fn build_table(fc: f64) -> Vec<f32> {
    let i0b = bessel_i0(KAISER_BETA);
    let mut table = vec![0f32; (PHASES + 1) * TAPS];
    for p in 0..=PHASES {
        let frac = p as f64 / PHASES as f64;
        let mut row = [0f64; TAPS];
        for (k, c) in row.iter_mut().enumerate() {
            // Distance from the interpolation point to tap k's input sample.
            let d = frac + (HALF - 1) as f64 - k as f64;
            let x = d / HALF as f64;
            let w = if x.abs() >= 1.0 {
                0.0
            } else {
                bessel_i0(KAISER_BETA * (1.0 - x * x).sqrt()) / i0b
            };
            let t = std::f64::consts::PI * fc * d;
            let sinc = if t.abs() < 1e-12 { 1.0 } else { t.sin() / t };
            *c = fc * sinc * w;
        }
        let sum: f64 = row.iter().sum();
        for (k, c) in row.iter().enumerate() {
            table[p * TAPS + k] = (c / sum) as f32;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds a stereo sine (L) and cosine (R) at `freq` through the
    /// converter and returns (output, input frames consumed).
    fn run(in_rate: u32, out_rate: u32, freq: f64, out_frames: usize) -> (Vec<f32>, usize) {
        let mut rs = Resampler::new(in_rate, out_rate);
        let mut n = 0usize;
        let mut out = vec![0f32; out_frames * 2];
        // Process in odd-sized blocks to exercise the streaming state.
        let mut done = 0;
        while done < out_frames {
            let blk = (out_frames - done).min(137);
            rs.process(&mut out[done * 2..(done + blk) * 2], |buf| {
                for f in buf.as_chunks_mut::<2>().0 {
                    let t = n as f64 / in_rate as f64;
                    f[0] = (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.5;
                    f[1] = (2.0 * std::f64::consts::PI * freq * t).cos() as f32 * 0.5;
                    n += 1;
                }
            });
            done += blk;
        }
        (out, n)
    }

    #[test]
    fn upsampling_preserves_a_sine() {
        let (out, _) = run(44_100, 48_000, 1000.0, 9600);
        let mut max_err = 0f64;
        // Skip the pre-roll at the start.
        for f in 100..9000 {
            let t = f as f64 / 48_000.0;
            let want = (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 0.5;
            max_err = max_err.max((out[2 * f] as f64 - want).abs());
        }
        assert!(max_err < 1e-3, "max error {max_err}");
    }

    #[test]
    fn downsampling_preserves_a_sine_and_removes_ultrasound() {
        let (out, _) = run(44_100, 32_000, 1000.0, 6400);
        for f in 100..6000 {
            let t = f as f64 / 32_000.0;
            let want = (2.0 * std::f64::consts::PI * 1000.0 * t).cos() * 0.5;
            assert!((out[2 * f + 1] as f64 - want).abs() < 1e-3);
        }
        // 20 kHz is above the 16 kHz output Nyquist and must be filtered out.
        let (out, _) = run(44_100, 32_000, 20_000.0, 6400);
        let peak = out[400..12_000].iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!(peak < 0.5 * 0.01, "alias leaked: {peak}");
    }

    #[test]
    fn consumes_input_at_the_rate_ratio() {
        for (i, o) in [
            (44_100, 48_000),
            (44_100, 96_000),
            (44_100, 22_050),
            (44_100, 44_100),
        ] {
            let out_frames = o as usize * 2; // two seconds
            let (_, consumed) = run(i, o, 440.0, out_frames);
            let expected = i as f64 * 2.0;
            // Within one refill block plus the kernel's look-ahead.
            assert!(
                (consumed as f64 - expected).abs() <= (PULL_FRAMES + TAPS) as f64,
                "{i}->{o}: consumed {consumed}, expected {expected}"
            );
        }
    }

    #[test]
    fn equal_rates_pass_through_bit_exact() {
        let mut rs = Resampler::new(44_100, 44_100);
        let mut out = vec![0f32; 10];
        let mut v = 0.0;
        rs.process(&mut out, |b| {
            for s in b.iter_mut() {
                *s = v;
                v += 1.0;
            }
        });
        assert_eq!(out, (0..10).map(|x| x as f32).collect::<Vec<_>>());
    }
}
