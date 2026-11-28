//! Streaming stereo sample-rate converter (windowed-sinc, polyphase table),
//! hosted in `splitsound-core` so release builds compile it at `-O3`.

pub use splitsound_core::playback::resample::Resampler;

#[cfg(test)]
mod tests {
    use super::*;

    const TAPS: usize = 64;
    const PULL_FRAMES: usize = 256;

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
