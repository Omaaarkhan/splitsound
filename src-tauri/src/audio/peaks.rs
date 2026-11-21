//! Waveform overviews (`Peaks` in the contract).

use std::io;
use std::path::Path;

use base64::Engine as _;

use super::mapped::MappedWav;
use crate::ipc::Peaks;

/// Largest bucket count a caller may ask for.
pub const MAX_BUCKETS: u32 = 65_536;

/// Min/max per bucket over interleaved stereo (both channels folded
/// together), as `2 * buckets` signed bytes where ±127 is ±1.0.
pub fn compute(samples: &[f32], buckets: u32) -> Vec<i8> {
    let buckets = buckets.max(1) as usize;
    let frames = samples.len() / 2;
    let mut out = vec![0i8; buckets * 2];
    if frames == 0 {
        return out;
    }
    let q = |x: f32| (x.clamp(-1.0, 1.0) * 127.0).round() as i8;
    for b in 0..buckets {
        let start = b * frames / buckets;
        let end = ((b + 1) * frames / buckets).max(start + 1).min(frames);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &s in &samples[start * 2..end * 2] {
            if s.is_finite() {
                lo = lo.min(s);
                hi = hi.max(s);
            }
        }
        if lo.is_finite() {
            out[2 * b] = q(lo);
            out[2 * b + 1] = q(hi);
        }
    }
    out
}

/// Peaks of a working-format WAV file.
pub fn peaks_of_file(path: &Path, buckets: u32) -> io::Result<Peaks> {
    let w = MappedWav::open(path)?;
    let bytes: Vec<u8> = compute(w.samples(), buckets)
        .into_iter()
        .map(|b| b as u8)
        .collect();
    Ok(Peaks {
        buckets: buckets.max(1),
        data: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_hold_min_then_max() {
        let s = [0.5, -0.25, 1.0, 0.0, -1.0, 0.1, 0.2, 0.2];
        let p = compute(&s, 2);
        assert_eq!(p, vec![-32, 127, -127, 25]);
        // More buckets than frames: each bucket still covers at least a frame.
        let p = compute(&s, 8);
        assert_eq!(p.len(), 16);
        assert!(p.iter().any(|x| *x != 0));
        assert_eq!(compute(&[], 4), vec![0; 8]);
        assert_eq!(compute(&[f32::NAN, f32::NAN], 1), vec![0, 0]);
    }
}
