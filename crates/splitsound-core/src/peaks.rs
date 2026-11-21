//! Waveform overviews for drawing.
//!
//! [`compute_peaks`] folds interleaved audio into `buckets` (min, max)
//! pairs, each scaled so that `-127..=127` maps to `-1..=1`. This is the
//! `Peaks` shape of the UI contract; [`Peaks::to_base64`] produces its
//! `data` string. Computing 2000 buckets for a 10 minute stereo file takes
//! a few tens of milliseconds straight from a [`crate::wav::MappedWav`], so
//! peaks can be made on demand and need not be cached.

/// Min/max waveform overview.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Peaks {
    /// Number of buckets.
    pub buckets: usize,
    /// `2 * buckets` signed bytes: min then max for each bucket.
    pub data: Vec<i8>,
}

impl Peaks {
    /// Standard base64 (with padding) of the signed bytes, as the UI
    /// contract's `Peaks.data` expects.
    pub fn to_base64(&self) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(self.data.len().div_ceil(3) * 4);
        for chunk in self.data.chunks(3) {
            let b = [
                chunk[0] as u8,
                chunk.get(1).map_or(0, |v| *v as u8),
                chunk.get(2).map_or(0, |v| *v as u8),
            ];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            out.push(TABLE[(n >> 18) as usize & 63] as char);
            out.push(TABLE[(n >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                TABLE[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                TABLE[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }
}

/// Largest bucket count [`compute_peaks`] will produce; larger requests are
/// clamped (a waveform never needs more columns than a very wide screen).
pub const MAX_BUCKETS: usize = 16_384;

fn quantise(v: f32) -> i8 {
    if v.is_nan() {
        0
    } else {
        (v.clamp(-1.0, 1.0) * 127.0).round() as i8
    }
}

/// Computes `buckets` min/max pairs over interleaved `samples` with
/// `channels` channels (all channels are folded together).
///
/// Buckets tile the whole file evenly; when there are fewer frames than
/// buckets, trailing buckets repeat the nearest frame so the overview still
/// spans the full width. `buckets` is clamped to `1..=MAX_BUCKETS`. Empty
/// audio gives all zeros.
pub fn compute_peaks(samples: &[f32], channels: usize, buckets: usize) -> Peaks {
    let buckets = buckets.clamp(1, MAX_BUCKETS);
    let channels = channels.max(1);
    let frames = samples.len() / channels;
    let mut data = vec![0i8; buckets * 2];
    if frames == 0 {
        return Peaks { buckets, data };
    }
    for b in 0..buckets {
        // Integer bucket edges: every frame belongs to exactly one bucket.
        let start = (b as u128 * frames as u128 / buckets as u128) as usize;
        let mut end = ((b as u128 + 1) * frames as u128 / buckets as u128) as usize;
        let start = start.min(frames - 1);
        if end <= start {
            end = start + 1;
        }
        let slice = &samples[start * channels..end * channels];
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in slice {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        data[2 * b] = quantise(lo);
        data[2 * b + 1] = quantise(hi);
    }
    Peaks { buckets, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_and_max_per_bucket() {
        // 8 stereo frames, 4 buckets of 2 frames each.
        let s = [
            0.0,
            0.0,
            1.0,
            -1.0, // bucket 0
            0.5,
            0.25,
            0.25,
            0.5, // bucket 1
            -0.5,
            -0.25,
            -0.1,
            -0.2, // bucket 2
            2.0,
            -3.0,
            f32::NAN,
            0.0, // bucket 3: clamped
        ];
        let p = compute_peaks(&s, 2, 4);
        assert_eq!(p.buckets, 4);
        assert_eq!(p.data, vec![-127, 127, 32, 64, -64, -13, -127, 127]);
    }

    #[test]
    fn fewer_frames_than_buckets_and_empty() {
        let p = compute_peaks(&[0.5, 0.5, -0.5, -0.5], 2, 6);
        assert_eq!(p.data.len(), 12);
        assert_eq!(&p.data[0..2], &[64, 64]);
        assert_eq!(&p.data[10..12], &[-64, -64]);
        let p = compute_peaks(&[], 2, 3);
        assert_eq!(p.data, vec![0; 6]);
        assert_eq!(compute_peaks(&[0.0; 4], 2, 0).buckets, 1);
        assert_eq!(compute_peaks(&[0.0; 4], 2, usize::MAX).buckets, MAX_BUCKETS);
    }

    #[test]
    fn every_frame_is_covered_once() {
        // A single spike must show up in exactly one bucket, wherever it is.
        let frames = 1009;
        for spike in [0, 1, 500, 1007, 1008] {
            let mut s = vec![0f32; frames];
            s[spike] = 1.0;
            let p = compute_peaks(&s, 1, 64);
            let hits = p.data.chunks(2).filter(|c| c[1] == 127).count();
            assert_eq!(hits, 1, "spike at {spike}");
        }
    }

    #[test]
    fn base64_matches_known_vectors() {
        let p = |bytes: &[u8]| Peaks {
            buckets: 0,
            data: bytes.iter().map(|b| *b as i8).collect(),
        };
        assert_eq!(p(b"").to_base64(), "");
        assert_eq!(p(b"f").to_base64(), "Zg==");
        assert_eq!(p(b"fo").to_base64(), "Zm8=");
        assert_eq!(p(b"foo").to_base64(), "Zm9v");
        assert_eq!(p(b"foobar").to_base64(), "Zm9vYmFy");
        assert_eq!(p(&[0x81, 0x7f]).to_base64(), "gX8=");
    }
}
