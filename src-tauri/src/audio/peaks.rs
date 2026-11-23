//! Waveform overviews (`Peaks` in the contract).

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::UNIX_EPOCH;

use base64::Engine as _;

use super::mapped::MappedWav;
use crate::ipc::Peaks;
use crate::util::{new_session_id, remove_file_retry, rename_retry};

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

/// Raw signed-byte peak values cast to `u8` (`2 * buckets` bytes).
pub fn compute_bytes(samples: &[f32], buckets: u32) -> Vec<u8> {
    compute(samples, buckets)
        .into_iter()
        .map(|b| b as u8)
        .collect()
}

/// Builds an API `Peaks` struct from already-computed raw peak bytes.
pub fn peaks_from_bytes(buckets: u32, bytes: &[u8]) -> Peaks {
    Peaks {
        buckets: buckets.max(1),
        data: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}

fn wav_fingerprint(path: &Path) -> io::Result<[u8; 20]> {
    let meta = fs::metadata(path)?;
    let len = meta.len();
    let (secs, nanos) = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or((0u64, 0u32), |d| (d.as_secs(), d.subsec_nanos()));
    let mut fp = [0u8; 20];
    fp[0..8].copy_from_slice(&len.to_le_bytes());
    fp[8..16].copy_from_slice(&secs.to_le_bytes());
    fp[16..20].copy_from_slice(&nanos.to_le_bytes());
    Ok(fp)
}

/// Returns raw `2 * buckets` peak bytes for `wav_path`, caching under
/// `<session_dir>/peaks/<source_key>-<buckets>.bin` keyed by WAV size + mtime.
pub fn cached_peak_bytes(
    session_dir: &Path,
    wav_path: &Path,
    source_key: &str,
    buckets: u32,
) -> io::Result<Vec<u8>> {
    let buckets = buckets.max(1);
    let expected = buckets as usize * 2;
    let fp = wav_fingerprint(wav_path)?;
    let peaks_dir = session_dir.join("peaks");
    let cache_path = peaks_dir.join(format!("{source_key}-{buckets}.bin"));
    if let Ok(mut f) = File::open(&cache_path)
        && f.metadata()
            .is_ok_and(|m| m.is_file() && m.len() == (20 + expected) as u64)
    {
        let mut buf = Vec::with_capacity(20 + expected);
        if f.read_to_end(&mut buf).is_ok() && buf.len() == 20 + expected && buf[..20] == fp {
            return Ok(buf.split_off(20));
        }
    }

    let w = MappedWav::open(wav_path)?;
    w.advise_sequential();
    let bytes = compute_bytes(w.samples(), buckets);

    if fs::create_dir_all(&peaks_dir).is_ok() {
        let tmp = peaks_dir.join(format!(
            ".{source_key}-{buckets}.tmp-{}",
            &new_session_id()[..8]
        ));
        if let Ok(mut out) = File::create_new(&tmp) {
            let ok =
                out.write_all(&fp).is_ok() && out.write_all(&bytes).is_ok() && out.flush().is_ok();
            drop(out);
            if !ok || rename_retry(&tmp, &cache_path).is_err() {
                let _ = remove_file_retry(&tmp);
            }
        }
    }
    Ok(bytes)
}

/// Peaks of a working-format WAV file.
pub fn peaks_of_file(path: &Path, buckets: u32) -> io::Result<Peaks> {
    let w = MappedWav::open(path)?;
    w.advise_sequential();
    let bytes = compute_bytes(w.samples(), buckets);
    Ok(peaks_from_bytes(buckets, &bytes))
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
