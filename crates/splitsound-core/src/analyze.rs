//! Per-stem measurements: level, how much of the song the stem is audible
//! in, a loudness envelope for the lane, and whether the stem is there at
//! all.
//!
//! All inputs are interleaved samples so the same code runs on in-memory
//! buffers and on a memory-mapped working file.

/// Number of bins in [`StemStats::activity`]; `ACTIVITY_BINS` in the UI
/// contract.
pub const ACTIVITY_BINS: usize = 240;

/// Window used to decide "audible right now": 100 ms at 44.1 kHz.
const WINDOW_FRAMES: usize = 4_410;

/// A window counts as audible when its RMS is above this level...
const ACTIVE_FLOOR_DB: f32 = -50.0;
/// ...and, when the mix is known, no more than this far below the mix in
/// the same window. Separation models leave faint bleed of the whole mix
/// in every stem; a relative gate keeps that from reading as "playing".
const ACTIVE_BELOW_MIX_DB: f32 = 30.0;

/// The envelope maps this level to 0 and full scale to 1.
const ENVELOPE_FLOOR_DB: f32 = -60.0;

/// A stem is `present` only if it is audible for at least this share of the
/// song...
const PRESENT_MIN_ACTIVE: f32 = 0.02;
/// ...or for at least this many seconds in total (a short solo in a long
/// song is still a real part)...
const PRESENT_MIN_ACTIVE_SEC: f32 = 4.0;
/// ...and its overall level is above this.
const PRESENT_MIN_RMS_DB: f32 = -55.0;

/// Measurements of one stem. Field for field the numeric part of `StemInfo`
/// in the UI contract.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StemStats {
    /// False when there is next to nothing in the stem.
    pub present: bool,
    /// Share of the song where the stem is audible, `0..=1`.
    pub active_fraction: f32,
    /// Highest absolute sample, in dBFS (floored at -120).
    pub peak_db: f32,
    /// RMS over the whole stem, in dBFS (floored at -120).
    pub rms_db: f32,
    /// Loudness envelope: [`ACTIVITY_BINS`] values in `0..=1`.
    pub activity: Vec<f32>,
}

/// Sum of squares and peak of frames `start..end`.
type Window<'a> = &'a dyn Fn(usize, usize) -> (f64, f32);

fn analyze_core(
    frames: usize,
    channels: usize,
    sample_rate: u32,
    stem: Window<'_>,
    mix_rms: Option<&[f32]>,
) -> StemStats {
    if frames == 0 {
        return StemStats {
            present: false,
            active_fraction: 0.0,
            peak_db: crate::audio::SILENCE_DB,
            rms_db: crate::audio::SILENCE_DB,
            activity: vec![0.0; ACTIVITY_BINS],
        };
    }
    let rms_of = |sum_sq: f64, count: usize| (sum_sq / count.max(1) as f64).sqrt() as f32;
    let bin_bounds = |b: usize| -> (usize, usize) {
        let s = ((b as u128 * frames as u128 / ACTIVITY_BINS as u128) as usize).min(frames - 1);
        let mut e = ((b as u128 + 1) * frames as u128 / ACTIVITY_BINS as u128) as usize;
        if e <= s {
            e = s + 1;
        }
        (s, e)
    };

    // Single pass over the stem combining 100 ms active-window gating and 240
    // activity-envelope bins cleanly at overlapping interval boundaries.
    let window = (WINDOW_FRAMES as u64 * u64::from(sample_rate.max(1)) / 44_100).max(1) as usize;
    let floor = crate::audio::db_to_gain(ACTIVE_FLOOR_DB);
    let rel = crate::audio::db_to_gain(-ACTIVE_BELOW_MIX_DB);
    let mut windows = 0usize;
    let mut active = 0usize;
    let mut total_sq = 0f64;
    let mut peak = 0f32;

    let mut activity = vec![0.0f32; ACTIVITY_BINS];
    let mut w_start = 0usize;
    let mut w_end = window.min(frames);
    let mut w_sq = 0f64;
    let mut b_idx = 0usize;
    let (mut b_start, mut b_end) = bin_bounds(0);
    let mut b_sq = 0f64;

    let mut pos = 0usize;
    while pos < frames {
        let next = w_end.min(b_end);
        let (seg_sq, seg_peak) = stem(pos, next);
        total_sq += seg_sq;
        peak = peak.max(seg_peak);
        w_sq += seg_sq;
        b_sq += seg_sq;
        pos = next;

        if pos == w_end {
            let w = rms_of(w_sq, (w_end - w_start) * channels);
            let mut is_active = w > floor;
            if is_active && let Some(mix_vals) = mix_rms {
                let m = mix_vals.get(windows).copied().unwrap_or(0.0);
                is_active = w > m * rel;
            }
            windows += 1;
            if is_active {
                active += 1;
            }
            w_start = w_end;
            w_end = (w_start + window).min(frames);
            w_sq = 0.0;
        }

        while b_idx < ACTIVITY_BINS && pos >= b_end {
            let db = crate::audio::amplitude_to_db(rms_of(b_sq, (b_end - b_start) * channels));
            activity[b_idx] = ((db - ENVELOPE_FLOOR_DB) / -ENVELOPE_FLOOR_DB).clamp(0.0, 1.0);
            b_idx += 1;
            b_sq = 0.0;
            if b_idx < ACTIVITY_BINS {
                (b_start, b_end) = bin_bounds(b_idx);
                if b_start < pos {
                    // Sub-frame tail when `frames < ACTIVITY_BINS`: exact slice fallback.
                    let (sub_sq, _) = stem(b_start, b_end);
                    b_sq = sub_sq;
                }
            }
        }
    }
    let active_fraction = active as f32 / windows.max(1) as f32;
    let active_sec = active as f32 * window as f32 / sample_rate.max(1) as f32;

    let rms_db = crate::audio::amplitude_to_db(rms_of(total_sq, frames * channels));
    let present = rms_db > PRESENT_MIN_RMS_DB
        && (active_fraction >= PRESENT_MIN_ACTIVE || active_sec >= PRESENT_MIN_ACTIVE_SEC);

    StemStats {
        present,
        active_fraction,
        peak_db: crate::audio::amplitude_to_db(peak),
        rms_db,
        activity,
    }
}

fn slice_window(samples: &[f32]) -> (f64, f32) {
    let mut sum = 0f64;
    let mut peak = 0f32;
    for &v in samples {
        sum += f64::from(v) * f64::from(v);
        peak = peak.max(v.abs());
    }
    (sum, peak)
}

/// Precomputes the per-window RMS of planar stereo `mix` once so analyzing all
/// stems against it does not rescan `mix` 6 times.
pub fn precompute_mix_rms(mix: &crate::audio::Stereo, sample_rate: u32) -> Vec<f32> {
    let frames = mix.len();
    if frames == 0 {
        return Vec::new();
    }
    let window = (WINDOW_FRAMES as u64 * u64::from(sample_rate.max(1)) / 44_100).max(1) as usize;
    let mut out = Vec::with_capacity(frames.div_ceil(window));
    let mut start = 0usize;
    while start < frames {
        let end = (start + window).min(frames);
        let (l, _) = slice_window(&mix.left[start..end]);
        let (r, _) = slice_window(&mix.right[start..end]);
        let count = (end - start) * 2;
        out.push(((l + r) / count.max(1) as f64).sqrt() as f32);
        start = end;
    }
    out
}

/// Measures a stem.
///
/// `stem` is interleaved with `channels` channels at `sample_rate`. `mix`,
/// when given, is the original mix in the same layout; it sharpens the
/// "audible" decision (see the module constants) and should be passed
/// whenever it is available.
pub fn analyze_stem(
    stem: &[f32],
    mix: Option<&[f32]>,
    channels: usize,
    sample_rate: u32,
) -> StemStats {
    let channels = channels.max(1);
    let frames = stem.len() / channels;
    let stem_window = |s: usize, e: usize| slice_window(&stem[s * channels..e * channels]);
    let mix = mix.filter(|m| m.len() >= frames * channels);
    let mix_rms = mix.map(|m| {
        let window =
            (WINDOW_FRAMES as u64 * u64::from(sample_rate.max(1)) / 44_100).max(1) as usize;
        let mut out = Vec::with_capacity(frames.div_ceil(window));
        let mut start = 0usize;
        while start < frames {
            let end = (start + window).min(frames);
            let (sq, _) = slice_window(&m[start * channels..end * channels]);
            let count = (end - start) * channels;
            out.push((sq / count.max(1) as f64).sqrt() as f32);
            start = end;
        }
        out
    });
    analyze_core(
        frames,
        channels,
        sample_rate,
        &stem_window,
        mix_rms.as_deref(),
    )
}

/// [`analyze_stem`] for planar stereo, using a precomputed `mix_rms` slice
/// from [`precompute_mix_rms`].
pub fn analyze_stereo_with_mix_rms(
    stem: &crate::audio::Stereo,
    mix_rms: Option<&[f32]>,
    sample_rate: u32,
) -> StemStats {
    let frames = stem.len();
    let stem_window = |s: usize, e: usize| {
        let (l, lp) = slice_window(&stem.left[s..e]);
        let (r, rp) = slice_window(&stem.right[s..e]);
        (l + r, lp.max(rp))
    };
    analyze_core(frames, 2, sample_rate, &stem_window, mix_rms)
}

/// [`analyze_stem`] for planar stereo, without copying anything.
pub fn analyze_stereo(
    stem: &crate::audio::Stereo,
    mix: Option<&crate::audio::Stereo>,
    sample_rate: u32,
) -> StemStats {
    let frames = stem.len();
    let mix_rms = mix
        .filter(|m| m.len() >= frames)
        .map(|m| precompute_mix_rms(m, sample_rate));
    analyze_stereo_with_mix_rms(stem, mix_rms.as_deref(), sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frames: usize, amp: f32) -> Vec<f32> {
        let mut v = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let s = amp * (i as f32 * 0.05).sin();
            v.push(s);
            v.push(s);
        }
        v
    }

    #[test]
    fn full_scale_tone() {
        let s = tone(44_100 * 4, 1.0);
        let st = analyze_stem(&s, None, 2, 44_100);
        assert!(st.present);
        assert!((st.active_fraction - 1.0).abs() < 1e-6);
        assert!(st.peak_db > -0.1 && st.peak_db <= 0.0);
        assert!((st.rms_db + 3.01).abs() < 0.05, "{}", st.rms_db);
        assert_eq!(st.activity.len(), ACTIVITY_BINS);
        assert!(st.activity.iter().all(|v| *v > 0.9 && *v <= 1.0));
    }

    #[test]
    fn silence_and_empty() {
        let st = analyze_stem(&vec![0.0; 44_100 * 2], None, 2, 44_100);
        assert!(!st.present);
        assert_eq!(st.active_fraction, 0.0);
        assert_eq!(st.peak_db, -120.0);
        assert_eq!(st.rms_db, -120.0);
        assert!(st.activity.iter().all(|v| *v == 0.0));
        let st = analyze_stem(&[], None, 2, 44_100);
        assert!(!st.present);
        assert_eq!(st.activity.len(), ACTIVITY_BINS);
    }

    #[test]
    fn half_active_and_envelope_shape() {
        // 10 s: silent first half, tone in the second half.
        let mut s = vec![0.0f32; 44_100 * 5 * 2];
        s.extend(tone(44_100 * 5, 0.5));
        let st = analyze_stem(&s, None, 2, 44_100);
        assert!(st.present);
        assert!(
            (st.active_fraction - 0.5).abs() < 0.02,
            "{}",
            st.active_fraction
        );
        assert!(st.activity[..118].iter().all(|v| *v == 0.0));
        assert!(st.activity[122..].iter().all(|v| *v > 0.8));
    }

    #[test]
    fn faint_bleed_is_not_present() {
        // The "stem" is the mix 40 dB down: bleed, not an instrument.
        let mix = tone(44_100 * 6, 0.8);
        let stem: Vec<f32> = mix.iter().map(|v| v * 0.01).collect();
        let with_mix = analyze_stem(&stem, Some(&mix), 2, 44_100);
        assert_eq!(with_mix.active_fraction, 0.0);
        assert!(!with_mix.present);
        // Without the mix for reference the absolute floor still lets it
        // through as "audible" (it is -45 dBFS); that is why callers pass
        // the mix.
        let alone = analyze_stem(&stem, None, 2, 44_100);
        assert!(alone.active_fraction > 0.9);
    }

    #[test]
    fn short_solo_in_long_song_counts() {
        // 5 s of playing in a 10 minute track is under 2% but is a real part.
        let mut s = vec![0.0f32; 44_100 * 595 * 2];
        s.extend(tone(44_100 * 5, 0.5));
        let st = analyze_stem(&s, None, 2, 44_100);
        assert!(st.active_fraction < 0.02);
        assert!(st.present);
        // One second is not enough.
        let mut s = vec![0.0f32; 44_100 * 599 * 2];
        s.extend(tone(44_100, 0.5));
        assert!(!analyze_stem(&s, None, 2, 44_100).present);
    }

    #[test]
    fn planar_and_interleaved_agree() {
        use crate::audio::Stereo;
        let mut s = vec![0.0f32; 44_100 * 3 * 2];
        s.extend(tone(44_100 * 4, 0.4));
        let mix: Vec<f32> = s.iter().map(|v| v * 1.5 + 0.01).collect();
        let a = analyze_stem(&s, Some(&mix), 2, 44_100);
        let b = analyze_stereo(
            &Stereo::from_interleaved(&s),
            Some(&Stereo::from_interleaved(&mix)),
            44_100,
        );
        assert_eq!(a, b);
        assert_eq!(
            analyze_stem(&s, None, 2, 44_100),
            analyze_stereo(&Stereo::from_interleaved(&s), None, 44_100)
        );
        assert!(!analyze_stereo(&Stereo::new(), None, 44_100).present);
    }

    #[test]
    fn serialises_like_the_contract() {
        let st = analyze_stem(&tone(4_410, 0.5), None, 2, 44_100);
        let v = serde_json::to_value(&st).unwrap();
        assert!(v.get("activeFraction").is_some());
        assert!(v.get("peakDb").is_some());
        assert!(v.get("rmsDb").is_some());
        assert_eq!(v["activity"].as_array().unwrap().len(), ACTIVITY_BINS);
    }
}
