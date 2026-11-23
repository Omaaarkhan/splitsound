//! Full-track separation common to every model kind: `ml/CONTRACT.md`
//! section 1 (track normalisation, chunk schedule, triangular overlap-add,
//! deterministic shifts), ported from `ml/reference.py`.
//!
//! The model itself is behind [`SegmentModel`]: something that turns one
//! fixed-size stereo segment into one stereo segment per source.
//!
//! # Memory
//! Only the sources the caller asks for are accumulated, each as one stereo
//! track, plus one mono weight track. Nothing is kept per pass or per
//! chunk beyond the model's own segment buffers.

use crate::audio::Stereo;
use crate::error::{Error, Result};

/// Range of the deterministic shifts: half a second at 44.1 kHz.
pub const MAX_SHIFT: usize = 22_050;

/// One fixed-size model invocation.
pub trait SegmentModel {
    /// Samples per segment (`N`).
    fn segment_len(&self) -> usize;

    /// Number of sources the model outputs (`S`).
    fn num_sources(&self) -> usize;

    /// Whether track-level mean/std normalisation is applied around segmentation.
    fn normalize(&self) -> bool {
        true
    }

    /// Separates one segment.
    ///
    /// `left` and `right` hold `segment_len()` samples. For each index in
    /// `wanted` (indices into the model's source list), the matching entry
    /// of `out` must be filled with that source's stereo segment: `out[i]`
    /// is `[left, right]` for source `wanted[i]`, each `segment_len()` long
    /// (the buffers arrive sized; previous contents are garbage).
    fn run_segment(
        &mut self,
        left: &[f32],
        right: &[f32],
        wanted: &[usize],
        out: &mut [[Vec<f32>; 2]],
    ) -> Result<()>;
}

/// The triangular transition weight of length `n` (even): rises from `1/h`
/// to 1 over the first half and falls back over the second, `h = n / 2`.
pub fn transition_weight(n: usize) -> Vec<f32> {
    let half = n / 2;
    let h = half.max(1) as f32;
    let mut w = Vec::with_capacity(n);
    for i in 0..half {
        w.push((i + 1) as f32 / h);
    }
    for i in half..n {
        w.push((n - i) as f32 / h);
    }
    w
}

/// Start offsets of the chunks for a track of `len` samples.
pub fn chunk_offsets(len: usize, stride: usize) -> Vec<usize> {
    (0..len).step_by(stride.max(1)).collect()
}

/// Zeros prepended in each pass of the shift trick.
pub fn shift_amounts(shifts: usize) -> Vec<usize> {
    let passes = shifts.max(1);
    (0..passes).map(|k| k * MAX_SHIFT / passes).collect()
}

/// Number of model calls [`separate_track`] will make.
pub fn count_segments(len: usize, stride: usize, shifts: usize) -> usize {
    shift_amounts(shifts)
        .iter()
        .map(|d| chunk_offsets(len + d, stride).len())
        .sum()
}

/// The summed transition weight at every sample of a track of `len`
/// samples (what the overlap-add is divided by).
fn weight_sum(len: usize, segment: usize, stride: usize, weight: &[f32]) -> Vec<f32> {
    let mut sum = vec![0f32; len];
    for o in chunk_offsets(len, stride) {
        let chunk_len = segment.min(len - o);
        for (s, w) in sum[o..o + chunk_len].iter_mut().zip(&weight[..chunk_len]) {
            *s += *w;
        }
    }
    sum
}

/// Separates a whole track.
///
/// * `mix`: the stereo mix at 44.1 kHz.
/// * `stride`: samples between chunk starts (from the manifest, or a
///   tighter value for the higher-overlap "best" setting).
/// * `wanted`: which of the model's sources to produce, as indices into its
///   source list. The result has one [`Stereo`] per entry, same order.
/// * `shifts`: deterministic test-time shifts; 0 and 1 both mean one pass.
/// * `on_segment`: called after every model call with (calls done, calls
///   total). Return an error (for example the `cancelled` error) to stop.
pub fn separate_track(
    model: &mut dyn SegmentModel,
    mix: &Stereo,
    stride: usize,
    wanted: &[usize],
    shifts: usize,
    on_segment: &mut dyn FnMut(usize, usize) -> Result<()>,
) -> Result<Vec<Stereo>> {
    let n = model.segment_len();
    let len = mix.len();
    if n == 0 || !n.is_multiple_of(2) || stride == 0 || stride > n {
        return Err(Error::internal("separate: bad segment or stride"));
    }
    if mix.left.len() != mix.right.len() {
        return Err(Error::internal("separate: ragged mix"));
    }
    if wanted.iter().any(|s| *s >= model.num_sources()) {
        return Err(Error::internal("separate: source index out of range"));
    }
    if len == 0 {
        return Ok(wanted.iter().map(|_| Stereo::new()).collect());
    }

    // 1.1 Track normalisation: mean and unbiased std of the mono fold-down.
    let (mean, std) = if model.normalize() {
        let mut sum = 0f64;
        for (l, r) in mix.left.iter().zip(&mix.right) {
            sum += 0.5 * (f64::from(*l) + f64::from(*r));
        }
        let mean = sum / len as f64;
        let mut var = 0f64;
        for (l, r) in mix.left.iter().zip(&mix.right) {
            let d = 0.5 * (f64::from(*l) + f64::from(*r)) - mean;
            var += d * d;
        }
        let std = if len > 1 {
            (var / (len - 1) as f64).sqrt()
        } else {
            0.0
        };
        // Digital silence (or NaN): the originals would divide by zero.
        // Written as a negated comparison so NaN lands here too.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let std = if !(std >= 1e-8) { 1.0 } else { std };
        (mean as f32, std as f32)
    } else {
        (0.0f32, 1.0f32)
    };
    let inv_std = 1.0 / std;

    let weight = transition_weight(n);
    let amounts = shift_amounts(shifts);
    let passes = amounts.len();
    let total_calls = count_segments(len, stride, shifts);
    let mut calls = 0usize;

    let mut acc: Vec<Stereo> = Vec::with_capacity(wanted.len());
    for _ in wanted {
        let mut l = Vec::new();
        let mut r = Vec::new();
        l.try_reserve_exact(len)
            .map_err(|_| Error::out_of_memory())?;
        r.try_reserve_exact(len)
            .map_err(|_| Error::out_of_memory())?;
        l.resize(len, 0.0);
        r.resize(len, 0.0);
        acc.push(Stereo::from_channels(l, r));
    }
    let mut seg_l = vec![0f32; n];
    let mut seg_r = vec![0f32; n];
    let mut out: Vec<[Vec<f32>; 2]> = wanted
        .iter()
        .map(|_| [vec![0f32; n], vec![0f32; n]])
        .collect();

    for &shift in &amounts {
        // This pass sees `shift` zeros followed by the normalised mix.
        let plen = len + shift;
        let inv_w: Vec<f32> = if passes > 1 {
            weight_sum(plen, n, stride, &weight)
                .into_iter()
                .map(|w| 1.0 / w)
                .collect()
        } else {
            Vec::new()
        };
        for o in chunk_offsets(plen, stride) {
            let chunk_len = n.min(plen - o);
            // 1.2: a short (last) chunk is centred in its segment and the
            // rest of the segment is the real audio around it where the
            // track has any, zeros outside the track.
            let delta = n - chunk_len;
            let start = o as isize - (delta / 2) as isize;
            // Contiguous split: leading zeros, mix copy/normalise, trailing zeros.
            let m_first = start - shift as isize;
            let valid_start = if start < 0 {
                ((-start) as usize).min(n)
            } else {
                0
            }
            .max(if m_first < 0 {
                ((-m_first) as usize).min(n)
            } else {
                0
            });
            let max_end_plen = if start < plen as isize {
                ((plen as isize - start) as usize).min(n)
            } else {
                0
            };
            let max_end_mix = if m_first < len as isize {
                ((len as isize - m_first) as usize).min(n)
            } else {
                0
            };
            let valid_end = max_end_plen.min(max_end_mix).max(valid_start);
            if valid_start > 0 {
                seg_l[..valid_start].fill(0.0);
                seg_r[..valid_start].fill(0.0);
            }
            if valid_end > valid_start {
                let mix_off = (m_first + valid_start as isize) as usize;
                let count = valid_end - valid_start;
                if mean == 0.0 && inv_std == 1.0 {
                    seg_l[valid_start..valid_end]
                        .copy_from_slice(&mix.left[mix_off..mix_off + count]);
                    seg_r[valid_start..valid_end]
                        .copy_from_slice(&mix.right[mix_off..mix_off + count]);
                } else {
                    for ((dl, dr), (&ml, &mr)) in seg_l[valid_start..valid_end]
                        .iter_mut()
                        .zip(&mut seg_r[valid_start..valid_end])
                        .zip(
                            mix.left[mix_off..mix_off + count]
                                .iter()
                                .zip(&mix.right[mix_off..mix_off + count]),
                        )
                    {
                        *dl = (ml - mean) * inv_std;
                        *dr = (mr - mean) * inv_std;
                    }
                }
            }
            if valid_end < n {
                seg_l[valid_end..].fill(0.0);
                seg_r[valid_end..].fill(0.0);
            }

            model.run_segment(&seg_l, &seg_r, wanted, &mut out)?;
            calls += 1;

            // Overlap-add the centre `chunk_len` samples with the FIRST
            // `chunk_len` weights; the first `shift` samples of the padded track
            // are dropped.
            let trim = delta / 2;
            let i0 = shift.saturating_sub(o);
            let dst_start = o + i0 - shift;
            if i0 < chunk_len {
                for (stem, est) in acc.iter_mut().zip(&out) {
                    for (c, est_c) in est.iter().enumerate() {
                        if est_c.len() != n {
                            return Err(Error::internal("separate: model returned a wrong size"));
                        }
                        let dst = &mut stem.channel_mut(c)[dst_start..];
                        let w_slice = &weight[i0..chunk_len];
                        let e_slice = &est_c[trim + i0..trim + chunk_len];
                        if passes == 1 {
                            for ((d, &w), &s) in dst.iter_mut().zip(w_slice).zip(e_slice) {
                                *d += w * s;
                            }
                        } else {
                            let inv_slice = &inv_w[o + i0..o + chunk_len];
                            for (((d, &w), &s), &iw) in
                                dst.iter_mut().zip(w_slice).zip(e_slice).zip(inv_slice)
                            {
                                *d += w * s * iw;
                            }
                        }
                    }
                }
            }
            on_segment(calls, total_calls)?;
        }
        if passes == 1 {
            let sum_w = weight_sum(plen, n, stride, &weight);
            for stem in acc.iter_mut() {
                for (d, &sw) in stem.left.iter_mut().zip(&sum_w[shift..]) {
                    *d = *d * (std / sw) + mean;
                }
                for (d, &sw) in stem.right.iter_mut().zip(&sum_w[shift..]) {
                    *d = *d * (std / sw) + mean;
                }
            }
        }
    }

    if passes > 1 {
        // Average the passes and undo the normalisation.
        let scale = std / passes as f32;
        for stem in acc.iter_mut() {
            for v in stem.left.iter_mut().chain(stem.right.iter_mut()) {
                *v = *v * scale + mean;
            }
        }
    }
    Ok(acc)
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A fake model for tests: source `s` is the input times `gains[s]`.
    /// Records every segment's first sample position hint for assertions.
    pub struct GainModel {
        pub n: usize,
        pub gains: Vec<f32>,
        pub calls: usize,
        pub seen_nonzero: Vec<(usize, usize)>,
    }

    impl SegmentModel for GainModel {
        fn segment_len(&self) -> usize {
            self.n
        }
        fn num_sources(&self) -> usize {
            self.gains.len()
        }
        fn run_segment(
            &mut self,
            left: &[f32],
            right: &[f32],
            wanted: &[usize],
            out: &mut [[Vec<f32>; 2]],
        ) -> Result<()> {
            self.calls += 1;
            let first = left.iter().position(|v| *v != 0.0).unwrap_or(self.n);
            let last = left.iter().rposition(|v| *v != 0.0).map_or(0, |p| p + 1);
            self.seen_nonzero.push((first, last));
            for (o, s) in out.iter_mut().zip(wanted) {
                for (d, v) in o[0].iter_mut().zip(left) {
                    *d = v * self.gains[*s];
                }
                for (d, v) in o[1].iter_mut().zip(right) {
                    *d = v * self.gains[*s];
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::GainModel;
    use super::*;
    use crate::testutil::Lcg;

    fn noise(len: usize, seed: u64) -> Stereo {
        let mut rng = Lcg(seed);
        let left: Vec<f32> = (0..len).map(|_| rng.next_f32() * 0.5 + 0.01).collect();
        let right: Vec<f32> = (0..len).map(|_| rng.next_f32() * 0.5 - 0.02).collect();
        Stereo::from_channels(left, right)
    }

    #[test]
    fn transition_weight_matches_the_contract() {
        let w = transition_weight(343_980);
        assert_eq!(w.len(), 343_980);
        assert_eq!(w[0], 1.0 / 171_990.0);
        assert_eq!(w[171_989], 1.0);
        assert_eq!(w[171_990], 1.0);
        assert_eq!(w[343_979], 1.0 / 171_990.0);
        // Small case, by hand: h = 4 -> 1/4 2/4 3/4 1 1 3/4 2/4 1/4.
        assert_eq!(
            transition_weight(8),
            vec![0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25]
        );
    }

    #[test]
    fn chunk_schedule_matches_the_contract() {
        assert_eq!(chunk_offsets(10, 4), vec![0, 4, 8]);
        assert_eq!(chunk_offsets(8, 4), vec![0, 4]);
        assert_eq!(chunk_offsets(1, 4), vec![0]);
        assert!(chunk_offsets(0, 4).is_empty());
        assert_eq!(shift_amounts(0), vec![0]);
        assert_eq!(shift_amounts(1), vec![0]);
        assert_eq!(shift_amounts(2), vec![0, 11_025]);
        assert_eq!(shift_amounts(4), vec![0, 5_512, 11_025, 16_537]);
        assert_eq!(count_segments(10, 4, 1), 3);
    }

    #[test]
    fn weight_sum_is_what_overlapping_triangles_add_up_to() {
        // n = 8, stride = 6, len = 14: chunks at 0, 6, 12.
        let w = transition_weight(8);
        let s = weight_sum(14, 8, 6, &w);
        let expected = [
            0.25,
            0.5,
            0.75,
            1.0,
            1.0,
            0.75, // chunk 0 alone
            0.5 + 0.25,
            0.25 + 0.5, // chunk 0 tail + chunk 1 head
            0.75,
            1.0,
            1.0,
            0.75, // chunk 1 alone
            0.5 + 0.25,
            0.25 + 0.5, // chunk 1 tail + first two weights of the short chunk 2
        ];
        for (a, b) in s.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{s:?}");
        }
        assert!(s.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn identity_model_reproduces_the_mix_for_many_lengths() {
        // With a linear model the whole pipeline (normalise, chunk, pad,
        // overlap-add, de-normalise) must be transparent.
        let n = 64;
        for (len, stride) in [
            (1usize, 48usize),
            (2, 48),
            (63, 48),
            (64, 48),
            (65, 48),
            (200, 48),
            (511, 48),
            (300, 16),
            (300, 64),
        ] {
            let mix = noise(len, len as u64 + 3);
            let mut model = GainModel {
                n,
                gains: vec![1.0, 0.5, -2.0],
                calls: 0,
                seen_nonzero: vec![],
            };
            let mut progress = Vec::new();
            let stems = separate_track(&mut model, &mix, stride, &[0, 1, 2], 0, &mut |d, t| {
                progress.push((d, t));
                Ok(())
            })
            .unwrap();
            assert_eq!(model.calls, len.div_ceil(stride), "len {len}");
            assert_eq!(progress.len(), model.calls);
            assert_eq!(progress.last().copied(), Some((model.calls, model.calls)));
            for (stem, gain) in stems.iter().zip([1.0f32, 0.5, -2.0]) {
                assert_eq!(stem.len(), len);
                for c in 0..2 {
                    for (i, (a, b)) in stem.channel(c).iter().zip(mix.channel(c)).enumerate() {
                        // Stems are gain * normalised mix, de-normalised:
                        // gain*(x-mean) + mean. Only gain 1 gives x back.
                        if gain == 1.0 {
                            assert!(
                                (a - b).abs() < 2e-5,
                                "len {len} stride {stride} ch {c} i {i}: {a} vs {b}"
                            );
                        } else {
                            assert!(a.is_finite());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn short_last_chunk_is_centred_with_real_context() {
        // len = 100, n = 64, stride = 48: chunks at 0 (64), 48 (52), 96 (4).
        // The last chunk has 4 samples; delta = 60, so the segment starts at
        // 96 - 30 = 66 and holds real audio from 66..100 (34 samples) then
        // zeros: non-zero range [0, 34).
        let mix = noise(100, 9);
        let mut model = GainModel {
            n: 64,
            gains: vec![1.0],
            calls: 0,
            seen_nonzero: vec![],
        };
        separate_track(&mut model, &mix, 48, &[0], 0, &mut |_, _| Ok(())).unwrap();
        assert_eq!(model.seen_nonzero, vec![(0, 64), (0, 58), (0, 34)]);
        // Chunk 1: offset 48, 52 samples, delta 12 -> starts at 42, real
        // audio 42..100 = 58 samples. (Checked above.)

        // A track shorter than one segment is centred: len 10 -> delta 54,
        // start -27: zeros for 27, then 10 real samples.
        let mix = noise(10, 2);
        let mut model = GainModel {
            n: 64,
            gains: vec![1.0],
            calls: 0,
            seen_nonzero: vec![],
        };
        separate_track(&mut model, &mix, 48, &[0], 0, &mut |_, _| Ok(())).unwrap();
        assert_eq!(model.seen_nonzero, vec![(27, 37)]);
    }

    #[test]
    fn shifts_average_to_the_same_answer_for_a_linear_model() {
        let mix = noise(100_000, 4);
        let mut model = GainModel {
            n: 44_100,
            gains: vec![1.0, 0.25],
            calls: 0,
            seen_nonzero: vec![],
        };
        let one = separate_track(&mut model, &mix, 33_075, &[0], 1, &mut |_, _| Ok(())).unwrap();
        let calls_one = model.calls;
        model.calls = 0;
        let four = separate_track(&mut model, &mix, 33_075, &[0], 4, &mut |_, _| Ok(())).unwrap();
        assert_eq!(model.calls, count_segments(100_000, 33_075, 4));
        assert!(model.calls > calls_one * 3);
        for c in 0..2 {
            let snr = crate::audio::snr_db(one[0].channel(c), four[0].channel(c));
            assert!(snr > 100.0, "{snr}");
            assert!(crate::audio::snr_db(mix.channel(c), four[0].channel(c)) > 100.0);
        }
    }

    #[test]
    fn silence_and_edge_cases() {
        let mut model = GainModel {
            n: 64,
            gains: vec![1.0],
            calls: 0,
            seen_nonzero: vec![],
        };
        // Digital silence must not divide by zero.
        let stems = separate_track(
            &mut model,
            &Stereo::silence(300),
            48,
            &[0],
            0,
            &mut |_, _| Ok(()),
        )
        .unwrap();
        assert!(stems[0].left.iter().all(|v| *v == 0.0));
        // Empty track.
        let stems =
            separate_track(&mut model, &Stereo::new(), 48, &[0], 0, &mut |_, _| Ok(())).unwrap();
        assert!(stems[0].is_empty());
        // Bad arguments are errors.
        let mix = noise(100, 1);
        assert!(separate_track(&mut model, &mix, 0, &[0], 0, &mut |_, _| Ok(())).is_err());
        assert!(separate_track(&mut model, &mix, 65, &[0], 0, &mut |_, _| Ok(())).is_err());
        assert!(separate_track(&mut model, &mix, 48, &[3], 0, &mut |_, _| Ok(())).is_err());
    }

    #[test]
    fn callback_error_stops_between_segments() {
        let mix = noise(1_000, 1);
        let mut model = GainModel {
            n: 64,
            gains: vec![1.0],
            calls: 0,
            seen_nonzero: vec![],
        };
        let err = separate_track(&mut model, &mix, 48, &[0], 0, &mut |done, _| {
            if done == 3 {
                Err(Error::cancelled())
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert!(err.is_cancelled());
        assert_eq!(model.calls, 3, "no further model calls after cancellation");
    }

    #[test]
    fn unnormalized_model_receives_raw_audio() {
        struct CaptureModel {
            n: usize,
            seen_first_left: Option<f32>,
        }
        impl SegmentModel for CaptureModel {
            fn segment_len(&self) -> usize {
                self.n
            }
            fn num_sources(&self) -> usize {
                1
            }
            fn normalize(&self) -> bool {
                false
            }
            fn run_segment(
                &mut self,
                left: &[f32],
                right: &[f32],
                _wanted: &[usize],
                out: &mut [[Vec<f32>; 2]],
            ) -> Result<()> {
                if self.seen_first_left.is_none() {
                    self.seen_first_left = Some(left[0]);
                }
                out[0][0].copy_from_slice(left);
                out[0][1].copy_from_slice(right);
                Ok(())
            }
        }
        // Large DC offset + tiny amplitude: normalization would shift and scale heavily.
        let mix = Stereo::from_channels(vec![10.0, 10.01, 10.02, 10.03], vec![10.0; 4]);
        let mut model = CaptureModel {
            n: 4,
            seen_first_left: None,
        };
        let stems = separate_track(&mut model, &mix, 2, &[0], 0, &mut |_, _| Ok(())).unwrap();
        assert!((model.seen_first_left.unwrap() - 10.0).abs() < 1e-6);
        assert!((stems[0].left[2] - 10.02).abs() < 1e-5);
    }
}
