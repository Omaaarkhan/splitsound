//! The mixing itself, as a deterministic state machine over sample buffers,
//! hosted in `splitsound-core` so release builds compile it at `-O3`.

pub use splitsound_core::playback::mixer::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn p(gain_db: f64, muted: bool, solo: bool) -> ChannelParams {
        ChannelParams {
            gain_db,
            muted,
            solo,
        }
    }

    fn gains(params: &[ChannelParams]) -> Vec<f32> {
        let mut out = vec![9.0; params.len()];
        effective_gains(params, &mut out);
        out
    }

    /// Constant stereo buffer.
    fn dc(l: f32, r: f32, frames: usize) -> Vec<f32> {
        (0..frames).flat_map(|_| [l, r]).collect()
    }

    fn render(
        m: &mut MixerCore,
        stems: &[&[f32]],
        mix: &[f32],
        frames: usize,
    ) -> (Vec<f32>, RenderOutcome) {
        let mut out = vec![f32::NAN; frames * 2];
        let o = m.render(&Sources { stems, mix }, &mut out);
        (out, o)
    }

    #[test]
    fn gain_mute_solo_matrix() {
        let unity = 1.0f32;
        // Nothing set: everything plays.
        assert_eq!(
            gains(&[p(0.0, false, false), p(0.0, false, false)]),
            vec![unity, unity]
        );
        // Mute silences only that channel.
        assert_eq!(
            gains(&[p(0.0, true, false), p(0.0, false, false)]),
            vec![0.0, unity]
        );
        // Any solo silences the others, muted or not.
        assert_eq!(
            gains(&[
                p(0.0, false, true),
                p(0.0, false, false),
                p(0.0, true, false)
            ]),
            vec![unity, 0.0, 0.0]
        );
        // Solo wins over mute on the same channel.
        assert_eq!(
            gains(&[p(0.0, true, true), p(0.0, false, false)]),
            vec![unity, 0.0]
        );
        // Two solos.
        assert_eq!(
            gains(&[
                p(0.0, false, true),
                p(-6.0, true, true),
                p(0.0, false, false)
            ]),
            vec![unity, db_to_gain(-6.0), 0.0]
        );
        // Gains are clamped; -60 dB and below is silence.
        assert!((db_to_gain(20.0) - db_to_gain(MAX_GAIN_DB)).abs() < 1e-6);
        assert!((db_to_gain(10.0) - 3.162_278).abs() < 1e-5);
        assert_eq!(db_to_gain(-60.0), 0.0);
        assert_eq!(db_to_gain(-200.0), 0.0);
        assert!(db_to_gain(-59.9) > 0.0);
        assert_eq!(clamp_db(f64::NAN), 0.0);
        assert_eq!(gains(&[p(-90.0, false, true)]), vec![0.0]);
    }

    #[test]
    fn stems_at_unity_sum_to_the_mix() {
        let a: Vec<f32> = (0..2000).map(|i| ((i as f32) * 0.01).sin() * 0.3).collect();
        let b: Vec<f32> = (0..2000)
            .map(|i| ((i as f32) * 0.037).cos() * 0.3)
            .collect();
        let mix: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let mut m = MixerCore::new(2, 1000);
        m.play();
        let (out, o) = render(&mut m, &[&a, &b], &mix, 900);
        assert_eq!(o, RenderOutcome::default());
        // After the fade-in the output is exactly the mix.
        for i in (TRANSPORT_FADE_FRAMES as usize * 2)..1800 {
            assert!((out[i] - mix[i]).abs() < 1e-5, "sample {i}");
        }
        // The fade-in starts near silence and rises.
        assert!(out[0].abs() <= mix[0].abs() * 0.01 + 1e-6);
    }

    #[test]
    fn mute_and_solo_change_the_output_smoothly() {
        let a = dc(0.5, 0.5, 20_000);
        let b = dc(0.25, 0.25, 20_000);
        let mix = dc(0.75, 0.75, 20_000);
        let mut m = MixerCore::new(2, 20_000);
        m.play();
        let (out, _) = render(&mut m, &[&a, &b], &mix, 1000);
        assert!((out[1998] - 0.75).abs() < 1e-5);
        m.set_channel(0, p(0.0, true, false));
        let (out, _) = render(&mut m, &[&a, &b], &mix, 3000);
        // No step: the first sample after muting is still close to the old level.
        assert!((out[0] - 0.75).abs() < 0.01);
        // Settles on channel b only, and never jumps by more than a smoothing step.
        assert!((out[5998] - 0.25).abs() < 1e-3);
        let max_step = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0f32, f32::max);
        assert!(max_step < 0.01, "step {max_step}");
        // Solo the muted channel: solo wins over mute, b goes silent.
        m.set_channel(0, p(0.0, true, true));
        let (out, _) = render(&mut m, &[&a, &b], &mix, 3000);
        assert!((out[5998] - 0.5).abs() < 1e-3);
        // Master gain applies to everything.
        m.set_master_db(-6.0);
        let (out, _) = render(&mut m, &[&a, &b], &mix, 3000);
        assert!((out[5998] - 0.5 * db_to_gain(-6.0)).abs() < 1e-3);
        // Meters: channel a is audible, b silent, master is the output.
        let mut meters = [0f32; 3];
        m.take_meters(&mut meters);
        render(&mut m, &[&a, &b], &mix, 1000);
        m.take_meters(&mut meters);
        assert!((meters[0] - 0.5).abs() < 0.01);
        assert!(meters[1] < 0.01);
        assert!((meters[2] - 0.5 * db_to_gain(-6.0)).abs() < 0.01);
        m.take_meters(&mut meters);
        assert_eq!(meters, [0.0; 3]);
    }

    #[test]
    fn source_switch_is_a_short_crossfade() {
        let a = dc(0.2, 0.2, 20_000);
        let mix = dc(0.8, 0.8, 20_000);
        let mut m = MixerCore::new(1, 20_000);
        m.play();
        render(&mut m, &[&a], &mix, 1000);
        m.set_original(true);
        let (out, _) = render(&mut m, &[&a], &mix, 4000);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        // Monotonic rise from the stems level to the mix level over SOURCE_FADE_FRAMES.
        assert!((left[0] - 0.2).abs() < 0.01);
        assert!(left.windows(2).all(|w| w[1] >= w[0] - 1e-6));
        let done = SOURCE_FADE_FRAMES as usize;
        assert!((left[done - 10] - 0.8).abs() > 1e-4, "not finished early");
        assert!((left[done + 1] - 0.8).abs() < 1e-5);
        // Stem meters drop to zero once the original is playing.
        let mut meters = [0f32; 2];
        m.take_meters(&mut meters);
        render(&mut m, &[&a], &mix, 100);
        m.take_meters(&mut meters);
        assert_eq!(meters[0], 0.0);
        assert!((meters[1] - 0.8).abs() < 1e-4);
        // Paused switches are immediate.
        m.pause();
        render(&mut m, &[&a], &mix, 1000);
        m.set_original(false);
        assert!(!m.is_original());
        m.play();
        let (out, _) = render(&mut m, &[&a], &mix, 1000);
        assert!((out[1998] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn without_stems_the_original_always_plays() {
        let mix = dc(0.5, -0.5, 2000);
        let mut m = MixerCore::new(0, 2000);
        m.set_original(false);
        assert!(m.is_original());
        m.play();
        let (out, _) = render(&mut m, &[], &mix, 1000);
        assert!((out[1998] - 0.5).abs() < 1e-6 && (out[1999] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn play_and_pause_fade_and_pause_reports_stop() {
        let a = dc(1.0, 1.0, 10_000);
        let mut m = MixerCore::new(1, 10_000);
        let (out, o) = render(&mut m, &[&a], &a, 10);
        assert_eq!(out, vec![0.0; 20], "silent before play");
        assert!(!o.stopped);
        m.play();
        let (out, _) = render(&mut m, &[&a], &a, 1000);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        assert!(left[0] < 0.01);
        assert!(left.windows(2).take(440).all(|w| w[1] > w[0]));
        assert!((left[500] - 1.0).abs() < 1e-6);
        let before = m.position();
        m.pause();
        let (out, o) = render(&mut m, &[&a], &a, 1000);
        assert!(o.stopped);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        assert!(left.windows(2).take(430).all(|w| w[1] < w[0]));
        assert!(
            left[TRANSPORT_FADE_FRAMES as usize..]
                .iter()
                .all(|x| *x == 0.0)
        );
        // The position advanced through the fade, then stopped.
        let after = m.position();
        assert!(after > before && after - before <= TRANSPORT_FADE_FRAMES as u64);
        render(&mut m, &[&a], &a, 1000);
        assert_eq!(m.position(), after);
        assert!(!m.is_active());
    }

    #[test]
    fn seek_while_playing_is_click_free() {
        let sine: Vec<f32> = (0..200_000)
            .flat_map(|i| {
                let s = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin() * 0.5;
                [s, s]
            })
            .collect();
        let mut m = MixerCore::new(1, 100_000);
        m.play();
        let (mut out, _) = render(&mut m, &[&sine], &sine, 2000);
        m.seek(50_011);
        assert_eq!(m.position(), 50_011);
        let (more, _) = render(&mut m, &[&sine], &sine, 2000);
        out.extend(more);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        let natural = 2.0 * std::f32::consts::PI * 1000.0 / 44_100.0 * 0.5;
        let max_step = left
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0f32, f32::max);
        assert!(
            max_step < natural * 1.1,
            "click: step {max_step} > {natural}"
        );
        assert_eq!(m.position(), 52_011);
        // Seeking while paused just moves the head.
        m.pause();
        render(&mut m, &[&sine], &sine, 1000);
        m.seek(10);
        assert_eq!(m.position(), 10);
        m.seek(u64::MAX);
        assert_eq!(m.position(), 100_000);
    }

    #[test]
    fn loop_wraps_with_a_crossfade() {
        // A ramp makes the read position visible in the output.
        let ramp: Vec<f32> = (0..20_000).flat_map(|i| [i as f32 / 20_000.0; 2]).collect();
        let mut m = MixerCore::new(1, 20_000);
        m.set_loop(Some((2_000, 6_000)));
        m.seek(1_500);
        m.play();
        let (out, o) = render(&mut m, &[&ramp], &ramp, 12_000);
        assert!(!o.reached_end);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        // Never reads past the loop end plus the crossfade tail.
        let max = left.iter().fold(0f32, |a, b| a.max(*b));
        assert!(max < (6_000 + JUMP_FADE_FRAMES) as f32 / 20_000.0);
        // After two wraps the head is back inside the loop.
        let pos = m.position();
        assert!((2_000..6_000).contains(&pos), "{pos}");
        // 1500 -> 6000 is 4500 frames, then 4000 per lap: 12000 frames end at 2000 + 3500.
        assert_eq!(pos, 5_500);
        // Wrap is gradual: no jump bigger than the crossfade allows.
        let max_step = left
            .windows(2)
            .skip(600)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0f32, f32::max);
        assert!(max_step < 4_000.0 / 20_000.0 / JUMP_FADE_FRAMES as f32 * 1.5 + 1.0 / 20_000.0);
        // Setting a loop that ends before the head jumps into it.
        m.set_loop(Some((100, 1_000)));
        assert_eq!(m.position(), 100);
        // Too-short or inverted loops are ignored.
        m.set_loop(Some((500, 510)));
        assert_eq!(m.loop_range(), None);
        m.set_loop(Some((900, 800)));
        assert_eq!(m.loop_range(), None);
        // Loop end is clamped to the track.
        m.set_loop(Some((19_000, 99_000)));
        assert_eq!(m.loop_range(), Some((19_000, 20_000)));
    }

    #[test]
    fn reaching_the_end_stops_at_the_end() {
        let a = dc(0.5, 0.5, 1_000);
        let mut m = MixerCore::new(1, 1_000);
        m.seek(400);
        m.play();
        let (out, o) = render(&mut m, &[&a], &a, 1_000);
        assert!(o.reached_end);
        assert!(m.has_ended() && !m.is_playing() && !m.is_active());
        assert_eq!(m.position(), 1_000);
        assert!(
            out[1200..].iter().all(|x| *x == 0.0),
            "silence after the end"
        );
        assert!(out[1198] != 0.0);
        // Playing again restarts from the top.
        m.play();
        assert_eq!(m.position(), 0);
        let (_, o) = render(&mut m, &[&a], &a, 10);
        assert!(!o.reached_end);
    }

    #[test]
    fn short_stems_and_bad_samples_are_silence() {
        let short = dc(0.5, 0.5, 10);
        let nan = dc(f32::NAN, f32::INFINITY, 2_000);
        let mut m = MixerCore::new(2, 2_000);
        m.play();
        let (out, _) = render(&mut m, &[&short, &nan], &nan, 2_000);
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out[1000..].iter().all(|x| *x == 0.0));
    }

    #[test]
    fn output_is_clipped_to_full_scale() {
        let loud = dc(0.9, -0.9, 2_000);
        let mut m = MixerCore::new(2, 2_000);
        m.set_channel(0, p(6.0, false, false));
        m.set_channel(1, p(6.0, false, false));
        m.set_master_db(6.0);
        m.play();
        let (out, _) = render(&mut m, &[&loud, &loud], &loud, 2_000);
        assert!(out.iter().all(|x| x.abs() <= 1.0));
        assert_eq!(out[3000], 1.0);
        assert_eq!(out[3001], -1.0);
    }
}
