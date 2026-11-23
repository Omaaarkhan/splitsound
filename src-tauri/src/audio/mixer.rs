//! The mixing itself, as a deterministic state machine over sample buffers.
//!
//! [`MixerCore::render`] turns the loaded stems (or the original mix) into
//! interleaved stereo at 44.1 kHz. It owns everything that has to be
//! sample-accurate: per-channel gain with smoothing, mute and solo, master
//! gain, the stems/original crossfade, play/pause fades, click-free seeks
//! and loop wraps, and stopping at the end. It knows nothing about threads
//! or devices, so all of it is unit-tested here.
//!
//! Conventions:
//! - Gains are decibels clamped to [`MIN_GAIN_DB`]..=[`MAX_GAIN_DB`];
//!   `MIN_GAIN_DB` itself means silence.
//! - Any solo silences every channel that is not soloed. A soloed channel
//!   plays even if it is also muted (solo wins, as on a console).
//! - Channel meters are post-fader (channel gain, mute/solo, source
//!   crossfade, transport fade) and pre-master; the master meter is the
//!   final output.

/// Mixer sample rate (every working file is 44.1 kHz stereo).
pub const RATE: u32 = 44_100;
pub const MIN_GAIN_DB: f64 = -60.0;
pub const MAX_GAIN_DB: f64 = 10.0;
/// Play/pause fade length (10 ms).
pub const TRANSPORT_FADE_FRAMES: u32 = 441;
/// Crossfade length for seeks and loop wraps (10 ms).
pub const JUMP_FADE_FRAMES: u32 = 441;
/// Crossfade length for the stems/original switch (40 ms).
pub const SOURCE_FADE_FRAMES: u32 = 1764;
/// Time constant of the gain smoothing (10 ms).
const GAIN_TAU_SEC: f64 = 0.010;
/// Most channels a session can have (every stem id).
pub const MAX_CHANNELS: usize = 7;

/// Clamps a gain to the supported range. NaN becomes 0 dB.
pub fn clamp_db(db: f64) -> f64 {
    if db.is_nan() {
        0.0
    } else {
        db.clamp(MIN_GAIN_DB, MAX_GAIN_DB)
    }
}

/// Decibels to linear gain, with `MIN_GAIN_DB` and below as exact silence.
pub fn db_to_gain(db: f64) -> f32 {
    let db = clamp_db(db);
    if db <= MIN_GAIN_DB {
        0.0
    } else {
        10f64.powf(db / 20.0) as f32
    }
}

/// What the user set on one channel strip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelParams {
    pub gain_db: f64,
    pub muted: bool,
    pub solo: bool,
}

impl Default for ChannelParams {
    fn default() -> Self {
        ChannelParams {
            gain_db: 0.0,
            muted: false,
            solo: false,
        }
    }
}

/// Linear gain each channel should have after mute and solo.
pub fn effective_gains(params: &[ChannelParams], out: &mut [f32]) {
    let any_solo = params.iter().any(|p| p.solo);
    for (p, o) in params.iter().zip(out.iter_mut()) {
        let audible = if any_solo { p.solo } else { !p.muted };
        *o = if audible { db_to_gain(p.gain_db) } else { 0.0 };
    }
}

/// The audio the mixer reads: interleaved stereo at 44.1 kHz. Reading past
/// the end of any buffer yields silence, so stems of slightly different
/// lengths are fine.
#[derive(Clone, Copy)]
pub struct Sources<'a> {
    pub stems: &'a [&'a [f32]],
    pub mix: &'a [f32],
}

#[inline]
fn frame(buf: &[f32], pos: u64) -> (f32, f32) {
    let i = (pos as usize).saturating_mul(2);
    match buf.get(i..i + 2) {
        Some(s) => (s[0], s[1]),
        None => (0.0, 0.0),
    }
}

#[inline]
fn finite(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// A seek or loop wrap in progress: the old read head keeps going and
/// fades out while the new one fades in.
#[derive(Debug, Clone, Copy)]
struct Jump {
    from: u64,
    left: u32,
}

/// What happened during one [`MixerCore::render`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderOutcome {
    /// Playback reached the end of the track (not looping) and stopped.
    pub reached_end: bool,
    /// A pause fade finished; the mixer is now silent.
    pub stopped: bool,
}

/// Mixer state. See the module docs.
pub struct MixerCore {
    len: u64,
    params: Vec<ChannelParams>,
    target: [f32; MAX_CHANNELS],
    gain: [f32; MAX_CHANNELS],
    master_db: f64,
    master_target: f32,
    master: f32,
    original: bool,
    /// 0 = stems, 1 = original mix.
    xfade: f32,
    playing: bool,
    /// Play/pause fade, 0..1.
    transport: f32,
    pos: u64,
    loop_range: Option<(u64, u64)>,
    jump: Option<Jump>,
    ended: bool,
    /// Peak per channel, then the master, since the last `take_meters`.
    meters: [f32; MAX_CHANNELS + 1],
    smooth_k: f32,
}

impl MixerCore {
    /// A mixer for `channels` stems (0 before a split) over `len_frames` frames.
    pub fn new(channels: usize, len_frames: u64) -> Self {
        let channels = channels.min(MAX_CHANNELS);
        let mut m = MixerCore {
            len: len_frames,
            params: vec![ChannelParams::default(); channels],
            target: [0.0; MAX_CHANNELS],
            gain: [0.0; MAX_CHANNELS],
            master_db: 0.0,
            master_target: 1.0,
            master: 1.0,
            original: channels == 0,
            xfade: if channels == 0 { 1.0 } else { 0.0 },
            playing: false,
            transport: 0.0,
            pos: 0,
            loop_range: None,
            jump: None,
            ended: false,
            meters: [0.0; MAX_CHANNELS + 1],
            smooth_k: (1.0 - (-1.0 / (GAIN_TAU_SEC * RATE as f64)).exp()) as f32,
        };
        m.recompute_targets();
        m.gain = m.target;
        m
    }

    pub fn channels(&self) -> usize {
        self.params.len()
    }

    pub fn len_frames(&self) -> u64 {
        self.len
    }

    pub fn position(&self) -> u64 {
        self.pos
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn has_ended(&self) -> bool {
        self.ended
    }

    /// True while there is something to render: playing, or fading out.
    pub fn is_active(&self) -> bool {
        self.playing || self.transport > 0.0
    }

    fn audible(&self) -> bool {
        self.transport > 0.0
    }

    fn recompute_targets(&mut self) {
        let n = self.params.len();
        effective_gains(&self.params, &mut self.target[..n]);
    }

    /// Sets one channel strip. Out-of-range indices are ignored.
    pub fn set_channel(&mut self, index: usize, params: ChannelParams) {
        if let Some(p) = self.params.get_mut(index) {
            *p = ChannelParams {
                gain_db: clamp_db(params.gain_db),
                ..params
            };
            self.recompute_targets();
            if !self.audible() {
                self.gain = self.target;
            }
        }
    }

    pub fn channel(&self, index: usize) -> Option<ChannelParams> {
        self.params.get(index).copied()
    }

    pub fn set_master_db(&mut self, db: f64) {
        self.master_db = clamp_db(db);
        self.master_target = db_to_gain(self.master_db);
        if !self.audible() {
            self.master = self.master_target;
        }
    }

    pub fn master_db(&self) -> f64 {
        self.master_db
    }

    /// Plays the original mix (`true`) or the stems. Without stems the
    /// original always plays.
    pub fn set_original(&mut self, original: bool) {
        self.original = original || self.params.is_empty();
        if !self.audible() {
            self.xfade = if self.original { 1.0 } else { 0.0 };
        }
    }

    pub fn is_original(&self) -> bool {
        self.original
    }

    pub fn xfade(&self) -> f32 {
        self.xfade
    }

    pub fn transport(&self) -> f32 {
        self.transport
    }

    /// Restores live playback state when hot-swapping audio without an
    /// audible dip; call after seeking and before `set_original` so a source
    /// change crossfades smoothly from `xfade`. `ended` carries over a track
    /// that already rendered to its end, so the next `play` restarts it.
    pub fn restore_live(&mut self, transport: f32, xfade: f32, playing: bool, ended: bool) {
        self.transport = transport.clamp(0.0, 1.0);
        self.xfade = xfade.clamp(0.0, 1.0);
        self.playing = playing;
        self.ended = ended;
    }

    pub fn play(&mut self) {
        if self.ended || self.pos >= self.len {
            self.pos = self.loop_range.map_or(0, |(a, _)| a);
        }
        if let Some((a, b)) = self.loop_range
            && self.pos >= b
        {
            self.pos = a;
        }
        self.ended = false;
        self.playing = true;
    }

    /// Starts the pause fade. [`RenderOutcome::stopped`] reports when it ends.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stops at once, without a fade (the output is already gone, e.g. the
    /// device disappeared).
    pub fn stop_now(&mut self) {
        self.playing = false;
        self.transport = 0.0;
        self.jump = None;
    }

    /// Moves the read head. While audible this crossfades from the old
    /// position, so it never clicks.
    pub fn seek(&mut self, frame: u64) {
        let frame = frame.min(self.len);
        if self.audible() && frame != self.pos {
            self.jump = Some(Jump {
                from: self.pos,
                left: JUMP_FADE_FRAMES,
            });
        }
        self.pos = frame;
        self.ended = false;
    }

    /// Sets or clears the loop. `end` is clamped to the track; a range
    /// shorter than 10 ms is treated as no loop. If the play head is past
    /// the new loop's end it jumps (click-free) to the loop start.
    pub fn set_loop(&mut self, range: Option<(u64, u64)>) {
        self.loop_range = range.and_then(|(a, b)| {
            let b = b.min(self.len);
            (b > a && b - a >= RATE as u64 / 100).then_some((a, b))
        });
        if let Some((a, b)) = self.loop_range
            && self.pos >= b
        {
            self.seek(a);
        }
    }

    pub fn loop_range(&self) -> Option<(u64, u64)> {
        self.loop_range
    }

    /// Copies the peak meters (channels, then master) gathered since the
    /// last call into `out` and resets them.
    pub fn take_meters(&mut self, out: &mut [f32]) {
        let n = self.params.len();
        for (i, o) in out.iter_mut().enumerate().take(n + 1) {
            let src = if i == n { MAX_CHANNELS } else { i };
            *o = self.meters[src].min(1.0);
            self.meters[src] = 0.0;
        }
    }

    /// Stereo frame of the full mix (stems weighted, crossfaded with the
    /// original) at `pos`, before master and transport gain. Used for the
    /// fading-out head of a jump.
    fn mixed_frame(&self, src: &Sources, pos: u64) -> (f32, f32) {
        let (mut sl, mut sr) = (0.0, 0.0);
        if self.xfade < 1.0 {
            for (c, stem) in src.stems.iter().enumerate().take(self.params.len()) {
                let g = self.gain[c];
                if g == 0.0 {
                    continue;
                }
                let (l, r) = frame(stem, pos);
                sl += finite(l) * g;
                sr += finite(r) * g;
            }
        }
        let (ml, mr) = if self.xfade > 0.0 {
            frame(src.mix, pos)
        } else {
            (0.0, 0.0)
        };
        let x = self.xfade;
        (
            sl * (1.0 - x) + finite(ml) * x,
            sr * (1.0 - x) + finite(mr) * x,
        )
    }

    /// Renders `out.len() / 2` stereo frames. Always fills `out` completely
    /// (with silence once stopped).
    pub fn render(&mut self, src: &Sources, out: &mut [f32]) -> RenderOutcome {
        let frames = out.len() / 2;
        let n = self.params.len().min(src.stems.len());
        let t_step = 1.0 / TRANSPORT_FADE_FRAMES as f32;
        let x_step = 1.0 / SOURCE_FADE_FRAMES as f32;
        let x_target = if self.original { 1.0 } else { 0.0 };
        let k = self.smooth_k;
        let mut outcome = RenderOutcome::default();
        let mut f = 0;
        while f < frames {
            if !self.is_active() {
                break;
            }
            // Transport fade.
            if self.playing {
                self.transport = (self.transport + t_step).min(1.0);
            } else {
                self.transport -= t_step;
                if self.transport <= 1e-6 {
                    self.transport = 0.0;
                    self.jump = None;
                    outcome.stopped = true;
                    break;
                }
            }
            // Parameter smoothing.
            for c in 0..n {
                let d = self.target[c] - self.gain[c];
                self.gain[c] = if d.abs() < 1e-6 {
                    self.target[c]
                } else {
                    self.gain[c] + d * k
                };
            }
            let d = self.master_target - self.master;
            self.master = if d.abs() < 1e-6 {
                self.master_target
            } else {
                self.master + d * k
            };
            if self.xfade < x_target {
                self.xfade = (self.xfade + x_step).min(1.0);
            } else if self.xfade > x_target {
                self.xfade = (self.xfade - x_step).max(0.0);
            }

            // Main read head, with channel meters.
            let x = self.xfade;
            let tr = self.transport;
            let (mut sl, mut sr) = (0.0f32, 0.0f32);
            if x < 1.0 {
                let meter_scale = (1.0 - x) * tr;
                for (c, stem) in src.stems.iter().enumerate().take(n) {
                    let g = self.gain[c];
                    if g == 0.0 {
                        continue;
                    }
                    let (l, r) = frame(stem, self.pos);
                    let (l, r) = (finite(l) * g, finite(r) * g);
                    sl += l;
                    sr += r;
                    let m = l.abs().max(r.abs()) * meter_scale;
                    if m > self.meters[c] {
                        self.meters[c] = m;
                    }
                }
            }
            let (ml, mr) = if x > 0.0 {
                frame(src.mix, self.pos)
            } else {
                (0.0, 0.0)
            };
            let mut l = sl * (1.0 - x) + finite(ml) * x;
            let mut r = sr * (1.0 - x) + finite(mr) * x;

            // Fading-out head of a seek or loop wrap.
            if let Some(mut j) = self.jump {
                let w = j.left as f32 / JUMP_FADE_FRAMES as f32;
                let (ol, or) = self.mixed_frame(src, j.from);
                l = l * (1.0 - w) + ol * w;
                r = r * (1.0 - w) + or * w;
                j.from += 1;
                j.left -= 1;
                self.jump = (j.left > 0).then_some(j);
            }

            let g = self.master * tr;
            let l = (l * g).clamp(-1.0, 1.0);
            let r = (r * g).clamp(-1.0, 1.0);
            out[2 * f] = l;
            out[2 * f + 1] = r;
            let m = l.abs().max(r.abs());
            if m > self.meters[MAX_CHANNELS] {
                self.meters[MAX_CHANNELS] = m;
            }
            f += 1;
            self.pos += 1;

            if let Some((a, b)) = self.loop_range
                && self.pos >= b
            {
                self.jump = Some(Jump {
                    from: self.pos,
                    left: JUMP_FADE_FRAMES,
                });
                self.pos = a;
                continue;
            }
            if self.pos >= self.len {
                self.pos = self.len;
                self.ended = true;
                self.playing = false;
                self.transport = 0.0;
                self.jump = None;
                outcome.reached_end = true;
                break;
            }
        }
        out[2 * f..].fill(0.0);
        outcome
    }
}

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
