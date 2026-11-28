//! Real-time playback DSP (polyphase resampler and multi-stem mixer) hosted
//! in `splitsound-core` so release builds compile it at `-O3` (`codegen-units
//! = 1`) with SIMD vectorization while the desktop shell stays `-Os`.

pub mod resample {
    //! Streaming stereo sample-rate converter (windowed-sinc, polyphase table).
    //!
    //! The mixer works at 44.1 kHz; output devices usually want 48 kHz or
    //! something else. This converts on the fly with a 64-tap Kaiser-windowed
    //! sinc whose coefficients are precomputed for 512 fractional positions
    //! and linearly interpolated between them. Passband is flat to about
    //! 19 kHz and images/aliases are attenuated by roughly 80 dB, which is
    //! inaudible for monitoring. Equal rates bypass the filter entirely.

    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    const TAPS: usize = 64;
    const HALF: usize = TAPS / 2;
    const PHASES: usize = 512;
    const KAISER_BETA: f64 = 9.0;
    /// Cutoff relative to the lower of the two Nyquist frequencies.
    const CUTOFF: f64 = 0.90;
    /// Input frames pulled from the source per refill.
    const PULL_FRAMES: usize = 256;
    /// Compact planar history only once at least this many frames have been
    /// consumed, avoiding per-block `memmove` churn.
    const COMPACT_FRAMES: usize = 1024;

    fn cached_table(in_rate: u32, out_rate: u32) -> Arc<[f32]> {
        static TABLES: OnceLock<Mutex<HashMap<u64, Arc<[f32]>>>> = OnceLock::new();
        let cache = TABLES.get_or_init(|| Mutex::new(HashMap::new()));
        let key = (u64::from(in_rate) << 32) | u64::from(out_rate);
        {
            let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(existing) = guard.get(&key) {
                return Arc::clone(existing);
            }
        }
        let built: Arc<[f32]> =
            build_table(CUTOFF * (out_rate as f64 / in_rate as f64).min(1.0)).into();
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(guard.entry(key).or_insert(built))
    }

    /// Converts interleaved stereo audio from `in_rate` to `out_rate`, pulling
    /// input from a callback as it needs it.
    pub struct Resampler {
        in_rate: u32,
        out_rate: u32,
        /// Input frames advanced per output frame (`in_rate / out_rate`).
        step: f64,
        /// `(PHASES + 1) * TAPS` coefficients, rows normalised to unity gain.
        table: Arc<[f32]>,
        /// Planar stereo input history so the 64-tap dot product executes over
        /// contiguous SIMD vectors instead of interleaved gathers.
        hist_l: Vec<f32>,
        hist_r: Vec<f32>,
        /// Read position, in frames, into `hist_l`/`hist_r` (centre of the kernel).
        pos: f64,
        pull: Vec<f32>,
    }

    impl Resampler {
        pub fn new(in_rate: u32, out_rate: u32) -> Self {
            let passthrough = in_rate == out_rate;
            let table: Arc<[f32]> = if passthrough {
                Arc::from(Vec::new())
            } else {
                cached_table(in_rate, out_rate)
            };
            let cap = COMPACT_FRAMES + PULL_FRAMES * 2 + TAPS;
            let mut r = Resampler {
                in_rate,
                out_rate,
                step: in_rate as f64 / out_rate as f64,
                table,
                hist_l: Vec::with_capacity(cap),
                hist_r: Vec::with_capacity(cap),
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
            self.hist_l.clear();
            self.hist_r.clear();
            // Pre-roll so output frame 0 lines up with input frame 0.
            self.hist_l.resize(HALF - 1, 0.0);
            self.hist_r.resize(HALF - 1, 0.0);
            self.pos = (HALF - 1) as f64;
        }

        /// Fills `out` (interleaved stereo) with converted audio. `source` is
        /// called to fill its argument (interleaved stereo, always the same
        /// length) with the next input frames.
        #[inline]
        pub fn process(&mut self, out: &mut [f32], mut source: impl FnMut(&mut [f32])) {
            self.process_dyn(out, &mut source);
        }

        /// Non-generic entry point so the hot polyphase convolution loop always
        /// compiles inside `splitsound-core` at `-O3` (`codegen-units = 1`)
        /// rather than monomorphizing or inlining into `-Os` downstream crates.
        #[inline(never)]
        fn process_dyn(&mut self, out: &mut [f32], source: &mut dyn FnMut(&mut [f32])) {
            if self.is_passthrough() {
                source(out);
                return;
            }
            let frames = out.len() / 2;
            if frames == 0 {
                return;
            }
            let pos0 = self.pos;
            let step = self.step;
            let max_i = (pos0 + (frames - 1) as f64 * step) as usize;
            let need = max_i + HALF + 1;
            while self.hist_l.len() < need {
                source(&mut self.pull);
                self.hist_l.reserve(PULL_FRAMES);
                self.hist_r.reserve(PULL_FRAMES);
                for frame in self.pull.as_chunks::<2>().0 {
                    self.hist_l.push(frame[0]);
                    self.hist_r.push(frame[1]);
                }
            }

            let hl = &self.hist_l[..];
            let hr = &self.hist_r[..];
            let table = &self.table[..];
            for (f, dst) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let p = pos0 + (f as f64) * step;
                let i = p as usize;
                let frac = p - i as f64;
                let ph = frac * PHASES as f64;
                let p0 = (ph as usize).min(PHASES - 1);
                let a = (ph - p0 as f64) as f32;
                let base = i + 1 - HALF;
                let wl: &[f32; TAPS] = hl[base..base + TAPS].try_into().expect("kernel window");
                let wr: &[f32; TAPS] = hr[base..base + TAPS].try_into().expect("kernel window");
                let c0: &[f32; TAPS] = table[p0 * TAPS..(p0 + 1) * TAPS]
                    .try_into()
                    .expect("phase row");
                let c1: &[f32; TAPS] = table[(p0 + 1) * TAPS..(p0 + 2) * TAPS]
                    .try_into()
                    .expect("phase row");
                let (l, r) = convolve64(wl, wr, c0, c1, a);
                dst[0] = l;
                dst[1] = r;
            }
            self.pos = pos0 + (frames as f64) * step;

            // Compact consumed history only when past COMPACT_FRAMES to avoid
            // per-block memmove overhead.
            let keep_from = (self.pos as usize).saturating_sub(HALF - 1);
            if keep_from >= COMPACT_FRAMES {
                let remain = self.hist_l.len() - keep_from;
                self.hist_l.copy_within(keep_from.., 0);
                self.hist_l.truncate(remain);
                self.hist_r.copy_within(keep_from.., 0);
                self.hist_r.truncate(remain);
                self.pos -= keep_from as f64;
            }
        }
    }

    #[inline(always)]
    fn convolve64(
        wl: &[f32; TAPS],
        wr: &[f32; TAPS],
        c0: &[f32; TAPS],
        c1: &[f32; TAPS],
        a: f32,
    ) -> (f32, f32) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on all aarch64 targets, and all four input
        // slices have compile-time fixed length TAPS == 64 (a multiple of 4).
        unsafe {
            use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
            let mut al0 = vdupq_n_f32(0.0);
            let mut al1 = vdupq_n_f32(0.0);
            let mut ar0 = vdupq_n_f32(0.0);
            let mut ar1 = vdupq_n_f32(0.0);
            let pwl = wl.as_ptr();
            let pwr = wr.as_ptr();
            let pc0 = c0.as_ptr();
            let pc1 = c1.as_ptr();
            let mut k = 0;
            while k < TAPS {
                let sl = vld1q_f32(pwl.add(k));
                let sr = vld1q_f32(pwr.add(k));
                let k0 = vld1q_f32(pc0.add(k));
                let k1 = vld1q_f32(pc1.add(k));
                al0 = vfmaq_f32(al0, sl, k0);
                al1 = vfmaq_f32(al1, sl, k1);
                ar0 = vfmaq_f32(ar0, sr, k0);
                ar1 = vfmaq_f32(ar1, sr, k1);
                k += 4;
            }
            let l0 = vaddvq_f32(al0);
            let l1 = vaddvq_f32(al1);
            let r0 = vaddvq_f32(ar0);
            let r1 = vaddvq_f32(ar1);
            (l0 + (l1 - l0) * a, r0 + (r1 - r0) * a)
        }

        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE2 is mandatory on all x86_64 targets, and all four input
        // slices have compile-time fixed length TAPS == 64 (a multiple of 4).
        unsafe {
            use std::arch::x86_64::{_mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_setzero_ps, _mm_storeu_ps};
            let mut al0 = _mm_setzero_ps();
            let mut al1 = _mm_setzero_ps();
            let mut ar0 = _mm_setzero_ps();
            let mut ar1 = _mm_setzero_ps();
            let pwl = wl.as_ptr();
            let pwr = wr.as_ptr();
            let pc0 = c0.as_ptr();
            let pc1 = c1.as_ptr();
            let mut k = 0;
            while k < TAPS {
                let sl = _mm_loadu_ps(pwl.add(k));
                let sr = _mm_loadu_ps(pwr.add(k));
                let k0 = _mm_loadu_ps(pc0.add(k));
                let k1 = _mm_loadu_ps(pc1.add(k));
                al0 = _mm_add_ps(al0, _mm_mul_ps(sl, k0));
                al1 = _mm_add_ps(al1, _mm_mul_ps(sl, k1));
                ar0 = _mm_add_ps(ar0, _mm_mul_ps(sr, k0));
                ar1 = _mm_add_ps(ar1, _mm_mul_ps(sr, k1));
                k += 4;
            }
            let hsum = |v| {
                let mut tmp = [0.0f32; 4];
                _mm_storeu_ps(tmp.as_mut_ptr(), v);
                (tmp[0] + tmp[1]) + (tmp[2] + tmp[3])
            };
            let l0 = hsum(al0);
            let l1 = hsum(al1);
            let r0 = hsum(ar0);
            let r1 = hsum(ar1);
            (l0 + (l1 - l0) * a, r0 + (r1 - r0) * a)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut al0 = [0.0f32; 4];
            let mut al1 = [0.0f32; 4];
            let mut ar0 = [0.0f32; 4];
            let mut ar1 = [0.0f32; 4];
            for (((wl4, wr4), c04), c14) in wl
                .as_chunks::<4>()
                .0
                .iter()
                .zip(wr.as_chunks::<4>().0)
                .zip(c0.as_chunks::<4>().0)
                .zip(c1.as_chunks::<4>().0)
            {
                for j in 0..4 {
                    let k0 = c04[j];
                    let k1 = c14[j];
                    let sl = wl4[j];
                    let sr = wr4[j];
                    al0[j] += sl * k0;
                    al1[j] += sl * k1;
                    ar0[j] += sr * k0;
                    ar1[j] += sr * k1;
                }
            }
            let l0 = (al0[0] + al0[1]) + (al0[2] + al0[3]);
            let l1 = (al1[0] + al1[1]) + (al1[2] + al1[3]);
            let r0 = (ar0[0] + ar0[1]) + (ar0[2] + ar0[3]);
            let r1 = (ar1[0] + ar1[1]) + (ar1[2] + ar1[3]);
            (l0 + (l1 - l0) * a, r0 + (r1 - r0) * a)
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
}

pub mod mixer {
    //! The mixing itself, as a deterministic state machine over sample buffers.
    //!
    //! [`MixerCore::render`] turns the loaded stems (or the original mix) into
    //! interleaved stereo at 44.1 kHz. It owns everything that has to be
    //! sample-accurate: per-channel gain with smoothing, mute and solo, master
    //! gain, the stems/original crossfade, play/pause fades, click-free seeks
    //! and loop wraps, and stopping at the end.

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

        /// True when the original mix file is currently being read or faded in.
        pub fn needs_mix_source(&self) -> bool {
            self.original || self.xfade > 0.0
        }

        /// True when stem channel `index` is currently audible or fading in/out.
        pub fn needs_stem_source(&self, index: usize) -> bool {
            if self.original && self.xfade >= 1.0 {
                return false;
            }
            index < self.params.len() && (self.gain[index] > 0.0 || self.target[index] > 0.0)
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

        /// Mix of all sources at `pos` with the current gains and crossfade.
        fn mixed_frame(&self, src: &Sources, pos: u64) -> (f32, f32) {
            let n = self.params.len().min(src.stems.len());
            let (mut sl, mut sr) = (0.0f32, 0.0f32);
            if self.xfade < 1.0 {
                for (c, stem) in src.stems.iter().enumerate().take(n) {
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

        #[inline]
        fn is_steady_state(&self, n: usize, x_target: f32) -> bool {
            self.playing
                && self.transport == 1.0
                && self.jump.is_none()
                && self.master == self.master_target
                && self.xfade == x_target
                && (0..n).all(|c| self.gain[c] == self.target[c])
        }

        /// Renders `out.len() / 2` stereo frames. Always fills `out` completely
        /// (with silence once stopped).
        #[inline(never)]
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

                // Steady-state fast path: when transport, xfade, jump, master
                // and all channel gains have settled, hoist all per-sample
                // smoothing/fade checks out of the inner sample loop.
                if self.is_steady_state(n, x_target) {
                    let stop_at = match self.loop_range {
                        Some((_, b)) if self.pos < b => b.min(self.len),
                        _ => self.len,
                    };
                    let run = (frames - f).min(stop_at.saturating_sub(self.pos) as usize);
                    if run > 0 {
                        let start_pos = self.pos as usize;
                        let g = self.master;
                        let mut master_peak = self.meters[MAX_CHANNELS];
                        if self.xfade == 1.0 {
                            let mix_end = (start_pos + run) * 2;
                            if src.mix.len() >= mix_end {
                                let mslice = &src.mix[start_pos * 2..mix_end];
                                let oslice = &mut out[f * 2..(f + run) * 2];
                                for (dst, pair) in
                                    oslice.as_chunks_mut::<2>().0.iter_mut().zip(mslice.as_chunks::<2>().0)
                                {
                                    let l = (finite(pair[0]) * g).clamp(-1.0, 1.0);
                                    let r = (finite(pair[1]) * g).clamp(-1.0, 1.0);
                                    dst[0] = l;
                                    dst[1] = r;
                                    master_peak = master_peak.max(l.abs().max(r.abs()));
                                }
                            } else {
                                for i in 0..run {
                                    let (ml, mr) = frame(src.mix, self.pos + i as u64);
                                    let l = (finite(ml) * g).clamp(-1.0, 1.0);
                                    let r = (finite(mr) * g).clamp(-1.0, 1.0);
                                    out[2 * (f + i)] = l;
                                    out[2 * (f + i) + 1] = r;
                                    master_peak = master_peak.max(l.abs().max(r.abs()));
                                }
                            }
                        } else {
                            let mut active = [(0usize, 0.0f32); MAX_CHANNELS];
                            let mut active_len = 0usize;
                            let need_end = (start_pos + run) * 2;
                            let mut all_in_bounds = true;
                            for c in 0..n {
                                let cg = self.gain[c];
                                if cg > 0.0 {
                                    active[active_len] = (c, cg);
                                    active_len += 1;
                                    if src.stems[c].len() < need_end {
                                        all_in_bounds = false;
                                    }
                                }
                            }
                            if active_len == 0 {
                                out[f * 2..(f + run) * 2].fill(0.0);
                            } else if all_in_bounds {
                                let oslice = &mut out[f * 2..(f + run) * 2];
                                let sample_start = start_pos * 2;
                                // Pre-slice active stems and keep peak meters in stack locals so
                                // the single-pass mix loop avoids double indirection and extra
                                // passes over the output buffer.
                                let mut slices: [&[f32]; MAX_CHANNELS] = [&[]; MAX_CHANNELS];
                                let mut ch_peaks = [0.0f32; MAX_CHANNELS];
                                for (k_idx, &(c, _)) in active[..active_len].iter().enumerate() {
                                    slices[k_idx] = &src.stems[c][sample_start..need_end];
                                    ch_peaks[k_idx] = self.meters[c];
                                }
                                for (i, dst) in oslice.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                                    let idx = i * 2;
                                    let (mut sl, mut sr) = (0.0f32, 0.0f32);
                                    for k_idx in 0..active_len {
                                        let cg = active[k_idx].1;
                                        let buf = slices[k_idx];
                                        let l = finite(buf[idx]) * cg;
                                        let r = finite(buf[idx + 1]) * cg;
                                        sl += l;
                                        sr += r;
                                        let m = l.abs().max(r.abs());
                                        if m > ch_peaks[k_idx] {
                                            ch_peaks[k_idx] = m;
                                        }
                                    }
                                    let l = (sl * g).clamp(-1.0, 1.0);
                                    let r = (sr * g).clamp(-1.0, 1.0);
                                    dst[0] = l;
                                    dst[1] = r;
                                    master_peak = master_peak.max(l.abs().max(r.abs()));
                                }
                                for (k_idx, &(c, _)) in active[..active_len].iter().enumerate() {
                                    self.meters[c] = ch_peaks[k_idx];
                                }
                            } else {
                                for i in 0..run {
                                    let p = self.pos + i as u64;
                                    let (mut sl, mut sr) = (0.0f32, 0.0f32);
                                    for &(c, cg) in &active[..active_len] {
                                        let (l, r) = frame(src.stems[c], p);
                                        let l = finite(l) * cg;
                                        let r = finite(r) * cg;
                                        sl += l;
                                        sr += r;
                                        let m = l.abs().max(r.abs());
                                        if m > self.meters[c] {
                                            self.meters[c] = m;
                                        }
                                    }
                                    let l = (sl * g).clamp(-1.0, 1.0);
                                    let r = (sr * g).clamp(-1.0, 1.0);
                                    out[2 * (f + i)] = l;
                                    out[2 * (f + i) + 1] = r;
                                    master_peak = master_peak.max(l.abs().max(r.abs()));
                                }
                            }
                        }
                        self.meters[MAX_CHANNELS] = master_peak;
                        f += run;
                        self.pos += run as u64;
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
                        continue;
                    }
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
}
