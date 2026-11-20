//! Core audio buffer type and small numeric helpers.

/// The engine's working sample rate: everything that is separated, stored or
/// played is 44.1 kHz.
pub const SAMPLE_RATE: u32 = 44_100;

/// Sample rate the instrument tagger listens at.
pub const TAGGER_SAMPLE_RATE: u32 = 32_000;

/// Level reported for digital silence, in dB.
pub const SILENCE_DB: f32 = -120.0;

/// Planar stereo float audio: two equally long channels in `[-1, 1]`.
///
/// The sample rate is not stored; unless a function says otherwise a
/// `Stereo` is at [`SAMPLE_RATE`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stereo {
    /// Left channel.
    pub left: Vec<f32>,
    /// Right channel.
    pub right: Vec<f32>,
}

impl Stereo {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// `frames` frames of silence.
    pub fn silence(frames: usize) -> Self {
        Stereo {
            left: vec![0.0; frames],
            right: vec![0.0; frames],
        }
    }

    /// Builds a buffer from two channels.
    ///
    /// # Panics
    /// Panics if the channels differ in length (a programming error).
    pub fn from_channels(left: Vec<f32>, right: Vec<f32>) -> Self {
        assert_eq!(left.len(), right.len(), "channels must be equally long");
        Stereo { left, right }
    }

    /// Dual mono: the same signal on both channels.
    pub fn from_mono(mono: Vec<f32>) -> Self {
        Stereo {
            right: mono.clone(),
            left: mono,
        }
    }

    /// De-interleaves `LRLR...` samples. A trailing odd sample is ignored.
    pub fn from_interleaved(samples: &[f32]) -> Self {
        let frames = samples.len() / 2;
        let mut left = Vec::with_capacity(frames);
        let mut right = Vec::with_capacity(frames);
        for frame in samples.as_chunks::<2>().0 {
            left.push(frame[0]);
            right.push(frame[1]);
        }
        Stereo { left, right }
    }

    /// Interleaves to `LRLR...`.
    pub fn to_interleaved(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len() * 2);
        for (l, r) in self.left.iter().zip(&self.right) {
            out.push(*l);
            out.push(*r);
        }
        out
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.left.len()
    }

    /// True when there are no frames.
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    /// Duration in seconds at `sample_rate`.
    pub fn duration_sec(&self, sample_rate: u32) -> f64 {
        self.len() as f64 / f64::from(sample_rate.max(1))
    }

    /// Channel `index` (0 = left, anything else = right).
    pub fn channel(&self, index: usize) -> &[f32] {
        if index == 0 { &self.left } else { &self.right }
    }

    /// Mutable channel `index` (0 = left, anything else = right).
    pub fn channel_mut(&mut self, index: usize) -> &mut Vec<f32> {
        if index == 0 {
            &mut self.left
        } else {
            &mut self.right
        }
    }

    /// The mono fold-down `0.5 * (L + R)`.
    pub fn to_mono(&self) -> Vec<f32> {
        self.left
            .iter()
            .zip(&self.right)
            .map(|(l, r)| 0.5 * (l + r))
            .collect()
    }

    /// `self -= other`, sample by sample over the common length.
    pub fn subtract(&mut self, other: &Stereo) {
        for (a, b) in self.left.iter_mut().zip(&other.left) {
            *a -= *b;
        }
        for (a, b) in self.right.iter_mut().zip(&other.right) {
            *a -= *b;
        }
    }

    /// `self += other`, sample by sample over the common length.
    pub fn add(&mut self, other: &Stereo) {
        for (a, b) in self.left.iter_mut().zip(&other.left) {
            *a += *b;
        }
        for (a, b) in self.right.iter_mut().zip(&other.right) {
            *a += *b;
        }
    }
}

/// Converts a linear amplitude to dB, floored at [`SILENCE_DB`].
pub fn amplitude_to_db(amplitude: f32) -> f32 {
    if amplitude > 0.0 && amplitude.is_finite() {
        (20.0 * amplitude.log10()).max(SILENCE_DB)
    } else {
        SILENCE_DB
    }
}

/// Converts dB to a linear gain.
pub fn db_to_gain(db: f32) -> f32 {
    if db <= SILENCE_DB {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

/// Signal-to-noise ratio in dB of `test` against `reference`
/// (`10*log10(sum(ref^2) / sum((ref-test)^2))`). Returns `f64::INFINITY` for
/// identical inputs.
pub fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let n = reference.len().min(test.len());
    let mut signal = 0f64;
    let mut noise = 0f64;
    for i in 0..n {
        let r = f64::from(reference[i]);
        let d = r - f64::from(test[i]);
        signal += r * r;
        noise += d * d;
    }
    if noise == 0.0 {
        f64::INFINITY
    } else if signal == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * (signal / noise).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_round_trip() {
        let s = Stereo::from_interleaved(&[1.0, -1.0, 0.5, -0.5, 0.25]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.left, vec![1.0, 0.5]);
        assert_eq!(s.right, vec![-1.0, -0.5]);
        assert_eq!(s.to_interleaved(), vec![1.0, -1.0, 0.5, -0.5]);
        assert_eq!(s.to_mono(), vec![0.0, 0.0]);
    }

    #[test]
    fn db_helpers() {
        assert!((amplitude_to_db(1.0)).abs() < 1e-6);
        assert!((amplitude_to_db(0.5) + 6.0206).abs() < 1e-3);
        assert_eq!(amplitude_to_db(0.0), SILENCE_DB);
        assert_eq!(amplitude_to_db(f32::NAN), SILENCE_DB);
        assert!((db_to_gain(-6.0206) - 0.5).abs() < 1e-4);
        assert_eq!(db_to_gain(-200.0), 0.0);
    }

    #[test]
    fn snr_of_scaled_copy() {
        let a: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = a.iter().map(|v| v * 0.999).collect();
        let snr = snr_db(&a, &b);
        assert!((snr - 60.0).abs() < 0.1, "{snr}");
        assert!(snr_db(&a, &a).is_infinite());
    }
}
