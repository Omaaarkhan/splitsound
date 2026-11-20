//! Stereo biquad filters (RBJ cookbook). Used by the mock engine to fake
//! stems by splitting the mix into frequency bands.

#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    /// Transposed direct form II state, per channel.
    z: [[f32; 2]; 2],
}

impl Biquad {
    fn new(b: [f64; 3], a: [f64; 3]) -> Self {
        Biquad {
            b0: (b[0] / a[0]) as f32,
            b1: (b[1] / a[0]) as f32,
            b2: (b[2] / a[0]) as f32,
            a1: (a[1] / a[0]) as f32,
            a2: (a[2] / a[0]) as f32,
            z: [[0.0; 2]; 2],
        }
    }

    fn params(freq: f64, rate: f64) -> (f64, f64) {
        let w0 = 2.0 * std::f64::consts::PI * (freq / rate).clamp(1e-5, 0.49);
        let alpha = w0.sin() / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
        (w0.cos(), alpha)
    }

    pub fn lowpass(freq: f64, rate: f64) -> Self {
        let (c, alpha) = Self::params(freq, rate);
        Self::new(
            [(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0],
            [1.0 + alpha, -2.0 * c, 1.0 - alpha],
        )
    }

    pub fn highpass(freq: f64, rate: f64) -> Self {
        let (c, alpha) = Self::params(freq, rate);
        Self::new(
            [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0],
            [1.0 + alpha, -2.0 * c, 1.0 - alpha],
        )
    }

    /// Filters interleaved stereo in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        for frame in buf.as_chunks_mut::<2>().0 {
            for (ch, x) in frame.iter_mut().enumerate() {
                let z = &mut self.z[ch];
                let y = self.b0 * *x + z[0];
                z[0] = self.b1 * *x - self.a1 * y + z[1];
                z[1] = self.b2 * *x - self.a2 * y;
                *x = y;
            }
        }
    }
}

/// A chain of biquads applied in order.
#[derive(Debug, Clone, Default)]
pub struct Chain(pub Vec<Biquad>);

impl Chain {
    /// Band-pass between `lo` and `hi` Hz (either may be `None` for an open
    /// end), 4th order on each side.
    pub fn band(lo: Option<f64>, hi: Option<f64>, rate: f64) -> Self {
        let mut v = Vec::new();
        if let Some(lo) = lo {
            v.push(Biquad::highpass(lo, rate));
            v.push(Biquad::highpass(lo, rate));
        }
        if let Some(hi) = hi {
            v.push(Biquad::lowpass(hi, rate));
            v.push(Biquad::lowpass(hi, rate));
        }
        Chain(v)
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        for b in &mut self.0 {
            b.process(buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_level(chain: &mut Chain, freq: f64) -> f32 {
        let rate = 44_100.0;
        let mut buf: Vec<f32> = (0..44_100)
            .flat_map(|i| {
                let s = (2.0 * std::f64::consts::PI * freq * i as f64 / rate).sin() as f32;
                [s, s]
            })
            .collect();
        chain.process(&mut buf);
        buf[44_100..].iter().fold(0f32, |m, x| m.max(x.abs()))
    }

    #[test]
    fn bands_pass_and_stop() {
        assert!(tone_level(&mut Chain::band(None, Some(150.0), 44_100.0), 60.0) > 0.9);
        assert!(tone_level(&mut Chain::band(None, Some(150.0), 44_100.0), 3000.0) < 0.01);
        assert!(
            tone_level(
                &mut Chain::band(Some(300.0), Some(3000.0), 44_100.0),
                1000.0
            ) > 0.9
        );
        assert!(tone_level(&mut Chain::band(Some(300.0), Some(3000.0), 44_100.0), 50.0) < 0.01);
        assert!(tone_level(&mut Chain::band(Some(5000.0), None, 44_100.0), 10_000.0) > 0.9);
    }
}
