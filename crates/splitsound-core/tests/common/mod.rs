//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

pub fn fixture(rel: &str) -> PathBuf {
    let p = fixtures().join(rel);
    assert!(p.exists(), "missing fixture {}", p.display());
    p
}

/// A unique scratch directory, removed on drop.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "splitsound-it-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Amplitude of the `freq` Hz component of `x` (sampled at `rate`), by
/// projecting onto a sine and cosine. Skips `edge` samples at both ends.
pub fn tone_amplitude(x: &[f32], freq: f64, rate: u32, edge: usize) -> f64 {
    let x = &x[edge.min(x.len())..x.len().saturating_sub(edge)];
    if x.is_empty() {
        return 0.0;
    }
    let (mut s, mut c) = (0f64, 0f64);
    for (i, v) in x.iter().enumerate() {
        let ph = 2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(rate);
        s += f64::from(*v) * ph.sin();
        c += f64::from(*v) * ph.cos();
    }
    2.0 * (s * s + c * c).sqrt() / x.len() as f64
}

/// Deterministic xorshift generator for mutation tests.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Splits an Ogg file into its packets (all logical streams together; the
/// fixtures have one). Minimal: no CRC check, no chaining.
pub fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0;
    while pos + 27 <= data.len() && &data[pos..pos + 4] == b"OggS" {
        let nsegs = data[pos + 26] as usize;
        let table = &data[pos + 27..pos + 27 + nsegs];
        let mut body = pos + 27 + nsegs;
        for &lace in table {
            current.extend_from_slice(&data[body..body + lace as usize]);
            body += lace as usize;
            if lace < 255 {
                packets.push(std::mem::take(&mut current));
            }
        }
        pos = body;
    }
    packets
}
