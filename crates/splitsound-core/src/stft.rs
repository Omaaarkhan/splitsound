//! Forward and inverse STFT exactly as `ml/CONTRACT.md` specifies for each
//! separation model kind.
//!
//! * [`HtdemucsStft`] (contract section 2): segment `N = 343980`, periodic
//!   Hann window, `T = 336` frames of `F = 2048` bins (the Nyquist bin is
//!   dropped), inverse divided by the constant window overlap `1.5`.
//! * [`ScnetStft`] (contract section 3): segment `N = 485100`, rectangular
//!   window, `T = 476` frames of all `F = 2049` bins, inverse divided by the
//!   number of overlapping frames.
//! * [`MelbandRoformerStft`] (contract section 4): segment `N = 352800`,
//!   periodic Hann window (`NFFT = 2048`, `HOP = 441`), unnormalized `rfft`,
//!   `T = 801` frames of all `F = 1025` bins, inverse divided by the exact
//!   overlap-add squared Hann window.
//!
//! `HtdemucsStft` and `ScnetStft` use a 4096-point FFT with hop 1024 and scale
//! by `1 / sqrt(NFFT)`. Spectrogram planes are laid out `[bin][frame]`
//! (index `k * frames + j`), one plane for the real parts and one for the
//! imaginary parts, which is the layout of one channel of the graphs'
//! complex-as-channels tensors.

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::error::{Error, Result};

/// FFT size (and window length) for HTDemucs and SCNet (Mel-Band RoFormer
/// uses `MB_NFFT`).
pub const NFFT: usize = 4096;
/// Hop length for HTDemucs and SCNet (Mel-Band RoFormer uses `MB_HOP`).
pub const HOP: usize = 1024;

/// HTDemucs segment length in samples (7.8 s at 44.1 kHz).
pub const SEGMENT: usize = 343_980;
/// HTDemucs frequency bins kept per frame.
pub const BINS: usize = 2048;
/// HTDemucs frames per segment.
pub const FRAMES: usize = 336;
/// HTDemucs reflect padding on the left.
pub const PAD: usize = 1536;
/// HTDemucs reflect padding on the right.
pub const PAD_R: usize = 1620;
/// HTDemucs padded signal length: `(FRAMES + 3) * HOP`.
pub const PADDED: usize = (FRAMES + 3) * HOP;
/// Values in one HTDemucs spectrogram plane (`BINS * FRAMES`).
pub const PLANE: usize = BINS * FRAMES;

/// SCNet segment length in samples (11 s at 44.1 kHz).
pub const SC_SEGMENT: usize = 485_100;
/// SCNet frequency bins per frame (all of them).
pub const SC_BINS: usize = 2049;
/// SCNet frames per segment.
pub const SC_FRAMES: usize = 476;
/// Zeros appended to an SCNet segment so the frame count comes out even.
pub const SC_TAIL: usize = 1300;
/// SCNet reflect padding on both sides (`NFFT / 2`).
pub const SC_CENTER: usize = 2048;
/// SCNet padded signal length: `NFFT + (SC_FRAMES - 1) * HOP`.
pub const SC_PADDED: usize = NFFT + (SC_FRAMES - 1) * HOP;
/// Values in one SCNet spectrogram plane (`SC_BINS * SC_FRAMES`).
pub const SC_PLANE: usize = SC_BINS * SC_FRAMES;

/// Mel-Band RoFormer FFT and window length.
pub const MB_NFFT: usize = 2048;
/// Mel-Band RoFormer hop length.
pub const MB_HOP: usize = 441;
/// Mel-Band RoFormer segment length in samples (8 s at 44.1 kHz).
pub const MB_SEGMENT: usize = 352_800;
/// Mel-Band RoFormer frequency bins per frame (`MB_NFFT / 2 + 1`).
pub const MB_BINS: usize = MB_NFFT / 2 + 1;
/// Mel-Band RoFormer frames per segment.
pub const MB_FRAMES: usize = 801;
/// Mel-Band RoFormer reflect padding on both sides (`MB_NFFT / 2`).
pub const MB_CENTER: usize = MB_NFFT / 2;
/// Mel-Band RoFormer padded signal length: `MB_NFFT + (MB_FRAMES - 1) * MB_HOP`.
pub const MB_PADDED: usize = MB_NFFT + (MB_FRAMES - 1) * MB_HOP;
/// Values in one Mel-Band RoFormer spectrogram plane (`MB_BINS * MB_FRAMES`).
pub const MB_PLANE: usize = MB_BINS * MB_FRAMES;

fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / len as f64).cos()) as f32)
        .collect()
}

// 8-frame tiles keep simultaneous plane writes within L1 load/fill buffers;
// +9 padding breaks power-of-two L1 cache-set aliasing at 2048 bins (8192 B).
const TILE: usize = 8;
const HT_STRIDE: usize = BINS + 9;
const SC_STRIDE: usize = SC_BINS + 9;
const MB_STRIDE: usize = MB_BINS + 9;

#[inline]
#[allow(clippy::too_many_arguments)]
fn flush_tile(
    tile_re: &[f32],
    tile_im: &[f32],
    tile_stride: usize,
    bins: usize,
    frames: usize,
    j0: usize,
    t: usize,
    re: &mut [f32],
    im: &mut [f32],
) {
    for k in 0..bins {
        let base = k * frames + j0;
        let dst_re = &mut re[base..base + t];
        let dst_im = &mut im[base..base + t];
        for tj in 0..t {
            let off = tj * tile_stride + k;
            dst_re[tj] = tile_re[off];
            dst_im[tj] = tile_im[off];
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn load_tile(
    re: &[f32],
    im: &[f32],
    bins: usize,
    frames: usize,
    j0: usize,
    t: usize,
    tile_stride: usize,
    tile_re: &mut [f32],
    tile_im: &mut [f32],
) {
    for k in 0..bins {
        let base = k * frames + j0;
        let src_re = &re[base..base + t];
        let src_im = &im[base..base + t];
        for tj in 0..t {
            let off = tj * tile_stride + k;
            tile_re[off] = src_re[tj];
            tile_im[off] = src_im[tj];
        }
    }
}

/// HTDemucs transform state: FFT plans, the window and scratch buffers.
///
/// Not `Sync`; make one per worker thread.
pub struct HtdemucsStft {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    window: Arc<Vec<f32>>,
    inv_window: Arc<Vec<f32>>,
    /// Shared time-domain scratch: holds reflect-padded input during `forward`
    /// and the overlap-add accumulator during `inverse_add` (never live together).
    work: Vec<f32>,
    frame: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
    tile_re: Vec<f32>,
    tile_im: Vec<f32>,
}

impl Default for HtdemucsStft {
    fn default() -> Self {
        Self::new()
    }
}

impl HtdemucsStft {
    /// Plans the transforms.
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(NFFT);
        let inverse = planner.plan_fft_inverse(NFFT);
        // Periodic Hann, computed in f64 for accuracy.
        let window = Arc::new(hann_window(NFFT));
        // realfft's inverse is unnormalised (no 1/NFFT), so the contract's
        // `sqrt(NFFT) * irfft` becomes a plain `1/sqrt(NFFT)`; the constant
        // overlap divisor 1.5 is folded in alongside the window.
        let scale = 1.0 / ((NFFT as f32).sqrt() * 1.5);
        let inv_window = Arc::new(window.iter().map(|w| w * scale).collect());
        HtdemucsStft {
            scratch_fwd: forward.make_scratch_vec(),
            scratch_inv: inverse.make_scratch_vec(),
            spectrum: forward.make_output_vec(),
            forward,
            inverse,
            window,
            inv_window,
            work: vec![0.0; PADDED],
            frame: vec![0.0; NFFT],
            tile_re: vec![0.0; TILE * HT_STRIDE],
            tile_im: vec![0.0; TILE * HT_STRIDE],
        }
    }

    /// Creates a sibling worker reusing the planned FFTs and precomputed windows.
    pub(crate) fn clone_worker(&self) -> Self {
        HtdemucsStft {
            scratch_fwd: self.forward.make_scratch_vec(),
            scratch_inv: self.inverse.make_scratch_vec(),
            spectrum: self.forward.make_output_vec(),
            forward: Arc::clone(&self.forward),
            inverse: Arc::clone(&self.inverse),
            window: Arc::clone(&self.window),
            inv_window: Arc::clone(&self.inv_window),
            work: vec![0.0; PADDED],
            frame: vec![0.0; NFFT],
            tile_re: vec![0.0; TILE * HT_STRIDE],
            tile_im: vec![0.0; TILE * HT_STRIDE],
        }
    }

    /// Forward STFT of one channel.
    ///
    /// `x` must hold [`SEGMENT`] samples. `re` and `im` receive the real and
    /// imaginary planes, each [`PLANE`] values laid out `[bin][frame]`
    /// (index `k * FRAMES + j`), which is the layout of one channel plane of
    /// the graph's `spec` input.
    pub fn forward(&mut self, x: &[f32], re: &mut [f32], im: &mut [f32]) -> Result<()> {
        if x.len() != SEGMENT || re.len() != PLANE || im.len() != PLANE {
            return Err(Error::internal("stft: wrong buffer size"));
        }
        // Reflect padding without repeating the edge sample.
        for i in 0..PAD {
            self.work[i] = x[PAD - i];
        }
        self.work[PAD..PAD + SEGMENT].copy_from_slice(x);
        for i in 0..PAD_R {
            self.work[PAD + SEGMENT + i] = x[SEGMENT - 2 - i];
        }
        let scale = 1.0 / (NFFT as f32).sqrt();
        for j0 in (0..FRAMES).step_by(TILE) {
            let t = TILE.min(FRAMES - j0);
            for tj in 0..t {
                let j = j0 + tj;
                let src = &self.work[j * HOP..j * HOP + NFFT];
                for ((f, s), w) in self.frame.iter_mut().zip(src).zip(self.window.iter()) {
                    *f = s * w;
                }
                self.forward
                    .process_with_scratch(
                        &mut self.frame,
                        &mut self.spectrum,
                        &mut self.scratch_fwd,
                    )
                    .map_err(|e| Error::internal(format!("fft: {e}")))?;
                let tre = &mut self.tile_re[tj * HT_STRIDE..tj * HT_STRIDE + BINS];
                let tim = &mut self.tile_im[tj * HT_STRIDE..tj * HT_STRIDE + BINS];
                for k in 0..BINS {
                    let z = self.spectrum[k];
                    tre[k] = z.re * scale;
                    tim[k] = z.im * scale;
                }
            }
            flush_tile(
                &self.tile_re,
                &self.tile_im,
                HT_STRIDE,
                BINS,
                FRAMES,
                j0,
                t,
                re,
                im,
            );
        }
        Ok(())
    }

    /// Inverse STFT of one source channel, **added** to `out`.
    ///
    /// `re` and `im` are [`PLANE`]-sized planes in the `[bin][frame]` layout;
    /// `out` holds [`SEGMENT`] samples and usually already contains the
    /// graph's time-branch output, so the sum is the final stem segment.
    pub fn inverse_add(&mut self, re: &[f32], im: &[f32], out: &mut [f32]) -> Result<()> {
        if out.len() != SEGMENT || re.len() != PLANE || im.len() != PLANE {
            return Err(Error::internal("istft: wrong buffer size"));
        }
        self.work.fill(0.0);
        for j0 in (0..FRAMES).step_by(TILE) {
            let t = TILE.min(FRAMES - j0);
            load_tile(
                re,
                im,
                BINS,
                FRAMES,
                j0,
                t,
                HT_STRIDE,
                &mut self.tile_re,
                &mut self.tile_im,
            );
            for tj in 0..t {
                let j = j0 + tj;
                let tre = &self.tile_re[tj * HT_STRIDE..tj * HT_STRIDE + BINS];
                let tim = &self.tile_im[tj * HT_STRIDE..tj * HT_STRIDE + BINS];
                for k in 0..BINS {
                    self.spectrum[k] = Complex32::new(tre[k], tim[k]);
                }
                // Zero Nyquist bin. The DC and Nyquist imaginary parts must be
                // zero for a real inverse; the DC bin from the network is not
                // guaranteed to be, and realfft would reject it, so drop it the
                // way numpy's irfft silently does.
                self.spectrum[BINS] = Complex32::new(0.0, 0.0);
                self.spectrum[0].im = 0.0;
                self.inverse
                    .process_with_scratch(
                        &mut self.spectrum,
                        &mut self.frame,
                        &mut self.scratch_inv,
                    )
                    .map_err(|e| Error::internal(format!("ifft: {e}")))?;
                let dst = &mut self.work[j * HOP..j * HOP + NFFT];
                for ((d, f), iw) in dst.iter_mut().zip(&self.frame).zip(self.inv_window.iter()) {
                    *d += f * iw;
                }
            }
        }
        for (o, a) in out.iter_mut().zip(&self.work[PAD..PAD + SEGMENT]) {
            *o += *a;
        }
        Ok(())
    }
}

/// SCNet transform state.
///
/// Not `Sync`; make one per worker thread.
pub struct ScnetStft {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    work: Vec<f32>,
    frame: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
    tile_re: Vec<f32>,
    tile_im: Vec<f32>,
}

impl Default for ScnetStft {
    fn default() -> Self {
        Self::new()
    }
}

impl ScnetStft {
    /// Plans the transforms.
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(NFFT);
        let inverse = planner.plan_fft_inverse(NFFT);
        ScnetStft {
            scratch_fwd: forward.make_scratch_vec(),
            scratch_inv: inverse.make_scratch_vec(),
            spectrum: forward.make_output_vec(),
            forward,
            inverse,
            work: vec![0.0; SC_PADDED],
            frame: vec![0.0; NFFT],
            tile_re: vec![0.0; TILE * SC_STRIDE],
            tile_im: vec![0.0; TILE * SC_STRIDE],
        }
    }

    /// Creates a sibling worker reusing the planned FFTs.
    pub(crate) fn clone_worker(&self) -> Self {
        ScnetStft {
            scratch_fwd: self.forward.make_scratch_vec(),
            scratch_inv: self.inverse.make_scratch_vec(),
            spectrum: self.forward.make_output_vec(),
            forward: Arc::clone(&self.forward),
            inverse: Arc::clone(&self.inverse),
            work: vec![0.0; SC_PADDED],
            frame: vec![0.0; NFFT],
            tile_re: vec![0.0; TILE * SC_STRIDE],
            tile_im: vec![0.0; TILE * SC_STRIDE],
        }
    }

    /// Forward STFT of one channel: [`SC_SEGMENT`] samples in, two
    /// [`SC_PLANE`]-sized planes out.
    pub fn forward(&mut self, x: &[f32], re: &mut [f32], im: &mut [f32]) -> Result<()> {
        if x.len() != SC_SEGMENT || re.len() != SC_PLANE || im.len() != SC_PLANE {
            return Err(Error::internal("scnet stft: wrong buffer size"));
        }
        // xp = x ++ zeros[TAIL]; then reflect CENTER samples on both sides.
        let xp_len = SC_SEGMENT + SC_TAIL;
        let xp = |i: usize| if i < SC_SEGMENT { x[i] } else { 0.0 };
        for (i, p) in self.work[..SC_CENTER].iter_mut().enumerate() {
            *p = xp(SC_CENTER - i);
        }
        self.work[SC_CENTER..SC_CENTER + SC_SEGMENT].copy_from_slice(x);
        self.work[SC_CENTER + SC_SEGMENT..SC_CENTER + xp_len].fill(0.0);
        for i in 0..SC_CENTER {
            self.work[SC_CENTER + xp_len + i] = xp(xp_len - 2 - i);
        }
        let scale = 1.0 / (NFFT as f32).sqrt();
        for j0 in (0..SC_FRAMES).step_by(TILE) {
            let t = TILE.min(SC_FRAMES - j0);
            for tj in 0..t {
                let j = j0 + tj;
                self.frame
                    .copy_from_slice(&self.work[j * HOP..j * HOP + NFFT]);
                self.forward
                    .process_with_scratch(
                        &mut self.frame,
                        &mut self.spectrum,
                        &mut self.scratch_fwd,
                    )
                    .map_err(|e| Error::internal(format!("fft: {e}")))?;
                let tre = &mut self.tile_re[tj * SC_STRIDE..tj * SC_STRIDE + SC_BINS];
                let tim = &mut self.tile_im[tj * SC_STRIDE..tj * SC_STRIDE + SC_BINS];
                for k in 0..SC_BINS {
                    let z = self.spectrum[k];
                    tre[k] = z.re * scale;
                    tim[k] = z.im * scale;
                }
            }
            flush_tile(
                &self.tile_re,
                &self.tile_im,
                SC_STRIDE,
                SC_BINS,
                SC_FRAMES,
                j0,
                t,
                re,
                im,
            );
        }
        Ok(())
    }

    /// Inverse STFT of one source channel, written to `out`
    /// ([`SC_SEGMENT`] samples; previous contents are replaced).
    pub fn inverse(&mut self, re: &[f32], im: &[f32], out: &mut [f32]) -> Result<()> {
        if out.len() != SC_SEGMENT || re.len() != SC_PLANE || im.len() != SC_PLANE {
            return Err(Error::internal("scnet istft: wrong buffer size"));
        }
        self.work.fill(0.0);
        // realfft's inverse is unnormalised, so `sqrt(NFFT) * irfft` is a
        // plain `1 / sqrt(NFFT)`, folded directly into the final overlap pass.
        let scale = 1.0 / (NFFT as f32).sqrt();
        for j0 in (0..SC_FRAMES).step_by(TILE) {
            let t = TILE.min(SC_FRAMES - j0);
            load_tile(
                re,
                im,
                SC_BINS,
                SC_FRAMES,
                j0,
                t,
                SC_STRIDE,
                &mut self.tile_re,
                &mut self.tile_im,
            );
            for tj in 0..t {
                let j = j0 + tj;
                let tre = &self.tile_re[tj * SC_STRIDE..tj * SC_STRIDE + SC_BINS];
                let tim = &self.tile_im[tj * SC_STRIDE..tj * SC_STRIDE + SC_BINS];
                for k in 0..SC_BINS {
                    self.spectrum[k] = Complex32::new(tre[k], tim[k]);
                }
                // A real inverse ignores the imaginary parts of DC and Nyquist;
                // realfft rejects them, so zero them explicitly.
                self.spectrum[0].im = 0.0;
                self.spectrum[SC_BINS - 1].im = 0.0;
                self.inverse
                    .process_with_scratch(
                        &mut self.spectrum,
                        &mut self.frame,
                        &mut self.scratch_inv,
                    )
                    .map_err(|e| Error::internal(format!("ifft: {e}")))?;
                let dst = &mut self.work[j * HOP..j * HOP + NFFT];
                for (d, f) in dst.iter_mut().zip(&self.frame) {
                    *d += f;
                }
            }
        }
        // Divide by the number of frames covering each sample (3 in the
        // first hop, 4 afterwards) multiplied by `scale`.
        let scale3 = scale / 3.0;
        let scale4 = scale / 4.0;
        let src = &self.work[SC_CENTER..SC_CENTER + SC_SEGMENT];
        for (o, a) in out[..HOP].iter_mut().zip(&src[..HOP]) {
            *o = *a * scale3;
        }
        for (o, a) in out[HOP..].iter_mut().zip(&src[HOP..]) {
            *o = *a * scale4;
        }
        Ok(())
    }
}

/// Mel-Band RoFormer transform state.
///
/// Not `Sync`; make one per worker thread.
pub struct MelbandRoformerStft {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    window: Arc<Vec<f32>>,
    inv_window: Arc<Vec<f32>>,
    inv_env: Arc<Vec<f32>>,
    work: Vec<f32>,
    frame: Vec<f32>,
    spectrum: Vec<Complex32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
    tile_re: Vec<f32>,
    tile_im: Vec<f32>,
}

impl Default for MelbandRoformerStft {
    fn default() -> Self {
        Self::new()
    }
}

impl MelbandRoformerStft {
    /// Plans the transforms and precomputes the periodic Hann window and
    /// overlap-add squared-window envelope.
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(MB_NFFT);
        let inverse = planner.plan_fft_inverse(MB_NFFT);
        let window = Arc::new(hann_window(MB_NFFT));
        // realfft's inverse is unnormalized, while numpy/torch unnormalized irfft
        // divides by MB_NFFT.
        let scale = 1.0 / (MB_NFFT as f32);
        let inv_window = Arc::new(window.iter().map(|w| w * scale).collect());
        let mut env_pad = vec![0.0f64; MB_PADDED];
        for j in 0..MB_FRAMES {
            for (dst, w) in env_pad[j * MB_HOP..j * MB_HOP + MB_NFFT]
                .iter_mut()
                .zip(window.iter())
            {
                let wf = f64::from(*w);
                *dst += wf * wf;
            }
        }
        let inv_env = Arc::new(
            env_pad[MB_CENTER..MB_CENTER + MB_SEGMENT]
                .iter()
                .map(|v| 1.0 / (*v as f32))
                .collect(),
        );
        MelbandRoformerStft {
            scratch_fwd: forward.make_scratch_vec(),
            scratch_inv: inverse.make_scratch_vec(),
            spectrum: forward.make_output_vec(),
            forward,
            inverse,
            window,
            inv_window,
            inv_env,
            work: vec![0.0; MB_PADDED],
            frame: vec![0.0; MB_NFFT],
            tile_re: vec![0.0; TILE * MB_STRIDE],
            tile_im: vec![0.0; TILE * MB_STRIDE],
        }
    }

    /// Creates a sibling worker reusing the planned FFTs and precomputed tables.
    pub(crate) fn clone_worker(&self) -> Self {
        MelbandRoformerStft {
            scratch_fwd: self.forward.make_scratch_vec(),
            scratch_inv: self.inverse.make_scratch_vec(),
            spectrum: self.forward.make_output_vec(),
            forward: Arc::clone(&self.forward),
            inverse: Arc::clone(&self.inverse),
            window: Arc::clone(&self.window),
            inv_window: Arc::clone(&self.inv_window),
            inv_env: Arc::clone(&self.inv_env),
            work: vec![0.0; MB_PADDED],
            frame: vec![0.0; MB_NFFT],
            tile_re: vec![0.0; TILE * MB_STRIDE],
            tile_im: vec![0.0; TILE * MB_STRIDE],
        }
    }

    /// Forward STFT of one channel: [`MB_SEGMENT`] samples in, two
    /// [`MB_PLANE`]-sized planes out (unnormalized rfft).
    pub fn forward(&mut self, x: &[f32], re: &mut [f32], im: &mut [f32]) -> Result<()> {
        if x.len() != MB_SEGMENT || re.len() != MB_PLANE || im.len() != MB_PLANE {
            return Err(Error::internal("melband stft: wrong buffer size"));
        }
        for (i, p) in self.work[..MB_CENTER].iter_mut().enumerate() {
            *p = x[MB_CENTER - i];
        }
        self.work[MB_CENTER..MB_CENTER + MB_SEGMENT].copy_from_slice(x);
        for i in 0..MB_CENTER {
            self.work[MB_CENTER + MB_SEGMENT + i] = x[MB_SEGMENT - 2 - i];
        }
        for j0 in (0..MB_FRAMES).step_by(TILE) {
            let t = TILE.min(MB_FRAMES - j0);
            for tj in 0..t {
                let j = j0 + tj;
                let src = &self.work[j * MB_HOP..j * MB_HOP + MB_NFFT];
                for ((f, s), w) in self.frame.iter_mut().zip(src).zip(self.window.iter()) {
                    *f = s * w;
                }
                self.forward
                    .process_with_scratch(
                        &mut self.frame,
                        &mut self.spectrum,
                        &mut self.scratch_fwd,
                    )
                    .map_err(|e| Error::internal(format!("fft: {e}")))?;
                let tre = &mut self.tile_re[tj * MB_STRIDE..tj * MB_STRIDE + MB_BINS];
                let tim = &mut self.tile_im[tj * MB_STRIDE..tj * MB_STRIDE + MB_BINS];
                for k in 0..MB_BINS {
                    let z = self.spectrum[k];
                    tre[k] = z.re;
                    tim[k] = z.im;
                }
            }
            flush_tile(
                &self.tile_re,
                &self.tile_im,
                MB_STRIDE,
                MB_BINS,
                MB_FRAMES,
                j0,
                t,
                re,
                im,
            );
        }
        Ok(())
    }

    /// Inverse STFT of one source channel, written to `out`
    /// ([`MB_SEGMENT`] samples; previous contents are replaced).
    pub fn inverse(&mut self, re: &[f32], im: &[f32], out: &mut [f32]) -> Result<()> {
        if out.len() != MB_SEGMENT || re.len() != MB_PLANE || im.len() != MB_PLANE {
            return Err(Error::internal("melband istft: wrong buffer size"));
        }
        self.work.fill(0.0);
        for j0 in (0..MB_FRAMES).step_by(TILE) {
            let t = TILE.min(MB_FRAMES - j0);
            load_tile(
                re,
                im,
                MB_BINS,
                MB_FRAMES,
                j0,
                t,
                MB_STRIDE,
                &mut self.tile_re,
                &mut self.tile_im,
            );
            for tj in 0..t {
                let j = j0 + tj;
                let tre = &self.tile_re[tj * MB_STRIDE..tj * MB_STRIDE + MB_BINS];
                let tim = &self.tile_im[tj * MB_STRIDE..tj * MB_STRIDE + MB_BINS];
                for k in 0..MB_BINS {
                    self.spectrum[k] = Complex32::new(tre[k], tim[k]);
                }
                self.spectrum[0].im = 0.0;
                self.spectrum[MB_BINS - 1].im = 0.0;
                self.inverse
                    .process_with_scratch(
                        &mut self.spectrum,
                        &mut self.frame,
                        &mut self.scratch_inv,
                    )
                    .map_err(|e| Error::internal(format!("ifft: {e}")))?;
                let dst = &mut self.work[j * MB_HOP..j * MB_HOP + MB_NFFT];
                for ((d, f), iw) in dst.iter_mut().zip(&self.frame).zip(self.inv_window.iter()) {
                    *d += f * iw;
                }
            }
        }
        for ((o, a), ie) in out
            .iter_mut()
            .zip(&self.work[MB_CENTER..MB_CENTER + MB_SEGMENT])
            .zip(self.inv_env.iter())
        {
            *o = *a * *ie;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Lcg;

    #[test]
    fn constants_agree_with_the_contract() {
        assert_eq!(FRAMES, SEGMENT.div_ceil(HOP));
        assert_eq!(PAD, 3 * HOP / 2);
        assert_eq!(PAD_R, PAD + FRAMES * HOP - SEGMENT);
        assert_eq!(PADDED, SEGMENT + PAD + PAD_R);
        assert_eq!(PADDED, 347_136);
    }

    #[test]
    fn window_overlap_is_one_point_five() {
        let s = HtdemucsStft::new();
        for n in 0..HOP {
            let sum: f32 = (0..4).map(|r| s.window[n + r * HOP].powi(2)).sum();
            assert!((sum - 1.5).abs() < 1e-6, "{n}: {sum}");
        }
    }

    #[test]
    fn forward_matches_a_direct_dft() {
        // Compare a few bins of a few frames against the textbook sum in f64.
        let mut rng = Lcg(7);
        let x: Vec<f32> = (0..SEGMENT).map(|_| rng.next_f32() * 0.5).collect();
        let mut st = HtdemucsStft::new();
        let mut re = vec![0f32; PLANE];
        let mut im = vec![0f32; PLANE];
        st.forward(&x, &mut re, &mut im).unwrap();

        let mut y = vec![0f64; PADDED];
        for i in 0..PAD {
            y[i] = f64::from(x[PAD - i]);
        }
        for i in 0..SEGMENT {
            y[PAD + i] = f64::from(x[i]);
        }
        for i in 0..PAD_R {
            y[PAD + SEGMENT + i] = f64::from(x[SEGMENT - 2 - i]);
        }
        for &j in &[0usize, 1, 167, 334, 335] {
            for &k in &[0usize, 1, 17, 1000, 2047] {
                let (mut sr, mut si) = (0f64, 0f64);
                for n in 0..NFFT {
                    let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / NFFT as f64).cos();
                    let ph = -2.0 * std::f64::consts::PI * (k * n % NFFT) as f64 / NFFT as f64;
                    sr += w * y[j * HOP + n] * ph.cos();
                    si += w * y[j * HOP + n] * ph.sin();
                }
                sr /= 64.0;
                si /= 64.0;
                assert!(
                    (f64::from(re[k * FRAMES + j]) - sr).abs() < 2e-5,
                    "re j={j} k={k}"
                );
                assert!(
                    (f64::from(im[k * FRAMES + j]) - si).abs() < 2e-5,
                    "im j={j} k={k}"
                );
            }
        }
    }

    #[test]
    fn round_trip_reconstructs_band_limited_audio() {
        // The transform drops the Nyquist bin, so use a signal with nothing
        // up there: two sines plus noise low-passed by six cascaded
        // two-tap averages (each has a zero at Nyquist).
        let mut rng = Lcg(42);
        let mut noise: Vec<f32> = (0..SEGMENT).map(|_| rng.next_f32()).collect();
        for _ in 0..6 {
            let mut prev = 0f32;
            for v in noise.iter_mut() {
                let cur = *v;
                *v = 0.5 * (cur + prev);
                prev = cur;
            }
        }
        let x: Vec<f32> = (0..SEGMENT)
            .map(|i| {
                let t = i as f32 / 44_100.0;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 3_000.0 * t).sin()
                    + 0.3 * noise[i]
            })
            .collect();
        let mut st = HtdemucsStft::new();
        let mut re = vec![0f32; PLANE];
        let mut im = vec![0f32; PLANE];
        st.forward(&x, &mut re, &mut im).unwrap();
        let mut y = vec![0f32; SEGMENT];
        st.inverse_add(&re, &im, &mut y).unwrap();
        // Away from the segment edges the reconstruction is exact. The
        // first and last 1.5 hops are deliberately not: demucs drops the
        // two STFT frames on each side that would complete the overlap
        // there (CONTRACT.md 1.4), and the model was trained with that.
        let edge = PAD + HOP;
        let snr = crate::audio::snr_db(&x[edge..SEGMENT - edge], &y[edge..SEGMENT - edge]);
        assert!(snr > 100.0, "interior round trip SNR {snr:.1} dB");
        // At the very first sample only two of the four frames exist.
        let ratio = y[1] / x[1];
        assert!(ratio > 0.2 && ratio < 0.95, "edge gain {ratio}");
        // inverse_add really adds.
        st.inverse_add(&re, &im, &mut y).unwrap();
        let doubled: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
        assert!(
            crate::audio::snr_db(&doubled[edge..SEGMENT - edge], &y[edge..SEGMENT - edge]) > 100.0
        );
    }

    #[test]
    fn wrong_sizes_are_errors_not_panics() {
        let mut st = HtdemucsStft::new();
        let mut re = vec![0f32; PLANE];
        let mut im = vec![0f32; PLANE];
        assert!(st.forward(&[0.0; 10], &mut re, &mut im).is_err());
        let mut out = vec![0f32; 10];
        assert!(st.inverse_add(&re, &im, &mut out).is_err());
    }

    #[test]
    fn scnet_constants_agree_with_the_contract() {
        assert_eq!((SC_SEGMENT + SC_TAIL) % HOP, 0);
        assert_eq!((SC_SEGMENT + SC_TAIL) / HOP + 1, SC_FRAMES);
        assert_eq!(SC_FRAMES % 2, 0);
        assert_eq!(SC_PADDED, 490_496);
        assert_eq!(SC_PADDED, SC_SEGMENT + SC_TAIL + 2 * SC_CENTER);
    }

    #[test]
    fn scnet_forward_matches_a_direct_dft() {
        let mut rng = Lcg(11);
        let x: Vec<f32> = (0..SC_SEGMENT).map(|_| rng.next_f32() * 0.5).collect();
        let mut st = ScnetStft::new();
        let mut re = vec![0f32; SC_PLANE];
        let mut im = vec![0f32; SC_PLANE];
        st.forward(&x, &mut re, &mut im).unwrap();

        // Build the padded signal independently, as the contract words it.
        let mut xp: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
        xp.extend(std::iter::repeat_n(0.0, SC_TAIL));
        let n = xp.len();
        let mut y: Vec<f64> = (1..=SC_CENTER).rev().map(|i| xp[i]).collect();
        y.extend_from_slice(&xp);
        y.extend((0..SC_CENTER).map(|i| xp[n - 2 - i]));
        assert_eq!(y.len(), SC_PADDED);
        for &j in &[0usize, 1, 2, 237, 473, 474, 475] {
            for &k in &[0usize, 1, 33, 1024, 2047, 2048] {
                let (mut sr, mut si) = (0f64, 0f64);
                for i in 0..NFFT {
                    let ph = -2.0 * std::f64::consts::PI * (k * i % NFFT) as f64 / NFFT as f64;
                    sr += y[j * HOP + i] * ph.cos();
                    si += y[j * HOP + i] * ph.sin();
                }
                sr /= 64.0;
                si /= 64.0;
                assert!(
                    (f64::from(re[k * SC_FRAMES + j]) - sr).abs() < 5e-5,
                    "re j={j} k={k}"
                );
                assert!(
                    (f64::from(im[k * SC_FRAMES + j]) - si).abs() < 5e-5,
                    "im j={j} k={k}"
                );
            }
        }
    }

    #[test]
    fn scnet_round_trip_is_exact() {
        // Rectangular window with all bins kept: the inverse is exact over
        // the whole segment, edges included.
        let mut rng = Lcg(5);
        let x: Vec<f32> = (0..SC_SEGMENT).map(|_| rng.next_f32() * 0.8).collect();
        let mut st = ScnetStft::new();
        let mut re = vec![0f32; SC_PLANE];
        let mut im = vec![0f32; SC_PLANE];
        st.forward(&x, &mut re, &mut im).unwrap();
        let mut y = vec![1.0f32; SC_SEGMENT];
        st.inverse(&re, &im, &mut y).unwrap();
        let snr = crate::audio::snr_db(&x, &y);
        assert!(snr > 110.0, "round trip SNR {snr:.1} dB");
        assert!(
            crate::audio::snr_db(&x[..2048], &y[..2048]) > 110.0,
            "left edge"
        );
        assert!(crate::audio::snr_db(&x[SC_SEGMENT - 2048..], &y[SC_SEGMENT - 2048..]) > 110.0);
        assert!(st.forward(&x[..100], &mut re, &mut im).is_err());
        assert!(st.inverse(&re, &im, &mut y[..100]).is_err());
    }

    #[test]
    fn melband_constants_agree_with_the_contract() {
        assert_eq!(MB_SEGMENT % MB_HOP, 0);
        assert_eq!(MB_SEGMENT / MB_HOP + 1, MB_FRAMES);
        assert_eq!(MB_PADDED, MB_SEGMENT + 2 * MB_CENTER);
        assert_eq!(MB_BINS, MB_NFFT / 2 + 1);
    }

    #[test]
    fn melband_round_trip_is_exact() {
        let mut rng = Lcg(17);
        let x: Vec<f32> = (0..MB_SEGMENT).map(|_| rng.next_f32() * 0.8).collect();
        let mut st = MelbandRoformerStft::new();
        let mut re = vec![0f32; MB_PLANE];
        let mut im = vec![0f32; MB_PLANE];
        st.forward(&x, &mut re, &mut im).unwrap();
        let mut y = vec![1.0f32; MB_SEGMENT];
        st.inverse(&re, &im, &mut y).unwrap();
        let snr = crate::audio::snr_db(&x, &y);
        assert!(snr > 105.0, "round trip SNR {snr:.1} dB");
        assert!(
            crate::audio::snr_db(&x[..1024], &y[..1024]) > 105.0,
            "left edge"
        );
        assert!(
            crate::audio::snr_db(&x[MB_SEGMENT - 1024..], &y[MB_SEGMENT - 1024..]) > 105.0,
            "right edge"
        );
        assert!(st.forward(&x[..100], &mut re, &mut im).is_err());
        assert!(st.inverse(&re, &im, &mut y[..100]).is_err());
    }
}
