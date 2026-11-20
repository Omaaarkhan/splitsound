//! The per-kind segment transforms of `ml/CONTRACT.md` sections 2 and 3:
//! host-side STFT, one graph call, host-side inverse STFT.
//!
//! Each type owns its ONNX Runtime session and scratch buffers and
//! implements [`SegmentModel`], so [`super::chunk::separate_track`] can
//! drive any of them. A third kind (the Mel-Band RoFormer of contract
//! section 4) slots in as one more type here plus one arm in
//! [`open_segment_model`].

use ort::session::Session;
use ort::value::TensorRef;

use super::chunk::SegmentModel;
use crate::error::{Error, Result};
use crate::runtime::run_error;
use crate::stft::{
    BINS, FRAMES, HtdemucsStft, PLANE, SC_BINS, SC_FRAMES, SC_PLANE, SC_SEGMENT, SEGMENT, ScnetStft,
};

/// The separation model families the engine knows how to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeparatorKind {
    /// SCNet: one spectrogram in, one per source out.
    Scnet,
    /// HTDemucs: spectrogram and waveform in, both branches out.
    Htdemucs,
    /// Mel-Band RoFormer (vocals). Declared so manifests that list it
    /// parse; running it is not implemented yet.
    MelbandRoformer,
}

/// Wraps `session` in the transform for `kind`. `num_sources` is the length
/// of the model's source list.
pub fn open_segment_model(
    kind: SeparatorKind,
    session: Session,
    num_sources: usize,
) -> Result<Box<dyn SegmentModel + Send>> {
    match kind {
        SeparatorKind::Scnet => Ok(Box::new(ScnetModel::new(session, num_sources))),
        SeparatorKind::Htdemucs => Ok(Box::new(HtdemucsModel::new(session, num_sources))),
        SeparatorKind::MelbandRoformer => Err(Error::internal(
            "the Mel-Band RoFormer transform is not implemented in this build",
        )),
    }
}

fn check_output(shape: &[i64], data_len: usize, expected: &[usize], what: &str) -> Result<()> {
    let ok = shape.len() == expected.len()
        && shape
            .iter()
            .zip(expected)
            .all(|(a, b)| usize::try_from(*a).ok() == Some(*b))
        && data_len == expected.iter().product::<usize>();
    if ok {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "{what}: unexpected output shape {shape:?}, wanted {expected:?}"
        )))
    }
}

/// SCNet (contract section 3).
pub struct ScnetModel {
    session: Session,
    stft: ScnetStft,
    sources: usize,
    /// `[1, 4, 2049, 476]`.
    spec: Vec<f32>,
}

impl ScnetModel {
    /// Wraps a session of an SCNet graph with `sources` outputs.
    pub fn new(session: Session, sources: usize) -> Self {
        ScnetModel {
            session,
            stft: ScnetStft::new(),
            sources,
            spec: vec![0.0; 4 * SC_PLANE],
        }
    }
}

impl SegmentModel for ScnetModel {
    fn segment_len(&self) -> usize {
        SC_SEGMENT
    }

    fn num_sources(&self) -> usize {
        self.sources
    }

    fn run_segment(
        &mut self,
        left: &[f32],
        right: &[f32],
        wanted: &[usize],
        out: &mut [[Vec<f32>; 2]],
    ) -> Result<()> {
        // Complex-as-channels: L re, L im, R re, R im.
        {
            let (l, r) = self.spec.split_at_mut(2 * SC_PLANE);
            let (l_re, l_im) = l.split_at_mut(SC_PLANE);
            let (r_re, r_im) = r.split_at_mut(SC_PLANE);
            self.stft.forward(left, l_re, l_im)?;
            self.stft.forward(right, r_re, r_im)?;
        }
        let input = TensorRef::from_array_view(([1usize, 4, SC_BINS, SC_FRAMES], &self.spec[..]))
            .map_err(run_error)?;
        let outputs = self
            .session
            .run(ort::inputs!["spec" => input])
            .map_err(run_error)?;
        let value = outputs
            .get("spec_out")
            .ok_or_else(|| Error::internal("scnet: graph has no `spec_out` output"))?;
        let (shape, data) = value.try_extract_tensor::<f32>().map_err(run_error)?;
        check_output(
            shape,
            data.len(),
            &[1, self.sources, 4, SC_BINS, SC_FRAMES],
            "scnet",
        )?;
        for (dst, &s) in out.iter_mut().zip(wanted) {
            if s >= self.sources {
                return Err(Error::internal("scnet: source index out of range"));
            }
            let base = s * 4 * SC_PLANE;
            for (c, channel) in dst.iter_mut().enumerate() {
                let re = &data[base + 2 * c * SC_PLANE..base + (2 * c + 1) * SC_PLANE];
                let im = &data[base + (2 * c + 1) * SC_PLANE..base + (2 * c + 2) * SC_PLANE];
                channel.resize(SC_SEGMENT, 0.0);
                self.stft.inverse(re, im, channel)?;
            }
        }
        Ok(())
    }
}

/// HTDemucs (contract section 2).
pub struct HtdemucsModel {
    session: Session,
    stft: HtdemucsStft,
    sources: usize,
    /// `[1, 4, 2048, 336]`.
    spec: Vec<f32>,
    /// `[1, 2, 343980]`.
    mix: Vec<f32>,
}

impl HtdemucsModel {
    /// Wraps a session of an HTDemucs graph with `sources` outputs.
    pub fn new(session: Session, sources: usize) -> Self {
        HtdemucsModel {
            session,
            stft: HtdemucsStft::new(),
            sources,
            spec: vec![0.0; 4 * PLANE],
            mix: vec![0.0; 2 * SEGMENT],
        }
    }
}

impl SegmentModel for HtdemucsModel {
    fn segment_len(&self) -> usize {
        SEGMENT
    }

    fn num_sources(&self) -> usize {
        self.sources
    }

    fn run_segment(
        &mut self,
        left: &[f32],
        right: &[f32],
        wanted: &[usize],
        out: &mut [[Vec<f32>; 2]],
    ) -> Result<()> {
        if left.len() != SEGMENT || right.len() != SEGMENT {
            return Err(Error::internal("htdemucs: wrong segment size"));
        }
        {
            let (l, r) = self.spec.split_at_mut(2 * PLANE);
            let (l_re, l_im) = l.split_at_mut(PLANE);
            let (r_re, r_im) = r.split_at_mut(PLANE);
            self.stft.forward(left, l_re, l_im)?;
            self.stft.forward(right, r_re, r_im)?;
        }
        self.mix[..SEGMENT].copy_from_slice(left);
        self.mix[SEGMENT..].copy_from_slice(right);
        let mix =
            TensorRef::from_array_view(([1usize, 2, SEGMENT], &self.mix[..])).map_err(run_error)?;
        let spec = TensorRef::from_array_view(([1usize, 4, BINS, FRAMES], &self.spec[..]))
            .map_err(run_error)?;
        let outputs = self
            .session
            .run(ort::inputs!["mix" => mix, "spec" => spec])
            .map_err(run_error)?;
        let spec_out = outputs
            .get("spec_out")
            .ok_or_else(|| Error::internal("htdemucs: graph has no `spec_out` output"))?;
        let wave_out = outputs
            .get("wave_out")
            .ok_or_else(|| Error::internal("htdemucs: graph has no `wave_out` output"))?;
        let (sshape, sdata) = spec_out.try_extract_tensor::<f32>().map_err(run_error)?;
        let (wshape, wdata) = wave_out.try_extract_tensor::<f32>().map_err(run_error)?;
        check_output(
            sshape,
            sdata.len(),
            &[1, self.sources, 4, BINS, FRAMES],
            "htdemucs spec",
        )?;
        check_output(
            wshape,
            wdata.len(),
            &[1, self.sources, 2, SEGMENT],
            "htdemucs wave",
        )?;
        for (dst, &s) in out.iter_mut().zip(wanted) {
            if s >= self.sources {
                return Err(Error::internal("htdemucs: source index out of range"));
            }
            let sbase = s * 4 * PLANE;
            let wbase = s * 2 * SEGMENT;
            for (c, channel) in dst.iter_mut().enumerate() {
                // Start from the waveform branch, then add the inverse STFT
                // of the spectrogram branch.
                channel.clear();
                channel.extend_from_slice(&wdata[wbase + c * SEGMENT..wbase + (c + 1) * SEGMENT]);
                let re = &sdata[sbase + 2 * c * PLANE..sbase + (2 * c + 1) * PLANE];
                let im = &sdata[sbase + (2 * c + 1) * PLANE..sbase + (2 * c + 2) * PLANE];
                self.stft.inverse_add(re, im, channel)?;
            }
        }
        Ok(())
    }
}
