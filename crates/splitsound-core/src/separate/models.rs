//! The per-kind segment transforms of `ml/CONTRACT.md` sections 2, 3 and 4:
//! host-side STFT, one graph call, host-side inverse STFT.
//!
//! Each type owns its ONNX Runtime session and scratch buffers and
//! implements [`SegmentModel`], so [`super::chunk::separate_track`] can
//! drive any of them.

use ort::session::Session;
use ort::value::TensorRef;

use super::chunk::SegmentModel;
use crate::error::{Error, Result};
use crate::runtime::run_error;
use crate::stft::{
    BINS, FRAMES, HtdemucsStft, MB_BINS, MB_FRAMES, MB_PLANE, MB_SEGMENT, MelbandRoformerStft,
    PLANE, SC_BINS, SC_FRAMES, SC_PLANE, SC_SEGMENT, SEGMENT, ScnetStft,
};

/// The separation model families the engine knows how to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeparatorKind {
    /// SCNet: one spectrogram in, one per source out.
    Scnet,
    /// HTDemucs: spectrogram and waveform in, both branches out.
    Htdemucs,
    /// Mel-Band RoFormer (vocals).
    MelbandRoformer,
}

/// Wraps `session` in the transform for `kind`. `num_sources` is the length
/// of the model's source list.
pub fn open_segment_model(
    kind: SeparatorKind,
    session: Session,
    num_sources: usize,
) -> Result<Box<dyn SegmentModel + Send>> {
    open_segment_model_with_threads(kind, session, num_sources, worker_count())
}

/// [`open_segment_model`] with an explicit upper bound on parallel STFT
/// workers (typically `Engine::threads()`), additionally capped by the
/// model's stereo channel count (`num_sources * 2`).
pub fn open_segment_model_with_threads(
    kind: SeparatorKind,
    session: Session,
    num_sources: usize,
    max_threads: usize,
) -> Result<Box<dyn SegmentModel + Send>> {
    match kind {
        SeparatorKind::Scnet => Ok(Box::new(ScnetModel::with_threads(
            session,
            num_sources,
            max_threads,
        ))),
        SeparatorKind::Htdemucs => Ok(Box::new(HtdemucsModel::with_threads(
            session,
            num_sources,
            max_threads,
        ))),
        SeparatorKind::MelbandRoformer => Ok(Box::new(MelbandRoformerModel::with_threads(
            session,
            num_sources,
            max_threads,
        ))),
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

fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, 12)
}

/// Number of auxiliary STFT workers (excluding the primary instance) needed
/// for a model with `sources` stereo outputs and at most `max_threads`
/// threads. Forward STFT uses at most 2 threads (L/R) and inverse STFT uses at
/// most `sources * 2` threads.
pub(crate) fn extra_worker_count(sources: usize, max_threads: usize) -> usize {
    let max_tasks = sources.saturating_mul(2).max(2);
    let total = max_threads.clamp(1, 12).min(max_tasks);
    total.saturating_sub(1)
}

fn make_workers<S>(first: &S, sources: usize, max_threads: usize, clone: impl Fn(&S) -> S) -> Vec<S> {
    let n = extra_worker_count(sources, max_threads);
    (0..n).map(|_| clone(first)).collect()
}

fn par_forward<S: Send>(
    primary: &mut S,
    workers: &mut [S],
    f_l: impl FnOnce(&mut S) -> Result<()> + Send,
    f_r: impl FnOnce(&mut S) -> Result<()> + Send,
) -> Result<()> {
    if let Some((w, _)) = workers.split_first_mut() {
        let (rl, rr) = std::thread::scope(|scope| {
            let h = scope.spawn(move || f_r(w));
            let rl = f_l(primary);
            let rr = h
                .join()
                .unwrap_or_else(|_| Err(Error::internal("stft worker panicked")));
            (rl, rr)
        });
        rl?;
        rr
    } else {
        f_l(primary)?;
        f_r(primary)
    }
}

struct InvTask<'a> {
    dst: &'a mut Vec<f32>,
    re: &'a [f32],
    im: &'a [f32],
    wave: Option<&'a [f32]>,
}

fn par_inverse<S: Send>(
    primary: &mut S,
    workers: &mut [S],
    tasks: &mut [InvTask<'_>],
    op: impl Fn(&mut S, &mut InvTask<'_>) -> Result<()> + Sync,
) -> Result<()> {
    if workers.is_empty() || tasks.len() <= 1 {
        for task in tasks.iter_mut() {
            op(primary, task)?;
        }
        return Ok(());
    }
    let active = (1 + workers.len()).min(tasks.len());
    let per = tasks.len().div_ceil(active);
    let mut stfts: Vec<&mut S> = std::iter::once(primary)
        .chain(workers.iter_mut())
        .take(active)
        .collect();
    std::thread::scope(|scope| {
        let mut chunks: Vec<&mut [InvTask<'_>]> = tasks.chunks_mut(per).collect();
        let first_chunk = chunks.remove(0);
        let first_stft = stfts.remove(0);
        let mut handles = Vec::with_capacity(chunks.len());
        let op_ref = &op;
        for (stft, chunk) in stfts.into_iter().zip(chunks) {
            handles.push(scope.spawn(move || -> Result<()> {
                for task in chunk.iter_mut() {
                    op_ref(stft, task)?;
                }
                Ok(())
            }));
        }
        for task in first_chunk.iter_mut() {
            op_ref(first_stft, task)?;
        }
        for h in handles {
            h.join()
                .unwrap_or_else(|_| Err(Error::internal("stft worker panicked")))?;
        }
        Ok(())
    })
}

/// SCNet (contract section 3).
pub struct ScnetModel {
    session: Session,
    stft: ScnetStft,
    workers: Vec<ScnetStft>,
    sources: usize,
    /// `[1, 4, 2049, 476]`.
    spec: Vec<f32>,
}

impl ScnetModel {
    /// Wraps a session of an SCNet graph with `sources` outputs.
    pub fn new(session: Session, sources: usize) -> Self {
        Self::with_threads(session, sources, worker_count())
    }

    /// [`ScnetModel::new`] capped to at most `max_threads` STFT workers.
    pub fn with_threads(session: Session, sources: usize, max_threads: usize) -> Self {
        let stft = ScnetStft::new();
        let workers = make_workers(&stft, sources, max_threads, ScnetStft::clone_worker);
        ScnetModel {
            session,
            stft,
            workers,
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
            par_forward(
                &mut self.stft,
                &mut self.workers,
                |s| s.forward(left, l_re, l_im),
                |s| s.forward(right, r_re, r_im),
            )?;
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
        let mut tasks = Vec::with_capacity(wanted.len() * 2);
        for (dst, &s) in out.iter_mut().zip(wanted) {
            if s >= self.sources {
                return Err(Error::internal("scnet: source index out of range"));
            }
            let base = s * 4 * SC_PLANE;
            let [dl, dr] = dst;
            tasks.push(InvTask {
                dst: dl,
                re: &data[base..base + SC_PLANE],
                im: &data[base + SC_PLANE..base + 2 * SC_PLANE],
                wave: None,
            });
            tasks.push(InvTask {
                dst: dr,
                re: &data[base + 2 * SC_PLANE..base + 3 * SC_PLANE],
                im: &data[base + 3 * SC_PLANE..base + 4 * SC_PLANE],
                wave: None,
            });
        }
        par_inverse(&mut self.stft, &mut self.workers, &mut tasks, |s, t| {
            t.dst.resize(SC_SEGMENT, 0.0);
            s.inverse(t.re, t.im, t.dst)
        })
    }
}

/// HTDemucs (contract section 2).
pub struct HtdemucsModel {
    session: Session,
    stft: HtdemucsStft,
    workers: Vec<HtdemucsStft>,
    sources: usize,
    /// `[1, 4, 2048, 336]`.
    spec: Vec<f32>,
    /// `[1, 2, 343980]`.
    mix: Vec<f32>,
}

impl HtdemucsModel {
    /// Wraps a session of an HTDemucs graph with `sources` outputs.
    pub fn new(session: Session, sources: usize) -> Self {
        Self::with_threads(session, sources, worker_count())
    }

    /// [`HtdemucsModel::new`] capped to at most `max_threads` STFT workers.
    pub fn with_threads(session: Session, sources: usize, max_threads: usize) -> Self {
        let stft = HtdemucsStft::new();
        let workers = make_workers(&stft, sources, max_threads, HtdemucsStft::clone_worker);
        HtdemucsModel {
            session,
            stft,
            workers,
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
            par_forward(
                &mut self.stft,
                &mut self.workers,
                |s| s.forward(left, l_re, l_im),
                |s| s.forward(right, r_re, r_im),
            )?;
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
        let mut tasks = Vec::with_capacity(wanted.len() * 2);
        for (dst, &s) in out.iter_mut().zip(wanted) {
            if s >= self.sources {
                return Err(Error::internal("htdemucs: source index out of range"));
            }
            let sbase = s * 4 * PLANE;
            let wbase = s * 2 * SEGMENT;
            let [dl, dr] = dst;
            tasks.push(InvTask {
                dst: dl,
                re: &sdata[sbase..sbase + PLANE],
                im: &sdata[sbase + PLANE..sbase + 2 * PLANE],
                wave: Some(&wdata[wbase..wbase + SEGMENT]),
            });
            tasks.push(InvTask {
                dst: dr,
                re: &sdata[sbase + 2 * PLANE..sbase + 3 * PLANE],
                im: &sdata[sbase + 3 * PLANE..sbase + 4 * PLANE],
                wave: Some(&wdata[wbase + SEGMENT..wbase + 2 * SEGMENT]),
            });
        }
        par_inverse(&mut self.stft, &mut self.workers, &mut tasks, |s, t| {
            t.dst.clear();
            t.dst.extend_from_slice(t.wave.unwrap_or(&[]));
            s.inverse_add(t.re, t.im, t.dst)
        })
    }
}

/// Mel-Band RoFormer (contract section 4).
pub struct MelbandRoformerModel {
    session: Session,
    stft: MelbandRoformerStft,
    workers: Vec<MelbandRoformerStft>,
    sources: usize,
    /// `[1, 4, 1025, 801]`.
    spec: Vec<f32>,
}

impl MelbandRoformerModel {
    /// Wraps a session of a Mel-Band RoFormer graph with `sources` outputs.
    pub fn new(session: Session, sources: usize) -> Self {
        Self::with_threads(session, sources, worker_count())
    }

    /// [`MelbandRoformerModel::new`] capped to at most `max_threads` STFT workers.
    pub fn with_threads(session: Session, sources: usize, max_threads: usize) -> Self {
        let stft = MelbandRoformerStft::new();
        let workers = make_workers(
            &stft,
            sources,
            max_threads,
            MelbandRoformerStft::clone_worker,
        );
        MelbandRoformerModel {
            session,
            stft,
            workers,
            sources,
            spec: vec![0.0; 4 * MB_PLANE],
        }
    }
}

impl SegmentModel for MelbandRoformerModel {
    fn segment_len(&self) -> usize {
        MB_SEGMENT
    }

    fn num_sources(&self) -> usize {
        self.sources
    }

    fn normalize(&self) -> bool {
        false
    }

    fn run_segment(
        &mut self,
        left: &[f32],
        right: &[f32],
        wanted: &[usize],
        out: &mut [[Vec<f32>; 2]],
    ) -> Result<()> {
        {
            let (l, r) = self.spec.split_at_mut(2 * MB_PLANE);
            let (l_re, l_im) = l.split_at_mut(MB_PLANE);
            let (r_re, r_im) = r.split_at_mut(MB_PLANE);
            par_forward(
                &mut self.stft,
                &mut self.workers,
                |s| s.forward(left, l_re, l_im),
                |s| s.forward(right, r_re, r_im),
            )?;
        }
        let input = TensorRef::from_array_view(([1usize, 4, MB_BINS, MB_FRAMES], &self.spec[..]))
            .map_err(run_error)?;
        let outputs = self
            .session
            .run(ort::inputs!["spec" => input])
            .map_err(run_error)?;
        let value = outputs
            .get("spec_out")
            .ok_or_else(|| Error::internal("melband_roformer: graph has no `spec_out` output"))?;
        let (shape, data) = value.try_extract_tensor::<f32>().map_err(run_error)?;
        check_output(
            shape,
            data.len(),
            &[1, self.sources, 4, MB_BINS, MB_FRAMES],
            "melband_roformer",
        )?;
        let mut tasks = Vec::with_capacity(wanted.len() * 2);
        for (dst, &s) in out.iter_mut().zip(wanted) {
            if s >= self.sources {
                return Err(Error::internal(
                    "melband_roformer: source index out of range",
                ));
            }
            let base = s * 4 * MB_PLANE;
            let [dl, dr] = dst;
            tasks.push(InvTask {
                dst: dl,
                re: &data[base..base + MB_PLANE],
                im: &data[base + MB_PLANE..base + 2 * MB_PLANE],
                wave: None,
            });
            tasks.push(InvTask {
                dst: dr,
                re: &data[base + 2 * MB_PLANE..base + 3 * MB_PLANE],
                im: &data[base + 3 * MB_PLANE..base + 4 * MB_PLANE],
                wave: None,
            });
        }
        par_inverse(&mut self.stft, &mut self.workers, &mut tasks, |s, t| {
            t.dst.resize(MB_SEGMENT, 0.0);
            s.inverse(t.re, t.im, t.dst)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_pool_is_bounded_by_sources_and_threads() {
        // Single-source stereo model (RoFormer): never needs more than 1 extra worker.
        assert_eq!(extra_worker_count(1, 12), 1);
        assert_eq!(extra_worker_count(1, 1), 0);
        // 4-source model (SCNet): bounded by min(8, threads) - 1.
        assert_eq!(extra_worker_count(4, 4), 3);
        assert_eq!(extra_worker_count(4, 12), 7);
        // 6-source model (HTDemucs): bounded by min(12, threads) - 1.
        assert_eq!(extra_worker_count(6, 4), 3);
        assert_eq!(extra_worker_count(6, 16), 11);
    }
}
