//! Progress reporting and cancellation.
//!
//! Long operations take a `&dyn Progress`. The engine calls
//! [`Progress::report`] from the thread doing the work (never concurrently
//! for one operation) and polls [`Progress::is_cancelled`] between units of
//! work: between model segments while separating, between packets while
//! decoding, between network reads while downloading.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::error::{Error, Result};

/// The stage of a split, matching `SplitStage` in the UI contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    /// Fetching a model that is not installed yet.
    Downloading,
    /// Loading models, preparing audio.
    Preparing,
    /// Running the separation models.
    Separating,
    /// Measuring each stem and listening for instruments inside it.
    Analyzing,
    /// Writing stems to disk.
    Writing,
}

/// One progress report. Field for field the engine half of `SplitProgress`
/// in the UI contract (the shell adds `sessionId`).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressUpdate {
    /// Current stage.
    pub stage: Stage,
    /// Overall progress in `0.0..=1.0`. Never decreases within one operation.
    pub fraction: f32,
    /// User-facing label, sentence case, no trailing period: "Separating drums".
    pub label: String,
    /// 1-based index of the current step.
    pub step: u32,
    /// Total number of steps, for "2 of 5".
    pub step_count: u32,
    /// Estimated seconds remaining; `None` until there is enough history.
    pub eta_sec: Option<f32>,
}

/// Receives progress and answers "should I stop?".
///
/// Implementations must be cheap and must not block: `report` is called on
/// the worker thread, up to a few times per second.
pub trait Progress: Send + Sync {
    /// Called with each progress update. The default does nothing.
    fn report(&self, _update: &ProgressUpdate) {}

    /// Polled regularly. Return `true` to make the operation stop with an
    /// error whose code is `cancelled`. The default never cancels.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// A [`Progress`] that ignores reports and never cancels.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProgress;

impl Progress for NoProgress {}

/// A shareable cancellation flag that is also a (silent) [`Progress`].
///
/// Clone it, hand one clone to the operation and call [`CancelFlag::cancel`]
/// on the other from any thread.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    /// A new, un-cancelled flag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation was requested.
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Clears the flag so it can be reused.
    pub fn reset(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl Progress for CancelFlag {
    fn is_cancelled(&self) -> bool {
        self.is_set()
    }
}

/// A [`Progress`] built from a closure and a [`CancelFlag`]. Handy for the
/// shell: forward updates as events, cancel through the flag.
pub struct FnProgress<F: Fn(&ProgressUpdate) + Send + Sync> {
    callback: F,
    cancel: CancelFlag,
}

impl<F: Fn(&ProgressUpdate) + Send + Sync> FnProgress<F> {
    /// Wraps `callback`; cancellation is driven by `cancel`.
    pub fn new(callback: F, cancel: CancelFlag) -> Self {
        Self { callback, cancel }
    }
}

impl<F: Fn(&ProgressUpdate) + Send + Sync> Progress for FnProgress<F> {
    fn report(&self, update: &ProgressUpdate) {
        (self.callback)(update);
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.is_set()
    }
}

/// Returns the cancellation error if `progress` asks to stop.
pub(crate) fn check_cancelled(progress: &dyn Progress) -> Result<()> {
    if progress.is_cancelled() {
        Err(Error::cancelled())
    } else {
        Ok(())
    }
}

/// One planned step of a multi-step operation.
#[derive(Debug, Clone)]
pub(crate) struct PlannedStep {
    pub stage: Stage,
    pub label: String,
    /// Relative cost. Only ratios matter.
    pub weight: f64,
}

/// Turns per-step progress into a single monotonic fraction with an ETA.
///
/// The plan is fixed up front (one entry per user-visible step, weighted by
/// expected cost), so the fraction is a weighted sum and can never go
/// backwards. The ETA comes from the measured rate of the heavy
/// ([`Stage::Separating`]) work so far, applied to the weight that remains;
/// it is withheld until a few seconds and a few percent of history exist.
pub(crate) struct Tracker<'a> {
    sink: &'a dyn Progress,
    steps: Vec<PlannedStep>,
    total_weight: f64,
    state: Mutex<TrackerState>,
}

struct TrackerState {
    current: usize,
    done_before_current: f64,
    last_fraction: f64,
    /// When the first weighted work was reported, and the fraction at that time.
    rate_origin: Option<(Instant, f64)>,
}

impl<'a> Tracker<'a> {
    pub fn new(sink: &'a dyn Progress, steps: Vec<PlannedStep>) -> Self {
        let total_weight = steps
            .iter()
            .map(|s| s.weight.max(0.0))
            .sum::<f64>()
            .max(1e-9);
        Tracker {
            sink,
            steps,
            total_weight,
            state: Mutex::new(TrackerState {
                current: 0,
                done_before_current: 0.0,
                last_fraction: 0.0,
                rate_origin: None,
            }),
        }
    }

    /// Moves to step `index` (0-based) and reports its start.
    pub fn begin(&self, index: usize) -> Result<()> {
        {
            let mut st = self.lock();
            let index = index.min(self.steps.len().saturating_sub(1));
            st.current = index;
            st.done_before_current = self.steps[..index].iter().map(|s| s.weight.max(0.0)).sum();
        }
        self.update(0.0)
    }

    /// Reports progress `within` (0..=1) the current step. Also the
    /// cancellation point: returns the `cancelled` error if asked to stop.
    pub fn update(&self, within: f64) -> Result<()> {
        check_cancelled(self.sink)?;
        if self.steps.is_empty() {
            return Ok(());
        }
        let update = {
            let mut st = self.lock();
            let step = &self.steps[st.current];
            let done = st.done_before_current + step.weight.max(0.0) * within.clamp(0.0, 1.0);
            let fraction = (done / self.total_weight)
                .clamp(0.0, 1.0)
                .max(st.last_fraction);
            st.last_fraction = fraction;
            let now = Instant::now();
            if st.rate_origin.is_none() && fraction > 0.0 {
                st.rate_origin = Some((now, fraction));
            }
            let eta_sec = st.rate_origin.and_then(|(t0, f0)| {
                let elapsed = now.duration_since(t0).as_secs_f64();
                let gained = fraction - f0;
                if elapsed >= 3.0 && gained >= 0.02 && fraction < 1.0 {
                    Some(((1.0 - fraction) * elapsed / gained) as f32)
                } else {
                    None
                }
            });
            ProgressUpdate {
                stage: step.stage,
                fraction: fraction as f32,
                label: step.label.clone(),
                step: (st.current + 1) as u32,
                step_count: self.steps.len() as u32,
                eta_sec,
            }
        };
        self.sink.report(&update);
        Ok(())
    }

    /// The underlying sink, for passing to nested operations that only need
    /// cancellation.
    pub fn sink(&self) -> &'a dyn Progress {
        self.sink
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TrackerState> {
        // A poisoned lock only means a reporter panicked; the state is still
        // usable for further reports.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Collect(Mutex<Vec<ProgressUpdate>>, CancelFlag);
    impl Progress for Collect {
        fn report(&self, update: &ProgressUpdate) {
            self.0.lock().unwrap().push(update.clone());
        }
        fn is_cancelled(&self) -> bool {
            self.1.is_set()
        }
    }

    fn plan() -> Vec<PlannedStep> {
        vec![
            PlannedStep {
                stage: Stage::Preparing,
                label: "Getting ready".into(),
                weight: 1.0,
            },
            PlannedStep {
                stage: Stage::Separating,
                label: "Separating drums".into(),
                weight: 8.0,
            },
            PlannedStep {
                stage: Stage::Writing,
                label: "Saving stems".into(),
                weight: 1.0,
            },
        ]
    }

    #[test]
    fn fraction_is_weighted_and_monotonic() {
        let sink = Collect(Mutex::new(Vec::new()), CancelFlag::new());
        let t = Tracker::new(&sink, plan());
        t.begin(0).unwrap();
        t.update(1.0).unwrap();
        t.begin(1).unwrap();
        t.update(0.5).unwrap();
        // Going "backwards" inside a step must not lower the fraction.
        t.update(0.25).unwrap();
        t.begin(2).unwrap();
        t.update(1.0).unwrap();
        let seen = sink.0.lock().unwrap();
        let fr: Vec<f32> = seen.iter().map(|u| u.fraction).collect();
        assert!(fr.windows(2).all(|w| w[1] >= w[0]), "{fr:?}");
        assert!((fr[1] - 0.1).abs() < 1e-6);
        assert!((fr[3] - 0.5).abs() < 1e-6);
        assert!((fr[4] - 0.5).abs() < 1e-6, "clamped, not lowered");
        assert!((fr.last().unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(seen[2].step, 2);
        assert_eq!(seen[2].step_count, 3);
        assert_eq!(seen[2].label, "Separating drums");
        assert_eq!(seen[2].stage, Stage::Separating);
    }

    #[test]
    fn cancellation_surfaces_as_error() {
        let sink = Collect(Mutex::new(Vec::new()), CancelFlag::new());
        let t = Tracker::new(&sink, plan());
        t.begin(0).unwrap();
        sink.1.cancel();
        let err = t.update(0.5).unwrap_err();
        assert!(err.is_cancelled());
        assert_eq!(err.code().as_str(), "cancelled");
    }

    #[test]
    fn progress_update_serialises_like_the_contract() {
        let u = ProgressUpdate {
            stage: Stage::Separating,
            fraction: 0.5,
            label: "Separating drums".into(),
            step: 2,
            step_count: 5,
            eta_sec: None,
        };
        let v: serde_json::Value = serde_json::to_value(&u).unwrap();
        assert_eq!(v["stage"], "separating");
        assert_eq!(v["stepCount"], 5);
        assert!(v["etaSec"].is_null());
    }
}
