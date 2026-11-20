//! A single background thread that runs jobs one at a time, in order,
//! behind a panic boundary. Used for the split queue and the instrument
//! scan queue.

use std::collections::VecDeque;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::util::{CancelToken, lock};

/// A job with a key (the session id) used to find it for cancellation.
pub trait Job: Send + 'static {
    fn key(&self) -> &str;
}

impl Job for String {
    fn key(&self) -> &str {
        self
    }
}

/// What [`Worker::cancel`] did.
#[derive(Debug, PartialEq, Eq)]
pub enum Cancelled<J> {
    /// The job had not started; it was removed from the queue.
    Queued(J),
    /// The job is running; its token was cancelled.
    Running,
    /// No such job.
    Idle,
}

struct State<J> {
    pending: VecDeque<J>,
    running: Option<(String, CancelToken)>,
    shutdown: bool,
}

struct Inner<J> {
    state: Mutex<State<J>>,
    cv: Condvar,
}

pub struct Worker<J: Job> {
    inner: Arc<Inner<J>>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

/// Turns a panic payload into text for the log.
pub fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

impl<J: Job> Worker<J> {
    /// Starts the thread. `run` executes each job; if it panics,
    /// `on_panic(key, message)` is called and the worker carries on.
    pub fn start(
        name: &str,
        run: impl Fn(J, &CancelToken) + Send + 'static,
        on_panic: impl Fn(String, String) + Send + 'static,
    ) -> Self {
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                pending: VecDeque::<J>::new(),
                running: None,
                shutdown: false,
            }),
            cv: Condvar::new(),
        });
        let i = inner.clone();
        let thread = thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                loop {
                    let (job, token) = {
                        let mut st = lock(&i.state);
                        loop {
                            if st.shutdown {
                                return;
                            }
                            if let Some(job) = st.pending.pop_front() {
                                let token = CancelToken::new();
                                st.running = Some((job.key().to_string(), token.clone()));
                                break (job, token);
                            }
                            st = i.cv.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let key = job.key().to_string();
                    let result = panic::catch_unwind(AssertUnwindSafe(|| run(job, &token)));
                    if let Err(p) = result {
                        let msg = panic_message(p.as_ref());
                        log::error!("job {key} panicked: {msg}");
                        let _ = panic::catch_unwind(AssertUnwindSafe(|| on_panic(key, msg)));
                    }
                    lock(&i.state).running = None;
                    i.cv.notify_all();
                }
            })
            .expect("failed to spawn a worker thread");
        Worker {
            inner,
            thread: Mutex::new(Some(thread)),
        }
    }

    pub fn push(&self, job: J) {
        lock(&self.inner.state).pending.push_back(job);
        self.inner.cv.notify_all();
    }

    pub fn cancel(&self, key: &str) -> Cancelled<J> {
        let mut st = lock(&self.inner.state);
        if let Some(i) = st.pending.iter().position(|j| j.key() == key) {
            return Cancelled::Queued(st.pending.remove(i).expect("index is valid"));
        }
        match &st.running {
            Some((k, token)) if k == key => {
                token.cancel();
                Cancelled::Running
            }
            _ => Cancelled::Idle,
        }
    }

    /// True if a job with this key is queued or running.
    pub fn contains(&self, key: &str) -> bool {
        let st = lock(&self.inner.state);
        st.pending.iter().any(|j| j.key() == key)
            || st.running.as_ref().is_some_and(|(k, _)| k == key)
    }

    pub fn is_running(&self, key: &str) -> bool {
        lock(&self.inner.state)
            .running
            .as_ref()
            .is_some_and(|(k, _)| k == key)
    }

    /// Waits up to `timeout` for the job `key` to stop running.
    pub fn wait_until_stopped(&self, key: &str, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut st = lock(&self.inner.state);
        while st.running.as_ref().is_some_and(|(k, _)| k == key) {
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            st = self
                .inner
                .cv
                .wait_timeout(st, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }

    /// Cancels everything and stops the thread.
    pub fn shutdown(&self) {
        {
            let mut st = lock(&self.inner.state);
            st.shutdown = true;
            st.pending.clear();
            if let Some((_, t)) = &st.running {
                t.cancel();
            }
        }
        self.inner.cv.notify_all();
        if let Some(t) = lock(&self.thread).take() {
            // The last owner may be the worker itself (it upgrades a weak
            // reference while running a job); never join from inside.
            if t.thread().id() != thread::current().id() {
                let _ = t.join();
            }
        }
    }
}

impl<J: Job> Drop for Worker<J> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn runs_in_order_survives_panics_and_cancels() {
        let (tx, rx) = mpsc::channel::<String>();
        let (ptx, prx) = mpsc::channel::<String>();
        let w = Worker::<String>::start(
            "test-worker",
            move |job, token| {
                if job == "boom" {
                    panic!("kaboom");
                }
                if job == "slow" {
                    while !token.is_cancelled() {
                        thread::sleep(Duration::from_millis(5));
                    }
                    tx.send("slow cancelled".into()).unwrap();
                    return;
                }
                tx.send(job).unwrap();
            },
            move |key, msg| ptx.send(format!("{key}: {msg}")).unwrap(),
        );
        w.push("a".into());
        w.push("boom".into());
        w.push("b".into());
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), "a");
        assert_eq!(
            prx.recv_timeout(Duration::from_secs(2)).unwrap(),
            "boom: kaboom"
        );
        assert_eq!(rx.recv_timeout(Duration::from_secs(2)).unwrap(), "b");
        w.push("slow".into());
        w.push("queued".into());
        while !w.is_running("slow") {
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(w.cancel("queued"), Cancelled::Queued("queued".to_string()));
        assert_eq!(w.cancel("slow"), Cancelled::Running);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            "slow cancelled"
        );
        assert!(w.wait_until_stopped("slow", Duration::from_secs(2)));
        assert_eq!(w.cancel("nothing"), Cancelled::Idle);
        assert!(!w.contains("queued"));
        w.shutdown();
    }
}
