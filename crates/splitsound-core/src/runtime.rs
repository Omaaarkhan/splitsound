//! ONNX Runtime: loading the shared library and building sessions.
//!
//! The engine links nothing at build time. The ONNX Runtime library is
//! opened at run time from one explicit **absolute** path (see
//! [`resolve_library_path`]); the system search path is never consulted,
//! so the only library that can be loaded is the one the caller points at
//! (the official Microsoft build that `scripts/fetch-onnxruntime.mjs`
//! verified and the app bundles).
//!
//! One library per process: ONNX Runtime is a process-wide singleton, so
//! the first [`Runtime::load`] wins and later calls return the same
//! runtime regardless of the path they pass.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel, SessionBuilder};

use crate::error::{Error, ErrorCode, Result};

/// Which hardware runs the models. `Settings.acceleration` in the UI
/// contract.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Acceleration {
    /// The fastest option known to work on this machine. See
    /// [`Runtime::auto_uses_gpu`].
    #[default]
    Auto,
    /// CPU only.
    Cpu,
    /// The GPU path (CoreML on macOS, DirectML on Windows) when one exists,
    /// falling back to CPU if it cannot be set up.
    Gpu,
}

/// What a session actually runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// ONNX Runtime's CPU execution provider.
    Cpu,
    /// Apple CoreML (GPU / Neural Engine), with CPU for unsupported nodes.
    CoreMl,
    /// DirectML on Windows.
    DirectMl,
}

impl Provider {
    /// Display name for `AppInfo.gpu.name`.
    pub fn display_name(self) -> &'static str {
        match self {
            Provider::Cpu => "CPU",
            Provider::CoreMl => "Apple CoreML",
            Provider::DirectMl => "DirectML",
        }
    }
}

/// File name of the ONNX Runtime library on this platform.
pub fn library_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// The `vendor/onnxruntime/<target>` folder name `fetch-onnxruntime.mjs`
/// uses for the host platform.
pub fn host_target_name() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "macos-arm64",
        ("macos", _) => "macos-x64",
        ("windows", _) => "windows-x64",
        ("linux", "aarch64") => "linux-arm64",
        _ => "linux-x64",
    }
}

/// Finds the ONNX Runtime library.
///
/// Order: `explicit` (what the app passes: the copy inside its bundle),
/// then the `ORT_DYLIB_PATH` environment variable, then (in debug and test
/// builds only) `vendor/onnxruntime/<host>/` in the source tree. The result
/// is always absolute and exists.
///
/// # Errors
/// `internal`, with a message that says how to get the library.
pub fn resolve_library_path(explicit: Option<&Path>) -> Result<PathBuf> {
    let mut tried: Vec<String> = Vec::new();
    let candidates = library_candidates(
        explicit,
        std::env::var_os("ORT_DYLIB_PATH")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from),
        cfg!(any(debug_assertions, test)),
    );
    for c in candidates {
        // A directory means "the library inside it".
        let c = if c.is_dir() {
            c.join(library_file_name())
        } else {
            c
        };
        match std::fs::canonicalize(&c) {
            Ok(abs) if abs.is_file() => return Ok(abs),
            _ => tried.push(c.display().to_string()),
        }
    }
    Err(Error::new(
        ErrorCode::Internal,
        "The ONNX Runtime library is missing, so models cannot be run",
    )
    .with_detail(format!(
        "looked in: {}. Run `node scripts/fetch-onnxruntime.mjs` or set ORT_DYLIB_PATH",
        tried.join(", ")
    )))
}

fn library_candidates(
    explicit: Option<&Path>,
    env_override: Option<PathBuf>,
    allow_source_tree: bool,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(p) = explicit {
        candidates.push(p.to_path_buf());
        return candidates;
    }
    if let Some(env) = env_override {
        candidates.push(env);
    }
    #[cfg(any(debug_assertions, test))]
    if allow_source_tree {
        // crates/splitsound-core -> repository root.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
        candidates.push(
            root.join("vendor")
                .join("onnxruntime")
                .join(host_target_name())
                .join(library_file_name()),
        );
    }
    #[cfg(not(any(debug_assertions, test)))]
    let _ = allow_source_tree;
    candidates
}

/// Runtime settings.
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfig {
    /// Absolute path of the ONNX Runtime library (or its folder). `None`
    /// uses the fallbacks of [`resolve_library_path`].
    pub library_path: Option<PathBuf>,
    /// Preferred hardware.
    pub acceleration: Acceleration,
    /// CPU threads for inference. 0 picks automatically.
    pub threads: usize,
    /// Folder where the CoreML provider may cache compiled models. `None`
    /// disables the cache (each session recompiles).
    pub cache_dir: Option<PathBuf>,
}

/// The loaded ONNX Runtime.
pub struct Runtime {
    env: Environment,
    library_path: PathBuf,
    version: String,
    config: Mutex<RuntimeConfig>,
}

static GLOBAL: OnceLock<std::result::Result<Arc<Runtime>, Error>> = OnceLock::new();

fn ort_err(context: &str, e: impl std::fmt::Display) -> Error {
    let text = e.to_string();
    let lower = text.to_ascii_lowercase();
    if lower.contains("bad_alloc")
        || lower.contains("out of memory")
        || lower.contains("failed to allocate")
    {
        Error::out_of_memory().with_detail(format!("{context}: {text}"))
    } else {
        Error::internal(format!("{context}: {text}"))
    }
}

/// Number of threads to use when the caller says "automatic".
///
/// ONNX Runtime's own default counts every core. On machines with
/// performance and efficiency cores (Apple Silicon, recent Intel) the slow
/// cores drag a compute-bound job down, so on Apple Silicon this asks the
/// OS for the number of performance cores. Elsewhere it uses the available
/// parallelism, capped at 16: these models stop scaling well before that.
pub fn auto_threads() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if let Some(p) = macos_performance_cores()
            && p >= 2
        {
            return p.min(logical);
        }
    }
    logical.clamp(1, 16)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn macos_performance_cores() -> Option<usize> {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut value: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // SAFETY: `name` is a NUL-terminated literal, `oldp` points at an i32
    // and `oldlenp` at its size; the call writes at most that many bytes.
    let rc = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut value).cast::<c_void>(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && value > 0 {
        Some(value as usize)
    } else {
        None
    }
}

impl Runtime {
    /// Loads ONNX Runtime (once per process) and returns the shared
    /// runtime. Later calls return the first runtime and only update its
    /// acceleration and thread settings.
    pub fn load(config: RuntimeConfig) -> Result<Arc<Runtime>> {
        let first = GLOBAL.get_or_init(|| Self::load_inner(&config).map(Arc::new));
        match first {
            Ok(rt) => {
                rt.reconfigure(config.acceleration, config.threads);
                Ok(Arc::clone(rt))
            }
            Err(e) => Err(e.clone()),
        }
    }

    fn load_inner(config: &RuntimeConfig) -> Result<Runtime> {
        let library_path = resolve_library_path(config.library_path.as_deref())?;
        let builder = ort::init_from(&library_path).map_err(|e| {
            Error::new(
                ErrorCode::Internal,
                "The ONNX Runtime library could not be loaded",
            )
            .with_detail(e)
        })?;
        let env = builder
            .with_name("splitsound")
            // Nothing leaves the machine. (Telemetry only exists in the
            // Windows build of ONNX Runtime; this turns it off there.)
            .with_telemetry(false)
            .build()
            .map_err(|e| ort_err("creating the ONNX Runtime environment", e))?;
        let version = ort::info().to_string();
        Ok(Runtime {
            env,
            library_path,
            version,
            config: Mutex::new(config.clone()),
        })
    }

    /// Changes the acceleration and thread settings for sessions built
    /// from now on. Existing sessions keep theirs.
    pub fn reconfigure(&self, acceleration: Acceleration, threads: usize) {
        let mut cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
        cfg.acceleration = acceleration;
        cfg.threads = threads;
    }

    /// The library that was loaded.
    pub fn library_path(&self) -> &Path {
        &self.library_path
    }

    /// ONNX Runtime's build information string.
    pub fn build_info(&self) -> &str {
        &self.version
    }

    /// Threads sessions are built with right now.
    pub fn threads(&self) -> usize {
        let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
        if cfg.threads == 0 {
            auto_threads()
        } else {
            cfg.threads
        }
    }

    /// The GPU execution path this build could use on this machine, if any.
    pub fn gpu_provider(&self) -> Option<Provider> {
        if cfg!(target_os = "macos") {
            use ort::ep::ExecutionProvider;
            if ort::ep::CoreML::is_available().unwrap_or(false) {
                return Some(Provider::CoreMl);
            }
        }
        None
    }

    /// Whether `Acceleration::Auto` uses the GPU path for separation
    /// models on this machine.
    ///
    /// It does not, anywhere, today. On Apple Silicon CoreML does nothing
    /// for SCNet (its LSTMs stay on the CPU) and helps only `htdemucs_6s`,
    /// at the price of a long model compilation (ml/BENCHMARKS.md); no
    /// other platform's GPU path could be tested. The GPU path stays an
    /// explicit opt-in.
    pub fn auto_uses_gpu(&self) -> bool {
        false
    }

    fn wants_gpu(&self) -> bool {
        let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
        match cfg.acceleration {
            Acceleration::Cpu => false,
            Acceleration::Gpu => true,
            Acceleration::Auto => self.auto_uses_gpu(),
        }
    }

    fn base_builder(&self) -> Result<SessionBuilder> {
        let threads = self.threads();
        Session::builder(&self.env)
            .map_err(|e| ort_err("session options", e))?
            // `ort` switches on ONNX Runtime's automatic device selection
            // by default, which on macOS silently registers CoreML (and
            // spends minutes compiling these graphs for the Neural Engine).
            // The engine decides the provider itself: never auto-register.
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(|e| ort_err("session options", e))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| ort_err("session options", e))?
            .with_intra_threads(threads)
            .map_err(|e| ort_err("session options", e))?
            // One model call at a time; no second pool competing for cores.
            .with_inter_threads(1)
            .map_err(|e| ort_err("session options", e))
        // The graphs' one symbolic dimension, `batch`, is deliberately
        // left symbolic on the CPU provider: fixing it makes ONNX Runtime
        // plan one arena for every intermediate tensor and triples peak
        // memory for no gain in speed (ml/BENCHMARKS.md, section 3). The
        // arena and memory-pattern defaults are the best measured there.
    }

    fn gpu_builder(&self) -> Result<(SessionBuilder, Provider)> {
        #[cfg(target_os = "macos")]
        {
            use ort::ep::coreml::{ComputeUnits, ModelFormat};
            let mut ep = ort::ep::CoreML::default()
                .with_model_format(ModelFormat::MLProgram)
                .with_compute_units(ComputeUnits::All)
                .with_static_input_shapes(true);
            let cache = {
                let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
                cfg.cache_dir.clone()
            };
            if let Some(dir) = cache
                && std::fs::create_dir_all(&dir).is_ok()
            {
                ep = ep.with_model_cache_dir(dir.display());
            }
            let builder = self
                .base_builder()?
                // CoreML wants static shapes.
                .with_dimension_override("batch", 1)
                .map_err(|e| ort_err("session options", e))?
                .with_execution_providers([ep.build().error_on_failure()])
                .map_err(|e| ort_err("registering CoreML", e))?;
            Ok((builder, Provider::CoreMl))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(Error::internal("no GPU execution path in this build"))
        }
    }

    /// Builds a session with `commit`, trying the GPU path first when the
    /// settings ask for it and falling back to CPU if the provider fails
    /// to register or the session fails to build on it.
    fn build_with_fallback(
        &self,
        commit: &dyn Fn(SessionBuilder) -> std::result::Result<Session, ort::Error>,
    ) -> Result<(Session, Provider)> {
        if self.wants_gpu()
            && let Ok((builder, provider)) = self.gpu_builder()
            && let Ok(session) = commit(builder)
        {
            return Ok((session, provider));
        }
        let session = commit(self.base_builder()?).map_err(|e| ort_err("loading the model", e))?;
        Ok((session, Provider::Cpu))
    }

    /// Builds a session from model bytes that are already verified. The
    /// model must be self-contained (no external data).
    pub fn session_from_memory(&self, model: &[u8]) -> Result<(Session, Provider)> {
        self.build_with_fallback(&|mut b| b.commit_from_memory(model))
    }

    /// Builds a session from verified bytes: the graph, plus (for a graph
    /// whose weights live in a sibling file) that file's verified contents
    /// under the name the graph refers to.
    ///
    /// The external data is handed to ONNX Runtime from memory, so the
    /// runtime never resolves a path next to the graph: what was verified
    /// is exactly what is loaded, and the graph cannot reach any other file.
    pub fn session_from_verified(
        &self,
        graph: &[u8],
        external: Option<(&str, std::sync::Arc<[u8]>)>,
    ) -> Result<(Session, Provider)> {
        match external {
            None => self.session_from_memory(graph),
            Some((name, bytes)) => self.build_with_fallback(&|b| {
                b.with_external_initializer_file_in_memory(name, bytes.clone())
                    .map_err(ort::Error::<()>::from)?
                    .commit_from_memory(graph)
            }),
        }
    }

    /// Builds a session from a model file. Needed for graphs whose weights
    /// live in a sibling external-data file, which ONNX Runtime resolves
    /// relative to the graph's own folder.
    pub fn session_from_file(&self, path: &Path) -> Result<(Session, Provider)> {
        self.build_with_fallback(&|mut b| b.commit_from_file(path))
    }
}

/// Converts an ONNX Runtime error raised while running a model.
pub(crate) fn run_error(e: impl std::fmt::Display) -> Error {
    ort_err("running the model", e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_consistent_with_the_fetch_script() {
        assert!(
            [
                "macos-arm64",
                "macos-x64",
                "windows-x64",
                "linux-x64",
                "linux-arm64"
            ]
            .contains(&host_target_name())
        );
        assert!(
            [
                "libonnxruntime.dylib",
                "onnxruntime.dll",
                "libonnxruntime.so"
            ]
            .contains(&library_file_name())
        );
    }

    #[test]
    fn explicit_missing_library_is_a_clear_error() {
        let e = resolve_library_path(Some(Path::new("/definitely/not/here/libonnxruntime.so")))
            .unwrap_err();
        assert_eq!(e.code(), ErrorCode::Internal);
        assert!(e.message().contains("ONNX Runtime"));
        assert!(e.detail().unwrap().contains("/definitely/not/here"));
    }

    #[test]
    fn explicit_path_is_never_replaced_by_a_fallback() {
        // Even if a vendored copy exists, an explicit path that is wrong
        // must fail rather than silently load something else.
        let dir = crate::testutil::TempDir::new("ortpath");
        assert!(resolve_library_path(Some(dir.path())).is_err());
        let fake = dir.path().join(library_file_name());
        std::fs::write(&fake, b"not a real library").unwrap();
        let got = resolve_library_path(Some(dir.path())).unwrap();
        assert!(got.is_absolute());
        assert_eq!(got.file_name().unwrap(), library_file_name());
    }

    #[test]
    fn auto_threads_is_sane() {
        let n = auto_threads();
        assert!((1..=64).contains(&n), "{n}");
    }

    #[test]
    fn release_mode_has_no_source_tree_fallback() {
        assert!(library_candidates(None, None, false).is_empty());
        let env_path = PathBuf::from("/custom/ort.so");
        assert_eq!(
            library_candidates(None, Some(env_path.clone()), false),
            vec![env_path]
        );
        assert_eq!(library_candidates(None, None, true).len(), 1);
    }
}
