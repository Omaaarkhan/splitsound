//! The model store: installing, verifying and loading model files.
//!
//! The rules ("Models are pinned" in `docs/architecture.md`):
//!
//! * The manifest is compiled in. A file is accepted only if its size and
//!   SHA-256 match the manifest; there is no way to load anything else.
//! * Downloads are HTTPS only, from a fixed base URL (project files) or an
//!   allow-listed publisher host (upstream checkpoints). Redirects are
//!   followed by hand and must stay on HTTPS and on allow-listed hosts.
//! * A download is cut off the moment it exceeds the pinned size, and its
//!   SHA-256 must match before it is decompressed.
//! * Decompression (pure-Rust zstd) is bounded by the pinned size of the
//!   result, whose SHA-256 must match too.
//! * Files are written to a temporary name and renamed into place only
//!   after they verified.
//! * Every load re-checks the digest and hands the *verified bytes* to the
//!   runtime, so what was checked is exactly what runs.
//! * A restricted model is never fetched or installed unless the caller
//!   passes `accept_restricted = true`.
//! * Files imported by hand go through the same checks.

use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::checkpoint::{convert_checkpoint_bytes, hex};
use crate::error::{Error, ErrorCode, Result};
use crate::manifest::{
    Manifest, ModelEntry, PinnedFile, UPSTREAM_HOSTS, is_safe_file_name, parse_https_url,
    tensor_table_for,
};
use crate::progress::Progress;
use crate::wav::{TempFile, create_temp_sibling};

/// Where project model files are downloaded from.
pub const DEFAULT_BASE_URL: &str =
    "https://github.com/Omaaarkhan/splitsound/releases/download/models-v1/";

/// Hosts a project download may be served or redirected from. GitHub
/// answers release downloads with a redirect to its asset storage.
const RELEASE_HOSTS: &[&str] = &[
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
];

const MAX_REDIRECTS: usize = 5;

/// Where the store gets files from.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// The real thing: `base_url` (HTTPS) for project files, the
    /// publisher's URL from the manifest for upstream checkpoints.
    Release {
        /// Base URL ending in `/`. Must be `https://`.
        base_url: String,
    },
    /// A local folder that holds the same files under the same names
    /// (an upstream checkpoint under the last component of its URL). For
    /// tests and offline mirrors. Files still have to match the manifest.
    Directory(PathBuf),
}

impl Default for ModelSource {
    fn default() -> Self {
        ModelSource::Release {
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }
}

/// Receives download progress and answers "should I stop?".
pub trait DownloadObserver: Send + Sync {
    /// `received` of `total` bytes fetched so far for `model_id`
    /// (`ModelDownloadProgress` in the UI contract). Called a few times a
    /// second at most.
    fn progress(&self, _model_id: &str, _received: u64, _total: u64) {}

    /// Return `true` to stop; the install fails with `cancelled`.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// An observer that reports nothing and never cancels.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoObserver;

impl DownloadObserver for NoObserver {}

/// Lets a [`DownloadObserver`] stand in where only cancellation is needed.
struct CancelOnly<'a>(&'a dyn DownloadObserver);

impl Progress for CancelOnly<'_> {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

/// Status of one model; `ModelStatus` in the UI contract (the shell fills
/// `downloadProgress` from the progress events).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    /// Model id.
    pub id: String,
    /// Display name.
    pub name: String,
    /// One plain sentence on what it is for.
    pub description: String,
    /// Size on disk once installed.
    pub size_bytes: u64,
    /// Size of the download.
    pub download_bytes: u64,
    /// Whether every file of the model is present with the right size.
    /// (The digest is checked when the model is loaded or verified.)
    pub installed: bool,
    /// Always `None` from the engine.
    pub download_progress: Option<f32>,
    /// Licence of the weights.
    pub license: String,
    /// Who made it.
    pub attribution: String,
    /// Who the download comes from: "GitHub" or "Meta".
    pub source: String,
    /// True when the weights are not open source.
    pub restricted: bool,
    /// Terms to show before a restricted model is fetched.
    pub terms: Option<String>,
}

/// Model bytes that passed their digest check, ready for the runtime.
pub struct VerifiedModel {
    /// The ONNX graph.
    pub graph: Vec<u8>,
    /// The graph's external data, for models whose weights live in a
    /// sibling file: (file name the graph refers to, verified contents).
    pub external: Option<(String, Arc<[u8]>)>,
}

impl std::fmt::Debug for VerifiedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedModel")
            .field("graph_bytes", &self.graph.len())
            .field(
                "external",
                &self.external.as_ref().map(|(n, b)| (n, b.len())),
            )
            .finish()
    }
}

/// The model store.
pub struct ModelStore {
    dir: PathBuf,
    manifest: Manifest,
    source: ModelSource,
    /// Serialises installs and removals.
    busy: Mutex<()>,
}

fn download_failed(detail: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::ModelDownloadFailed,
        "The model download did not complete correctly. Please try again",
    )
    .with_detail(detail)
}

fn offline(detail: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::Offline,
        "SplitSound could not reach the internet to download a model",
    )
    .with_detail(detail)
}

fn terms_required(entry: &ModelEntry) -> Error {
    Error::new(
        ErrorCode::TermsRequired,
        format!(
            "\u{201c}{}\u{201d} has licence terms that need to be accepted first",
            entry.name
        ),
    )
}

fn missing(entry: &ModelEntry) -> Error {
    Error::new(
        ErrorCode::ModelMissing,
        format!("The model \u{201c}{}\u{201d} is not installed", entry.name),
    )
}

fn damaged(entry: &ModelEntry, detail: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::ModelMissing,
        format!(
            "The model \u{201c}{}\u{201d} is damaged and needs to be downloaded again",
            entry.name
        ),
    )
    .with_detail(detail)
}

impl ModelStore {
    /// Opens (creating if needed) the store at `dir` with the compiled-in
    /// manifest and the default download source.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        Self::with_source(dir, Manifest::builtin().clone(), ModelSource::default())
    }

    /// Opens a store with an explicit manifest and source (tests, mirrors).
    pub fn with_source(
        dir: impl Into<PathBuf>,
        manifest: Manifest,
        source: ModelSource,
    ) -> Result<Self> {
        let dir = dir.into();
        if let ModelSource::Release { base_url } = &source {
            let ok = base_url.ends_with('/') && parse_https_url(base_url).is_some();
            if !ok {
                return Err(Error::internal(
                    "the model base URL must be https:// and end with a slash",
                ));
            }
        }
        std::fs::create_dir_all(&dir).map_err(|e| Error::from_write(&e, &dir))?;
        Ok(ModelStore {
            dir,
            manifest,
            source,
            busy: Mutex::new(()),
        })
    }

    /// The folder models are installed in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The manifest in use.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn path_of(&self, file: &PinnedFile) -> PathBuf {
        // Names were validated when the manifest was parsed: one plain
        // component each.
        self.dir.join(&file.name)
    }

    fn has_file(&self, file: &PinnedFile) -> bool {
        std::fs::metadata(self.path_of(file))
            .is_ok_and(|m| m.is_file() && m.len() == file.size_bytes)
    }

    /// Whether every file of `id` is present with the right size.
    pub fn is_installed(&self, id: &str) -> bool {
        self.manifest.find(id).is_some_and(|m| {
            self.has_file(&m.file) && m.weights.as_ref().is_none_or(|w| self.has_file(&w.file))
        })
    }

    fn status_of(&self, m: &ModelEntry) -> ModelStatus {
        ModelStatus {
            id: m.id.clone(),
            name: m.name.clone(),
            description: m.description.clone(),
            size_bytes: m.installed_bytes(),
            download_bytes: m.download_bytes(),
            installed: self.is_installed(&m.id),
            download_progress: None,
            license: m.license.clone(),
            attribution: m.attribution.clone(),
            source: m.source.clone(),
            restricted: m.restricted,
            terms: m.terms.clone(),
        }
    }

    /// Status of every model, in manifest order.
    pub fn list(&self) -> Vec<ModelStatus> {
        self.manifest
            .models
            .iter()
            .map(|m| self.status_of(m))
            .collect()
    }

    /// Status of one model.
    pub fn status(&self, id: &str) -> Result<ModelStatus> {
        Ok(self.status_of(self.manifest.get(id)?))
    }

    /// Installs `id` if it is not installed yet.
    ///
    /// `accept_restricted` must be `true` for a restricted model; without
    /// it nothing is fetched and the error code is `terms_required`.
    ///
    /// # Errors
    /// `not_found`, `terms_required`, `offline`, `model_download_failed`,
    /// `cancelled`, `disk_full`.
    pub fn install(
        &self,
        id: &str,
        accept_restricted: bool,
        observer: &dyn DownloadObserver,
    ) -> Result<()> {
        let entry = self.manifest.get(id)?;
        if entry.restricted && !accept_restricted {
            return Err(terms_required(entry));
        }
        let _guard = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        if self.is_installed(id) {
            return Ok(());
        }
        let total = entry.download_bytes();
        let mut done = 0u64;

        if !self.has_file(&entry.file) {
            let Some(download) = &entry.download else {
                return Err(Error::new(
                    ErrorCode::ModelDownloadFailed,
                    format!(
                        "The model \u{201c}{}\u{201d} is not available for download yet",
                        entry.name
                    ),
                ));
            };
            let (file, _tmp) = self.fetch_project_file(
                download,
                &mut |n| {
                    observer.progress(&entry.id, done + n, total);
                },
                observer,
            )?;
            self.install_compressed(file, &entry.file, observer)?;
            done += download.size_bytes;
        } else if let Some(d) = &entry.download {
            done += d.size_bytes;
        }

        if let Some(weights) = &entry.weights
            && !self.has_file(&weights.file)
        {
            let table = tensor_table_for(entry)?;
            let name = upstream_file_name(&weights.upstream.url)?;
            let pinned = PinnedFile {
                name: name.to_string(),
                sha256: weights.upstream.sha256.clone(),
                size_bytes: weights.upstream.size_bytes,
            };
            // The checkpoint itself is never kept on disk: verify it directly
            // in memory and convert those exact verified bytes.
            let bytes = self.fetch_upstream_bytes(
                &weights.upstream.url,
                &pinned,
                &mut |n| {
                    observer.progress(&entry.id, done + n, total);
                },
                observer,
            )?;
            convert_checkpoint_bytes(
                &bytes,
                &table,
                &self.path_of(&weights.file),
                &CancelOnly(observer),
            )?;
        }
        observer.progress(&entry.id, total, total);
        Ok(())
    }

    /// Installs every model in `ids` that is missing.
    pub fn ensure(
        &self,
        ids: &[&str],
        accept_restricted: bool,
        observer: &dyn DownloadObserver,
    ) -> Result<()> {
        // Refuse up front, before fetching anything, if any model needs
        // terms that were not accepted.
        for id in ids {
            let entry = self.manifest.get(id)?;
            if entry.restricted && !accept_restricted && !self.is_installed(id) {
                return Err(terms_required(entry));
            }
        }
        for id in ids {
            self.install(id, accept_restricted, observer)?;
        }
        Ok(())
    }

    /// Removes a model's files. Not an error if it was not installed.
    pub fn remove(&self, id: &str) -> Result<()> {
        let entry = self.manifest.get(id)?;
        let _guard = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let mut files = vec![&entry.file];
        files.extend(entry.weights.as_ref().map(|w| &w.file));
        for f in files {
            let path = self.path_of(f);
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(Error::from_write(&e, &path)),
            }
        }
        Ok(())
    }

    /// Fully re-checks an installed model's files against the manifest.
    ///
    /// # Errors
    /// `model_missing` if a file is absent or does not match.
    pub fn verify(&self, id: &str) -> Result<()> {
        self.load(id).map(|_| ())
    }

    /// Reads a model's files, checks size and SHA-256 of each, and returns
    /// the verified bytes.
    ///
    /// For a model with external weights the graph is also checked to
    /// reference exactly its own weights file and nothing else; for a
    /// self-contained model, to reference no external file at all.
    pub fn load(&self, id: &str) -> Result<VerifiedModel> {
        let entry = self.manifest.get(id)?;
        let graph = self.read_verified(entry, &entry.file)?;
        let external = match &entry.weights {
            None => {
                check_external_locations(&graph, None).map_err(|e| damaged(entry, e))?;
                None
            }
            Some(w) => {
                check_external_locations(&graph, Some(&w.file.name))
                    .map_err(|e| damaged(entry, e))?;
                let bytes = self.read_verified(entry, &w.file)?;
                Some((w.file.name.clone(), Arc::<[u8]>::from(bytes)))
            }
        };
        Ok(VerifiedModel { graph, external })
    }

    fn read_verified(&self, entry: &ModelEntry, file: &PinnedFile) -> Result<Vec<u8>> {
        let path = self.path_of(file);
        let mut f = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(missing(entry)),
            Err(e) => return Err(Error::from_read(&e, &path)),
        };
        let len = f.metadata().map_err(|e| Error::from_read(&e, &path))?.len();
        if len != file.size_bytes {
            return Err(damaged(
                entry,
                format!("{}: size {len}, expected {}", file.name, file.size_bytes),
            ));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len as usize)
            .map_err(|_| Error::out_of_memory())?;
        // `take` so a file that grows underneath us cannot outrun the size.
        (&mut f)
            .take(len + 1)
            .read_to_end(&mut data)
            .map_err(|e| Error::from_read(&e, &path))?;
        if data.len() as u64 != file.size_bytes || hex(&Sha256::digest(&data)) != file.sha256 {
            return Err(damaged(entry, format!("{}: SHA-256 mismatch", file.name)));
        }
        Ok(data)
    }

    /// Installs a model file the user obtained by hand.
    ///
    /// The file is recognised purely by content: its size and SHA-256 must
    /// equal a pinned installed file, a pinned compressed download or a
    /// pinned upstream checkpoint. Returns the id of the model it belongs
    /// to, or `None` if it is not a known model file (nothing is copied in
    /// that case).
    ///
    /// # Errors
    /// `terms_required` when the file belongs to a restricted model and
    /// `accept_restricted` is false; `file_not_found`, `disk_full`,
    /// `model_download_failed` (a recognised archive that does not unpack
    /// to the pinned content).
    pub fn import_file(&self, path: &Path, accept_restricted: bool) -> Result<Option<String>> {
        let mut file = File::open(path).map_err(|e| Error::from_read(&e, path))?;
        let meta = file.metadata().map_err(|e| Error::from_read(&e, path))?;
        if !meta.is_file() {
            return Ok(None);
        }
        let len = meta.len();
        // Cheap pre-filter: only read files whose size is a pinned size.
        let known_size = self.manifest.models.iter().any(|m| {
            m.file.size_bytes == len
                || m.download.as_ref().is_some_and(|d| d.size_bytes == len)
                || m.weights
                    .as_ref()
                    .is_some_and(|w| w.file.size_bytes == len || w.upstream.size_bytes == len)
        });
        if !known_size {
            return Ok(None);
        }
        // Read once from the open handle (bounded by the pinned size) so the
        // exact bytes that are hashed are the bytes installed or parsed.
        let mut data = Vec::new();
        data.try_reserve_exact(len as usize)
            .map_err(|_| Error::out_of_memory())?;
        (&mut file)
            .take(len.saturating_add(1))
            .read_to_end(&mut data)
            .map_err(|e| Error::from_read(&e, path))?;
        if data.len() as u64 != len {
            return Err(download_failed("file changed while it was being imported"));
        }
        let digest = hex(&Sha256::digest(&data));
        let _guard = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        for entry in &self.manifest.models {
            let gate = || {
                if entry.restricted && !accept_restricted {
                    Err(terms_required(entry))
                } else {
                    Ok(())
                }
            };
            let matches = |f: &PinnedFile| f.size_bytes == len && f.sha256 == digest;
            if matches(&entry.file) {
                gate()?;
                self.write_verified_bytes(&data, &entry.file)?;
                return Ok(Some(entry.id.clone()));
            }
            if let Some(d) = entry.download.as_ref().filter(|d| matches(d)) {
                let _ = d;
                gate()?;
                self.install_compressed(Cursor::new(&data), &entry.file, &NoObserver)?;
                return Ok(Some(entry.id.clone()));
            }
            if let Some(w) = &entry.weights {
                if matches(&w.file) {
                    gate()?;
                    self.write_verified_bytes(&data, &w.file)?;
                    return Ok(Some(entry.id.clone()));
                }
                if w.upstream.size_bytes == len && w.upstream.sha256 == digest {
                    gate()?;
                    let table = tensor_table_for(entry)?;
                    convert_checkpoint_bytes(
                        &data,
                        &table,
                        &self.path_of(&w.file),
                        &crate::progress::NoProgress,
                    )?;
                    return Ok(Some(entry.id.clone()));
                }
            }
        }
        Ok(None)
    }

    /// Writes already-verified `bytes` to the installed location of `file`
    /// via an unpredictable `create_new` sibling temp file.
    fn write_verified_bytes(&self, bytes: &[u8], file: &PinnedFile) -> Result<()> {
        if bytes.len() as u64 != file.size_bytes || hex(&Sha256::digest(bytes)) != file.sha256 {
            return Err(download_failed("file changed while it was being imported"));
        }
        let dst = self.path_of(file);
        let (mut out, tmp) = create_temp_sibling(&dst).map_err(|e| Error::from_write(&e, &dst))?;
        out.write_all(bytes)
            .map_err(|e| Error::from_write(&e, &dst))?;
        out.sync_all().map_err(|e| Error::from_write(&e, &dst))?;
        drop(out);
        tmp.persist(&dst).map_err(|e| Error::from_write(&e, &dst))
    }

    /// Decompresses the verified zstd stream from `compressed` into the
    /// installed location of `target`, bounded by and checked against the
    /// pinned size and digest.
    fn install_compressed<R: Read>(
        &self,
        compressed: R,
        target: &PinnedFile,
        observer: &dyn DownloadObserver,
    ) -> Result<()> {
        let dst = self.path_of(target);
        let mut input = std::io::BufReader::with_capacity(1 << 16, compressed);
        let (out, tmp) = create_temp_sibling(&dst).map_err(|e| Error::from_write(&e, &dst))?;
        let mut out = std::io::BufWriter::with_capacity(1 << 16, out);
        let mut hasher = Sha256::new();
        let mut written = 0u64;
        {
            let mut decoder = ruzstd::decoding::StreamingDecoder::new(&mut input)
                .map_err(|e| download_failed(format!("zstd: {e}")))?;
            let mut buf = vec![0u8; 1 << 16];
            loop {
                if observer.is_cancelled() {
                    return Err(Error::cancelled());
                }
                let n = decoder
                    .read(&mut buf)
                    .map_err(|e| download_failed(format!("zstd: {e}")))?;
                if n == 0 {
                    break;
                }
                written += n as u64;
                // The bound: stop the moment the output is larger than
                // it is allowed to be. A decompression bomb gets no
                // further than one buffer past the pinned size.
                if written > target.size_bytes {
                    return Err(download_failed("decompressed data is larger than expected"));
                }
                hasher.update(&buf[..n]);
                out.write_all(&buf[..n])
                    .map_err(|e| Error::from_write(&e, &dst))?;
            }
        }
        // Nothing may follow the frame.
        let mut extra = [0u8; 1];
        if input
            .read(&mut extra)
            .map_err(|e| download_failed(format!("zstd: {e}")))?
            != 0
        {
            return Err(download_failed(
                "unexpected data after the compressed stream",
            ));
        }
        let file = out
            .into_inner()
            .map_err(|e| Error::from_write(&e.into_error(), &dst))?;
        file.sync_all().map_err(|e| Error::from_write(&e, &dst))?;
        drop(file);
        if written != target.size_bytes || hex(&hasher.finalize()) != target.sha256 {
            return Err(download_failed(
                "decompressed file does not match its checksum",
            ));
        }
        tmp.persist(&dst).map_err(|e| Error::from_write(&e, &dst))
    }

    fn fetch_project_file(
        &self,
        file: &PinnedFile,
        on_bytes: &mut dyn FnMut(u64),
        observer: &dyn DownloadObserver,
    ) -> Result<(File, TempFile)> {
        let (reader, length) = match &self.source {
            ModelSource::Directory(dir) => (open_local(&dir.join(&file.name))?, None),
            ModelSource::Release { base_url } => {
                let url = format!("{base_url}{}", file.name);
                http_get(&url, RELEASE_HOSTS, file.size_bytes)?
            }
        };
        let dst = self.path_of(file);
        let (mut out, tmp) = create_temp_sibling(&dst).map_err(|e| Error::from_write(&e, &dst))?;
        fetch_from_reader(reader, length, file, &mut out, &dst, on_bytes, observer)?;
        out.sync_all().map_err(|e| Error::from_write(&e, &dst))?;
        out.seek(SeekFrom::Start(0))
            .map_err(|e| Error::from_read(&e, tmp.path()))?;
        Ok((out, tmp))
    }

    fn fetch_upstream_bytes(
        &self,
        url: &str,
        file: &PinnedFile,
        on_bytes: &mut dyn FnMut(u64),
        observer: &dyn DownloadObserver,
    ) -> Result<Vec<u8>> {
        let (reader, length) = match &self.source {
            ModelSource::Directory(dir) => (open_local(&dir.join(&file.name))?, None),
            ModelSource::Release { .. } => http_get(url, UPSTREAM_HOSTS, file.size_bytes)?,
        };
        let mut out = Vec::new();
        out.try_reserve_exact(file.size_bytes as usize)
            .map_err(|_| Error::out_of_memory())?;
        let label = self.dir.join(&file.name);
        fetch_from_reader(reader, length, file, &mut out, &label, on_bytes, observer)?;
        Ok(out)
    }

    /// Removes leftover temporary files from interrupted installs. Call
    /// when no install is running (for example at start-up).
    pub fn clean_temp_files(&self) {
        let _guard = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                if e.file_name().to_string_lossy().ends_with(".part") {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
}

/// The file name an upstream checkpoint is stored under while it is being
/// processed: the last path component of its URL, if that is a safe name.
fn upstream_file_name(url: &str) -> Result<&str> {
    parse_https_url(url)
        .and_then(|(_, path)| path.rsplit('/').next())
        .filter(|n| is_safe_file_name(n))
        .ok_or_else(|| Error::internal("upstream URL has no usable file name"))
}

fn open_local(path: &Path) -> Result<Box<dyn Read>> {
    match File::open(path) {
        Ok(f) => Ok(Box::new(f)),
        Err(e) => Err(download_failed(format!("{}: {e}", path.display()))),
    }
}

/// Streams `reader` into `out`, enforcing the pinned size while the bytes
/// arrive and the pinned SHA-256 at the end.
fn fetch_from_reader<W: Write>(
    reader: Box<dyn Read>,
    declared_length: Option<u64>,
    file: &PinnedFile,
    out: &mut W,
    dest: &Path,
    on_bytes: &mut dyn FnMut(u64),
    observer: &dyn DownloadObserver,
) -> Result<()> {
    if let Some(len) = declared_length
        && len != file.size_bytes
    {
        return Err(download_failed(format!(
            "{}: server announced {len} bytes, expected {}",
            file.name, file.size_bytes
        )));
    }
    let mut reader = reader;
    let mut hasher = Sha256::new();
    let mut received = 0u64;
    let mut buf = vec![0u8; 1 << 16];
    let mut last_report = std::time::Instant::now();
    loop {
        if observer.is_cancelled() {
            return Err(Error::cancelled());
        }
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(download_failed(format!("{}: {e}", file.name))),
        };
        if n == 0 {
            break;
        }
        received += n as u64;
        // The size cap: never keep more than the manifest says.
        if received > file.size_bytes {
            return Err(download_failed(format!(
                "{}: larger than expected",
                file.name
            )));
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])
            .map_err(|e| Error::from_write(&e, dest))?;
        if last_report.elapsed() >= Duration::from_millis(150) {
            last_report = std::time::Instant::now();
            on_bytes(received);
        }
    }
    if received != file.size_bytes {
        return Err(download_failed(format!(
            "{}: received {received} of {} bytes",
            file.name, file.size_bytes
        )));
    }
    if hex(&hasher.finalize()) != file.sha256 {
        return Err(download_failed(format!("{}: SHA-256 mismatch", file.name)));
    }
    on_bytes(received);
    Ok(())
}

/// GETs `url` over HTTPS, following at most [`MAX_REDIRECTS`] redirects by
/// hand so every hop can be checked: HTTPS only, host on `allowed_hosts`.
/// Returns the body (hard-capped at `max_bytes`) and its announced length.
fn http_get(
    url: &str,
    allowed_hosts: &[&str],
    max_bytes: u64,
) -> Result<(Box<dyn Read>, Option<u64>)> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .max_redirects(0)
        .max_redirects_will_error(false)
        .http_status_as_error(false)
        .timeout_resolve(Some(Duration::from_secs(20)))
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .user_agent("SplitSound")
        .build()
        .into();
    let mut current = url.to_string();
    for _ in 0..=MAX_REDIRECTS {
        let Some((host, _)) = parse_https_url(&current) else {
            return Err(download_failed(format!(
                "refusing non-HTTPS or malformed URL {current}"
            )));
        };
        if !allowed_hosts.contains(&host) {
            return Err(download_failed(format!("refusing unexpected host {host}")));
        }
        let response = agent.get(&current).call().map_err(|e| map_ureq_error(&e))?;
        let status = response.status().as_u16();
        if (300..400).contains(&status) {
            let location = response
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| download_failed("redirect without a location"))?;
            // Only absolute HTTPS redirects are followed.
            current = location.to_string();
            continue;
        }
        if status != 200 {
            return Err(download_failed(format!("{current}: HTTP {status}")));
        }
        let length = response.body().content_length();
        let reader = response
            .into_body()
            .into_with_config()
            .limit(max_bytes.saturating_add(1))
            .reader();
        return Ok((Box::new(reader), length));
    }
    Err(download_failed("too many redirects"))
}

fn map_ureq_error(e: &ureq::Error) -> Error {
    match e {
        ureq::Error::HostNotFound
        | ureq::Error::ConnectionFailed
        | ureq::Error::Timeout(_)
        | ureq::Error::Io(_) => offline(e),
        other => download_failed(other),
    }
}

/// Checks which external data files an ONNX graph refers to.
///
/// ONNX stores an external tensor's file as a `location` key/value entry.
/// This scans the serialized graph for every such entry, wherever it is
/// nested, and requires each to name exactly `expected` (`None`: there
/// must be none at all, the model is self-contained). The graph files are
/// digest-pinned, so this can only fail if the project itself shipped a
/// bad graph; it is a second lock on the same door.
pub fn check_external_locations(
    graph: &[u8],
    expected: Option<&str>,
) -> std::result::Result<(), String> {
    // StringStringEntryProto { key = 1: "location", value = 2: <name> }
    const NEEDLE: &[u8] = b"\x0a\x08location\x12";
    let mut found = 0usize;
    let mut pos = 0usize;
    while let Some(i) = find(&graph[pos..], NEEDLE) {
        let at = pos + i + NEEDLE.len();
        // Value length: a one-byte varint (names are short).
        let Some(&len) = graph.get(at) else {
            return Err("truncated external data entry".into());
        };
        if len >= 0x80 {
            return Err("external data location is too long".into());
        }
        let Some(value) = graph.get(at + 1..at + 1 + usize::from(len)) else {
            return Err("truncated external data entry".into());
        };
        match expected {
            Some(name) if value == name.as_bytes() => found += 1,
            _ => {
                return Err(format!(
                    "graph refers to an unexpected external file {:?}",
                    String::from_utf8_lossy(value)
                ));
            }
        }
        pos = at;
    }
    if expected.is_some() && found == 0 {
        return Err("graph has no external data but a weights file was expected".into());
    }
    Ok(())
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::testing::sample_checkpoint;
    use crate::testutil::TempDir;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn sha(data: &[u8]) -> String {
        hex(&Sha256::digest(data))
    }

    /// A zstd frame holding `data` as raw (uncompressed) blocks.
    fn zstd_raw(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x28, 0xB5, 0x2F, 0xFD];
        // Frame header descriptor: single segment, 4-byte content size.
        out.push(0b1010_0000);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        let chunks: Vec<&[u8]> = if data.is_empty() {
            vec![&[][..]]
        } else {
            data.chunks(100_000).collect()
        };
        for (i, chunk) in chunks.iter().enumerate() {
            let last = u32::from(i + 1 == chunks.len());
            let header = last | ((chunk.len() as u32) << 3); // block type 0 = raw
            out.extend_from_slice(&header.to_le_bytes()[..3]);
            out.extend_from_slice(chunk);
        }
        out
    }

    /// A zstd bomb: `blocks` RLE blocks of 128 KiB each, 4 bytes apiece.
    fn zstd_bomb(blocks: usize) -> Vec<u8> {
        let mut out = vec![0x28, 0xB5, 0x2F, 0xFD];
        // No single-segment flag, no content size; window descriptor 1 MiB.
        out.push(0b0000_0000);
        out.push(0x50);
        for i in 0..blocks {
            let last = u32::from(i + 1 == blocks);
            let header = last | (1 << 1) | ((128 * 1024u32) << 3); // type 1 = RLE
            out.extend_from_slice(&header.to_le_bytes()[..3]);
            out.push(0x41);
        }
        out
    }

    const FAKE_ONNX: &[u8] =
        b"\x08\x07fake onnx graph bytes, self-contained, no external data......";

    fn manifest_for(graph: &[u8], compressed: &[u8], extra: &str) -> Manifest {
        Manifest::from_json(&format!(
            r#"{{"schemaVersion":2,"models":[{{"id":"m","kind":"tagger","revision":1,"name":"Test model","description":"d",
            "source":"GitHub","restricted":false,"terms":null,"license":"MIT","attribution":"a","origin":"https://example.org",
            "file":"m.onnx","sha256":"{}","sizeBytes":{},
            "acquire":{{"type":"download","file":"m.onnx.zst","sha256":"{}","sizeBytes":{}}},"opset":17,
            "sampleRate":32000,"channels":1,"windowSamples":320000,"hopSamples":160000,"numClasses":527,
            "dynamicBatch":false}}{extra}]}}"#,
            sha(graph),
            graph.len(),
            sha(compressed),
            compressed.len()
        ))
        .unwrap()
    }

    struct Setup {
        _tmp: TempDir,
        src: PathBuf,
        store: ModelStore,
    }

    /// A store whose "server" is a local folder containing `served`.
    fn setup(manifest: Manifest, served: &[(&str, &[u8])]) -> Setup {
        let tmp = TempDir::new("store");
        let src = tmp.path().join("server");
        std::fs::create_dir_all(&src).unwrap();
        for (name, bytes) in served {
            std::fs::write(src.join(name), bytes).unwrap();
        }
        let store = ModelStore::with_source(
            tmp.path().join("models"),
            manifest,
            ModelSource::Directory(src.clone()),
        )
        .unwrap();
        Setup {
            _tmp: tmp,
            src,
            store,
        }
    }

    fn files_in(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn download_decompress_verify_and_load() {
        let z = zstd_raw(FAKE_ONNX);
        let s = setup(manifest_for(FAKE_ONNX, &z, ""), &[("m.onnx.zst", &z)]);
        assert!(!s.store.is_installed("m"));
        let st = s.store.status("m").unwrap();
        assert!(!st.installed && st.size_bytes == FAKE_ONNX.len() as u64);
        assert_eq!(st.download_bytes, z.len() as u64);
        assert_eq!(
            s.store.load("m").unwrap_err().code(),
            ErrorCode::ModelMissing
        );

        struct Seen(AtomicU64, AtomicU64);
        impl DownloadObserver for Seen {
            fn progress(&self, id: &str, received: u64, total: u64) {
                assert_eq!(id, "m");
                assert!(received <= total);
                assert!(
                    received >= self.0.load(Ordering::SeqCst),
                    "progress went backwards"
                );
                self.0.store(received, Ordering::SeqCst);
                self.1.store(total, Ordering::SeqCst);
            }
        }
        let seen = Seen(AtomicU64::new(0), AtomicU64::new(0));
        s.store.install("m", false, &seen).unwrap();
        assert_eq!(seen.0.load(Ordering::SeqCst), z.len() as u64);
        assert_eq!(seen.1.load(Ordering::SeqCst), z.len() as u64);
        assert!(s.store.is_installed("m"));
        // Only the installed file remains: no archive, no temp files.
        assert_eq!(files_in(s.store.dir()), vec!["m.onnx"]);
        let loaded = s.store.load("m").unwrap();
        assert_eq!(loaded.graph, FAKE_ONNX);
        assert!(loaded.external.is_none());
        s.store.verify("m").unwrap();
        // Installing again is a no-op even if the server is gone.
        std::fs::remove_file(s.src.join("m.onnx.zst")).unwrap();
        s.store.install("m", false, &NoObserver).unwrap();
        // Remove, twice.
        s.store.remove("m").unwrap();
        s.store.remove("m").unwrap();
        assert!(!s.store.is_installed("m"));
        assert_eq!(
            s.store.install("m", false, &NoObserver).unwrap_err().code(),
            ErrorCode::ModelDownloadFailed
        );
        assert_eq!(
            s.store.status("zzz").unwrap_err().code(),
            ErrorCode::NotFound
        );
    }

    #[test]
    fn digest_is_rechecked_at_load() {
        let z = zstd_raw(FAKE_ONNX);
        let s = setup(manifest_for(FAKE_ONNX, &z, ""), &[("m.onnx.zst", &z)]);
        s.store.install("m", false, &NoObserver).unwrap();
        let path = s.store.dir().join("m.onnx");
        // Same size, one byte changed.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[10] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            s.store.is_installed("m"),
            "the cheap check only looks at size"
        );
        let e = s.store.load("m").unwrap_err();
        assert_eq!(e.code(), ErrorCode::ModelMissing);
        assert!(e.message().contains("damaged"));
        assert!(s.store.verify("m").is_err());
        // A different size is not even "installed".
        std::fs::write(&path, &bytes[..20]).unwrap();
        assert!(!s.store.is_installed("m"));
    }

    fn expect_install_failure(manifest: Manifest, served: &[(&str, &[u8])], detail_has: &str) {
        let s = setup(manifest, served);
        let e = s.store.install("m", false, &NoObserver).unwrap_err();
        assert_eq!(e.code(), ErrorCode::ModelDownloadFailed, "{e}");
        assert!(
            e.detail().unwrap_or("").contains(detail_has),
            "detail {:?} does not mention {detail_has:?}",
            e.detail()
        );
        assert!(!s.store.is_installed("m"));
        assert!(
            files_in(s.store.dir()).is_empty(),
            "left behind: {:?}",
            files_in(s.store.dir())
        );
    }

    #[test]
    fn wrong_hash_is_rejected() {
        let z = zstd_raw(FAKE_ONNX);
        let mut tampered = z.clone();
        let n = tampered.len();
        tampered[n - 3] ^= 0x20;
        expect_install_failure(
            manifest_for(FAKE_ONNX, &z, ""),
            &[("m.onnx.zst", &tampered)],
            "SHA-256 mismatch",
        );
    }

    #[test]
    fn oversized_download_is_cut_off() {
        let z = zstd_raw(FAKE_ONNX);
        let mut big = z.clone();
        big.extend(std::iter::repeat_n(0u8, 5_000_000));
        expect_install_failure(
            manifest_for(FAKE_ONNX, &z, ""),
            &[("m.onnx.zst", &big)],
            "larger than expected",
        );
    }

    #[test]
    fn truncated_download_is_rejected() {
        let z = zstd_raw(FAKE_ONNX);
        expect_install_failure(
            manifest_for(FAKE_ONNX, &z, ""),
            &[("m.onnx.zst", &z[..z.len() - 7])],
            "received",
        );
        expect_install_failure(
            manifest_for(FAKE_ONNX, &z, ""),
            &[("m.onnx.zst", b"")],
            "received",
        );
    }

    #[test]
    fn zstd_bomb_is_stopped_at_the_pinned_size() {
        // 4 kB that would inflate to 128 MiB; the manifest pins the
        // archive's own digest, so only the output bound can stop it.
        let bomb = zstd_bomb(1000);
        assert!(bomb.len() < 5000);
        let t0 = std::time::Instant::now();
        expect_install_failure(
            manifest_for(FAKE_ONNX, &bomb, ""),
            &[("m.onnx.zst", &bomb)],
            "larger than expected",
        );
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "the bomb was allowed to run"
        );
    }

    #[test]
    fn archive_with_wrong_or_extra_content_is_rejected() {
        // Valid archive, pinned archive digest, but it unpacks to
        // something that is not the pinned graph.
        let mut different = FAKE_ONNX.to_vec();
        different[5] ^= 0x40;
        let other = zstd_raw(&different);
        expect_install_failure(
            manifest_for(FAKE_ONNX, &other, ""),
            &[("m.onnx.zst", &other)],
            "does not match",
        );
        // Shorter than pinned.
        let short = zstd_raw(&FAKE_ONNX[..10]);
        expect_install_failure(
            manifest_for(FAKE_ONNX, &short, ""),
            &[("m.onnx.zst", &short)],
            "does not match",
        );
        // Garbage after the frame.
        let mut trailing = zstd_raw(FAKE_ONNX);
        trailing.extend_from_slice(b"extra");
        expect_install_failure(
            manifest_for(FAKE_ONNX, &trailing, ""),
            &[("m.onnx.zst", &trailing)],
            "after the compressed stream",
        );
        // Not zstd at all.
        let junk = b"PK\x03\x04 this is not a zstd frame".to_vec();
        expect_install_failure(
            manifest_for(FAKE_ONNX, &junk, ""),
            &[("m.onnx.zst", &junk)],
            "zstd",
        );
    }

    #[test]
    fn cancellation_leaves_nothing_behind() {
        struct Stop;
        impl DownloadObserver for Stop {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let z = zstd_raw(FAKE_ONNX);
        let s = setup(manifest_for(FAKE_ONNX, &z, ""), &[("m.onnx.zst", &z)]);
        let e = s.store.install("m", false, &Stop).unwrap_err();
        assert_eq!(e.code(), ErrorCode::Cancelled);
        assert!(files_in(s.store.dir()).is_empty());
    }

    #[test]
    fn hand_imported_files_must_match_a_pinned_digest() {
        let z = zstd_raw(FAKE_ONNX);
        let s = setup(manifest_for(FAKE_ONNX, &z, ""), &[]);
        let dir = TempDir::new("import");
        // The plain graph.
        let plain = dir.path().join("whatever-name.bin");
        std::fs::write(&plain, FAKE_ONNX).unwrap();
        assert_eq!(
            s.store.import_file(&plain, false).unwrap().as_deref(),
            Some("m")
        );
        assert!(s.store.is_installed("m"));
        assert_eq!(s.store.load("m").unwrap().graph, FAKE_ONNX);
        s.store.remove("m").unwrap();
        // The compressed download.
        let packed = dir.path().join("download.zst");
        std::fs::write(&packed, &z).unwrap();
        assert_eq!(
            s.store.import_file(&packed, false).unwrap().as_deref(),
            Some("m")
        );
        assert_eq!(files_in(s.store.dir()), vec!["m.onnx"]);
        s.store.remove("m").unwrap();
        // Same size as the graph, different content: not recognised.
        let mut wrong = FAKE_ONNX.to_vec();
        wrong[0] ^= 1;
        let p = dir.path().join("m.onnx");
        std::fs::write(&p, &wrong).unwrap();
        assert_eq!(s.store.import_file(&p, false).unwrap(), None);
        // Unrelated files, a directory, a missing file.
        let p = dir.path().join("song.mp3");
        std::fs::write(&p, b"not a model").unwrap();
        assert_eq!(s.store.import_file(&p, false).unwrap(), None);
        assert_eq!(s.store.import_file(dir.path(), false).unwrap(), None);
        assert_eq!(
            s.store
                .import_file(&dir.path().join("nope"), false)
                .unwrap_err()
                .code(),
            ErrorCode::FileNotFound
        );
        assert!(files_in(s.store.dir()).is_empty());
    }

    /// A manifest with a restricted, converted model built from the
    /// synthetic checkpoint. Returns (manifest, graph, graph.zst, checkpoint, weights).
    fn restricted_setup() -> (String, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let (checkpoint, weights, table_json) = sample_checkpoint();
        // A "graph" that refers to its weights file the way ONNX does.
        let mut graph = b"\x08\x07graph".to_vec();
        graph.extend_from_slice(b"\x0a\x08location\x12\x09r.weights");
        graph.extend_from_slice(b" more graph bytes");
        let z = zstd_raw(&graph);
        let json = format!(
            r#"{{"schemaVersion":2,"models":[{{"id":"r","kind":"tagger","revision":1,"name":"Restricted model","description":"d",
            "source":"Meta","restricted":true,"terms":"Research use only.","license":"Research only","attribution":"a",
            "origin":"https://example.org","file":"r.onnx","sha256":"{}","sizeBytes":{},
            "acquire":{{"type":"download","file":"r.onnx.zst","sha256":"{}","sizeBytes":{}}},"opset":17,
            "sampleRate":32000,"channels":1,"windowSamples":320000,"hopSamples":160000,"numClasses":527,
            "dynamicBatch":false,
            "weights":{{"file":"r.weights","sha256":"{}","sizeBytes":{},
                "tensors":{{"file":"r.tensors.json","sha256":"{}","sizeBytes":{}}},
                "upstream":{{"url":"https://dl.fbaipublicfiles.com/demucs/test-checkpoint.th","sha256":"{}","sizeBytes":{},"format":"torch-zip"}}}}}}]}}"#,
            sha(&graph),
            graph.len(),
            sha(&z),
            z.len(),
            sha(&weights),
            weights.len(),
            sha(table_json.as_bytes()),
            table_json.len(),
            sha(&checkpoint),
            checkpoint.len(),
        );
        (json, graph, z, checkpoint, weights)
    }

    #[test]
    fn restricted_models_need_the_accept_flag() {
        let (json, _graph, z, checkpoint, _weights) = restricted_setup();
        let s = setup(
            Manifest::from_json(&json).unwrap(),
            &[("r.onnx.zst", &z), ("test-checkpoint.th", &checkpoint)],
        );
        let st = s.store.status("r").unwrap();
        assert!(st.restricted);
        assert_eq!(st.terms.as_deref(), Some("Research use only."));
        assert_eq!(st.source, "Meta");
        assert_eq!(st.download_bytes, (z.len() + checkpoint.len()) as u64);

        // Without the flag: refused, and nothing at all was fetched.
        let e = s.store.install("r", false, &NoObserver).unwrap_err();
        assert_eq!(e.code(), ErrorCode::TermsRequired);
        assert!(e.message().contains("Restricted model"));
        assert!(files_in(s.store.dir()).is_empty());
        let e = s.store.ensure(&["r"], false, &NoObserver).unwrap_err();
        assert_eq!(e.code(), ErrorCode::TermsRequired);
        assert!(files_in(s.store.dir()).is_empty());

        // Importing the restricted files by hand is gated the same way.
        let dir = TempDir::new("import-r");
        let ck = dir.path().join("ck.th");
        std::fs::write(&ck, &checkpoint).unwrap();
        assert_eq!(
            s.store.import_file(&ck, false).unwrap_err().code(),
            ErrorCode::TermsRequired
        );
        assert!(files_in(s.store.dir()).is_empty());
    }

    #[test]
    fn converted_model_installs_only_the_graph_and_weights() {
        // The synthetic tensor table is not the compiled-in one, so the
        // full install path cannot run here; exercise its two halves:
        // the upstream fetch rules, and loading with external data.
        let (json, graph, z, checkpoint, weights) = restricted_setup();
        let s = setup(
            Manifest::from_json(&json).unwrap(),
            &[("r.onnx.zst", &z), ("test-checkpoint.th", &checkpoint)],
        );
        let entry = s.store.manifest().get("r").unwrap().clone();
        let w = entry.weights.as_ref().unwrap();
        assert_eq!(
            upstream_file_name(&w.upstream.url).unwrap(),
            "test-checkpoint.th"
        );

        // Upstream fetch: pinned size and digest are enforced in memory (never written raw to disk).
        let pinned = PinnedFile {
            name: "test-checkpoint.th".into(),
            sha256: w.upstream.sha256.clone(),
            size_bytes: w.upstream.size_bytes,
        };
        let fetched = s
            .store
            .fetch_upstream_bytes(&w.upstream.url, &pinned, &mut |_| {}, &NoObserver)
            .unwrap();
        assert_eq!(fetched, checkpoint);
        let mut bad = checkpoint.clone();
        bad[100] ^= 1;
        std::fs::write(s.src.join("test-checkpoint.th"), &bad).unwrap();
        let e = s
            .store
            .fetch_upstream_bytes(&w.upstream.url, &pinned, &mut |_| {}, &NoObserver)
            .unwrap_err();
        assert_eq!(e.code(), ErrorCode::ModelDownloadFailed);
        assert!(files_in(s.store.dir()).is_empty());

        // Put the two installed files in place and load them.
        std::fs::write(s.store.dir().join("r.onnx"), &graph).unwrap();
        assert!(!s.store.is_installed("r"), "weights still missing");
        std::fs::write(s.store.dir().join("r.weights"), &weights).unwrap();
        assert!(s.store.is_installed("r"));
        let loaded = s.store.load("r").unwrap();
        assert_eq!(loaded.graph, graph);
        let (name, bytes) = loaded.external.unwrap();
        assert_eq!(name, "r.weights");
        assert_eq!(&bytes[..], &weights[..]);

        // Tampered weights are caught at load.
        let mut bad = weights.clone();
        bad[3] ^= 0x80;
        std::fs::write(s.store.dir().join("r.weights"), &bad).unwrap();
        assert_eq!(
            s.store.load("r").unwrap_err().code(),
            ErrorCode::ModelMissing
        );
        s.store.remove("r").unwrap();
        assert!(files_in(s.store.dir()).is_empty());
    }

    #[test]
    fn external_data_locations_are_checked() {
        let entry = |name: &str| {
            let mut v = b"\x0a\x08location\x12".to_vec();
            v.push(name.len() as u8);
            v.extend_from_slice(name.as_bytes());
            v
        };
        let graph = |names: &[&str]| {
            let mut g = b"header".to_vec();
            for n in names {
                g.extend_from_slice(&entry(n));
                g.extend_from_slice(b"..padding..");
            }
            g
        };
        assert!(check_external_locations(&graph(&[]), None).is_ok());
        assert!(
            check_external_locations(&graph(&["m.weights", "m.weights"]), Some("m.weights"))
                .is_ok()
        );
        // A self-contained model must not reference any file.
        assert!(check_external_locations(&graph(&["m.weights"]), None).is_err());
        // Expected weights but none referenced.
        assert!(check_external_locations(&graph(&[]), Some("m.weights")).is_err());
        // Any other file, by any path.
        for evil in [
            "other.weights",
            "../m.weights",
            "/etc/passwd",
            "m.weights/../x",
            "C:\\x",
            "",
        ] {
            assert!(
                check_external_locations(&graph(&["m.weights", evil]), Some("m.weights")).is_err(),
                "{evil:?}"
            );
        }
        // Truncated entry.
        let mut g = graph(&["m.weights"]);
        g.truncate(g.len() - 15);
        assert!(check_external_locations(&g, Some("m.weights")).is_err());
    }

    #[test]
    fn base_url_must_be_https() {
        let tmp = TempDir::new("store-url");
        let m = || Manifest::from_json(r#"{"schemaVersion":2,"models":[]}"#).unwrap();
        for bad in [
            "http://example.com/models/",
            "https://example.com/models",
            "file:///tmp/",
            "ftp://x/",
        ] {
            let r = ModelStore::with_source(
                tmp.path(),
                m(),
                ModelSource::Release {
                    base_url: bad.into(),
                },
            );
            assert!(r.is_err(), "{bad}");
        }
        assert!(ModelStore::with_source(tmp.path(), m(), ModelSource::default()).is_ok());
        assert!(
            DEFAULT_BASE_URL.starts_with(
                "https://github.com/Omaaarkhan/splitsound/releases/download/models-v1/"
            )
        );
    }

    #[test]
    fn redirect_and_host_rules() {
        // No network is used: these fail before any connection is made.
        let e = http_get("http://github.com/x", RELEASE_HOSTS, 10)
            .err()
            .unwrap();
        assert_eq!(e.code(), ErrorCode::ModelDownloadFailed);
        let e = http_get("https://evil.example/x", RELEASE_HOSTS, 10)
            .err()
            .unwrap();
        assert!(e.detail().unwrap().contains("unexpected host"));
        let e = http_get("https://github.com/x", UPSTREAM_HOSTS, 10)
            .err()
            .unwrap();
        assert!(e.detail().unwrap().contains("unexpected host"));
    }

    #[test]
    fn stale_temp_files_can_be_cleaned() {
        let z = zstd_raw(FAKE_ONNX);
        let s = setup(manifest_for(FAKE_ONNX, &z, ""), &[("m.onnx.zst", &z)]);
        std::fs::write(s.store.dir().join("m.onnx.zst.123.part"), b"junk").unwrap();
        s.store.install("m", false, &NoObserver).unwrap();
        s.store.clean_temp_files();
        assert_eq!(files_in(s.store.dir()), vec!["m.onnx"]);
    }

    #[cfg(unix)]
    #[test]
    fn install_does_not_follow_or_remove_preexisting_symlinks() {
        let z = zstd_raw(FAKE_ONNX);
        let s = setup(manifest_for(FAKE_ONNX, &z, ""), &[("m.onnx.zst", &z)]);
        let outside = TempDir::new("store-symlink-target");
        let target = outside.path().join("keep.txt");
        std::fs::write(&target, b"keep this").unwrap();
        let pid = std::process::id();
        let link_zst = s.store.dir().join(format!("m.onnx.zst.{pid}.part"));
        let link_onnx = s.store.dir().join(format!("m.onnx.{pid}.part"));
        std::os::unix::fs::symlink(&target, &link_zst).unwrap();
        std::os::unix::fs::symlink(&target, &link_onnx).unwrap();

        s.store.install("m", false, &NoObserver).unwrap();
        assert_eq!(s.store.load("m").unwrap().graph, FAKE_ONNX);
        assert_eq!(std::fs::read(&target).unwrap(), b"keep this");
        assert_eq!(std::fs::read_link(&link_zst).unwrap(), target);
        assert_eq!(std::fs::read_link(&link_onnx).unwrap(), target);
    }
}
