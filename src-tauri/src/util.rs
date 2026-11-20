//! Small shared helpers: session ids, cancellation, timestamps, atomic file
//! writes, file-name sanitising and mutex poisoning.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(any(windows, test))]
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

/// Locks a mutex, recovering the data if a panicking thread poisoned it.
/// Jobs run behind a panic boundary, so a poisoned lock must not take the
/// rest of the app down with it.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn read_lock<T>(m: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    m.read().unwrap_or_else(|e| e.into_inner())
}

pub fn write_lock<T>(m: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    m.write().unwrap_or_else(|e| e.into_inner())
}

#[cfg(windows)]
const WINDOWS_RETRY_DELAYS_MS: &[u64] = &[20, 40, 80, 120, 180, 250, 310];

#[cfg(any(windows, test))]
fn is_transient_windows_lock(err: &io::Error) -> bool {
    err.kind() == io::ErrorKind::PermissionDenied || matches!(err.raw_os_error(), Some(5 | 32 | 33))
}

#[cfg(any(windows, test))]
fn retry_transient<T>(delays_ms: &[u64], mut op: impl FnMut() -> io::Result<T>) -> io::Result<T> {
    let mut attempt = 0;
    loop {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) if attempt < delays_ms.len() && is_transient_windows_lock(&e) => {
                std::thread::sleep(Duration::from_millis(delays_ms[attempt]));
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Retries a filesystem mutation over ~1 s on Windows when antivirus or the
/// search indexer transiently holds a sharing lock on a new file or folder.
/// Executes once on Unix.
pub fn retry_on_windows<T>(op: impl FnMut() -> io::Result<T>) -> io::Result<T> {
    #[cfg(windows)]
    {
        retry_transient(WINDOWS_RETRY_DELAYS_MS, op)
    }
    #[cfg(not(windows))]
    {
        let mut op = op;
        op()
    }
}

pub fn rename_retry(from: &Path, to: &Path) -> io::Result<()> {
    retry_on_windows(|| fs::rename(from, to))
}

pub fn remove_file_retry(path: &Path) -> io::Result<()> {
    retry_on_windows(|| fs::remove_file(path))
}

pub fn remove_dir_all_retry(path: &Path) -> io::Result<()> {
    retry_on_windows(|| fs::remove_dir_all(path))
}

/* ------------------------------------------------------------------ */
/* Session ids                                                         */
/* ------------------------------------------------------------------ */

/// Length of a session id: 32 lowercase hex digits (a v4 UUID without dashes).
pub const SESSION_ID_LEN: usize = 32;

/// Creates a new random session id.
pub fn new_session_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// True when `id` matches `^[0-9a-f]{32}$`. Every command checks this
/// before an id is used to build a path.
pub fn is_valid_session_id(id: &str) -> bool {
    id.len() == SESSION_ID_LEN && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/* ------------------------------------------------------------------ */
/* Cancellation                                                        */
/* ------------------------------------------------------------------ */

/// Cooperative cancellation flag shared between the shell and the engine.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// True if both handles control the same flag.
    pub fn same(&self, other: &CancelToken) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/* ------------------------------------------------------------------ */
/* Time                                                                */
/* ------------------------------------------------------------------ */

/// Formats a time as ISO 8601 UTC with milliseconds: `2026-11-19T19:30:12.345Z`.
pub fn iso8601(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let ms = d.subsec_millis();
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (y, m, day) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{day:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// Now, as ISO 8601.
pub fn now_iso8601() -> String {
    iso8601(SystemTime::now())
}

/// Days since 1970-01-01 to (year, month, day). Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/* ------------------------------------------------------------------ */
/* Files                                                               */
/* ------------------------------------------------------------------ */

/// Writes `bytes` to `path` atomically: a temporary file in the same folder
/// is written, flushed to disk and renamed over the target, so readers see
/// either the old or the new content, never a torn file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let tmp = dir.join(format!(
        ".{}.tmp-{}",
        name.to_string_lossy(),
        &new_session_id()[..8]
    ));
    let mut f = fs::File::create_new(&tmp)?;
    let result = (|| {
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        rename_retry(&tmp, path)?;
        // Make the rename itself durable where the platform allows it.
        #[cfg(unix)]
        if let Ok(d) = fs::File::open(dir) {
            let _ = d.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = remove_file_retry(&tmp);
    }
    result
}

/// True for temporary files made by [`write_atomic`].
pub fn is_atomic_temp(name: &str) -> bool {
    name.starts_with('.') && name.contains(".tmp-")
}

/// Total size of the regular files under `dir` (symlinks are not followed).
pub fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for entry in rd.flatten() {
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

/// Reads a file only if it is a regular file no larger than `max` bytes.
pub fn read_bounded(path: &Path, max: u64) -> io::Result<Vec<u8>> {
    // Check type before opening so directories on Windows return InvalidInput
    // rather than PermissionDenied (os error 5).
    let meta = fs::metadata(path)?;
    if !meta.is_file() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a file"));
    }
    if meta.len() > max {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file too large"));
    }
    let file = fs::File::open(path)?;
    let opened_meta = file.metadata()?;
    if !opened_meta.is_file() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a file"));
    }
    if opened_meta.len() > max {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file too large"));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(opened_meta.len() as usize)
        .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "out of memory"))?;
    file.take(max.saturating_add(1)).read_to_end(&mut data)?;
    if data.len() as u64 > max {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file too large"));
    }
    Ok(data)
}

/* ------------------------------------------------------------------ */
/* File names                                                          */
/* ------------------------------------------------------------------ */

/// Longest file stem we produce, in bytes. Leaves room for " (Instrumental)",
/// a counter and an extension inside the 255-byte limit of every file system.
const MAX_STEM_BYTES: usize = 180;

/// Turns arbitrary text (usually a song title from tags) into a file-name
/// stem that is valid on macOS, Windows and Linux:
///
/// - path separators, `: * ? " < > |`, control characters and bidi
///   overrides (which can disguise an extension) are removed or replaced;
/// - leading dots (hidden files) and trailing dots/spaces (invalid on
///   Windows) are trimmed, runs of whitespace collapse;
/// - Windows device names (`CON`, `NUL`, `COM1`, … also with an extension)
///   get a leading underscore;
/// - the result is at most 180 bytes and never empty ("Untitled").
pub fn sanitize_file_stem(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_space = false;
    for c in input.chars() {
        let mapped = match c {
            '/' | '\\' | ':' | '|' => Some('-'),
            '*' | '?' | '"' | '<' | '>' => None,
            c if c.is_control() => Some(' '),
            // Bidirectional overrides and isolates, zero-width characters.
            '\u{202A}'..='\u{202E}'
            | '\u{2066}'..='\u{2069}'
            | '\u{200B}'..='\u{200F}'
            | '\u{FEFF}' => None,
            c if c.is_whitespace() => Some(' '),
            c => Some(c),
        };
        if let Some(c) = mapped {
            if c == ' ' {
                if last_space || out.is_empty() {
                    continue;
                }
                last_space = true;
            } else {
                last_space = false;
            }
            out.push(c);
        }
    }
    let mut s = out.trim_start_matches(['.', ' ']).to_string();
    truncate_utf8(&mut s, MAX_STEM_BYTES);
    let s = s.trim_end_matches(['.', ' ']).to_string();
    let s = if s.is_empty() {
        "Untitled".to_string()
    } else {
        s
    };
    if is_windows_reserved(&s) {
        format!("_{s}")
    } else {
        s
    }
}

fn truncate_utf8(s: &mut String, max: usize) {
    if s.len() <= max {
        return;
    }
    let mut cut = max;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s.truncate(cut);
}

/// Windows treats these names as devices, with or without an extension.
fn is_windows_reserved(stem: &str) -> bool {
    let base = stem.split('.').next().unwrap_or("").trim_end_matches(' ');
    let upper = base.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ((upper.starts_with("COM") || upper.starts_with("LPT"))
        && base.chars().count() == 4
        && base
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, '¹' | '²' | '³')))
}

/// Returns `dir/stem.ext`, or `dir/stem 2.ext`, `dir/stem 3.ext`, … for the
/// first name that does not exist yet. Never overwrites.
pub fn unique_path(dir: &Path, stem: &str, ext: &str, taken: &[PathBuf]) -> io::Result<PathBuf> {
    for n in 1u32.. {
        let name = if n == 1 {
            format!("{stem}.{ext}")
        } else {
            format!("{stem} {n}.{ext}")
        };
        let p = dir.join(name);
        if taken.contains(&p) {
            continue;
        }
        match p.symlink_metadata() {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(p),
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::other("no available file name"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_are_strict() {
        let id = new_session_id();
        assert!(is_valid_session_id(&id), "{id}");
        assert!(!is_valid_session_id(""));
        assert!(!is_valid_session_id(&id.to_uppercase()));
        assert!(!is_valid_session_id("../../../../etc/passwd0000000000"));
        assert!(!is_valid_session_id(&format!("{}/", &id[..31])));
        assert!(!is_valid_session_id(&format!("{id}0")));
    }

    #[test]
    fn iso8601_formats_known_instants() {
        use std::time::Duration;
        assert_eq!(iso8601(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        let t = UNIX_EPOCH + Duration::from_millis(1_795_635_012_345);
        assert_eq!(iso8601(t), "2026-11-25T19:30:12.345Z");
        let leap = UNIX_EPOCH + Duration::from_secs(951_782_400); // 2000-02-29
        assert_eq!(iso8601(leap), "2000-02-29T00:00:00.000Z");
    }

    #[test]
    fn sanitises_file_names_for_every_os() {
        assert_eq!(
            sanitize_file_stem("Song title (Vocals)"),
            "Song title (Vocals)"
        );
        assert_eq!(
            sanitize_file_stem("AC/DC: Back in Black?"),
            "AC-DC- Back in Black"
        );
        assert_eq!(sanitize_file_stem("..hidden"), "hidden");
        assert_eq!(sanitize_file_stem("trailing. . "), "trailing");
        assert_eq!(sanitize_file_stem("a\tb\n\nc"), "a b c");
        assert_eq!(sanitize_file_stem(""), "Untitled");
        assert_eq!(sanitize_file_stem("???"), "Untitled");
        assert_eq!(sanitize_file_stem("con"), "_con");
        assert_eq!(sanitize_file_stem("NUL.song"), "_NUL.song");
        assert_eq!(sanitize_file_stem("COM1"), "_COM1");
        assert_eq!(sanitize_file_stem("lpt9 "), "_lpt9");
        assert_eq!(sanitize_file_stem("Con (Vocals)"), "Con (Vocals)");
        assert_eq!(sanitize_file_stem("Company"), "Company");
        assert_eq!(sanitize_file_stem("evil\u{202E}vaw.exe"), "evilvaw.exe");
        let long = "é".repeat(200);
        let s = sanitize_file_stem(&long);
        assert!(s.len() <= 180 && s.chars().all(|c| c == 'é'));
    }

    #[test]
    fn unique_path_never_overwrites() {
        let dir = std::env::temp_dir().join(format!("ss-test-{}", new_session_id()));
        fs::create_dir_all(&dir).unwrap();
        let a = unique_path(&dir, "x", "wav", &[]).unwrap();
        assert_eq!(a, dir.join("x.wav"));
        fs::write(&a, b"1").unwrap();
        let b = unique_path(&dir, "x", "wav", &[]).unwrap();
        assert_eq!(b, dir.join("x 2.wav"));
        let c = unique_path(&dir, "x", "wav", std::slice::from_ref(&b)).unwrap();
        assert_eq!(c, dir.join("x 3.wav"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn atomic_write_replaces_content() {
        let dir = std::env::temp_dir().join(format!("ss-test-{}", new_session_id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.json");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"two");
        let names: Vec<_> = fs::read_dir(&dir).unwrap().flatten().collect();
        assert_eq!(names.len(), 1, "no temp files left behind");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_bounded_rejects_directories_and_oversized_files() {
        let dir = std::env::temp_dir().join(format!("ss-test-{}", new_session_id()));
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            read_bounded(&dir, 1024).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        let p = dir.join("big.bin");
        fs::write(&p, b"0123456789").unwrap();
        assert_eq!(
            read_bounded(&p, 4).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(read_bounded(&p, 10).unwrap(), b"0123456789");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn transient_retry_retries_permission_denied_only() {
        let mut calls = 0;
        let v = retry_transient(&[0, 0, 0], || {
            calls += 1;
            if calls < 3 {
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            } else {
                Ok("ok")
            }
        })
        .unwrap();
        assert_eq!(v, "ok");
        assert_eq!(calls, 3);

        let mut nf_calls = 0;
        let err = retry_transient(&[0, 0, 0], || -> io::Result<()> {
            nf_calls += 1;
            Err(io::Error::from(io::ErrorKind::NotFound))
        })
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert_eq!(nf_calls, 1);
    }
}
