//! Finding the bundled ONNX Runtime shared library.
//!
//! `scripts/fetch-onnxruntime.mjs` places Microsoft's official build at
//! `src-tauri/resources/onnxruntime/<name>`; the bundler ships it:
//!
//! | OS      | Bundled at                                         |
//! |---------|----------------------------------------------------|
//! | macOS   | `SplitSound.app/Contents/Frameworks/libonnxruntime.dylib` (code-signed with the app) |
//! | Windows | `<install dir>\onnxruntime\onnxruntime.dll`        |
//! | Linux   | `<resource dir>/onnxruntime/libonnxruntime.so` (`/usr/lib/SplitSound/…` for deb/rpm, inside the AppImage) |
//!
//! The engine must load the library from the absolute path returned here,
//! never by bare name: Windows ships its own, older `onnxruntime.dll` in
//! System32 that a by-name load could pick up.
//!
//! In development builds the source location is also tried, so `tauri dev`
//! and `cargo run` work once the library has been fetched. A file that is
//! missing, empty (the build script's development placeholder) or not a
//! shared library of this platform is ignored.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Canonical file name of the library on this platform.
pub const LIB_NAME: &str = if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else if cfg!(target_os = "windows") {
    "onnxruntime.dll"
} else {
    "libonnxruntime.so"
};

/// Smallest plausible library (the real one is tens of MB).
const MIN_SIZE: u64 = 64 * 1024;

/// Places to look, most specific first. `exe` is the running executable,
/// `resource_dir` Tauri's resource folder.
pub fn candidates(exe: Option<&Path>, resource_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if cfg!(target_os = "macos")
        && let Some(macos_dir) = exe.and_then(Path::parent)
    {
        // SplitSound.app/Contents/MacOS/<exe> -> Contents/Frameworks
        v.push(macos_dir.join("..").join("Frameworks").join(LIB_NAME));
    }
    if let Some(r) = resource_dir {
        v.push(r.join("onnxruntime").join(LIB_NAME));
    }
    if cfg!(debug_assertions) {
        v.push(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("resources")
                .join("onnxruntime")
                .join(LIB_NAME),
        );
    }
    v
}

/// True if `path` is a non-trivial shared library for this platform
/// (Mach-O, PE or ELF by magic number).
pub fn looks_like_library(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() || meta.len() < MIN_SIZE {
        return false;
    }
    let mut magic = [0u8; 4];
    if File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_err()
    {
        return false;
    }
    if cfg!(target_os = "macos") {
        // 64-bit Mach-O (either byte order) or a universal ("fat") binary.
        matches!(
            magic,
            [0xCF, 0xFA, 0xED, 0xFE]
                | [0xFE, 0xED, 0xFA, 0xCF]
                | [0xCA, 0xFE, 0xBA, 0xBE]
                | [0xBE, 0xBA, 0xFE, 0xCA]
        )
    } else if cfg!(target_os = "windows") {
        magic[0..2] == *b"MZ"
    } else {
        magic == [0x7F, b'E', b'L', b'F']
    }
}

/// Absolute path of the bundled library, if a valid one is found.
pub fn resolve(exe: Option<&Path>, resource_dir: Option<&Path>) -> Option<PathBuf> {
    candidates(exe, resource_dir)
        .into_iter()
        .find(|p| looks_like_library(p))
        .and_then(|p| crate::app::dunce_canonicalize(&p).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_placeholders_and_wrong_formats() {
        let dir = std::env::temp_dir().join(format!("ss-ort-{}", crate::util::new_session_id()));
        let res = dir.join("res");
        std::fs::create_dir_all(res.join("onnxruntime")).unwrap();
        let lib = res.join("onnxruntime").join(LIB_NAME);
        std::fs::write(&lib, b"").unwrap();
        assert!(!looks_like_library(&lib), "empty placeholder");
        std::fs::write(&lib, vec![b'x'; 100_000]).unwrap();
        assert!(!looks_like_library(&lib), "not a library");
        let mut bytes = if cfg!(target_os = "macos") {
            vec![0xCF, 0xFA, 0xED, 0xFE]
        } else if cfg!(target_os = "windows") {
            b"MZ\0\0".to_vec()
        } else {
            vec![0x7F, b'E', b'L', b'F']
        };
        bytes.resize(100_000, 0);
        std::fs::write(&lib, &bytes).unwrap();
        assert!(looks_like_library(&lib));
        let found = resolve(None, Some(&res)).unwrap();
        assert!(found.is_absolute());
        assert!(found.ends_with(Path::new("onnxruntime").join(LIB_NAME)));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_looks_in_frameworks_first() {
        let c = candidates(
            Some(Path::new(
                "/Applications/SplitSound.app/Contents/MacOS/SplitSound",
            )),
            Some(Path::new("/Applications/SplitSound.app/Contents/Resources")),
        );
        assert_eq!(
            c[0],
            Path::new(
                "/Applications/SplitSound.app/Contents/MacOS/../Frameworks/libonnxruntime.dylib"
            )
        );
    }
}
