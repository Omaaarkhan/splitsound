//! Read-only access to a working-format WAV (32-bit float, 44.1 kHz stereo)
//! as a `&[f32]`, memory-mapped when possible.
//!
//! The files mapped here live in the app's own library folder and are
//! written once, atomically, before they are ever mapped. If another
//! process truncated a mapped file the OS could fault the reading thread;
//! that is the accepted cost of mapping (it is how every DAW reads audio).

#[cfg(debug_assertions)]
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;
#[cfg(debug_assertions)]
use std::path::PathBuf;
#[cfg(debug_assertions)]
use std::sync::{Mutex, OnceLock};

use memmap2::{Mmap, MmapOptions};

use super::wav::{self, WavInfo};

#[cfg(debug_assertions)]
fn live_map() -> &'static Mutex<HashMap<PathBuf, usize>> {
    static LIVE: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(debug_assertions)]
fn normalize_key(path: &Path) -> PathBuf {
    if let Ok(c) = crate::app::dunce_canonicalize(path) {
        return c;
    }
    let mut missing = Vec::new();
    let mut cur = path;
    while let Some(parent) = cur.parent() {
        if let Some(name) = cur.file_name() {
            missing.push(name.to_os_string());
        }
        if let Ok(mut base) = crate::app::dunce_canonicalize(parent) {
            for part in missing.into_iter().rev() {
                base.push(part);
            }
            return base;
        }
        cur = parent;
    }
    path.to_path_buf()
}

enum Storage {
    Mapped {
        map: Mmap,
        offset: usize,
        len: usize,
    },
    Owned(Vec<f32>),
}

/// Interleaved stereo `f32` samples of a working-format WAV.
pub struct MappedWav {
    storage: Storage,
    pub info: WavInfo,
    #[cfg(debug_assertions)]
    mapped_path: Option<PathBuf>,
}

impl MappedWav {
    /// Opens and maps `path`. Fails with `InvalidInput` if `path` is not a
    /// regular file, or `InvalidData` if the file is not in the working format.
    pub fn open(path: &Path) -> io::Result<Self> {
        let path_meta = std::fs::metadata(path)?;
        if !path_meta.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        let mut file = File::open(path)?;
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        let info = wav::read_info(&mut file)?;
        if !info.is_working_format() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a 32-bit float 44.1 kHz stereo WAV",
            ));
        }
        let offset = info.data_offset as usize;
        let len = (info.data_len / 4) as usize;
        if len == 0 {
            return Ok(MappedWav {
                storage: Storage::Owned(Vec::new()),
                info,
                #[cfg(debug_assertions)]
                mapped_path: None,
            });
        }
        // SAFETY: the file is opened read-only and the map is never written.
        // See the module docs for the truncation caveat.
        let map = unsafe { MmapOptions::new().map(&file)? };
        if map.len() < offset + len * 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "file shorter than header says",
            ));
        }
        let aligned = (map.as_ptr() as usize + offset).is_multiple_of(std::mem::align_of::<f32>());
        if aligned && cfg!(target_endian = "little") {
            #[cfg(unix)]
            let _ = map.advise(memmap2::Advice::Sequential);
            #[cfg(debug_assertions)]
            let mapped_path = {
                let key = normalize_key(path);
                let mut m = crate::util::lock(live_map());
                *m.entry(key.clone()).or_insert(0) += 1;
                Some(key)
            };
            Ok(MappedWav {
                storage: Storage::Mapped { map, offset, len },
                info,
                #[cfg(debug_assertions)]
                mapped_path,
            })
        } else {
            // Unaligned data (another writer's layout) or a big-endian host: copy.
            let bytes = &map[offset..offset + len * 4];
            let v = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| wav::sanitize(f32::from_le_bytes(*c)))
                .collect();
            Ok(MappedWav {
                storage: Storage::Owned(v),
                info,
                #[cfg(debug_assertions)]
                mapped_path: None,
            })
        }
    }

    /// Number of currently live memory mappings rooted under `dir` (debug/tests).
    #[cfg(debug_assertions)]
    pub fn live_mappings_under(dir: &Path) -> usize {
        let prefix = normalize_key(dir);
        let m = crate::util::lock(live_map());
        m.iter()
            .filter(|(p, _)| p.starts_with(&prefix))
            .map(|(_, count)| *count)
            .sum()
    }

    /// Wraps samples already in memory (tests, synthetic sources).
    pub fn from_samples(samples: Vec<f32>) -> Self {
        let frames = (samples.len() / 2) as u64;
        MappedWav {
            info: WavInfo {
                format: wav::SampleFormat::Float(32),
                channels: 2,
                sample_rate: wav::WORK_RATE,
                data_offset: 0,
                data_len: frames * 8,
                block_align: 8,
                title: None,
                artist: None,
                album: None,
            },
            storage: Storage::Owned(samples),
            #[cfg(debug_assertions)]
            mapped_path: None,
        }
    }

    /// Interleaved L/R samples.
    pub fn samples(&self) -> &[f32] {
        match &self.storage {
            Storage::Mapped { map, offset, len } => {
                let ptr = map[*offset..].as_ptr() as *const f32;
                // SAFETY: bounds were checked in `open`, alignment was checked,
                // the host is little-endian (WAV byte order) and any bit
                // pattern is a valid f32 (non-finite values are filtered by
                // the mixer).
                unsafe { std::slice::from_raw_parts(ptr, *len) }
            }
            Storage::Owned(v) => v,
        }
    }

    pub fn frames(&self) -> u64 {
        (self.samples().len() / 2) as u64
    }
}

#[cfg(debug_assertions)]
impl Drop for MappedWav {
    fn drop(&mut self) {
        // Release the OS mapping before decrementing the live counter so any
        // observer that sees zero knows UnmapViewOfFile / munmap already ran.
        self.storage = Storage::Owned(Vec::new());
        if let Some(key) = self.mapped_path.take() {
            let mut m = crate::util::lock(live_map());
            if let Some(count) = m.get_mut(&key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    m.remove(&key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_working_files_and_rejects_others() {
        let d = std::env::temp_dir().join(format!("ss-map-{}", crate::util::new_session_id()));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(
            MappedWav::open(&d).err().unwrap().kind(),
            io::ErrorKind::InvalidInput
        );
        let p = d.join("a.wav");
        let s: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        wav::write_working(&p, &s).unwrap();
        let m = MappedWav::open(&p).unwrap();
        assert!(matches!(m.storage, Storage::Mapped { .. }));
        assert_eq!(MappedWav::live_mappings_under(&d), 1);
        assert_eq!(m.samples(), &s[..]);
        assert_eq!(m.frames(), 500);

        let q = d.join("b.wav");
        let mut w = wav::WavWriter::create(&q, wav::WriteFormat::Pcm16, 2, 44_100).unwrap();
        w.write(&[0.0; 8]).unwrap();
        w.finish().unwrap();
        assert!(MappedWav::open(&q).is_err());
        drop(m);
        assert_eq!(MappedWav::live_mappings_under(&d), 0);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
