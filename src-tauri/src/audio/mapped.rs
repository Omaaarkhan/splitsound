//! Read-only access to a working-format WAV (32-bit float, 44.1 kHz stereo)
//! as a `&[f32]`, memory-mapped when possible.
//!
//! The files mapped here live in the app's own library folder and are
//! written once, atomically, before they are ever mapped. If another
//! process truncated a mapped file the OS could fault the reading thread;
//! that is the accepted cost of mapping (it is how every DAW reads audio).

use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use super::wav::{self, WavInfo};

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
}

impl MappedWav {
    /// Opens and maps `path`. Fails with `InvalidData` if the file is not in
    /// the working format.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
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
            Ok(MappedWav {
                storage: Storage::Mapped { map, offset, len },
                info,
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
            })
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_working_files_and_rejects_others() {
        let d = std::env::temp_dir().join(format!("ss-map-{}", crate::util::new_session_id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("a.wav");
        let s: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        wav::write_working(&p, &s).unwrap();
        let m = MappedWav::open(&p).unwrap();
        assert!(matches!(m.storage, Storage::Mapped { .. }));
        assert_eq!(m.samples(), &s[..]);
        assert_eq!(m.frames(), 500);

        let q = d.join("b.wav");
        let mut w = wav::WavWriter::create(&q, wav::WriteFormat::Pcm16, 2, 44_100).unwrap();
        w.write(&[0.0; 8]).unwrap();
        w.finish().unwrap();
        assert!(MappedWav::open(&q).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
