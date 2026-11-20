//! Opt-in debug-bundle smoke test. Uses a throwaway library and the actual
//! WebView, IPC handlers and output backend. Set `SPLITSOUND_SMOKE_TEST=1`
//! with either `SPLITSOUND_MOCK_ENGINE=1` or `SPLITSOUND_SMOKE_INPUT` pointing
//! at a real MP3/Opus excerpt (at least 20 seconds). The real run uses ffmpeg
//! to verify the exported audio. `SPLITSOUND_SMOKE_EMPTY_STORE=1` checks the
//! missing-model path instead, with an empty models directory.

use std::f32::consts::TAU;
use std::io;
use std::path::PathBuf;

use tauri::{Manager, Runtime, WebviewWindowBuilder};

use crate::app::{AppCore, AppPaths};
use crate::audio::mapped::MappedWav;
use crate::audio::wav::{WORK_RATE, write_working};
use crate::ipc::{SessionStatus, SplitMode};
use crate::open_files::OpenFilesGate;
use crate::picker::PickedPaths;
use crate::util::new_session_id;

pub(crate) struct Smoke {
    root: PathBuf,
    song: PathBuf,
    denied: PathBuf,
    export: PathBuf,
    real: bool,
    empty_store: bool,
}

impl Smoke {
    pub fn from_env() -> io::Result<Option<Self>> {
        if std::env::var("SPLITSOUND_SMOKE_TEST").as_deref() != Ok("1") {
            return Ok(None);
        }
        let mock = std::env::var(crate::engine::MOCK_ENV).as_deref() == Ok("1");
        let input = std::env::var_os("SPLITSOUND_SMOKE_INPUT").map(PathBuf::from);
        let extension = match (mock, &input) {
            (true, None) => "wav",
            (false, Some(path)) => match path.extension().and_then(|ext| ext.to_str()) {
                Some(ext @ ("mp3" | "opus")) => ext,
                _ => {
                    return Err(io::Error::other(
                        "the smoke input must be an MP3 or Opus file",
                    ));
                }
            },
            _ => {
                return Err(io::Error::other(
                    "set either SPLITSOUND_MOCK_ENGINE=1 or SPLITSOUND_SMOKE_INPUT",
                ));
            }
        };
        let root = std::env::temp_dir().join(format!("splitsound-smoke-{}", new_session_id()));
        std::fs::create_dir(&root)?;
        let smoke = Self {
            song: root.join(format!("SplitSound smoke.{extension}")),
            denied: root.join(format!("Unselected.{extension}")),
            export: root.join("export"),
            real: !mock,
            empty_store: std::env::var("SPLITSOUND_SMOKE_EMPTY_STORE").as_deref() == Ok("1"),
            root,
        };
        std::fs::create_dir(&smoke.export)?;
        if let Some(input) = input {
            std::fs::copy(input, &smoke.song)?;
        } else {
            let samples: Vec<_> = (0..3 * WORK_RATE)
                .flat_map(|i| {
                    let sample = 0.02 * (TAU * i as f32 * 440.0 / WORK_RATE as f32).sin();
                    [sample, sample]
                })
                .collect();
            write_working(&smoke.song, &samples)?;
        }
        std::fs::copy(&smoke.song, &smoke.denied)?;
        Ok(Some(smoke))
    }

    pub fn paths(&self) -> AppPaths {
        AppPaths::under(&self.root)
    }

    pub fn configure<'a, R: Runtime, M: Manager<R>>(
        self,
        builder: WebviewWindowBuilder<'a, R, M>,
        picked: &PickedPaths,
        gate: &OpenFilesGate,
        sink: &dyn crate::ipc::EventSink,
    ) -> WebviewWindowBuilder<'a, R, M> {
        picked.grant(&self.export, true);
        gate.push(vec![self.song.clone()], sink);
        let args = serde_json::json!({
            "denied": self.denied, "export": self.export, "song": self.song,
            "real": self.real, "emptyStore": self.empty_store,
        });
        builder
            .initialization_script(format!("({})({args});", include_str!("smoke.js")))
            .on_document_title_changed(move |window, title| {
                if title == "SplitSound smoke phase all" {
                    log::info!("bundled smoke test: starting all/fast");
                }
                if let Some(result) = title.strip_prefix("SplitSound smoke result ") {
                    let core = window.state::<std::sync::Arc<AppCore>>();
                    let verification = self.verify(&core, result);
                    match &verification {
                        Ok(()) => log::info!("bundled smoke test: {result}; verified"),
                        Err(error) => log::error!("bundled smoke test: {result}; {error}"),
                    }
                    core.shutdown();
                    if let Err(error) = std::fs::remove_dir_all(&self.root) {
                        log::warn!("couldn't clean the smoke-test library: {error}");
                    }
                    window.app_handle().exit(i32::from(verification.is_err()));
                }
            })
    }

    fn verify(&self, core: &AppCore, result: &str) -> Result<(), Box<dyn std::error::Error>> {
        if self.empty_store && result == "passed consent and unavailable preview" {
            return Ok(());
        }
        if self.real {
            let report: RealReport = serde_json::from_str(
                result
                    .strip_prefix("passed real ")
                    .ok_or("real IPC checks failed")?,
            )?;
            verify_split(core, &report.band, SplitMode::Band)?;
            verify_split(core, &report.all, SplitMode::All)?;
            for ext in ["flac", "mp3"] {
                for file in ["Vocals", "Mix"] {
                    let path = self.export.join(format!("SplitSound smoke ({file}).{ext}"));
                    let status = std::process::Command::new("ffmpeg")
                        .args(["-nostdin", "-v", "error", "-i"])
                        .arg(&path)
                        .args(["-map", "0:a:0", "-f", "null", "-"])
                        .status()?;
                    if !status.success() {
                        return Err(format!("ffmpeg could not decode {}", path.display()).into());
                    }
                    log::info!(
                        "ffmpeg decoded {}",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    );
                }
            }
        } else if !matches!(
            result,
            "passed with audio" | "passed without an audio device"
        ) || !["Vocals", "Mix"].iter().all(|file| {
            self.export
                .join(format!("SplitSound smoke ({file}).wav"))
                .is_file()
        }) {
            return Err("mock IPC or export checks failed".into());
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct RealReport {
    band: String,
    all: String,
}

fn verify_split(
    core: &AppCore,
    id: &str,
    mode: SplitMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let session = core.get_session(id)?;
    let dir = core.library.dir(id)?;
    let summary = session.split.ok_or("missing split summary")?;
    if session.status != SessionStatus::Ready
        || summary.mode != mode
        || session.stems.len() != mode.stems().len()
    {
        return Err(format!("unexpected {mode:?} result").into());
    }
    let mix = MappedWav::open(&dir.join("mix.wav"))?;
    let mut stems = Vec::new();
    for (stem, expected) in session.stems.iter().zip(mode.stems()) {
        if stem.id != *expected {
            return Err("unexpected stem order".into());
        }
        let audio = MappedWav::open(&crate::library::stem_path(&dir, stem.id))?;
        if audio.samples().len() != mix.samples().len() {
            return Err("stem and mix lengths differ".into());
        }
        let energy = audio
            .samples()
            .iter()
            .map(|s| f64::from(*s).powi(2))
            .sum::<f64>();
        let rms_db = 10.0 * (energy / audio.samples().len() as f64).log10();
        if !rms_db.is_finite() || rms_db < -80.0 || (stem.rms_db - rms_db).abs() > 0.01 {
            return Err(format!(
                "{} is silent, invalid, or has inconsistent metadata",
                stem.id.as_str()
            )
            .into());
        }
        log::info!(
            "{mode:?} {}: RMS {rms_db:.2} dBFS, peak {:.2} dBFS",
            stem.id.as_str(),
            stem.peak_db
        );
        stems.push(audio);
    }
    let mut max_error = 0.0_f64;
    for (i, original) in mix.samples().iter().enumerate() {
        let sum: f64 = stems.iter().map(|stem| f64::from(stem.samples()[i])).sum();
        max_error = max_error.max((sum - f64::from(*original)).abs());
    }
    if max_error > 5e-6 {
        return Err(format!("{mode:?} max reconstruction error {max_error:.3e}").into());
    }
    log::info!(
        "{mode:?} completed in {:.1}s, max reconstruction error {max_error:.3e}",
        summary.elapsed_sec
    );
    Ok(())
}

impl Drop for Smoke {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
