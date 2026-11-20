//! `splitsound`: command-line front end for the SplitSound engine.
//!
//! ```text
//! splitsound separate <file> [--mode vocals|band|all] [--quality fast|best]
//!                            [--out DIR] [--format wav32f|wav24|wav16|flac|mp3]
//!                            [--download] [--accept-restricted] [--no-tag]
//! splitsound detect <file>
//! splitsound models list
//! splitsound models fetch [ID...] [--accept-restricted]     (no ids: every model)
//! splitsound models import <FILE...> [--accept-restricted]
//! splitsound models verify [ID...]
//! splitsound models remove <ID...>
//! splitsound bench [file] [--seconds N] [--only MODE/QUALITY,...]
//!
//! Options for every command:
//!   --models-dir DIR    where models live (default: $SPLITSOUND_MODELS_DIR,
//!                       else vendor/models in the source tree)
//!   --models-from DIR   install models from a local folder holding the release
//!                       files instead of downloading (same checks apply)
//!   --ort PATH          ONNX Runtime library (default: $ORT_DYLIB_PATH, else
//!                       vendor/onnxruntime/<host>/)
//!   --threads N         inference threads (default: automatic)
//!   --accel auto|cpu|gpu
//!   --json              machine-readable result on stdout
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::Instant;

use splitsound_core::engine::{export_file, peaks_for_file};
use splitsound_core::store::DownloadObserver;
use splitsound_core::{
    Acceleration, Engine, EngineConfig, Error, ExportFormat, Progress, ProgressUpdate, Quality,
    SplitMode, SplitOptions, SplitRequest,
};

struct Args {
    positional: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

/// Flags that take a value.
const VALUE_FLAGS: &[&str] = &[
    "--mode",
    "--quality",
    "--out",
    "--format",
    "--models-dir",
    "--ort",
    "--threads",
    "--accel",
    "--seconds",
    "--only",
    "--models-from",
];

impl Args {
    fn parse(raw: &[String]) -> Result<Args, String> {
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some((name, value)) = a.split_once('=').filter(|_| a.starts_with("--")) {
                flags.push((name.to_string(), Some(value.to_string())));
            } else if a.starts_with("--") {
                if VALUE_FLAGS.contains(&a.as_str()) {
                    let value = raw.get(i + 1).ok_or_else(|| format!("{a} needs a value"))?;
                    flags.push((a.clone(), Some(value.clone())));
                    i += 1;
                } else {
                    flags.push((a.clone(), None));
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Ok(Args { positional, flags })
    }

    fn value(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .and_then(|(_, v)| v.as_deref())
    }

    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(n, _)| n == name)
    }
}

fn default_models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SPLITSOUND_MODELS_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/models")
}

fn build_engine(args: &Args) -> Result<Engine, String> {
    let mut config = EngineConfig::new(
        args.value("--models-dir")
            .map(PathBuf::from)
            .unwrap_or_else(default_models_dir),
    );
    config.onnxruntime_path = args.value("--ort").map(PathBuf::from);
    if let Some(dir) = args.value("--models-from") {
        config.model_source = splitsound_core::ModelSource::Directory(PathBuf::from(dir));
    }
    if let Some(t) = args.value("--threads") {
        config.threads = t
            .parse()
            .map_err(|_| format!("--threads: not a number: {t}"))?;
    }
    config.acceleration = match args.value("--accel").unwrap_or("auto") {
        "auto" => Acceleration::Auto,
        "cpu" => Acceleration::Cpu,
        "gpu" => Acceleration::Gpu,
        other => return Err(format!("--accel: expected auto, cpu or gpu, got {other}")),
    };
    Engine::new(config).map_err(|e| describe(&e))
}

fn describe(e: &Error) -> String {
    match e.detail() {
        Some(d) => format!("{} [{}] ({d})", e.message(), e.code()),
        None => format!("{} [{}]", e.message(), e.code()),
    }
}

fn parse_options(args: &Args) -> Result<SplitOptions, String> {
    let mode = match args.value("--mode").unwrap_or("band") {
        "vocals" => SplitMode::Vocals,
        "band" => SplitMode::Band,
        "all" => SplitMode::All,
        other => return Err(format!("--mode: expected vocals, band or all, got {other}")),
    };
    let quality = match args.value("--quality").unwrap_or("fast") {
        "fast" => Quality::Fast,
        "best" => Quality::Best,
        other => return Err(format!("--quality: expected fast or best, got {other}")),
    };
    Ok(SplitOptions { mode, quality })
}

/// Progress on stderr: one line, rewritten in place.
struct Console {
    last: Mutex<(Instant, String)>,
    quiet: bool,
}

impl Console {
    fn new(quiet: bool) -> Self {
        Console {
            last: Mutex::new((Instant::now(), String::new())),
            quiet,
        }
    }

    fn finish(&self) {
        if !self.quiet {
            eprintln!();
        }
    }
}

impl Progress for Console {
    fn report(&self, u: &ProgressUpdate) {
        if self.quiet {
            return;
        }
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if u.label == last.1 && now.duration_since(last.0).as_millis() < 200 && u.fraction < 1.0 {
            return;
        }
        *last = (now, u.label.clone());
        let eta = u
            .eta_sec
            .map(|s| format!(", about {s:.0}s left"))
            .unwrap_or_default();
        eprint!(
            "\r\x1b[K[{}/{}] {:<38} {:5.1}%{eta}",
            u.step,
            u.step_count,
            u.label,
            u.fraction * 100.0
        );
        let _ = std::io::stderr().flush();
    }
}

impl DownloadObserver for Console {
    fn progress(&self, model_id: &str, received: u64, total: u64) {
        if !self.quiet {
            eprint!(
                "\r\x1b[K{model_id}: {:.1} / {:.1} MB",
                received as f64 / 1e6,
                total as f64 / 1e6
            );
            let _ = std::io::stderr().flush();
        }
    }
}

/// Peak resident memory of this process in bytes, where the OS says.
fn peak_rss_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: the pointer is valid for writing one libc::rusage.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: a successful getrusage initialized the struct.
        let max = unsafe { usage.assume_init() }.ru_maxrss.max(0) as u64;
        // macOS reports bytes, Linux kilobytes.
        Some(if cfg!(target_os = "macos") {
            max
        } else {
            max * 1024
        })
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn cmd_separate(args: &Args) -> Result<(), String> {
    let file = args.positional.get(1).ok_or("separate: which file?")?;
    let options = parse_options(args)?;
    let format = match args.value("--format") {
        None => ExportFormat::Wav32f,
        Some(f) => ExportFormat::parse(f).ok_or_else(|| format!("--format: unknown format {f}"))?,
    };
    if !format.is_available() {
        return Err("this build cannot write MP3".into());
    }
    let out_dir = args.value("--out").map(PathBuf::from).unwrap_or_else(|| {
        let stem = Path::new(file)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "song".into());
        PathBuf::from(format!("{stem}-stems"))
    });
    let engine = build_engine(args)?;
    let json = args.has("--json");
    let console = Console::new(json);

    let t0 = Instant::now();
    let track = engine
        .import(Path::new(file), &console)
        .map_err(|e| describe(&e))?;
    let decode_sec = t0.elapsed().as_secs_f64();
    let duration = track.info.duration_sec;
    if !json {
        eprintln!(
            "{}: {} {} Hz {} ch, {:.1}s (decoded in {decode_sec:.2}s)",
            file, track.info.codec, track.info.sample_rate, track.info.channels, duration
        );
    }

    let mut request = SplitRequest::new(options);
    request.download_missing = args.has("--download");
    request.accept_restricted = args.has("--accept-restricted");
    request.tag_stems = !args.has("--no-tag");
    let work_dir = if format == ExportFormat::Wav32f {
        out_dir.clone()
    } else {
        out_dir.join(".work")
    };
    let t1 = Instant::now();
    let result = engine
        .split_to_dir(&track.audio, request, &work_dir, &console)
        .map_err(|e| {
            console.finish();
            if e.code() == splitsound_core::ErrorCode::TermsRequired {
                let terms = engine
                    .plan(options)
                    .ok()
                    .and_then(|p| p.models.into_iter().find_map(|m| m.terms))
                    .unwrap_or_default();
                format!(
                    "{}\n\n{terms}\n\nRe-run with --accept-restricted if you agree.",
                    describe(&e)
                )
            } else {
                describe(&e)
            }
        })?;
    console.finish();
    let split_sec = t1.elapsed().as_secs_f64();

    let mut files = Vec::new();
    for stem in &result.stems {
        let path = if format == ExportFormat::Wav32f {
            stem.path.clone()
        } else {
            let dst = out_dir.join(format!("{}.{}", stem.id.as_str(), format.extension()));
            export_file(&stem.path, format, &dst, &console).map_err(|e| describe(&e))?;
            dst
        };
        files.push(path);
    }
    if format != ExportFormat::Wav32f {
        let _ = std::fs::remove_dir_all(&work_dir);
    }

    if json {
        let value = serde_json::json!({
            "file": file,
            "durationSec": duration,
            "options": options,
            "decodeSec": decode_sec,
            "splitSec": split_sec,
            "realtimeFactor": duration / split_sec.max(1e-9),
            "peakRssBytes": peak_rss_bytes(),
            "modelIds": result.model_ids,
            "files": files,
            "stems": result.stems,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!(
        "split in {split_sec:.1}s ({:.1}x realtime) with {}",
        duration / split_sec.max(1e-9),
        result.model_ids.join(" + ")
    );
    if let Some(rss) = peak_rss_bytes() {
        println!("peak memory {:.2} GB", rss as f64 / (1u64 << 30) as f64);
    }
    println!(
        "{:<13} {:>7} {:>7} {:>7} {:>7}  file",
        "stem", "present", "active", "peak", "rms"
    );
    for (stem, path) in result.stems.iter().zip(&files) {
        println!(
            "{:<13} {:>7} {:>6.0}% {:>6.1}dB {:>6.1}dB  {}",
            stem.id.as_str(),
            if stem.stats.present { "yes" } else { "no" },
            stem.stats.active_fraction * 100.0,
            stem.stats.peak_db,
            stem.stats.rms_db,
            path.display()
        );
        if !stem.contains.is_empty() {
            let heard: Vec<String> = stem
                .contains
                .iter()
                .take(6)
                .map(|d| format!("{} {:.0}%", d.label, d.confidence * 100.0))
                .collect();
            println!("{:<13} contains: {}", "", heard.join(", "));
        }
    }
    Ok(())
}

fn cmd_detect(args: &Args) -> Result<(), String> {
    let file = args.positional.get(1).ok_or("detect: which file?")?;
    let engine = build_engine(args)?;
    let console = Console::new(true);
    let track = engine
        .import(Path::new(file), &console)
        .map_err(|e| describe(&e))?;
    let t0 = Instant::now();
    let outcome = engine
        .detect(&track.audio, &console)
        .map_err(|e| describe(&e))?;
    let report = match outcome {
        splitsound_core::DetectOutcome::Done(report) => report,
        splitsound_core::DetectOutcome::Unavailable { missing } => {
            if args.has("--json") {
                println!(
                    "{}",
                    serde_json::json!({ "status": "unavailable", "missing": missing })
                );
                return Ok(());
            }
            return Err(format!(
                "detection is unavailable: install {} first (splitsound models fetch)",
                missing.join(" and ")
            ));
        }
    };
    if args.has("--json") {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!(
        "{}: {:.1}s, listened in {:.2}s",
        file,
        track.info.duration_sec,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "preview of {} segment(s): vocals {}, drums {}, bass {}",
        report.segments,
        if report.vocals_heard {
            "heard"
        } else {
            "not heard"
        },
        if report.drums_heard {
            "heard"
        } else {
            "not heard"
        },
        if report.bass_heard {
            "heard"
        } else {
            "not heard"
        }
    );
    if report.detections.is_empty() {
        println!("no instruments recognised");
    }
    for d in &report.detections {
        println!(
            "{:>5.1}%  {:<8} {}",
            d.confidence * 100.0,
            d.family.as_str(),
            d.label
        );
    }
    println!(
        "suggested split: {}",
        serde_json::to_string(&report.suggested_mode)
            .unwrap_or_default()
            .trim_matches('"')
    );
    Ok(())
}

fn cmd_models(args: &Args) -> Result<(), String> {
    let action = args
        .positional
        .get(1)
        .map(String::as_str)
        .ok_or("models: list, fetch, import, verify or remove?")?;
    let engine = build_engine(args)?;
    let store = engine.models();
    let rest: Vec<&str> = args.positional.iter().skip(2).map(String::as_str).collect();
    let all_ids: Vec<String> = store
        .manifest()
        .models
        .iter()
        .map(|m| m.id.clone())
        .collect();
    let accept = args.has("--accept-restricted");
    match action {
        "list" => {
            if args.has("--json") {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&store.list()).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!("models in {}", store.dir().display());
            println!(
                "{:<22} {:<10} {:>9} {:>9}  {:<10} licence",
                "id", "installed", "size MB", "dl MB", "source"
            );
            for m in store.list() {
                println!(
                    "{:<22} {:<10} {:>9.1} {:>9.1}  {:<10} {}{}",
                    m.id,
                    if m.installed { "yes" } else { "no" },
                    m.size_bytes as f64 / 1e6,
                    m.download_bytes as f64 / 1e6,
                    m.source,
                    m.license,
                    if m.restricted {
                        "  [restricted: needs --accept-restricted]"
                    } else {
                        ""
                    }
                );
            }
            Ok(())
        }
        "fetch" => {
            let ids: Vec<&str> = if rest.is_empty() {
                all_ids.iter().map(String::as_str).collect()
            } else {
                rest
            };
            let console = Console::new(false);
            let mut failed = false;
            for id in ids {
                match store.install(id, accept, &console) {
                    Ok(()) => eprintln!("\r\x1b[K{id}: installed"),
                    Err(e) => {
                        failed = true;
                        eprintln!("\r\x1b[K{id}: {}", describe(&e));
                        if e.code() == splitsound_core::ErrorCode::TermsRequired
                            && let Ok(status) = store.status(id)
                            && let Some(terms) = status.terms
                        {
                            eprintln!(
                                "    {terms}\n    Re-run with --accept-restricted if you agree."
                            );
                        }
                    }
                }
            }
            if failed {
                Err("some models were not installed".into())
            } else {
                Ok(())
            }
        }
        "import" => {
            if rest.is_empty() {
                return Err("models import: which files?".into());
            }
            for file in rest {
                match store.import_file(Path::new(file), accept) {
                    Ok(Some(id)) => println!("{file}: installed as {id}"),
                    Ok(None) => println!("{file}: not a known model file, ignored"),
                    Err(e) => println!("{file}: {}", describe(&e)),
                }
            }
            Ok(())
        }
        "verify" => {
            let ids: Vec<&str> = if rest.is_empty() {
                all_ids.iter().map(String::as_str).collect()
            } else {
                rest
            };
            let mut bad = false;
            for id in ids {
                if !store.is_installed(id) && store.manifest().find(id).is_some() {
                    println!("{id}: not installed");
                    continue;
                }
                match store.verify(id) {
                    Ok(()) => println!("{id}: ok (size and SHA-256 match the manifest)"),
                    Err(e) => {
                        bad = true;
                        println!("{id}: FAILED: {}", describe(&e));
                    }
                }
            }
            if bad {
                Err("verification failed".into())
            } else {
                Ok(())
            }
        }
        "remove" => {
            for id in rest {
                engine.remove_model(id).map_err(|e| describe(&e))?;
                println!("{id}: removed");
            }
            Ok(())
        }
        other => Err(format!("models: unknown action {other}")),
    }
}

fn cmd_bench(args: &Args) -> Result<(), String> {
    let engine = build_engine(args)?;
    let console = Console::new(true);
    let audio = match args.positional.get(1) {
        Some(file) => {
            engine
                .import(Path::new(file), &console)
                .map_err(|e| describe(&e))?
                .audio
        }
        None => {
            // Deterministic noise: speed does not depend on the content.
            let seconds: f64 = args
                .value("--seconds")
                .unwrap_or("60")
                .parse()
                .map_err(|_| "--seconds: not a number")?;
            let n = (seconds * 44_100.0) as usize;
            let mut seed = 0x1234_5678_9ABC_DEF1u64;
            let mut noise = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.3
            };
            let left: Vec<f32> = (0..n).map(|_| noise()).collect();
            let right: Vec<f32> = (0..n).map(|_| noise()).collect();
            splitsound_core::audio::Stereo::from_channels(left, right)
        }
    };
    let seconds = audio.len() as f64 / 44_100.0;
    let only: Option<Vec<&str>> = args.value("--only").map(|s| s.split(',').collect());
    println!(
        "bench: {seconds:.1}s of audio, {} threads",
        engine.threads()
    );
    println!(
        "{:<8} {:<6} {:>9} {:>10} {:>11}  models",
        "mode", "quality", "seconds", "x realtime", "peak RSS GB"
    );
    for (mode, mode_name) in [
        (SplitMode::Vocals, "vocals"),
        (SplitMode::Band, "band"),
        (SplitMode::All, "all"),
    ] {
        for (quality, quality_name) in [(Quality::Fast, "fast"), (Quality::Best, "best")] {
            let key = format!("{mode_name}/{quality_name}");
            if only.as_ref().is_some_and(|o| !o.contains(&key.as_str())) {
                continue;
            }
            let options = SplitOptions { mode, quality };
            let t0 = Instant::now();
            match engine.split(&audio, options, &console) {
                Ok(split) => {
                    let dt = t0.elapsed().as_secs_f64();
                    println!(
                        "{:<8} {:<6} {:>9.1} {:>10.1} {:>11}  {}",
                        mode_name,
                        quality_name,
                        dt,
                        seconds / dt,
                        peak_rss_bytes()
                            .map(|b| format!("{:.2}", b as f64 / (1u64 << 30) as f64))
                            .unwrap_or_else(|| "?".into()),
                        split.model_ids.join(" + ")
                    );
                }
                Err(e) => println!("{mode_name:<8} {quality_name:<6} skipped: {}", describe(&e)),
            }
            engine.unload();
        }
    }
    Ok(())
}

fn cmd_peaks(args: &Args) -> Result<(), String> {
    let file = args.positional.get(1).ok_or("peaks: which working file?")?;
    let t0 = Instant::now();
    let peaks = peaks_for_file(Path::new(file), 2000).map_err(|e| describe(&e))?;
    println!(
        "{} buckets in {:.1} ms, base64 length {}",
        peaks.buckets,
        t0.elapsed().as_secs_f64() * 1e3,
        peaks.to_base64().len()
    );
    Ok(())
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("splitsound: {e}");
            return ExitCode::from(2);
        }
    };
    let result = match args.positional.first().map(String::as_str) {
        Some("separate") => cmd_separate(&args),
        Some("detect") => cmd_detect(&args),
        Some("models") => cmd_models(&args),
        Some("bench") => cmd_bench(&args),
        Some("peaks") => cmd_peaks(&args),
        _ => {
            eprintln!(
                "usage: splitsound <separate|detect|models|bench> ...\n\
                 \n  separate <file> [--mode vocals|band|all] [--quality fast|best] [--out DIR]\n\
                 \x20          [--format wav32f|wav24|wav16|flac|mp3] [--download] [--accept-restricted]\n\
                 \x20 detect <file>\n\
                 \x20 models list | fetch [ID...] | import <FILE...> | verify [ID...] | remove <ID...>\n\
                 \x20 bench [file] [--seconds N] [--only mode/quality,...]\n\
                 \n  common: --models-dir DIR  --models-from DIR  --ort PATH  --threads N  --accel auto|cpu|gpu  --json"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("splitsound: {e}");
            ExitCode::FAILURE
        }
    }
}
