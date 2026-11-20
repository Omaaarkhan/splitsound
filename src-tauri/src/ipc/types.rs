//! Rust mirror of `src/ipc/contract.ts`.
//!
//! Every type here serialises to exactly the JSON shape the contract
//! describes: struct fields in camelCase, enums as lowercase strings,
//! optional values as `null` (never omitted). `tests` at the bottom of this
//! file pin the key sets so drift is caught by `cargo test`.

use serde::{Deserialize, Deserializer, Serialize};

/* ------------------------------------------------------------------ */
/* Vocabulary                                                          */
/* ------------------------------------------------------------------ */

/// A stem the engine can produce. `Instrumental` only appears in two-stem mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StemId {
    Vocals,
    Drums,
    Bass,
    Guitar,
    Piano,
    Other,
    Instrumental,
}

impl StemId {
    /// Canonical top-to-bottom lane order (`STEM_ORDER` in the contract).
    pub const ORDER: [StemId; 7] = [
        StemId::Vocals,
        StemId::Drums,
        StemId::Bass,
        StemId::Guitar,
        StemId::Piano,
        StemId::Other,
        StemId::Instrumental,
    ];

    /// Lowercase id, also used as the stem's file name (`stems/<id>.wav`).
    pub fn as_str(self) -> &'static str {
        match self {
            StemId::Vocals => "vocals",
            StemId::Drums => "drums",
            StemId::Bass => "bass",
            StemId::Guitar => "guitar",
            StemId::Piano => "piano",
            StemId::Other => "other",
            StemId::Instrumental => "instrumental",
        }
    }

    /// Display name used in file names: "Song title (Vocals).wav".
    pub fn display_name(self) -> &'static str {
        match self {
            StemId::Vocals => "Vocals",
            StemId::Drums => "Drums",
            StemId::Bass => "Bass",
            StemId::Guitar => "Guitar",
            StemId::Piano => "Piano",
            StemId::Other => "Other",
            StemId::Instrumental => "Instrumental",
        }
    }

    /// Position in [`StemId::ORDER`].
    pub fn lane_index(self) -> usize {
        Self::ORDER
            .iter()
            .position(|s| *s == self)
            .unwrap_or(usize::MAX)
    }
}

/// What can be played or drawn: a stem, or the original mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceId {
    Vocals,
    Drums,
    Bass,
    Guitar,
    Piano,
    Other,
    Instrumental,
    Mix,
}

impl SourceId {
    /// The stem this source names, or `None` for the original mix.
    pub fn stem(self) -> Option<StemId> {
        Some(match self {
            SourceId::Vocals => StemId::Vocals,
            SourceId::Drums => StemId::Drums,
            SourceId::Bass => StemId::Bass,
            SourceId::Guitar => StemId::Guitar,
            SourceId::Piano => StemId::Piano,
            SourceId::Other => StemId::Other,
            SourceId::Instrumental => StemId::Instrumental,
            SourceId::Mix => return None,
        })
    }
}

impl From<StemId> for SourceId {
    fn from(s: StemId) -> Self {
        match s {
            StemId::Vocals => SourceId::Vocals,
            StemId::Drums => SourceId::Drums,
            StemId::Bass => SourceId::Bass,
            StemId::Guitar => SourceId::Guitar,
            StemId::Piano => SourceId::Piano,
            StemId::Other => SourceId::Other,
            StemId::Instrumental => SourceId::Instrumental,
        }
    }
}

/// How finely to split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    Vocals,
    Band,
    All,
}

impl SplitMode {
    /// `STEMS_BY_MODE` in the contract, in lane order.
    pub fn stems(self) -> &'static [StemId] {
        match self {
            SplitMode::Vocals => &[StemId::Vocals, StemId::Instrumental],
            SplitMode::Band => &[StemId::Vocals, StemId::Drums, StemId::Bass, StemId::Other],
            SplitMode::All => &[
                StemId::Vocals,
                StemId::Drums,
                StemId::Bass,
                StemId::Guitar,
                StemId::Piano,
                StemId::Other,
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Fast,
    Best,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    Wav16,
    Wav24,
    Wav32f,
    Flac,
    Mp3,
}

impl ExportFormat {
    /// File extension without the dot.
    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Wav16 | ExportFormat::Wav24 | ExportFormat::Wav32f => "wav",
            ExportFormat::Flac => "flac",
            ExportFormat::Mp3 => "mp3",
        }
    }
}

/// `IpcError.code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    FileNotFound,
    UnsupportedFormat,
    DecodeFailed,
    ModelMissing,
    /// A model with a restricted licence is needed and the user has not agreed to its terms.
    TermsRequired,
    ModelDownloadFailed,
    Offline,
    Cancelled,
    Busy,
    OutOfMemory,
    DiskFull,
    AudioDevice,
    NotFound,
    Internal,
}

/// Every command rejects with this shape. `message` is plain language, ready to show.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IpcError {
    pub code: ErrorCode,
    pub message: String,
}

/* ------------------------------------------------------------------ */
/* Sessions                                                            */
/* ------------------------------------------------------------------ */

/// An instrument the tagger heard, mapped to the stem it would land in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Detection {
    pub label: String,
    pub family: StemId,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackInfo {
    pub path: String,
    pub file_name: String,
    pub title: String,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub duration_sec: f64,
    pub sample_rate: u32,
    pub channels: u32,
    pub codec: String,
    pub bitrate_kbps: Option<u32>,
    pub file_size_bytes: u64,
    pub has_artwork: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    Imported,
    Queued,
    Splitting,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalysisStatus {
    Pending,
    Done,
    Unavailable,
    Failed,
}

/// Number of bins in [`StemInfo::activity`].
pub const ACTIVITY_BINS: usize = 240;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StemInfo {
    pub id: StemId,
    pub path: String,
    pub present: bool,
    pub active_fraction: f64,
    pub peak_db: f64,
    pub rms_db: f64,
    pub activity: Vec<f64>,
    pub contains: Vec<Detection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitSummary {
    pub mode: SplitMode,
    pub quality: Quality,
    pub finished_at: String,
    pub elapsed_sec: f64,
    pub model_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub id: String,
    pub created_at: String,
    pub track: TrackInfo,
    pub status: SessionStatus,
    pub analysis: AnalysisStatus,
    pub detections: Vec<Detection>,
    pub suggested_mode: SplitMode,
    pub split: Option<SplitSummary>,
    pub stems: Vec<StemInfo>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
    pub id: String,
    pub created_at: String,
    pub title: String,
    pub artist: Option<String>,
    pub duration_sec: f64,
    pub status: SessionStatus,
    pub stem_ids: Vec<StemId>,
    pub has_artwork: bool,
    pub size_bytes: u64,
}

/// Waveform overview for drawing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Peaks {
    pub buckets: u32,
    /// Base64 of `2 * buckets` signed bytes: min then max per bucket.
    pub data: String,
}

/* ------------------------------------------------------------------ */
/* Splitting                                                           */
/* ------------------------------------------------------------------ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitOptions {
    pub mode: SplitMode,
    pub quality: Quality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitEstimate {
    pub estimated_sec: f64,
    pub models: Vec<ModelStatus>,
    pub download_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitStage {
    Downloading,
    Preparing,
    Separating,
    Analyzing,
    Writing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SplitProgress {
    pub session_id: String,
    pub stage: SplitStage,
    pub fraction: f64,
    pub label: String,
    pub step: u32,
    pub step_count: u32,
    pub eta_sec: Option<f64>,
}

/* ------------------------------------------------------------------ */
/* Playback                                                            */
/* ------------------------------------------------------------------ */

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelState {
    pub id: SourceId,
    pub gain_db: f64,
    pub muted: bool,
    pub solo: bool,
}

/// `PlaybackState.source`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlaybackSource {
    Stems,
    Original,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoopRange {
    pub start_sec: f64,
    pub end_sec: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaybackState {
    pub session_id: Option<String>,
    pub playing: bool,
    pub position_sec: f64,
    pub duration_sec: f64,
    pub source: PlaybackSource,
    #[serde(rename = "loop")]
    pub loop_range: Option<LoopRange>,
    pub master_gain_db: f64,
    pub channels: Vec<ChannelState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Level {
    pub id: SourceId,
    pub peak: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaybackTick {
    pub position_sec: f64,
    pub levels: Vec<Level>,
}

/* ------------------------------------------------------------------ */
/* Export                                                              */
/* ------------------------------------------------------------------ */

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportOptions {
    pub stem_ids: Vec<StemId>,
    pub format: ExportFormat,
    pub destination_dir: String,
    pub include_mixdown: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub destination_dir: String,
    pub files: Vec<String>,
}

/* ------------------------------------------------------------------ */
/* Models                                                              */
/* ------------------------------------------------------------------ */

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: String,
    pub name: String,
    pub description: String,
    pub size_bytes: u64,
    pub download_bytes: u64,
    pub installed: bool,
    pub download_progress: Option<f64>,
    pub license: String,
    pub attribution: String,
    /// Who the download comes from, for display: "GitHub" or "Meta".
    pub source: String,
    /// True when the weights are not under an open-source licence. Such a
    /// model is never downloaded unless the user agreed to `terms`.
    pub restricted: bool,
    /// Plain-language terms to show before a restricted model is downloaded.
    pub terms: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadProgress {
    pub model_id: String,
    pub received_bytes: u64,
    pub total_bytes: u64,
}

/* ------------------------------------------------------------------ */
/* Settings and system                                                 */
/* ------------------------------------------------------------------ */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Acceleration {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub default_quality: Quality,
    pub export_format: ExportFormat,
    pub export_dir: Option<String>,
    pub acceleration: Acceleration,
    pub threads: u32,
    pub check_for_updates: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            default_quality: Quality::Fast,
            export_format: ExportFormat::Wav24,
            export_dir: None,
            acceleration: Acceleration::Auto,
            threads: 0,
            check_for_updates: true,
        }
    }
}

/// `Partial<Settings>`. `exportDir` distinguishes "absent" (outer `None`)
/// from "set to null" (`Some(None)`).
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SettingsPatch {
    pub default_quality: Option<Quality>,
    pub export_format: Option<ExportFormat>,
    #[serde(deserialize_with = "double_option")]
    pub export_dir: Option<Option<String>>,
    pub acceleration: Option<Acceleration>,
    pub threads: Option<u32>,
    pub check_for_updates: Option<bool>,
}

fn double_option<'de, D, T>(de: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(de).map(Some)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    Macos,
    Windows,
    Linux,
}

impl Platform {
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            Platform::Macos
        } else if cfg!(target_os = "windows") {
            Platform::Windows
        } else {
            Platform::Linux
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub available: bool,
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppInfo {
    pub version: String,
    pub platform: Platform,
    pub arch: String,
    pub gpu: GpuInfo,
    pub cpu_threads: u32,
    pub models_dir: String,
    pub library_dir: String,
    pub library_bytes: u64,
}

/* ------------------------------------------------------------------ */
/* Events without a named type in the contract                         */
/* ------------------------------------------------------------------ */

/// Payload of `app:open-files`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenFiles {
    pub paths: Vec<String>,
}

#[cfg(test)]
mod tests {
    //! Pins the JSON shape of every payload type against the contract.
    use super::*;
    use serde_json::{Value, json};

    fn keys(v: &Value) -> Vec<String> {
        let mut k: Vec<String> = v
            .as_object()
            .expect("expected a JSON object")
            .keys()
            .cloned()
            .collect();
        k.sort();
        k
    }

    fn assert_keys<T: Serialize>(value: &T, expected: &[&str]) -> Value {
        let v = serde_json::to_value(value).unwrap();
        let mut want: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(keys(&v), want, "key mismatch for {v}");
        v
    }

    fn detection() -> Detection {
        Detection {
            label: "Electric guitar".into(),
            family: StemId::Guitar,
            confidence: 0.8,
        }
    }

    fn track() -> TrackInfo {
        TrackInfo {
            path: "/music/a.wav".into(),
            file_name: "a.wav".into(),
            title: "A".into(),
            artist: None,
            album: Some("B".into()),
            duration_sec: 1.0,
            sample_rate: 44100,
            channels: 2,
            codec: "WAV".into(),
            bitrate_kbps: None,
            file_size_bytes: 10,
            has_artwork: false,
        }
    }

    fn stem_info() -> StemInfo {
        StemInfo {
            id: StemId::Vocals,
            path: "/x/stems/vocals.wav".into(),
            present: true,
            active_fraction: 0.5,
            peak_db: -1.0,
            rms_db: -20.0,
            activity: vec![0.0; ACTIVITY_BINS],
            contains: vec![detection()],
        }
    }

    fn summary() -> SplitSummary {
        SplitSummary {
            mode: SplitMode::All,
            quality: Quality::Best,
            finished_at: "2026-01-01T00:00:00.000Z".into(),
            elapsed_sec: 3.0,
            model_ids: vec!["m".into()],
        }
    }

    fn model() -> ModelStatus {
        ModelStatus {
            id: "m".into(),
            name: "M".into(),
            description: "d".into(),
            size_bytes: 1,
            download_bytes: 1,
            installed: false,
            download_progress: None,
            license: "MIT".into(),
            attribution: "a".into(),
            source: "GitHub".into(),
            restricted: false,
            terms: None,
        }
    }

    #[test]
    fn enums_serialise_as_lowercase_strings() {
        assert_eq!(json!(StemId::Instrumental), json!("instrumental"));
        assert_eq!(json!(SourceId::Mix), json!("mix"));
        assert_eq!(json!(SplitMode::Band), json!("band"));
        assert_eq!(json!(Quality::Best), json!("best"));
        assert_eq!(json!(ExportFormat::Wav32f), json!("wav32f"));
        assert_eq!(json!(ExportFormat::Mp3), json!("mp3"));
        assert_eq!(
            json!(ErrorCode::ModelDownloadFailed),
            json!("model_download_failed")
        );
        assert_eq!(json!(ErrorCode::AudioDevice), json!("audio_device"));
        assert_eq!(json!(ErrorCode::TermsRequired), json!("terms_required"));
        assert_eq!(json!(SessionStatus::Splitting), json!("splitting"));
        assert_eq!(json!(AnalysisStatus::Unavailable), json!("unavailable"));
        assert_eq!(json!(SplitStage::Analyzing), json!("analyzing"));
        assert_eq!(json!(PlaybackSource::Original), json!("original"));
        assert_eq!(json!(Acceleration::Gpu), json!("gpu"));
        assert_eq!(json!(Platform::Macos), json!("macos"));
        for (i, s) in StemId::ORDER.iter().enumerate() {
            assert_eq!(json!(s), json!(s.as_str()));
            assert_eq!(s.lane_index(), i);
        }
    }

    #[test]
    fn payload_keys_match_contract() {
        assert_keys(
            &IpcError {
                code: ErrorCode::Internal,
                message: "m".into(),
            },
            &["code", "message"],
        );
        assert_keys(&detection(), &["label", "family", "confidence"]);
        let t = assert_keys(
            &track(),
            &[
                "path",
                "fileName",
                "title",
                "artist",
                "album",
                "durationSec",
                "sampleRate",
                "channels",
                "codec",
                "bitrateKbps",
                "fileSizeBytes",
                "hasArtwork",
            ],
        );
        assert!(t["artist"].is_null(), "None must serialise as null");
        assert_keys(
            &stem_info(),
            &[
                "id",
                "path",
                "present",
                "activeFraction",
                "peakDb",
                "rmsDb",
                "activity",
                "contains",
            ],
        );
        assert_keys(
            &summary(),
            &["mode", "quality", "finishedAt", "elapsedSec", "modelIds"],
        );
        let session = Session {
            id: "0".repeat(32),
            created_at: "2026-01-01T00:00:00.000Z".into(),
            track: track(),
            status: SessionStatus::Ready,
            analysis: AnalysisStatus::Done,
            detections: vec![detection()],
            suggested_mode: SplitMode::All,
            split: Some(summary()),
            stems: vec![stem_info()],
            error: None,
        };
        assert_keys(
            &session,
            &[
                "id",
                "createdAt",
                "track",
                "status",
                "analysis",
                "detections",
                "suggestedMode",
                "split",
                "stems",
                "error",
            ],
        );
        assert_keys(
            &SessionSummary {
                id: "x".into(),
                created_at: "c".into(),
                title: "t".into(),
                artist: None,
                duration_sec: 1.0,
                status: SessionStatus::Imported,
                stem_ids: vec![StemId::Vocals],
                has_artwork: false,
                size_bytes: 1,
            },
            &[
                "id",
                "createdAt",
                "title",
                "artist",
                "durationSec",
                "status",
                "stemIds",
                "hasArtwork",
                "sizeBytes",
            ],
        );
        assert_keys(
            &Peaks {
                buckets: 1,
                data: "AAA=".into(),
            },
            &["buckets", "data"],
        );
        assert_keys(
            &SplitOptions {
                mode: SplitMode::All,
                quality: Quality::Fast,
            },
            &["mode", "quality"],
        );
        assert_keys(
            &SplitEstimate {
                estimated_sec: 1.0,
                models: vec![model()],
                download_bytes: 0,
            },
            &["estimatedSec", "models", "downloadBytes"],
        );
        assert_keys(
            &SplitProgress {
                session_id: "s".into(),
                stage: SplitStage::Separating,
                fraction: 0.5,
                label: "Separating drums".into(),
                step: 2,
                step_count: 5,
                eta_sec: None,
            },
            &[
                "sessionId",
                "stage",
                "fraction",
                "label",
                "step",
                "stepCount",
                "etaSec",
            ],
        );
        assert_keys(
            &ChannelState {
                id: SourceId::Vocals,
                gain_db: 0.0,
                muted: false,
                solo: false,
            },
            &["id", "gainDb", "muted", "solo"],
        );
        let ps = assert_keys(
            &PlaybackState {
                session_id: None,
                playing: false,
                position_sec: 0.0,
                duration_sec: 0.0,
                source: PlaybackSource::Stems,
                loop_range: Some(LoopRange {
                    start_sec: 1.0,
                    end_sec: 2.0,
                }),
                master_gain_db: 0.0,
                channels: vec![],
            },
            &[
                "sessionId",
                "playing",
                "positionSec",
                "durationSec",
                "source",
                "loop",
                "masterGainDb",
                "channels",
            ],
        );
        assert_eq!(keys(&ps["loop"]), vec!["endSec", "startSec"]);
        let tick = assert_keys(
            &PlaybackTick {
                position_sec: 1.0,
                levels: vec![Level {
                    id: SourceId::Mix,
                    peak: 0.5,
                }],
            },
            &["positionSec", "levels"],
        );
        assert_eq!(keys(&tick["levels"][0]), vec!["id", "peak"]);
        assert_keys(
            &ExportOptions {
                stem_ids: vec![StemId::Bass],
                format: ExportFormat::Flac,
                destination_dir: "/tmp".into(),
                include_mixdown: true,
            },
            &["stemIds", "format", "destinationDir", "includeMixdown"],
        );
        assert_keys(
            &ExportResult {
                destination_dir: "/tmp".into(),
                files: vec![],
            },
            &["destinationDir", "files"],
        );
        assert_keys(
            &model(),
            &[
                "id",
                "name",
                "description",
                "sizeBytes",
                "downloadBytes",
                "installed",
                "downloadProgress",
                "license",
                "attribution",
                "source",
                "restricted",
                "terms",
            ],
        );
        assert_keys(
            &ModelDownloadProgress {
                model_id: "m".into(),
                received_bytes: 1,
                total_bytes: 2,
            },
            &["modelId", "receivedBytes", "totalBytes"],
        );
        assert_keys(
            &Settings::default(),
            &[
                "defaultQuality",
                "exportFormat",
                "exportDir",
                "acceleration",
                "threads",
                "checkForUpdates",
            ],
        );
        let info = assert_keys(
            &AppInfo {
                version: "0.1.0".into(),
                platform: Platform::current(),
                arch: "aarch64".into(),
                gpu: GpuInfo {
                    available: false,
                    name: None,
                },
                cpu_threads: 8,
                models_dir: "/m".into(),
                library_dir: "/l".into(),
                library_bytes: 0,
            },
            &[
                "version",
                "platform",
                "arch",
                "gpu",
                "cpuThreads",
                "modelsDir",
                "libraryDir",
                "libraryBytes",
            ],
        );
        assert_eq!(keys(&info["gpu"]), vec!["available", "name"]);
        assert_keys(
            &OpenFiles {
                paths: vec!["/a.mp3".into()],
            },
            &["paths"],
        );
    }

    #[test]
    fn command_arguments_deserialise_from_contract_shapes() {
        let o: ExportOptions = serde_json::from_value(json!({
            "stemIds": ["vocals", "other"],
            "format": "wav24",
            "destinationDir": "/x",
            "includeMixdown": false
        }))
        .unwrap();
        assert_eq!(o.stem_ids, vec![StemId::Vocals, StemId::Other]);
        let s: SplitOptions =
            serde_json::from_value(json!({"mode": "vocals", "quality": "best"})).unwrap();
        assert_eq!(s.mode, SplitMode::Vocals);
        let l: Option<LoopRange> =
            serde_json::from_value(json!({"startSec": 1.5, "endSec": 3})).unwrap();
        assert_eq!(l.unwrap().end_sec, 3.0);
        assert!(serde_json::from_value::<SourceId>(json!("Vocals")).is_err());
    }

    #[test]
    fn settings_patch_distinguishes_absent_from_null() {
        let p: SettingsPatch = serde_json::from_value(json!({})).unwrap();
        assert_eq!(p.export_dir, None);
        let p: SettingsPatch = serde_json::from_value(json!({"exportDir": null})).unwrap();
        assert_eq!(p.export_dir, Some(None));
        let p: SettingsPatch =
            serde_json::from_value(json!({"exportDir": "/a", "threads": 4})).unwrap();
        assert_eq!(p.export_dir, Some(Some("/a".into())));
        assert_eq!(p.threads, Some(4));
    }
}
