/**
 * SplitSound IPC contract: the single source of truth for everything that
 * crosses the webview <-> Rust boundary.
 *
 * - Commands are Tauri commands, invoked by their snake_case name with a
 *   camelCase argument object. They resolve with `result` or reject with an
 *   `IpcError`.
 * - Events are emitted by Rust to the main window. Payloads are camelCase JSON.
 * - All times are seconds, all gains are decibels, all paths are absolute.
 *
 * The Rust side mirrors these types with serde (`rename_all = "camelCase"`,
 * enums as lowercase strings). Change this file first, then both sides.
 */

/* ------------------------------------------------------------------ */
/* Vocabulary                                                          */
/* ------------------------------------------------------------------ */

/**
 * A stem the engine can produce. `instrumental` only appears in the
 * two-stem mode (everything that is not vocals).
 */
export type StemId =
  | "vocals"
  | "drums"
  | "bass"
  | "guitar"
  | "piano"
  | "other"
  | "instrumental";

/** Canonical top-to-bottom lane order. */
export const STEM_ORDER: readonly StemId[] = [
  "vocals",
  "drums",
  "bass",
  "guitar",
  "piano",
  "other",
  "instrumental",
];

/** What can be played or drawn: a stem, or the original mix. */
export type SourceId = StemId | "mix";

/**
 * How finely to split.
 * - `vocals`: vocals + instrumental (2 stems)
 * - `band`:   vocals, drums, bass, other (4 stems)
 * - `all`:    vocals, drums, bass, guitar, piano, other (6 stems)
 */
export type SplitMode = "vocals" | "band" | "all";

export const STEMS_BY_MODE: Readonly<Record<SplitMode, readonly StemId[]>> = {
  vocals: ["vocals", "instrumental"],
  band: ["vocals", "drums", "bass", "other"],
  all: ["vocals", "drums", "bass", "guitar", "piano", "other"],
};

/**
 * - `fast`: the small models, one pass.
 * - `best`: the large models, with more overlap. Several times slower.
 * - `studio`: Mel-Band RoFormer vocal isolation followed by separation for minimal bleed.
 */
export type Quality = "fast" | "best" | "studio";

export type ExportFormat = "wav16" | "wav24" | "wav32f" | "flac" | "mp3";

/** Every command rejects with this shape. `message` is plain language, ready to show. */
export interface IpcError {
  code:
    | "file_not_found"
    | "unsupported_format"
    | "decode_failed"
    | "model_missing"
    /** A model with a restricted licence is needed and the user has not agreed to its terms. */
    | "terms_required"
    | "model_download_failed"
    | "offline"
    | "cancelled"
    | "busy"
    | "out_of_memory"
    | "disk_full"
    | "audio_device"
    | "not_found"
    | "internal";
  message: string;
}

/* ------------------------------------------------------------------ */
/* Sessions                                                            */
/* ------------------------------------------------------------------ */

/** An instrument the tagger heard, mapped to the stem it would land in. */
export interface Detection {
  /** Short human name, e.g. "Electric guitar", "Singing", "Synthesizer". */
  label: string;
  /** The stem this instrument ends up in. */
  family: StemId;
  /** 0..1 */
  confidence: number;
}

export interface TrackInfo {
  /** Original file the user chose. May have moved or been deleted since. */
  path: string;
  fileName: string;
  /** From tags, falling back to the file name without extension. */
  title: string;
  artist: string | null;
  album: string | null;
  durationSec: number;
  sampleRate: number;
  channels: number;
  /** Display name: "MP3", "FLAC", "Opus", "WAV", "AAC", ... */
  codec: string;
  bitrateKbps: number | null;
  fileSizeBytes: number;
  hasArtwork: boolean;
}

export type SessionStatus =
  /** Decoded and ready to split. */
  | "imported"
  /** Waiting behind another split. */
  | "queued"
  | "splitting"
  /** Stems exist. */
  | "ready"
  | "failed";

/** Instrument scan of the original mix. */
export type AnalysisStatus = "pending" | "done" | "unavailable" | "failed";

/** Number of bins in `StemInfo.activity`. */
export const ACTIVITY_BINS = 240;

export interface StemInfo {
  id: StemId;
  /** Absolute path of the lossless working file (32-bit float WAV, 44.1 kHz stereo). */
  path: string;
  /** False when the engine found next to nothing for this stem in this song. */
  present: boolean;
  /** Share of the song where this stem is audible, 0..1. */
  activeFraction: number;
  peakDb: number;
  rmsDb: number;
  /** Loudness envelope over the song, `ACTIVITY_BINS` values in 0..1. */
  activity: number[];
  /** Instruments heard inside this stem. Mostly interesting for `other`. */
  contains: Detection[];
}

export interface SplitSummary {
  mode: SplitMode;
  quality: Quality;
  /** ISO 8601 */
  finishedAt: string;
  elapsedSec: number;
  modelIds: string[];
}

export interface Session {
  id: string;
  /** ISO 8601 */
  createdAt: string;
  track: TrackInfo;
  status: SessionStatus;
  analysis: AnalysisStatus;
  /** Instruments heard in the original mix, strongest first. Empty until analysis is done. */
  detections: Detection[];
  /** The mode the app suggests from what it heard. */
  suggestedMode: SplitMode;
  /** Set while status is `queued` or `splitting`, and kept once `ready`. */
  split: SplitSummary | null;
  /** Empty until status is `ready`. In lane order. */
  stems: StemInfo[];
  /** Plain-language reason when status is `failed`. */
  error: string | null;
}

export interface SessionSummary {
  id: string;
  createdAt: string;
  title: string;
  artist: string | null;
  durationSec: number;
  status: SessionStatus;
  stemIds: StemId[];
  hasArtwork: boolean;
  /** Bytes this session uses on disk. */
  sizeBytes: number;
}

/** Waveform overview for drawing. */
export interface Peaks {
  buckets: number;
  /**
   * Base64 of `2 * buckets` signed bytes: min then max for each bucket, each
   * scaled so that -127..127 maps to -1..1 (both channels folded together).
   */
  data: string;
}

/* ------------------------------------------------------------------ */
/* Splitting                                                           */
/* ------------------------------------------------------------------ */

export interface SplitOptions {
  mode: SplitMode;
  quality: Quality;
}

export interface SplitEstimate {
  /** Rough processing time on this machine. */
  estimatedSec: number;
  /** Models this split needs, with whether each is installed. */
  models: ModelStatus[];
  /** Bytes that still have to be downloaded before the split can start. 0 when ready. */
  downloadBytes: number;
}

export type SplitStage =
  /** Fetching a model that is not installed yet. */
  | "downloading"
  | "preparing"
  | "separating"
  /** Measuring each stem and listening for instruments inside it. */
  | "analyzing"
  | "writing";

export interface SplitProgress {
  sessionId: string;
  stage: SplitStage;
  /** Overall progress, 0..1, never goes backwards. */
  fraction: number;
  /** User-facing, sentence case, no trailing period: "Separating drums". */
  label: string;
  /** 1-based index of the current step and the number of steps, for "2 of 5". */
  step: number;
  stepCount: number;
  /** Null until there is enough history to estimate. */
  etaSec: number | null;
}

/* ------------------------------------------------------------------ */
/* Playback                                                            */
/* ------------------------------------------------------------------ */

export interface ChannelState {
  id: SourceId;
  gainDb: number;
  muted: boolean;
  solo: boolean;
}

export interface PlaybackState {
  /** Null when nothing is loaded. */
  sessionId: string | null;
  playing: boolean;
  positionSec: number;
  durationSec: number;
  /**
   * `stems`: the mixer plays the separated stems.
   * `original`: the untouched mix plays instead (always the case before a split).
   */
  source: "stems" | "original";
  loop: { startSec: number; endSec: number } | null;
  masterGainDb: number;
  /** One entry per stem of the loaded session, in lane order. */
  channels: ChannelState[];
}

/** Sent about 30 times a second while playing. */
export interface PlaybackTick {
  positionSec: number;
  /** Post-fader peak level per channel over the last tick, linear 0..1. `mix` is the master bus. */
  levels: { id: SourceId; peak: number }[];
}

/* ------------------------------------------------------------------ */
/* Export                                                              */
/* ------------------------------------------------------------------ */

export interface ExportOptions {
  stemIds: StemId[];
  format: ExportFormat;
  destinationDir: string;
  /** Also write one file with the chosen stems mixed together at the current mixer levels. */
  includeMixdown: boolean;
}

export interface ExportResult {
  destinationDir: string;
  files: string[];
}

/* ------------------------------------------------------------------ */
/* Models                                                              */
/* ------------------------------------------------------------------ */

export interface ModelStatus {
  id: string;
  /** Display name, e.g. "Demucs 6-stem". */
  name: string;
  /** One plain sentence on what it is for. */
  description: string;
  /** Size on disk once installed. */
  sizeBytes: number;
  /** Size of the download. */
  downloadBytes: number;
  installed: boolean;
  /** 0..1 while downloading, otherwise null. */
  downloadProgress: number | null;
  license: string;
  attribution: string;
  /** Who the download comes from, for display: "GitHub" or "Meta". */
  source: string;
  /**
   * True when the weights are not under an open-source licence (for example
   * research or non-commercial use only). Such a model is never downloaded
   * unless the user has agreed to `terms`.
   */
  restricted: boolean;
  /** Plain-language terms to show before a restricted model is downloaded. Null otherwise. */
  terms: string | null;
}

export interface ModelDownloadProgress {
  modelId: string;
  receivedBytes: number;
  totalBytes: number;
}

/* ------------------------------------------------------------------ */
/* Settings and system                                                 */
/* ------------------------------------------------------------------ */

export interface Settings {
  defaultQuality: Quality;
  exportFormat: ExportFormat;
  /** Last folder stems were exported to. */
  exportDir: string | null;
  /** `auto` picks the fastest option that works on this machine. */
  acceleration: "auto" | "cpu" | "gpu";
  /** CPU threads for separation. 0 means choose automatically. */
  threads: number;
  checkForUpdates: boolean;
}

export interface AppInfo {
  version: string;
  platform: "macos" | "windows" | "linux";
  arch: string;
  /** Whether a GPU execution path exists on this machine, and what it is called. */
  gpu: { available: boolean; name: string | null };
  cpuThreads: number;
  modelsDir: string;
  libraryDir: string;
  /** Bytes used by all sessions. */
  libraryBytes: number;
}

/** A newer version of the app that can be installed. */
export interface UpdateInfo {
  version: string;
  /** Release notes as plain text, or null. Never rendered as markup. */
  notes: string | null;
}

export interface UpdateProgress {
  downloadedBytes: number;
  /** Null when the server did not say how big the download is. */
  totalBytes: number | null;
}

/* ------------------------------------------------------------------ */
/* Commands                                                            */
/* ------------------------------------------------------------------ */

export interface Commands {
  /* Library */
  list_sessions: { args: Record<string, never>; result: SessionSummary[] };
  /** Decodes the file, creates a session and starts the instrument scan in the background. */
  import_track: { args: { path: string }; result: Session };
  get_session: { args: { sessionId: string }; result: Session };
  delete_session: { args: { sessionId: string }; result: null };
  /** Removes every session. */
  clear_library: { args: Record<string, never>; result: null };
  /** Cover art as a data URL, or null. */
  get_artwork: { args: { sessionId: string }; result: string | null };
  get_peaks: {
    args: { sessionId: string; source: SourceId; buckets: number };
    result: Peaks;
  };

  /* Splitting */
  estimate_split: {
    args: { sessionId: string; options: SplitOptions };
    result: SplitEstimate;
  };
  /**
   * Queues the split and returns at once. Missing models are downloaded as
   * the first stage. Progress arrives as `split:progress`; the result as
   * `session:updated` with status `ready` or `failed`.
   *
   * If a missing model is `restricted`, the command rejects with
   * `terms_required` unless `acceptRestricted` is true. Show the model's
   * `terms` (from `estimate_split`), and call again with `acceptRestricted`
   * once the user agrees.
   */
  start_split: {
    args: { sessionId: string; options: SplitOptions; acceptRestricted?: boolean };
    result: null;
  };
  /** Cancels a queued or running split. The session returns to `imported` (or `ready` if it had stems). */
  cancel_split: { args: { sessionId: string }; result: null };

  /* Playback: one session is loaded at a time */
  playback_load: { args: { sessionId: string }; result: PlaybackState };
  playback_unload: { args: Record<string, never>; result: null };
  playback_play: { args: Record<string, never>; result: null };
  playback_pause: { args: Record<string, never>; result: null };
  playback_seek: { args: { positionSec: number }; result: null };
  playback_set_channel: {
    args: { id: SourceId; gainDb?: number; muted?: boolean; solo?: boolean };
    result: null;
  };
  playback_set_master: { args: { gainDb: number }; result: null };
  playback_set_source: {
    args: { source: "stems" | "original" };
    result: null;
  };
  playback_set_loop: {
    args: { loop: { startSec: number; endSec: number } | null };
    result: null;
  };

  /* Export */
  export_stems: {
    args: { sessionId: string; options: ExportOptions };
    result: ExportResult;
  };
  /**
   * Makes sure well-named files exist for dragging out of the window
   * ("Song title (Vocals).wav") and returns their paths, in the order asked.
   */
  prepare_drag: {
    args: { sessionId: string; stemIds: StemId[] };
    result: string[];
  };

  /* Models */
  list_models: { args: Record<string, never>; result: ModelStatus[] };
  /** Rejects with `terms_required` if a listed model is `restricted` and `acceptRestricted` is not true. */
  download_models: { args: { modelIds: string[]; acceptRestricted?: boolean }; result: null };
  cancel_model_downloads: { args: Record<string, never>; result: null };
  remove_model: { args: { modelId: string }; result: null };
  /** Installs model files the user downloaded by hand. Returns the ids that were recognised. */
  import_model_files: { args: { paths: string[] }; result: string[] };

  /* Settings and system */
  get_settings: { args: Record<string, never>; result: Settings };
  update_settings: { args: { patch: Partial<Settings> }; result: Settings };
  get_app_info: { args: Record<string, never>; result: AppInfo };
  reveal_path: { args: { path: string }; result: null };

  /* Updates. The interface cannot choose where updates come from: the
     endpoint and the signing key are fixed in the app. */
  /** Resolves with the newer version, or null when the app is up to date. */
  check_for_update: { args: Record<string, never>; result: UpdateInfo | null };
  /**
   * Downloads the update found by the last `check_for_update`, verifies its
   * signature, installs it and restarts the app. Progress arrives as
   * `update:progress`. Rejects if there is nothing to install.
   */
  install_update: { args: Record<string, never>; result: null };
}

/* ------------------------------------------------------------------ */
/* Events                                                              */
/* ------------------------------------------------------------------ */

export interface Events {
  /** A session changed: scan finished, split queued, finished or failed. */
  "session:updated": Session;
  /** The list of sessions changed. */
  "library:changed": SessionSummary[];
  "split:progress": SplitProgress;
  /** Sent whenever anything but the position changes. */
  "playback:state": PlaybackState;
  "playback:tick": PlaybackTick;
  "models:progress": ModelDownloadProgress;
  /** A model was installed, removed, or a download failed. */
  "models:changed": ModelStatus[];
  /**
   * The operating system asked the app to open audio files ("Open with",
   * a file passed on the command line, or a second launch). The UI imports them.
   */
  "app:open-files": { paths: string[] };
  "update:progress": UpdateProgress;
}

export type CommandName = keyof Commands;
export type EventName = keyof Events;
