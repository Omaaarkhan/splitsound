import type {
  AppInfo,
  ChannelState,
  CommandName,
  Commands,
  Detection,
  EventName,
  Events,
  IpcError,
  ModelStatus,
  PlaybackState,
  Session,
  SessionSummary,
  Settings,
  SourceId,
  SplitEstimate,
  SplitMode,
  SplitOptions,
  SplitProgress,
  StemId,
  StemInfo,
} from "../contract";
import { STEMS_BY_MODE } from "../contract";
import { AUDIO_EXTENSIONS, EXPORT_FORMATS, stemName } from "../../utils/stems";
import { COPY } from "../../utils/errors";
import { getNowMs } from "../../utils/clock";
import {
  createInitialAppInfo,
  createInitialModels,
  createInitialSessions,
  createInitialSettings,
  MOCK_ARTWORK,
} from "./fixtures";
import { generateActivity, generatePeaks } from "./waveforms";

export type MockScene =
  | "home"
  | "home-empty"
  | "home-first-run"
  | "imported"
  | "consent"
  | "scan-downloading"
  | "scan-unavailable"
  | "splitting"
  | "ready"
  | "ready-solo"
  | "loop"
  | "export"
  | "settings"
  | "update"
  | "error";

export interface MockBackendOptions {
  scene?: MockScene | null;
  /** Speed multiplier for async timers (e.g. 0.05 in unit tests). */
  timeScale?: number;
  /** Simulate model downloads failing mid-flight. */
  failModelDownloads?: boolean;
}

type Listener<E extends EventName> = (payload: Events[E]) => void;

export interface MockBackend {
  invoke<K extends CommandName>(
    command: K,
    args: Commands[K]["args"],
  ): Promise<Commands[K]["result"]>;
  listen<E extends EventName>(event: E, handler: Listener<E>): Promise<() => void>;
  emit<E extends EventName>(event: E, payload: Events[E]): void;
  getInitialSplitProgress(): Record<string, SplitProgress>;
  getInitialPlaybackLevels(): Partial<Record<SourceId, number>>;
  dispose(): void;
}

function makeError(code: IpcError["code"], message: string): IpcError {
  return { code, message };
}

function clone<T>(value: T): T {
  return structuredClone(value);
}

function fileNameFromPath(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

function extensionFromName(fileName: string): string {
  const idx = fileName.lastIndexOf(".");
  return idx === -1 ? "" : fileName.slice(idx + 1).toLowerCase();
}

function titleFromFileName(fileName: string): string {
  const idx = fileName.lastIndexOf(".");
  const base = idx === -1 ? fileName : fileName.slice(0, idx);
  return base.replace(/[_-]+/g, " ").trim() || "Untitled track";
}

function codecFromExt(ext: string): string {
  switch (ext) {
    case "mp3":
      return "MP3";
    case "wav":
      return "WAV";
    case "flac":
      return "FLAC";
    case "opus":
      return "Opus";
    case "ogg":
      return "OGG";
    case "m4a":
    case "aac":
      return "AAC";
    case "aiff":
    case "aif":
      return "AIFF";
    default:
      return "Audio";
  }
}

export function requiredModelIdsForSplit(options: SplitOptions): string[] {
  const ids = [options.quality === "best" ? "scnet_large" : "scnet_small"];
  if (options.mode === "all") {
    ids.push("htdemucs_6s");
  }
  return ids;
}

export function createMockBackend(opts: MockBackendOptions = {}): MockBackend {
  const scene = opts.scene ?? null;
  const timeScale = opts.timeScale ?? 1;
  const failModelDownloads = opts.failModelDownloads ?? false;
  const listeners = new Map<EventName, Set<(payload: unknown) => void>>();

  const models: ModelStatus[] = createInitialModels();
  let settings: Settings = createInitialSettings();
  const sessions = new Map<string, Session>();
  const artwork = new Map<string, string>(Object.entries(MOCK_ARTWORK));
  const initialSplitProgress: Record<string, SplitProgress> = {};
  let initialPlaybackLevels: Partial<Record<SourceId, number>> = {};

  if (scene !== "home-empty" && scene !== "home-first-run") {
    for (const s of createInitialSessions()) {
      sessions.set(s.id, s);
    }
  }

  if (scene === "home-first-run") {
    for (const m of models) {
      m.installed = false;
    }
  }

  if (scene === "scan-downloading") {
    const small = models.find((m) => m.id === "scnet_small");
    const tagger = models.find((m) => m.id === "tagger");
    if (small) {
      small.installed = false;
      // 16_357_786 / (20_971_520 + 19_922_944) = 40.0% overall across the two scan models
      small.downloadProgress = 0.78;
    }
    if (tagger) {
      tagger.installed = false;
      tagger.downloadProgress = 0;
    }
    const target = sessions.get("velvet-static");
    if (target) {
      target.analysis = "pending";
      target.detections = [];
    }
  }

  if (scene === "scan-unavailable") {
    const small = models.find((m) => m.id === "scnet_small");
    const tagger = models.find((m) => m.id === "tagger");
    if (small) {
      small.installed = false;
      small.downloadProgress = null;
    }
    if (tagger) {
      tagger.installed = false;
      tagger.downloadProgress = null;
    }
    const target = sessions.get("velvet-static");
    if (target) {
      target.analysis = "unavailable";
      target.detections = [];
    }
  }

  // Scene-specific deterministic snapshots
  if (scene === "home" || scene === null) {
    initialSplitProgress["silver-lines"] = {
      sessionId: "silver-lines",
      stage: "separating",
      fraction: 0.58,
      label: "Separating drums",
      step: 3,
      stepCount: 5,
      etaSec: 42,
    };
  }

  if (scene === "splitting") {
    initialSplitProgress["silver-lines"] = {
      sessionId: "silver-lines",
      stage: "separating",
      fraction: 0.44,
      label: "Separating drums",
      step: 2,
      stepCount: 5,
      etaSec: 58,
    };
  }

  if (scene === "error") {
    const failed = sessions.get("velvet-static");
    if (failed) {
      failed.status = "failed";
      failed.error = COPY.notEnoughMemory;
    }
  }

  // For ready / ready-solo / export / settings scenes, keep htdemucs_6s uninstalled in settings
  // so Settings clearly demonstrates the restricted Meta model download flow, while keeping
  // Golden Hour already split from an earlier run.

  let playback: PlaybackState = {
    sessionId: null,
    playing: false,
    positionSec: 0,
    durationSec: 0,
    source: "original",
    loop: null,
    masterGainDb: 0,
    channels: [],
  };

  const applyReadyPlaybackSnapshot = (soloed: boolean) => {
    const s = sessions.get("golden-hour");
    if (!s) return;
    playback = {
      sessionId: s.id,
      playing: false,
      positionSec: 84.3,
      durationSec: s.track.durationSec,
      source: "stems",
      loop: null,
      masterGainDb: 0,
      channels: s.stems.map((stem) => {
        if (!soloed) {
          return { id: stem.id, gainDb: 0, muted: false, solo: false };
        }
        return {
          id: stem.id,
          gainDb: stem.id === "bass" ? -4.5 : stem.id === "other" ? 2.0 : 0,
          muted: stem.id === "bass",
          solo: stem.id === "drums",
        };
      }),
    };
    initialPlaybackLevels = soloed
      ? { vocals: 0.62, drums: 0.81, bass: 0, guitar: 0.44, piano: 0, other: 0.38, mix: 0.74 }
      : { vocals: 0.64, drums: 0.78, bass: 0.56, guitar: 0.48, piano: 0, other: 0.36, mix: 0.72 };
  };

  if (scene === "ready" || scene === "export" || scene === "settings" || scene === "update") {
    applyReadyPlaybackSnapshot(false);
  } else if (scene === "loop") {
    applyReadyPlaybackSnapshot(false);
    playback.loop = { startSec: 52, endSec: 128 };
  } else if (scene === "ready-solo") {
    applyReadyPlaybackSnapshot(true);
  } else if (
    scene === "imported" ||
    scene === "consent" ||
    scene === "error" ||
    scene === "scan-downloading" ||
    scene === "scan-unavailable"
  ) {
    const s = sessions.get("velvet-static");
    if (s) {
      playback = {
        sessionId: s.id,
        playing: false,
        positionSec: 52.4,
        durationSec: s.track.durationSec,
        source: "original",
        loop: null,
        masterGainDb: 0,
        channels: [],
      };
    }
  } else if (scene === "splitting") {
    const s = sessions.get("silver-lines");
    if (s) {
      playback = {
        sessionId: s.id,
        playing: false,
        positionSec: 28.4,
        durationSec: s.track.durationSec,
        source: "original",
        loop: null,
        masterGainDb: 0,
        channels: [],
      };
    }
  }

  let playbackTimer: ReturnType<typeof setInterval> | null = null;
  let lastTickWallMs = 0;
  const activeTimers = new Set<ReturnType<typeof setTimeout>>();
  const activeSplits = new Map<
    string,
    {
      cancelled: boolean;
      prevStatus: Session["status"];
      prevStems: StemInfo[];
      prevSplit: Session["split"];
    }
  >();
  let downloadCancelled = false;
  let importCounter = 1;

  function emit<E extends EventName>(event: E, payload: Events[E]): void {
    const set = listeners.get(event);
    if (!set) return;
    for (const fn of set) {
      fn(clone(payload));
    }
  }

  function scheduleTimeout(fn: () => void, ms: number): ReturnType<typeof setTimeout> {
    const handle = setTimeout(
      () => {
        activeTimers.delete(handle);
        fn();
      },
      Math.max(1, Math.round(ms * timeScale)),
    );
    activeTimers.add(handle);
    return handle;
  }

  function computeSessionSizeBytes(s: Session): number {
    const base = s.track.fileSizeBytes;
    const stemsBytes = s.stems.length * Math.round(s.track.durationSec * 352_800);
    return base + stemsBytes;
  }

  function toSummary(s: Session): SessionSummary {
    return {
      id: s.id,
      createdAt: s.createdAt,
      title: s.track.title,
      artist: s.track.artist,
      durationSec: s.track.durationSec,
      status: s.status,
      stemIds: s.stems.map((st) => st.id),
      hasArtwork: s.track.hasArtwork,
      sizeBytes: computeSessionSizeBytes(s),
    };
  }

  function listSummaries(): SessionSummary[] {
    return Array.from(sessions.values())
      .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt))
      .map(toSummary);
  }

  function buildAppInfo(): AppInfo {
    const totalBytes = Array.from(sessions.values()).reduce(
      (acc, s) => acc + computeSessionSizeBytes(s),
      0,
    );
    return createInitialAppInfo(totalBytes);
  }

  function stopPlaybackTimer(): void {
    if (playbackTimer !== null) {
      clearInterval(playbackTimer);
      playbackTimer = null;
    }
  }

  function computeTickLevels(posSec: number): { id: SourceId; peak: number }[] {
    const session = playback.sessionId ? sessions.get(playback.sessionId) : null;
    if (!session) return [];
    const frac = playback.durationSec > 0 ? Math.min(0.999, posSec / playback.durationSec) : 0;
    const masterGain = Math.pow(10, playback.masterGainDb / 20);

    if (playback.source === "original" || session.stems.length === 0) {
      const env = 0.35 + 0.45 * Math.abs(Math.sin(posSec * 3.7));
      return [{ id: "mix", peak: Math.min(1, env * masterGain) }];
    }

    const anySolo = playback.channels.some((c) => c.solo);
    let mixPeak = 0;
    const levels: { id: SourceId; peak: number }[] = [];

    for (const stem of session.stems) {
      const ch = playback.channels.find((c) => c.id === stem.id);
      const audible = stem.present && !!ch && !ch.muted && (!anySolo || ch.solo);
      if (!audible || !ch) {
        levels.push({ id: stem.id, peak: 0 });
        continue;
      }
      const bin = Math.min(stem.activity.length - 1, Math.floor(frac * stem.activity.length));
      const base = stem.activity[bin] ?? 0;
      const flutter = 0.82 + 0.18 * Math.abs(Math.sin(posSec * 9.3 + stem.id.length));
      const gain = Math.pow(10, ch.gainDb / 20);
      const peak = Math.min(1, base * flutter * gain * masterGain);
      levels.push({ id: stem.id, peak });
      mixPeak = Math.max(mixPeak, peak);
    }
    levels.push({ id: "mix", peak: mixPeak });
    return levels;
  }

  function startPlaybackTimer(): void {
    stopPlaybackTimer();
    lastTickWallMs = Date.now();
    playbackTimer = setInterval(
      () => {
        if (!playback.playing) {
          stopPlaybackTimer();
          return;
        }
        const now = Date.now();
        const dt = Math.max(0, (now - lastTickWallMs) / 1000);
        lastTickWallMs = now;
        let nextPos = playback.positionSec + dt;

        if (playback.loop && nextPos >= playback.loop.endSec) {
          nextPos = playback.loop.startSec;
        } else if (nextPos >= playback.durationSec) {
          playback.positionSec = playback.durationSec;
          playback.playing = false;
          stopPlaybackTimer();
          emit("playback:tick", { positionSec: playback.positionSec, levels: [] });
          emit("playback:state", playback);
          return;
        }

        playback.positionSec = nextPos;
        emit("playback:tick", {
          positionSec: playback.positionSec,
          levels: computeTickLevels(playback.positionSec),
        });
      },
      Math.max(16, Math.round(33 * timeScale)),
    );
  }

  function makeStemsForSplit(session: Session, mode: SplitMode): StemInfo[] {
    const stemIds = STEMS_BY_MODE[mode];
    const detByFamily = new Map<StemId, Detection[]>();
    for (const d of session.detections) {
      const list = detByFamily.get(d.family) ?? [];
      list.push(d);
      detByFamily.set(d.family, list);
    }

    return stemIds.map((id) => {
      const contains =
        id === "instrumental"
          ? session.detections.filter((d) => d.family !== "vocals").slice(0, 3)
          : (detByFamily.get(id) ?? []);
      // Make piano absent on tracks that had no piano detection when splitting in 6-stem mode
      const present =
        id === "other" || id === "instrumental"
          ? true
          : contains.length > 0 || (id !== "piano" && id !== "guitar");
      return {
        id,
        path: `/Users/demo/Library/Application Support/SplitSound/library/${session.id}/${id}.wav`,
        present,
        activeFraction: present ? 0.68 : 0.01,
        peakDb: present ? -1.9 : -68,
        rmsDb: present ? -14.8 : -75,
        activity: generateActivity(session.id, id, present),
        contains,
      };
    });
  }

  async function runModelDownload(modelId: string, shouldFail = false): Promise<boolean> {
    const model = models.find((m) => m.id === modelId);
    if (!model || model.installed) return true;
    const steps = 6;
    for (let i = 1; i <= steps; i++) {
      await new Promise<void>((resolve) => scheduleTimeout(resolve, 140));
      if (downloadCancelled) {
        model.downloadProgress = null;
        emit("models:changed", models);
        return false;
      }
      const receivedBytes = Math.round((model.downloadBytes * i) / steps);
      model.downloadProgress = i / steps;
      emit("models:progress", {
        modelId: model.id,
        receivedBytes,
        totalBytes: model.downloadBytes,
      });
      if ((shouldFail || failModelDownloads) && i === 2) {
        model.downloadProgress = null;
        model.installed = false;
        emit("models:changed", models);
        return false;
      }
    }
    model.installed = true;
    model.downloadProgress = null;
    emit("models:changed", models);
    return true;
  }

  function completeInstrumentScan(session: Session): void {
    session.analysis = "done";
    session.detections = [
      { label: "Singing", family: "vocals", confidence: 0.93 },
      { label: "Drum kit", family: "drums", confidence: 0.89 },
      { label: "Electric guitar", family: "guitar", confidence: 0.84 },
      { label: "Bass guitar", family: "bass", confidence: 0.8 },
      { label: "Synthesizer", family: "other", confidence: 0.62 },
    ];
    session.suggestedMode = "all";
    emit("session:updated", session);
  }

  function rescanWaitingSessionsIfReady(): void {
    const small = models.find((m) => m.id === "scnet_small");
    const tagger = models.find((m) => m.id === "tagger");
    if (!small?.installed || !tagger?.installed) return;
    for (const s of sessions.values()) {
      if (s.status === "imported" && (s.analysis === "unavailable" || s.analysis === "pending")) {
        s.analysis = "pending";
        emit("session:updated", s);
        scheduleTimeout(() => {
          const cur = sessions.get(s.id);
          if (cur && cur.analysis === "pending") {
            completeInstrumentScan(cur);
          }
        }, 400);
      }
    }
  }

  async function runSplitJob(
    session: Session,
    options: SplitOptions,
    neededModels: ModelStatus[],
  ): Promise<void> {
    const tracker = activeSplits.get(session.id);
    if (!tracker) return;

    const missing = neededModels.filter((m) => !m.installed);
    const stemIds = STEMS_BY_MODE[options.mode];
    const hasDownload = missing.length > 0;
    const steps: { stage: SplitProgress["stage"]; label: string; durationMs: number }[] = [];

    if (hasDownload) {
      for (const m of missing) {
        steps.push({
          stage: "downloading",
          label: `Downloading ${m.name}`,
          durationMs: 700,
        });
      }
    }
    steps.push({ stage: "preparing", label: "Preparing audio", durationMs: 360 });
    for (const stemId of stemIds.slice(0, Math.min(3, stemIds.length))) {
      steps.push({
        stage: "separating",
        label: `Separating ${stemName(stemId).toLowerCase()}`,
        durationMs: 650,
      });
    }
    steps.push({ stage: "analyzing", label: "Checking stems", durationMs: 420 });
    steps.push({ stage: "writing", label: "Saving stems", durationMs: 340 });

    const stepCount = steps.length;
    const totalEstSec = options.quality === "best" ? 210 : 42;
    let missingIdx = 0;

    for (let idx = 0; idx < stepCount; idx++) {
      if (tracker.cancelled) return;
      const step = steps[idx];
      const baseFrac = idx / stepCount;
      const span = 1 / stepCount;

      if (step.stage === "downloading" && missing[missingIdx]) {
        const model = missing[missingIdx++];
        const subSteps = 5;
        for (let k = 1; k <= subSteps; k++) {
          if (tracker.cancelled) return;
          const subFrac = baseFrac + (span * k) / subSteps;
          const receivedBytes = Math.round((model.downloadBytes * k) / subSteps);
          model.downloadProgress = k / subSteps;
          emit("models:progress", {
            modelId: model.id,
            receivedBytes,
            totalBytes: model.downloadBytes,
          });
          emit("split:progress", {
            sessionId: session.id,
            stage: "downloading",
            fraction: Number(subFrac.toFixed(3)),
            label: step.label,
            step: idx + 1,
            stepCount,
            etaSec: Math.max(2, Math.round(totalEstSec * (1 - subFrac))),
          });
          await new Promise<void>((resolve) =>
            scheduleTimeout(resolve, step.durationMs / subSteps),
          );
          if (failModelDownloads && k === 2) {
            model.downloadProgress = null;
            model.installed = false;
            session.status = "failed";
            session.error = COPY.modelDownloadFailed;
            activeSplits.delete(session.id);
            emit("models:changed", models);
            emit("session:updated", session);
            emit("library:changed", listSummaries());
            return;
          }
        }
        if (tracker.cancelled) {
          model.downloadProgress = null;
          emit("models:changed", models);
          return;
        }
        model.installed = true;
        model.downloadProgress = null;
        emit("models:changed", models);
        continue;
      }

      const subSteps = 4;
      for (let k = 1; k <= subSteps; k++) {
        if (tracker.cancelled) return;
        const subFrac = baseFrac + (span * k) / subSteps;
        emit("split:progress", {
          sessionId: session.id,
          stage: step.stage,
          fraction: Number(Math.min(0.99, subFrac).toFixed(3)),
          label: step.label,
          step: idx + 1,
          stepCount,
          etaSec: Math.max(1, Math.round(totalEstSec * (1 - subFrac))),
        });
        await new Promise<void>((resolve) => scheduleTimeout(resolve, step.durationMs / subSteps));
      }
    }

    if (tracker.cancelled) return;

    session.status = "ready";
    session.error = null;
    session.stems = makeStemsForSplit(session, options.mode);
    session.split = {
      mode: options.mode,
      quality: options.quality,
      finishedAt: new Date(getNowMs()).toISOString(),
      elapsedSec: options.quality === "best" ? 184 : 44,
      modelIds: neededModels.map((m) => m.id),
    };
    activeSplits.delete(session.id);

    emit("session:updated", session);
    emit("library:changed", listSummaries());

    if (playback.sessionId === session.id) {
      playback = {
        ...playback,
        source: "stems",
        channels: session.stems.map((st) => ({
          id: st.id,
          gainDb: 0,
          muted: false,
          solo: false,
        })),
      };
      emit("playback:state", playback);
    }
  }

  async function invoke<K extends CommandName>(
    command: K,
    args: Commands[K]["args"],
  ): Promise<Commands[K]["result"]> {
    switch (command) {
      case "list_sessions":
        return clone(listSummaries()) as Commands[K]["result"];

      case "import_track": {
        const { path } = args as Commands["import_track"]["args"];
        const fileName = fileNameFromPath(path);
        const lower = fileName.toLowerCase();
        const ext = extensionFromName(fileName);

        if (lower.startsWith("missing")) {
          throw makeError("file_not_found", "This file could not be found.");
        }
        if (!ext || !AUDIO_EXTENSIONS.includes(ext as (typeof AUDIO_EXTENSIONS)[number])) {
          throw makeError("unsupported_format", COPY.unsupportedFile);
        }
        if (lower.includes("broken") || lower.includes("corrupt")) {
          throw makeError("decode_failed", COPY.damagedFile);
        }
        if (lower.includes("oom") || lower.includes("huge")) {
          throw makeError("out_of_memory", COPY.notEnoughMemory);
        }

        const id = `session-${importCounter++}`;
        const title = titleFromFileName(fileName);
        const newSession: Session = {
          id,
          createdAt: new Date(getNowMs()).toISOString(),
          track: {
            path,
            fileName,
            title,
            artist: "Local recording",
            album: null,
            durationSec: 214,
            sampleRate: 44100,
            channels: 2,
            codec: codecFromExt(ext),
            bitrateKbps: ext === "mp3" ? 320 : 1411,
            fileSizeBytes: 22_400_000,
            hasArtwork: false,
          },
          status: "imported",
          analysis: "pending",
          detections: [],
          suggestedMode: "band",
          split: null,
          stems: [],
          error: null,
        };

        sessions.set(id, newSession);
        emit("library:changed", listSummaries());

        // Fetch missing scan models (scnet_small + tagger) in the background before scanning,
        // matching the real engine's first-open behaviour.
        const simulateFail = lower.includes("offline") || lower.includes("faildownload");
        void (async () => {
          const scanModelIds = ["scnet_small", "tagger"];
          const missingScan = scanModelIds.filter((mid) => {
            const m = models.find((x) => x.id === mid);
            return m && !m.installed;
          });

          if (missingScan.length > 0) {
            downloadCancelled = false;
            for (const mid of missingScan) {
              const ok = await runModelDownload(mid, simulateFail);
              if (!ok) {
                const cur = sessions.get(id);
                if (cur && cur.analysis === "pending") {
                  cur.analysis = "unavailable";
                  emit("session:updated", cur);
                }
                return;
              }
            }
          }

          scheduleTimeout(() => {
            const current = sessions.get(id);
            if (!current || current.analysis !== "pending") return;
            completeInstrumentScan(current);
          }, 500);
        })();

        return clone(newSession) as Commands[K]["result"];
      }

      case "get_session": {
        const { sessionId } = args as Commands["get_session"]["args"];
        const found = sessions.get(sessionId);
        if (!found) {
          throw makeError("not_found", "That song is no longer in the library.");
        }
        return clone(found) as Commands[K]["result"];
      }

      case "delete_session": {
        const { sessionId } = args as Commands["delete_session"]["args"];
        const running = activeSplits.get(sessionId);
        if (running) {
          running.cancelled = true;
          activeSplits.delete(sessionId);
        }
        sessions.delete(sessionId);
        if (playback.sessionId === sessionId) {
          stopPlaybackTimer();
          playback = {
            sessionId: null,
            playing: false,
            positionSec: 0,
            durationSec: 0,
            source: "original",
            loop: null,
            masterGainDb: 0,
            channels: [],
          };
          emit("playback:state", playback);
        }
        emit("library:changed", listSummaries());
        return null as Commands[K]["result"];
      }

      case "clear_library": {
        for (const t of activeSplits.values()) {
          t.cancelled = true;
        }
        activeSplits.clear();
        sessions.clear();
        stopPlaybackTimer();
        playback = {
          sessionId: null,
          playing: false,
          positionSec: 0,
          durationSec: 0,
          source: "original",
          loop: null,
          masterGainDb: 0,
          channels: [],
        };
        emit("playback:state", playback);
        emit("library:changed", []);
        return null as Commands[K]["result"];
      }

      case "get_artwork": {
        const { sessionId } = args as Commands["get_artwork"]["args"];
        return (artwork.get(sessionId) ?? null) as Commands[K]["result"];
      }

      case "get_peaks": {
        const { sessionId, source, buckets } = args as Commands["get_peaks"]["args"];
        const s = sessions.get(sessionId);
        const stem = s?.stems.find((st) => st.id === source);
        const present = stem ? stem.present : true;
        return generatePeaks(sessionId, source, buckets, present) as Commands[K]["result"];
      }

      case "estimate_split": {
        const { sessionId, options } = args as Commands["estimate_split"]["args"];
        const s = sessions.get(sessionId);
        const duration = s?.track.durationSec ?? 230;
        const neededIds = requiredModelIdsForSplit(options);
        const needed = neededIds
          .map((id) => models.find((m) => m.id === id))
          .filter((m): m is ModelStatus => Boolean(m));
        const downloadBytes = needed
          .filter((m) => !m.installed)
          .reduce((sum, m) => sum + m.downloadBytes, 0);
        const baseFactor = options.quality === "best" ? 1.0 : 0.18;
        const modeBonus = options.mode === "all" ? 1.15 : 1.0;
        const estimate: SplitEstimate = {
          estimatedSec: Math.round(duration * baseFactor * modeBonus),
          models: clone(needed),
          downloadBytes,
        };
        return estimate as Commands[K]["result"];
      }

      case "start_split": {
        const { sessionId, options, acceptRestricted } = args as Commands["start_split"]["args"];
        const s = sessions.get(sessionId);
        if (!s) {
          throw makeError("not_found", "That song is no longer in the library.");
        }
        const neededIds = requiredModelIdsForSplit(options);
        const needed = neededIds
          .map((id) => models.find((m) => m.id === id))
          .filter((m): m is ModelStatus => Boolean(m));

        const missingRestricted = needed.find((m) => !m.installed && m.restricted);
        if (missingRestricted && acceptRestricted !== true) {
          throw makeError(
            "terms_required",
            `${missingRestricted.name} requires accepting its licence terms before downloading.`,
          );
        }

        const prevStatus = s.status === "ready" ? "ready" : "imported";
        const prevStems = clone(s.stems);
        const prevSplit = clone(s.split);

        s.status = "splitting";
        s.error = null;
        s.split = {
          mode: options.mode,
          quality: options.quality,
          finishedAt: "",
          elapsedSec: 0,
          modelIds: neededIds,
        };

        activeSplits.set(sessionId, {
          cancelled: false,
          prevStatus,
          prevStems,
          prevSplit,
        });

        emit("session:updated", s);
        emit("library:changed", listSummaries());
        emit("split:progress", {
          sessionId,
          stage: needed.some((m) => !m.installed) ? "downloading" : "preparing",
          fraction: 0.02,
          label: needed.some((m) => !m.installed)
            ? `Downloading ${needed.find((m) => !m.installed)!.name}`
            : "Preparing audio",
          step: 1,
          stepCount: needed.filter((m) => !m.installed).length + 6,
          etaSec: null,
        });

        void runSplitJob(s, options, needed);
        return null as Commands[K]["result"];
      }

      case "cancel_split": {
        const { sessionId } = args as Commands["cancel_split"]["args"];
        const s = sessions.get(sessionId);
        const tracker = activeSplits.get(sessionId);
        if (tracker) {
          tracker.cancelled = true;
          activeSplits.delete(sessionId);
        }
        if (s && (s.status === "splitting" || s.status === "queued")) {
          s.status = tracker ? tracker.prevStatus : s.stems.length > 0 ? "ready" : "imported";
          if (tracker) {
            s.stems = tracker.prevStems;
            s.split = tracker.prevSplit;
          } else if (s.stems.length === 0) {
            s.split = null;
          }
          emit("session:updated", s);
          emit("library:changed", listSummaries());
        }
        return null as Commands[K]["result"];
      }

      case "playback_load": {
        const { sessionId } = args as Commands["playback_load"]["args"];
        const s = sessions.get(sessionId);
        if (!s) {
          throw makeError("not_found", "That song is no longer in the library.");
        }
        if (playback.sessionId === sessionId) {
          return clone(playback) as Commands[K]["result"];
        }
        stopPlaybackTimer();
        const channels: ChannelState[] =
          s.status === "ready"
            ? s.stems.map((st) => ({
                id: st.id,
                gainDb: 0,
                muted: false,
                solo: false,
              }))
            : [];
        playback = {
          sessionId: s.id,
          playing: false,
          positionSec: 0,
          durationSec: s.track.durationSec,
          source: s.status === "ready" ? "stems" : "original",
          loop: null,
          masterGainDb: 0,
          channels,
        };
        emit("playback:state", playback);
        return clone(playback) as Commands[K]["result"];
      }

      case "playback_unload": {
        stopPlaybackTimer();
        playback = {
          sessionId: null,
          playing: false,
          positionSec: 0,
          durationSec: 0,
          source: "original",
          loop: null,
          masterGainDb: 0,
          channels: [],
        };
        emit("playback:state", playback);
        return null as Commands[K]["result"];
      }

      case "playback_play": {
        if (!playback.sessionId) return null as Commands[K]["result"];
        if (playback.positionSec >= playback.durationSec) {
          playback.positionSec = 0;
        }
        playback.playing = true;
        emit("playback:state", playback);
        startPlaybackTimer();
        return null as Commands[K]["result"];
      }

      case "playback_pause": {
        playback.playing = false;
        stopPlaybackTimer();
        emit("playback:state", playback);
        emit("playback:tick", { positionSec: playback.positionSec, levels: [] });
        return null as Commands[K]["result"];
      }

      case "playback_seek": {
        const { positionSec } = args as Commands["playback_seek"]["args"];
        playback.positionSec = Math.max(0, Math.min(playback.durationSec, positionSec));
        lastTickWallMs = Date.now();
        emit("playback:tick", {
          positionSec: playback.positionSec,
          levels: playback.playing ? computeTickLevels(playback.positionSec) : [],
        });
        return null as Commands[K]["result"];
      }

      case "playback_set_channel": {
        const patch = args as Commands["playback_set_channel"]["args"];
        const ch = playback.channels.find((c) => c.id === patch.id);
        if (ch) {
          if (patch.gainDb !== undefined) ch.gainDb = patch.gainDb;
          if (patch.muted !== undefined) ch.muted = patch.muted;
          if (patch.solo !== undefined) ch.solo = patch.solo;
          emit("playback:state", playback);
        }
        return null as Commands[K]["result"];
      }

      case "playback_set_master": {
        const { gainDb } = args as Commands["playback_set_master"]["args"];
        playback.masterGainDb = gainDb;
        emit("playback:state", playback);
        return null as Commands[K]["result"];
      }

      case "playback_set_source": {
        const { source } = args as Commands["playback_set_source"]["args"];
        playback.source = source;
        emit("playback:state", playback);
        return null as Commands[K]["result"];
      }

      case "playback_set_loop": {
        const { loop } = args as Commands["playback_set_loop"]["args"];
        playback.loop = loop;
        emit("playback:state", playback);
        return null as Commands[K]["result"];
      }

      case "export_stems": {
        const { sessionId, options } = args as Commands["export_stems"]["args"];
        const s = sessions.get(sessionId);
        if (!s) {
          throw makeError("not_found", "That song is no longer in the library.");
        }
        await new Promise<void>((resolve) => scheduleTimeout(resolve, 260));
        const fmt = EXPORT_FORMATS.find((f) => f.value === options.format);
        const ext = fmt?.ext ?? "wav";
        const files = options.stemIds.map(
          (stemId) => `${options.destinationDir}/${s.track.title} (${stemName(stemId)}).${ext}`,
        );
        if (options.includeMixdown) {
          files.push(`${options.destinationDir}/${s.track.title} (Mix).${ext}`);
        }
        settings = {
          ...settings,
          exportFormat: options.format,
          exportDir: options.destinationDir,
        };
        return {
          destinationDir: options.destinationDir,
          files,
        } as Commands[K]["result"];
      }

      case "prepare_drag": {
        const { sessionId, stemIds } = args as Commands["prepare_drag"]["args"];
        const s = sessions.get(sessionId);
        const title = s?.track.title ?? "Track";
        return stemIds.map(
          (id) => `/Users/demo/Library/Caches/SplitSound/drag/${title} (${stemName(id)}).wav`,
        ) as Commands[K]["result"];
      }

      case "list_models":
        return clone(models) as Commands[K]["result"];

      case "download_models": {
        const { modelIds, acceptRestricted } = args as Commands["download_models"]["args"];
        const targets = modelIds
          .map((id) => models.find((m) => m.id === id))
          .filter((m): m is ModelStatus => Boolean(m));
        const missingRestricted = targets.find((m) => !m.installed && m.restricted);
        if (missingRestricted && acceptRestricted !== true) {
          throw makeError(
            "terms_required",
            `${missingRestricted.name} requires accepting its licence terms before downloading.`,
          );
        }
        downloadCancelled = false;
        void (async () => {
          for (const m of targets) {
            if (downloadCancelled) break;
            const ok = await runModelDownload(m.id, false);
            if (!ok) break;
          }
          rescanWaitingSessionsIfReady();
        })();
        return null as Commands[K]["result"];
      }

      case "cancel_model_downloads": {
        downloadCancelled = true;
        for (const m of models) {
          m.downloadProgress = null;
        }
        emit("models:changed", models);
        return null as Commands[K]["result"];
      }

      case "remove_model": {
        const { modelId } = args as Commands["remove_model"]["args"];
        const m = models.find((x) => x.id === modelId);
        if (m) {
          m.installed = false;
          m.downloadProgress = null;
          emit("models:changed", models);
        }
        return null as Commands[K]["result"];
      }

      case "import_model_files": {
        const { paths } = args as Commands["import_model_files"]["args"];
        const recognised: string[] = [];
        for (const p of paths) {
          const lower = fileNameFromPath(p).toLowerCase();
          for (const m of models) {
            if (lower.includes(m.id)) {
              m.installed = true;
              m.downloadProgress = null;
              recognised.push(m.id);
            }
          }
        }
        if (recognised.length > 0) {
          emit("models:changed", models);
        }
        return recognised as Commands[K]["result"];
      }

      case "get_settings":
        return clone(settings) as Commands[K]["result"];

      case "update_settings": {
        const { patch } = args as Commands["update_settings"]["args"];
        settings = { ...settings, ...patch };
        return clone(settings) as Commands[K]["result"];
      }

      case "get_app_info":
        return buildAppInfo() as Commands[K]["result"];

      case "reveal_path":
        return null as Commands[K]["result"];

      default:
        throw makeError("internal", `Unknown command: ${String(command)}`);
    }
  }

  async function listen<E extends EventName>(event: E, handler: Listener<E>): Promise<() => void> {
    let set = listeners.get(event);
    if (!set) {
      set = new Set();
      listeners.set(event, set);
    }
    const wrapped = handler as (payload: unknown) => void;
    set.add(wrapped);
    return () => {
      set?.delete(wrapped);
    };
  }

  function dispose(): void {
    stopPlaybackTimer();
    for (const handle of activeTimers) {
      clearTimeout(handle);
    }
    activeTimers.clear();
    listeners.clear();
  }

  return {
    invoke,
    listen,
    emit,
    getInitialSplitProgress: () => clone(initialSplitProgress),
    getInitialPlaybackLevels: () => clone(initialPlaybackLevels),
    dispose,
  };
}
