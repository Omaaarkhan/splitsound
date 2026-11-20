import type { ModelStatus, SplitOptions } from "../ipc/contract";
import { getTransport, invoke, isTauriRuntime, listen, type UnlistenFn } from "../ipc/client";
import { checkForAppUpdate } from "../ipc/platform";
import { COPY, formatIpcError, isIpcError } from "../utils/errors";
import { getScanModels } from "../utils/format";
import {
  applyLibraryChanged,
  applyLibraryProgress,
  refreshLibrary,
  seedLibraryProgress,
} from "./libraryStore";
import {
  applySessionSplitProgress,
  applySessionUpdated,
  closeSession,
  openSession,
  refreshSplitEstimate,
  sessionStore,
} from "./sessionStore";
import {
  applyPlaybackState,
  applyPlaybackTick,
  loadPlaybackSession,
  seedPlaybackLevels,
  unloadPlaybackSession,
} from "./playbackStore";
import {
  applyModelProgress,
  applyModelsChanged,
  downloadModels,
  findUninstalledRestrictedModel,
  markUserCancelledDownloads,
  markUserTriggeredDownloads,
  modelsStore,
  refreshModels,
} from "./modelsStore";
import { refreshAppInfo, refreshSettingsAndInfo, settingsStore } from "./settingsStore";
import { dismissNotice, openPanel, showNotice } from "./uiStore";

let startupUpdateCheckDone = false;

export function resetStartupUpdateCheckForTesting(): void {
  startupUpdateCheckDone = false;
}

export function showUpdateAvailableNotice(
  version: string,
  install: (onProgress?: (fraction: number) => void) => Promise<void>,
): void {
  showNotice(`SplitSound ${version} is available.`, {
    actionLabel: "Update and restart",
    dismissLabel: "Later",
    autoDismissMs: null,
    onAction: () => {
      showNotice("Downloading the update, 0 %", {
        autoDismissMs: null,
        dismissLabel: null,
      });
      void install((fraction) => {
        const pct = Math.max(0, Math.min(100, Math.round(fraction * 100)));
        showNotice(`Downloading the update, ${pct} %`, {
          autoDismissMs: null,
          dismissLabel: null,
        });
      }).catch(() => {
        dismissNotice();
      });
    },
  });
}

export async function runStartupUpdateCheck(
  checker: typeof checkForAppUpdate = checkForAppUpdate,
): Promise<void> {
  if (startupUpdateCheckDone) return;
  startupUpdateCheckDone = true;
  if (!settingsStore.getState().settings.checkForUpdates) return;
  try {
    const res = await checker();
    if (res.status === "available") {
      showUpdateAvailableNotice(res.version, res.install);
    }
  } catch {
    // Missing or failing updater must stay silent at startup
  }
}

export async function openSongById(sessionId: string): Promise<void> {
  try {
    const defaultQuality = settingsStore.getState().settings.defaultQuality;
    const session = await openSession(sessionId, defaultQuality);
    await loadPlaybackSession(session.id);
  } catch (err) {
    showNotice(formatIpcError(err), { tone: "error" });
  }
}

export async function goHome(): Promise<void> {
  closeSession();
  try {
    await unloadPlaybackSession();
  } catch {
    // ignore unload failure
  }
}

export async function importAudioPaths(paths: string[]): Promise<void> {
  if (paths.length === 0) return;
  let firstImportedId: string | null = null;
  for (const path of paths) {
    try {
      const session = await invoke("import_track", { path });
      if (!firstImportedId) {
        firstImportedId = session.id;
      }
    } catch (err) {
      showNotice(formatIpcError(err), { tone: "error" });
    }
  }
  if (firstImportedId) {
    await openSongById(firstImportedId);
  }
}

export async function requestStartSplit(
  sessionId: string,
  options: SplitOptions,
  acceptRestricted = false,
): Promise<void> {
  const est = await invoke("estimate_split", { sessionId, options }).catch(
    () => sessionStore.getState().estimate,
  );
  const restrictedModel: ModelStatus | undefined =
    est?.models.find((m) => m.restricted && !m.installed) ??
    (options.mode === "all" ? findUninstalledRestrictedModel() : undefined);

  if (restrictedModel && !acceptRestricted) {
    openPanel({
      type: "consent",
      request: { kind: "split", sessionId, options, model: restrictedModel },
    });
    return;
  }

  const missingIds = (est?.models ?? []).filter((m) => !m.installed).map((m) => m.id);
  if (missingIds.length > 0) {
    markUserTriggeredDownloads(missingIds);
  }

  try {
    await invoke("start_split", {
      sessionId,
      options,
      acceptRestricted: acceptRestricted || undefined,
    });
  } catch (err) {
    if (missingIds.length > 0) {
      markUserCancelledDownloads(missingIds);
    }
    if (isIpcError(err) && err.code === "terms_required") {
      const model =
        est?.models.find((m) => m.restricted) ??
        modelsStore.getState().models.find((m) => m.restricted);
      if (model) {
        openPanel({
          type: "consent",
          request: { kind: "split", sessionId, options, model },
        });
        return;
      }
    }
    showNotice(formatIpcError(err), { tone: "error" });
  }
}

export async function requestModelDownload(
  model: ModelStatus,
  acceptRestricted = false,
): Promise<void> {
  if (model.restricted && !model.installed && !acceptRestricted) {
    openPanel({
      type: "consent",
      request: { kind: "model", model },
    });
    return;
  }
  markUserTriggeredDownloads([model.id]);
  try {
    await downloadModels([model.id], acceptRestricted || undefined);
  } catch (err) {
    markUserCancelledDownloads([model.id]);
    if (isIpcError(err) && err.code === "terms_required") {
      openPanel({
        type: "consent",
        request: { kind: "model", model },
      });
      return;
    }
    showNotice(formatIpcError(err), { tone: "error" });
  }
}

export async function retryInstrumentScan(): Promise<void> {
  const missingScanIds = getScanModels(modelsStore.getState().models)
    .filter((m) => !m.installed)
    .map((m) => m.id);
  if (missingScanIds.length === 0) return;
  markUserTriggeredDownloads(missingScanIds);
  try {
    await downloadModels(missingScanIds);
  } catch (err) {
    markUserCancelledDownloads(missingScanIds);
    showNotice(formatIpcError(err), { tone: "error" });
  }
}

export async function bootstrapApp(): Promise<UnlistenFn> {
  const unlisteners: UnlistenFn[] = [];

  unlisteners.push(
    await listen("library:changed", (sessions) => {
      applyLibraryChanged(sessions);
      const current = sessionStore.getState().current;
      if (current && !sessions.some((s) => s.id === current.id)) {
        closeSession();
      }
      void refreshAppInfo();
    }),
  );

  unlisteners.push(
    await listen("session:updated", (session) => {
      const prev = sessionStore.getState();
      const wasDownloadingSplit =
        prev.current?.id === session.id &&
        prev.current.status === "splitting" &&
        prev.splitProgress?.stage === "downloading";
      applySessionUpdated(session);
      if (wasDownloadingSplit && session.status === "failed") {
        showNotice(COPY.modelDownloadFailed, { tone: "error" });
      }
    }),
  );

  unlisteners.push(
    await listen("split:progress", (progress) => {
      applyLibraryProgress(progress);
      applySessionSplitProgress(progress);
    }),
  );

  unlisteners.push(
    await listen("playback:state", (state) => {
      applyPlaybackState(state);
    }),
  );

  unlisteners.push(
    await listen("playback:tick", (tick) => {
      applyPlaybackTick(tick);
    }),
  );

  unlisteners.push(
    await listen("models:changed", (models) => {
      const { failedModelIds } = applyModelsChanged(models);
      if (failedModelIds.length > 0) {
        showNotice(COPY.modelDownloadFailed, { tone: "error" });
      }
      const cur = sessionStore.getState().current;
      if (cur && (cur.status === "imported" || cur.status === "failed")) {
        void refreshSplitEstimate(cur.id);
      }
    }),
  );

  unlisteners.push(
    await listen("models:progress", (progress) => {
      applyModelProgress(progress);
    }),
  );

  unlisteners.push(
    await listen("app:open-files", ({ paths }) => {
      void importAudioPaths(paths);
    }),
  );

  await Promise.all([refreshSettingsAndInfo(), refreshModels(), refreshLibrary()]);

  // Seed deterministic scene state when running against the browser mock
  if (!isTauriRuntime()) {
    const transport = await getTransport();
    const initialProgress = transport.getInitialSplitProgress?.();
    if (initialProgress) {
      seedLibraryProgress(initialProgress);
    }

    const scene =
      typeof window !== "undefined"
        ? new URLSearchParams(window.location.search).get("scene")
        : null;

    if (
      scene === "imported" ||
      scene === "consent" ||
      scene === "error" ||
      scene === "scan-downloading" ||
      scene === "scan-unavailable"
    ) {
      await openSongById("velvet-static");
      if (scene === "consent") {
        const restricted = modelsStore.getState().models.find((m) => m.restricted);
        const cur = sessionStore.getState().current;
        if (restricted && cur) {
          openPanel({
            type: "consent",
            request: {
              kind: "split",
              sessionId: cur.id,
              options: sessionStore.getState().splitOptions,
              model: restricted,
            },
          });
        }
      }
    } else if (scene === "splitting") {
      await openSongById("silver-lines");
    } else if (
      scene === "ready" ||
      scene === "ready-solo" ||
      scene === "loop" ||
      scene === "export" ||
      scene === "settings" ||
      scene === "update"
    ) {
      await openSongById("golden-hour");
      const initialLevels = transport.getInitialPlaybackLevels?.();
      if (initialLevels) {
        seedPlaybackLevels(initialLevels);
      }
      if (scene === "export") {
        openPanel({ type: "export" });
      } else if (scene === "settings") {
        openPanel({ type: "settings" });
      } else if (scene === "update") {
        showUpdateAvailableNotice("26.48.0", async (onProgress) => {
          for (const frac of [0.15, 0.4, 0.72, 1]) {
            await new Promise((r) => setTimeout(r, 250));
            onProgress?.(frac);
          }
        });
      }
    }
  } else {
    void runStartupUpdateCheck();
  }

  return () => {
    for (const unlisten of unlisteners) {
      unlisten();
    }
  };
}
