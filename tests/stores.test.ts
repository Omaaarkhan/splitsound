import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransportForTesting } from "../src/ipc/client";
import { createMockBackend, type MockBackend } from "../src/ipc/mock/backend";
import {
  bootstrapApp,
  importAudioPaths,
  openSongById,
  requestModelDownload,
  requestStartSplit,
  resetStartupUpdateCheckForTesting,
  retryInstrumentScan,
  runStartupUpdateCheck,
} from "../src/stores/index";
import { libraryStore } from "../src/stores/libraryStore";
import {
  cancelModelDownloads,
  modelsStore,
  resetModelDownloadTracking,
} from "../src/stores/modelsStore";
import { COPY } from "../src/utils/errors";
import {
  isChannelAudible,
  playbackStore,
  resetPlaybackRuntime,
  setLoopRegion,
  toggleChannelMute,
  toggleChannelSolo,
  toggleLoop,
} from "../src/stores/playbackStore";
import { sessionStore } from "../src/stores/sessionStore";
import { settingsStore } from "../src/stores/settingsStore";
import { uiStore } from "../src/stores/uiStore";

describe("application stores", () => {
  let backend: MockBackend;
  let unlisten: (() => void) | null = null;

  beforeEach(async () => {
    libraryStore.reset();
    sessionStore.reset();
    resetPlaybackRuntime();
    modelsStore.reset();
    resetModelDownloadTracking();
    settingsStore.reset();
    uiStore.reset();

    backend = createMockBackend({ timeScale: 0.02 });
    setTransportForTesting(backend);
    unlisten = await bootstrapApp();
  });

  afterEach(() => {
    unlisten?.();
    unlisten = null;
    backend.dispose();
    setTransportForTesting(null);
  });

  it("loads library, models, and settings on bootstrap", () => {
    expect(libraryStore.getState().sessions.length).toBeGreaterThanOrEqual(4);
    expect(modelsStore.getState().models.length).toBe(4);
    expect(settingsStore.getState().loaded).toBe(true);
  });

  it("opens consent panel when starting a 6-stem split with restricted model uninstalled", async () => {
    await openSongById("velvet-static");
    expect(sessionStore.getState().current?.id).toBe("velvet-static");

    await requestStartSplit("velvet-static", { mode: "all", quality: "fast" });
    const panel = uiStore.getState().panel;
    expect(panel.type).toBe("consent");
    if (panel.type === "consent") {
      expect(panel.request.kind).toBe("split");
      expect(panel.request.model.id).toBe("htdemucs_6s");
    }

    // Accepting starts the split and transitions to ready with reveal flag
    await requestStartSplit("velvet-static", { mode: "all", quality: "fast" }, true);
    expect(sessionStore.getState().current?.status).toBe("splitting");
    await new Promise((r) => setTimeout(r, 250));
    expect(sessionStore.getState().current?.status).toBe("ready");
    expect(sessionStore.getState().revealing).toBe(true);
  });

  it("opens consent panel when downloading a restricted model from settings", async () => {
    const demucs = modelsStore.getState().models.find((m) => m.id === "htdemucs_6s")!;
    await requestModelDownload(demucs);
    const panel = uiStore.getState().panel;
    expect(panel.type).toBe("consent");
    if (panel.type === "consent") {
      expect(panel.request.kind).toBe("model");
    }
  });

  it("handles app:open-files event and error notices", async () => {
    await importAudioPaths(["/tmp/broken.mp3"]);
    expect(uiStore.getState().notice?.tone).toBe("error");
    expect(uiStore.getState().notice?.text).toContain("damaged");

    backend.emit("app:open-files", { paths: ["/tmp/Late_Take.flac"] });
    await new Promise((r) => setTimeout(r, 80));
    expect(sessionStore.getState().current?.track.title).toBe("Late Take");
  });

  it("updates solo and mute audibility accurately", async () => {
    await openSongById("golden-hour");
    const before = playbackStore.getState().playback.channels;
    expect(before.every((c) => isChannelAudible(c, before))).toBe(true);

    await toggleChannelSolo("drums");
    const soloed = playbackStore.getState().playback.channels;
    expect(
      isChannelAudible(
        soloed.find((c) => c.id === "drums"),
        soloed,
      ),
    ).toBe(true);
    expect(
      isChannelAudible(
        soloed.find((c) => c.id === "vocals"),
        soloed,
      ),
    ).toBe(false);

    await toggleChannelMute("drums");
    const mutedSolo = playbackStore.getState().playback.channels;
    expect(
      isChannelAudible(
        mutedSolo.find((c) => c.id === "drums"),
        mutedSolo,
      ),
    ).toBe(false);
  });

  it("sets section loop regions, latches Loop on/off, and falls back to full song", async () => {
    await openSongById("golden-hour");
    expect(playbackStore.getState().playback.loop).toBeNull();

    // Turning Loop on with no region yet loops the whole song
    await toggleLoop();
    expect(playbackStore.getState().playback.loop).toEqual({ startSec: 0, endSec: 231 });

    // Setting a section loop updates PlaybackState.loop
    await setLoopRegion({ startSec: 40, endSec: 95 });
    expect(playbackStore.getState().playback.loop).toEqual({ startSec: 40, endSec: 95 });

    // Toggling off clears active loop, toggling back on restores last section loop
    await toggleLoop();
    expect(playbackStore.getState().playback.loop).toBeNull();
    await toggleLoop();
    expect(playbackStore.getState().playback.loop).toEqual({ startSec: 40, endSec: 95 });
  });

  it("checks for updates at most once per launch and reports download progress", async () => {
    resetStartupUpdateCheckForTesting();
    let calls = 0;
    const fakeChecker = async () => {
      calls++;
      return {
        status: "available" as const,
        version: "26.48.0",
        install: async (onProgress?: (f: number) => void) => {
          onProgress?.(0.4);
        },
      };
    };

    await runStartupUpdateCheck(fakeChecker);
    await runStartupUpdateCheck(fakeChecker);
    expect(calls).toBe(1);

    const notice = uiStore.getState().notice;
    expect(notice?.text).toBe("SplitSound 26.48.0 is available.");
    expect(notice?.actionLabel).toBe("Update and restart");
    expect(notice?.dismissLabel).toBe("Later");

    notice?.onAction?.();
    await new Promise((r) => setTimeout(r, 10));
    expect(uiStore.getState().notice?.text).toBe("Downloading the update, 40 %");
  });

  it("shows download-failed error notice on silent failure but never on user cancel", async () => {
    // 1. User cancels a Settings download -> NO error notice
    const large = modelsStore.getState().models.find((m) => m.id === "scnet_large")!;
    await requestModelDownload(large);
    await new Promise((r) => setTimeout(r, 15));
    await cancelModelDownloads();
    await new Promise((r) => setTimeout(r, 40));
    expect(uiStore.getState().notice).toBeNull();

    // 2. Backend fails a Settings download mid-flight -> shows COPY.modelDownloadFailed
    unlisten?.();
    backend.dispose();
    modelsStore.reset();
    resetModelDownloadTracking();
    uiStore.reset();
    backend = createMockBackend({ timeScale: 0.01, failModelDownloads: true });
    setTransportForTesting(backend);
    unlisten = await bootstrapApp();

    const largeModel = modelsStore.getState().models.find((m) => m.id === "scnet_large")!;
    await requestModelDownload(largeModel);
    await new Promise((r) => setTimeout(r, 80));
    expect(uiStore.getState().notice?.tone).toBe("error");
    expect(uiStore.getState().notice?.text).toBe(COPY.modelDownloadFailed);

    // 3. Split download fails mid-flight -> shows COPY.modelDownloadFailed
    uiStore.reset();
    await openSongById("velvet-static");
    await requestStartSplit("velvet-static", { mode: "band", quality: "best" });
    await new Promise((r) => setTimeout(r, 120));
    expect(sessionStore.getState().current?.status).toBe("failed");
    expect(uiStore.getState().notice?.tone).toBe("error");
    expect(uiStore.getState().notice?.text).toBe(COPY.modelDownloadFailed);
  });

  it("retries unavailable instrument scan by downloading missing scan models", async () => {
    unlisten?.();
    backend.dispose();
    modelsStore.reset();
    resetModelDownloadTracking();
    sessionStore.reset();
    uiStore.reset();
    backend = createMockBackend({ scene: "scan-unavailable", timeScale: 0.01 });
    setTransportForTesting(backend);
    unlisten = await bootstrapApp();

    await openSongById("velvet-static");
    expect(sessionStore.getState().current?.analysis).toBe("unavailable");

    await retryInstrumentScan();
    await new Promise((r) => setTimeout(r, 160));
    expect(sessionStore.getState().current?.analysis).toBe("done");
    expect(sessionStore.getState().current?.detections.length).toBeGreaterThan(0);
  });
});
