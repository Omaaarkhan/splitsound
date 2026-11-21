import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { setTransportForTesting } from "../src/ipc/client";
import { createMockBackend, type MockBackend } from "../src/ipc/mock/backend";
import {
  bootstrapApp,
  checkForUpdatesManually,
  formatUpdateProgressNotice,
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
    expect(modelsStore.getState().models.length).toBe(5);
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

  it("updates solo and mute audibility accurately and keeps them mutually exclusive", async () => {
    await openSongById("golden-hour");
    const stems = sessionStore.getState().current!.stems;
    const before = playbackStore.getState().playback.channels;
    expect(
      before.every((c) => {
        const present = stems.find((s) => s.id === c.id)?.present ?? false;
        return isChannelAudible(c, before) === present;
      }),
    ).toBe(true);

    // Editing a channel while listening to the original automatically switches to stems
    await backend.invoke("playback_set_source", { source: "original" });
    playbackStore.setState((prev) => ({
      ...prev,
      playback: { ...prev.playback, source: "original" },
    }));
    await toggleChannelMute("vocals");
    expect(playbackStore.getState().playback.source).toBe("stems");
    await toggleChannelMute("vocals");

    // Soloing drums silences vocals
    await toggleChannelSolo("drums");
    const soloed = playbackStore.getState().playback.channels;
    expect(soloed.find((c) => c.id === "drums")).toMatchObject({ solo: true, muted: false });
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

    // Muting drums clears solo on drums and makes vocals audible again
    await toggleChannelMute("drums");
    const muted = playbackStore.getState().playback.channels;
    expect(muted.find((c) => c.id === "drums")).toMatchObject({ solo: false, muted: true });
    expect(
      isChannelAudible(
        muted.find((c) => c.id === "drums"),
        muted,
      ),
    ).toBe(false);
    expect(
      isChannelAudible(
        muted.find((c) => c.id === "vocals"),
        muted,
      ),
    ).toBe(true);

    // Soloing drums again clears mute on drums
    await toggleChannelSolo("drums");
    const resoled = playbackStore.getState().playback.channels;
    expect(resoled.find((c) => c.id === "drums")).toMatchObject({ solo: true, muted: false });

    // With drums and vocals both soloed, muting drums keeps vocals soloed
    await toggleChannelSolo("vocals");
    await toggleChannelMute("drums");
    const twoSoloThenMute = playbackStore.getState().playback.channels;
    expect(twoSoloThenMute.find((c) => c.id === "drums")).toMatchObject({
      solo: false,
      muted: true,
    });
    expect(twoSoloThenMute.find((c) => c.id === "vocals")).toMatchObject({
      solo: true,
      muted: false,
    });
    expect(
      isChannelAudible(
        twoSoloThenMute.find((c) => c.id === "drums"),
        twoSoloThenMute,
      ),
    ).toBe(false);
    expect(
      isChannelAudible(
        twoSoloThenMute.find((c) => c.id === "vocals"),
        twoSoloThenMute,
      ),
    ).toBe(true);
    expect(
      isChannelAudible(
        twoSoloThenMute.find((c) => c.id === "bass"),
        twoSoloThenMute,
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

  it("checks for updates at most once per launch, reports progress, and handles errors", async () => {
    // 1. Progress formatting with and without totalBytes
    expect(
      formatUpdateProgressNotice({ downloadedBytes: 16_000_000, totalBytes: 40_000_000 }),
    ).toBe("Downloading the update, 40 %");
    expect(formatUpdateProgressNotice({ downloadedBytes: 5_000_000, totalBytes: null })).toBe(
      "Downloading the update",
    );

    // 2. Startup check finds 26.48.0 once per launch and installs via IPC + update:progress
    unlisten?.();
    backend.dispose();
    uiStore.reset();
    resetStartupUpdateCheckForTesting();
    backend = createMockBackend({
      timeScale: 0.01,
      availableUpdate: { version: "26.48.0", notes: "Notes not shown" },
    });
    setTransportForTesting(backend);
    unlisten = await bootstrapApp();

    await runStartupUpdateCheck();
    const firstNotice = uiStore.getState().notice;
    expect(firstNotice?.text).toBe("SplitSound 26.48.0 is available.");
    expect(firstNotice?.actionLabel).toBe("Update and restart");
    expect(firstNotice?.dismissLabel).toBe("Later");

    // Calling runStartupUpdateCheck again does not re-check
    uiStore.reset();
    await runStartupUpdateCheck();
    expect(uiStore.getState().notice).toBeNull();

    // Triggering Update and restart streams update:progress into notice
    const seenNotices: string[] = [];
    const unsubUi = uiStore.subscribe(() => {
      const txt = uiStore.getState().notice?.text;
      if (txt) seenNotices.push(txt);
    });
    firstNotice?.onAction?.();
    await new Promise((r) => setTimeout(r, 80));
    unsubUi();
    expect(seenNotices).toContain("Downloading the update, 40 %");

    // 3. Automatic check stays silent on failure; manual check shows error notice
    unlisten?.();
    backend.dispose();
    uiStore.reset();
    resetStartupUpdateCheckForTesting();
    backend = createMockBackend({ timeScale: 0.01, failUpdateCheck: true });
    setTransportForTesting(backend);
    unlisten = await bootstrapApp();

    await runStartupUpdateCheck();
    expect(uiStore.getState().notice).toBeNull();

    const manual = await checkForUpdatesManually();
    expect(manual.status).toBe("error");
    expect(uiStore.getState().notice?.tone).toBe("error");
    expect(uiStore.getState().notice?.text).toBe("Couldn't check for updates. Try again later.");

    // 4. Manual check when up to date returns up-to-date cleanly
    unlisten?.();
    backend.dispose();
    uiStore.reset();
    backend = createMockBackend({ timeScale: 0.01, availableUpdate: null });
    setTransportForTesting(backend);
    unlisten = await bootstrapApp();

    const upToDate = await checkForUpdatesManually();
    expect(upToDate.status).toBe("up-to-date");
    expect(uiStore.getState().notice).toBeNull();
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
