import { afterEach, describe, expect, it } from "vitest";
import type { IpcError, Session, SplitProgress } from "../src/ipc/contract";
import { createMockBackend, type MockBackend } from "../src/ipc/mock/backend";

describe("mock backend state machine", () => {
  let backend: MockBackend | null = null;

  afterEach(() => {
    backend?.dispose();
    backend = null;
  });

  it("provides the expected model line-up and enforces restricted consent", async () => {
    backend = createMockBackend({ timeScale: 0.02 });
    const models = await backend.invoke("list_models", {});
    expect(models.map((m) => m.id)).toEqual([
      "scnet_small",
      "scnet_large",
      "melband_roformer_vocals",
      "htdemucs_6s",
      "tagger",
    ]);

    const small = models.find((m) => m.id === "scnet_small")!;
    const demucs = models.find((m) => m.id === "htdemucs_6s")!;
    expect(small.installed).toBe(true);
    expect(small.source).toBe("GitHub");
    expect(small.restricted).toBe(false);
    expect(demucs.installed).toBe(false);
    expect(demucs.source).toBe("Meta");
    expect(demucs.restricted).toBe(true);
    expect(demucs.license).toBe("Research use (CC BY-NC)");
    expect(demucs.terms).toContain("Meta");

    // Rejects restricted split without acceptRestricted
    await expect(
      backend.invoke("start_split", {
        sessionId: "velvet-static",
        options: { mode: "all", quality: "fast" },
      }),
    ).rejects.toMatchObject<Partial<IpcError>>({ code: "terms_required" });

    // Rejects restricted download_models without acceptRestricted
    await expect(
      backend.invoke("download_models", { modelIds: ["htdemucs_6s"] }),
    ).rejects.toMatchObject<Partial<IpcError>>({ code: "terms_required" });

    // Succeeds when acceptRestricted is true
    const progressEvents: SplitProgress[] = [];
    let finalSession: Session | null = null;
    await backend.listen("split:progress", (p) => progressEvents.push(p));
    await backend.listen("session:updated", (s) => {
      if (s.id === "velvet-static" && s.status === "ready") {
        finalSession = s;
      }
    });

    await backend.invoke("start_split", {
      sessionId: "velvet-static",
      options: { mode: "all", quality: "fast" },
      acceptRestricted: true,
    });

    await new Promise((r) => setTimeout(r, 250));
    expect(progressEvents.length).toBeGreaterThan(2);
    expect(finalSession).not.toBeNull();
    expect(finalSession!.stems.length).toBe(6);

    const updatedModels = await backend.invoke("list_models", {});
    expect(updatedModels.find((m) => m.id === "htdemucs_6s")!.installed).toBe(true);
  });

  it("rejects damaged and unsupported files and runs background instrument scans", async () => {
    backend = createMockBackend({ timeScale: 0.02 });

    await expect(backend.invoke("import_track", { path: "/tmp/broken.mp3" })).rejects.toMatchObject<
      Partial<IpcError>
    >({ code: "decode_failed" });

    await expect(backend.invoke("import_track", { path: "/tmp/notes.txt" })).rejects.toMatchObject<
      Partial<IpcError>
    >({ code: "unsupported_format" });

    let scanned: Session | null = null;
    await backend.listen("session:updated", (s) => {
      if (s.analysis === "done") scanned = s;
    });

    const imported = await backend.invoke("import_track", { path: "/tmp/New_Demo.wav" });
    expect(imported.status).toBe("imported");
    expect(imported.analysis).toBe("pending");
    expect(imported.track.title).toBe("New Demo");

    await new Promise((r) => setTimeout(r, 60));
    expect(scanned).not.toBeNull();
    expect(scanned!.detections.length).toBeGreaterThan(0);
  });

  it("cancels a running split cleanly and restores prior status", async () => {
    backend = createMockBackend({ timeScale: 0.2 });
    await backend.invoke("start_split", {
      sessionId: "velvet-static",
      options: { mode: "band", quality: "fast" },
    });

    const mid = await backend.invoke("get_session", { sessionId: "velvet-static" });
    expect(mid.status).toBe("splitting");

    await backend.invoke("cancel_split", { sessionId: "velvet-static" });
    const after = await backend.invoke("get_session", { sessionId: "velvet-static" });
    expect(after.status).toBe("imported");
  });

  it("advances playback with ticks, handles export and persists settings", async () => {
    backend = createMockBackend({ timeScale: 0.2 });
    const state = await backend.invoke("playback_load", { sessionId: "golden-hour" });
    expect(state.source).toBe("stems");
    expect(state.channels.length).toBe(6);

    const ticks: number[] = [];
    await backend.listen("playback:tick", (t) => ticks.push(t.positionSec));
    await backend.invoke("playback_play", {});
    await new Promise((r) => setTimeout(r, 45));
    await backend.invoke("playback_pause", {});
    expect(ticks.length).toBeGreaterThan(0);

    const exp = await backend.invoke("export_stems", {
      sessionId: "golden-hour",
      options: {
        stemIds: ["vocals", "drums"],
        format: "flac",
        destinationDir: "/Users/demo/Desktop",
        includeMixdown: true,
      },
    });
    expect(exp.files.length).toBe(3);
    expect(exp.files[0]).toContain("(Vocals).flac");
    expect(exp.files[2]).toContain("(Mix).flac");

    const updated = await backend.invoke("update_settings", {
      patch: { acceleration: "cpu", threads: 4 },
    });
    expect(updated.acceleration).toBe("cpu");
    expect(updated.threads).toBe(4);
    const reread = await backend.invoke("get_settings", {});
    expect(reread.threads).toBe(4);
    expect(reread.exportFormat).toBe("flac");
  });

  it("downloads scan models on first-run import and marks unavailable on failure", async () => {
    backend = createMockBackend({ scene: "home-first-run", timeScale: 0.01 });
    const progressModelIds = new Set<string>();
    let finalSession: Session | null = null;
    await backend.listen("models:progress", (p) => progressModelIds.add(p.modelId));
    await backend.listen("session:updated", (s) => {
      if (s.analysis === "done") finalSession = s;
    });

    await backend.invoke("import_track", { path: "/tmp/First_Song.wav" });
    await new Promise((r) => setTimeout(r, 120));
    expect(progressModelIds.has("scnet_small")).toBe(true);
    expect(progressModelIds.has("tagger")).toBe(true);
    expect(finalSession).not.toBeNull();
    expect(finalSession!.detections.length).toBeGreaterThan(0);

    // Simulate download failure on a fresh first-run backend
    backend.dispose();
    backend = createMockBackend({ scene: "home-first-run", timeScale: 0.01 });
    let failedScan: Session | null = null;
    await backend.listen("session:updated", (s) => {
      if (s.analysis === "unavailable") failedScan = s;
    });

    await backend.invoke("import_track", { path: "/tmp/offline_song.mp3" });
    await new Promise((r) => setTimeout(r, 100));
    expect(failedScan).not.toBeNull();
  });

  it("supports check_for_update, install_update, update:progress, and update errors", async () => {
    // Up to date by default; install_update rejects when nothing found
    backend = createMockBackend({ timeScale: 0.01 });
    expect(await backend.invoke("check_for_update", {})).toBeNull();
    await expect(backend.invoke("install_update", {})).rejects.toMatchObject<Partial<IpcError>>({
      code: "not_found",
    });

    // Update available -> emits update:progress with bytes
    backend.dispose();
    backend = createMockBackend({
      timeScale: 0.01,
      availableUpdate: { version: "26.48.0", notes: "Plain release notes." },
    });
    const info = await backend.invoke("check_for_update", {});
    expect(info).toEqual({ version: "26.48.0", notes: "Plain release notes." });

    const seen: { downloadedBytes: number; totalBytes: number | null }[] = [];
    await backend.listen("update:progress", (p) => seen.push(p));
    await backend.invoke("install_update", {});
    expect(seen.length).toBeGreaterThanOrEqual(4);
    expect(seen.some((p) => p.totalBytes === 40_000_000)).toBe(true);

    // Failed update check
    backend.dispose();
    backend = createMockBackend({ timeScale: 0.01, failUpdateCheck: true });
    await expect(backend.invoke("check_for_update", {})).rejects.toMatchObject<Partial<IpcError>>({
      code: "offline",
    });
  });

  it("keeps solo and mute mutually exclusive in playback_set_channel", async () => {
    backend = createMockBackend({ timeScale: 0.01 });
    let latestState = await backend.invoke("playback_load", { sessionId: "golden-hour" });
    await backend.listen("playback:state", (st) => {
      latestState = st;
    });

    await backend.invoke("playback_set_channel", { id: "vocals", solo: true });
    expect(latestState.channels.find((c) => c.id === "vocals")).toMatchObject({
      solo: true,
      muted: false,
    });

    await backend.invoke("playback_set_channel", { id: "vocals", muted: true });
    expect(latestState.channels.find((c) => c.id === "vocals")).toMatchObject({
      solo: false,
      muted: true,
    });

    await backend.invoke("playback_set_channel", { id: "vocals", solo: true });
    expect(latestState.channels.find((c) => c.id === "vocals")).toMatchObject({
      solo: true,
      muted: false,
    });

    // Adjusting a channel while listening to original automatically switches to stems
    await backend.invoke("playback_set_source", { source: "original" });
    expect(latestState.source).toBe("original");
    await backend.invoke("playback_set_channel", { id: "vocals", gainDb: -3 });
    expect(latestState.source).toBe("stems");

    // Re-splitting preserves manual gain/mute on loaded playback
    await backend.invoke("playback_set_channel", { id: "drums", muted: true });
    await backend.invoke("start_split", {
      sessionId: "golden-hour",
      options: { mode: "all", quality: "fast" },
      acceptRestricted: true,
    });
    await new Promise((r) => setTimeout(r, 250));
    expect(latestState.channels.find((c) => c.id === "vocals")?.gainDb).toBe(-3);
    expect(latestState.channels.find((c) => c.id === "drums")?.muted).toBe(true);
    expect(latestState.channels.find((c) => c.id === "piano")?.muted).toBe(true);
  });

  it("supports studio quality splits with RoFormer model plan", async () => {
    backend = createMockBackend({ timeScale: 0.02 });
    const est = await backend.invoke("estimate_split", {
      sessionId: "velvet-static",
      options: { mode: "all", quality: "studio" },
    });
    expect(est.models.map((m) => m.id)).toEqual([
      "melband_roformer_vocals",
      "scnet_large",
      "htdemucs_6s",
    ]);
  });
});
