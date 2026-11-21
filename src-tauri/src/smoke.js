async ({ denied, export: destinationDir, song, real, emptyStore }) => {
  const errors = [];
  addEventListener("securitypolicyviolation", (e) => errors.push(`CSP ${e.violatedDirective}`));
  addEventListener("error", (e) => errors.push(e.message || "resource failed to load"), true);
  addEventListener("unhandledrejection", (e) => errors.push(String(e.reason?.message || e.reason)));
  const internals = () => window.__TAURI_INTERNALS__;
  const invoke = (cmd, args) => internals().invoke(cmd, args);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const session = (sessionId) => invoke("get_session", { sessionId });
  function require(condition, message) {
    if (!condition) throw new Error(message);
  }
  async function until(check, description, timeout = 30_000) {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      const value = await check();
      if (value) return value;
      await sleep(40);
    }
    throw new Error(`timed out waiting for ${description}`);
  }
  async function rejects(command, args, code) {
    let rejection;
    try {
      await invoke(command, args);
    } catch (error) {
      rejection = error;
    }
    require(rejection?.code === code, `${command} did not reject with ${code}`);
  }
  async function startSplit(sessionId, mode, acceptRestricted) {
    await invoke("start_split", { sessionId, options: { mode, quality: "fast" }, acceptRestricted });
  }
  async function ready(sessionId, stemCount) {
    return until(async () => {
      const current = await session(sessionId);
      require(current.status !== "failed", current.error || "split failed");
      return current.status === "ready" && current.stems.length === stemCount && current;
    }, `${stemCount} finished stems`, 180_000);
  }
  async function exportAudio(sessionId, format, stemIds) {
    const result = await invoke("export_stems", {
      sessionId,
      options: { stemIds, destinationDir, format, includeMixdown: true },
    });
    require(result.files.length === stemIds.length + 1, `wrong number of ${format} exports`);
  }
  try {
    await until(() => document.body?.innerText.includes("SplitSound smoke"), "the UI to import the OS-open file");
    require(!!document.querySelector("#root")?.firstElementChild, "the React UI did not mount");
    const [{ id: sessionId }] = await invoke("list_sessions");
    await rejects("import_track", { path: denied }, "file_not_found");
    await rejects("download_models", { modelIds: ["htdemucs_6s"] }, "terms_required");
    if (!real || emptyStore) {
      await rejects("start_split", { sessionId, options: { mode: "all", quality: "fast" } }, "terms_required");
    }
    if (real) {
      const imported = await session(sessionId);
      require(imported.track.durationSec >= 20, "the compressed excerpt is too short");
      const info = await invoke("get_app_info");
      require(!info.gpu.available && info.gpu.name === null, "GPU must be disabled");
      await invoke("update_settings", { patch: { acceleration: "gpu", threads: 4 } });
      require((await invoke("get_app_info")).cpuThreads === 4, "thread settings were not applied");
    }

    const state = await invoke("playback_load", { sessionId });
    require(state.source === "original", "pre-split playback must use the original");
    let heard = 0;
    await invoke("plugin:event|listen", {
      event: "playback:tick",
      target: { kind: "Any" },
      handler: internals().transformCallback(({ payload }) => {
        if (payload.levels.some(({ id, peak }) => id === "mix" && peak > 0)) heard = payload.positionSec;
      }),
    });
    let detections;
    if (real && !emptyStore) {
      const scanned = await until(async () => {
        const current = await session(sessionId);
        require(current.analysis === "pending" || current.analysis === "done", `instrument scan ${current.analysis}`);
        return current.analysis === "done" && current;
      }, "the real instrument scan", 180_000);
      detections = scanned.detections.map(({ label, confidence }) => [label, Number(confidence.toFixed(3))]);
      require(["bass", "drums"].every((family) =>
        scanned.detections.some((d) => d.family === family && d.confidence > 0.1)
      ), "the preview did not hear bass and drums in the reference music");
    }
    if (!emptyStore) await startSplit(sessionId, "band", false);
    let audio = "with audio";
    try {
      await invoke("playback_play");
    } catch (error) {
      if (error?.code !== "audio_device") throw error;
      audio = "without an audio device";
    }
    if (audio === "with audio") {
      await until(() => heard >= 1, "one second of audible playback ticks");
      await invoke("playback_pause");
    }
    if (emptyStore) {
      await until(async () => {
        const entry = await session(sessionId);
        require(entry.analysis !== "failed", "a missing preview must be unavailable");
        return entry.analysis === "unavailable" && entry.status === "imported";
      }, "the unavailable preview", 100_000);
      const models = await invoke("list_models");
      require(!models.find((model) => model.id === "htdemucs_6s")?.installed, "Demucs was installed without consent");
      require(errors.length === 0, errors.join(" | "));
      document.title = "SplitSound smoke result passed consent and unavailable preview";
      return;
    }

    await ready(sessionId, 4);
    if (real) {
      await invoke("playback_load", { sessionId });
      await invoke("playback_set_channel", { id: "vocals", gainDb: -6 });
      await invoke("playback_set_channel", { id: "drums", muted: true });
      for (const format of ["flac", "mp3"]) {
        await exportAudio(sessionId, format, ["vocals", "drums"]);
      }
    } else {
      await exportAudio(sessionId, "wav16", ["vocals"]);
      const drag = await invoke("prepare_drag", { sessionId, stemIds: ["vocals"] });
      require(drag[0]?.endsWith("SplitSound smoke (Vocals).wav"), "drag file has the wrong name");
    }
    let result = `passed ${audio}`;
    if (real) {
      const { id: all } = await invoke("import_track", { path: song });
      document.title = "SplitSound smoke phase all";
      await startSplit(all, "all", true);
      await ready(all, 6);
      result = `passed real ${JSON.stringify({ band: sessionId, all, audio, detections })}`;
    }
    require(errors.length === 0, errors.join(" | "));
    document.title = `SplitSound smoke result ${result}`;
  } catch (error) {
    const detail = error?.message || String(error);
    document.title = `SplitSound smoke result failed: ${detail} ${errors.join(" | ")}`;
  }
}
