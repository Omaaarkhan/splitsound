import { describe, expect, it } from "vitest";
import {
  computeFirstRunDownloadBytes,
  computeModelDownloadFraction,
  formatBytes,
  formatDb,
  formatDurationEstimate,
  formatElapsed,
  formatEta,
  formatImportedModelsNotice,
  formatModelMeta,
  formatSessionDate,
  formatTime,
  formatTimeTenths,
  getScanModels,
  rulerTickInterval,
} from "../src/utils/format";
import { createInitialModels } from "../src/ipc/mock/fixtures";
import { dbToFaderFraction, faderFractionToDb, snapDb } from "../src/utils/fader";
import { decodePeaks, encodePeaks } from "../src/utils/waveform";
import { sortStemsForDisplay } from "../src/utils/stems";
import { COPY, formatIpcError } from "../src/utils/errors";
import { MOCK_NOW_MS } from "../src/utils/clock";

describe("formatters", () => {
  it("formats time and tenths with tabular stability", () => {
    expect(formatTime(0)).toBe("0:00");
    expect(formatTime(84.3)).toBe("1:24");
    expect(formatTime(231)).toBe("3:51");
    expect(formatTimeTenths(0)).toBe("0:00.0");
    expect(formatTimeTenths(84.3)).toBe("1:24.3");
    expect(formatTimeTenths(-5)).toBe("0:00.0");
  });

  it("formats decibels cleanly", () => {
    expect(formatDb(0)).toBe("0.0");
    expect(formatDb(-4.5)).toBe("-4.5");
    expect(formatDb(2)).toBe("+2.0");
    expect(formatDb(-60)).toBe("-60");
  });

  it("formats byte sizes, elapsed seconds, estimates, and dates", () => {
    expect(formatBytes(57_671_680)).toBe("55 MB");
    expect(formatBytes(41_943_040)).toBe("40 MB");
    expect(formatBytes(2_147_483_648)).toBe("2.0 GB");
    expect(formatElapsed(48)).toBe("48 s");
    expect(formatElapsed(120)).toBe("2 m");
    expect(formatElapsed(198)).toBe("3 m 18 s");
    expect(formatDurationEstimate(41, "fast")).toBe("About 40 seconds");
    expect(formatDurationEstimate(235, "best")).toBe("About 4 minutes, cleaner result");
    expect(formatDurationEstimate(520, "studio")).toBe("About 9 minutes, cleanest vocals");
    expect(formatEta(null)).toBe("Estimating time left");
    expect(formatEta(58)).toBe("About a minute left");
    expect(formatSessionDate("2026-11-19T18:00:00.000Z", MOCK_NOW_MS)).toBe("Today");
    expect(formatSessionDate("2026-11-18T12:00:00.000Z", MOCK_NOW_MS)).toBe("Yesterday");
    expect(formatSessionDate("2026-11-14T12:00:00.000Z", MOCK_NOW_MS)).toBe("Nov 14");
    expect(rulerTickInterval(231)).toBe(30);
    expect(rulerTickInterval(40)).toBe(5);
  });

  it("formats model metadata without middle dots, import notices, and first-run scan size", () => {
    const models = createInitialModels();
    const small = models.find((m) => m.id === "scnet_small")!;
    const demucs = models.find((m) => m.id === "htdemucs_6s")!;
    expect(formatModelMeta(small)).toBe("21 MB, from GitHub, MIT licence");
    expect(formatModelMeta(demucs)).toBe("55 MB, from Meta, research use (CC BY-NC)");
    expect(formatModelMeta(small)).not.toContain("·");

    expect(formatImportedModelsNotice(0)).toBe("None of those files is a SplitSound model.");
    expect(formatImportedModelsNotice(1)).toBe("Added 1 model");
    expect(formatImportedModelsNotice(2)).toBe("Added 2 models");

    // Recognises both mock "tagger" and real engine "efficientat_mn10_as" plus "scnet_small"
    const realIds = models.map((m) =>
      m.id === "tagger" ? { ...m, id: "efficientat_mn10_as", name: "Instrument tagger" } : m,
    );
    expect(getScanModels(realIds).map((m) => m.id)).toEqual(["scnet_small", "efficientat_mn10_as"]);

    // When all models are uninstalled, scnet_small (20 MB) + detector (19 MB) = 39 MB
    const noneInstalled = models.map((m) => ({ ...m, installed: false }));
    expect(formatBytes(computeFirstRunDownloadBytes(noneInstalled))).toBe("39 MB");
    expect(computeFirstRunDownloadBytes(models)).toBe(0);

    // Overall progress across two scan models (sequentially downloading)
    const downloading = noneInstalled.map((m) =>
      m.id === "scnet_small" ? { ...m, downloadProgress: 0.78 } : m,
    );
    const frac = computeModelDownloadFraction(downloading, {}, new Set(["scnet_small", "tagger"]));
    expect(Math.round((frac ?? 0) * 100)).toBe(40);
  });

  it("maps IPC error codes to spec copy", () => {
    expect(formatIpcError({ code: "unsupported_format", message: "raw" })).toBe(
      COPY.unsupportedFile,
    );
    expect(formatIpcError({ code: "decode_failed", message: "raw" })).toBe(COPY.damagedFile);
    expect(formatIpcError({ code: "model_download_failed", message: "raw" })).toBe(
      COPY.modelDownloadFailed,
    );
    expect(formatIpcError({ code: "out_of_memory", message: "raw" })).toBe(COPY.notEnoughMemory);
    expect(formatIpcError({ code: "audio_device", message: "raw" })).toBe(COPY.noAudioDevice);
    expect(formatIpcError({ code: "cancelled", message: "raw" })).toBe(COPY.splitCancelled);
  });
});

describe("fader taper and peaks encoding", () => {
  it("maps 0 dB near 68%, -4.5 dB near 55%, +2 dB near 74% and round-trips", () => {
    expect(dbToFaderFraction(0)).toBeCloseTo(0.68, 2);
    expect(dbToFaderFraction(-4.5)).toBeCloseTo(0.545, 2);
    expect(dbToFaderFraction(2)).toBeCloseTo(0.744, 2);
    expect(faderFractionToDb(dbToFaderFraction(0))).toBe(0);
    expect(faderFractionToDb(dbToFaderFraction(-4.5))).toBe(-4.5);
    expect(faderFractionToDb(dbToFaderFraction(2))).toBe(2);
    expect(snapDb(1.3)).toBe(1.5);
  });

  it("encodes and decodes signed byte waveform peaks", () => {
    const min = new Float32Array([-0.5, -1, 0, -0.25]);
    const max = new Float32Array([0.5, 1, 0, 0.75]);
    const encoded = encodePeaks(min, max);
    const decoded = decodePeaks(encoded);
    expect(decoded.buckets).toBe(4);
    expect(decoded.min[1]).toBeCloseTo(-1, 2);
    expect(decoded.max[1]).toBeCloseTo(1, 2);
    expect(decoded.max[3]).toBeCloseTo(0.75, 1);
  });

  it("places absent stems at the bottom while preserving canonical order", () => {
    const sorted = sortStemsForDisplay([
      { id: "other", present: true },
      { id: "piano", present: false },
      { id: "vocals", present: true },
      { id: "drums", present: true },
    ]);
    expect(sorted.map((s) => s.id)).toEqual(["vocals", "drums", "other", "piano"]);
  });
});
