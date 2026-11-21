import type { ModelDownloadProgress, ModelStatus, Quality } from "../ipc/contract";

export const QUALITY_LABEL: Record<Quality, string> = {
  fast: "Fast",
  best: "Best",
  studio: "Studio",
};

export function formatTime(seconds: number): string {
  const clamped = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const total = Math.floor(clamped);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

export function formatTimeTenths(seconds: number): string {
  const clamped = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const tenthsTotal = Math.floor(clamped * 10 + 1e-6);
  const tenths = tenthsTotal % 10;
  const totalSecs = Math.floor(tenthsTotal / 10);
  const mins = Math.floor(totalSecs / 60);
  const secs = totalSecs % 60;
  return `${mins}:${String(secs).padStart(2, "0")}.${tenths}`;
}

export function formatDb(db: number): string {
  if (!Number.isFinite(db) || db <= -59.5) {
    return "-60";
  }
  const rounded = Math.round(db * 2) / 2;
  if (Math.abs(rounded) < 0.05) {
    return "0.0";
  }
  const sign = rounded > 0 ? "+" : "";
  return `${sign}${rounded.toFixed(1)}`;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  const mb = bytes / (1024 * 1024);
  if (mb < 1) {
    const kb = Math.max(1, Math.round(bytes / 1024));
    return `${kb} KB`;
  }
  if (mb < 1024) {
    return `${Math.round(mb)} MB`;
  }
  const gb = mb / 1024;
  return `${gb.toFixed(gb >= 10 ? 0 : 1)} GB`;
}

export function formatElapsed(seconds: number): string {
  const s = Math.max(1, Math.round(seconds));
  if (s < 60) {
    return `${s} s`;
  }
  const mins = Math.floor(s / 60);
  const rem = s % 60;
  if (rem === 0) {
    return `${mins} m`;
  }
  return `${mins} m ${rem} s`;
}

export function formatDurationEstimate(seconds: number, quality: Quality): string {
  const s = Math.max(5, Math.round(seconds));
  let base: string;
  if (s < 55) {
    const rounded = Math.max(10, Math.round(s / 5) * 5);
    base = `About ${rounded} seconds`;
  } else if (s < 90) {
    base = "About a minute";
  } else {
    const mins = Math.max(2, Math.round(s / 60));
    base = `About ${mins} minutes`;
  }
  if (quality === "studio") return `${base}, cleanest vocals`;
  return quality === "best" ? `${base}, cleaner result` : base;
}

export function formatEta(etaSec: number | null): string {
  if (etaSec === null || !Number.isFinite(etaSec)) {
    return "Estimating time left";
  }
  const s = Math.max(1, Math.round(etaSec));
  if (s < 10) {
    return "A few seconds left";
  }
  if (s < 50) {
    const rounded = Math.round(s / 5) * 5;
    return `About ${rounded} seconds left`;
  }
  if (s < 90) {
    return "About a minute left";
  }
  const mins = Math.max(2, Math.round(s / 60));
  return `About ${mins} minutes left`;
}

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
] as const;

export function formatSessionDate(iso: string, nowMs: number = Date.now()): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date(nowMs);
  const startOfToday = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const startOfTarget = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
  const diffDays = Math.round((startOfToday - startOfTarget) / 86_400_000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (d.getUTCFullYear() === now.getUTCFullYear()) {
    return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
  }
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
}

export function rulerTickInterval(durationSec: number): number {
  if (durationSec <= 45) return 5;
  if (durationSec <= 100) return 15;
  if (durationSec <= 360) return 30;
  if (durationSec <= 720) return 60;
  return 120;
}

/**
 * Plain sentence-style metadata line for Settings -> Models (no middle dots):
 * "40 MB, from GitHub, MIT licence" / "55 MB, from Meta, research use (CC BY-NC)"
 */
export function formatModelMeta(model: ModelStatus): string {
  const size = formatBytes(model.installed ? model.sizeBytes : model.downloadBytes);
  const rawLicense = model.license.trim();
  let licensePhrase: string;
  if (model.restricted || /use|licen[cs]e/i.test(rawLicense)) {
    licensePhrase = rawLicense.charAt(0).toLowerCase() + rawLicense.slice(1);
  } else {
    licensePhrase = `${rawLicense} licence`;
  }
  return `${size}, from ${model.source}, ${licensePhrase}`;
}

export function formatImportedModelsNotice(count: number): string {
  if (count <= 0) {
    return "None of those files is a SplitSound model.";
  }
  return `Added ${count} ${count === 1 ? "model" : "models"}`;
}

/**
 * Identifies the two unrestricted models needed by the instrument scan:
 * the small separation model (`scnet_small`) and the detector (`efficientat_mn10_as` / `tagger`).
 */
export function getScanModels(models: readonly ModelStatus[]): ModelStatus[] {
  return models.filter(
    (m) =>
      !m.restricted &&
      (m.id === "scnet_small" ||
        m.id === "efficientat_mn10_as" ||
        m.id === "tagger" ||
        /tagger|detector|efficientat/i.test(`${m.id} ${m.name}`)),
  );
}

/**
 * Total download size of not-yet-installed unrestricted models among the two scan models
 * (`scnet_small` and the detector).
 */
export function computeFirstRunDownloadBytes(models: readonly ModelStatus[]): number {
  return getScanModels(models)
    .filter((m) => !m.installed)
    .reduce((sum, m) => sum + m.downloadBytes, 0);
}

/**
 * Overall download progress (0..1) across models currently being fetched, or null when idle.
 * When `batchIds` includes models from a multi-model fetch (such as the two scan models),
 * already-finished models in that batch count toward received bytes so progress stays monotonic.
 */
export function computeModelDownloadFraction(
  models: readonly ModelStatus[],
  progressById: Readonly<Record<string, ModelDownloadProgress>>,
  batchIds?: ReadonlySet<string>,
): number | null {
  const active = models.filter((m) => m.downloadProgress !== null || m.id in progressById);
  if (active.length === 0) return null;

  const targetIds = new Set<string>(active.map((m) => m.id));
  if (batchIds) {
    for (const id of batchIds) {
      targetIds.add(id);
    }
  }

  let total = 0;
  let received = 0;
  for (const m of models) {
    if (!targetIds.has(m.id)) continue;
    const prog = progressById[m.id];
    const size = prog && prog.totalBytes > 0 ? prog.totalBytes : Math.max(1, m.downloadBytes);
    total += size;
    if (m.installed) {
      received += size;
    } else if (prog) {
      received += Math.min(size, Math.max(0, prog.receivedBytes));
    } else if (m.downloadProgress !== null) {
      received += Math.min(size, Math.max(0, m.downloadProgress * size));
    }
  }

  if (total <= 0) return 0;
  return Math.max(0, Math.min(1, received / total));
}
