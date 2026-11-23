import type {
  Peaks,
  Quality,
  Session,
  SourceId,
  SplitEstimate,
  SplitMode,
  SplitOptions,
  SplitProgress,
} from "../ipc/contract";
import { invoke } from "../ipc/client";
import { decodeSessionPeaksBinary, type DecodedPeaks } from "../utils/waveform";
import { createStore } from "./createStore";
import { libraryStore } from "./libraryStore";

export const DEFAULT_PEAK_BUCKETS = 960;
const MAX_PEAKS_CACHE_ENTRIES = 12;

export interface SessionState {
  current: Session | null;
  peaks: Partial<Record<SourceId, Peaks | DecodedPeaks>>;
  splitOptions: SplitOptions;
  modeTouchedByUser: boolean;
  estimate: SplitEstimate | null;
  splitProgress: SplitProgress | null;
  /** Non-null briefly right after a split transitions splitting -> ready to drive the reveal animation. */
  revealToken: number;
  revealing: boolean;
  loading: boolean;
}

const initialSessionState = (): SessionState => ({
  current: null,
  peaks: {},
  splitOptions: { mode: "all", quality: "fast" },
  modeTouchedByUser: false,
  estimate: null,
  splitProgress: null,
  revealToken: 0,
  revealing: false,
  loading: false,
});

export const sessionStore = createStore<SessionState>(initialSessionState);

interface CachedSessionPeaks {
  version: string;
  peaks: Partial<Record<SourceId, DecodedPeaks>>;
}

const peaksCache = new Map<string, CachedSessionPeaks>();
const pendingMixRequests = new Map<string, Promise<void>>();

let revealTimer: ReturnType<typeof setTimeout> | null = null;
let openRequestSeq = 0;
let estimateRequestSeq = 0;

function clearRevealTimer(): void {
  if (revealTimer !== null) {
    clearTimeout(revealTimer);
    revealTimer = null;
  }
}

function sessionCacheVersion(session: Session): string {
  const present = session.stems
    .filter((s) => s.present)
    .map((s) => s.id)
    .join(",");
  return `${session.status}:${session.split?.finishedAt ?? ""}:${present}`;
}

function initialSourcesForSession(session: Session): SourceId[] {
  return session.status === "ready"
    ? session.stems.filter((s) => s.present).map((s) => s.id)
    : ["mix"];
}

function getCacheEntry(session: Session): CachedSessionPeaks {
  const version = sessionCacheVersion(session);
  const existing = peaksCache.get(session.id);
  if (!existing) {
    return { version, peaks: {} };
  }
  peaksCache.delete(session.id);
  if (existing.version === version) {
    peaksCache.set(session.id, existing);
    return existing;
  }
  // Keep mix peaks across status/resplit transitions while invalidating stale stems.
  const preserved: CachedSessionPeaks = {
    version,
    peaks: existing.peaks.mix ? { mix: existing.peaks.mix } : {},
  };
  peaksCache.set(session.id, preserved);
  return preserved;
}

function putCachePeaks(
  sessionId: string,
  version: string,
  added: Partial<Record<SourceId, DecodedPeaks>>,
): Partial<Record<SourceId, DecodedPeaks>> {
  const existing = peaksCache.get(sessionId);
  peaksCache.delete(sessionId);
  const basePeaks =
    existing && existing.version === version
      ? existing.peaks
      : existing?.peaks.mix
        ? { mix: existing.peaks.mix }
        : {};
  const merged = { ...basePeaks, ...added };
  peaksCache.set(sessionId, { version, peaks: merged });
  while (peaksCache.size > MAX_PEAKS_CACHE_ENTRIES) {
    const oldest = peaksCache.keys().next().value;
    if (oldest === undefined) break;
    peaksCache.delete(oldest);
  }
  return merged;
}

async function fetchPeaksForSession(session: Session, seq: number): Promise<void> {
  const sources = initialSourcesForSession(session);
  if (sources.length === 0) return;

  const entry = getCacheEntry(session);
  const missing = sources.filter((source) => !(source in entry.peaks));
  if (missing.length === 0) {
    sessionStore.setState((prev) =>
      prev.current?.id === session.id ? { ...prev, peaks: { ...entry.peaks, ...prev.peaks } } : prev,
    );
    return;
  }

  let decoded: Partial<Record<SourceId, DecodedPeaks>>;
  try {
    const buffer = await invoke("get_session_peaks", {
      sessionId: session.id,
      sources: missing,
      buckets: DEFAULT_PEAK_BUCKETS,
    });
    decoded = decodeSessionPeaksBinary(buffer);
  } catch {
    return;
  }

  const merged = putCachePeaks(session.id, entry.version, decoded);

  if (seq !== openRequestSeq && sessionStore.getState().current?.id !== session.id) {
    return;
  }

  sessionStore.setState((prev) => {
    if (prev.current?.id !== session.id) return prev;
    return { ...prev, peaks: { ...prev.peaks, ...merged } };
  });
}

export async function ensureMixPeaks(sessionId?: string): Promise<void> {
  const current = sessionStore.getState().current;
  const target = sessionId && current?.id === sessionId ? current : !sessionId ? current : null;
  if (!target) return;

  const currentPeaks = sessionStore.getState().peaks;
  if (currentPeaks.mix) return;

  const entry = getCacheEntry(target);
  if (entry.peaks.mix) {
    const mix = entry.peaks.mix;
    sessionStore.setState((prev) =>
      prev.current?.id === target.id ? { ...prev, peaks: { ...prev.peaks, mix } } : prev,
    );
    return;
  }

  const existingReq = pendingMixRequests.get(target.id);
  if (existingReq) return existingReq;

  const version = entry.version;
  const req = (async () => {
    try {
      const buffer = await invoke("get_session_peaks", {
        sessionId: target.id,
        sources: ["mix"],
        buckets: DEFAULT_PEAK_BUCKETS,
      });
      const decoded = decodeSessionPeaksBinary(buffer);
      if (!decoded.mix) return;
      putCachePeaks(target.id, version, { mix: decoded.mix });
      sessionStore.setState((prev) =>
        prev.current?.id === target.id
          ? { ...prev, peaks: { ...prev.peaks, mix: decoded.mix } }
          : prev,
      );
    } catch {
      // keep silent when waveform peaks cannot be loaded
    } finally {
      pendingMixRequests.delete(target.id);
    }
  })();
  pendingMixRequests.set(target.id, req);
  return req;
}

export async function refreshSplitEstimate(
  sessionId?: string,
  options?: SplitOptions,
): Promise<SplitEstimate | null> {
  const st = sessionStore.getState();
  const id = sessionId ?? st.current?.id;
  if (!id) return null;
  const opts = options ?? st.splitOptions;
  const seq = ++estimateRequestSeq;
  try {
    const estimate = await invoke("estimate_split", { sessionId: id, options: opts });
    if (seq === estimateRequestSeq && sessionStore.getState().current?.id === id) {
      sessionStore.setState((prev) => ({ ...prev, estimate }));
    }
    return estimate;
  } catch {
    return null;
  }
}

export async function openSession(
  sessionId: string,
  defaultQuality: Quality = "fast",
): Promise<Session> {
  const seq = ++openRequestSeq;
  clearRevealTimer();
  sessionStore.setState((prev) => ({ ...prev, loading: true, revealing: false }));
  const session = await invoke("get_session", { sessionId });
  if (seq !== openRequestSeq) {
    return session;
  }

  const existingProgress = libraryStore.getState().progressById[session.id] ?? null;
  const initialOptions: SplitOptions = {
    mode: session.split?.mode ?? session.suggestedMode,
    quality: session.split?.quality ?? defaultQuality,
  };

  const cachedPeaks = getCacheEntry(session).peaks;

  sessionStore.setState({
    current: session,
    peaks: cachedPeaks,
    splitOptions: initialOptions,
    modeTouchedByUser: false,
    estimate: null,
    splitProgress:
      session.status === "splitting" || session.status === "queued" ? existingProgress : null,
    revealToken: 0,
    revealing: false,
    loading: false,
  });

  const tasks: Promise<unknown>[] = [fetchPeaksForSession(session, seq)];
  if (session.status === "imported" || session.status === "failed") {
    tasks.push(refreshSplitEstimate(session.id, initialOptions));
  }
  await Promise.all(tasks);
  return session;
}

export function closeSession(): void {
  openRequestSeq++;
  clearRevealTimer();
  sessionStore.setState(initialSessionState());
}

export function setSplitMode(mode: SplitMode): void {
  const prev = sessionStore.getState();
  const nextOptions: SplitOptions = { ...prev.splitOptions, mode };
  sessionStore.setState({
    ...prev,
    splitOptions: nextOptions,
    modeTouchedByUser: true,
  });
  void refreshSplitEstimate(prev.current?.id, nextOptions);
}

export function setSplitQuality(quality: Quality): void {
  const prev = sessionStore.getState();
  const nextOptions: SplitOptions = { ...prev.splitOptions, quality };
  sessionStore.setState({
    ...prev,
    splitOptions: nextOptions,
  });
  void refreshSplitEstimate(prev.current?.id, nextOptions);
}

export function applySessionUpdated(updated: Session): void {
  const prev = sessionStore.getState();
  if (!prev.current || prev.current.id !== updated.id) return;

  const wasSplitting = prev.current.status === "splitting" || prev.current.status === "queued";
  const becameReady = wasSplitting && updated.status === "ready";
  const nextMode =
    !prev.modeTouchedByUser && updated.status === "imported"
      ? updated.suggestedMode
      : prev.splitOptions.mode;
  const nextOptions: SplitOptions = { ...prev.splitOptions, mode: nextMode };

  if (becameReady) {
    clearRevealTimer();
    revealTimer = setTimeout(() => {
      revealTimer = null;
      sessionStore.setState((s) => (s.revealing ? { ...s, revealing: false } : s));
    }, 900);
  }

  sessionStore.setState({
    ...prev,
    current: updated,
    splitOptions: nextOptions,
    splitProgress:
      updated.status === "splitting" || updated.status === "queued" ? prev.splitProgress : null,
    revealToken: becameReady ? prev.revealToken + 1 : prev.revealToken,
    revealing: becameReady ? true : prev.revealing,
  });

  if (becameReady) {
    void fetchPeaksForSession(updated, openRequestSeq);
  } else if (updated.status === "imported" || updated.status === "failed") {
    void refreshSplitEstimate(updated.id, nextOptions);
  }
}

export function applySessionSplitProgress(progress: SplitProgress): void {
  const prev = sessionStore.getState();
  if (!prev.current || prev.current.id !== progress.sessionId) return;
  sessionStore.setState({
    ...prev,
    splitProgress: progress,
  });
}
