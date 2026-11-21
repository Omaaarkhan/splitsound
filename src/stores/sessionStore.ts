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
import { createStore } from "./createStore";
import { libraryStore } from "./libraryStore";

export const DEFAULT_PEAK_BUCKETS = 960;

export interface SessionState {
  current: Session | null;
  peaks: Partial<Record<SourceId, Peaks>>;
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

let revealTimer: ReturnType<typeof setTimeout> | null = null;
let openRequestSeq = 0;
let estimateRequestSeq = 0;

function clearRevealTimer(): void {
  if (revealTimer !== null) {
    clearTimeout(revealTimer);
    revealTimer = null;
  }
}

async function fetchPeaksForSession(session: Session, seq: number): Promise<void> {
  const sources: SourceId[] =
    session.status === "ready"
      ? ["mix", ...session.stems.filter((s) => s.present).map((s) => s.id)]
      : ["mix"];

  const entries = await Promise.all(
    sources.map(async (source) => {
      try {
        const peaks = await invoke("get_peaks", {
          sessionId: session.id,
          source,
          buckets: DEFAULT_PEAK_BUCKETS,
        });
        return [source, peaks] as const;
      } catch {
        return null;
      }
    }),
  );

  if (seq !== openRequestSeq && sessionStore.getState().current?.id !== session.id) {
    return;
  }

  sessionStore.setState((prev) => {
    if (prev.current?.id !== session.id) return prev;
    const nextPeaks = { ...prev.peaks };
    for (const item of entries) {
      if (item) {
        nextPeaks[item[0]] = item[1];
      }
    }
    return { ...prev, peaks: nextPeaks };
  });
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

  sessionStore.setState({
    current: session,
    peaks: {},
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
