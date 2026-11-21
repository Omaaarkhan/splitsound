import type { SessionSummary, SplitProgress } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { createStore } from "./createStore";

export interface LibraryState {
  sessions: SessionSummary[];
  loaded: boolean;
  artworkById: Record<string, string | null>;
  progressById: Record<string, SplitProgress>;
}

const initialLibraryState = (): LibraryState => ({
  sessions: [],
  loaded: false,
  artworkById: {},
  progressById: {},
});

export const libraryStore = createStore<LibraryState>(initialLibraryState);

const pendingArtwork = new Set<string>();

export async function refreshLibrary(): Promise<SessionSummary[]> {
  const sessions = await invoke("list_sessions");
  libraryStore.setState((prev) => ({
    ...prev,
    sessions,
    loaded: true,
  }));
  return sessions;
}

export function applyLibraryChanged(sessions: SessionSummary[]): void {
  libraryStore.setState((prev) => {
    const activeIds = new Set(sessions.map((s) => s.id));
    const progressById: Record<string, SplitProgress> = {};
    for (const [id, prog] of Object.entries(prev.progressById)) {
      const summary = sessions.find((s) => s.id === id);
      if (summary && (summary.status === "splitting" || summary.status === "queued")) {
        progressById[id] = prog;
      }
    }
    const artworkById: Record<string, string | null> = {};
    for (const [id, art] of Object.entries(prev.artworkById)) {
      if (activeIds.has(id)) {
        artworkById[id] = art;
      }
    }
    return {
      ...prev,
      sessions,
      loaded: true,
      progressById,
      artworkById,
    };
  });
}

export function applyLibraryProgress(progress: SplitProgress): void {
  libraryStore.setState((prev) => ({
    ...prev,
    progressById: {
      ...prev.progressById,
      [progress.sessionId]: progress,
    },
  }));
}

export function seedLibraryProgress(map: Record<string, SplitProgress>): void {
  libraryStore.setState((prev) => ({
    ...prev,
    progressById: { ...prev.progressById, ...map },
  }));
}

export async function loadSessionArtwork(sessionId: string): Promise<void> {
  const state = libraryStore.getState();
  if (sessionId in state.artworkById || pendingArtwork.has(sessionId)) {
    return;
  }
  pendingArtwork.add(sessionId);
  try {
    const dataUrl = await invoke("get_artwork", { sessionId });
    libraryStore.setState((prev) => ({
      ...prev,
      artworkById: {
        ...prev.artworkById,
        [sessionId]: dataUrl,
      },
    }));
  } catch {
    libraryStore.setState((prev) => ({
      ...prev,
      artworkById: {
        ...prev.artworkById,
        [sessionId]: null,
      },
    }));
  } finally {
    pendingArtwork.delete(sessionId);
  }
}

export async function deleteLibrarySession(sessionId: string): Promise<void> {
  await invoke("delete_session", { sessionId });
}

export async function clearAllLibrary(): Promise<void> {
  await invoke("clear_library");
}
