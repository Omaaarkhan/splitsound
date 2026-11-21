import type { ChannelState, PlaybackState, PlaybackTick, SourceId } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { createStore } from "./createStore";
import { snapDb } from "../utils/fader";

export interface PlaybackStoreState {
  playback: PlaybackState;
  lastLoopRegion: { startSec: number; endSec: number } | null;
  lastTickPos: number;
  lastTickPerfMs: number;
}

const emptyPlayback = (): PlaybackState => ({
  sessionId: null,
  playing: false,
  positionSec: 0,
  durationSec: 0,
  source: "original",
  loop: null,
  masterGainDb: 0,
  channels: [],
});

const initialPlaybackStoreState = (): PlaybackStoreState => ({
  playback: emptyPlayback(),
  lastLoopRegion: null,
  lastTickPos: 0,
  lastTickPerfMs: 0,
});

export const playbackStore = createStore<PlaybackStoreState>(initialPlaybackStoreState);

/** High-frequency tick channel (30 Hz) to animate playhead & meters without full React re-renders. */
let currentLevels: Partial<Record<SourceId, number>> = {};
let currentTickPos = 0;
let currentTickPerfMs = 0;
const tickListeners = new Set<() => void>();

function perfNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function notifyTickListeners(): void {
  for (const fn of tickListeners) {
    fn();
  }
}

export function subscribePlaybackTicks(listener: () => void): () => void {
  tickListeners.add(listener);
  return () => {
    tickListeners.delete(listener);
  };
}

export function getPlaybackLevels(): Partial<Record<SourceId, number>> {
  return currentLevels;
}

export function seedPlaybackLevels(levels: Partial<Record<SourceId, number>>): void {
  currentLevels = { ...levels };
  notifyTickListeners();
}

export function resetPlaybackRuntime(): void {
  currentLevels = {};
  currentTickPos = 0;
  currentTickPerfMs = 0;
  playbackStore.reset();
  notifyTickListeners();
}

/**
 * Interpolates smoothly from the last reported tick without drifting ahead more
 * than ~120 ms when playing, and returns the exact reported position when paused.
 */
export function getInterpolatedPositionSec(nowPerfMs: number = perfNow()): number {
  const { playback } = playbackStore.getState();
  if (!playback.playing) {
    return currentTickPos;
  }
  const elapsedSec = Math.max(0, Math.min(0.12, (nowPerfMs - currentTickPerfMs) / 1000));
  const raw = currentTickPos + elapsedSec;
  if (playback.loop && raw > playback.loop.endSec) {
    return playback.loop.startSec;
  }
  return Math.max(0, Math.min(playback.durationSec, raw));
}

export function applyPlaybackState(next: PlaybackState): void {
  const now = perfNow();
  currentTickPos = next.positionSec;
  currentTickPerfMs = now;
  const prev = playbackStore.getState();
  const sameSession = next.sessionId === prev.playback.sessionId;
  if (!next.playing && !sameSession) {
    currentLevels = {};
  }
  playbackStore.setState({
    playback: next,
    lastLoopRegion: next.loop ?? (sameSession ? prev.lastLoopRegion : null),
    lastTickPos: next.positionSec,
    lastTickPerfMs: now,
  });
  notifyTickListeners();
}

export function applyPlaybackTick(tick: PlaybackTick): void {
  const now = perfNow();
  currentTickPos = tick.positionSec;
  currentTickPerfMs = now;
  const nextLevels: Partial<Record<SourceId, number>> = {};
  for (const entry of tick.levels) {
    nextLevels[entry.id] = entry.peak;
  }
  currentLevels = nextLevels;
  notifyTickListeners();
}

export function isChannelAudible(
  channel: ChannelState | undefined,
  channels: readonly ChannelState[],
): boolean {
  if (!channel) return true;
  const anySolo = channels.some((c) => c.solo);
  return anySolo ? channel.solo : !channel.muted;
}

export async function loadPlaybackSession(sessionId: string): Promise<PlaybackState> {
  const state = await invoke("playback_load", { sessionId });
  applyPlaybackState(state);
  return state;
}

export async function unloadPlaybackSession(): Promise<void> {
  await invoke("playback_unload");
  currentLevels = {};
  currentTickPos = 0;
  playbackStore.setState(initialPlaybackStoreState());
  notifyTickListeners();
}

export async function togglePlayback(): Promise<void> {
  const { playback } = playbackStore.getState();
  if (!playback.sessionId) return;
  if (playback.playing) {
    await invoke("playback_pause");
  } else {
    await invoke("playback_play");
  }
}

export async function seekPlayback(positionSec: number): Promise<void> {
  const { playback } = playbackStore.getState();
  if (!playback.sessionId) return;
  const clamped = Math.max(0, Math.min(playback.durationSec, positionSec));
  const now = perfNow();
  currentTickPos = clamped;
  currentTickPerfMs = now;
  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, positionSec: clamped },
    lastTickPos: clamped,
    lastTickPerfMs: now,
  }));
  notifyTickListeners();
  await invoke("playback_seek", { positionSec: clamped });
}

export async function seekPlaybackBy(deltaSec: number): Promise<void> {
  const current = getInterpolatedPositionSec();
  await seekPlayback(current + deltaSec);
}

export async function setChannelGain(id: SourceId, gainDb: number): Promise<void> {
  const snapped = snapDb(gainDb);
  playbackStore.setState((prev) => ({
    ...prev,
    playback: {
      ...prev.playback,
      channels: prev.playback.channels.map((ch) =>
        ch.id === id ? { ...ch, gainDb: snapped } : ch,
      ),
    },
  }));
  await invoke("playback_set_channel", { id, gainDb: snapped });
}

export async function toggleChannelSolo(id: SourceId): Promise<void> {
  const ch = playbackStore.getState().playback.channels.find((c) => c.id === id);
  if (!ch) return;
  const solo = !ch.solo;
  const muted = solo ? false : ch.muted;
  playbackStore.setState((prev) => ({
    ...prev,
    playback: {
      ...prev.playback,
      channels: prev.playback.channels.map((c) => (c.id === id ? { ...c, solo, muted } : c)),
    },
  }));
  await invoke(
    "playback_set_channel",
    solo ? { id, solo: true, muted: false } : { id, solo: false },
  );
}

export async function toggleChannelMute(id: SourceId): Promise<void> {
  const ch = playbackStore.getState().playback.channels.find((c) => c.id === id);
  if (!ch) return;
  const muted = !ch.muted;
  const solo = muted ? false : ch.solo;
  playbackStore.setState((prev) => ({
    ...prev,
    playback: {
      ...prev.playback,
      channels: prev.playback.channels.map((c) => (c.id === id ? { ...c, muted, solo } : c)),
    },
  }));
  await invoke(
    "playback_set_channel",
    muted ? { id, muted: true, solo: false } : { id, muted: false },
  );
}

export async function setMasterGain(gainDb: number): Promise<void> {
  const snapped = snapDb(gainDb);
  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, masterGainDb: snapped },
  }));
  await invoke("playback_set_master", { gainDb: snapped });
}

export async function toggleHearOriginal(): Promise<void> {
  const { playback } = playbackStore.getState();
  const source = playback.source === "original" ? "stems" : "original";
  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, source },
  }));
  await invoke("playback_set_source", { source });
}

export async function setLoopRegion(
  loop: { startSec: number; endSec: number } | null,
): Promise<void> {
  const { playback } = playbackStore.getState();
  if (!playback.sessionId || playback.durationSec <= 0) return;

  let normalized: { startSec: number; endSec: number } | null = null;
  if (loop) {
    const startSec = Math.max(
      0,
      Math.min(playback.durationSec, Math.min(loop.startSec, loop.endSec)),
    );
    const endSec = Math.max(
      0,
      Math.min(playback.durationSec, Math.max(loop.startSec, loop.endSec)),
    );
    if (endSec - startSec >= 0.5) {
      normalized = {
        startSec: Number(startSec.toFixed(2)),
        endSec: Number(endSec.toFixed(2)),
      };
    }
  }

  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, loop: normalized },
    lastLoopRegion: normalized ?? prev.lastLoopRegion,
  }));
  await invoke("playback_set_loop", { loop: normalized });
}

export async function toggleLoop(): Promise<void> {
  const { playback, lastLoopRegion } = playbackStore.getState();
  if (!playback.sessionId || playback.durationSec <= 0) return;
  const nextLoop = playback.loop
    ? null
    : (lastLoopRegion ?? { startSec: 0, endSec: playback.durationSec });
  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, loop: nextLoop },
    lastLoopRegion: nextLoop ?? prev.lastLoopRegion,
  }));
  await invoke("playback_set_loop", { loop: nextLoop });
}
