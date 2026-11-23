import type { ChannelState, PlaybackState, PlaybackTick, SourceId } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { createStore } from "./createStore";
import { ensureMixPeaks } from "./sessionStore";
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

/** High-frequency tick + single central display rAF channel for waveforms, playhead, meters & clock. */
let currentLevels: Partial<Record<SourceId, number>> = {};
let currentTickPos = 0;
let currentTickPerfMs = 0;
const tickListeners = new Set<() => void>();
const frameListeners = new Set<() => void>();
let extraAnimCount = 0;
let rafId = 0;

// In-flight IPC coalescing + optimistic edit guards so stale state/tick echoes never fight drags.
const inFlightChannelGain = new Map<SourceId, Promise<void>>();
const pendingChannelGain = new Map<SourceId, number>();
let inFlightMasterGain: Promise<void> | null = null;
let pendingMasterGain: number | null = null;
let inFlightSeek: Promise<void> | null = null;
let pendingSeekPos: number | null = null;

const pendingChannelEdits = new Map<SourceId, number>();
let pendingMasterEdits = 0;
let pendingSeekEdits = 0;
let pendingSourceEdits = 0;
let pendingLoopEdits = 0;

function perfNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function shouldRunRaf(): boolean {
  return playbackStore.getState().playback.playing || extraAnimCount > 0;
}

function stopRafLoop(): void {
  if (rafId !== 0 && typeof cancelAnimationFrame === "function") {
    cancelAnimationFrame(rafId);
  }
  rafId = 0;
}

function rafStep(): void {
  rafId = 0;
  if (!shouldRunRaf() || frameListeners.size === 0) return;
  for (const fn of frameListeners) {
    fn();
  }
  if (shouldRunRaf() && frameListeners.size > 0 && typeof requestAnimationFrame === "function") {
    rafId = requestAnimationFrame(rafStep);
  }
}

function syncRafState(): void {
  if (shouldRunRaf() && frameListeners.size > 0) {
    if (rafId === 0 && typeof requestAnimationFrame === "function") {
      rafId = requestAnimationFrame(rafStep);
    }
  } else {
    stopRafLoop();
  }
}

function notifyTickListeners(): void {
  for (const fn of tickListeners) {
    fn();
  }
  for (const fn of frameListeners) {
    fn();
  }
}

function clearPendingGuards(): void {
  inFlightChannelGain.clear();
  pendingChannelGain.clear();
  inFlightMasterGain = null;
  pendingMasterGain = null;
  inFlightSeek = null;
  pendingSeekPos = null;
  pendingChannelEdits.clear();
  pendingMasterEdits = 0;
  pendingSeekEdits = 0;
  pendingSourceEdits = 0;
  pendingLoopEdits = 0;
}

export function subscribePlaybackTicks(listener: () => void): () => void {
  tickListeners.add(listener);
  return () => {
    tickListeners.delete(listener);
  };
}

/**
 * Subscribe to both high-frequency discrete ticks/seeks and the single shared
 * display-rate `requestAnimationFrame` loop while audio or transitions animate.
 */
export function subscribePlaybackFrame(listener: () => void): () => void {
  frameListeners.add(listener);
  syncRafState();
  return () => {
    frameListeners.delete(listener);
    if (frameListeners.size === 0) {
      stopRafLoop();
    }
  };
}

/** Keeps the shared display-rate `requestAnimationFrame` loop running for non-playback animations. */
export function acquirePlaybackAnimation(): () => void {
  extraAnimCount++;
  syncRafState();
  let released = false;
  return () => {
    if (released) return;
    released = true;
    extraAnimCount = Math.max(0, extraAnimCount - 1);
    syncRafState();
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
  stopRafLoop();
  clearPendingGuards();
  extraAnimCount = 0;
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
  if (!playback.playing || pendingSeekEdits > 0) {
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
  const prev = playbackStore.getState();
  const sameSession = next.sessionId !== null && next.sessionId === prev.playback.sessionId;

  if (!sameSession) {
    clearPendingGuards();
  }

  let effectivePos = next.positionSec;
  if (sameSession && pendingSeekEdits > 0) {
    effectivePos = prev.playback.positionSec;
  } else {
    currentTickPos = next.positionSec;
    currentTickPerfMs = now;
  }

  if (!next.playing && !sameSession) {
    currentLevels = {};
  }

  const effectiveChannels =
    sameSession && pendingChannelEdits.size > 0
      ? next.channels.map((ch) => {
          if ((pendingChannelEdits.get(ch.id) ?? 0) > 0) {
            return prev.playback.channels.find((p) => p.id === ch.id) ?? ch;
          }
          return ch;
        })
      : next.channels;

  const effectiveMaster =
    sameSession && pendingMasterEdits > 0 ? prev.playback.masterGainDb : next.masterGainDb;
  const effectiveSource =
    sameSession && pendingSourceEdits > 0 ? prev.playback.source : next.source;
  const effectiveLoop = sameSession && pendingLoopEdits > 0 ? prev.playback.loop : next.loop;

  const merged: PlaybackState = {
    ...next,
    positionSec: effectivePos,
    channels: effectiveChannels,
    masterGainDb: effectiveMaster,
    source: effectiveSource,
    loop: effectiveLoop,
  };

  playbackStore.setState({
    playback: merged,
    lastLoopRegion: merged.loop ?? (sameSession ? prev.lastLoopRegion : null),
    lastTickPos: currentTickPos,
    lastTickPerfMs: now,
  });
  notifyTickListeners();
  syncRafState();
}

export function applyPlaybackTick(tick: PlaybackTick): void {
  const now = perfNow();
  if (pendingSeekEdits === 0) {
    currentTickPos = tick.positionSec;
    currentTickPerfMs = now;
  }
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
  clearPendingGuards();
  const state = await invoke("playback_load", { sessionId });
  applyPlaybackState(state);
  return state;
}

export async function unloadPlaybackSession(): Promise<void> {
  stopRafLoop();
  clearPendingGuards();
  await invoke("playback_unload");
  currentLevels = {};
  currentTickPos = 0;
  playbackStore.setState(initialPlaybackStoreState());
  notifyTickListeners();
}

export async function togglePlayback(): Promise<void> {
  const { playback } = playbackStore.getState();
  if (!playback.sessionId) return;
  const targetPlaying = !playback.playing;
  const sessionId = playback.sessionId;
  const now = perfNow();
  currentTickPos = getInterpolatedPositionSec(now);
  currentTickPerfMs = now;
  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, playing: targetPlaying },
    lastTickPos: currentTickPos,
    lastTickPerfMs: now,
  }));
  notifyTickListeners();
  syncRafState();
  try {
    if (targetPlaying) {
      await invoke("playback_play");
    } else {
      await invoke("playback_pause");
    }
  } catch (err) {
    const rollbackNow = perfNow();
    playbackStore.setState((prev) =>
      prev.playback.sessionId === sessionId
        ? {
            ...prev,
            playback: { ...prev.playback, playing: !targetPlaying },
            lastTickPos: currentTickPos,
            lastTickPerfMs: rollbackNow,
          }
        : prev,
    );
    notifyTickListeners();
    syncRafState();
    throw err;
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

  pendingSeekPos = clamped;
  if (inFlightSeek) return inFlightSeek;

  pendingSeekEdits++;
  inFlightSeek = (async () => {
    try {
      while (pendingSeekPos !== null) {
        const nextPos = pendingSeekPos;
        pendingSeekPos = null;
        await invoke("playback_seek", { positionSec: nextPos });
      }
    } finally {
      inFlightSeek = null;
      pendingSeekEdits = Math.max(0, pendingSeekEdits - 1);
    }
  })();
  return inFlightSeek;
}

export async function seekPlaybackBy(deltaSec: number): Promise<void> {
  const current = getInterpolatedPositionSec();
  await seekPlayback(current + deltaSec);
}

function activeStemsSource(playback: PlaybackState, id: SourceId): PlaybackState["source"] {
  return playback.source === "original" && playback.channels.some((c) => c.id === id)
    ? "stems"
    : playback.source;
}

function bumpPendingChannel(id: SourceId, delta: number): void {
  const next = Math.max(0, (pendingChannelEdits.get(id) ?? 0) + delta);
  if (next === 0) pendingChannelEdits.delete(id);
  else pendingChannelEdits.set(id, next);
}

export async function setChannelGain(id: SourceId, gainDb: number): Promise<void> {
  if (!playbackStore.getState().playback.channels.some((c) => c.id === id)) return;
  const snapped = snapDb(gainDb);
  playbackStore.setState((prev) => ({
    ...prev,
    playback: {
      ...prev.playback,
      source: activeStemsSource(prev.playback, id),
      channels: prev.playback.channels.map((ch) =>
        ch.id === id ? { ...ch, gainDb: snapped } : ch,
      ),
    },
  }));

  pendingChannelGain.set(id, snapped);
  const existing = inFlightChannelGain.get(id);
  if (existing) return existing;

  bumpPendingChannel(id, 1);
  const task = (async () => {
    try {
      while (pendingChannelGain.has(id)) {
        const nextGain = pendingChannelGain.get(id)!;
        pendingChannelGain.delete(id);
        await invoke("playback_set_channel", { id, gainDb: nextGain });
      }
    } finally {
      inFlightChannelGain.delete(id);
      bumpPendingChannel(id, -1);
    }
  })();
  inFlightChannelGain.set(id, task);
  return task;
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
      source: activeStemsSource(prev.playback, id),
      channels: prev.playback.channels.map((c) => (c.id === id ? { ...c, solo, muted } : c)),
    },
  }));
  bumpPendingChannel(id, 1);
  try {
    await invoke(
      "playback_set_channel",
      solo ? { id, solo: true, muted: false } : { id, solo: false },
    );
  } finally {
    bumpPendingChannel(id, -1);
  }
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
      source: activeStemsSource(prev.playback, id),
      channels: prev.playback.channels.map((c) => (c.id === id ? { ...c, muted, solo } : c)),
    },
  }));
  bumpPendingChannel(id, 1);
  try {
    await invoke(
      "playback_set_channel",
      muted ? { id, muted: true, solo: false } : { id, muted: false },
    );
  } finally {
    bumpPendingChannel(id, -1);
  }
}

export async function setMasterGain(gainDb: number): Promise<void> {
  const snapped = snapDb(gainDb);
  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, masterGainDb: snapped },
  }));

  pendingMasterGain = snapped;
  if (inFlightMasterGain) return inFlightMasterGain;

  pendingMasterEdits++;
  inFlightMasterGain = (async () => {
    try {
      while (pendingMasterGain !== null) {
        const nextGain = pendingMasterGain;
        pendingMasterGain = null;
        await invoke("playback_set_master", { gainDb: nextGain });
      }
    } finally {
      inFlightMasterGain = null;
      pendingMasterEdits = Math.max(0, pendingMasterEdits - 1);
    }
  })();
  return inFlightMasterGain;
}

export async function toggleHearOriginal(): Promise<void> {
  const { playback } = playbackStore.getState();
  const source = playback.source === "original" ? "stems" : "original";
  if (source === "original" && playback.sessionId) {
    void ensureMixPeaks(playback.sessionId);
  }
  playbackStore.setState((prev) => ({
    ...prev,
    playback: { ...prev.playback, source },
  }));
  pendingSourceEdits++;
  try {
    await invoke("playback_set_source", { source });
  } finally {
    pendingSourceEdits = Math.max(0, pendingSourceEdits - 1);
  }
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
  pendingLoopEdits++;
  try {
    await invoke("playback_set_loop", { loop: normalized });
  } finally {
    pendingLoopEdits = Math.max(0, pendingLoopEdits - 1);
  }
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
  pendingLoopEdits++;
  try {
    await invoke("playback_set_loop", { loop: nextLoop });
  } finally {
    pendingLoopEdits = Math.max(0, pendingLoopEdits - 1);
  }
}
