import { memo, useEffect, useMemo, useRef } from "react";
import type { SessionSummary, SplitProgress, StemId } from "../ipc/contract";
import { useShallowStore } from "../stores/createStore";
import { libraryStore, loadSessionArtwork } from "../stores/libraryStore";
import { modelsStore } from "../stores/modelsStore";
import { settingsStore } from "../stores/settingsStore";
import { getNowMs } from "../utils/clock";
import {
  computeFirstRunDownloadBytes,
  formatBytes,
  formatSessionDate,
  formatTime,
} from "../utils/format";
import { STEM_CSS_VAR } from "../utils/stems";
import { drawWaveform, resolveCssColor, type DecodedPeaks } from "../utils/waveform";
import { TapeSlivers } from "../components/primitives/TapeSlivers";
import { formatModShortcut, tip } from "../components/primitives/Tooltip";
import "./HomeView.css";

interface HomeViewProps {
  draggingFiles: boolean;
  onChooseFile: () => void;
  onOpenSession: (sessionId: string) => void;
  onRequestRemove: (sessionId: string, title: string) => void;
}

const ILLUSTRATION_STEMS: readonly StemId[] = [
  "vocals",
  "drums",
  "bass",
  "guitar",
  "piano",
  "other",
];

function makeIllustrationPeaks(seed: number, kind: StemId | "mix", buckets: number): DecodedPeaks {
  let s = seed >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const raw = new Float32Array(buckets);
  for (let i = 0; i < buckets; i++) {
    const t = i / buckets;
    let v = 0.2;
    switch (kind) {
      case "mix": {
        const intro = t < 0.14 ? Math.pow(t / 0.14, 0.75) : 1;
        v =
          (0.36 +
            0.24 * Math.sin(t * 22 + 0.5) +
            0.18 * Math.abs(Math.sin(t * 88)) +
            0.14 * rand()) *
          intro;
        break;
      }
      case "vocals":
        v =
          Math.sin(t * 28) > -0.15
            ? 0.28 + 0.52 * Math.abs(Math.sin(t * 140 + rand()))
            : 0.03 * rand();
        break;
      case "drums": {
        const beat = (t * 64) % 1;
        const snare = ((t * 32 + 0.5) % 1) * 8;
        v = Math.min(1, 0.08 + 0.8 * Math.exp(-beat * 6) + 0.48 * Math.exp(-snare));
        break;
      }
      case "bass": {
        const note = Math.floor(t * 28);
        const decay = (t * 28) % 1;
        v = (0.45 + 0.25 * Math.sin(note * 2.1)) * (1 - 0.35 * decay);
        break;
      }
      case "guitar":
        v = (0.24 + 0.42 * rand()) * (0.65 + 0.35 * Math.exp(-(((t * 96) % 1) * 3)));
        break;
      case "piano":
        v = 0.16 + 0.62 * Math.exp(-(((t * 20) % 1) * 3.5)) * (0.75 + 0.25 * rand());
        break;
      case "other":
        v = (0.14 + 0.28 * (0.5 + 0.5 * Math.sin(t * 10 - 0.6))) * (0.8 + 0.2 * rand());
        break;
      case "instrumental":
        v = 0.4 + 0.4 * rand();
        break;
    }
    raw[i] = Math.max(0, Math.min(1, v));
  }

  const min = new Float32Array(buckets);
  const max = new Float32Array(buckets);
  for (let i = 0; i < buckets; i++) {
    const prev = i > 0 ? raw[i - 1] : raw[i];
    const next = i < buckets - 1 ? raw[i + 1] : raw[i];
    const smoothed = (raw[i] + prev * 0.5 + next * 0.5) / 2;
    max[i] = smoothed;
    min[i] = -smoothed * 0.94;
  }
  return { buckets, min, max };
}

const HomeIllustration = memo(function HomeIllustration({
  draggingFiles,
}: {
  draggingFiles: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const peaksMap = useMemo(() => {
    const mix = makeIllustrationPeaks(4099, "mix", 280);
    const stems = ILLUSTRATION_STEMS.map((id, idx) => ({
      id,
      peaks: makeIllustrationPeaks(7919 + idx * 997, id, 420),
    }));
    return { mix, stems };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const inkColor = resolveCssColor("--ink", "#f3f0e6");
    const ink3Color = resolveCssColor("--ink-3", "#95a8b3");
    const lineColor = resolveCssColor("--well-line", "#24323c");
    const lampColor = resolveCssColor("--lamp", "#fbc044");
    const stemColors = ILLUSTRATION_STEMS.map((id) => resolveCssColor(STEM_CSS_VAR[id]));

    const paint = () => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      if (w <= 0 || h <= 0) return;

      const dpr = Math.min(3, window.devicePixelRatio || 1);
      const targetW = Math.round(w * dpr);
      const targetH = Math.round(h * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // 45-degree forward-slash cut centered near 35% of the illustration width
      const cutCenter = w * 0.35;
      const halfRun = h * 0.5;
      const topX = cutCenter + halfRun;
      const botX = cutCenter - halfRun;

      // Left side of the cut: single off-white waveform arriving from the left
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(topX, 0);
      ctx.lineTo(botX, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      ctx.clip();

      drawWaveform(ctx, {
        peaks: peaksMap.mix,
        color: inkColor,
        x: 0,
        y: h * 0.14,
        width: Math.max(topX, w * 0.52),
        height: h * 0.72,
        playFraction: 1,
        audible: true,
        paddingY: 12,
        filledPlayedAlpha: 0.92,
      });
      ctx.restore();

      // Right side of the cut: six coloured waveforms stacked in lanes
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(topX, 0);
      ctx.lineTo(w, 0);
      ctx.lineTo(w, h);
      ctx.lineTo(botX, h);
      ctx.closePath();
      ctx.clip();

      const laneCount = peaksMap.stems.length;
      const laneH = h / laneCount;
      const startX = Math.min(botX, cutCenter - halfRun);
      const spanW = w - startX;

      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 1;
      for (let i = 1; i < laneCount; i++) {
        const ly = Math.round(i * laneH) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, ly);
        ctx.lineTo(w, ly);
        ctx.stroke();
      }

      for (let i = 0; i < laneCount; i++) {
        drawWaveform(ctx, {
          peaks: peaksMap.stems[i].peaks,
          color: stemColors[i],
          x: startX,
          y: i * laneH,
          width: spanW,
          height: laneH,
          playFraction: 1,
          audible: true,
          paddingY: 6,
          filledPlayedAlpha: 0.9,
        });
      }
      ctx.restore();

      // The 45-degree splice cut line itself
      ctx.save();
      ctx.strokeStyle = draggingFiles ? lampColor : ink3Color;
      ctx.lineWidth = draggingFiles ? 2 : 1.5;
      ctx.beginPath();
      ctx.moveTo(botX, h);
      ctx.lineTo(topX, 0);
      ctx.stroke();
      ctx.restore();

      ctx.restore();
    };

    paint();
    const ro = new ResizeObserver(paint);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draggingFiles, peaksMap]);

  return (
    <div className="home-drop__art" aria-hidden="true">
      <canvas ref={canvasRef} className="home-drop__canvas" />
    </div>
  );
});

const artworkObserverTargets = new WeakMap<Element, string>();
let sharedArtworkObserver: IntersectionObserver | null = null;

function observeRowArtwork(el: Element, sessionId: string): () => void {
  if (typeof IntersectionObserver === "undefined") {
    void loadSessionArtwork(sessionId);
    return () => {};
  }
  if (!sharedArtworkObserver) {
    sharedArtworkObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const id = artworkObserverTargets.get(entry.target);
          artworkObserverTargets.delete(entry.target);
          sharedArtworkObserver?.unobserve(entry.target);
          if (id) {
            void loadSessionArtwork(id);
          }
        }
      },
      { rootMargin: "200px" },
    );
  }
  artworkObserverTargets.set(el, sessionId);
  sharedArtworkObserver.observe(el);
  return () => {
    artworkObserverTargets.delete(el);
    sharedArtworkObserver?.unobserve(el);
  };
}

interface LibraryRowProps {
  session: SessionSummary;
  artworkUrl: string | null | undefined;
  progress: SplitProgress | undefined;
  onOpenSession: (sessionId: string) => void;
  onRequestRemove: (sessionId: string, title: string) => void;
}

const LibraryRow = memo(function LibraryRow({
  session,
  artworkUrl,
  progress,
  onOpenSession,
  onRequestRemove,
}: LibraryRowProps) {
  const rowRef = useRef<HTMLLIElement | null>(null);
  const fillRef = useRef<HTMLDivElement | null>(null);
  const pct = progress ? Math.round(progress.fraction * 100) : 0;

  useEffect(() => {
    if (!session.hasArtwork || artworkUrl !== undefined) return;
    const el = rowRef.current;
    if (!el) {
      void loadSessionArtwork(session.id);
      return;
    }
    return observeRowArtwork(el, session.id);
  }, [session.id, session.hasArtwork, artworkUrl]);

  useEffect(() => {
    const el = fillRef.current;
    if (el) {
      el.style.setProperty("--progress", (pct / 100).toFixed(4));
    }
  }, [pct]);

  const isSplitting = session.status === "splitting" || session.status === "queued";

  return (
    <li ref={rowRef} className="library-row">
      <button type="button" className="library-row__main" onClick={() => onOpenSession(session.id)}>
        <span className="library-row__art" aria-hidden="true">
          {artworkUrl ? <img src={artworkUrl} alt="" /> : null}
        </span>

        <span className="library-row__meta">
          <span className="library-row__title">{session.title}</span>
          {session.artist ? <span className="library-row__artist">{session.artist}</span> : null}
        </span>

        <span className="library-row__dur">{formatTime(session.durationSec)}</span>

        <span className="library-row__stems">
          {session.stemIds.length > 0 ? (
            <TapeSlivers stemIds={session.stemIds} />
          ) : session.status === "failed" ? (
            <span className="library-row__status-text">
              <span className="library-row__lamp" aria-hidden="true" />
              Split failed
            </span>
          ) : (
            <span className="library-row__status-text">
              {isSplitting ? "Splitting" : "Not split"}
            </span>
          )}
        </span>

        {isSplitting ? (
          <span className="library-row__prog">
            <span className="library-row__prog-label">
              {session.status === "queued" ? "Queued" : `Splitting, ${pct}%`}
            </span>
            <span className="library-row__prog-track" aria-hidden="true">
              <span ref={fillRef} className="library-row__prog-fill" />
            </span>
          </span>
        ) : (
          <span className="library-row__date">
            {formatSessionDate(session.createdAt, getNowMs())}
          </span>
        )}
      </button>

      <div className="library-row__remove-cell">
        <button
          type="button"
          className="console-key console-key--compact library-row__remove"
          aria-label={`Remove ${session.title}`}
          onClick={() => onRequestRemove(session.id, session.title)}
          {...tip("Remove from library", {
            detail:
              "Deletes this song's separated stems from SplitSound. Your original file is not touched.",
            placement: "left",
          })}
        >
          Remove
        </button>
      </div>
    </li>
  );
});

export const HomeView = memo(function HomeView({
  draggingFiles,
  onChooseFile,
  onOpenSession,
  onRequestRemove,
}: HomeViewProps) {
  const { sessions, artworkById, progressById, loaded } = useShallowStore(libraryStore, (s) => ({
    sessions: s.sessions,
    artworkById: s.artworkById,
    progressById: s.progressById,
    loaded: s.loaded,
  }));
  const { models, modelsLoaded } = useShallowStore(modelsStore, (s) => ({
    models: s.models,
    modelsLoaded: s.loaded,
  }));
  const platform = useShallowStore(settingsStore, (s) => s.appInfo.platform);

  const firstRunBytes = modelsLoaded ? computeFirstRunDownloadBytes(models) : 0;

  return (
    <main className="home">
      <section
        className={draggingFiles ? "home-drop home-drop--over" : "home-drop"}
        aria-label="Import audio"
      >
        <div className="home-drop__copy">
          <h1 className="home-drop__headline">
            {draggingFiles ? "Drop it here" : "Drop a song to split it"}
          </h1>
          <p className="home-drop__sub">
            Separate vocals, drums, bass, guitar and piano on this computer.
          </p>
          {firstRunBytes > 0 ? (
            <p className="home-drop__first-run">
              The first song you open downloads the models it needs (about{" "}
              {formatBytes(firstRunBytes)}), once.
            </p>
          ) : null}
          <div className="home-drop__actions">
            <button
              type="button"
              className="primary-btn"
              onClick={onChooseFile}
              {...tip("Open an audio file", {
                detail: "Or drag and drop a song anywhere on this window.",
                shortcut: formatModShortcut(platform, "O"),
                placement: "bottom",
              })}
            >
              Choose a file
            </button>
            <span className="home-drop__formats">MP3, WAV, FLAC, Opus, OGG, M4A, AIFF</span>
          </div>
        </div>
        <HomeIllustration draggingFiles={draggingFiles} />
      </section>

      <section className="home-library" aria-label="Library">
        <div className="home-library__head">
          <h2 className="home-library__title">Library</h2>
          {sessions.length > 0 ? (
            <span className="home-library__count">
              {sessions.length} {sessions.length === 1 ? "song" : "songs"}
            </span>
          ) : null}
        </div>
        <div className="home-library__scroll">
          {loaded && sessions.length === 0 ? (
            <p className="home-library__empty">
              Songs you split are kept here so you can come back to them.
            </p>
          ) : (
            <ul className="library-list">
              {sessions.map((s) => (
                <LibraryRow
                  key={s.id}
                  session={s}
                  artworkUrl={artworkById[s.id]}
                  progress={progressById[s.id]}
                  onOpenSession={onOpenSession}
                  onRequestRemove={onRequestRemove}
                />
              ))}
            </ul>
          )}
        </div>
      </section>
    </main>
  );
});
