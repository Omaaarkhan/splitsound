import { memo, useEffect, useMemo, useRef, type KeyboardEvent, type PointerEvent } from "react";
import type { Peaks, SourceId, StemId } from "../ipc/contract";
import {
  acquirePlaybackAnimation,
  getInterpolatedPositionSec,
  subscribePlaybackFrame,
} from "../stores/playbackStore";
import { formatTime, rulerTickInterval } from "../utils/format";
import { STEM_CSS_VAR } from "../utils/stems";
import {
  type DecodedPeaks,
  decodePeaks,
  drawWaveform,
  drawWaveformRibbons,
  isDecodedPeaks,
  resolveCssColor,
} from "../utils/waveform";
import { tip } from "./primitives/Tooltip";
import "./WaveformWell.css";

interface LoopRange {
  startSec: number;
  endSec: number;
}

interface TimeRulerProps {
  durationSec: number;
  loop?: LoopRange | null;
  onSeek?: (positionSec: number) => void;
  onChangeLoop?: (loop: LoopRange | null) => void;
}

const MIN_LOOP_SEC = 1;
const DRAG_THRESHOLD_PX = 5;

export const TimeRuler = memo(function TimeRuler({
  durationSec,
  loop,
  onSeek,
  onChangeLoop,
}: TimeRulerProps) {
  const rulerRef = useRef<HTMLDivElement | null>(null);
  const rulerRectRef = useRef<{ left: number; width: number }>({ left: 0, width: 1 });
  const loopRef = useRef<HTMLDivElement | null>(null);
  const startHandleRef = useRef<HTMLDivElement | null>(null);
  const endHandleRef = useRef<HTMLDivElement | null>(null);

  const dragRef = useRef<{
    mode: "none" | "create" | "start" | "end";
    anchorX: number;
    anchorSec: number;
    moved: boolean;
    fixedStart: number;
    fixedEnd: number;
  }>({
    mode: "none",
    anchorX: 0,
    anchorSec: 0,
    moved: false,
    fixedStart: 0,
    fixedEnd: 0,
  });

  const ticks = useMemo(() => {
    const safeDur = Math.max(1, durationSec);
    const step = rulerTickInterval(safeDur);
    const out: { sec: number; leftPct: number }[] = [];
    for (let s = 0; s <= safeDur; s += step) {
      out.push({ sec: s, leftPct: (s / safeDur) * 100 });
    }
    return out;
  }, [durationSec]);

  useEffect(() => {
    const el = rulerRef.current;
    if (!el) return;
    const nodes = el.querySelectorAll<HTMLElement>(".ruler__tick");
    nodes.forEach((node, idx) => {
      const tick = ticks[idx];
      if (tick) {
        node.style.setProperty("--tick-left", `${tick.leftPct.toFixed(4)}%`);
      }
    });
  }, [ticks]);

  useEffect(() => {
    if (!loop || durationSec <= 0) return;
    const startPct = (loop.startSec / durationSec) * 100;
    const endPct = (loop.endSec / durationSec) * 100;
    const widthPct = Math.max(0, endPct - startPct);
    if (loopRef.current) {
      loopRef.current.style.setProperty("--loop-start", `${startPct.toFixed(3)}%`);
      loopRef.current.style.setProperty("--loop-width", `${widthPct.toFixed(3)}%`);
    }
    if (startHandleRef.current) {
      startHandleRef.current.style.setProperty("--handle-left", `${startPct.toFixed(3)}%`);
    }
    if (endHandleRef.current) {
      endHandleRef.current.style.setProperty("--handle-left", `${endPct.toFixed(3)}%`);
    }
  }, [loop, durationSec]);

  useEffect(() => {
    const el = rulerRef.current;
    if (!el) return;
    const syncRect = () => {
      const r = el.getBoundingClientRect();
      rulerRectRef.current = { left: r.left, width: r.width };
    };
    syncRect();
    const ro = new ResizeObserver(syncRect);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const refreshRulerRect = () => {
    const el = rulerRef.current;
    if (el) {
      const r = el.getBoundingClientRect();
      rulerRectRef.current = { left: r.left, width: r.width };
    }
  };

  const clientXToSec = (clientX: number): number => {
    if (durationSec <= 0) return 0;
    const rect = rulerRectRef.current;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
    return frac * durationSec;
  };

  const handleRulerPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || durationSec <= 0) return;
    e.preventDefault();
    e.stopPropagation();
    refreshRulerRect();
    const sec = clientXToSec(e.clientX);
    dragRef.current = {
      mode: "create",
      anchorX: e.clientX,
      anchorSec: sec,
      moved: false,
      fixedStart: sec,
      fixedEnd: sec,
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const startEdgePointerDown = (edge: "start" | "end", e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || !loop || durationSec <= 0) return;
    e.preventDefault();
    e.stopPropagation();
    refreshRulerRect();
    dragRef.current = {
      mode: edge,
      anchorX: e.clientX,
      anchorSec: edge === "start" ? loop.startSec : loop.endSec,
      moved: true,
      fixedStart: loop.startSec,
      fixedEnd: loop.endSec,
    };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.currentTarget.focus();
  };

  const handlePointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag.mode === "none") return;
    e.stopPropagation();
    const sec = clientXToSec(e.clientX);

    if (drag.mode === "create") {
      if (!drag.moved && Math.abs(e.clientX - drag.anchorX) >= DRAG_THRESHOLD_PX) {
        drag.moved = true;
      }
      if (drag.moved && onChangeLoop) {
        const startSec = Math.min(drag.anchorSec, sec);
        const endSec = Math.max(drag.anchorSec, sec);
        if (endSec - startSec >= 0.5) {
          onChangeLoop({ startSec, endSec });
        }
      }
      return;
    }

    if (drag.mode === "start" && onChangeLoop) {
      const nextStart = Math.max(0, Math.min(drag.fixedEnd - MIN_LOOP_SEC, sec));
      onChangeLoop({ startSec: nextStart, endSec: drag.fixedEnd });
      return;
    }

    if (drag.mode === "end" && onChangeLoop) {
      const nextEnd = Math.min(durationSec, Math.max(drag.fixedStart + MIN_LOOP_SEC, sec));
      onChangeLoop({ startSec: drag.fixedStart, endSec: nextEnd });
    }
  };

  const handlePointerUp = (e: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag.mode === "none") return;
    e.stopPropagation();
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    const wasCreateClick = drag.mode === "create" && !drag.moved;
    const clickSec = drag.anchorSec;
    dragRef.current.mode = "none";

    if (wasCreateClick) {
      onSeek?.(clickSec);
    }
  };

  const handleEdgeKeyDown = (edge: "start" | "end") => (e: KeyboardEvent<HTMLDivElement>) => {
    if (!loop || !onChangeLoop || durationSec <= 0) return;
    const step = e.shiftKey ? 5 : 1;
    let nextStart = loop.startSec;
    let nextEnd = loop.endSec;

    switch (e.key) {
      case "ArrowLeft":
        if (edge === "start") nextStart = Math.max(0, loop.startSec - step);
        else nextEnd = Math.max(loop.startSec + MIN_LOOP_SEC, loop.endSec - step);
        break;
      case "ArrowRight":
        if (edge === "start")
          nextStart = Math.min(loop.endSec - MIN_LOOP_SEC, loop.startSec + step);
        else nextEnd = Math.min(durationSec, loop.endSec + step);
        break;
      case "Home":
        if (edge === "start") nextStart = 0;
        else nextEnd = Math.min(durationSec, loop.startSec + MIN_LOOP_SEC);
        break;
      case "End":
        if (edge === "start") nextStart = Math.max(0, loop.endSec - MIN_LOOP_SEC);
        else nextEnd = durationSec;
        break;
      default:
        return;
    }

    e.preventDefault();
    e.stopPropagation();
    onChangeLoop({ startSec: nextStart, endSec: nextEnd });
  };

  return (
    <div
      ref={rulerRef}
      className="ruler"
      onPointerDown={handleRulerPointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      {...tip("Timeline", {
        detail: "Click to jump, or drag across to loop a section.",
        placement: "bottom",
      })}
    >
      {ticks.map((t) => (
        <span key={t.sec} className="ruler__tick" aria-hidden="true">
          {formatTime(t.sec)}
        </span>
      ))}
      {loop ? (
        <>
          <div ref={loopRef} className="ruler__loop" aria-hidden="true" />
          <div
            ref={startHandleRef}
            role="slider"
            tabIndex={0}
            className="ruler__handle"
            aria-label="Loop start"
            aria-valuemin={0}
            aria-valuemax={Math.max(0, Math.floor(loop.endSec - MIN_LOOP_SEC))}
            aria-valuenow={Math.round(loop.startSec)}
            aria-valuetext={formatTime(loop.startSec)}
            onPointerDown={(e) => startEdgePointerDown("start", e)}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onKeyDown={handleEdgeKeyDown("start")}
            {...tip(`Loop start (${formatTime(loop.startSec)})`, {
              detail: "Drag or use arrow keys to adjust.",
              placement: "bottom",
            })}
          />
          <div
            ref={endHandleRef}
            role="slider"
            tabIndex={0}
            className="ruler__handle"
            aria-label="Loop end"
            aria-valuemin={Math.min(
              Math.round(durationSec),
              Math.ceil(loop.startSec + MIN_LOOP_SEC),
            )}
            aria-valuemax={Math.round(durationSec)}
            aria-valuenow={Math.round(loop.endSec)}
            aria-valuetext={formatTime(loop.endSec)}
            onPointerDown={(e) => startEdgePointerDown("end", e)}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onKeyDown={handleEdgeKeyDown("end")}
            {...tip(`Loop end (${formatTime(loop.endSec)})`, {
              detail: "Drag or use arrow keys to adjust.",
              placement: "bottom",
            })}
          />
        </>
      ) : null}
    </div>
  );
});

export const LoopWash = memo(function LoopWash({
  loop,
  durationSec,
}: {
  loop: LoopRange | null;
  durationSec: number;
}) {
  const washRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = washRef.current;
    if (!el || !loop || durationSec <= 0) return;
    const startPct = (loop.startSec / durationSec) * 100;
    const widthPct = Math.max(0, ((loop.endSec - loop.startSec) / durationSec) * 100);
    el.style.setProperty("--loop-start", `${startPct.toFixed(3)}%`);
    el.style.setProperty("--loop-width", `${widthPct.toFixed(3)}%`);
  }, [loop, durationSec]);

  if (!loop) return null;
  return <div ref={washRef} className="well__loop-wash" aria-hidden="true" />;
});

interface WaveformCanvasProps {
  source: SourceId;
  peaks: Peaks | DecodedPeaks | undefined;
  audible: boolean;
  durationSec: number;
  playing: boolean;
  /** Optional splitting cut line bands to draw behind it. */
  splitOverlay?: {
    fraction?: number;
    stemIds: readonly StemId[];
    animate: boolean;
  };
  /** Cut fraction passed cleanly outside `splitOverlay` so 10 Hz updates don't re-subscribe listeners. */
  splitFraction?: number;
}

interface WaveformCacheLayer {
  w: number;
  h: number;
  dpr: number;
  playedCanvas: HTMLCanvasElement;
  aheadCanvas: HTMLCanvasElement;
  ribbonsCanvas: HTMLCanvasElement | null;
  lampColor: string;
}

function makeOffscreenCanvas(wPx: number, hPx: number): HTMLCanvasElement {
  const off = document.createElement("canvas");
  off.width = wPx;
  off.height = hPx;
  return off;
}

export const WaveformCanvas = memo(function WaveformCanvas({
  source,
  peaks,
  audible,
  durationSec,
  playing: _playing,
  splitOverlay,
  splitFraction,
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const decoded = useMemo(
    () => (peaks ? (isDecodedPeaks(peaks) ? peaks : decodePeaks(peaks)) : null),
    [peaks],
  );
  const initialCut = splitFraction ?? splitOverlay?.fraction ?? 0;
  const smoothCutRef = useRef<number>(initialCut);
  const targetCutRef = useRef<number>(initialCut);
  const sizeRef = useRef<{ w: number; h: number; dpr: number }>({ w: 0, h: 0, dpr: 1 });
  const cacheRef = useRef<WaveformCacheLayer | null>(null);
  const paintRef = useRef<() => void>(() => {});

  useEffect(() => {
    const frac = splitFraction ?? splitOverlay?.fraction;
    if (frac !== undefined) {
      targetCutRef.current = frac;
      if (!splitOverlay?.animate) {
        smoothCutRef.current = frac;
        paintRef.current();
      }
    }
  }, [splitFraction, splitOverlay]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const color = resolveCssColor(STEM_CSS_VAR[source]);
    const ink3Color = resolveCssColor("--ink-3", "#95a8b3");
    const lampColor = resolveCssColor("--lamp", "#fbc044");
    const stemColors = splitOverlay
      ? splitOverlay.stemIds.map((id) => resolveCssColor(STEM_CSS_VAR[id]))
      : [];
    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const animateSplit = Boolean(splitOverlay?.animate) && !reducedMotion;

    const rebuildCache = (w: number, h: number, dpr: number): WaveformCacheLayer => {
      const targetW = Math.max(1, Math.round(w * dpr));
      const targetH = Math.max(1, Math.round(h * dpr));
      const isMix = source === "mix" || Boolean(splitOverlay);
      const padY = isMix ? h * 0.15 : 7;

      const playedCanvas = makeOffscreenCanvas(targetW, targetH);
      const aheadCanvas = makeOffscreenCanvas(targetW, targetH);
      let ribbonsCanvas: HTMLCanvasElement | null = null;

      if (decoded) {
        const playedCtx = playedCanvas.getContext("2d");
        if (playedCtx) {
          playedCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
          drawWaveform(playedCtx, {
            peaks: decoded,
            color,
            unplayedColor: isMix ? ink3Color : undefined,
            x: 0,
            y: 0,
            width: w,
            height: h,
            playFraction: 1,
            audible: splitOverlay ? true : audible,
            paddingY: padY,
            filledPlayedAlpha: 1,
            filledAheadAlpha: isMix ? 1 : 0.55,
          });
        }

        const aheadCtx = aheadCanvas.getContext("2d");
        if (aheadCtx) {
          aheadCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
          drawWaveform(aheadCtx, {
            peaks: decoded,
            color,
            unplayedColor: isMix ? ink3Color : undefined,
            x: 0,
            y: 0,
            width: w,
            height: h,
            playFraction: 0,
            audible: splitOverlay ? true : audible,
            paddingY: padY,
            filledPlayedAlpha: 1,
            filledAheadAlpha: isMix ? 1 : 0.55,
          });
        }

        if (splitOverlay && stemColors.length > 0) {
          ribbonsCanvas = makeOffscreenCanvas(targetW, targetH);
          const ribCtx = ribbonsCanvas.getContext("2d");
          if (ribCtx) {
            ribCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
            drawWaveformRibbons(ribCtx, {
              peaks: decoded,
              colors: stemColors,
              x: 0,
              y: 0,
              width: w,
              height: h,
              paddingY: padY,
              gapPx: 2,
            });
          }
        }
      }

      const built: WaveformCacheLayer = {
        w,
        h,
        dpr,
        playedCanvas,
        aheadCanvas,
        ribbonsCanvas,
        lampColor,
      };
      cacheRef.current = built;
      return built;
    };

    const drawPlayheadSplit = (
      ctx: CanvasRenderingContext2D,
      cache: WaveformCacheLayer,
      w: number,
      h: number,
      playFrac: number,
    ) => {
      const playX = w * playFrac;
      if (playX > 0) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, playX, h);
        ctx.clip();
        ctx.drawImage(cache.playedCanvas, 0, 0, w, h);
        ctx.restore();
      }
      if (playX < w) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(playX, 0, w - playX, h);
        ctx.clip();
        ctx.drawImage(cache.aheadCanvas, 0, 0, w, h);
        ctx.restore();
      }
    };

    const paint = () => {
      if (animateSplit) {
        const diff = targetCutRef.current - smoothCutRef.current;
        smoothCutRef.current =
          Math.abs(diff) < 0.0005 ? targetCutRef.current : smoothCutRef.current + diff * 0.12;
      }

      const { w, h, dpr } = sizeRef.current;
      if (w <= 0 || h <= 0) return;

      const targetW = Math.round(w * dpr);
      const targetH = Math.round(h * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      let cache = cacheRef.current;
      if (!cache || cache.w !== w || cache.h !== h || cache.dpr !== dpr) {
        cache = rebuildCache(w, h, dpr);
      }

      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const playFrac =
        durationSec > 0 ? Math.max(0, Math.min(1, getInterpolatedPositionSec() / durationSec)) : 0;

      if (splitOverlay) {
        const cutFrac = Math.max(0, Math.min(1, smoothCutRef.current));
        const cx = cutFrac * w;
        const dx = h * 0.5;
        const topX = cx + dx;
        const botX = cx - dx;

        if (decoded) {
          if (cache.ribbonsCanvas && topX > 0) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(topX, 0);
            ctx.lineTo(botX, h);
            ctx.lineTo(0, h);
            ctx.closePath();
            ctx.clip();
            ctx.drawImage(cache.ribbonsCanvas, 0, 0, w, h);
            ctx.restore();
          }

          ctx.save();
          ctx.beginPath();
          ctx.moveTo(topX, 0);
          ctx.lineTo(w, 0);
          ctx.lineTo(w, h);
          ctx.lineTo(botX, h);
          ctx.closePath();
          ctx.clip();
          drawPlayheadSplit(ctx, cache, w, h, playFrac);
          ctx.restore();
        }

        ctx.save();
        ctx.strokeStyle = cache.lampColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(botX, h);
        ctx.lineTo(topX, 0);
        ctx.stroke();
        ctx.restore();
      } else if (decoded) {
        drawPlayheadSplit(ctx, cache, w, h, playFrac);
      }

      ctx.restore();
    };

    paintRef.current = paint;

    const syncSizeAndPaint = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(3, window.devicePixelRatio || 1);
      sizeRef.current = { w: rect.width, h: rect.height, dpr };
      paint();
    };

    syncSizeAndPaint();

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      const dpr = Math.min(3, window.devicePixelRatio || 1);
      if (entry) {
        sizeRef.current = {
          w: entry.contentRect.width,
          h: entry.contentRect.height,
          dpr,
        };
        paint();
      } else {
        syncSizeAndPaint();
      }
    });
    ro.observe(canvas);

    const unsubFrame = subscribePlaybackFrame(paint);
    const releaseAnim = animateSplit ? acquirePlaybackAnimation() : null;

    return () => {
      paintRef.current = () => {};
      ro.disconnect();
      unsubFrame();
      releaseAnim?.();
    };
  }, [source, decoded, audible, durationSec, splitOverlay]);

  return <canvas ref={canvasRef} className="lane__canvas" aria-hidden="true" />;
});

interface PlayheadLineProps {
  durationSec: number;
  playing?: boolean;
}

export const PlayheadLine = memo(function PlayheadLine({ durationSec }: PlayheadLineProps) {
  const laneRef = useRef<HTMLDivElement | null>(null);
  const lastValRef = useRef<string>("");

  useEffect(() => {
    const update = () => {
      const el = laneRef.current;
      if (!el) return;
      const frac =
        durationSec > 0 ? Math.max(0, Math.min(1, getInterpolatedPositionSec() / durationSec)) : 0;
      const next = frac.toFixed(5);
      if (next !== lastValRef.current) {
        lastValRef.current = next;
        el.style.setProperty("--playhead", next);
      }
    };
    update();
    return subscribePlaybackFrame(update);
  }, [durationSec]);

  return (
    <div ref={laneRef} className="playhead-lane" aria-hidden="true">
      <div className="playhead" />
    </div>
  );
});

export function useWellSeekHandlers(
  durationSec: number,
  enabled: boolean,
  onSeek: (positionSec: number) => void,
) {
  const draggingRef = useRef(false);
  const rectRef = useRef<{ left: number; width: number }>({ left: 0, width: 1 });
  const paramsRef = useRef({ durationSec, enabled, onSeek });

  useEffect(() => {
    paramsRef.current = { durationSec, enabled, onSeek };
  }, [durationSec, enabled, onSeek]);

  return useMemo(() => {
    const seekFromClientX = (clientX: number) => {
      const { durationSec: dur, enabled: en, onSeek: cb } = paramsRef.current;
      if (!en || dur <= 0) return;
      const rect = rectRef.current;
      const frac = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
      cb(frac * dur);
    };

    return {
      onPointerDown: (e: PointerEvent<HTMLElement>) => {
        if (!paramsRef.current.enabled || e.button !== 0) return;
        e.preventDefault();
        draggingRef.current = true;
        const r = e.currentTarget.getBoundingClientRect();
        rectRef.current = { left: r.left, width: r.width };
        e.currentTarget.setPointerCapture(e.pointerId);
        seekFromClientX(e.clientX);
      },
      onPointerMove: (e: PointerEvent<HTMLElement>) => {
        if (!draggingRef.current) return;
        seekFromClientX(e.clientX);
      },
      onPointerUp: (e: PointerEvent<HTMLElement>) => {
        if (!draggingRef.current) return;
        draggingRef.current = false;
        if (e.currentTarget.hasPointerCapture(e.pointerId)) {
          e.currentTarget.releasePointerCapture(e.pointerId);
        }
      },
      onPointerCancel: (e: PointerEvent<HTMLElement>) => {
        if (!draggingRef.current) return;
        draggingRef.current = false;
        if (e.currentTarget.hasPointerCapture(e.pointerId)) {
          e.currentTarget.releasePointerCapture(e.pointerId);
        }
      },
    };
  }, []);
}
