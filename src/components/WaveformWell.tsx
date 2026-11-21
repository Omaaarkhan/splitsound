import { useEffect, useMemo, useRef, type KeyboardEvent, type PointerEvent } from "react";
import type { Peaks, SourceId, StemId } from "../ipc/contract";
import { getInterpolatedPositionSec, subscribePlaybackTicks } from "../stores/playbackStore";
import { formatTime, rulerTickInterval } from "../utils/format";
import { STEM_CSS_VAR } from "../utils/stems";
import { decodePeaks, drawWaveform, drawWaveformRibbons, resolveCssColor } from "../utils/waveform";
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

export function TimeRuler({ durationSec, loop, onSeek, onChangeLoop }: TimeRulerProps) {
  const rulerRef = useRef<HTMLDivElement | null>(null);
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

  const clientXToSec = (clientX: number): number => {
    const el = rulerRef.current;
    if (!el || durationSec <= 0) return 0;
    const rect = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
    return frac * durationSec;
  };

  const handleRulerPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || durationSec <= 0) return;
    e.preventDefault();
    e.stopPropagation();
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
          />
        </>
      ) : null}
    </div>
  );
}

export function LoopWash({ loop, durationSec }: { loop: LoopRange | null; durationSec: number }) {
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
}

interface WaveformCanvasProps {
  source: SourceId;
  peaks: Peaks | undefined;
  audible: boolean;
  durationSec: number;
  playing: boolean;
  /** Optional splitting cut line fraction (0..1) and bands to draw behind it. */
  splitOverlay?: {
    fraction: number;
    stemIds: readonly StemId[];
    animate: boolean;
  };
}

export function WaveformCanvas({
  source,
  peaks,
  audible,
  durationSec,
  playing,
  splitOverlay,
}: WaveformCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const decoded = useMemo(() => (peaks ? decodePeaks(peaks) : null), [peaks]);
  const smoothCutRef = useRef<number>(splitOverlay?.fraction ?? 0);
  const targetCutRef = useRef<number>(splitOverlay?.fraction ?? 0);

  useEffect(() => {
    if (splitOverlay) {
      targetCutRef.current = splitOverlay.fraction;
      if (!splitOverlay.animate) {
        smoothCutRef.current = splitOverlay.fraction;
      }
    }
  }, [splitOverlay]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const color = resolveCssColor(STEM_CSS_VAR[source]);
    const ink3Color = resolveCssColor("--ink-3", "#95a8b3");
    const lampColor = resolveCssColor("--lamp", "#fbc044");
    const stemColors = splitOverlay
      ? splitOverlay.stemIds.map((id) => resolveCssColor(STEM_CSS_VAR[id]))
      : [];

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

      const playFrac =
        durationSec > 0 ? Math.max(0, Math.min(1, getInterpolatedPositionSec() / durationSec)) : 0;

      if (splitOverlay) {
        // 45-degree diagonal cut (leaning forward slash `/` across full lane height)
        const cutFrac = Math.max(0, Math.min(1, smoothCutRef.current));
        const cx = cutFrac * w;
        const dx = h * 0.5;
        const topX = cx + dx;
        const botX = cx - dx;
        // Cap original mix waveform amplitude at ~70% of lane height
        const mixPadY = h * 0.15;

        if (decoded) {
          // Passed side of the 45-degree cut: ribbons that follow the waveform's own height
          // at every x, split into N equal stem colours with 2px gaps of well colour.
          if (stemColors.length > 0 && topX > 0) {
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(topX, 0);
            ctx.lineTo(botX, h);
            ctx.lineTo(0, h);
            ctx.closePath();
            ctx.clip();

            drawWaveformRibbons(ctx, {
              peaks: decoded,
              colors: stemColors,
              x: 0,
              y: 0,
              width: w,
              height: h,
              paddingY: mixPadY,
              gapPx: 2,
            });
            ctx.restore();
          }

          // Ahead side of the 45-degree cut: plain original waveform (--ink played, --ink-3 unplayed, full opacity)
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(topX, 0);
          ctx.lineTo(w, 0);
          ctx.lineTo(w, h);
          ctx.lineTo(botX, h);
          ctx.closePath();
          ctx.clip();
          drawWaveform(ctx, {
            peaks: decoded,
            color,
            unplayedColor: ink3Color,
            x: 0,
            y: 0,
            width: w,
            height: h,
            playFraction: playFrac,
            audible: true,
            paddingY: mixPadY,
            filledPlayedAlpha: 1,
            filledAheadAlpha: 1,
          });
          ctx.restore();
        }

        // 45-degree lamp cut line
        ctx.save();
        ctx.strokeStyle = lampColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(botX, h);
        ctx.lineTo(topX, 0);
        ctx.stroke();
        ctx.restore();
      } else if (decoded) {
        const isMix = source === "mix";
        drawWaveform(ctx, {
          peaks: decoded,
          color,
          unplayedColor: isMix ? ink3Color : undefined,
          x: 0,
          y: 0,
          width: w,
          height: h,
          playFraction: playFrac,
          audible,
          paddingY: isMix ? h * 0.15 : 7,
          filledPlayedAlpha: 1,
          filledAheadAlpha: isMix ? 1 : 0.55,
        });
      }

      ctx.restore();
    };

    paint();

    const ro = new ResizeObserver(paint);
    ro.observe(canvas);
    const unsubTicks = subscribePlaybackTicks(paint);

    let rafId = 0;
    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const shouldAnimate = playing || (Boolean(splitOverlay?.animate) && !reducedMotion);

    if (shouldAnimate) {
      const loop = () => {
        if (splitOverlay?.animate) {
          const diff = targetCutRef.current - smoothCutRef.current;
          smoothCutRef.current =
            Math.abs(diff) < 0.0005 ? targetCutRef.current : smoothCutRef.current + diff * 0.12;
        }
        paint();
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    }

    return () => {
      ro.disconnect();
      unsubTicks();
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [source, decoded, audible, durationSec, playing, splitOverlay]);

  return <canvas ref={canvasRef} className="lane__canvas" aria-hidden="true" />;
}

interface PlayheadLineProps {
  durationSec: number;
  playing: boolean;
}

export function PlayheadLine({ durationSec, playing }: PlayheadLineProps) {
  const lineRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const update = () => {
      const el = lineRef.current;
      if (!el) return;
      const frac =
        durationSec > 0 ? Math.max(0, Math.min(1, getInterpolatedPositionSec() / durationSec)) : 0;
      el.style.setProperty("--playhead", `${(frac * 100).toFixed(4)}%`);
    };
    update();
    const unsub = subscribePlaybackTicks(update);

    if (!playing) return unsub;

    let rafId = 0;
    const loop = () => {
      update();
      rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);

    return () => {
      unsub();
      cancelAnimationFrame(rafId);
    };
  }, [durationSec, playing]);

  return <div ref={lineRef} className="playhead" aria-hidden="true" />;
}

export function useWellSeekHandlers(
  durationSec: number,
  enabled: boolean,
  onSeek: (positionSec: number) => void,
) {
  const draggingRef = useRef(false);

  const seekFromEvent = (e: PointerEvent<HTMLElement>) => {
    if (!enabled || durationSec <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
    onSeek(frac * durationSec);
  };

  return {
    onPointerDown: (e: PointerEvent<HTMLElement>) => {
      if (!enabled || e.button !== 0) return;
      e.preventDefault();
      draggingRef.current = true;
      e.currentTarget.setPointerCapture(e.pointerId);
      seekFromEvent(e);
    },
    onPointerMove: (e: PointerEvent<HTMLElement>) => {
      if (!draggingRef.current) return;
      seekFromEvent(e);
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
}
