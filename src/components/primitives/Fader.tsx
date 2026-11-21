import { useEffect, useRef, type KeyboardEvent, type PointerEvent } from "react";
import {
  dbToFaderFraction,
  FADER_MAX_DB,
  FADER_MIN_DB,
  FADER_STEP_DB,
  faderFractionToDb,
  snapDb,
} from "../../utils/fader";
import { formatDb } from "../../utils/format";
import type { TooltipAttrs } from "./Tooltip";
import "./Fader.css";

interface FaderProps {
  valueDb: number;
  label: string;
  onChange: (db: number) => void;
  className?: string;
  tooltip?: TooltipAttrs;
}

export function Fader({ valueDb, label, onChange, className, tooltip }: FaderProps) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);

  const fraction = dbToFaderFraction(valueDb);

  useEffect(() => {
    const el = trackRef.current;
    if (el) {
      el.style.setProperty("--pos", fraction.toFixed(4));
    }
  }, [fraction]);

  const updateFromClientX = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const usable = Math.max(1, rect.width - 10);
    const raw = (clientX - rect.left - 5) / usable;
    const nextDb = faderFractionToDb(raw);
    if (nextDb !== valueDb) {
      onChange(nextDb);
    }
  };

  const handlePointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    draggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    e.currentTarget.focus();
    updateFromClientX(e.clientX);
  };

  const handlePointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    updateFromClientX(e.clientX);
  };

  const handlePointerUp = (e: PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  const handleDoubleClick = () => {
    onChange(0);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    let next: number;
    switch (e.key) {
      case "ArrowRight":
      case "ArrowUp":
        next = snapDb(valueDb + (e.shiftKey ? 2 : FADER_STEP_DB));
        break;
      case "ArrowLeft":
      case "ArrowDown":
        next = snapDb(valueDb - (e.shiftKey ? 2 : FADER_STEP_DB));
        break;
      case "PageUp":
        next = snapDb(valueDb + 3);
        break;
      case "PageDown":
        next = snapDb(valueDb - 3);
        break;
      case "Home":
        next = FADER_MIN_DB;
        break;
      case "End":
        next = FADER_MAX_DB;
        break;
      case "0":
      case "Backspace":
      case "Delete":
        next = 0;
        break;
      default:
        return;
    }
    e.preventDefault();
    e.stopPropagation();
    onChange(next);
  };

  return (
    <div
      ref={trackRef}
      role="slider"
      tabIndex={0}
      className={className ? `fader ${className}` : "fader"}
      aria-label={label}
      aria-valuemin={FADER_MIN_DB}
      aria-valuemax={FADER_MAX_DB}
      aria-valuenow={snapDb(valueDb)}
      aria-valuetext={`${formatDb(valueDb)} dB`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onDoubleClick={handleDoubleClick}
      onKeyDown={handleKeyDown}
      {...tooltip}
    >
      <span className="fader__cap" aria-hidden="true" />
    </div>
  );
}
