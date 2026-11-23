import { memo, useEffect, useRef } from "react";
import type { StemId } from "../ipc/contract";
import { getPlaybackLevels, subscribePlaybackFrame } from "../stores/playbackStore";
import "./MeterBar.css";

interface MeterBarProps {
  stemId: StemId;
}

export const MeterBar = memo(function MeterBar({ stemId }: MeterBarProps) {
  const fillRef = useRef<HTMLSpanElement | null>(null);
  const lastLevelRef = useRef<string>("");

  useEffect(() => {
    const sync = () => {
      const el = fillRef.current;
      if (!el) return;
      const lvl = Math.max(0, Math.min(1, getPlaybackLevels()[stemId] ?? 0));
      const next = lvl.toFixed(3);
      if (next !== lastLevelRef.current) {
        lastLevelRef.current = next;
        el.style.setProperty("--level", next);
      }
    };
    sync();
    return subscribePlaybackFrame(sync);
  }, [stemId]);

  return (
    <div className={`meter stem-${stemId}`} aria-hidden="true">
      <span ref={fillRef} className="meter__fill" />
    </div>
  );
});
