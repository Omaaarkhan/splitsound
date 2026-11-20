import { useEffect, useRef } from "react";
import type { StemId } from "../ipc/contract";
import { getPlaybackLevels, subscribePlaybackTicks } from "../stores/playbackStore";
import "./MeterBar.css";

interface MeterBarProps {
  stemId: StemId;
}

export function MeterBar({ stemId }: MeterBarProps) {
  const fillRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    const sync = () => {
      const el = fillRef.current;
      if (!el) return;
      const lvl = Math.max(0, Math.min(1, getPlaybackLevels()[stemId] ?? 0));
      el.style.setProperty("--level", lvl.toFixed(3));
    };
    sync();
    return subscribePlaybackTicks(sync);
  }, [stemId]);

  return (
    <div className={`meter stem-${stemId}`} aria-hidden="true">
      <span ref={fillRef} className="meter__fill" />
    </div>
  );
}
