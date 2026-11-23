import { memo, useEffect, useMemo, useRef } from "react";
import type { Session } from "../ipc/contract";
import { STEMS_BY_MODE } from "../ipc/contract";
import { useShallowStore } from "../stores/createStore";
import { playbackStore, seekPlayback, setLoopRegion } from "../stores/playbackStore";
import { sessionStore } from "../stores/sessionStore";
import { formatEta } from "../utils/format";
import {
  LoopWash,
  PlayheadLine,
  TimeRuler,
  useWellSeekHandlers,
  WaveformCanvas,
} from "../components/WaveformWell";
import "./SplittingView.css";

interface SplittingViewProps {
  session: Session;
  onCancel: () => void;
}

export const SplittingView = memo(function SplittingView({
  session,
  onCancel,
}: SplittingViewProps) {
  const { peaks, splitProgress } = useShallowStore(sessionStore, (s) => ({
    peaks: s.peaks,
    splitProgress: s.splitProgress,
  }));
  const { playing, playbackDurationSec, loop } = useShallowStore(playbackStore, (s) => ({
    playing: s.playback.playing,
    playbackDurationSec: s.playback.durationSec,
    loop: s.playback.loop,
  }));
  const fillRef = useRef<HTMLDivElement | null>(null);

  const isQueued = session.status === "queued";
  const fraction = isQueued ? 0 : Math.max(0, Math.min(1, splitProgress?.fraction ?? 0));
  const label = isQueued
    ? "Waiting for another split"
    : (splitProgress?.label ?? "Preparing audio");
  const stepText =
    !isQueued && splitProgress
      ? `Step ${splitProgress.step} of ${splitProgress.stepCount}`
      : isQueued
        ? "Queued"
        : "Starting";

  const stemIds = STEMS_BY_MODE[session.split?.mode ?? "all"];
  const isStaticScene =
    typeof window !== "undefined" && new URLSearchParams(window.location.search).has("scene");

  const splitOverlay = useMemo(
    () => ({
      stemIds,
      animate: !isStaticScene && !isQueued,
    }),
    [stemIds, isStaticScene, isQueued],
  );

  const durationSec = playbackDurationSec || session.track.durationSec;
  const seekHandlers = useWellSeekHandlers(durationSec, true, (pos) => {
    void seekPlayback(pos);
  });

  useEffect(() => {
    const el = fillRef.current;
    if (el) {
      el.style.setProperty("--progress", fraction.toFixed(4));
    }
  }, [fraction]);

  return (
    <main className="desk">
      <aside className="splitting-strip" aria-label="Split progress">
        <div className="strip-head">{isQueued ? "Queued" : "Splitting"}</div>
        <div className="splitting-strip__body" aria-live="polite">
          <h2 className="splitting-strip__label">{label}</h2>
          <p className="splitting-strip__step">{stepText}</p>
          <div
            className="splitting-strip__bar"
            role="progressbar"
            aria-label="Split progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(fraction * 100)}
          >
            <div ref={fillRef} className="splitting-strip__fill" />
          </div>
          <p className="splitting-strip__eta">
            {isQueued
              ? "Starts when the current split finishes"
              : formatEta(splitProgress?.etaSec ?? null)}
          </p>
          <div className="splitting-strip__actions">
            <button type="button" className="console-key" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      </aside>

      <section
        className="well well--single well--seekable"
        aria-label="Waveform splitting progress"
        {...seekHandlers}
      >
        <TimeRuler
          durationSec={durationSec}
          loop={loop}
          onSeek={(pos) => void seekPlayback(pos)}
          onChangeLoop={(l) => void setLoopRegion(l)}
        />
        <LoopWash loop={loop} durationSec={durationSec} />
        <div className="lane">
          <WaveformCanvas
            source="mix"
            peaks={peaks.mix}
            audible
            durationSec={durationSec}
            playing={playing}
            splitOverlay={splitOverlay}
            splitFraction={fraction}
          />
        </div>
        <PlayheadLine durationSec={durationSec} playing={playing} />
      </section>
    </main>
  );
});
