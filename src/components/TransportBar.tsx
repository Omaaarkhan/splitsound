import { useEffect, useRef } from "react";
import type { Session } from "../ipc/contract";
import { useStore } from "../stores/createStore";
import {
  getInterpolatedPositionSec,
  playbackStore,
  setMasterGain,
  subscribePlaybackTicks,
  toggleHearOriginal,
  toggleLoop,
  togglePlayback,
} from "../stores/playbackStore";
import { settingsStore } from "../stores/settingsStore";
import { formatDb, formatTime, formatTimeTenths } from "../utils/format";
import { Fader } from "./primitives/Fader";
import { formatModShortcut, tip } from "./primitives/Tooltip";
import "./TransportBar.css";

interface TransportBarProps {
  session: Session;
  onOpenExport: () => void;
}

export function TransportBar({ session, onOpenExport }: TransportBarProps) {
  const { playback } = useStore(playbackStore);
  const { appInfo } = useStore(settingsStore);
  const posRef = useRef<HTMLSpanElement | null>(null);

  const isReady = session.status === "ready";
  const durationSec = playback.durationSec || session.track.durationSec;
  const masterTip = tip(`Master volume (${formatDb(playback.masterGainDb)} dB)`, {
    detail:
      "Overall volume for listening and mixdown export. Double-click to reset to 0 dB. Individual stem exports keep their original level.",
    placement: "top",
  });

  useEffect(() => {
    const syncText = () => {
      if (posRef.current) {
        posRef.current.textContent = formatTimeTenths(getInterpolatedPositionSec());
      }
    };
    syncText();
    const unsub = subscribePlaybackTicks(syncText);

    if (!playback.playing) {
      return unsub;
    }

    let rafId = 0;
    const frame = () => {
      syncText();
      rafId = requestAnimationFrame(frame);
    };
    rafId = requestAnimationFrame(frame);

    return () => {
      unsub();
      cancelAnimationFrame(rafId);
    };
  }, [playback.playing, playback.sessionId, playback.positionSec]);

  return (
    <footer className="transport" aria-label="Playback controls">
      <button
        type="button"
        className="transport__play"
        aria-label={playback.playing ? "Pause" : "Play"}
        onClick={() => void togglePlayback()}
        {...tip(playback.playing ? "Pause" : "Play", {
          detail: "Left and Right arrows jump 5 seconds. Home jumps to the start.",
          shortcut: "Space",
          placement: "top",
        })}
      >
        {playback.playing ? (
          <svg
            className="transport__play-glyph"
            width="14"
            height="16"
            viewBox="0 0 14 16"
            aria-hidden="true"
          >
            <rect x="0" y="0" width="5" height="16" />
            <rect x="9" y="0" width="5" height="16" />
          </svg>
        ) : (
          <svg
            className="transport__play-glyph"
            width="16"
            height="18"
            viewBox="0 0 16 18"
            aria-hidden="true"
          >
            <path d="M1 0l15 9-15 9z" />
          </svg>
        )}
      </button>

      <div className="transport__time" aria-label="Playback position">
        <span ref={posRef}>{formatTimeTenths(playback.positionSec)}</span>
        <span className="transport__total">/ {formatTime(durationSec)}</span>
      </div>

      <button
        type="button"
        className={playback.loop ? "console-key console-key--selected" : "console-key"}
        aria-pressed={playback.loop !== null}
        onClick={() => void toggleLoop()}
        {...tip(playback.loop ? "Turn off loop" : "Loop playback", {
          detail: playback.loop
            ? `Looping ${formatTime(playback.loop.startSec)} to ${formatTime(playback.loop.endSec)}. Drag handles on the timeline to resize.`
            : "Repeats the whole song, or drag across the timeline above to loop a section.",
          placement: "top",
        })}
      >
        Loop
      </button>

      {isReady ? (
        <button
          type="button"
          className={
            playback.source === "original" ? "console-key console-key--lit" : "console-key"
          }
          aria-pressed={playback.source === "original"}
          onClick={() => void toggleHearOriginal()}
          {...tip(playback.source === "original" ? "Return to stems" : "Hear original", {
            detail:
              playback.source === "original"
                ? "Switch back from the original song to your separated stems."
                : "Temporarily switch to the untouched song to compare against your stems.",
            placement: "top",
          })}
        >
          Hear original
        </button>
      ) : null}

      <div className="transport__spacer" />

      <div className="transport__master" {...masterTip}>
        <span>Master</span>
        <Fader
          valueDb={playback.masterGainDb}
          label="Master level"
          onChange={(db) => void setMasterGain(db)}
          tooltip={masterTip}
        />
      </div>

      {isReady ? (
        <button
          type="button"
          className="primary-btn"
          onClick={onOpenExport}
          {...tip("Export stems", {
            detail: "Save individual stems or a custom mixdown as WAV, FLAC, or MP3.",
            shortcut: formatModShortcut(appInfo.platform, "E"),
            placement: "top",
          })}
        >
          Export stems
        </button>
      ) : null}
    </footer>
  );
}
