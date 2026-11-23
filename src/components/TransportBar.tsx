import { memo, useCallback, useEffect, useRef } from "react";
import type { Session } from "../ipc/contract";
import { useShallowStore } from "../stores/createStore";
import {
  getInterpolatedPositionSec,
  playbackStore,
  setMasterGain,
  subscribePlaybackFrame,
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

export const TransportBar = memo(function TransportBar({
  session,
  onOpenExport,
}: TransportBarProps) {
  const { playing, sessionId, positionSec, playbackDurationSec, loop, source, masterGainDb } =
    useShallowStore(playbackStore, (s) => ({
      playing: s.playback.playing,
      sessionId: s.playback.sessionId,
      positionSec: s.playback.positionSec,
      playbackDurationSec: s.playback.durationSec,
      loop: s.playback.loop,
      source: s.playback.source,
      masterGainDb: s.playback.masterGainDb,
    }));
  const { platform } = useShallowStore(settingsStore, (s) => ({
    platform: s.appInfo.platform,
  }));
  const posRef = useRef<HTMLSpanElement | null>(null);
  const lastTextRef = useRef<string>("");

  const isReady = session.status === "ready";
  const durationSec = playbackDurationSec || session.track.durationSec;
  const masterTip = tip(`Master volume (${formatDb(masterGainDb)} dB)`, {
    detail:
      "Overall volume for listening and mixdown export. Double-click to reset to 0 dB. Individual stem exports keep their original level.",
    placement: "top",
  });

  const handleMasterChange = useCallback((db: number) => {
    void setMasterGain(db);
  }, []);

  useEffect(() => {
    const syncText = () => {
      if (!posRef.current) return;
      const next = formatTimeTenths(getInterpolatedPositionSec());
      if (next !== lastTextRef.current) {
        lastTextRef.current = next;
        posRef.current.textContent = next;
      }
    };
    syncText();
    return subscribePlaybackFrame(syncText);
  }, [playing, sessionId, positionSec]);

  return (
    <footer className="transport" aria-label="Playback controls">
      <button
        type="button"
        className="transport__play"
        aria-label={playing ? "Pause" : "Play"}
        onClick={() => void togglePlayback()}
        {...tip(playing ? "Pause" : "Play", {
          detail: "Left and Right arrows jump 5 seconds. Home jumps to the start.",
          shortcut: "Space",
          placement: "top",
        })}
      >
        {playing ? (
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
        <span ref={posRef}>{formatTimeTenths(positionSec)}</span>
        <span className="transport__total">/ {formatTime(durationSec)}</span>
      </div>

      <button
        type="button"
        className={loop ? "console-key console-key--selected" : "console-key"}
        aria-pressed={loop !== null}
        onClick={() => void toggleLoop()}
        {...tip(loop ? "Turn off loop" : "Loop playback", {
          detail: loop
            ? `Looping ${formatTime(loop.startSec)} to ${formatTime(loop.endSec)}. Drag handles on the timeline to resize.`
            : "Repeats the whole song, or drag across the timeline above to loop a section.",
          placement: "top",
        })}
      >
        Loop
      </button>

      {isReady ? (
        <button
          type="button"
          className={source === "original" ? "console-key console-key--lit" : "console-key"}
          aria-pressed={source === "original"}
          onClick={() => void toggleHearOriginal()}
          {...tip(source === "original" ? "Return to stems" : "Hear original", {
            detail:
              source === "original"
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
          valueDb={masterGainDb}
          label="Master level"
          onChange={handleMasterChange}
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
            shortcut: formatModShortcut(platform, "E"),
            placement: "top",
          })}
        >
          Export stems
        </button>
      ) : null}
    </footer>
  );
});
