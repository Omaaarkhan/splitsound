import { memo, useCallback, useEffect, useMemo, useRef } from "react";
import type { Peaks, Session, StemId, StemInfo } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { startNativeStemDrag } from "../ipc/platform";
import { useShallowStore } from "../stores/createStore";
import {
  isChannelAudible,
  playbackStore,
  seekPlayback,
  setChannelGain,
  setLoopRegion,
  toggleChannelMute,
  toggleChannelSolo,
} from "../stores/playbackStore";
import { sessionStore } from "../stores/sessionStore";
import { renderTapeDragPreview } from "../utils/dragPreview";
import { formatDb, formatElapsed } from "../utils/format";
import { sortStemsForDisplay, stemName, stemNameLower } from "../utils/stems";
import type { DecodedPeaks } from "../utils/waveform";
import { Fader } from "../components/primitives/Fader";
import { TapeLabel } from "../components/primitives/TapeLabel";
import { tip } from "../components/primitives/Tooltip";
import { MeterBar } from "../components/MeterBar";
import {
  LoopWash,
  PlayheadLine,
  TimeRuler,
  useWellSeekHandlers,
  WaveformCanvas,
} from "../components/WaveformWell";
import "./StemsView.css";

interface StemChannelStripProps {
  stem: StemInfo;
  index: number;
  solo: boolean;
  muted: boolean;
  gainDb: number;
  revealing: boolean;
  onDragOut: (stemId: StemId) => void;
}

const StemChannelStrip = memo(function StemChannelStrip({
  stem,
  index,
  solo,
  muted,
  gainDb,
  revealing,
  onDragOut,
}: StemChannelStripProps) {
  const lower = stemNameLower(stem.id);
  const heardText = useMemo(() => stem.contains.map((d) => d.label).join(", "), [stem.contains]);
  const shortcutNum = index < 6 ? String(index + 1) : undefined;
  const faderTip = useMemo(
    () =>
      tip(`${stemName(stem.id)} volume`, {
        detail: "Adjust loudness for listening and mixdown export. Double-click to reset to 0 dB.",
        placement: "top",
      }),
    [stem.id],
  );
  const handleStartDragOut = useCallback(() => {
    onDragOut(stem.id);
  }, [onDragOut, stem.id]);
  const handleGainChange = useCallback(
    (db: number) => {
      void setChannelGain(stem.id, db);
    },
    [stem.id],
  );
  const handleSolo = useCallback(() => {
    void toggleChannelSolo(stem.id);
  }, [stem.id]);
  const handleMute = useCallback(() => {
    void toggleChannelMute(stem.id);
  }, [stem.id]);

  return (
    <div className={revealing ? `ch ch--reveal stem-${stem.id}` : `ch stem-${stem.id}`}>
      <TapeLabel stemId={stem.id} draggableHandle onStartDragOut={handleStartDragOut} />

      <div
        className="ch__heard"
        {...(heardText
          ? tip("Heard in this stem", { detail: heardText, placement: "right" })
          : {})}
      >
        {heardText}
      </div>

      <div className="ch__ctl">
        <button
          type="button"
          className={solo ? "ch__key ch__key--solo-on" : "ch__key"}
          aria-label={`Solo ${lower}`}
          aria-pressed={solo}
          onClick={handleSolo}
          {...tip(solo ? `Stop soloing ${lower}` : `Solo ${lower}`, {
            detail: solo
              ? "Return to hearing all unmuted stems."
              : "Listen to only this stem. You can solo more than one at once.",
            shortcut: shortcutNum,
            placement: "top",
          })}
        >
          S
        </button>
        <button
          type="button"
          className={muted ? "ch__key ch__key--mute-on" : "ch__key"}
          aria-label={`Mute ${lower}`}
          aria-pressed={muted}
          onClick={handleMute}
          {...tip(muted ? `Unmute ${lower}` : `Mute ${lower}`, {
            detail: muted
              ? "Bring this stem back into the mix."
              : "Silence this stem while the others keep playing.",
            shortcut: shortcutNum ? `Shift+${shortcutNum}` : undefined,
            placement: "top",
          })}
        >
          M
        </button>
        <Fader
          className="ch__fader"
          valueDb={gainDb}
          label={`${stemName(stem.id)} level`}
          onChange={handleGainChange}
          tooltip={faderTip}
        />
        <span
          className="ch__db"
          {...tip(`Level: ${formatDb(gainDb)} dB`, {
            detail: "0.0 dB is the original volume. Double-click the slider to reset.",
            placement: "top",
          })}
        >
          {formatDb(gainDb)}
        </span>
      </div>

      <MeterBar stemId={stem.id} />
    </div>
  );
});

interface StemLaneProps {
  stemId: StemId;
  peaks: Peaks | DecodedPeaks | undefined;
  audible: boolean;
  durationSec: number;
  revealing: boolean;
}

const StemLane = memo(function StemLane({
  stemId,
  peaks,
  audible,
  durationSec,
  revealing,
}: StemLaneProps) {
  return (
    <div className={revealing ? "lane lane--reveal" : "lane"}>
      <WaveformCanvas source={stemId} peaks={peaks} audible={audible} durationSec={durationSec} />
    </div>
  );
});

interface StemsViewProps {
  session: Session;
}

export const StemsView = memo(function StemsView({ session }: StemsViewProps) {
  const { peaks, revealing } = useShallowStore(sessionStore, (s) => ({
    peaks: s.peaks,
    revealing: s.revealing,
  }));
  const { channels, playbackDurationSec, loop } = useShallowStore(playbackStore, (s) => ({
    channels: s.playback.channels,
    playbackDurationSec: s.playback.durationSec,
    loop: s.playback.loop,
  }));
  const deskRef = useRef<HTMLElement | null>(null);

  const orderedStems = useMemo(() => sortStemsForDisplay(session.stems), [session.stems]);
  const presentStems = useMemo(() => orderedStems.filter((s) => s.present), [orderedStems]);
  const absentStems = useMemo(() => orderedStems.filter((s) => !s.present), [orderedStems]);

  const gridRowsTemplate = useMemo(() => {
    const parts = ["var(--ruler-height)"];
    if (presentStems.length > 0) {
      parts.push(`repeat(${presentStems.length}, minmax(0, 1fr))`);
    }
    if (absentStems.length > 0) {
      parts.push(`repeat(${absentStems.length}, 40px)`);
    }
    return parts.join(" ");
  }, [presentStems.length, absentStems.length]);

  useEffect(() => {
    const desk = deskRef.current;
    if (!desk) return;
    desk.style.setProperty("--desk-rows", gridRowsTemplate);

    if (revealing) {
      const stripRows = desk.querySelectorAll<HTMLElement>(".ch--reveal");
      stripRows.forEach((row, i) => {
        row.style.setProperty("--reveal-delay", `${i * 40}ms`);
      });
      const lanes = desk.querySelectorAll<HTMLElement>(".lane--reveal");
      const midIndex = (lanes.length - 1) / 2;
      lanes.forEach((lane, i) => {
        lane.style.setProperty("--reveal-delay", `${i * 40}ms`);
        const offsetPx = Math.round((midIndex - i) * 36);
        lane.style.setProperty("--reveal-from-y", `${offsetPx}px`);
      });
    }
  }, [gridRowsTemplate, revealing, orderedStems.length]);

  const durationSec = playbackDurationSec || session.track.durationSec;
  const handleSeek = useCallback((pos: number) => {
    void seekPlayback(pos);
  }, []);
  const handleChangeLoop = useCallback((nextLoop: { startSec: number; endSec: number } | null) => {
    void setLoopRegion(nextLoop);
  }, []);
  const seekHandlers = useWellSeekHandlers(durationSec, true, handleSeek);

  const handleDragOut = useCallback(
    async (stemId: StemId) => {
      try {
        const preview = renderTapeDragPreview(stemId);
        const paths = await invoke("prepare_drag", {
          sessionId: session.id,
          stemIds: [stemId],
        });
        await startNativeStemDrag(paths, preview);
      } catch {
        // ignore cancelled drag
      }
    },
    [session.id],
  );
  const triggerDragOut = useCallback(
    (stemId: StemId) => {
      void handleDragOut(stemId);
    },
    [handleDragOut],
  );

  const elapsed = session.split?.elapsedSec ?? 0;
  const stemCountLabel = `${session.stems.length} ${session.stems.length === 1 ? "stem" : "stems"}`;
  const stripHeader =
    elapsed > 0 ? `${stemCountLabel}, split in ${formatElapsed(elapsed)}` : stemCountLabel;

  return (
    <main ref={deskRef} className="stems-desk">
      <aside className="stems-strip" aria-label="Stem mixer">
        <div className="stems-strip__head">{stripHeader}</div>

        {presentStems.map((stem, idx) => {
          const ch = channels.find((c) => c.id === stem.id);
          return (
            <StemChannelStrip
              key={stem.id}
              stem={stem}
              index={idx}
              solo={ch?.solo ?? false}
              muted={ch?.muted ?? false}
              gainDb={ch?.gainDb ?? 0}
              revealing={revealing}
              onDragOut={triggerDragOut}
            />
          );
        })}

        {absentStems.map((stem) => (
          <div
            key={stem.id}
            className={`ch ch--absent stem-${stem.id}`}
            {...tip(`Nothing heard in ${stemNameLower(stem.id)}`, {
              detail: "This stem is essentially silent in this song.",
              placement: "right",
            })}
          >
            <TapeLabel stemId={stem.id} size="absent" />
            <div className="ch__heard">None heard in this song</div>
          </div>
        ))}
      </aside>

      <section
        className="well stems-well well--seekable"
        aria-label="Stem waveforms"
        {...seekHandlers}
      >
        <TimeRuler
          durationSec={durationSec}
          loop={loop}
          onSeek={handleSeek}
          onChangeLoop={handleChangeLoop}
        />
        <LoopWash loop={loop} durationSec={durationSec} />

        {presentStems.map((stem) => {
          const ch = channels.find((c) => c.id === stem.id);
          const audible = isChannelAudible(ch, channels);
          return (
            <StemLane
              key={stem.id}
              stemId={stem.id}
              peaks={peaks[stem.id]}
              audible={audible}
              durationSec={durationSec}
              revealing={revealing}
            />
          );
        })}

        {absentStems.map((stem) => (
          <div key={stem.id} className="lane lane--flat" aria-hidden="true" />
        ))}

        <PlayheadLine durationSec={durationSec} />
      </section>
    </main>
  );
});
