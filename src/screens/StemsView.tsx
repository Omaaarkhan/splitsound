import { useEffect, useMemo, useRef } from "react";
import type { Session, StemId } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { startNativeStemDrag } from "../ipc/platform";
import { useStore } from "../stores/createStore";
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
import { Fader } from "../components/primitives/Fader";
import { TapeLabel } from "../components/primitives/TapeLabel";
import { MeterBar } from "../components/MeterBar";
import {
  LoopWash,
  PlayheadLine,
  TimeRuler,
  useWellSeekHandlers,
  WaveformCanvas,
} from "../components/WaveformWell";
import "./StemsView.css";

interface StemsViewProps {
  session: Session;
}

export function StemsView({ session }: StemsViewProps) {
  const { peaks, revealing } = useStore(sessionStore);
  const { playback } = useStore(playbackStore);
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

  const durationSec = playback.durationSec || session.track.durationSec;
  const seekHandlers = useWellSeekHandlers(durationSec, true, (pos) => {
    void seekPlayback(pos);
  });

  const handleDragOut = async (stemId: StemId) => {
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
  };

  const elapsed = session.split?.elapsedSec ?? 0;
  const stemCountLabel = `${session.stems.length} ${session.stems.length === 1 ? "stem" : "stems"}`;
  const stripHeader =
    elapsed > 0 ? `${stemCountLabel}, split in ${formatElapsed(elapsed)}` : stemCountLabel;

  return (
    <main ref={deskRef} className="stems-desk">
      <aside className="stems-strip" aria-label="Stem mixer">
        <div className="stems-strip__head">{stripHeader}</div>

        {presentStems.map((stem) => {
          const ch = playback.channels.find((c) => c.id === stem.id);
          const solo = ch?.solo ?? false;
          const muted = ch?.muted ?? false;
          const gainDb = ch?.gainDb ?? 0;
          const lower = stemNameLower(stem.id);
          const heardText = stem.contains.map((d) => d.label).join(", ");

          return (
            <div
              key={stem.id}
              className={revealing ? `ch ch--reveal stem-${stem.id}` : `ch stem-${stem.id}`}
            >
              <TapeLabel
                stemId={stem.id}
                draggableHandle
                onStartDragOut={() => void handleDragOut(stem.id)}
              />

              <div className="ch__heard" title={heardText || undefined}>
                {heardText}
              </div>

              <div className="ch__ctl">
                <button
                  type="button"
                  className={solo ? "ch__key ch__key--solo-on" : "ch__key"}
                  aria-label={`Solo ${lower}`}
                  aria-pressed={solo}
                  onClick={() => void toggleChannelSolo(stem.id)}
                >
                  S
                </button>
                <button
                  type="button"
                  className={muted ? "ch__key ch__key--mute-on" : "ch__key"}
                  aria-label={`Mute ${lower}`}
                  aria-pressed={muted}
                  onClick={() => void toggleChannelMute(stem.id)}
                >
                  M
                </button>
                <Fader
                  className="ch__fader"
                  valueDb={gainDb}
                  label={`${stemName(stem.id)} level`}
                  onChange={(db) => void setChannelGain(stem.id, db)}
                />
                <span className="ch__db">{formatDb(gainDb)}</span>
              </div>

              <MeterBar stemId={stem.id} />
            </div>
          );
        })}

        {absentStems.map((stem) => (
          <div key={stem.id} className={`ch ch--absent stem-${stem.id}`}>
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
          loop={playback.loop}
          onSeek={(pos) => void seekPlayback(pos)}
          onChangeLoop={(loop) => void setLoopRegion(loop)}
        />
        <LoopWash loop={playback.loop} durationSec={durationSec} />

        {presentStems.map((stem) => {
          const ch = playback.channels.find((c) => c.id === stem.id);
          const audible = isChannelAudible(ch, playback.channels);
          return (
            <div key={stem.id} className={revealing ? "lane lane--reveal" : "lane"}>
              <WaveformCanvas
                source={stem.id}
                peaks={peaks[stem.id]}
                audible={audible}
                durationSec={durationSec}
                playing={playback.playing}
              />
            </div>
          );
        })}

        {absentStems.map((stem) => (
          <div key={stem.id} className="lane lane--flat" aria-hidden="true" />
        ))}

        <PlayheadLine durationSec={durationSec} playing={playback.playing} />
      </section>
    </main>
  );
}
