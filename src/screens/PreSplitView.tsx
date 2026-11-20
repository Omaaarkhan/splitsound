import { useEffect, useMemo, useRef } from "react";
import type { Quality, Session, SplitMode } from "../ipc/contract";
import { STEMS_BY_MODE } from "../ipc/contract";
import { useStore } from "../stores/createStore";
import { retryInstrumentScan } from "../stores/index";
import { modelsStore } from "../stores/modelsStore";
import { playbackStore, seekPlayback, setLoopRegion } from "../stores/playbackStore";
import { sessionStore, setSplitMode, setSplitQuality } from "../stores/sessionStore";
import {
  computeModelDownloadFraction,
  formatBytes,
  formatDurationEstimate,
  getScanModels,
} from "../utils/format";
import { SPLIT_MODE_LABELS, SPLIT_MODES_ORDER, stemName } from "../utils/stems";
import { TapeSlivers } from "../components/primitives/TapeSlivers";
import {
  LoopWash,
  PlayheadLine,
  TimeRuler,
  useWellSeekHandlers,
  WaveformCanvas,
} from "../components/WaveformWell";
import "./PreSplitView.css";

interface PreSplitViewProps {
  session: Session;
  onStartSplit: () => void;
}

export function PreSplitView({ session, onStartSplit }: PreSplitViewProps) {
  const { peaks, splitOptions, estimate } = useStore(sessionStore);
  const { playback } = useStore(playbackStore);
  const { models, progressById, scanBatchIds } = useStore(modelsStore);
  const scanFillRef = useRef<HTMLDivElement | null>(null);

  const durationSec = playback.durationSec || session.track.durationSec;
  const seekHandlers = useWellSeekHandlers(durationSec, true, (pos) => {
    void seekPlayback(pos);
  });

  const restrictedMissing = models.some((m) => m.restricted && !m.installed);
  const topDetections = session.detections.slice(0, 8);
  const sampleRateKhz = (session.track.sampleRate / 1000).toFixed(1).replace(/\.0$/, "");
  const headerText = `${session.track.codec}, ${sampleRateKhz} kHz`;

  const scanBatchSet = useMemo(() => new Set(scanBatchIds), [scanBatchIds]);
  const downloadFraction = computeModelDownloadFraction(models, progressById, scanBatchSet);
  const scanModelDownloading = getScanModels(models).some(
    (m) => m.downloadProgress !== null || m.id in progressById,
  );
  const showScanDownloading =
    (session.analysis === "pending" && downloadFraction !== null) ||
    (session.analysis === "unavailable" && scanModelDownloading && downloadFraction !== null);
  const scanDownloadPct =
    downloadFraction !== null ? Math.max(0, Math.min(100, Math.round(downloadFraction * 100))) : 0;

  useEffect(() => {
    const el = scanFillRef.current;
    if (el) {
      el.style.setProperty("--progress", `${scanDownloadPct}%`);
    }
  }, [scanDownloadPct, showScanDownloading]);

  const estimateText = estimate
    ? formatDurationEstimate(estimate.estimatedSec, splitOptions.quality)
    : splitOptions.quality === "best"
      ? "About 4 minutes, cleaner result"
      : "About 40 seconds";

  return (
    <main className="desk">
      <aside className="presplit-strip" aria-label="Split setup">
        <div className="strip-head">{headerText}</div>

        <div className="presplit-strip__body">
          {session.status === "failed" ? (
            <section className="presplit-error" role="alert">
              <div className="presplit-error__head">
                <span className="presplit-error__lamp" aria-hidden="true" />
                <span>Split didn&apos;t finish</span>
              </div>
              <p className="presplit-error__text">
                {session.error ??
                  "This song couldn't be split right now. Try again or choose another quality."}
              </p>
            </section>
          ) : null}

          <section className="presplit-section" aria-label="Heard in this song">
            <h2 className="presplit-section__title">Heard in this song</h2>
            {showScanDownloading ? (
              <div className="presplit-scan-prog" aria-live="polite">
                <p className="presplit-note">
                  Downloading what&apos;s needed to listen, {scanDownloadPct} %
                </p>
                <div
                  className="presplit-scan-bar"
                  role="progressbar"
                  aria-label="Downloading models to listen"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={scanDownloadPct}
                >
                  <div ref={scanFillRef} className="presplit-scan-fill" />
                </div>
              </div>
            ) : session.analysis === "pending" ? (
              <p className="presplit-note">Listening for instruments</p>
            ) : session.analysis === "unavailable" ? (
              <div className="presplit-scan-retry">
                <p className="presplit-note">
                  Couldn&apos;t listen for instruments. The models it needs aren&apos;t installed,
                  and they couldn&apos;t be downloaded.
                </p>
                <button
                  type="button"
                  className="console-key console-key--compact"
                  onClick={() => void retryInstrumentScan()}
                >
                  Try again
                </button>
              </div>
            ) : session.analysis === "failed" ? (
              <p className="presplit-note">Couldn&apos;t listen for instruments.</p>
            ) : topDetections.length === 0 ? (
              <p className="presplit-note">No specific instruments stood out in the scan.</p>
            ) : (
              <ul className="detections">
                {topDetections.map((det) => {
                  const segments = Math.max(1, Math.min(5, Math.round(det.confidence * 5)));
                  return (
                    <li
                      key={`${det.family}-${det.label}`}
                      className={`detection stem-${det.family}`}
                    >
                      <span className="detection__swatch" aria-hidden="true" />
                      <span className="detection__name">{det.label}</span>
                      <span className="detection__meter" aria-hidden="true">
                        {[1, 2, 3, 4, 5].map((n) => (
                          <span
                            key={n}
                            className={
                              n <= segments ? "detection__seg detection__seg--on" : "detection__seg"
                            }
                          />
                        ))}
                      </span>
                      <span className="sr-only">
                        , lands in {stemName(det.family)}, confidence {segments} of 5
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>

          <section className="presplit-section" aria-label="Split into">
            <h2 className="presplit-section__title">Split into</h2>
            <div className="mode-list" role="radiogroup" aria-label="Split into">
              {SPLIT_MODES_ORDER.map((mode: SplitMode) => {
                const selected = splitOptions.mode === mode;
                const suggested = session.suggestedMode === mode && session.analysis === "done";
                const showRestrictedNote = mode === "all" && restrictedMissing;
                return (
                  <button
                    key={mode}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    className={selected ? "mode-row mode-row--selected" : "mode-row"}
                    onClick={() => setSplitMode(mode)}
                  >
                    <span className="mode-row__lamp" aria-hidden="true" />
                    <span className="mode-row__content">
                      <span className="mode-row__top">
                        <span className="mode-row__title">{SPLIT_MODE_LABELS[mode]}</span>
                        {suggested ? <span className="mode-row__suggested">Suggested</span> : null}
                      </span>
                      <TapeSlivers stemIds={STEMS_BY_MODE[mode]} labelPrefix="Produces" />
                      {showRestrictedNote ? (
                        <span className="mode-row__restricted">
                          Needs a research-licensed model
                        </span>
                      ) : null}
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="presplit-section" aria-label="Quality">
            <h2 className="presplit-section__title">Quality</h2>
            <div className="quality-switch" role="radiogroup" aria-label="Quality">
              {(["fast", "best"] as Quality[]).map((q) => {
                const active = splitOptions.quality === q;
                return (
                  <button
                    key={q}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    className={active ? "console-key console-key--selected" : "console-key"}
                    onClick={() => setSplitQuality(q)}
                  >
                    {q === "fast" ? "Fast" : "Best"}
                  </button>
                );
              })}
            </div>
            <p className="presplit-note">{estimateText}</p>
            {estimate && estimate.downloadBytes > 0 ? (
              <p className="presplit-note">
                Needs a one-time {formatBytes(estimate.downloadBytes)} download
              </p>
            ) : null}
          </section>
        </div>

        <div className="presplit-strip__foot">
          <button type="button" className="primary-btn primary-btn--full" onClick={onStartSplit}>
            {session.status === "failed" ? "Try again" : "Split song"}
          </button>
        </div>
      </aside>

      <section
        className="well well--single well--seekable"
        aria-label="Original song waveform"
        {...seekHandlers}
      >
        <TimeRuler
          durationSec={durationSec}
          loop={playback.loop}
          onSeek={(pos) => void seekPlayback(pos)}
          onChangeLoop={(loop) => void setLoopRegion(loop)}
        />
        <LoopWash loop={playback.loop} durationSec={durationSec} />
        <div className="lane">
          <WaveformCanvas
            source="mix"
            peaks={peaks.mix}
            audible
            durationSec={durationSec}
            playing={playback.playing}
          />
        </div>
        <PlayheadLine durationSec={durationSec} playing={playback.playing} />
      </section>
    </main>
  );
}
