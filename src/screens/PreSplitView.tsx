import { memo, useEffect, useMemo, useRef } from "react";
import type { Quality, Session, SplitMode } from "../ipc/contract";
import { STEMS_BY_MODE } from "../ipc/contract";
import { useShallowStore } from "../stores/createStore";
import { retryInstrumentScan } from "../stores/index";
import { modelsStore } from "../stores/modelsStore";
import { playbackStore, seekPlayback, setLoopRegion } from "../stores/playbackStore";
import { sessionStore, setSplitMode, setSplitQuality } from "../stores/sessionStore";
import {
  QUALITY_LABEL,
  computeModelDownloadFraction,
  formatBytes,
  formatDurationEstimate,
  getScanModels,
} from "../utils/format";
import { SPLIT_MODE_LABELS, SPLIT_MODES_ORDER, stemName } from "../utils/stems";
import { TapeSlivers } from "../components/primitives/TapeSlivers";
import { tip } from "../components/primitives/Tooltip";
import {
  LoopWash,
  PlayheadLine,
  TimeRuler,
  useWellSeekHandlers,
  WaveformCanvas,
} from "../components/WaveformWell";
import "./PreSplitView.css";

const MODE_TIPS: Record<SplitMode, { title: string; detail: string }> = {
  vocals: {
    title: "2 stems",
    detail: "Separates vocals from the backing track.",
  },
  band: {
    title: "4 stems",
    detail: "Separates vocals, drums, bass, and everything else.",
  },
  all: {
    title: "6 stems",
    detail: "Adds separate guitar and piano stems alongside vocals, drums, bass, and other.",
  },
};

const QUALITY_TIPS: Record<Quality, { title: string; detail: string }> = {
  fast: {
    title: "Fast quality",
    detail: "Quicker split using the lighter model. Good for previews.",
  },
  best: {
    title: "Best quality",
    detail: "Takes longer using the larger model for cleaner separation and less bleed.",
  },
  studio: {
    title: "Studio quality",
    detail:
      "Isolates vocals first with Mel-Band RoFormer to dramatically cut vocal bleed into other stems.",
  },
};

interface PreSplitViewProps {
  session: Session;
  onStartSplit: () => void;
}

function handleSeekPlayback(pos: number): void {
  void seekPlayback(pos);
}

function handleChangeLoop(loop: { startSec: number; endSec: number } | null): void {
  void setLoopRegion(loop);
}

export const PreSplitView = memo(function PreSplitView({
  session,
  onStartSplit,
}: PreSplitViewProps) {
  const { peaks, splitOptions, estimate } = useShallowStore(sessionStore, (s) => ({
    peaks: s.peaks,
    splitOptions: s.splitOptions,
    estimate: s.estimate,
  }));
  const { playbackDurationSec, loop } = useShallowStore(playbackStore, (s) => ({
    playbackDurationSec: s.playback.durationSec,
    loop: s.playback.loop,
  }));
  const { models, progressById, scanBatchIds } = useShallowStore(modelsStore, (s) => ({
    models: s.models,
    progressById: s.progressById,
    scanBatchIds: s.scanBatchIds,
  }));
  const scanFillRef = useRef<HTMLDivElement | null>(null);

  const durationSec = playbackDurationSec || session.track.durationSec;
  const seekHandlers = useWellSeekHandlers(durationSec, true, handleSeekPlayback);

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
      el.style.setProperty("--progress", (scanDownloadPct / 100).toFixed(4));
    }
  }, [scanDownloadPct, showScanDownloading]);

  const estimateText = estimate
    ? formatDurationEstimate(estimate.estimatedSec, splitOptions.quality)
    : splitOptions.quality === "studio"
      ? "About 12 minutes, cleanest vocals"
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
                      {...tip(det.label, {
                        detail: `Lands in ${stemName(det.family)} · detection strength ${segments} of 5.`,
                        placement: "right",
                      })}
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
                    {...tip(MODE_TIPS[mode].title, {
                      detail: MODE_TIPS[mode].detail,
                      placement: "right",
                    })}
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
              {(["fast", "best", "studio"] as Quality[]).map((q) => {
                const active = splitOptions.quality === q;
                const label = QUALITY_LABEL[q];
                return (
                  <button
                    key={q}
                    type="button"
                    role="radio"
                    aria-checked={active}
                    className={active ? "console-key console-key--selected" : "console-key"}
                    onClick={() => setSplitQuality(q)}
                    {...tip(QUALITY_TIPS[q].title, {
                      detail: QUALITY_TIPS[q].detail,
                      placement: "right",
                    })}
                  >
                    {label}
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
          loop={loop}
          onSeek={handleSeekPlayback}
          onChangeLoop={handleChangeLoop}
        />
        <LoopWash loop={loop} durationSec={durationSec} />
        <div className="lane">
          <WaveformCanvas source="mix" peaks={peaks.mix} audible durationSec={durationSec} />
        </div>
        <PlayheadLine durationSec={durationSec} />
      </section>
    </main>
  );
});
