import { useCallback, useEffect } from "react";
import { invoke } from "./ipc/client";
import { pickAudioFiles, subscribeWindowFileDrop } from "./ipc/platform";
import { useStore } from "./stores/createStore";
import {
  bootstrapApp,
  goHome,
  importAudioPaths,
  openSongById,
  requestModelDownload,
  requestStartSplit,
} from "./stores/index";
import { clearAllLibrary, deleteLibrarySession } from "./stores/libraryStore";
import { markUserCancelledDownloads } from "./stores/modelsStore";
import {
  seekPlayback,
  seekPlaybackBy,
  toggleChannelMute,
  toggleChannelSolo,
  togglePlayback,
} from "./stores/playbackStore";
import { sessionStore, setSplitMode } from "./stores/sessionStore";
import { settingsStore } from "./stores/settingsStore";
import { closePanel, openPanel, setDraggingFiles, showNotice, uiStore } from "./stores/uiStore";
import { COPY, formatIpcError } from "./utils/errors";
import { sortStemsForDisplay } from "./utils/stems";
import { TopBar } from "./components/TopBar";
import { TransportBar } from "./components/TransportBar";
import { NoticeBar } from "./components/NoticeBar";
import { TooltipLayer } from "./components/primitives/Tooltip";
import { HomeView } from "./screens/HomeView";
import { PreSplitView } from "./screens/PreSplitView";
import { SplittingView } from "./screens/SplittingView";
import { StemsView } from "./screens/StemsView";
import { ExportPanel } from "./panels/ExportPanel";
import { SettingsPanel } from "./panels/SettingsPanel";
import { ConsentPanel } from "./panels/ConsentPanel";
import { ConfirmPanel } from "./panels/ConfirmPanel";
import { LicensesPanel } from "./panels/LicensesPanel";
import "./App.css";

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  return false;
}

function isSliderOrInteractiveRole(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const role = target.getAttribute("role");
  return role === "slider" || role === "listbox" || role === "option";
}

export default function App() {
  const { current: session, splitOptions } = useStore(sessionStore);
  const { appInfo } = useStore(settingsStore);
  const { panel, notice, draggingFiles } = useStore(uiStore);

  useEffect(() => {
    let disposed = false;
    let unlistenEvents: (() => void) | null = null;
    let unlistenDrop: (() => void) | null = null;

    void bootstrapApp().then((fn) => {
      if (disposed) fn();
      else unlistenEvents = fn;
    });

    void subscribeWindowFileDrop({
      onDragEnter: () => setDraggingFiles(true),
      onDragLeave: () => setDraggingFiles(false),
      onDrop: (paths) => {
        void importAudioPaths(paths);
      },
    }).then((fn) => {
      if (disposed) fn();
      else unlistenDrop = fn;
    });

    return () => {
      disposed = true;
      unlistenEvents?.();
      unlistenDrop?.();
    };
  }, []);

  const handleChooseFile = useCallback(async () => {
    const paths = await pickAudioFiles();
    if (paths.length > 0) {
      await importAudioPaths(paths);
    }
  }, []);

  const handleCancelSplit = useCallback(async () => {
    if (!session) return;
    markUserCancelledDownloads();
    try {
      await invoke("cancel_split", { sessionId: session.id });
      showNotice(COPY.splitCancelled);
    } catch (err) {
      showNotice(formatIpcError(err), { tone: "error" });
    }
  }, [session]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      const keyLower = e.key.toLowerCase();

      if (mod && keyLower === "o") {
        e.preventDefault();
        void handleChooseFile();
        return;
      }

      if (mod && keyLower === ",") {
        e.preventDefault();
        openPanel({ type: "settings" });
        return;
      }

      if (mod && keyLower === "e") {
        const cur = sessionStore.getState().current;
        if (cur?.status === "ready") {
          e.preventDefault();
          openPanel({ type: "export" });
        }
        return;
      }

      // Skip playback/mixer shortcuts while a panel is open or typing in an input
      if (uiStore.getState().panel.type !== "none") return;
      if (isEditableTarget(e.target)) return;

      const cur = sessionStore.getState().current;
      if (!cur) return;

      if (e.code === "Space") {
        // Allow focused buttons/checkboxes to keep normal Space activation
        if (
          e.target instanceof HTMLElement &&
          (e.target.tagName === "BUTTON" || e.target.getAttribute("role") === "radio")
        ) {
          return;
        }
        e.preventDefault();
        void togglePlayback();
        return;
      }

      if (isSliderOrInteractiveRole(e.target)) return;

      if (e.key === "ArrowLeft") {
        e.preventDefault();
        void seekPlaybackBy(-5);
        return;
      }

      if (e.key === "ArrowRight") {
        e.preventDefault();
        void seekPlaybackBy(5);
        return;
      }

      if (e.key === "Home") {
        e.preventDefault();
        void seekPlayback(0);
        return;
      }

      if (cur.status === "ready" && !mod && !e.altKey) {
        const match = /^Digit([1-6])$/.exec(e.code);
        if (match) {
          const idx = Number(match[1]) - 1;
          const ordered = sortStemsForDisplay(cur.stems).filter((s) => s.present);
          const stem = ordered[idx];
          if (stem) {
            e.preventDefault();
            if (e.shiftKey) {
              void toggleChannelMute(stem.id);
            } else {
              void toggleChannelSolo(stem.id);
            }
          }
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleChooseFile]);

  const showTransport = session !== null;

  return (
    <div className={showTransport ? "app-shell app-shell--with-transport" : "app-shell"}>
      <TopBar
        platform={appInfo.platform}
        session={session}
        onGoHome={() => void goHome()}
        onOpenSettings={() => openPanel({ type: "settings" })}
      />

      {session === null ? (
        <HomeView
          draggingFiles={draggingFiles}
          onChooseFile={() => void handleChooseFile()}
          onOpenSession={(id) => void openSongById(id)}
          onRequestRemove={(sessionId, title) =>
            openPanel({ type: "confirm-remove", sessionId, title })
          }
        />
      ) : session.status === "ready" ? (
        <StemsView session={session} />
      ) : session.status === "splitting" || session.status === "queued" ? (
        <SplittingView session={session} onCancel={() => void handleCancelSplit()} />
      ) : (
        <PreSplitView
          session={session}
          onStartSplit={() => void requestStartSplit(session.id, splitOptions)}
        />
      )}

      {showTransport ? (
        <TransportBar session={session} onOpenExport={() => openPanel({ type: "export" })} />
      ) : null}

      {notice ? <NoticeBar notice={notice} aboveTransport={showTransport} /> : null}

      {panel.type === "export" && session?.status === "ready" ? (
        <ExportPanel session={session} onClose={closePanel} />
      ) : null}

      {panel.type === "settings" ? (
        <SettingsPanel
          onClose={closePanel}
          onDownloadModel={(model) => void requestModelDownload(model)}
          onRequestClearLibrary={() => openPanel({ type: "confirm-clear" })}
          onOpenLicenses={() => openPanel({ type: "licenses" })}
        />
      ) : null}

      {panel.type === "consent" ? (
        <ConsentPanel
          request={panel.request}
          onCancel={() => {
            if (panel.request.kind === "model") {
              openPanel({ type: "settings" });
            } else {
              closePanel();
            }
          }}
          onAccept={() => {
            const req = panel.request;
            if (req.kind === "split") {
              closePanel();
              void requestStartSplit(req.sessionId, req.options, true);
            } else {
              openPanel({ type: "settings" });
              void requestModelDownload(req.model, true);
            }
          }}
          onSplitWithout={
            panel.request.kind === "split"
              ? () => {
                  const req = panel.request;
                  if (req.kind !== "split") return;
                  closePanel();
                  setSplitMode("band");
                  void requestStartSplit(req.sessionId, { ...req.options, mode: "band" }, false);
                }
              : undefined
          }
        />
      ) : null}

      {panel.type === "confirm-remove" ? (
        <ConfirmPanel
          title="Remove song"
          body={COPY.confirmRemove}
          confirmLabel="Remove"
          onCancel={closePanel}
          onConfirm={() => {
            const id = panel.sessionId;
            closePanel();
            void deleteLibrarySession(id).catch((err) => {
              showNotice(formatIpcError(err), { tone: "error" });
            });
          }}
        />
      ) : null}

      {panel.type === "confirm-clear" ? (
        <ConfirmPanel
          title="Clear library"
          body={COPY.confirmClearLibrary}
          confirmLabel="Clear library"
          onCancel={() => openPanel({ type: "settings" })}
          onConfirm={() => {
            openPanel({ type: "settings" });
            void clearAllLibrary().catch((err) => {
              showNotice(formatIpcError(err), { tone: "error" });
            });
          }}
        />
      ) : null}

      {panel.type === "licenses" ? (
        <LicensesPanel onClose={() => openPanel({ type: "settings" })} />
      ) : null}

      <TooltipLayer resetKey={`${session?.id ?? ""}:${session?.status ?? ""}:${panel.type}`} />
    </div>
  );
}
