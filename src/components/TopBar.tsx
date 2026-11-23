import { memo } from "react";
import type { AppInfo, Session } from "../ipc/contract";
import { formatModShortcut, tip } from "./primitives/Tooltip";
import "./TopBar.css";

interface TopBarProps {
  platform: AppInfo["platform"];
  session: Session | null;
  onGoHome: () => void;
  onOpenSettings: () => void;
}

export const TopBar = memo(function TopBar({
  platform,
  session,
  onGoHome,
  onOpenSettings,
}: TopBarProps) {
  const isMac = platform === "macos";
  const dragProps = isMac ? { "data-tauri-drag-region": true } : {};

  return (
    <header className={isMac ? "topbar topbar--macos" : "topbar topbar--standard"} {...dragProps}>
      <div className="topbar__left" {...dragProps}>
        <button
          type="button"
          className="wordmark"
          onClick={onGoHome}
          aria-label="SplitSound, return to library"
          {...(session ? tip("Back to library", { placement: "bottom" }) : {})}
        >
          <span>Split</span>
          <svg
            className="wordmark__cut"
            width="9"
            height="15"
            viewBox="0 0 9 15"
            aria-hidden="true"
          >
            <line x1="1.2" y1="13.8" x2="7.8" y2="1.2" stroke="currentColor" strokeWidth="1.6" />
          </svg>
          <span>Sound</span>
        </button>
      </div>

      <div className="topbar__song" {...dragProps}>
        {session ? (
          <>
            <span className="topbar__title">{session.track.title}</span>
            {session.track.artist ? (
              <span className="topbar__artist">{session.track.artist}</span>
            ) : null}
          </>
        ) : null}
      </div>

      <nav className="topbar__nav" aria-label="Application">
        <button
          type="button"
          className={session ? "topbar__link" : "topbar__link topbar__link--active"}
          onClick={onGoHome}
          {...tip("Library", {
            detail: "View saved songs or drop a new audio file.",
            placement: "bottom",
          })}
        >
          Library
        </button>
        <button
          type="button"
          className="topbar__link"
          onClick={onOpenSettings}
          {...tip("Settings", {
            detail: "Quality defaults, models, library storage, and updates.",
            shortcut: formatModShortcut(platform, ","),
            placement: "bottom",
          })}
        >
          Settings
        </button>
      </nav>
    </header>
  );
});
