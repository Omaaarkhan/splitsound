import { dismissNotice, uiStore, type Notice } from "../stores/uiStore";
import "./NoticeBar.css";

interface NoticeBarProps {
  notice: Notice;
  aboveTransport: boolean;
}

export function NoticeBar({ notice, aboveTransport }: NoticeBarProps) {
  const classes = [
    "notice",
    aboveTransport ? "notice--above-transport" : "",
    notice.tone === "error" ? "notice--error" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const dismissLabel = notice.dismissLabel === undefined ? "Dismiss" : notice.dismissLabel;

  return (
    <div
      className={classes}
      role={notice.tone === "error" ? "alert" : "status"}
      aria-live={notice.tone === "error" ? "assertive" : "polite"}
    >
      <p className="notice__text">{notice.text}</p>
      <div className="notice__actions">
        {notice.actionLabel && notice.onAction ? (
          <button
            type="button"
            className="console-key console-key--compact"
            onClick={() => {
              const prevId = notice.id;
              notice.onAction?.();
              if (uiStore.getState().notice?.id === prevId) {
                dismissNotice();
              }
            }}
          >
            {notice.actionLabel}
          </button>
        ) : null}
        {dismissLabel ? (
          <button
            type="button"
            className="console-key console-key--compact"
            onClick={dismissNotice}
          >
            {dismissLabel}
          </button>
        ) : null}
      </div>
    </div>
  );
}
