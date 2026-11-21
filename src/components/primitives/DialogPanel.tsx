import { useEffect, useId, useRef, type ReactNode } from "react";
import "./DialogPanel.css";

interface DialogPanelProps {
  title: string;
  size?: "narrow" | "default" | "wide";
  onClose: () => void;
  headerAction?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
}

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const selector = [
    "button:not(:disabled)",
    "[href]",
    "input:not(:disabled)",
    "select:not(:disabled)",
    "textarea:not(:disabled)",
    '[tabindex]:not([tabindex="-1"])',
  ].join(",");
  return Array.from(container.querySelectorAll<HTMLElement>(selector)).filter(
    (el) => el.offsetParent !== null,
  );
}

export function DialogPanel({
  title,
  size = "default",
  onClose,
  headerAction,
  footer,
  children,
}: DialogPanelProps) {
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    previousFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const panel = panelRef.current;
    if (panel) {
      panel.focus({ preventScroll: true });
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key === "Tab" && panelRef.current) {
        const focusable = getFocusableElements(panelRef.current);
        if (focusable.length === 0) {
          e.preventDefault();
          return;
        }
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    window.addEventListener("keydown", onKeyDown, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      previousFocusRef.current?.focus();
    };
  }, [onClose]);

  const sizeClass =
    size === "wide"
      ? "dialog-panel dialog-panel--wide"
      : size === "narrow"
        ? "dialog-panel dialog-panel--narrow"
        : "dialog-panel";

  return (
    <div
      className="dialog-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        ref={panelRef}
        className={sizeClass}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
      >
        <div className="dialog-panel__head">
          <h2 id={titleId} className="dialog-panel__title">
            {title}
          </h2>
          {headerAction}
        </div>
        <div className="dialog-panel__body">{children}</div>
        {footer ? <div className="dialog-panel__foot">{footer}</div> : null}
      </div>
    </div>
  );
}
