import type { ModelStatus, SplitOptions } from "../ipc/contract";
import { createStore } from "./createStore";

export type ConsentRequest =
  | {
      kind: "split";
      sessionId: string;
      options: SplitOptions;
      model: ModelStatus;
    }
  | {
      kind: "model";
      model: ModelStatus;
    };

export type ActivePanel =
  | { type: "none" }
  | { type: "export" }
  | { type: "settings" }
  | { type: "consent"; request: ConsentRequest }
  | { type: "confirm-remove"; sessionId: string; title: string }
  | { type: "confirm-clear" }
  | { type: "licenses" };

export interface Notice {
  id: number;
  tone: "info" | "error";
  text: string;
  actionLabel?: string;
  onAction?: () => void;
  /** Label for the dismiss button; pass null to hide dismiss (e.g. while downloading an update). */
  dismissLabel?: string | null;
}

export interface UiState {
  panel: ActivePanel;
  notice: Notice | null;
  draggingFiles: boolean;
}

const initialUiState = (): UiState => ({
  panel: { type: "none" },
  notice: null,
  draggingFiles: false,
});

export const uiStore = createStore<UiState>(initialUiState);

let noticeSeq = 1;
let noticeTimer: ReturnType<typeof setTimeout> | null = null;

function clearNoticeTimer(): void {
  if (noticeTimer !== null) {
    clearTimeout(noticeTimer);
    noticeTimer = null;
  }
}

export function openPanel(panel: ActivePanel): void {
  uiStore.setState((prev) => ({ ...prev, panel }));
}

export function closePanel(): void {
  uiStore.setState((prev) => ({ ...prev, panel: { type: "none" } }));
}

export function showNotice(
  text: string,
  options?: {
    tone?: "info" | "error";
    actionLabel?: string;
    onAction?: () => void;
    dismissLabel?: string | null;
    autoDismissMs?: number | null;
  },
): void {
  clearNoticeTimer();
  const tone = options?.tone ?? "info";
  const id = noticeSeq++;
  uiStore.setState((prev) => ({
    ...prev,
    notice: {
      id,
      tone,
      text,
      actionLabel: options?.actionLabel,
      onAction: options?.onAction,
      dismissLabel: options?.dismissLabel === undefined ? "Dismiss" : options.dismissLabel,
    },
  }));
  const autoMs =
    options?.autoDismissMs === undefined ? (tone === "error" ? 9000 : 7000) : options.autoDismissMs;
  if (autoMs !== null && autoMs > 0) {
    noticeTimer = setTimeout(() => {
      noticeTimer = null;
      uiStore.setState((prev) => (prev.notice?.id === id ? { ...prev, notice: null } : prev));
    }, autoMs);
  }
}

export function dismissNotice(): void {
  clearNoticeTimer();
  uiStore.setState((prev) => (prev.notice ? { ...prev, notice: null } : prev));
}

export function setDraggingFiles(draggingFiles: boolean): void {
  uiStore.setState((prev) =>
    prev.draggingFiles === draggingFiles ? prev : { ...prev, draggingFiles },
  );
}
