import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { AppInfo } from "../../ipc/contract";
import "./Tooltip.css";

export type TooltipPlacement = "top" | "bottom" | "left" | "right";

export interface TooltipOptions {
  detail?: string;
  shortcut?: string;
  placement?: TooltipPlacement;
}

export type TooltipAttrs = {
  "data-tip"?: string;
  "data-tip-detail"?: string;
  "data-tip-key"?: string;
  "data-tip-place"?: TooltipPlacement;
};

const TOOLTIP_ID = "splitsound-tooltip";
const HOVER_DELAY_MS = 350;
const WARM_DELAY_MS = 70;
const WARM_WINDOW_MS = 300;
const FOCUS_DELAY_MS = 120;
const GAP = 8;
const MARGIN = 8;

export function tip(title: string | null | undefined, options?: TooltipOptions): TooltipAttrs {
  if (!title) return {};
  const attrs: TooltipAttrs = { "data-tip": title };
  if (options?.detail) attrs["data-tip-detail"] = options.detail;
  if (options?.shortcut) attrs["data-tip-key"] = options.shortcut;
  if (options?.placement) attrs["data-tip-place"] = options.placement;
  return attrs;
}

export function formatModShortcut(platform: AppInfo["platform"], key: string): string {
  return platform === "macos" ? `⌘${key}` : `Ctrl+${key}`;
}

interface ActiveTooltip {
  anchor: HTMLElement;
  title: string;
  detail: string;
  shortcut: string;
  placement: TooltipPlacement;
  viaFocus: boolean;
}

function parsePlacement(raw: string | undefined): TooltipPlacement {
  if (raw === "bottom" || raw === "left" || raw === "right") return raw;
  return "top";
}

function readTooltip(anchor: HTMLElement, viaFocus: boolean): ActiveTooltip | null {
  const title = anchor.dataset.tip;
  if (!title) return null;
  return {
    anchor,
    title,
    detail: anchor.dataset.tipDetail ?? "",
    shortcut: anchor.dataset.tipKey ?? "",
    placement: parsePlacement(anchor.dataset.tipPlace),
    viaFocus,
  };
}

function findAnchor(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof Element)) return null;
  const el = target.closest<HTMLElement>("[data-tip]");
  return el && el.dataset.tip ? el : null;
}

function TooltipLayerInner() {
  const [active, setActive] = useState<ActiveTooltip | null>(null);
  const tipRef = useRef<HTMLDivElement | null>(null);
  const activeRef = useRef<ActiveTooltip | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let suppressedAnchor: HTMLElement | null = null;
    let lastClosedAt = 0;
    let described: { el: HTMLElement; prev: string | null } | null = null;

    const clearTimer = () => {
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    };

    const restoreDescribed = () => {
      if (!described) return;
      const { el, prev } = described;
      described = null;
      if (prev === null) {
        el.removeAttribute("aria-describedby");
      } else {
        el.setAttribute("aria-describedby", prev);
      }
    };

    const show = (anchor: HTMLElement, viaFocus: boolean) => {
      const data = readTooltip(anchor, viaFocus);
      if (!data) return;
      restoreDescribed();
      if (viaFocus) {
        const prev = anchor.getAttribute("aria-describedby");
        described = { el: anchor, prev };
        const parts = prev ? prev.split(/\s+/).filter(Boolean) : [];
        if (!parts.includes(TOOLTIP_ID)) {
          parts.push(TOOLTIP_ID);
          anchor.setAttribute("aria-describedby", parts.join(" "));
        }
      }
      activeRef.current = data;
      setActive(data);
    };

    const hide = (recordClose: boolean) => {
      clearTimer();
      restoreDescribed();
      if (activeRef.current) {
        if (recordClose) {
          lastClosedAt = Date.now();
        }
        activeRef.current = null;
        setActive(null);
      }
    };

    const onPointerOver = (e: PointerEvent) => {
      const anchor = findAnchor(e.target);
      if (suppressedAnchor && anchor !== suppressedAnchor) {
        suppressedAnchor = null;
      }
      if (!anchor) {
        hide(true);
        return;
      }
      if (anchor === suppressedAnchor) {
        return;
      }
      if (activeRef.current?.anchor === anchor && !activeRef.current.viaFocus) {
        return;
      }
      clearTimer();
      const isWarm = activeRef.current !== null || Date.now() - lastClosedAt < WARM_WINDOW_MS;
      const delay = isWarm ? WARM_DELAY_MS : HOVER_DELAY_MS;
      timer = setTimeout(() => {
        timer = null;
        if (anchor.isConnected) {
          show(anchor, false);
        }
      }, delay);
    };

    const onPointerLeaveDoc = () => {
      suppressedAnchor = null;
      hide(false);
    };

    const onPointerDown = (e: PointerEvent) => {
      const anchor = findAnchor(e.target);
      if (anchor) {
        suppressedAnchor = anchor;
      }
      hide(false);
    };

    const onFocusIn = (e: FocusEvent) => {
      const anchor = findAnchor(e.target);
      if (!anchor) return;
      let focusVisible: boolean;
      try {
        focusVisible = anchor.matches(":focus-visible");
      } catch {
        focusVisible = false;
      }
      if (!focusVisible) return;
      clearTimer();
      timer = setTimeout(() => {
        timer = null;
        if (anchor.isConnected && document.activeElement === anchor) {
          show(anchor, true);
        }
      }, FOCUS_DELAY_MS);
    };

    const onFocusOut = (e: FocusEvent) => {
      const anchor = findAnchor(e.target);
      if (!anchor || activeRef.current?.anchor === anchor) {
        hide(false);
      }
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        hide(false);
      }
    };

    const onViewportChange = () => {
      hide(false);
    };

    document.addEventListener("pointerover", onPointerOver);
    document.documentElement.addEventListener("pointerleave", onPointerLeaveDoc);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("blur", onViewportChange);
    window.addEventListener("resize", onViewportChange);
    window.addEventListener("scroll", onViewportChange, true);

    return () => {
      clearTimer();
      restoreDescribed();
      activeRef.current = null;
      document.removeEventListener("pointerover", onPointerOver);
      document.documentElement.removeEventListener("pointerleave", onPointerLeaveDoc);
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", onFocusOut);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("blur", onViewportChange);
      window.removeEventListener("resize", onViewportChange);
      window.removeEventListener("scroll", onViewportChange, true);
    };
  }, []);

  useLayoutEffect(() => {
    const tipEl = tipRef.current;
    if (!active || !tipEl || !active.anchor.isConnected) return;

    const anchorRect = active.anchor.getBoundingClientRect();
    const tipRect = tipEl.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let place = active.placement;
    if (
      place === "top" &&
      anchorRect.top - tipRect.height - GAP < MARGIN &&
      anchorRect.bottom + GAP + tipRect.height <= vh - MARGIN
    ) {
      place = "bottom";
    } else if (
      place === "bottom" &&
      anchorRect.bottom + GAP + tipRect.height > vh - MARGIN &&
      anchorRect.top - tipRect.height - GAP >= MARGIN
    ) {
      place = "top";
    } else if (
      place === "left" &&
      anchorRect.left - tipRect.width - GAP < MARGIN &&
      anchorRect.right + GAP + tipRect.width <= vw - MARGIN
    ) {
      place = "right";
    } else if (
      place === "right" &&
      anchorRect.right + GAP + tipRect.width > vw - MARGIN &&
      anchorRect.left - tipRect.width - GAP >= MARGIN
    ) {
      place = "left";
    }

    let x: number;
    let y: number;
    switch (place) {
      case "bottom":
        x = anchorRect.left + (anchorRect.width - tipRect.width) / 2;
        y = anchorRect.bottom + GAP;
        break;
      case "left":
        x = anchorRect.left - tipRect.width - GAP;
        y = anchorRect.top + (anchorRect.height - tipRect.height) / 2;
        break;
      case "right":
        x = anchorRect.right + GAP;
        y = anchorRect.top + (anchorRect.height - tipRect.height) / 2;
        break;
      case "top":
      default:
        x = anchorRect.left + (anchorRect.width - tipRect.width) / 2;
        y = anchorRect.top - tipRect.height - GAP;
        break;
    }

    const maxLeft = Math.max(MARGIN, vw - tipRect.width - MARGIN);
    const maxTop = Math.max(MARGIN, vh - tipRect.height - MARGIN);
    const left = Math.round(Math.max(MARGIN, Math.min(maxLeft, x)));
    const top = Math.round(Math.max(MARGIN, Math.min(maxTop, y)));

    tipEl.style.setProperty("--tip-left", `${left}px`);
    tipEl.style.setProperty("--tip-top", `${top}px`);
    tipEl.dataset.visible = "true";
  }, [active]);

  if (!active || typeof document === "undefined") {
    return null;
  }

  return createPortal(
    <div ref={tipRef} id={TOOLTIP_ID} role="tooltip" className="tooltip" data-visible="false">
      <div className="tooltip__head">
        <span className="tooltip__title">{active.title}</span>
        {active.shortcut ? <kbd className="tooltip__key">{active.shortcut}</kbd> : null}
      </div>
      {active.detail ? <p className="tooltip__detail">{active.detail}</p> : null}
    </div>,
    document.body,
  );
}

export function TooltipLayer({ resetKey }: { resetKey: string }) {
  return <TooltipLayerInner key={resetKey} />;
}
