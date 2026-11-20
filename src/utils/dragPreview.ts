import type { StemId } from "../ipc/contract";
import { STEM_CSS_VAR, stemName } from "./stems";
import { resolveCssColor } from "./waveform";

const FALLBACK_ICON_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/**
 * Renders a stem's coloured artist-tape strip (spliced 45-degree end, marker writing)
 * to an offscreen 2x canvas and returns a PNG data URL for the native drag preview.
 */
export function renderTapeDragPreview(stemId: StemId): string {
  if (typeof document === "undefined") {
    return FALLBACK_ICON_DATA_URL;
  }

  const label = stemName(stemId);
  const tapeColor = resolveCssColor(STEM_CSS_VAR[stemId], "#f2eddd");
  const markerColor = resolveCssColor("--marker", "#131822");
  const edgeColor = resolveCssColor("--well-deep", "#0d151b");
  const handFont = resolveCssColor("--font-hand", '"Just Another Hand", cursive');

  const dpr = 2;
  const stripHeight = 30;
  const cut = 15;
  const padLeft = 12;
  const padRight = 26;
  const fontSpec = `400 27px ${handFont}`;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return FALLBACK_ICON_DATA_URL;
  }

  ctx.font = fontSpec;
  const measured = Math.ceil(ctx.measureText(label).width);
  const width = Math.max(76, padLeft + measured + padRight);
  const height = stripHeight + 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;

  ctx.save();
  ctx.scale(dpr, dpr);

  // Hard 1px bottom edge under the tape
  ctx.fillStyle = edgeColor;
  ctx.beginPath();
  ctx.moveTo(0, 1);
  ctx.lineTo(width, 1);
  ctx.lineTo(width - cut, stripHeight + 1);
  ctx.lineTo(0, stripHeight + 1);
  ctx.closePath();
  ctx.fill();

  // Tape strip body with 45-degree forward-slash cut end
  ctx.fillStyle = tapeColor;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(width, 0);
  ctx.lineTo(width - cut, stripHeight);
  ctx.lineTo(0, stripHeight);
  ctx.closePath();
  ctx.fill();

  // Stem name in the marker handwriting face, optically centred with 3px top offset
  ctx.fillStyle = markerColor;
  ctx.font = fontSpec;
  ctx.textBaseline = "middle";
  ctx.fillText(label, padLeft, stripHeight / 2 + 2);

  ctx.restore();

  try {
    return canvas.toDataURL("image/png");
  } catch {
    return FALLBACK_ICON_DATA_URL;
  }
}
