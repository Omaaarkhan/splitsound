import type { Peaks, SourceId } from "../ipc/contract";

export interface DecodedPeaks {
  buckets: number;
  min: Float32Array;
  max: Float32Array;
}

export function isDecodedPeaks(value: Peaks | DecodedPeaks): value is DecodedPeaks {
  return "min" in value && value.min instanceof Float32Array;
}

export function encodePeaks(min: Float32Array, max: Float32Array): Peaks {
  const buckets = Math.min(min.length, max.length);
  const bytes = new Uint8Array(buckets * 2);
  for (let i = 0; i < buckets; i++) {
    const lo = Math.max(-127, Math.min(127, Math.round(min[i] * 127)));
    const hi = Math.max(-127, Math.min(127, Math.round(max[i] * 127)));
    bytes[i * 2] = lo & 0xff;
    bytes[i * 2 + 1] = hi & 0xff;
  }
  return { buckets, data: uint8ToBase64(bytes) };
}

function decodePeakBytes(bytes: Uint8Array, offset: number, buckets: number): DecodedPeaks {
  const safeBuckets = Math.min(buckets, Math.max(0, Math.floor((bytes.length - offset) / 2)));
  const min = new Float32Array(safeBuckets);
  const max = new Float32Array(safeBuckets);
  for (let i = 0; i < safeBuckets; i++) {
    const rawLo = bytes[offset + i * 2];
    const rawHi = bytes[offset + i * 2 + 1];
    const signedLo = rawLo > 127 ? rawLo - 256 : rawLo;
    const signedHi = rawHi > 127 ? rawHi - 256 : rawHi;
    min[i] = signedLo / 127;
    max[i] = signedHi / 127;
  }
  return { buckets: safeBuckets, min, max };
}

export function decodePeaks(peaks: Peaks): DecodedPeaks {
  const bytes = base64ToUint8(peaks.data);
  return decodePeakBytes(bytes, 0, peaks.buckets);
}

/**
 * Decodes the binary response from `get_session_peaks`:
 * `[count: u8]` followed by `[id_len: u8][id_utf8][buckets: u32 LE][2 * buckets u8]` entries.
 */
export function decodeSessionPeaksBinary(
  buffer: ArrayBuffer,
): Partial<Record<SourceId, DecodedPeaks>> {
  const bytes = new Uint8Array(buffer);
  const out: Partial<Record<SourceId, DecodedPeaks>> = {};
  if (bytes.length < 1) return out;

  const view = new DataView(buffer);
  const decoder = new TextDecoder();
  const count = bytes[0];
  let offset = 1;

  for (let idx = 0; idx < count && offset < bytes.length; idx++) {
    const idLen = bytes[offset++];
    if (offset + idLen + 4 > bytes.length) break;
    const id = decoder.decode(bytes.subarray(offset, offset + idLen)) as SourceId;
    offset += idLen;
    const buckets = view.getUint32(offset, true);
    offset += 4;
    const byteLen = buckets * 2;
    if (offset + byteLen > bytes.length) break;
    out[id] = decodePeakBytes(bytes, offset, buckets);
    offset += byteLen;
  }

  return out;
}

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function base64ToUint8(base64: string): Uint8Array {
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

export interface DrawWaveformOptions {
  peaks: DecodedPeaks;
  color: string;
  /** Optional distinct colour for the unplayed region (e.g. --ink-3 for original mix). */
  unplayedColor?: string;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Fraction 0..1 where played ends and unplayed begins. If >= 1, entire waveform is full strength. */
  playFraction?: number;
  /** True when stem is audible (filled); false when muted/unsoloed (outline only). */
  audible?: boolean;
  /** Vertical inset padding in CSS px. */
  paddingY?: number;
  /** Override played/unplayed alphas for subtle contexts. */
  filledPlayedAlpha?: number;
  filledAheadAlpha?: number;
}

export function traceWaveformPath(
  ctx: CanvasRenderingContext2D,
  peaks: DecodedPeaks,
  x: number,
  y: number,
  width: number,
  height: number,
  paddingY = 7,
): void {
  const mid = y + height / 2;
  const amp = Math.max(2, height / 2 - paddingY);
  const n = peaks.buckets;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const px = x + (i / (n - 1)) * width;
    const py = mid - peaks.max[i] * amp;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  for (let i = n - 1; i >= 0; i--) {
    const px = x + (i / (n - 1)) * width;
    ctx.lineTo(px, mid - peaks.min[i] * amp);
  }
  ctx.closePath();
}

/**
 * Shared waveform path drawer used by stem lanes, the pre-split/splitting mix lane,
 * and the home screen split illustration.
 */
export function drawWaveform(ctx: CanvasRenderingContext2D, opts: DrawWaveformOptions): void {
  const {
    peaks,
    color,
    unplayedColor,
    x,
    y,
    width,
    height,
    playFraction = 1,
    audible = true,
    paddingY = 7,
    filledPlayedAlpha = 1,
    filledAheadAlpha = 0.55,
  } = opts;

  if (peaks.buckets < 2 || width <= 0 || height <= 0) return;

  const clampedPlay = Math.max(0, Math.min(1, playFraction));
  const playX = x + width * clampedPlay;

  for (let pass = 0; pass < 2; pass++) {
    const clipX = pass === 0 ? x : playX;
    const clipW = pass === 0 ? playX - x : x + width - playX;
    if (clipW <= 0) continue;

    ctx.save();
    ctx.beginPath();
    ctx.rect(clipX, y, clipW, height);
    ctx.clip();

    traceWaveformPath(ctx, peaks, x, y, width, height, paddingY);
    const passColor = pass === 1 && unplayedColor ? unplayedColor : color;
    if (!audible) {
      ctx.strokeStyle = passColor;
      ctx.globalAlpha = pass === 0 ? 0.6 : 0.38;
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      ctx.fillStyle = passColor;
      ctx.globalAlpha = pass === 0 ? filledPlayedAlpha : filledAheadAlpha;
      ctx.fill();
    }
    ctx.restore();
  }
}

export interface DrawWaveformRibbonsOptions {
  peaks: DecodedPeaks;
  colors: readonly string[];
  x: number;
  y: number;
  width: number;
  height: number;
  paddingY?: number;
  gapPx?: number;
}

/**
 * Splits the waveform's own height at every x (from its top edge to its bottom edge)
 * into N equal horizontal ribbons, one per stem colour, separated by a crisp gap of well
 * colour so every stem colour remains visible even through quiet passages.
 */
export function drawWaveformRibbons(
  ctx: CanvasRenderingContext2D,
  opts: DrawWaveformRibbonsOptions,
): void {
  const { peaks, colors, x, y, width, height, paddingY = 7, gapPx = 2 } = opts;
  const bandCount = colors.length;
  const n = peaks.buckets;
  if (bandCount === 0 || n < 2 || width <= 0 || height <= 0) return;

  const mid = y + height / 2;
  const amp = Math.max(2, height / 2 - paddingY);
  const topY = new Float32Array(n);
  const ribbonH = new Float32Array(n);
  const effGap = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const t = mid - peaks.max[i] * amp;
    const b = mid - peaks.min[i] * amp;
    const span = Math.max(0, b - t);
    topY[i] = t;
    const gap =
      bandCount > 1 ? Math.min(gapPx, Math.max(0, (span - bandCount * 1.1) / (bandCount - 1))) : 0;
    effGap[i] = gap;
    ribbonH[i] = Math.max(0.6, (span - gap * (bandCount - 1)) / bandCount);
  }

  ctx.save();
  ctx.globalAlpha = 1;
  for (let k = 0; k < bandCount; k++) {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const px = x + (i / (n - 1)) * width;
      const py = topY[i] + k * (ribbonH[i] + effGap[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    for (let i = n - 1; i >= 0; i--) {
      const px = x + (i / (n - 1)) * width;
      const py = topY[i] + k * (ribbonH[i] + effGap[i]) + ribbonH[i];
      ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = colors[k];
    ctx.fill();
  }
  ctx.restore();
}

export function resolveCssColor(varName: string, fallback = "#f3f0e6"): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  return value || fallback;
}
