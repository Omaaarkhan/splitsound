import type { Peaks, SourceId } from "../contract";
import { ACTIVITY_BINS } from "../contract";
import { encodePeaks } from "../../utils/waveform";

export function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function hashString(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

type ShapeFn = (t: number, r: () => number, phase: number) => number;

const SHAPES: Record<SourceId, ShapeFn> = {
  vocals: (t, r, p) => {
    const phrase = Math.sin(t * 46 + p) > -0.25 && !(t > 0.44 && t < 0.56) && t > 0.06 && t < 0.95;
    return phrase
      ? 0.28 + 0.5 * Math.abs(Math.sin(t * 190 + r() * 2)) * (0.5 + r() * 0.5)
      : 0.012 * r();
  },
  drums: (t, r, p) => {
    if (t < 0.03) return 0.01;
    const beat = (t * 128 + p * 0.1) % 1;
    const hit = Math.exp(-beat * 7);
    const snare = Math.exp(-(((t * 64 + 0.5 + p * 0.1) % 1) * 9));
    return (
      Math.min(1, 0.08 + 0.78 * hit * (0.7 + 0.3 * r()) + 0.5 * snare) *
      (t > 0.44 && t < 0.5 ? 0.25 : 1)
    );
  },
  bass: (t, r, p) => {
    if (t < 0.06) return 0.008;
    const note = Math.floor(t * 64 + p);
    const rr = createRng(note * 7919 + 3);
    const lvl = 0.38 + 0.3 * rr();
    const d = (t * 64) % 1;
    return (t > 0.44 && t < 0.5 ? 0.05 : lvl) * (1 - 0.35 * d) * (0.9 + 0.1 * r());
  },
  guitar: (t, r, p) => {
    const sect = (t > 0.25 && t < 0.44) || (t > 0.56 && t < 0.9) ? 1 : 0.42;
    const strum = 0.6 + 0.4 * Math.exp(-(((t * 256 + p) % 1) * 3));
    return (0.16 + 0.4 * r()) * sect * strum;
  },
  piano: (t, r, p) => {
    if (t < 0.04 || (t > 0.48 && t < 0.54)) return 0.015 * r();
    const chord = (t * 32 + p * 0.5) % 1;
    const attack = Math.exp(-chord * 3.8);
    const body = 0.16 + 0.56 * attack * (0.72 + 0.28 * r());
    return Math.min(0.92, body * (t > 0.93 ? (1 - t) * 14 : 1));
  },
  other: (t, r, p) => {
    const swell = 0.5 + 0.5 * Math.sin(t * 9 - 1 + p);
    return (0.1 + 0.24 * swell) * (0.75 + 0.25 * r()) * (t > 0.92 ? (1 - t) * 12 : 1);
  },
  instrumental: (t, r, p) => {
    const d = SHAPES.drums(t, r, p) * 0.55;
    const b = SHAPES.bass(t, r, p) * 0.4;
    const g = SHAPES.guitar(t, r, p) * 0.35;
    return Math.min(0.95, 0.1 + d + b * 0.6 + g * 0.4);
  },
  mix: (t, r, p) => {
    // Musical structure: quiet intro, verse, pre-chorus breath, chorus, quiet bridge,
    // drum build, second chorus, brief quiet break, final chorus, outro fade.
    let section: number;
    if (t < 0.055) {
      section = 0.14 + (t / 0.055) * 0.36;
    } else if (t < 0.21) {
      section = 0.56 + 0.08 * Math.sin(t * 28 + p);
    } else if (t < 0.255) {
      section = 0.22 + 0.14 * ((t - 0.21) / 0.045);
    } else if (t < 0.44) {
      section = 0.84 + 0.06 * Math.sin(t * 42 + p);
    } else if (t < 0.545) {
      const dip = Math.sin(((t - 0.44) / 0.105) * Math.PI);
      section = 0.26 - 0.13 * dip + 0.08 * Math.abs(Math.sin(t * 60 + p));
    } else if (t < 0.59) {
      section = 0.3 + 0.56 * ((t - 0.545) / 0.045);
    } else if (t < 0.79) {
      section = 0.87 + 0.06 * Math.sin(t * 36 + p);
    } else if (t < 0.845) {
      section = 0.21 + 0.07 * Math.abs(Math.sin(t * 50 + p));
    } else if (t < 0.945) {
      section = 0.85;
    } else {
      section = 0.85 * Math.pow(Math.max(0.02, (1 - t) / 0.055), 1.3);
    }

    // Dense rhythmic transients + fine per-bucket peak detail with realistic mastered fullness
    const barPos = (t * 220 + p * 0.2) % 1;
    const kick = Math.exp(-barPos * 9);
    const snare = Math.exp(-(((t * 110 + 0.5 + p * 0.2) % 1) * 11));
    const groove = 0.7 + 0.22 * kick + 0.16 * snare;
    const microDetail = 0.66 + 0.34 * Math.pow(r(), 0.75);
    const spike = r() > 0.85 ? 0.12 * r() : 0;

    return Math.min(0.98, Math.max(0.02, section * groove * microDetail + spike * section));
  },
};

export function generateStemEnvelope(
  sessionId: string,
  source: SourceId,
  buckets: number,
  present = true,
): Float32Array {
  const out = new Float32Array(buckets);
  if (!present) {
    return out;
  }
  // Use the comp's exact seed for Golden Hour so the waveforms match app.png closely.
  const isGoldenHour = sessionId === "golden-hour";
  const seed = isGoldenHour ? source.length * 9973 + 17 : hashString(`${sessionId}:${source}`);
  const phase = isGoldenHour ? 0 : ((hashString(sessionId) % 100) / 100) * Math.PI * 2;
  const r = createRng(seed);
  const shape = SHAPES[source];

  const raw = new Float32Array(buckets);
  for (let i = 0; i < buckets; i++) {
    raw[i] = Math.max(0, Math.min(1, shape(i / buckets, r, phase)));
  }

  // Keep sharp per-bucket transient detail on full mixes instead of blurring into a blob.
  if (source === "mix") {
    return raw;
  }

  for (let i = 0; i < buckets; i++) {
    const prev = i > 0 ? raw[i - 1] : raw[i];
    const next = i < buckets - 1 ? raw[i + 1] : raw[i];
    out[i] = Math.max(0, Math.min(1, (raw[i] + prev * 0.5 + next * 0.5) / 2));
  }
  return out;
}

export function generatePeaks(
  sessionId: string,
  source: SourceId,
  buckets: number,
  present = true,
): Peaks {
  const env = generateStemEnvelope(sessionId, source, buckets, present);
  const min = new Float32Array(buckets);
  const max = new Float32Array(buckets);
  if (source === "mix") {
    const asym = createRng(hashString(`${sessionId}:asym`));
    for (let i = 0; i < buckets; i++) {
      max[i] = Math.min(1, env[i] * (0.9 + 0.16 * asym()));
      min[i] = -Math.min(1, env[i] * (0.88 + 0.16 * asym()));
    }
  } else {
    for (let i = 0; i < buckets; i++) {
      max[i] = env[i];
      min[i] = -env[i] * 0.94;
    }
  }
  return encodePeaks(min, max);
}

export function generateActivity(sessionId: string, source: SourceId, present = true): number[] {
  const env = generateStemEnvelope(sessionId, source, ACTIVITY_BINS, present);
  return Array.from(env, (v) => Number(v.toFixed(3)));
}
