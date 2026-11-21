/**
 * Console-style fader taper: -60 dB .. +10 dB with 0.0 dB at 68% travel,
 * -4.5 dB near 55%, and +2.0 dB near 74%, matching hardware console feel.
 */
export const FADER_MIN_DB = -60;
export const FADER_MAX_DB = 10;
export const FADER_STEP_DB = 0.5;

const STOPS: readonly { db: number; pos: number }[] = [
  { db: -60, pos: 0 },
  { db: -18, pos: 0.24 },
  { db: -6, pos: 0.5 },
  { db: 0, pos: 0.68 },
  { db: 10, pos: 1 },
];

export function clampDb(db: number): number {
  if (!Number.isFinite(db)) return 0;
  return Math.max(FADER_MIN_DB, Math.min(FADER_MAX_DB, db));
}

export function snapDb(db: number): number {
  const clamped = clampDb(db);
  return Math.round(clamped / FADER_STEP_DB) * FADER_STEP_DB;
}

export function dbToFaderFraction(db: number): number {
  const v = clampDb(db);
  for (let i = 0; i < STOPS.length - 1; i++) {
    const a = STOPS[i];
    const b = STOPS[i + 1];
    if (v <= b.db) {
      const t = (v - a.db) / (b.db - a.db);
      return a.pos + t * (b.pos - a.pos);
    }
  }
  return 1;
}

export function faderFractionToDb(fraction: number): number {
  const f = Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0));
  for (let i = 0; i < STOPS.length - 1; i++) {
    const a = STOPS[i];
    const b = STOPS[i + 1];
    if (f <= b.pos) {
      const t = (f - a.pos) / (b.pos - a.pos);
      return snapDb(a.db + t * (b.db - a.db));
    }
  }
  return FADER_MAX_DB;
}
