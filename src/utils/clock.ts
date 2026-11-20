export const MOCK_NOW_ISO = "2026-11-19T19:00:00.000Z";
export const MOCK_NOW_MS = Date.parse(MOCK_NOW_ISO);

let fixedNowMs: number | null = null;

export function setFixedNowMs(ms: number | null): void {
  fixedNowMs = ms;
}

export function getNowMs(): number {
  return fixedNowMs ?? Date.now();
}
