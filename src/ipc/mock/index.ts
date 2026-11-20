import { createMockBackend, type MockBackend, type MockScene } from "./backend";
import { MOCK_NOW_MS, setFixedNowMs } from "../../utils/clock";

const VALID_SCENES = new Set<MockScene>([
  "home",
  "home-empty",
  "home-first-run",
  "imported",
  "consent",
  "scan-downloading",
  "scan-unavailable",
  "splitting",
  "ready",
  "ready-solo",
  "loop",
  "export",
  "settings",
  "update",
  "error",
]);

export function readSceneFromLocation(): MockScene | null {
  if (typeof window === "undefined") return null;
  const raw = new URLSearchParams(window.location.search).get("scene");
  if (raw && VALID_SCENES.has(raw as MockScene)) {
    return raw as MockScene;
  }
  return null;
}

let singleton: MockBackend | null = null;

export function getMockBackend(): MockBackend {
  if (!singleton) {
    const scene = readSceneFromLocation();
    if (scene !== null) {
      setFixedNowMs(MOCK_NOW_MS);
    }
    singleton = createMockBackend({ scene });
  }
  return singleton;
}

export type { MockBackend, MockScene };
export { createMockBackend };
