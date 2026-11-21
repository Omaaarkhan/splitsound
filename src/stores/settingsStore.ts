import type { AppInfo, Settings } from "../ipc/contract";
import { invoke } from "../ipc/client";
import { createStore } from "./createStore";

export interface SettingsState {
  settings: Settings;
  appInfo: AppInfo;
  loaded: boolean;
}

const initialSettingsState = (): SettingsState => ({
  settings: {
    defaultQuality: "fast",
    exportFormat: "wav24",
    exportDir: null,
    acceleration: "auto",
    threads: 0,
    checkForUpdates: true,
  },
  appInfo: {
    version: "0.1.0",
    platform: "macos",
    arch: "aarch64",
    gpu: { available: false, name: null },
    cpuThreads: 4,
    modelsDir: "",
    libraryDir: "",
    libraryBytes: 0,
  },
  loaded: false,
});

export const settingsStore = createStore<SettingsState>(initialSettingsState);

export async function refreshSettingsAndInfo(): Promise<void> {
  const [settings, appInfo] = await Promise.all([invoke("get_settings"), invoke("get_app_info")]);
  settingsStore.setState({ settings, appInfo, loaded: true });
}

export async function refreshAppInfo(): Promise<void> {
  const appInfo = await invoke("get_app_info");
  settingsStore.setState((prev) => ({ ...prev, appInfo }));
}

export async function updateSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = await invoke("update_settings", { patch });
  settingsStore.setState((prev) => ({ ...prev, settings: next }));
  return next;
}
