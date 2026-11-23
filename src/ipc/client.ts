import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { listen as tauriListen } from "@tauri-apps/api/event";
import type { CommandName, Commands, EventName, Events, SourceId, SplitProgress } from "./contract";

declare const __ENABLE_MOCK__: boolean | undefined;

const ENABLE_MOCK = typeof __ENABLE_MOCK__ !== "undefined" ? __ENABLE_MOCK__ : true;

export type UnlistenFn = () => void;

export interface IpcTransport {
  invoke<K extends CommandName>(
    command: K,
    args: Commands[K]["args"],
  ): Promise<Commands[K]["result"]>;
  listen<E extends EventName>(event: E, handler: (payload: Events[E]) => void): Promise<UnlistenFn>;
  getInitialSplitProgress?(): Record<string, SplitProgress>;
  getInitialPlaybackLevels?(): Partial<Record<SourceId, number>>;
}

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

let transportPromise: Promise<IpcTransport> | null = null;

function createTauriTransport(): Promise<IpcTransport> {
  return Promise.resolve({
    invoke: (command, args) => tauriInvoke(command, args as Record<string, unknown>),
    listen: async (event, handler) => {
      const unlisten = await tauriListen(event, (ev) => {
        handler(ev.payload as Events[typeof event]);
      });
      return unlisten;
    },
  });
}

async function createMockTransport(): Promise<IpcTransport> {
  if (!ENABLE_MOCK) {
    throw new Error("Tauri IPC runtime unavailable");
  }
  const mod = await import("./mock/index");
  return mod.getMockBackend();
}

export function getTransport(): Promise<IpcTransport> {
  if (!transportPromise) {
    transportPromise = isTauriRuntime() ? createTauriTransport() : createMockTransport();
  }
  return transportPromise;
}

export function setTransportForTesting(transport: IpcTransport | null): void {
  transportPromise = transport ? Promise.resolve(transport) : null;
}

type EmptyArgs = Record<string, never>;

export async function invoke<K extends CommandName>(
  command: K,
  ...rest: Commands[K]["args"] extends EmptyArgs ? [args?: EmptyArgs] : [args: Commands[K]["args"]]
): Promise<Commands[K]["result"]> {
  const transport = await getTransport();
  const args = (rest[0] ?? {}) as Commands[K]["args"];
  return transport.invoke(command, args);
}

export async function listen<E extends EventName>(
  event: E,
  handler: (payload: Events[E]) => void,
): Promise<UnlistenFn> {
  const transport = await getTransport();
  return transport.listen(event, handler);
}
