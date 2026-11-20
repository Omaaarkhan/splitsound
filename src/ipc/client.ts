import type { CommandName, Commands, EventName, Events, SourceId, SplitProgress } from "./contract";

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

async function createTauriTransport(): Promise<IpcTransport> {
  const [{ invoke: tauriInvoke }, { listen: tauriListen }] = await Promise.all([
    import("@tauri-apps/api/core"),
    import("@tauri-apps/api/event"),
  ]);

  return {
    invoke: (command, args) => tauriInvoke(command, args as Record<string, unknown>),
    listen: async (event, handler) => {
      const unlisten = await tauriListen(event, (ev) => {
        handler(ev.payload as Events[typeof event]);
      });
      return unlisten;
    },
  };
}

async function createMockTransport(): Promise<IpcTransport> {
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
