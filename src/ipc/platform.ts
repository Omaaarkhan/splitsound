import { isTauriRuntime, type UnlistenFn } from "./client";
import { AUDIO_EXTENSIONS } from "../utils/stems";

export async function pickAudioFiles(): Promise<string[]> {
  if (isTauriRuntime()) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({
      title: "Choose a song",
      multiple: true,
      filters: [{ name: "Audio", extensions: [...AUDIO_EXTENSIONS] }],
    });
    if (!selected) return [];
    return Array.isArray(selected) ? selected : [selected];
  }

  return new Promise<string[]>((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = AUDIO_EXTENSIONS.map((ext) => `.${ext}`).join(",");
    let settled = false;
    const finish = (files: string[]) => {
      if (settled) return;
      settled = true;
      resolve(files);
    };
    input.addEventListener("change", () => {
      const list = Array.from(input.files ?? []).map((f) => f.name);
      finish(list);
    });
    input.addEventListener("cancel", () => finish([]));
    input.click();
  });
}

export async function pickDirectory(defaultPath?: string | null): Promise<string | null> {
  if (isTauriRuntime()) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({
      title: "Choose export folder",
      directory: true,
      multiple: false,
      defaultPath: defaultPath ?? undefined,
    });
    if (typeof selected === "string") return selected;
    return null;
  }

  // Deterministic browser fallback when running outside Tauri
  const candidates = [
    "/Users/demo/Music/SplitSound",
    "/Users/demo/Desktop/Stems",
    "/Users/demo/Documents/Projects",
  ];
  const currentIdx = defaultPath ? candidates.indexOf(defaultPath) : -1;
  return candidates[(currentIdx + 1) % candidates.length];
}

export async function pickAnyFiles(): Promise<string[]> {
  if (isTauriRuntime()) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({
      title: "Add model files",
      multiple: true,
    });
    if (!selected) return [];
    return Array.isArray(selected) ? selected : [selected];
  }

  return new Promise<string[]>((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    let settled = false;
    const finish = (files: string[]) => {
      if (settled) return;
      settled = true;
      resolve(files);
    };
    input.addEventListener("change", () => {
      const list = Array.from(input.files ?? []).map((f) => f.name);
      finish(list);
    });
    input.addEventListener("cancel", () => finish([]));
    input.click();
  });
}

export interface FileDropHandlers {
  onDragEnter: () => void;
  onDragLeave: () => void;
  onDrop: (paths: string[]) => void;
}

export async function subscribeWindowFileDrop(handlers: FileDropHandlers): Promise<UnlistenFn> {
  if (isTauriRuntime()) {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    return getCurrentWindow().onDragDropEvent((event) => {
      const payload = event.payload;
      if (payload.type === "enter" || payload.type === "over") {
        handlers.onDragEnter();
      } else if (payload.type === "leave") {
        handlers.onDragLeave();
      } else if (payload.type === "drop") {
        handlers.onDragLeave();
        if (payload.paths.length > 0) {
          handlers.onDrop(payload.paths);
        }
      }
    });
  }

  let depth = 0;
  const hasFiles = (e: DragEvent): boolean =>
    Array.from(e.dataTransfer?.types ?? []).includes("Files");

  const onDragEnter = (e: DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    handlers.onDragEnter();
  };
  const onDragOver = (e: DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = "copy";
    }
  };
  const onDragLeave = (e: DragEvent) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) {
      handlers.onDragLeave();
    }
  };
  const onDrop = (e: DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    handlers.onDragLeave();
    const files = Array.from(e.dataTransfer?.files ?? []).map((f) => f.name);
    if (files.length > 0) {
      handlers.onDrop(files);
    }
  };

  window.addEventListener("dragenter", onDragEnter);
  window.addEventListener("dragover", onDragOver);
  window.addEventListener("dragleave", onDragLeave);
  window.addEventListener("drop", onDrop);

  return () => {
    window.removeEventListener("dragenter", onDragEnter);
    window.removeEventListener("dragover", onDragOver);
    window.removeEventListener("dragleave", onDragLeave);
    window.removeEventListener("drop", onDrop);
  };
}

export async function startNativeStemDrag(paths: string[], iconDataUrl?: string): Promise<void> {
  if (!isTauriRuntime() || paths.length === 0) return;
  try {
    const { startDrag } = await import("@crabnebula/tauri-plugin-drag");
    const fallbackIcon =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    await startDrag({ item: paths, icon: iconDataUrl || fallbackIcon, mode: "copy" });
  } catch {
    // Ignore native drag cancellation or unsupported host errors
  }
}
