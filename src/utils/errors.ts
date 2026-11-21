import type { IpcError } from "../ipc/contract";

export const COPY = {
  unsupportedFile:
    "SplitSound can't read this file. It opens MP3, WAV, FLAC, Opus, OGG, M4A and AIFF.",
  damagedFile: "This file is damaged or incomplete, so it couldn't be opened.",
  modelDownloadFailed: "The model couldn't be downloaded. Check your connection and try again.",
  notEnoughMemory:
    "This song needs more memory than is free right now. Close other apps and try again.",
  noAudioDevice:
    "No audio output was found, so nothing can be played. Splitting and exporting still work.",
  splitCancelled: "Split cancelled.",
  confirmRemove:
    "Remove this song and its stems from the library? The original file is not touched.",
  confirmClearLibrary:
    "Remove every song and its stems from the library? Your original files are not touched.",
} as const;

export function isIpcError(err: unknown): err is IpcError {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    "message" in err &&
    typeof (err as { code: unknown }).code === "string" &&
    typeof (err as { message: unknown }).message === "string"
  );
}

export function formatIpcError(err: unknown): string {
  if (isIpcError(err)) {
    switch (err.code) {
      case "unsupported_format":
        return COPY.unsupportedFile;
      case "decode_failed":
        return COPY.damagedFile;
      case "model_download_failed":
      case "offline":
        return COPY.modelDownloadFailed;
      case "out_of_memory":
        return COPY.notEnoughMemory;
      case "audio_device":
        return COPY.noAudioDevice;
      case "cancelled":
        return COPY.splitCancelled;
      default:
        return err.message || "Something went wrong.";
    }
  }
  if (err instanceof Error && err.message) {
    return err.message;
  }
  return "Something went wrong.";
}
