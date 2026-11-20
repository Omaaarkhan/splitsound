import type { ExportFormat, SourceId, SplitMode, StemId } from "../ipc/contract";
import { STEM_ORDER } from "../ipc/contract";

export const STEM_NAMES: Readonly<Record<StemId, string>> = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  guitar: "Guitar",
  piano: "Piano",
  other: "Everything else",
  instrumental: "Backing track",
};

export const STEM_CSS_VAR: Readonly<Record<SourceId, string>> = {
  vocals: "--stem-vocals",
  drums: "--stem-drums",
  bass: "--stem-bass",
  guitar: "--stem-guitar",
  piano: "--stem-piano",
  other: "--stem-other",
  instrumental: "--stem-instrumental",
  mix: "--stem-mix",
};

export const SPLIT_MODE_LABELS: Readonly<Record<SplitMode, string>> = {
  all: "Every instrument",
  band: "Band",
  vocals: "Vocals and backing",
};

export const SPLIT_MODES_ORDER: readonly SplitMode[] = ["all", "band", "vocals"];

export const EXPORT_FORMATS: readonly { value: ExportFormat; label: string; ext: string }[] = [
  { value: "wav24", label: "WAV 24-bit", ext: "wav" },
  { value: "wav16", label: "WAV 16-bit", ext: "wav" },
  { value: "wav32f", label: "WAV 32-bit float", ext: "wav" },
  { value: "flac", label: "FLAC", ext: "flac" },
  { value: "mp3", label: "MP3", ext: "mp3" },
];

export const AUDIO_EXTENSIONS = [
  "mp3",
  "wav",
  "flac",
  "opus",
  "ogg",
  "m4a",
  "aac",
  "aiff",
  "aif",
] as const;

export function stemName(id: StemId): string {
  return STEM_NAMES[id];
}

export function stemNameLower(id: StemId): string {
  return STEM_NAMES[id].toLowerCase();
}

export function sortStemsForDisplay<T extends { id: StemId; present: boolean }>(
  stems: readonly T[],
): T[] {
  const rank = (id: StemId): number => {
    const idx = STEM_ORDER.indexOf(id);
    return idx === -1 ? STEM_ORDER.length : idx;
  };
  return [...stems].sort((a, b) => {
    if (a.present !== b.present) return a.present ? -1 : 1;
    return rank(a.id) - rank(b.id);
  });
}

export function stemColorVar(id: SourceId): string {
  return `var(${STEM_CSS_VAR[id]})`;
}
