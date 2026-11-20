import type { AppInfo, ModelStatus, Session, Settings, StemId, StemInfo } from "../contract";
import { MOCK_NOW_ISO } from "../../utils/clock";
import { generateActivity } from "./waveforms";

export const DEMUCS_TERMS =
  "Demucs is published by Meta for research. Its weights are not under an open-source licence and are likely limited to non-commercial use (CC BY-NC). SplitSound downloads them directly from Meta and keeps them on this computer.";

export function createInitialModels(): ModelStatus[] {
  return [
    {
      id: "scnet_small",
      name: "SCNet Small",
      description: "Separates vocals, drums, bass and everything else quickly.",
      sizeBytes: 22_516_980,
      downloadBytes: 20_971_520,
      installed: true,
      downloadProgress: null,
      license: "MIT",
      attribution: "SCNet authors",
      source: "GitHub",
      restricted: false,
      terms: null,
    },
    {
      id: "scnet_large",
      name: "SCNet Large",
      description: "Higher-quality separation for vocals, drums, bass and everything else.",
      sizeBytes: 85_750_454,
      downloadBytes: 77_996_104,
      installed: false,
      downloadProgress: null,
      license: "MIT",
      attribution: "SCNet authors",
      source: "GitHub",
      restricted: false,
      terms: null,
    },
    {
      id: "htdemucs_6s",
      name: "Demucs 6-stem",
      description: "Separates guitar and piano from the rest of the mix.",
      sizeBytes: 57_800_000,
      downloadBytes: 57_671_680,
      installed: false,
      downloadProgress: null,
      license: "Research use (CC BY-NC)",
      attribution: "Meta Platforms, Inc.",
      source: "Meta",
      restricted: true,
      terms: DEMUCS_TERMS,
    },
    {
      id: "tagger",
      name: "Instrument detector",
      description: "Listens to songs and identifies which instruments are present.",
      sizeBytes: 23_741_689,
      downloadBytes: 19_922_944,
      installed: true,
      downloadProgress: null,
      license: "MIT",
      attribution: "Open audio tagger weights",
      source: "GitHub",
      restricted: false,
      terms: null,
    },
  ];
}

export function createInitialSettings(): Settings {
  return {
    defaultQuality: "fast",
    exportFormat: "wav24",
    exportDir: "/Users/demo/Music/SplitSound",
    acceleration: "auto",
    threads: 0,
    checkForUpdates: true,
  };
}

export function createInitialAppInfo(libraryBytes: number): AppInfo {
  return {
    version: "0.1.0",
    platform: "macos",
    arch: "aarch64",
    gpu: { available: true, name: "Apple M3 Metal" },
    cpuThreads: 8,
    modelsDir: "/Users/demo/Library/Application Support/SplitSound/models",
    libraryDir: "/Users/demo/Library/Application Support/SplitSound/library",
    libraryBytes,
  };
}

function makeArtworkSvg(bg: string, stripe1: string, stripe2: string, accent: string): string {
  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">`,
    `<rect width="80" height="80" fill="${bg}"/>`,
    `<polygon points="0,56 80,22 80,46 0,80" fill="${stripe1}"/>`,
    `<polygon points="0,34 80,0 80,14 0,48" fill="${stripe2}"/>`,
    `<circle cx="24" cy="24" r="10" fill="${accent}"/>`,
    `</svg>`,
  ].join("");
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

export const MOCK_ARTWORK: Record<string, string> = {
  "golden-hour": makeArtworkSvg("#1f2e38", "#f9914d", "#f4da57", "#fe9cb8"),
  "midnight-trio": makeArtworkSvg("#16232d", "#67a1eb", "#6bddb5", "#f3f0e6"),
  "rooftop-signal": makeArtworkSvg("#25343f", "#c0a1e4", "#fe9cb8", "#fbc044"),
  "silver-lines": makeArtworkSvg("#1b2a34", "#abcde0", "#67a1eb", "#f4da57"),
};

function makeStem(
  sessionId: string,
  id: StemId,
  present: boolean,
  containsLabels: string[],
  peakDb: number,
  rmsDb: number,
): StemInfo {
  return {
    id,
    path: `/Users/demo/Library/Application Support/SplitSound/library/${sessionId}/${id}.wav`,
    present,
    activeFraction: present ? 0.72 : 0.01,
    peakDb: present ? peakDb : -68,
    rmsDb: present ? rmsDb : -74,
    activity: generateActivity(sessionId, id, present),
    contains: containsLabels.map((label, idx) => ({
      label,
      family: id,
      confidence: Math.max(0.45, 0.92 - idx * 0.14),
    })),
  };
}

export function createInitialSessions(): Session[] {
  const goldenHour: Session = {
    id: "golden-hour",
    createdAt: MOCK_NOW_ISO,
    track: {
      path: "/Users/demo/Music/Golden Hour on Larkin Street.flac",
      fileName: "Golden Hour on Larkin Street.flac",
      title: "Golden Hour on Larkin Street",
      artist: "The Paper Kites Club",
      album: "Larkin Sessions",
      durationSec: 231,
      sampleRate: 44100,
      channels: 2,
      codec: "FLAC",
      bitrateKbps: 980,
      fileSizeBytes: 28_400_000,
      hasArtwork: true,
    },
    status: "ready",
    analysis: "done",
    detections: [
      { label: "Drum kit", family: "drums", confidence: 0.96 },
      { label: "Singing", family: "vocals", confidence: 0.93 },
      { label: "Electric guitar", family: "guitar", confidence: 0.88 },
      { label: "Bass guitar", family: "bass", confidence: 0.85 },
      { label: "Acoustic guitar", family: "guitar", confidence: 0.72 },
      { label: "Synthesizer", family: "other", confidence: 0.66 },
      { label: "Tambourine", family: "drums", confidence: 0.58 },
      { label: "Strings", family: "other", confidence: 0.52 },
    ],
    suggestedMode: "all",
    split: {
      mode: "all",
      quality: "fast",
      finishedAt: MOCK_NOW_ISO,
      elapsedSec: 48,
      modelIds: ["scnet_small", "htdemucs_6s"],
    },
    stems: [
      makeStem("golden-hour", "vocals", true, ["Singing", "choir"], -1.8, -14.2),
      makeStem("golden-hour", "drums", true, ["Drum kit", "tambourine"], -0.6, -11.4),
      makeStem("golden-hour", "bass", true, ["Bass guitar"], -2.4, -15.1),
      makeStem("golden-hour", "guitar", true, ["Electric guitar", "acoustic guitar"], -3.1, -16.8),
      makeStem("golden-hour", "piano", false, [], -68, -74),
      makeStem("golden-hour", "other", true, ["Synthesizer", "strings"], -4.5, -19.2),
    ],
    error: null,
  };

  const midnightTrio: Session = {
    id: "midnight-trio",
    createdAt: "2026-11-18T21:15:00.000Z",
    track: {
      path: "/Users/demo/Music/Midnight Trio in E Minor.wav",
      fileName: "Midnight Trio in E Minor.wav",
      title: "Midnight Trio in E Minor",
      artist: "Harbor Quartet",
      album: "Late Tape",
      durationSec: 274,
      sampleRate: 48000,
      channels: 2,
      codec: "WAV",
      bitrateKbps: 2304,
      fileSizeBytes: 78_900_000,
      hasArtwork: true,
    },
    status: "ready",
    analysis: "done",
    detections: [
      { label: "Acoustic piano", family: "piano", confidence: 0.95 },
      { label: "Double bass", family: "bass", confidence: 0.91 },
      { label: "Drum kit", family: "drums", confidence: 0.87 },
      { label: "Trumpet", family: "other", confidence: 0.74 },
      { label: "Ride cymbal", family: "drums", confidence: 0.65 },
    ],
    suggestedMode: "all",
    split: {
      mode: "all",
      quality: "best",
      finishedAt: "2026-11-18T21:19:00.000Z",
      elapsedSec: 198,
      modelIds: ["scnet_large", "htdemucs_6s"],
    },
    stems: [
      makeStem("midnight-trio", "vocals", false, [], -70, -76),
      makeStem("midnight-trio", "drums", true, ["Drum kit", "ride cymbal"], -2.1, -15.6),
      makeStem("midnight-trio", "bass", true, ["Double bass"], -1.5, -13.9),
      makeStem("midnight-trio", "guitar", false, [], -69, -75),
      makeStem("midnight-trio", "piano", true, ["Acoustic piano"], -1.2, -14.1),
      makeStem("midnight-trio", "other", true, ["Trumpet", "room reverb"], -3.4, -18.5),
    ],
    error: null,
  };

  const rooftopSignal: Session = {
    id: "rooftop-signal",
    createdAt: "2026-11-14T16:40:00.000Z",
    track: {
      path: "/Users/demo/Music/Rooftop Signal.mp3",
      fileName: "Rooftop Signal.mp3",
      title: "Rooftop Signal",
      artist: "Nora Vance",
      album: "Night FM",
      durationSec: 218,
      sampleRate: 44100,
      channels: 2,
      codec: "MP3",
      bitrateKbps: 320,
      fileSizeBytes: 8_720_000,
      hasArtwork: true,
    },
    status: "ready",
    analysis: "done",
    detections: [
      { label: "Singing", family: "vocals", confidence: 0.94 },
      { label: "Drum machine", family: "drums", confidence: 0.89 },
      { label: "Synth bass", family: "bass", confidence: 0.84 },
      { label: "Synthesizer", family: "other", confidence: 0.81 },
    ],
    suggestedMode: "band",
    split: {
      mode: "band",
      quality: "fast",
      finishedAt: "2026-11-14T16:41:00.000Z",
      elapsedSec: 34,
      modelIds: ["scnet_small"],
    },
    stems: [
      makeStem("rooftop-signal", "vocals", true, ["Singing"], -1.4, -13.6),
      makeStem("rooftop-signal", "drums", true, ["Drum machine"], -0.9, -11.9),
      makeStem("rooftop-signal", "bass", true, ["Synth bass"], -2.2, -14.7),
      makeStem("rooftop-signal", "other", true, ["Synthesizer", "pad"], -3.8, -17.3),
    ],
    error: null,
  };

  const velvetStatic: Session = {
    id: "velvet-static",
    createdAt: "2026-11-19T18:45:00.000Z",
    track: {
      path: "/Users/demo/Music/Velvet Static.flac",
      fileName: "Velvet Static.flac",
      title: "Velvet Static",
      artist: "Cedar & Wire",
      album: "Northbound",
      durationSec: 256,
      sampleRate: 44100,
      channels: 2,
      codec: "FLAC",
      bitrateKbps: 912,
      fileSizeBytes: 29_100_000,
      hasArtwork: false,
    },
    status: "imported",
    analysis: "done",
    detections: [
      { label: "Singing", family: "vocals", confidence: 0.95 },
      { label: "Drum kit", family: "drums", confidence: 0.92 },
      { label: "Electric guitar", family: "guitar", confidence: 0.89 },
      { label: "Acoustic piano", family: "piano", confidence: 0.81 },
      { label: "Bass guitar", family: "bass", confidence: 0.78 },
      { label: "Synthesizer", family: "other", confidence: 0.61 },
      { label: "Tambourine", family: "drums", confidence: 0.48 },
    ],
    suggestedMode: "all",
    split: null,
    stems: [],
    error: null,
  };

  const silverLines: Session = {
    id: "silver-lines",
    createdAt: "2026-11-19T18:55:00.000Z",
    track: {
      path: "/Users/demo/Music/Silver Lines (Demo).aiff",
      fileName: "Silver Lines (Demo).aiff",
      title: "Silver Lines (Demo)",
      artist: "M. Lindqvist",
      album: null,
      durationSec: 194,
      sampleRate: 44100,
      channels: 2,
      codec: "AIFF",
      bitrateKbps: 1411,
      fileSizeBytes: 34_200_000,
      hasArtwork: true,
    },
    status: "splitting",
    analysis: "done",
    detections: [
      { label: "Singing", family: "vocals", confidence: 0.91 },
      { label: "Drum kit", family: "drums", confidence: 0.88 },
      { label: "Electric guitar", family: "guitar", confidence: 0.84 },
      { label: "Bass guitar", family: "bass", confidence: 0.79 },
      { label: "Acoustic piano", family: "piano", confidence: 0.67 },
    ],
    suggestedMode: "all",
    split: {
      mode: "all",
      quality: "fast",
      finishedAt: "",
      elapsedSec: 0,
      modelIds: ["scnet_small", "htdemucs_6s"],
    },
    stems: [],
    error: null,
  };

  return [goldenHour, silverLines, midnightTrio, rooftopSignal, velvetStatic];
}
