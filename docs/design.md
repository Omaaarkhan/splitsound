# Design

SplitSound looks like the place stems come from: a recording console. Not a
photo of one, a drawing of one. Flat colour, hard edges, printed legends.

## The idea

- **The faceplate.** The window is one slate-blue metal panel (the blue-grey of
  a 1970s Neve), not a page with cards on it. Regions are separated by printed
  hairlines and by depth, never by floating boxes.
- **The well.** Waveforms live in a recessed dark display that runs the full
  height of the desk. It is the only dark surface.
- **The tape.** On a real console every channel is named with a strip of
  artist tape and a marker. Here every stem is a strip of coloured tape with
  its name written on it, cut at 45 degrees the way tape is spliced. The tape
  colour is the stem's colour everywhere else: its waveform, its meter.
  This is the one loud thing in the interface. Everything else stays quiet so
  it can be.
- **The cut.** The 45-degree splice is the brand: in the wordmark, on the end
  of each tape strip, and as the line that sweeps across the song while it is
  being split.

## What to avoid

- No cards, no soft drop shadows, no gradients as decoration, no glassy blur.
- No rounded pills. Radii are 2 to 4 px.
- No all-caps labels, no letter-spaced eyebrow text, no monospace.
- No arrows or icons added to button text. A button says what it does.
- The handwriting face is for stem names on tape and nothing else.
- Colour is never the only signal: every coloured thing also has a name.

## Tokens

All values are in `src/styles/tokens.css`. Use the variables, never raw values.

| Role | Token | Notes |
| --- | --- | --- |
| Faceplate | `--plate` | Window background. `--plate-hi` for raised keys, `--plate-top` on hover, `--plate-lo` for pressed keys and inset fields. |
| Display | `--well` | Waveform area. `--well-line` for lane dividers and ruler ticks, `--well-deep` for slots, tracks and the hard edge under raised keys. |
| Legends | `--ink`, `--ink-2`, `--ink-3` | Primary, secondary, tertiary text. `--ink-3` only on wells or at 15px and up. |
| Hairlines | `--rule` | Lines printed on the plate. |
| Tape | `--tape` with `--marker` text | Primary buttons and neutral labels. |
| Lamps | `--lamp`, `--mute` | Solo, focus, progress, drop target; mute and destructive actions. |
| Stems | `--stem-*` | Tape background (with `--marker` text) and waveform colour. |

### Type

One family does the interface: **Archivo** (variable, weight and width axes),
bundled with the app. Width is the voice:

- 125% width, weight 800: the wordmark and the one headline on the home screen.
- 112% width, weight 700: primary buttons.
- 100% width, weight 400 to 600: everything you read.
- 88% width with tabular figures, weight 500 to 600: time, decibels, the ruler.

**Just Another Hand** is the marker. Stem names on tape only, 27px on a 30px
strip (22px on a 24px strip for absent stems), `--marker` colour, 3px of top
padding to centre it optically.

Scale: 11, 12, 13 (base), 15, 19, 27, 46 px. Sentence case everywhere.
Line length under 60 characters for anything longer than a label.

### Depth

Three levels, drawn with 1px edges (`--edge-*` tokens):

1. Recessed: the well (inner shadow at the top).
2. Flush: the plate.
3. Raised: keys and fader caps (1px light edge on top, 1 to 2px `--well-deep` edge below). A pressed key drops to `--plate-lo` and loses its bottom edge.

Lit keys fill with their lamp colour and switch to `--marker` text.

## Layout

One window, three rows: top bar (44px), desk (flexible), transport (76px).
The desk is a 276px strip on the left (flush plate) and the well on the right.
Rows in the strip line up exactly with lanes in the well. Text is left-aligned.
Minimum window 980 x 640.

```
+--------------------------------------------------------------------------+
| Split/Sound            Song title  Artist              Library  Settings |
+----------------------+---------------------------------------------------+
| 6 stems, split in 48s| 0:00    0:30    1:00    1:30    2:00    2:30      |
+----------------------+---------------------------------------------------+
| [Vocals/]          | |  ~~~~ waveform in the stem's colour ~~~~          |
| Singing, choir     | |                                                   |
| [S][M] ---o--- 0.0 | |                                                   |
+----------------------+---------------------------------------------------+
| [Drums/]           | |  ...                                              |
+----------------------+---------------------------------------------------+
| [Piano/] None heard in this song   (collapsed, 40px)                     |
+----------------------+---------------------------------------------------+
| [>]  1:24.3 / 3:51   [Loop] [Hear original]      Master --o-- [Export stems] |
+--------------------------------------------------------------------------+
```

On macOS the top bar is the title bar: leave 86px on the left for the window
buttons and mark the bar as a drag region. On Windows and Linux the native
title bar stays, and the top bar is an ordinary toolbar with 16px padding.

## Screens

### Home (no song open)

The top 46% of the desk is one full-width well, and it is the drop target.
Left third: the headline "Drop a song to split it" (46px, wide, 800), one
sentence under it (15px, `--ink-2`), the primary button "Choose a file" and,
beside it in 12px, the formats. Right two thirds: a drawing, on canvas, of one
off-white waveform arriving from the left, meeting a 45-degree cut, and
leaving as six thinner coloured waveforms stacked in lanes. It is static.
When a file is dragged over the window the cut line turns `--lamp`, the well
gets a 2px `--lamp` inner border, and the headline reads "Drop it here".

Below, on the plate: the library. A plain list, one row per song: artwork
(40px square, 2px radius; when there is none, a square in `--plate-lo`),
title and artist, length, the stems it was split into as small tape slivers
(14 x 8 px parallelograms in stem colours, with the names as the row's
accessible description), date, and a "Remove" action that appears on hover
and focus. A row that is still splitting shows its progress instead of the
date. Clicking a row opens the song.

Empty library: "Songs you split are kept here so you can come back to them."

Before the models are installed, the sentence under the headline gains a
second line: "The first split downloads its models (about 64 MB), once."
There is no onboarding beyond that.

When a newer version exists, a notice stays at the bottom of the window:
"SplitSound 26.48.0 is available." with "Update and restart" and "Later".

### Song, before splitting

Well: a single tall lane with the original waveform, drawn to at most 70% of
the lane height: `--ink` for the part already played, `--ink-3` for the part
ahead. You can play it.

Strip, top to bottom:

1. "Heard in this song": the detected instruments, strongest first, at most
   eight. Each row: a 10px square in the colour of the stem it will land in,
   the instrument name, and a five-segment level showing confidence. While
   the scan runs: "Listening for instruments". If the scan is unavailable:
   say so in one line and carry on.
2. "Split into": three choices, each a row with a square lamp (lit `--lamp`
   when chosen), a title, and the stems it produces as tape slivers:
   "Every instrument" (6), "Band" (4), "Vocals and backing" (2). The one the
   app suggests carries the word "Suggested" in `--ink-3`.
3. Quality: a two-key switch, "Fast" and "Best" (the chosen key looks
   pressed, `--plate-lo` with a 2px `--lamp` line along its top edge, so that
   "Split song" stays the only bright thing here), and one line under it with
   the time estimate ("About 40 seconds" / "About 4 minutes, cleaner
   result"). If a model has to be fetched first: "Needs a one-time 58 MB
   download".
4. The primary button: "Split song".

### Asking before a restricted model

Guitar and piano need a model that is not open source. When the chosen split
needs it and it is not installed, the "Every instrument" row says "Needs a
research-licensed model", and pressing "Split song" opens a panel instead of
starting. Title: "Guitar and piano need one more model". Body: the model's
terms, verbatim, then its size. Actions, from the right: the primary button
"Download from Meta and split", then "Split without guitar and piano", then
"Cancel". The app never describes this model as open source.

### Splitting

Strip: the current step in 19px ("Separating drums"), "Step 2 of 5", a 3px
progress bar in `--lamp` on a `--well-deep` slot, the time left ("About a
minute left"), and a plain key: "Cancel".

Well: the original waveform with the cut, a 45-degree `--lamp` line the full
height of the lane, at the position that matches overall progress. Ahead of
the cut the waveform is plain. Behind it, the waveform itself is drawn as
ribbons in the colours of the stems being made: at every point its height is
divided equally between them, top to bottom in lane order, with 2px of well
between ribbons, so they swell and thin with the music. One pale strip of
tape goes in, coloured strips come out. The line moves smoothly between progress
updates. The original can still be played while this runs.

When the split finishes, the one orchestrated moment in the app (about half
a second): the single lane parts into the stem lanes, each waveform sliding
from the centre to its own lane, 40 ms apart, top to bottom, and each tape
label wipes in from the left as its lane lands. With reduced motion: a plain
cross-fade.

### Stems (the main screen)

Reference: the comp screenshot.

Each strip row: the tape label (drag handle, see below), under it in 12px the
instruments heard inside that stem, then the controls on one line: Solo key
(S), Mute key (M), a horizontal fader, the level in dB (88% width, tabular).
At the right edge of the row, a 5px vertical meter in the stem's colour.

Each lane: the waveform, filled, in the stem's colour. The part already
played is at full strength; the part ahead is at 55%. A stem you cannot hear
right now (muted, or not soloed while another is) is drawn as an outline
only, so the picture always shows what you are hearing. Click or drag in any
lane or the ruler to move the playhead. The playhead is a 1px `--ink` line
across all lanes with a small triangle in the ruler.

A stem with nothing in it is collapsed to a 40px row at the bottom: the tape
at half opacity, "None heard in this song", and a dashed centre line in the
lane.

Dragging a tape label out of the window drags that stem's audio file, so it
can be dropped into a DAW or a folder. The label lifts slightly (2px up) on
hover to hint at it; the tooltip says "Drag into your DAW or a folder".

Dragging a label out carries a picture of that strip of tape with it.

Dragging along the ruler marks a section to loop: a `--lamp` bar along the
bottom of the ruler and a faint wash down the lanes. Its edges can be
dragged. A click without a drag just moves the playhead.

Transport, left to right: Play/Pause (58 x 42 raised key, solid glyph), the
time (27px, 88% width, tabular) with the total after it in `--ink-3`, "Loop"
(latching; with no section marked it loops the whole song), "Hear original"
(latching; while lit you hear the untouched song), then on the right the
master fader and the primary button "Export stems".

### Export

A panel (4px radius, `--plate` with a 1px `--well-deep` outline, over a
dimmed window). Title "Export stems". A checklist of the stems as tape
strips with checkboxes (absent stems unchecked); a format menu (WAV 24-bit,
WAV 16-bit, WAV 32-bit float, FLAC, MP3); a checkbox "Also export one file
with these stems mixed at the current levels"; the destination folder with a
"Change" key; and the buttons "Cancel" and "Export 5 files" (the number
follows the selection). Afterwards a notice at the bottom of the window:
"Exported 5 files" with the action "Show in folder".

### Settings

Same panel style, sections separated by hairlines:

- Splitting: default quality; "Use the graphics processor" (Automatic / Never / Always); processor threads.
- Models: each model with its name, one line such as "40 MB, from GitHub, MIT licence", what it is for, and "Download" or "Remove". A restricted model also shows its terms, and its "Download" opens the consent panel. Below the list, "Add model files" installs files fetched by hand.
- Library: where it is, how much space it uses, "Clear library".
- Updates: the version, "Check for updates", "Check automatically".
- About: one line, "SplitSound never uploads your music. Everything happens on this computer.", and a link to the licences.

## Words

Plain verbs, sentence case, no exclamation marks, no "please", no "oops".
An action keeps its name all the way through: the button says "Export
stems", the panel is titled "Export stems", the notice says "Exported".

| Situation | Text |
| --- | --- |
| Unsupported file | "SplitSound can't read this file. It opens MP3, WAV, FLAC, Opus, OGG, M4A and AIFF." |
| Damaged file | "This file is damaged or incomplete, so it couldn't be opened." |
| Model download failed | "The model couldn't be downloaded. Check your connection and try again." |
| Not enough memory | "This song needs more memory than is free right now. Close other apps and try again." |
| No audio device | "No audio output was found, so nothing can be played. Splitting and exporting still work." |
| Split cancelled | "Split cancelled." |
| Confirm remove | "Remove this song and its stems from the library? The original file is not touched." |

Stem names: Vocals, Drums, Bass, Guitar, Piano, Everything else, Backing track.

## Motion

Controls answer in 90 ms; panels and lanes move in 180 ms; the split reveal
takes about 520 ms. Easing `--ease-out`. Nothing loops or moves on its own
except the cut line while a split runs and the meters while music plays.
Honour `prefers-reduced-motion`.

## Access

- Every control is reachable and operable by keyboard, with a visible focus
  ring (`--focus-ring`).
- Space plays and pauses. Left and right arrows move 5 seconds. Home returns
  to the start. 1 to 6 solo the lane with that number; Shift with the number
  mutes it. Ctrl/Cmd+O chooses a file, Ctrl/Cmd+E opens export, Ctrl/Cmd+,
  opens settings, Escape closes a panel.
- Faders are real sliders (arrow keys step 0.5 dB, double-click returns to 0).
- Keys that latch expose their pressed state; the S and M keys are named in
  full for screen readers ("Solo vocals", "Mute vocals").
- Text contrast is at least 4.5:1.
