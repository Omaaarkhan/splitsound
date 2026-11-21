#!/bin/sh
# Regenerates every binary fixture in this folder from synthetic signals.
#
# Needs: python3 (standard library only), an ffmpeg build with libopus and
# libmp3lame, and, for the two Ogg Vorbis files, a Python that has the
# `soundfile` package (libsndfile with libvorbis); point SF_PYTHON at it.
# Nothing here is downloaded. The committed files were made with ffmpeg
# 8.0.1 (libopus, libmp3lame) and libsndfile 1.2.2 on macOS; a different
# encoder version gives different (equally valid) bytes, so regenerate the
# whole set together, never one file.
#
#   SF_PYTHON=../../../../ml/.venv/bin/python sh make_fixtures.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ff() { ffmpeg -hide_banner -loglevel error -nostdin -y "$@"; }
bitexact="-map_metadata -1 -fflags +bitexact -flags:a +bitexact"
SF_PYTHON=${SF_PYTHON:-python3}

python3 "$here/make_sources.py" "$tmp"
ff -f lavfi -i "color=c=0x3366cc:s=16x16" -frames:v 1 "$tmp/cover.png"

a="$here/audio"
rm -f "$a"/* "$here"/opus/* "$here"/corrupt/*

# --- One file per supported format -------------------------------------
# Signal: left 440 Hz, right 880 Hz, amplitude 0.5 (see make_sources.py).
ff -i "$tmp/tone_44k_half.wav" $bitexact -c:a pcm_s16le "$a/tone_s16.wav"
ff -i "$tmp/tone_48k_half.wav" $bitexact -c:a pcm_s24le "$a/tone_s24_48k.wav"
ff -i "$tmp/tone_mono_44k.wav" $bitexact -t 0.5 -c:a pcm_f32le "$a/tone_f32_mono.wav"
ff -i "$tmp/tone_44k_half.wav" $bitexact -c:a pcm_s16be "$a/tone.aiff"
ff -i "$tmp/tone_44k.wav" $bitexact -c:a flac "$a/tone.flac"
ff -i "$tmp/tone_44k.wav" $bitexact -c:a libmp3lame -b:a 128k "$a/tone.mp3"
ff -i "$tmp/tone_mono_44k.wav" $bitexact -c:a libmp3lame -b:a 64k "$a/tone_mono.mp3"
# Vorbis comes from libvorbis through libsndfile: ffmpeg's built-in Vorbis
# encoder is experimental and produced a stream that ffmpeg's and
# Symphonia's decoders disagree on (right channel), so it is no reference.
"$SF_PYTHON" - "$tmp/tone_44k.wav" "$a/tone.ogg" "$a/tagged.ogg" <<'PY'
import sys
import soundfile as sf
data, rate = sf.read(sys.argv[1], dtype="float32")
sf.write(sys.argv[2], data, rate, format="OGG", subtype="VORBIS")
with sf.SoundFile(sys.argv[3], "w", samplerate=rate, channels=data.shape[1], format="OGG", subtype="VORBIS") as f:
    f.title = "Fixture Title"
    f.artist = "Fixture Artist"
    f.album = "Fixture Album"
    f.write(data)
PY
ff -i "$tmp/tone_48k.wav" $bitexact -c:a libopus -b:a 96k "$a/tone.opus"
ff -i "$tmp/tone_44k.wav" $bitexact -c:a aac -b:a 128k "$a/tone.m4a"
ff -i "$tmp/tone_44k.wav" $bitexact -c:a aac -b:a 128k -f adts "$a/tone.aac"
ff -i "$tmp/tone_44k.wav" $bitexact -c:a alac "$a/tone_alac.m4a"
ff -i "$tmp/tone_44k_half.wav" $bitexact -c:a alac "$a/tone_alac.caf"
ff -i "$tmp/tone_51_44k.wav" $bitexact -c:a flac "$a/tone_51.flac"

# --- Tags and cover art ---------------------------------------------------
tagged() { # output [encoder options...]
  out=$1; shift
  ff -i "$tmp/tone_44k.wav" -i "$tmp/cover.png" -map 0:a -map 1:v \
    -fflags +bitexact -flags:a +bitexact "$@" -c:v copy -disposition:v attached_pic \
    -metadata "title=Fixture Title" -metadata "artist=Fixture Artist" \
    -metadata "album=Fixture Album" "$out"
}
tagged "$a/tagged.mp3" -c:a libmp3lame -b:a 96k -id3v2_version 3
tagged "$a/tagged.flac" -c:a flac
tagged "$a/tagged.m4a" -c:a aac -b:a 96k
ff -i "$tmp/tone_48k.wav" -fflags +bitexact -flags:a +bitexact -c:a libopus -b:a 64k \
  -metadata "title=Fixture Title" -metadata "artist=Fixture Artist" \
  -metadata "album=Fixture Album" "$a/tagged.opus"

# --- Opus parity set --------------------------------------------------------
# Every Opus mode, mono and stereo, low to high bitrate. Each file gets a
# reference decode made by libopus itself (through ffmpeg), stored as 16-bit
# FLAC so it stays small; tests/opus_parity.rs compares against it.
o="$here/opus"
enc() { # name source [encoder options...]
  name=$1; src=$2; shift 2
  ff -i "$tmp/$src" $bitexact -c:a libopus "$@" "$o/$name.opus"
}
enc silk_nb_8k_mono      speech_48k.wav        -ac 1 -application voip  -b:a 8k  -cutoff 4000
enc silk_mb_10k_mono_60ms speech_48k.wav       -ac 1 -application voip  -b:a 10k -cutoff 6000 -frame_duration 60
enc silk_12k_mono        speech_48k.wav        -ac 1 -application voip  -b:a 12k
enc silk_24k_stereo      speech_stereo_48k.wav -ac 2 -application voip  -b:a 24k -cutoff 8000
enc hybrid_20k_mono_60ms speech_48k.wav        -ac 1 -application voip  -b:a 20k -frame_duration 60
enc hybrid_32k_mono      speech_48k.wav        -ac 1 -application voip  -b:a 32k
enc hybrid_48k_stereo    speech_stereo_48k.wav -ac 2 -application voip  -b:a 48k
enc celt_64k_stereo      music_48k.wav         -ac 2 -application audio -b:a 64k
enc celt_128k_stereo     music_48k.wav         -ac 2 -application audio -b:a 128k
enc celt_256k_stereo     music_48k.wav         -ac 2 -application audio -b:a 256k
enc celt_96k_lowdelay_5ms music_48k.wav        -ac 2 -application lowdelay -b:a 96k -frame_duration 5
enc celt_96k_mono_2_5ms  music_48k.wav         -ac 1 -application lowdelay -b:a 96k -frame_duration 2.5
enc celt_nb_24k_mono_10ms music_48k.wav        -ac 1 -application lowdelay -b:a 24k -cutoff 4000 -frame_duration 10
enc surround_51_192k     tone_51_48k.wav       -ac 6 -mapping_family 1 -b:a 192k
# A 44.1 kHz original: ffmpeg resamples to 48 kHz before libopus, so write
# the original rate into the header the way opusenc would.
enc from_44k1_96k        music_44k.wav         -ac 2 -application audio -b:a 96k
python3 "$here/patch_opus_head.py" "$o/from_44k1_96k.opus" "$o/from_44k1_96k.opus" --rate 44100
# A stream whose header asks for a longer pre-skip than the encoder delay,
# and one with a -6 dB output gain (Q7.8 = -1536).
python3 "$here/patch_opus_head.py" "$o/celt_128k_stereo.opus" "$o/preskip_3840.opus" --pre-skip 3840
python3 "$here/patch_opus_head.py" "$o/celt_64k_stereo.opus" "$o/gain_minus_6db.opus" --gain -1536
for f in "$o"/*.opus; do
  ff -c:a libopus -i "$f" -map_metadata -1 -fflags +bitexact -flags:a +bitexact \
    -c:a flac -sample_fmt s16 "${f%.opus}.ref.flac"
done

# --- Damaged files ------------------------------------------------------------
# Truncations and bit flips of the fixtures above. Made by tests' own
# helper at run time for most cases; these few are committed so the exact
# bytes that once mattered stay under test.
c="$here/corrupt"
python3 - "$a" "$o" "$c" <<'PY'
import sys
a, o, c = sys.argv[1:4]
def cut(src, dst, keep):
    d = open(src, "rb").read()
    open(dst, "wb").write(d[: int(len(d) * keep)])
def flip(src, dst, offsets):
    d = bytearray(open(src, "rb").read())
    for off in offsets:
        d[int(len(d) * off)] ^= 0x5A
    open(dst, "wb").write(bytes(d))
cut(f"{o}/celt_128k_stereo.opus", f"{c}/opus_truncated_half.opus", 0.5)
cut(f"{o}/celt_128k_stereo.opus", f"{c}/opus_header_only.opus", 0.02)
flip(f"{o}/celt_128k_stereo.opus", f"{c}/opus_bitflips.opus", [0.3, 0.5, 0.7])
flip(f"{o}/silk_12k_mono.opus", f"{c}/opus_silk_bitflips.opus", [0.4, 0.6])
cut(f"{a}/tone.flac", f"{c}/flac_truncated.flac", 0.5)
cut(f"{a}/tone.m4a", f"{c}/m4a_truncated.m4a", 0.5)
cut(f"{a}/tone_s16.wav", f"{c}/wav_truncated.wav", 0.5)
cut(f"{a}/tone.mp3", f"{c}/mp3_truncated.mp3", 0.5)
flip(f"{a}/tone.flac", f"{c}/flac_bitflips.flac", [0.3, 0.5, 0.7])
PY
ls -l "$a" "$o" "$c" | awk '{s+=$5} END {print "total bytes:", s}'
