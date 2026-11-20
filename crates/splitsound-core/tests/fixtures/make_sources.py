#!/usr/bin/env python3
"""Generates the synthetic source signals for the decoder fixtures.

Pure standard library (no numpy), deterministic. Usage:

    python3 make_sources.py <out_dir>

Writes 16-bit WAV files that `make_fixtures.sh` then encodes with ffmpeg.
The signals are described in README.md; tests rely on the exact recipe of
the `tone*` files (frequencies and amplitudes), so keep them in sync.
"""
import math
import random
import struct
import sys
import wave


def write_wav(path, rate, channels):
    """channels: list of equally long float lists in [-1, 1]."""
    n = len(channels[0])
    with wave.open(path, "wb") as w:
        w.setnchannels(len(channels))
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            for ch in channels:
                v = max(-1.0, min(1.0, ch[i]))
                frames += struct.pack("<h", int(round(v * 32767.0)))
        w.writeframes(bytes(frames))


def fade(sig, rate, ms=10.0):
    k = int(rate * ms / 1000.0)
    n = len(sig)
    for i in range(min(k, n)):
        g = 0.5 - 0.5 * math.cos(math.pi * i / k)
        sig[i] *= g
        sig[n - 1 - i] *= g
    return sig


def tone(freq, amp, rate, seconds):
    n = int(round(rate * seconds))
    return fade([amp * math.sin(2 * math.pi * freq * i / rate) for i in range(n)], rate)


def speech_like(rate, seconds, seed=1):
    """A crude vowel/fricative sequence: a gliding pulse train through three
    formant resonators, syllable-rate amplitude modulation, noise bursts."""
    rnd = random.Random(seed)
    n = int(round(rate * seconds))
    out = [0.0] * n
    # Source: band-limited pulse train with a gliding pitch.
    phase = 0.0
    src = [0.0] * n
    for i in range(n):
        t = i / rate
        f0 = 110.0 + 50.0 * math.sin(2 * math.pi * 0.9 * t) + 15.0 * math.sin(2 * math.pi * 5.1 * t)
        phase += f0 / rate
        if phase >= 1.0:
            phase -= 1.0
            src[i] = 1.0
    # Formants move between two "vowels".
    def resonate(x, f_a, f_b, bw):
        y = [0.0] * n
        y1 = y2 = 0.0
        for i in range(n):
            t = i / rate
            mix = 0.5 + 0.5 * math.sin(2 * math.pi * 1.7 * t)
            f = f_a + (f_b - f_a) * mix
            r = math.exp(-math.pi * bw / rate)
            a1 = 2 * r * math.cos(2 * math.pi * f / rate)
            a2 = -r * r
            v = x[i] + a1 * y1 + a2 * y2
            y2 = y1
            y1 = v
            y[i] = v
        return y
    f1 = resonate(src, 700.0, 300.0, 90.0)
    f2 = resonate(src, 1200.0, 2300.0, 120.0)
    f3 = resonate(src, 2600.0, 3000.0, 180.0)
    for i in range(n):
        t = i / rate
        syll = max(0.0, math.sin(2 * math.pi * 3.3 * t)) ** 0.6
        voiced = (f1[i] + 0.6 * f2[i] + 0.3 * f3[i]) * 0.035 * syll
        # Fricative noise in the gaps between syllables.
        gap = max(0.0, -math.sin(2 * math.pi * 3.3 * t)) ** 2
        noise = (rnd.random() * 2 - 1) * 0.05 * gap
        out[i] = voiced + noise
    peak = max(abs(v) for v in out) or 1.0
    return fade([0.6 * v / peak for v in out], rate)


def music_like(rate, seconds, seed=2):
    """Stereo: a chord with harmonics panned across the field, a bass note,
    kick-like thumps and hi-hat noise bursts."""
    rnd = random.Random(seed)
    n = int(round(rate * seconds))
    left = [0.0] * n
    right = [0.0] * n
    notes = [(261.63, 0.2), (329.63, 0.8), (392.00, 0.5), (523.25, 0.35)]
    for i in range(n):
        t = i / rate
        l = r = 0.0
        for freq, pan in notes:
            v = 0.0
            for h, a in ((1, 1.0), (2, 0.5), (3, 0.3), (5, 0.12), (7, 0.06)):
                v += a * math.sin(2 * math.pi * freq * h * t + 0.3 * h)
            v *= 0.055 * (0.75 + 0.25 * math.sin(2 * math.pi * 2.0 * t + freq))
            l += v * math.cos(pan * math.pi / 2)
            r += v * math.sin(pan * math.pi / 2)
        bass = 0.22 * math.sin(2 * math.pi * 65.41 * t) * (0.6 + 0.4 * math.sin(2 * math.pi * 1.0 * t))
        l += bass
        r += bass
        beat = t % 0.25
        kick = 0.5 * math.exp(-beat * 28.0) * math.sin(2 * math.pi * (50.0 + 90.0 * math.exp(-beat * 40.0)) * beat)
        l += kick
        r += kick
        hbeat = (t + 0.125) % 0.25
        hat = (rnd.random() * 2 - 1) * 0.12 * math.exp(-hbeat * 90.0)
        l += hat * 0.4
        r += hat
        left[i] = l
        right[i] = r
    peak = max(max(abs(v) for v in left), max(abs(v) for v in right)) or 1.0
    g = 0.7 / peak
    return fade([v * g for v in left], rate), fade([v * g for v in right], rate)


def main():
    out = sys.argv[1]
    # Decoder fixtures: left 440 Hz, right 880 Hz, amplitude 0.5.
    write_wav(f"{out}/tone_44k.wav", 44100, [tone(440.0, 0.5, 44100, 1.0), tone(880.0, 0.5, 44100, 1.0)])
    write_wav(f"{out}/tone_44k_half.wav", 44100, [tone(440.0, 0.5, 44100, 0.5), tone(880.0, 0.5, 44100, 0.5)])
    write_wav(f"{out}/tone_48k_half.wav", 48000, [tone(440.0, 0.5, 48000, 0.5), tone(880.0, 0.5, 48000, 0.5)])
    write_wav(f"{out}/tone_48k.wav", 48000, [tone(440.0, 0.5, 48000, 1.0), tone(880.0, 0.5, 48000, 1.0)])
    write_wav(f"{out}/tone_mono_44k.wav", 44100, [tone(440.0, 0.5, 44100, 1.0)])
    # 5.1 in WAV channel order: FL FR FC LFE BL BR.
    write_wav(
        f"{out}/tone_51_44k.wav",
        44100,
        [
            tone(440.0, 0.25, 44100, 0.5),
            tone(880.0, 0.25, 44100, 0.5),
            tone(1320.0, 0.25, 44100, 0.5),
            tone(60.0, 0.25, 44100, 0.5),
            tone(220.0, 0.25, 44100, 0.5),
            tone(330.0, 0.25, 44100, 0.5),
        ],
    )
    # Opus parity sources. Lengths are deliberately not multiples of an Opus
    # frame so the final packet carries end padding to trim.
    write_wav(f"{out}/speech_48k.wav", 48000, [speech_like(48000, 1.2345)])
    sl, sr = music_like(48000, 1.2345)
    write_wav(f"{out}/music_48k.wav", 48000, [sl, sr])
    ml, mr = music_like(44100, 1.2345, seed=3)
    write_wav(f"{out}/music_44k.wav", 44100, [ml, mr])
    sp = speech_like(48000, 1.2345, seed=4)
    sp2 = speech_like(48000, 1.2345, seed=5)
    write_wav(f"{out}/speech_stereo_48k.wav", 48000, [sp, [0.6 * a + 0.4 * b for a, b in zip(sp, sp2)]])
    s51 = [tone(f, 0.2, 48000, 1.0) for f in (440.0, 880.0, 1320.0, 60.0, 220.0, 330.0)]
    write_wav(f"{out}/tone_51_48k.wav", 48000, s51)


if __name__ == "__main__":
    main()
