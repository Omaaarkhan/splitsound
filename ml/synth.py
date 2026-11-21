"""Deterministic synthetic test audio (numpy only).

    uv run --locked python synth.py dist/testdata/synth_12s.wav

Produces the 12 s stereo 44.1 kHz float32 clip used by the test vectors: a
seeded mix of sustained tones, chirps, decaying noise bursts and amplitude
modulation. It is not music; it only has to exercise every part of the
pipeline (two full chunks, a short centred last chunk, both branches of the
model) with a signal that has energy across the spectrum.

The noise comes from SplitMix64 applied to the sample index, so it is a pure
function of (seed, index) and trivial to reproduce in any language. The tones
use cos/sin/exp from the platform maths library, so regenerating on another
machine may differ in the last bit of a few samples; the WAV file written
next to the test vectors is the authoritative input.
"""
from __future__ import annotations

import sys

import numpy as np

SAMPLE_RATE = 44100


def splitmix64_uniform(count: int, seed: int) -> np.ndarray:
    """count uniform doubles in [0, 1): SplitMix64 of (seed + index + 1)."""
    with np.errstate(over="ignore"):
        z = (np.arange(1, count + 1, dtype=np.uint64) + np.uint64(seed)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def synth_clip(seconds: float = 12.0, seed: int = 20240607, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return [2, round(seconds * sample_rate)] float32, peak below 1."""
    n = int(round(seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    two_pi = 2.0 * np.pi
    left = np.zeros(n)
    right = np.zeros(n)

    # 1. Bass: 55 Hz with three harmonics, slow amplitude modulation, centred.
    bass = sum(np.sin(two_pi * 55.0 * h * t) / h for h in (1, 2, 3, 4))
    bass *= 0.18 * (0.6 + 0.4 * np.sin(two_pi * 0.5 * t))
    left += bass
    right += bass

    # 2. A chord (A3, C#4, E4) with tremolo, slightly detuned between channels.
    for k, freq in enumerate((220.0, 277.18, 329.63)):
        tremolo = 0.5 + 0.5 * np.sin(two_pi * (3.0 + k) * t)
        left += 0.07 * tremolo * np.sin(two_pi * freq * t)
        right += 0.07 * tremolo * np.sin(two_pi * freq * 1.003 * t + 0.4)

    # 3. Chirps: a 3 s exponential sweep 200 Hz -> 6 kHz, restarting every 3 s,
    #    panned left on even repeats and right on odd ones.
    sweep = 3.0
    local = np.mod(t, sweep)
    ratio = 6000.0 / 200.0
    phase = two_pi * 200.0 * sweep / np.log(ratio) * (np.power(ratio, local / sweep) - 1.0)
    envelope = np.sin(np.pi * local / sweep) ** 2
    chirp = 0.12 * envelope * np.sin(phase)
    odd = (np.floor(t / sweep).astype(np.int64) % 2) == 1
    left += np.where(odd, 0.3, 1.0) * chirp
    right += np.where(odd, 1.0, 0.3) * chirp

    # 4. Noise bursts: every 0.5 s, 120 ms, exponential decay (drum-like).
    #    Independent noise per channel.
    burst_time = np.mod(t, 0.5)
    burst_envelope = np.where(burst_time < 0.12, np.exp(-burst_time / 0.025), 0.0)
    left += 0.30 * burst_envelope * (2.0 * splitmix64_uniform(n, seed) - 1.0)
    right += 0.30 * burst_envelope * (2.0 * splitmix64_uniform(n, seed + 0x5EED) - 1.0)

    # 5. A high tone with fast amplitude modulation, right-heavy.
    high = 0.05 * (0.5 + 0.5 * np.sin(two_pi * 11.0 * t)) * np.sin(two_pi * 3520.0 * t)
    left += 0.4 * high
    right += high

    # 6. Low-level broadband noise floor so no bin is exactly silent.
    left += 0.004 * (2.0 * splitmix64_uniform(n, seed + 1) - 1.0)
    right += 0.004 * (2.0 * splitmix64_uniform(n, seed + 2) - 1.0)

    # Short fades so the clip starts and ends at zero.
    fade = int(0.01 * sample_rate)
    ramp = np.linspace(0.0, 1.0, fade)
    for channel in (left, right):
        channel[:fade] *= ramp
        channel[-fade:] *= ramp[::-1]

    clip = np.stack([left, right]).astype(np.float32)
    assert np.abs(clip).max() < 1.0
    return clip


def main() -> int:
    import soundfile

    out = sys.argv[1] if len(sys.argv) > 1 else "synth_12s.wav"
    clip = synth_clip()
    soundfile.write(out, clip.T, SAMPLE_RATE, subtype="FLOAT")
    print(f"{out}: {clip.shape[1]} samples, peak {np.abs(clip).max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
