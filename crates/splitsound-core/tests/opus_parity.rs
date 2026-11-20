//! Proof that the pure-Rust Opus decoder (`opuscule`) is correct on real
//! files: every fixture is compared, sample aligned, against libopus' own
//! decode of the same file (made by ffmpeg, stored as 16-bit FLAC next to
//! it by `tests/fixtures/make_fixtures.sh`).
//!
//! The set covers SILK (narrowband, mediumband, wideband), hybrid and CELT
//! (narrowband and fullband) modes, mono, stereo and 5.1, 8 kbit/s to
//! 256 kbit/s, 2.5 ms to 60 ms frames including multi-frame packets, a
//! 44.1 kHz original, a non-default pre-skip and a non-zero output gain.
//!
//! Run with `--nocapture` to see the per-file numbers.

mod common;

use common::{fixture, fixtures, ogg_packets};
use splitsound_core::decode::{DecodeLimits, decode_native};
use splitsound_core::progress::NoProgress;

/// Opus mode of a packet from its TOC byte (RFC 6716 section 3.1).
fn mode_of(toc: u8) -> &'static str {
    match toc >> 3 {
        0..=11 => "SILK",
        12..=15 => "hybrid",
        _ => "CELT",
    }
}

struct Expect {
    name: &'static str,
    mode: &'static str,
    channels: usize,
}

const FILES: &[Expect] = &[
    Expect {
        name: "silk_nb_8k_mono",
        mode: "SILK",
        channels: 1,
    },
    Expect {
        name: "silk_mb_10k_mono_60ms",
        mode: "SILK",
        channels: 1,
    },
    Expect {
        name: "silk_12k_mono",
        mode: "SILK",
        channels: 1,
    },
    Expect {
        name: "silk_24k_stereo",
        mode: "SILK",
        channels: 2,
    },
    Expect {
        name: "hybrid_20k_mono_60ms",
        mode: "hybrid",
        channels: 1,
    },
    Expect {
        name: "hybrid_32k_mono",
        mode: "hybrid",
        channels: 1,
    },
    Expect {
        name: "hybrid_48k_stereo",
        mode: "hybrid",
        channels: 2,
    },
    Expect {
        name: "celt_nb_24k_mono_10ms",
        mode: "CELT",
        channels: 1,
    },
    Expect {
        name: "celt_64k_stereo",
        mode: "CELT",
        channels: 2,
    },
    Expect {
        name: "celt_128k_stereo",
        mode: "CELT",
        channels: 2,
    },
    Expect {
        name: "celt_256k_stereo",
        mode: "CELT",
        channels: 2,
    },
    Expect {
        name: "celt_96k_lowdelay_5ms",
        mode: "CELT",
        channels: 2,
    },
    Expect {
        name: "celt_96k_mono_2_5ms",
        mode: "CELT",
        channels: 1,
    },
    Expect {
        name: "from_44k1_96k",
        mode: "CELT",
        channels: 2,
    },
    Expect {
        name: "preskip_3840",
        mode: "CELT",
        channels: 2,
    },
    Expect {
        name: "gain_minus_6db",
        mode: "CELT",
        channels: 2,
    },
    Expect {
        name: "surround_51_192k",
        mode: "CELT",
        channels: 6,
    },
];

#[test]
fn every_fixture_is_listed() {
    let mut on_disk: Vec<String> = std::fs::read_dir(fixtures().join("opus"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".opus"))
        .map(|n| n.trim_end_matches(".opus").to_string())
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = FILES.iter().map(|f| f.name.to_string()).collect();
    listed.sort();
    assert_eq!(on_disk, listed);
}

#[test]
fn matches_libopus_on_every_mode() {
    println!(
        "{:26} {:7} {:>3} {:>7} {:>9} {:>8} {:>9}",
        "file", "mode", "ch", "frames", "SNR dB", "max LSB", "exact %"
    );
    let mut worst = f64::INFINITY;
    for f in FILES {
        let opus_path = fixture(&format!("opus/{}.opus", f.name));
        let ref_path = fixture(&format!("opus/{}.ref.flac", f.name));

        // The file really exercises the mode it is named for.
        let bytes = std::fs::read(&opus_path).unwrap();
        let packets = ogg_packets(&bytes);
        assert!(packets.len() > 3, "{}", f.name);
        let audio_packets = &packets[2..];
        let in_mode = audio_packets
            .iter()
            .filter(|p| !p.is_empty() && mode_of(p[0]) == f.mode)
            .count();
        assert!(
            in_mode * 10 >= audio_packets.len() * 8,
            "{}: only {in_mode} of {} packets are {}",
            f.name,
            audio_packets.len(),
            f.mode
        );

        let ours = decode_native(&opus_path, &DecodeLimits::default(), &NoProgress)
            .unwrap_or_else(|e| panic!("{}: {e} ({:?})", f.name, e.detail()));
        let reference = decode_native(&ref_path, &DecodeLimits::default(), &NoProgress).unwrap();
        assert_eq!(ours.sample_rate, 48_000, "{}", f.name);
        assert_eq!(reference.sample_rate, 48_000, "{}", f.name);
        assert_eq!(ours.channels, f.channels, "{}", f.name);
        assert_eq!(reference.channels, f.channels, "{}", f.name);
        // Pre-skip and end trimming: the length must match libopus exactly.
        assert_eq!(
            ours.samples.len(),
            reference.samples.len(),
            "{}: decoded {} frames, libopus {}",
            f.name,
            ours.samples.len() / f.channels,
            reference.samples.len() / f.channels
        );

        // Opus carries 5.1 in Vorbis order (L C R Ls Rs LFE); ffmpeg wrote
        // the reference FLAC in its own order (L R C LFE Ls Rs). Put ours
        // in the same order before comparing.
        let ours_samples: Vec<f32> = if f.channels == 6 {
            const FROM_VORBIS: [usize; 6] = [0, 2, 1, 5, 3, 4];
            ours.samples
                .as_chunks::<6>()
                .0
                .iter()
                .flat_map(|fr| FROM_VORBIS.map(|i| fr[i]))
                .collect()
        } else {
            ours.samples.clone()
        };

        // The reference is 16-bit, so compare at 16 bits as well as in float.
        let mut max_lsb = 0i32;
        let mut exact = 0usize;
        let (mut signal, mut noise) = (0f64, 0f64);
        for (a, b) in ours_samples.iter().zip(&reference.samples) {
            let qa = (f64::from(*a) * 32768.0).round().clamp(-32768.0, 32767.0) as i32;
            let qb = (f64::from(*b) * 32768.0).round() as i32;
            let d = (qa - qb).abs();
            max_lsb = max_lsb.max(d);
            if d == 0 {
                exact += 1;
            }
            let diff = f64::from(*a) - f64::from(*b);
            signal += f64::from(*b) * f64::from(*b);
            noise += diff * diff;
        }
        let snr = 10.0 * (signal / noise.max(1e-30)).log10();
        worst = worst.min(snr);
        println!(
            "{:26} {:7} {:>3} {:>7} {:>9.1} {:>8} {:>9.2}",
            f.name,
            f.mode,
            f.channels,
            ours.samples.len() / f.channels,
            snr,
            max_lsb,
            100.0 * exact as f64 / ours.samples.len() as f64
        );
        // The float decode can differ from libopus' 16-bit output only by
        // that output's own rounding (half an LSB) plus float noise.
        assert!(
            max_lsb <= 1,
            "{}: differs from libopus by {max_lsb} LSB",
            f.name
        );
        assert!(snr > 60.0, "{}: SNR {snr:.1} dB against libopus", f.name);
    }
    println!("worst SNR against libopus: {worst:.1} dB");
}

#[test]
fn header_fields_are_honoured() {
    let frames = |name: &str| {
        let a = decode_native(
            &fixture(&format!("opus/{name}.opus")),
            &DecodeLimits::default(),
            &NoProgress,
        )
        .unwrap();
        (a.samples.len() / a.channels, a)
    };
    // 1.2345 s of audio: the encoder's 312 frame pre-skip and the padding in
    // the last packet are both removed.
    let (n, _) = frames("celt_128k_stereo");
    assert_eq!(n, 59_256);
    // Same packets, but the header asks to skip 3840 instead of 312.
    let (m, _) = frames("preskip_3840");
    assert_eq!(n - m, 3_840 - 312);
    // -6 dB output gain halves the signal.
    let (_, plain) = frames("celt_64k_stereo");
    let (_, quiet) = frames("gain_minus_6db");
    let rms =
        |v: &[f32]| (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    let ratio = rms(&quiet.samples) / rms(&plain.samples);
    assert!(
        (ratio - 10f64.powf(-6.0 / 20.0)).abs() < 1e-3,
        "gain ratio {ratio}"
    );
    // The original sample rate in the header does not change the decode rate.
    let (_, a) = frames("from_44k1_96k");
    assert_eq!(a.sample_rate, 48_000);
    assert_eq!(a.info.codec, "Opus");
}
