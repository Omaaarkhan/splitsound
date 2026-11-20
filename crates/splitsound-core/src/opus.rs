//! Opus decoding on top of the `opuscule` crate.
//!
//! Symphonia demuxes Ogg and hands this module the identification header
//! (`OpusHead`, RFC 7845 section 5.1) and the raw packets. This module
//! parses the header, sets up a single-stream or multistream decoder,
//! applies the header's output gain, drops the pre-skip at the start and
//! the end padding the demuxer reports, and always produces 48 kHz
//! interleaved float.
//!
//! `opuscule` is safe Rust but young and machine-translated from the C
//! reference, so it is treated as able to fail in any way short of memory
//! corruption: every packet decode is wrapped so that a decoder error *or
//! panic* becomes a clean `decode_failed`. Its output is checked against
//! libopus in `tests/opus_parity.rs`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use opuscule::{Channels as OpusChannels, Decoder, MsDecoder, SampleRate};

use crate::error::{Error, Result};

/// Longest Opus packet duration at 48 kHz: 120 ms.
const MAX_PACKET_FRAMES: usize = 5760;

/// Largest Opus packet accepted. RFC 7845 recommends rejecting packets
/// over about 61 kB per stream; this allows that for 8 streams.
const MAX_PACKET_BYTES: usize = 8 * 61_440;

/// The parsed identification header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    /// Output channel count (1..=255).
    pub channels: usize,
    /// Frames at 48 kHz to discard from the start of the decode.
    pub pre_skip: u16,
    /// Sample rate of the original input, informational only.
    pub input_sample_rate: u32,
    /// Gain to apply when decoding, in Q7.8 dB.
    pub output_gain_q8: i16,
    /// Channel mapping family: 0 mono/stereo, 1 Vorbis-order surround,
    /// 255 unspecified discrete channels.
    pub mapping_family: u8,
    /// Number of Opus streams in each packet.
    pub streams: usize,
    /// How many of those streams are stereo-coupled.
    pub coupled_streams: usize,
    /// For each output channel, the decoded channel that feeds it
    /// (255 = silence).
    pub mapping: Vec<u8>,
}

fn bad_header(detail: &str) -> Error {
    Error::decode("This Opus file has a damaged header").with_detail(detail)
}

impl OpusHead {
    /// Parses an `OpusHead` packet.
    pub fn parse(data: &[u8]) -> Result<OpusHead> {
        if data.len() < 19 || &data[0..8] != b"OpusHead" {
            return Err(bad_header("missing OpusHead"));
        }
        let version = data[8];
        // The major version is the high nibble; only major 0 is defined.
        if version >> 4 != 0 {
            return Err(Error::unsupported(
                "This Opus file uses a newer format version than SplitSound understands",
            ));
        }
        let channels = usize::from(data[9]);
        if channels == 0 {
            return Err(bad_header("zero channels"));
        }
        let pre_skip = u16::from_le_bytes([data[10], data[11]]);
        let input_sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let output_gain_q8 = i16::from_le_bytes([data[16], data[17]]);
        let mapping_family = data[18];
        let (streams, coupled_streams, mapping) = if mapping_family == 0 {
            if channels > 2 {
                return Err(bad_header("family 0 with more than two channels"));
            }
            (
                1,
                channels - 1,
                if channels == 2 { vec![0, 1] } else { vec![0] },
            )
        } else {
            if data.len() < 21 + channels {
                return Err(bad_header("truncated channel mapping"));
            }
            let streams = usize::from(data[19]);
            let coupled = usize::from(data[20]);
            if streams == 0 || coupled > streams || streams + coupled > 255 {
                return Err(bad_header("invalid stream counts"));
            }
            let mapping = data[21..21 + channels].to_vec();
            if mapping
                .iter()
                .any(|m| *m != 255 && usize::from(*m) >= streams + coupled)
            {
                return Err(bad_header("mapping index out of range"));
            }
            if mapping_family == 1 && channels > 8 {
                return Err(bad_header("family 1 with more than eight channels"));
            }
            (streams, coupled, mapping)
        };
        Ok(OpusHead {
            channels,
            pre_skip,
            input_sample_rate,
            output_gain_q8,
            mapping_family,
            streams,
            coupled_streams,
            mapping,
        })
    }
}

enum Inner {
    Single(Box<Decoder>),
    Multi(Box<MsDecoder>, Vec<Decoder>),
}

/// Decoder for one Ogg Opus logical stream.
pub struct OpusStreamDecoder {
    head: OpusHead,
    inner: Inner,
    gain: f32,
    /// Frames still to discard from the start (pre-skip).
    skip: usize,
    pcm: Vec<f32>,
}

fn damaged(detail: impl std::fmt::Display) -> Error {
    Error::decode("This file is damaged and could not be read").with_detail(detail)
}

impl OpusStreamDecoder {
    /// Creates a decoder from the `OpusHead` packet. `max_channels` bounds
    /// the channel count accepted.
    pub fn new(opus_head: &[u8], max_channels: usize) -> Result<Self> {
        let head = OpusHead::parse(opus_head)?;
        if head.channels > max_channels {
            return Err(Error::unsupported(
                "This file has more channels than SplitSound can handle",
            )
            .with_detail(format!("{} channels", head.channels)));
        }
        let inner = if head.mapping_family == 0 {
            let layout = if head.channels == 2 {
                OpusChannels::Stereo
            } else {
                OpusChannels::Mono
            };
            Inner::Single(Box::new(Decoder::new(SampleRate::Hz48000, layout)))
        } else {
            let ms = MsDecoder::new(
                head.channels,
                head.streams,
                head.coupled_streams,
                &head.mapping,
            )
            .map_err(|e| bad_header(&format!("multistream layout rejected: {e:?}")))?;
            let decoders = (0..ms.nb_streams())
                .map(|s| Decoder::new(SampleRate::Hz48000, ms.stream_channels(s)))
                .collect();
            Inner::Multi(Box::new(ms), decoders)
        };
        // Q7.8 dB -> linear.
        let gain = if head.output_gain_q8 == 0 {
            1.0
        } else {
            10f32.powf(f32::from(head.output_gain_q8) / (20.0 * 256.0))
        };
        Ok(OpusStreamDecoder {
            pcm: vec![0.0; MAX_PACKET_FRAMES * head.channels],
            skip: usize::from(head.pre_skip),
            gain,
            inner,
            head,
        })
    }

    /// Output channel count.
    pub fn channels(&self) -> usize {
        self.head.channels
    }

    /// The parsed header.
    pub fn head(&self) -> &OpusHead {
        &self.head
    }

    /// Decodes one packet and appends interleaved 48 kHz samples to `out`.
    ///
    /// `trim_end` is the number of frames the demuxer says to drop from the
    /// end of this packet (end padding, from the final granule position).
    /// Pre-skip is handled here.
    ///
    /// Any decoder error or panic is returned as `decode_failed`.
    pub fn decode_packet(
        &mut self,
        packet: &[u8],
        trim_end: u64,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        if packet.is_empty() {
            return Err(damaged("empty Opus packet"));
        }
        if packet.len() > MAX_PACKET_BYTES {
            return Err(damaged("oversized Opus packet"));
        }
        let channels = self.head.channels;
        let pcm = &mut self.pcm;
        let inner = &mut self.inner;
        let decoded = catch_unwind(AssertUnwindSafe(|| match inner {
            Inner::Single(d) => d.decode(Some(packet), pcm, false),
            Inner::Multi(ms, decoders) => ms.decode(decoders, Some(packet), pcm, false),
        }));
        let frames = match decoded {
            Ok(Ok(frames)) => frames,
            Ok(Err(e)) => return Err(damaged(format!("opus: {e:?}"))),
            Err(_) => return Err(damaged("opus decoder panicked")),
        };
        if frames > MAX_PACKET_FRAMES {
            return Err(damaged("opus decoder returned an impossible frame count"));
        }
        let mut start = 0usize;
        if self.skip > 0 {
            start = self.skip.min(frames);
            self.skip -= start;
        }
        let keep_end = frames.saturating_sub(usize::try_from(trim_end).unwrap_or(usize::MAX));
        if keep_end <= start {
            return Ok(());
        }
        let slice = &self.pcm[start * channels..keep_end * channels];
        if self.gain == 1.0 {
            out.extend_from_slice(slice);
        } else {
            out.extend(slice.iter().map(|s| s * self.gain));
        }
        Ok(())
    }

    /// Per-output-channel gains into (left, right) for a stereo downmix.
    ///
    /// Family 1 uses the Vorbis channel order (RFC 7845 section 5.1.1.2);
    /// the centre and surrounds are mixed in at -3 dB, the LFE is dropped,
    /// and the result is normalised so it cannot clip. Other layouts fall
    /// back to "first channel left, second right, the rest to both".
    pub fn downmix_matrix(&self) -> Vec<(f32, f32)> {
        const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
        const L: (f32, f32) = (1.0, 0.0);
        const R: (f32, f32) = (0.0, 1.0);
        const C: (f32, f32) = (H, H);
        const SL: (f32, f32) = (H, 0.0);
        const SR: (f32, f32) = (0.0, H);
        const RC: (f32, f32) = (0.5, 0.5);
        const LFE: (f32, f32) = (0.0, 0.0);
        let n = self.head.channels;
        let mut gains: Vec<(f32, f32)> = match (self.head.mapping_family, n) {
            (_, 1) => return vec![(1.0, 1.0)],
            (_, 2) => return vec![L, R],
            (1, 3) => vec![L, C, R],
            (1, 4) => vec![L, R, SL, SR],
            (1, 5) => vec![L, C, R, SL, SR],
            (1, 6) => vec![L, C, R, SL, SR, LFE],
            (1, 7) => vec![L, C, R, SL, SR, RC, LFE],
            (1, 8) => vec![L, C, R, SL, SR, SL, SR, LFE],
            _ => (0..n)
                .map(|i| match i {
                    0 => L,
                    1 => R,
                    _ => C,
                })
                .collect(),
        };
        let sum_l: f32 = gains.iter().map(|g| g.0).sum();
        let sum_r: f32 = gains.iter().map(|g| g.1).sum();
        let norm = 1.0 / sum_l.max(sum_r).max(1.0);
        for g in gains.iter_mut() {
            g.0 *= norm;
            g.1 *= norm;
        }
        gains
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(channels: u8, pre_skip: u16, gain: i16, family: u8, tail: &[u8]) -> Vec<u8> {
        let mut v = b"OpusHead".to_vec();
        v.push(1);
        v.push(channels);
        v.extend_from_slice(&pre_skip.to_le_bytes());
        v.extend_from_slice(&48_000u32.to_le_bytes());
        v.extend_from_slice(&gain.to_le_bytes());
        v.push(family);
        v.extend_from_slice(tail);
        v
    }

    #[test]
    fn parses_stereo_header() {
        let h = OpusHead::parse(&head(2, 312, 0, 0, &[])).unwrap();
        assert_eq!(h.channels, 2);
        assert_eq!(h.pre_skip, 312);
        assert_eq!(h.streams, 1);
        assert_eq!(h.coupled_streams, 1);
        assert_eq!(h.mapping, vec![0, 1]);
    }

    #[test]
    fn parses_surround_header() {
        // 5.1: 4 streams, 2 coupled.
        let h = OpusHead::parse(&head(6, 312, 0, 1, &[4, 2, 0, 4, 1, 2, 3, 5])).unwrap();
        assert_eq!(h.streams, 4);
        assert_eq!(h.coupled_streams, 2);
        assert_eq!(h.mapping, vec![0, 4, 1, 2, 3, 5]);
        let d = OpusStreamDecoder::new(&head(6, 312, 0, 1, &[4, 2, 0, 4, 1, 2, 3, 5]), 8).unwrap();
        let m = d.downmix_matrix();
        assert_eq!(m.len(), 6);
        assert_eq!(m[5], (0.0, 0.0));
        assert!(m.iter().map(|g| g.0).sum::<f32>() <= 1.0 + 1e-6);
    }

    #[test]
    fn rejects_bad_headers() {
        let cases: Vec<Vec<u8>> = vec![
            vec![],
            b"OpusHead".to_vec(),
            b"NotOpus!xxxxxxxxxxxxxxxx".to_vec(),
            head(0, 0, 0, 0, &[]),
            head(3, 0, 0, 0, &[]),
            // Family 1 but the mapping table is cut short.
            head(6, 0, 0, 1, &[4, 2, 0, 4]),
            // Coupled > streams.
            head(2, 0, 0, 1, &[1, 2, 0, 1]),
            // Mapping index out of range.
            head(2, 0, 0, 1, &[1, 1, 0, 9]),
            // Zero streams.
            head(2, 0, 0, 1, &[0, 0, 0, 1]),
        ];
        for c in cases {
            assert!(OpusHead::parse(&c).is_err(), "{c:?}");
        }
        // Unknown major version.
        let mut v = head(2, 0, 0, 0, &[]);
        v[8] = 0x10;
        assert_eq!(
            OpusHead::parse(&v).unwrap_err().code().as_str(),
            "unsupported_format"
        );
        // Channel cap.
        let e = OpusStreamDecoder::new(&head(6, 0, 0, 1, &[4, 2, 0, 4, 1, 2, 3, 5]), 2);
        assert!(e.is_err());
    }

    #[test]
    fn garbage_packets_are_errors_not_panics() {
        let mut d = OpusStreamDecoder::new(&head(2, 0, 0, 0, &[]), 8).unwrap();
        let mut out = Vec::new();
        assert!(d.decode_packet(&[], 0, &mut out).is_err());
        // Random bytes: each call must return (Ok or Err), never unwind.
        let mut seed = 0x1234_5678_9abc_def0u64;
        for len in [1usize, 2, 3, 7, 50, 400, 1275, 4000] {
            for _ in 0..200 {
                let p: Vec<u8> = (0..len)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        seed as u8
                    })
                    .collect();
                let _ = d.decode_packet(&p, 0, &mut out);
            }
        }
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
