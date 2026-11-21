//! Writes a file's native decode as raw little-endian f32 to stdout.
//! `cargo run --example dumpnative -- <file> > out.f32`
use std::io::Write;
fn main() {
    let path = std::env::args().nth(1).expect("usage: dumpnative <file>");
    let a = splitsound_core::decode::decode_native(
        std::path::Path::new(&path),
        &splitsound_core::decode::DecodeLimits::default(),
        &splitsound_core::progress::NoProgress,
    )
    .unwrap_or_else(|e| {
        eprintln!("{}: {} ({:?})", e.code(), e, e.detail());
        std::process::exit(1)
    });
    eprintln!(
        "{} Hz, {} ch, {} frames",
        a.sample_rate,
        a.channels,
        a.samples.len() / a.channels
    );
    let mut out = std::io::stdout().lock();
    let bytes: Vec<u8> = a.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    out.write_all(&bytes).unwrap();
}
