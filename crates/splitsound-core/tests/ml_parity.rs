//! Parity with the Python reference, using the vectors in
//! `ml/dist/testdata` (see its README).
//!
//! Three tiers, each skipping cleanly (with a message) when what it needs
//! is absent, so a checkout without models passes:
//! * transforms and the checkpoint converter need only the test vectors;
//! * stems and tagger parity also need the ONNX Runtime library
//!   (`ORT_DYLIB_PATH` or `vendor/onnxruntime/<host>/`) and the models in
//!   `$SPLITSOUND_MODELS_DIR` or `vendor/models`.
//!
//! Run with `--nocapture` to see the measured numbers.

use std::path::{Path, PathBuf};

use splitsound_core::audio::snr_db;
use splitsound_core::checkpoint::{TensorTable, convert_checkpoint};
use splitsound_core::manifest::{Manifest, ModelKind};
use splitsound_core::progress::NoProgress;
use splitsound_core::runtime::{
    Acceleration, Runtime, RuntimeConfig, SessionConfig, resolve_library_path,
};
use splitsound_core::separate::chunk::separate_track;
use splitsound_core::separate::models::open_segment_model;
use splitsound_core::stft::{
    HtdemucsStft, MB_FRAMES, MB_PLANE, MB_SEGMENT, MelbandRoformerStft, PLANE, SC_FRAMES, SC_PLANE,
    SC_SEGMENT, SEGMENT, ScnetStft,
};
use splitsound_core::store::{ModelSource, ModelStore};
use splitsound_core::tagger::Tagger;

fn testdata() -> Option<PathBuf> {
    let dir = std::env::var_os("SPLITSOUND_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ml/dist/testdata"));
    if dir.join("testdata.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "skipped: no test vectors at {} (run ml/make_testdata.py)",
            dir.display()
        );
        None
    }
}

/// Reads a little-endian float32 or complex64 `.npy` (version 1.0, C order)
/// as a flat list of f32 (complex values as re, im pairs).
fn read_npy(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{}", path.display());
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(
        header.contains("'<f4'") || header.contains("'<c8'"),
        "{header}"
    );
    assert!(header.contains("'fortran_order': False"), "{header}");
    bytes[10 + header_len..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn stft_htdemucs_matches_the_reference_vectors() {
    let Some(dir) = testdata() else { return };
    let input = read_npy(&dir.join("stft_htdemucs_input.npy"));
    let spec = read_npy(&dir.join("stft_htdemucs_spec.npy"));
    let roundtrip = read_npy(&dir.join("stft_htdemucs_roundtrip.npy"));
    assert_eq!(input.len(), SEGMENT);
    assert_eq!(spec.len(), 2 * PLANE);
    let expected_re: Vec<f32> = spec.as_chunks::<2>().0.iter().map(|c| c[0]).collect();
    let expected_im: Vec<f32> = spec.as_chunks::<2>().0.iter().map(|c| c[1]).collect();

    let mut st = HtdemucsStft::new();
    let mut re = vec![0f32; PLANE];
    let mut im = vec![0f32; PLANE];
    st.forward(&input, &mut re, &mut im).unwrap();
    let d_re = max_abs_diff(&re, &expected_re);
    let d_im = max_abs_diff(&im, &expected_im);
    // Inverse of the REFERENCE spectrogram, so the two directions are
    // checked independently.
    let mut out = vec![0f32; SEGMENT];
    st.inverse_add(&expected_re, &expected_im, &mut out)
        .unwrap();
    let d_inv = max_abs_diff(&out, &roundtrip);
    println!(
        "htdemucs STFT: max |diff| re {d_re:.2e}, im {d_im:.2e}; inverse {d_inv:.2e}, SNR {:.1} dB",
        snr_db(&roundtrip, &out)
    );
    assert!(d_re < 1e-5 && d_im < 1e-5, "forward: {d_re} {d_im}");
    assert!(d_inv < 1e-5, "inverse: {d_inv}");
}

#[test]
fn stft_scnet_matches_the_reference_vectors() {
    let Some(dir) = testdata() else { return };
    let input = read_npy(&dir.join("stft_scnet_input.npy"));
    let spec = read_npy(&dir.join("stft_scnet_spec.npy"));
    let roundtrip = read_npy(&dir.join("stft_scnet_roundtrip.npy"));
    assert_eq!(input.len(), SC_SEGMENT);
    assert_eq!(spec.len(), 2 * SC_PLANE);
    assert_eq!(SC_PLANE % SC_FRAMES, 0);
    let expected_re: Vec<f32> = spec.as_chunks::<2>().0.iter().map(|c| c[0]).collect();
    let expected_im: Vec<f32> = spec.as_chunks::<2>().0.iter().map(|c| c[1]).collect();

    let mut st = ScnetStft::new();
    let mut re = vec![0f32; SC_PLANE];
    let mut im = vec![0f32; SC_PLANE];
    st.forward(&input, &mut re, &mut im).unwrap();
    let d_re = max_abs_diff(&re, &expected_re);
    let d_im = max_abs_diff(&im, &expected_im);
    let mut out = vec![0f32; SC_SEGMENT];
    st.inverse(&expected_re, &expected_im, &mut out).unwrap();
    let d_inv = max_abs_diff(&out, &roundtrip);
    println!(
        "scnet STFT: max |diff| re {d_re:.2e}, im {d_im:.2e}; inverse {d_inv:.2e}, SNR {:.1} dB",
        snr_db(&roundtrip, &out)
    );
    // The rectangular window lets full-scale content through unattenuated,
    // so float32 rounding in a 4096-point FFT is a little larger here.
    assert!(d_re < 3e-5 && d_im < 3e-5, "forward: {d_re} {d_im}");
    assert!(d_inv < 1e-5, "inverse: {d_inv}");
}

#[test]
fn stft_melband_matches_the_reference_vectors() {
    let Some(dir) = testdata() else { return };
    let input = read_npy(&dir.join("stft_melband_input.npy"));
    let spec = read_npy(&dir.join("stft_melband_spec.npy"));
    let roundtrip = read_npy(&dir.join("stft_melband_roundtrip.npy"));
    assert_eq!(input.len(), MB_SEGMENT);
    assert_eq!(spec.len(), 2 * MB_PLANE);
    assert_eq!(MB_PLANE % MB_FRAMES, 0);
    let expected_re: Vec<f32> = spec.as_chunks::<2>().0.iter().map(|c| c[0]).collect();
    let expected_im: Vec<f32> = spec.as_chunks::<2>().0.iter().map(|c| c[1]).collect();

    let mut st = MelbandRoformerStft::new();
    let mut re = vec![0f32; MB_PLANE];
    let mut im = vec![0f32; MB_PLANE];
    st.forward(&input, &mut re, &mut im).unwrap();
    let d_re = max_abs_diff(&re, &expected_re);
    let d_im = max_abs_diff(&im, &expected_im);
    let mut out = vec![0f32; MB_SEGMENT];
    st.inverse(&expected_re, &expected_im, &mut out).unwrap();
    let d_inv = max_abs_diff(&out, &roundtrip);
    println!(
        "melband STFT: max |diff| re {d_re:.2e}, im {d_im:.2e}; inverse {d_inv:.2e}, SNR {:.1} dB",
        snr_db(&roundtrip, &out)
    );
    assert!(d_re < 3e-5 && d_im < 3e-5, "forward: {d_re} {d_im}");
    assert!(d_inv < 1e-5, "inverse: {d_inv}");
}

#[test]
fn converter_matches_the_reference_and_rejects_what_it_rejects() {
    let Some(dir) = testdata() else { return };
    let conv = dir.join("converter");
    let scratch = std::env::temp_dir().join(format!("splitsound-conv-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let table = |p: &Path| TensorTable::from_json(&std::fs::read_to_string(p).unwrap());

    // The fixture converts to exactly the expected bytes.
    let out = scratch.join("fixture.weights");
    convert_checkpoint(
        &conv.join("fixture.th"),
        &table(&conv.join("fixture.tensors.json")).unwrap(),
        &out,
        &NoProgress,
    )
    .unwrap();
    assert_eq!(
        std::fs::read(&out).unwrap(),
        std::fs::read(conv.join("fixture.expected.weights")).unwrap(),
        "converted weights differ from the reference's"
    );
    std::fs::remove_file(&out).unwrap();

    // Every reject_* input is refused, and nothing is left behind.
    let mut rejected = 0;
    for entry in std::fs::read_dir(&conv).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.starts_with("reject_") || !name.ends_with(".th") {
            continue;
        }
        let table_path = conv.join(format!("{name}.tensors.json"));
        let result = match table(&table_path) {
            Ok(t) => convert_checkpoint(&path, &t, &out, &NoProgress),
            Err(e) => Err(e),
        };
        let err = result
            .err()
            .unwrap_or_else(|| panic!("{name} was accepted"));
        println!("{name}: rejected ({})", err.detail().unwrap_or(""));
        assert!(!out.exists(), "{name} left an output file");
        rejected += 1;
    }
    assert!(rejected >= 10, "only {rejected} reject fixtures found");
    // The right checkpoint with a table that pins another hash.
    let wrong = table(&conv.join("reject_wrong_pin.tensors.json")).unwrap();
    assert!(convert_checkpoint(&conv.join("fixture.th"), &wrong, &out, &NoProgress).is_err());
    assert!(!out.exists());
    assert_eq!(
        std::fs::read_dir(&scratch).unwrap().count(),
        0,
        "temporary files left behind"
    );
    let _ = std::fs::remove_dir_all(&scratch);
}

/// The runtime and the installed models, or `None` (with a message).
fn models() -> Option<(std::sync::Arc<Runtime>, ModelStore)> {
    if let Err(e) = resolve_library_path(None) {
        eprintln!("skipped: {} ({})", e.message(), e.detail().unwrap_or(""));
        return None;
    }
    let dir = std::env::var_os("SPLITSOUND_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/models"));
    if !dir.is_dir() {
        eprintln!("skipped: no models folder at {}", dir.display());
        return None;
    }
    let runtime = match Runtime::load(RuntimeConfig {
        acceleration: Acceleration::Cpu,
        ..Default::default()
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skipped: {} ({})", e.message(), e.detail().unwrap_or(""));
            return None;
        }
    };
    let store =
        ModelStore::with_source(dir, Manifest::builtin().clone(), ModelSource::default()).unwrap();
    Some((runtime, store))
}

#[test]
fn stems_match_the_reference() {
    let Some(dir) = testdata() else { return };
    let Some((runtime, store)) = models() else {
        return;
    };
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("testdata.json")).unwrap()).unwrap();
    let (mix, rate) =
        splitsound_core::wav::read_wav_stereo(&dir.join("synth_12s.wav"), u64::MAX).unwrap();
    assert_eq!(rate, 44_100);
    let mut checked = 0;
    for (model_id, info) in index["stems"].as_object().unwrap() {
        let entry = store.manifest().get(model_id).unwrap().clone();
        if !store.is_installed(model_id) {
            eprintln!("skipped {model_id}: not installed");
            continue;
        }
        if info["modelSha256"].as_str() != Some(entry.file.sha256.as_str()) {
            eprintln!("skipped {model_id}: test vectors were made with a different model file");
            continue;
        }
        let verified = store.load(model_id).unwrap();
        let external = verified
            .external
            .as_ref()
            .map(|(n, b)| (n.as_str(), b.clone()));
        let (session, _) = runtime
            .session_from_verified_with(
                &verified.graph,
                external,
                SessionConfig {
                    allow_gpu: false,
                    memory_pattern: !matches!(entry.kind, ModelKind::MelbandRoformer),
                },
            )
            .unwrap();
        let mut model = open_segment_model(
            entry.kind.separator().unwrap(),
            session,
            entry.sources.len(),
        )
        .unwrap();
        let wanted: Vec<usize> = (0..entry.sources.len()).collect();
        let stride = info["strideSamples"].as_u64().unwrap() as usize;
        assert_eq!(stride, entry.stride_samples);
        let stems =
            separate_track(model.as_mut(), &mix, stride, &wanted, 0, &mut |_, _| Ok(())).unwrap();
        for (source, stem) in entry.sources.iter().zip(&stems) {
            let file = dir.join(info["files"][source]["file"].as_str().unwrap());
            let (reference, _) = splitsound_core::wav::read_wav_stereo(&file, u64::MAX).unwrap();
            assert_eq!(reference.len(), stem.len());
            let snr =
                snr_db(&reference.left, &stem.left).min(snr_db(&reference.right, &stem.right));
            println!("{model_id:<12} {source:<7} SNR {snr:6.1} dB");
            assert!(
                snr > 50.0,
                "{model_id} {source}: only {snr:.1} dB against the reference"
            );
        }
        checked += 1;
    }
    if checked == 0 {
        eprintln!("skipped: none of the reference models are installed");
    }
}

#[test]
fn tagger_matches_the_reference() {
    let Some(dir) = testdata() else { return };
    let Some((runtime, store)) = models() else {
        return;
    };
    let id = splitsound_core::engine::TAGGER_MODEL;
    if !store.is_installed(id) {
        eprintln!("skipped: the tagger is not installed");
        return;
    }
    let verified = store.load(id).unwrap();
    let (session, _) = runtime
        .session_from_verified(&verified.graph, None)
        .unwrap();
    let mut tagger = Tagger::new(session);

    let one = tagger
        .score(&read_npy(&dir.join("tagger_input_32k.npy")), &mut |_, _| {
            Ok(())
        })
        .unwrap();
    assert_eq!(one.windows, 1);
    let d = max_abs_diff(&one.max, &read_npy(&dir.join("tagger_probs.npy")));
    let long = tagger
        .score(
            &read_npy(&dir.join("tagger_long_input_32k.npy")),
            &mut |_, _| Ok(()),
        )
        .unwrap();
    assert_eq!(long.windows, 4);
    let d_max = max_abs_diff(&long.max, &read_npy(&dir.join("tagger_long_max.npy")));
    let d_mean = max_abs_diff(&long.mean, &read_npy(&dir.join("tagger_long_mean.npy")));
    println!(
        "tagger: max |diff| single window {d:.2e}, long max {d_max:.2e}, long mean {d_mean:.2e}"
    );
    assert!(d < 1e-4 && d_max < 1e-4 && d_mean < 1e-4);
}
