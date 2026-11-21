//! The model manifest: which models exist, where each comes from, and the
//! size and SHA-256 every file must have.
//!
//! The manifest is compiled into the binary (`models/manifest.json`, a copy
//! of `ml/dist/manifest.json`). It is never fetched, so a network attacker
//! cannot change what the engine considers a valid model.

use std::sync::OnceLock;

use crate::checkpoint::TensorTable;
use crate::error::{Error, ErrorCode, Result};
use crate::separate::models::SeparatorKind;

/// A file pinned by name, size and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedFile {
    /// File name (no directories).
    pub name: String,
    /// SHA-256, lowercase hex.
    pub sha256: String,
    /// Size in bytes.
    pub size_bytes: u64,
}

/// A file fetched from its original publisher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamFile {
    /// HTTPS URL on an allow-listed host.
    pub url: String,
    /// SHA-256, lowercase hex.
    pub sha256: String,
    /// Size in bytes.
    pub size_bytes: u64,
}

/// Weights that the engine builds itself from an upstream checkpoint
/// instead of downloading them from the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedWeights {
    /// The weights file the conversion produces (the ONNX graph's external
    /// data file).
    pub file: PinnedFile,
    /// The original publisher's checkpoint.
    pub upstream: UpstreamFile,
    /// SHA-256 of the tensor table compiled into the engine.
    pub tensor_table_sha256: String,
}

/// What a model is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// SCNet separator.
    Scnet,
    /// HTDemucs separator.
    Htdemucs,
    /// Mel-Band RoFormer vocal separator.
    MelbandRoformer,
    /// AudioSet instrument tagger.
    Tagger,
}

impl ModelKind {
    /// The separator transform for this kind, if it is a separator.
    pub fn separator(self) -> Option<SeparatorKind> {
        match self {
            ModelKind::Scnet => Some(SeparatorKind::Scnet),
            ModelKind::Htdemucs => Some(SeparatorKind::Htdemucs),
            ModelKind::MelbandRoformer => Some(SeparatorKind::MelbandRoformer),
            ModelKind::Tagger => None,
        }
    }
}

/// One model, as the engine uses it (see `RawModel` for the file's shape).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelEntry {
    /// Stable id: `scnet_small`, `htdemucs_6s`, ...
    pub id: String,
    /// Model family.
    pub kind: ModelKind,
    /// Display name.
    pub name: String,
    /// One plain sentence on what it is for.
    pub description: String,
    /// Licence of the weights.
    pub license: String,
    /// Who made it.
    pub attribution: String,
    /// Who the download comes from, for display: "GitHub" or "Meta".
    pub source: String,
    /// True when the weights are not under an open-source licence. Such a
    /// model is never fetched without the explicit accept flag.
    pub restricted: bool,
    /// Plain-language terms to show before a restricted model is fetched.
    pub terms: Option<String>,
    /// Output order of the graph (separators only).
    pub sources: Vec<String>,
    /// The sources this model is trusted for.
    pub targets: Vec<String>,
    /// Samples per model call (separators only).
    pub segment_samples: usize,
    /// Default samples between chunk starts (separators only).
    pub stride_samples: usize,
    /// Whether section 1.1 track normalisation runs around separation. The
    /// engine takes this from the model kind; validation only checks that
    /// the manifest agrees with it.
    pub normalize: bool,
    /// The ONNX graph as installed.
    pub file: PinnedFile,
    /// The zstd-compressed graph in the project's models release. `None`
    /// means the model cannot be downloaded yet, only imported by hand.
    pub download: Option<PinnedFile>,
    /// Set when the weights are converted on the device.
    pub weights: Option<ConvertedWeights>,
    /// Pinned digests of the asset files compiled into the engine for this
    /// model (the tagger's label list and instrument map): (file name,
    /// SHA-256).
    pub asset_digests: Vec<(String, String)>,
}

impl ModelEntry {
    /// Bytes on disk once installed.
    pub fn installed_bytes(&self) -> u64 {
        self.file.size_bytes + self.weights.as_ref().map_or(0, |w| w.file.size_bytes)
    }

    /// Bytes that are downloaded to install it.
    pub fn download_bytes(&self) -> u64 {
        self.download.as_ref().map_or(0, |d| d.size_bytes)
            + self.weights.as_ref().map_or(0, |w| w.upstream.size_bytes)
    }

    /// Index of `source` in the graph's output order.
    pub fn source_index(&self, source: &str) -> Option<usize> {
        self.sources.iter().position(|s| s == source)
    }
}

/// The whole manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// Format version of this file; 2.
    pub schema_version: u32,
    /// The models.
    pub models: Vec<ModelEntry>,
}

/// Largest model file the engine will ever accept, whatever a manifest
/// says: a backstop against a typo turning into a 1 TB download.
pub const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Hosts an upstream checkpoint may be fetched from.
pub const UPSTREAM_HOSTS: &[&str] = &["dl.fbaipublicfiles.com"];

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A file name that is safe to join onto the models folder: one plain
/// component, conservative characters, not hidden.
pub fn is_safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 100
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        && !name.contains("..")
}

/// Splits an `https://host/path` URL into host and path. Anything else
/// (other schemes, credentials, ports, queries with odd characters) is
/// rejected.
pub fn parse_https_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("https://")?;
    let (host, path) = rest.split_once('/')?;
    let host_ok = !host.is_empty()
        && host.len() <= 253
        && host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
        && !host.starts_with('.')
        && !host.ends_with('.');
    let path_ok = path
        .bytes()
        .all(|b| b.is_ascii_graphic() && !matches!(b, b'\\' | b'"' | b'<' | b'>' | b'#'));
    (host_ok && path_ok).then_some((host, path))
}

/// A pinned file as the manifest writes it: `file`, `sha256`, `sizeBytes`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawFile {
    file: String,
    sha256: String,
    /// Present in the file; the compiled-in assets are checked by digest.
    #[allow(dead_code)]
    size_bytes: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawAcquire {
    #[serde(rename = "type")]
    kind: String,
    file: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawUpstream {
    url: String,
    sha256: String,
    size_bytes: u64,
    #[serde(default)]
    format: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawWeights {
    file: String,
    sha256: String,
    size_bytes: u64,
    tensors: RawFile,
    upstream: RawUpstream,
}

/// One model exactly as `ml/dist/manifest.json` (schemaVersion 2,
/// `ml/CONTRACT.md` section 7) writes it. Every documented field is listed
/// and unknown ones are an error, so a change of shape is noticed.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawModel {
    id: String,
    kind: ModelKind,
    #[allow(dead_code)]
    revision: u32,
    name: String,
    description: String,
    source: String,
    restricted: bool,
    terms: Option<String>,
    license: String,
    attribution: String,
    #[allow(dead_code)]
    origin: String,
    file: String,
    sha256: String,
    size_bytes: u64,
    acquire: RawAcquire,
    #[allow(dead_code)]
    opset: u32,
    #[serde(default)]
    sources: Vec<String>,
    #[serde(default)]
    targets: Vec<String>,
    sample_rate: u32,
    channels: u32,
    #[serde(default)]
    segment_samples: usize,
    #[serde(default)]
    stride_samples: usize,
    #[serde(default)]
    #[allow(dead_code)]
    nfft: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    hop: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    bins: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    frames: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    window: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    mel_bands: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    win_length: Option<u32>,
    #[serde(default)]
    normalize: Option<bool>,
    #[allow(dead_code)]
    dynamic_batch: bool,
    #[serde(default)]
    #[allow(dead_code)]
    weight_storage: Option<String>,
    #[serde(default)]
    weights: Option<RawWeights>,
    #[serde(default)]
    window_samples: Option<u64>,
    #[serde(default)]
    hop_samples: Option<u64>,
    #[serde(default)]
    num_classes: Option<u64>,
    #[serde(default)]
    labels: Option<RawFile>,
    #[serde(default)]
    instrument_map: Option<RawFile>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawManifest {
    schema_version: u32,
    models: Vec<RawModel>,
}

impl RawModel {
    fn into_entry(self) -> Result<ModelEntry> {
        let fail = |what: &str| {
            Err(Error::internal(format!(
                "model manifest: {}: {what}",
                self.id
            )))
        };
        if self.acquire.kind != "download" {
            return fail("unknown acquire type");
        }
        let expected_rate = if self.kind == ModelKind::Tagger {
            32_000
        } else {
            44_100
        };
        let expected_channels = if self.kind == ModelKind::Tagger { 1 } else { 2 };
        if self.sample_rate != expected_rate || self.channels != expected_channels {
            return fail("unexpected sample rate or channel count");
        }
        if self.kind == ModelKind::Tagger
            && (self.window_samples != Some(crate::tagger::WINDOW as u64)
                || self.hop_samples != Some(crate::tagger::HOP as u64)
                || self.num_classes != Some(crate::tagger::NUM_CLASSES as u64))
        {
            return fail("tagger constants differ from the engine's");
        }
        if let Some(w) = &self.weights
            && w.upstream.format.as_deref() != Some("torch-zip")
        {
            return fail("unknown upstream format");
        }
        let normalize = self.normalize.unwrap_or(true);
        let expected_normalize = self.kind != ModelKind::MelbandRoformer;
        if self.kind != ModelKind::Tagger && normalize != expected_normalize {
            return fail("normalize setting differs from the engine's");
        }
        Ok(ModelEntry {
            file: PinnedFile {
                name: self.file,
                sha256: self.sha256,
                size_bytes: self.size_bytes,
            },
            download: Some(PinnedFile {
                name: self.acquire.file,
                sha256: self.acquire.sha256,
                size_bytes: self.acquire.size_bytes,
            }),
            weights: self.weights.map(|w| ConvertedWeights {
                file: PinnedFile {
                    name: w.file,
                    sha256: w.sha256,
                    size_bytes: w.size_bytes,
                },
                upstream: UpstreamFile {
                    url: w.upstream.url,
                    sha256: w.upstream.sha256,
                    size_bytes: w.upstream.size_bytes,
                },
                tensor_table_sha256: w.tensors.sha256,
            }),
            asset_digests: [self.labels, self.instrument_map]
                .into_iter()
                .flatten()
                .map(|f| (f.file, f.sha256))
                .collect(),
            id: self.id,
            kind: self.kind,
            name: self.name,
            description: self.description,
            license: self.license,
            attribution: self.attribution,
            source: self.source,
            restricted: self.restricted,
            terms: self.terms,
            sources: self.sources,
            targets: self.targets,
            segment_samples: self.segment_samples,
            stride_samples: self.stride_samples,
            normalize,
        })
    }
}

impl Manifest {
    /// Parses and validates a manifest. Every rule a file must obey is
    /// checked here once, so the rest of the engine can trust the entries.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: RawManifest = serde_json::from_str(json)
            .map_err(|e| Error::internal(format!("model manifest: {e}")))?;
        let mut models = Vec::with_capacity(raw.models.len());
        for model in raw.models {
            models.push(model.into_entry()?);
        }
        let manifest = Manifest {
            schema_version: raw.schema_version,
            models,
        };
        let fail = |what: String| Err(Error::internal(format!("model manifest: {what}")));
        if manifest.schema_version != 2 {
            return fail(format!(
                "unsupported schemaVersion {}",
                manifest.schema_version
            ));
        }
        let mut names: Vec<&str> = Vec::new();
        let mut ids: Vec<&str> = Vec::new();
        for m in &manifest.models {
            if !is_safe_file_name(&m.id) || m.id.contains('.') || ids.contains(&m.id.as_str()) {
                return fail(format!("bad or duplicate id {:?}", m.id));
            }
            ids.push(&m.id);
            let mut files: Vec<&PinnedFile> = vec![&m.file];
            files.extend(m.download.as_ref());
            files.extend(m.weights.as_ref().map(|w| &w.file));
            for f in files {
                if !is_safe_file_name(&f.name)
                    || !is_sha256(&f.sha256)
                    || f.size_bytes == 0
                    || f.size_bytes > MAX_MODEL_BYTES
                    || names.contains(&f.name.as_str())
                {
                    return fail(format!("{}: bad file entry {:?}", m.id, f.name));
                }
                names.push(&f.name);
            }
            if let Some(w) = &m.weights {
                let host_ok = parse_https_url(&w.upstream.url)
                    .is_some_and(|(host, _)| UPSTREAM_HOSTS.contains(&host));
                if !host_ok
                    || !is_sha256(&w.upstream.sha256)
                    || !is_sha256(&w.tensor_table_sha256)
                    || w.upstream.size_bytes == 0
                    || w.upstream.size_bytes > MAX_MODEL_BYTES
                {
                    return fail(format!("{}: bad upstream entry", m.id));
                }
            }
            if m.restricted && m.terms.as_deref().is_none_or(|t| t.trim().is_empty()) {
                return fail(format!("{}: restricted model without terms", m.id));
            }
            if m.kind.separator().is_some() {
                let targets_ok = m.targets.iter().all(|t| m.sources.contains(t));
                if m.sources.is_empty()
                    || m.sources.len() > 16
                    || !targets_ok
                    || m.segment_samples == 0
                    || m.segment_samples % 2 != 0
                    || m.stride_samples == 0
                    || m.stride_samples > m.segment_samples
                {
                    return fail(format!("{}: bad separator parameters", m.id));
                }
            }
        }
        Ok(manifest)
    }

    /// The manifest compiled into the engine.
    pub fn builtin() -> &'static Manifest {
        static MANIFEST: OnceLock<Manifest> = OnceLock::new();
        MANIFEST.get_or_init(|| {
            // The embedded file is checked by the test suite; a failure
            // here would be a build defect, not a runtime condition.
            match Manifest::from_json(include_str!("../models/manifest.json")) {
                Ok(m) => m,
                Err(_) => Manifest {
                    schema_version: 2,
                    models: Vec::new(),
                },
            }
        })
    }

    /// Looks a model up by id.
    pub fn get(&self, id: &str) -> Result<&ModelEntry> {
        self.models.iter().find(|m| m.id == id).ok_or_else(|| {
            Error::new(
                ErrorCode::NotFound,
                format!("There is no model called \u{201c}{id}\u{201d}"),
            )
        })
    }

    /// Looks a model up by id, `None` if the manifest does not list it.
    pub fn find(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id)
    }
}

/// The tensor table for the model whose weights are converted on the
/// device, compiled into the engine.
pub fn builtin_tensor_table_json(model_id: &str) -> Option<&'static str> {
    match model_id {
        "htdemucs_6s" => Some(include_str!("../models/htdemucs_6s.tensors.json")),
        _ => None,
    }
}

/// Parses the compiled-in tensor table for `entry` and checks it against
/// the digest and pins in the manifest.
pub fn tensor_table_for(entry: &ModelEntry) -> Result<TensorTable> {
    use sha2::{Digest, Sha256};
    let weights = entry
        .weights
        .as_ref()
        .ok_or_else(|| Error::internal("model has no converted weights"))?;
    let json = builtin_tensor_table_json(&entry.id)
        .ok_or_else(|| Error::internal("no tensor table for this model"))?;
    if crate::checkpoint::hex(&Sha256::digest(json.as_bytes())) != weights.tensor_table_sha256 {
        return Err(Error::internal("tensor table does not match the manifest"));
    }
    let table = TensorTable::from_json(json)?;
    if table.upstream.sha256 != weights.upstream.sha256
        || table.upstream.size_bytes != weights.upstream.size_bytes
        || table.weights.sha256 != weights.file.sha256
        || table.weights.size_bytes != weights.file.size_bytes
    {
        return Err(Error::internal("tensor table and manifest disagree"));
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /// A minimal manifest in the shape of `ml/dist/manifest.json`.
    fn entry(extra: &str) -> String {
        format!(
            r#"{{"schemaVersion":2,"models":[{{"id":"m","kind":"scnet","revision":1,"name":"M","description":"d",
            "source":"GitHub","restricted":false,"terms":null,"license":"MIT","attribution":"a",
            "origin":"https://example.org","file":"m.onnx","sha256":"{SHA}","sizeBytes":10,
            "acquire":{{"type":"download","file":"m.onnx.zst","sha256":"{SHA}","sizeBytes":5}},"opset":17,
            "sources":["drums","vocals"],"targets":["vocals"],"sampleRate":44100,"channels":2,
            "segmentSamples":100,"strideSamples":75,"dynamicBatch":true{extra}}}]}}"#
        )
    }

    #[test]
    fn builtin_manifest_is_valid_and_consistent() {
        let m = Manifest::from_json(include_str!("../models/manifest.json")).unwrap();
        assert!(!m.models.is_empty());
        assert_eq!(Manifest::builtin(), &m);
        for model in &m.models {
            if model.weights.is_some() {
                tensor_table_for(model).unwrap();
                assert!(
                    model.restricted,
                    "{}: converted models are the restricted ones",
                    model.id
                );
            }
            if model.restricted {
                assert!(model.terms.is_some());
            }
        }
        // The assets compiled into the engine are the ones the manifest pins.
        use sha2::{Digest, Sha256};
        let tagger = m.get("efficientat_mn10_as").unwrap();
        assert_eq!(tagger.asset_digests.len(), 2);
        for (file, digest) in &tagger.asset_digests {
            let bytes: &[u8] = match file.as_str() {
                "audioset_labels.json" => include_bytes!("../models/audioset_labels.json"),
                "instrument_map.json" => include_bytes!("../models/instrument_map.json"),
                other => panic!("unexpected asset {other}"),
            };
            assert_eq!(
                &crate::checkpoint::hex(&Sha256::digest(bytes)),
                digest,
                "{file}"
            );
        }
        // Every model can be downloaded.
        assert!(m.models.iter().all(|x| x.download.is_some()));
        // The pipelines need these.
        for id in [
            "scnet_small",
            "scnet_large",
            "melband_roformer_vocals",
            "htdemucs_6s",
            "efficientat_mn10_as",
        ] {
            m.get(id).unwrap();
        }
        assert_eq!(m.get("nope").unwrap_err().code(), ErrorCode::NotFound);
        assert!(m.get("scnet_large").unwrap().normalize);
        assert!(!m.get("melband_roformer_vocals").unwrap().normalize);
    }

    #[test]
    fn valid_minimal_manifest() {
        let m = Manifest::from_json(&entry("")).unwrap();
        assert_eq!(m.models[0].installed_bytes(), 10);
        assert_eq!(m.models[0].download_bytes(), 5);
        assert_eq!(m.models[0].source_index("vocals"), Some(1));
    }

    #[test]
    fn invalid_manifests_are_rejected() {
        let ok = entry("");
        let cases = [
            ok.replace("\"schemaVersion\":2", "\"schemaVersion\":1"),
            // Path tricks in names.
            ok.replace("m.onnx", "../m.onnx"),
            ok.replace("m.onnx", "a/m.onnx"),
            ok.replace("m.onnx", ".hidden"),
            ok.replace("m.onnx", "C:\\\\m.onnx"),
            ok.replace("m.onnx", ""),
            // Digests and sizes.
            ok.replace(SHA, "abc"),
            ok.replace(SHA, &SHA.to_uppercase()),
            ok.replace("\"sizeBytes\":10", "\"sizeBytes\":0"),
            ok.replace("\"sizeBytes\":10", "\"sizeBytes\":99999999999999"),
            // Separator parameters.
            ok.replace("\"strideSamples\":75", "\"strideSamples\":0"),
            ok.replace("\"strideSamples\":75", "\"strideSamples\":101"),
            ok.replace("\"segmentSamples\":100", "\"segmentSamples\":101"),
            ok.replace("\"targets\":[\"vocals\"]", "\"targets\":[\"kazoo\"]"),
            // Unknown fields are a mistake, not something to ignore.
            ok.replace(
                "\"restricted\":false",
                "\"restricted\":false,\"url\":\"http://evil\"",
            ),
            // Restricted without terms.
            ok.replace("\"restricted\":false", "\"restricted\":true"),
            // The same file name twice.
            ok.replace("m.onnx.zst", "m.onnx"),
            // Shape rules of the official manifest.
            ok.replace("\"type\":\"download\"", "\"type\":\"execute\""),
            ok.replace("\"sampleRate\":44100", "\"sampleRate\":48000"),
            ok.replace("\"channels\":2", "\"channels\":6"),
            ok.replace("\"opset\":17,", ""),
            ok.replace(
                "\"dynamicBatch\":true",
                "\"dynamicBatch\":true,\"normalize\":false",
            ),
            ok.replace("\"kind\":\"scnet\"", "\"kind\":\"melband_roformer\""),
        ];
        for (i, c) in cases.iter().enumerate() {
            assert!(Manifest::from_json(c).is_err(), "case {i} accepted: {c}");
        }
    }

    #[test]
    fn upstream_urls_are_allow_listed() {
        let with_url = |url: &str| {
            entry(&format!(
                r#","weights":{{"file":"m.weights","sha256":"{SHA}","sizeBytes":8,
                "tensors":{{"file":"m.tensors.json","sha256":"{SHA}","sizeBytes":3}},
                "upstream":{{"url":"{url}","sha256":"{SHA}","sizeBytes":4,"format":"torch-zip"}}}}"#
            ))
        };
        assert!(
            Manifest::from_json(&with_url("https://dl.fbaipublicfiles.com/demucs/x.th")).is_ok()
        );
        for bad in [
            "http://dl.fbaipublicfiles.com/demucs/x.th",
            "https://evil.example/demucs/x.th",
            "https://dl.fbaipublicfiles.com.evil.example/x.th",
            "https://user@dl.fbaipublicfiles.com/x.th",
            "https://dl.fbaipublicfiles.com:8443/x.th",
            "https://dl.fbaipublicfiles.com",
            "file:///etc/passwd",
            "ftp://dl.fbaipublicfiles.com/x.th",
        ] {
            assert!(Manifest::from_json(&with_url(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn url_and_name_helpers() {
        assert_eq!(
            parse_https_url("https://github.com/a/b/releases/download/v1/x.zst"),
            Some(("github.com", "a/b/releases/download/v1/x.zst"))
        );
        assert_eq!(parse_https_url("https://github.com"), None);
        assert_eq!(parse_https_url("http://github.com/x"), None);
        assert_eq!(parse_https_url("https://git hub.com/x"), None);
        assert_eq!(parse_https_url("https://github.com/x y"), None);
        assert!(is_safe_file_name("scnet_small.onnx.zst"));
        for bad in [
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            "a..b",
            ".env",
            "nul\0",
            "über.onnx",
        ] {
            assert!(!is_safe_file_name(bad), "{bad:?}");
        }
    }
}
