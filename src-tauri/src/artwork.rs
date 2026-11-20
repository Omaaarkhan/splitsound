//! Cover art handling. Only PNG and JPEG are accepted, identified by their
//! bytes (never by a file name or a MIME type from tags), so the webview
//! only ever receives `data:image/png` or `data:image/jpeg` URLs; SVG or
//! anything else is dropped.

use base64::Engine as _;

/// Largest cover we keep.
pub const MAX_ARTWORK_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    Png,
    Jpeg,
}

impl ImageKind {
    pub fn extension(self) -> &'static str {
        match self {
            ImageKind::Png => "png",
            ImageKind::Jpeg => "jpg",
        }
    }

    pub fn mime(self) -> &'static str {
        match self {
            ImageKind::Png => "image/png",
            ImageKind::Jpeg => "image/jpeg",
        }
    }
}

/// Identifies PNG and JPEG by their signatures.
pub fn sniff(bytes: &[u8]) -> Option<ImageKind> {
    if bytes.len() < 8 || bytes.len() > MAX_ARTWORK_BYTES {
        return None;
    }
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some(ImageKind::Png)
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(ImageKind::Jpeg)
    } else {
        None
    }
}

/// `data:` URL for PNG or JPEG bytes; `None` for anything else.
pub fn data_url(bytes: &[u8]) -> Option<String> {
    let kind = sniff(bytes)?;
    Some(format!(
        "data:{};base64,{}",
        kind.mime(),
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_png_and_jpeg_pass() {
        let png = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0];
        assert_eq!(sniff(&png), Some(ImageKind::Png));
        assert!(
            data_url(&png)
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        let jpg = [0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0];
        assert!(
            data_url(&jpg)
                .unwrap()
                .starts_with("data:image/jpeg;base64,")
        );
        assert_eq!(
            sniff(b"<svg xmlns='http://www.w3.org/2000/svg'><script>"),
            None
        );
        assert_eq!(sniff(b"GIF89a\0\0\0"), None);
        assert_eq!(sniff(&[0xFF, 0xD8]), None);
    }
}
