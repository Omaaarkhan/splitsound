//! Cover art handling. Only PNG and JPEG are accepted, identified by their
//! bytes (never by a file name or a MIME type from tags), so the webview
//! only ever receives `data:image/png` or `data:image/jpeg` URLs; SVG or
//! anything else is dropped.

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};

use base64::Engine as _;
use image::GenericImageView;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;

/// Largest cover we keep.
pub const MAX_ARTWORK_BYTES: usize = 16 * 1024 * 1024;
const THUMB_MAX_DIM: u32 = 128;
const MAX_SOURCE_DIM: u32 = 8192;
const THUMB_JPEG_QUALITY: u8 = 82;

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

/// Decodes a verified PNG or JPEG cover and produces a `<= 128x128` JPEG
/// thumbnail (quality 82), bounded by dimension/allocation limits and a panic
/// boundary. Returns `None` if invalid, oversized, or decoding fails.
pub fn make_thumbnail(bytes: &[u8]) -> Option<Vec<u8>> {
    let kind = sniff(bytes)?;
    let fmt = match kind {
        ImageKind::Png => image::ImageFormat::Png,
        ImageKind::Jpeg => image::ImageFormat::Jpeg,
    };
    catch_unwind(AssertUnwindSafe(|| {
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_SOURCE_DIM);
        limits.max_image_height = Some(MAX_SOURCE_DIM);
        limits.max_alloc = Some(64 * 1024 * 1024);
        let mut reader = image::ImageReader::with_format(Cursor::new(bytes), fmt);
        reader.limits(limits);
        let img = reader.decode().ok()?;
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 || w > MAX_SOURCE_DIM || h > MAX_SOURCE_DIM {
            return None;
        }
        let thumb = if w > THUMB_MAX_DIM || h > THUMB_MAX_DIM {
            img.resize(THUMB_MAX_DIM, THUMB_MAX_DIM, FilterType::Triangle)
        } else {
            img
        };
        let rgb = thumb.into_rgb8();
        let mut out = Vec::with_capacity(8192);
        let mut enc = JpegEncoder::new_with_quality(&mut out, THUMB_JPEG_QUALITY);
        enc.encode(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )
        .ok()?;
        Some(out)
    }))
    .ok()
    .flatten()
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

    #[test]
    fn thumbnail_resizes_valid_png_and_rejects_bad_bytes() {
        assert_eq!(make_thumbnail(b"<svg></svg>"), None);
        assert_eq!(
            make_thumbnail(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
            None
        );

        let img = image::RgbImage::from_fn(256, 128, |x, y| {
            image::Rgb([(x % 255) as u8, (y % 255) as u8, 120])
        });
        let mut png_bytes = Vec::new();
        use image::ImageEncoder as _;
        image::codecs::png::PngEncoder::new(&mut png_bytes)
            .write_image(
                img.as_raw(),
                img.width(),
                img.height(),
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let thumb = make_thumbnail(&png_bytes).expect("thumbnail created");
        assert_eq!(sniff(&thumb), Some(ImageKind::Jpeg));
        let decoded = image::load_from_memory(&thumb).unwrap();
        assert_eq!(decoded.dimensions(), (128, 64));
    }
}
