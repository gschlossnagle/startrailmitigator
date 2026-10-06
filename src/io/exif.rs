//! EXIF metadata extraction for TIFF (and JPEG) files.

use crate::image::ShotInfo;
use anyhow::Result;
use exif::{In, Tag, Value};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

fn rational_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Rational(r) if !r.is_empty() && r[0].denom != 0 => Some(r[0].to_f64()),
        Value::SRational(r) if !r.is_empty() && r[0].denom != 0 => Some(r[0].to_f64()),
        Value::Short(s) if !s.is_empty() => Some(s[0] as f64),
        Value::Long(l) if !l.is_empty() => Some(l[0] as f64),
        _ => None,
    }
}

fn ascii_string(v: &Value) -> Option<String> {
    match v {
        Value::Ascii(parts) if !parts.is_empty() => {
            Some(String::from_utf8_lossy(&parts[0]).trim().to_string())
        }
        _ => None,
    }
}

pub fn read_exif(path: &Path) -> Result<ShotInfo> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let exif = exif::Reader::new().read_from_container(&mut reader)?;
    // Look fields up by tag number so that a tag stored directly in the image
    // directory (as this tool writes them) is found as well as one in the EXIF
    // sub-directory; prefer the EXIF sub-directory when both exist.
    let get = |tag: Tag| {
        exif.fields()
            .filter(|f| f.ifd_num == In::PRIMARY && f.tag.number() == tag.number())
            .max_by_key(|f| u8::from(f.tag.context() == exif::Context::Exif))
            .map(|f| &f.value)
    };
    Ok(ShotInfo {
        exposure_s: get(Tag::ExposureTime).and_then(rational_f64),
        focal_length_mm: get(Tag::FocalLength).and_then(rational_f64),
        focal_length_35mm: get(Tag::FocalLengthIn35mmFilm)
            .and_then(rational_f64)
            .filter(|v| *v > 0.0),
        sensor_width_mm: None,
        camera_make: get(Tag::Make).and_then(ascii_string),
        camera_model: get(Tag::Model).and_then(ascii_string),
    })
}
