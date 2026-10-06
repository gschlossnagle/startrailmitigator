//! Image file input and output.
//!
//! Reads 8/16-bit TIFF and camera RAW files into [`Image`]; writes 16-bit (or
//! 8-bit, matching the input) TIFF with the ICC profile carried over.

pub mod exif;
pub mod raw;
pub mod tiff;

use crate::image::Image;
use anyhow::{Context, Result};
use std::path::Path;

const RAW_EXTENSIONS: &[&str] = &[
    "dng", "nef", "nrw", "cr2", "cr3", "crw", "arw", "srf", "sr2", "raf", "orf", "rw2", "pef",
    "iiq", "3fr", "fff", "erf", "mef", "mos", "x3f", "kdc", "dcr", "mrw", "raw",
];

pub fn is_raw_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| RAW_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Load an image by file extension: RAW formats through rawler, everything
/// else as TIFF.
pub fn load(path: &Path) -> Result<Image> {
    if is_raw_path(path) {
        raw::load_raw(path).with_context(|| format!("decoding RAW {}", path.display()))
    } else {
        tiff::load_tiff(path).with_context(|| format!("reading TIFF {}", path.display()))
    }
}

/// Save as TIFF, using the bit depth recorded in `image.format`.
pub fn save(path: &Path, image: &Image) -> Result<()> {
    tiff::save_tiff(path, image).with_context(|| format!("writing TIFF {}", path.display()))
}
