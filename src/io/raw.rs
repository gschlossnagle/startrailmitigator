//! Camera RAW decoding via rawler: demosaic, white balance and colour matrix
//! to scene-linear RGB (sRGB primaries, no gamma).

use crate::image::{Image, Plane, ShotInfo, SourceFormat};
use anyhow::{anyhow, bail, Result};
use rawler::decoders::{Orientation, RawDecodeParams};
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use rawler::rawsource::RawSource;
use std::path::Path;

pub fn load_raw(path: &Path) -> Result<Image> {
    let source = RawSource::new(path)?;
    let decoder = rawler::get_decoder(&source)?;
    let params = RawDecodeParams { image_index: 0 };
    let rawimage = decoder.raw_image(&source, &params, false)?;
    let metadata = decoder.raw_metadata(&source, &params).ok();

    let develop = RawDevelop::new_with(&[
        ProcessingStep::Rescale,
        ProcessingStep::Demosaic,
        ProcessingStep::FujiRotate,
        ProcessingStep::CropActiveArea,
        ProcessingStep::WhiteBalance,
        ProcessingStep::Calibrate,
        ProcessingStep::CropDefault,
    ]);
    let intermediate = develop
        .develop_intermediate(&rawimage)
        .map_err(|e| anyhow!("developing RAW: {e}"))?;

    let (width, height, planes) = match intermediate {
        Intermediate::Monochrome(px) => {
            let plane = Plane::from_vec(px.width, px.height, px.data);
            (px.width, px.height, vec![plane])
        }
        Intermediate::ThreeColor(px) => {
            let n = px.width * px.height;
            let mut planes: Vec<Plane> = (0..3).map(|_| Plane::new(px.width, px.height)).collect();
            for i in 0..n {
                let p = px.data[i];
                planes[0].data[i] = p[0];
                planes[1].data[i] = p[1];
                planes[2].data[i] = p[2];
            }
            (px.width, px.height, planes)
        }
        Intermediate::FourColor(_) => bail!("four-colour sensor data was not reduced to RGB"),
    };
    let _ = (width, height);

    let mut img = Image::from_planes(planes);
    img = apply_orientation(img, rawimage.orientation);
    img.format = SourceFormat {
        bits: 16,
        icc_profile: None,
        linear: true,
    };
    img.shot = ShotInfo {
        exposure_s: metadata
            .as_ref()
            .and_then(|m| m.exif.exposure_time.as_ref())
            .filter(|r| r.d != 0)
            .map(|r| r.n as f64 / r.d as f64),
        focal_length_mm: metadata
            .as_ref()
            .and_then(|m| m.exif.focal_length.as_ref())
            .filter(|r| r.d != 0)
            .map(|r| r.n as f64 / r.d as f64),
        focal_length_35mm: None,
        sensor_width_mm: None,
        camera_make: Some(rawimage.clean_make.clone()),
        camera_model: Some(rawimage.clean_model.clone()),
    };
    Ok(img)
}

/// Rotate the developed image so it appears the way the camera's orientation
/// tag says it was held.
fn apply_orientation(img: Image, orientation: Orientation) -> Image {
    let (w, h) = (img.width(), img.height());
    let rot = |src: &Plane, mode: u8| -> Plane {
        match mode {
            1 => {
                // 90 degrees clockwise: new (x, y) = old (y, h-1-x)
                let mut out = Plane::new(h, w);
                for y in 0..w {
                    for x in 0..h {
                        out.data[y * h + x] = src.get(y, h - 1 - x);
                    }
                }
                out
            }
            2 => {
                let mut out = Plane::new(w, h);
                for y in 0..h {
                    for x in 0..w {
                        out.data[y * w + x] = src.get(w - 1 - x, h - 1 - y);
                    }
                }
                out
            }
            3 => {
                // 270 degrees clockwise (90 ccw): new (x, y) = old (w-1-y, x)
                let mut out = Plane::new(h, w);
                for y in 0..w {
                    for x in 0..h {
                        out.data[y * h + x] = src.get(w - 1 - y, x);
                    }
                }
                out
            }
            _ => src.clone(),
        }
    };
    let mode = match orientation {
        Orientation::Rotate90 => 1,
        Orientation::Rotate180 => 2,
        Orientation::Rotate270 => 3,
        _ => 0,
    };
    if mode == 0 {
        return img;
    }
    let planes = img.planes.iter().map(|p| rot(p, mode)).collect();
    Image {
        planes,
        shot: img.shot,
        format: img.format,
    }
}
