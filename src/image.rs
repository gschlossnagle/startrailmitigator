//! Planar floating-point image containers.
//!
//! All processing happens on `f32` planes with values normalized so that the
//! input's white level maps to 1.0. Planes are row-major.

use serde::{Deserialize, Serialize};

/// A single-channel image plane.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn filled(width: usize, height: usize, v: f32) -> Self {
        Self {
            width,
            height,
            data: vec![v; width * height],
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), width * height, "plane data size mismatch");
        Self {
            width,
            height,
            data,
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        self.data[y * self.width + x] = v;
    }

    #[inline]
    pub fn in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    /// Value with coordinates clamped to the plane edges.
    #[inline]
    pub fn get_clamped(&self, x: i64, y: i64) -> f32 {
        let xc = x.clamp(0, self.width as i64 - 1) as usize;
        let yc = y.clamp(0, self.height as i64 - 1) as usize;
        self.get(xc, yc)
    }

    /// Bilinear sample at fractional coordinates (pixel centers at integer coords).
    pub fn sample_bilinear(&self, x: f64, y: f64) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = (x - x0) as f32;
        let fy = (y - y0) as f32;
        let (xi, yi) = (x0 as i64, y0 as i64);
        let a = self.get_clamped(xi, yi);
        let b = self.get_clamped(xi + 1, yi);
        let c = self.get_clamped(xi, yi + 1);
        let d = self.get_clamped(xi + 1, yi + 1);
        let top = a + (b - a) * fx;
        let bot = c + (d - c) * fx;
        top + (bot - top) * fy
    }

    pub fn map_inplace(&mut self, f: impl Fn(f32) -> f32) {
        for v in &mut self.data {
            *v = f(*v);
        }
    }

    pub fn clamp_inplace(&mut self, lo: f32, hi: f32) {
        self.map_inplace(|v| v.clamp(lo, hi));
    }

    pub fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        &mut self.data[y * self.width..(y + 1) * self.width]
    }
}

/// Metadata describing the shot, used to seed the sky-rotation model.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ShotInfo {
    /// Exposure time in seconds.
    pub exposure_s: Option<f64>,
    /// Physical focal length in millimetres.
    pub focal_length_mm: Option<f64>,
    /// 35 mm-equivalent focal length, when the camera recorded it.
    pub focal_length_35mm: Option<f64>,
    /// Sensor width in millimetres, when known.
    pub sensor_width_mm: Option<f64>,
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
}

impl ShotInfo {
    /// Focal length in pixels for an image of the given width, from whichever
    /// piece of metadata is available. `None` when it cannot be derived.
    pub fn focal_px(&self, image_width: usize) -> Option<f64> {
        let w = image_width as f64;
        if let Some(f35) = self.focal_length_35mm {
            if f35 > 0.0 {
                return Some(f35 * w / 36.0);
            }
        }
        match (self.focal_length_mm, self.sensor_width_mm) {
            (Some(f), Some(sw)) if f > 0.0 && sw > 0.0 => Some(f * w / sw),
            _ => None,
        }
    }
}

/// How the pixel values were encoded in the source file; needed to write the
/// output in the same form.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceFormat {
    /// 8 or 16.
    pub bits: u16,
    /// Raw ICC profile bytes to carry over, if any.
    pub icc_profile: Option<Vec<u8>>,
    /// Whether the data is scene-linear (RAW) or display-encoded (typical TIFF).
    pub linear: bool,
}

impl Default for SourceFormat {
    fn default() -> Self {
        Self {
            bits: 16,
            icc_profile: None,
            linear: false,
        }
    }
}

/// A planar RGB (or single-channel) image with normalized `f32` samples.
#[derive(Clone, Debug)]
pub struct Image {
    pub planes: Vec<Plane>,
    pub shot: ShotInfo,
    pub format: SourceFormat,
}

impl Image {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            planes: (0..channels).map(|_| Plane::new(width, height)).collect(),
            shot: ShotInfo::default(),
            format: SourceFormat::default(),
        }
    }

    pub fn from_planes(planes: Vec<Plane>) -> Self {
        assert!(!planes.is_empty());
        let (w, h) = (planes[0].width, planes[0].height);
        assert!(planes.iter().all(|p| p.width == w && p.height == h));
        Self {
            planes,
            shot: ShotInfo::default(),
            format: SourceFormat::default(),
        }
    }

    pub fn width(&self) -> usize {
        self.planes[0].width
    }

    pub fn height(&self) -> usize {
        self.planes[0].height
    }

    pub fn channels(&self) -> usize {
        self.planes.len()
    }

    /// Rec. 709 luminance for RGB, or the plane itself for single-channel images.
    pub fn luminance(&self) -> Plane {
        if self.planes.len() == 1 {
            return self.planes[0].clone();
        }
        let (r, g, b) = (&self.planes[0], &self.planes[1], &self.planes[2]);
        let mut out = Plane::new(self.width(), self.height());
        for i in 0..out.data.len() {
            out.data[i] = 0.2126 * r.data[i] + 0.7152 * g.data[i] + 0.0722 * b.data[i];
        }
        out
    }

    /// Per-pixel maximum across channels; catches strongly colored stars that
    /// luminance would under-weight.
    pub fn channel_max(&self) -> Plane {
        let mut out = self.planes[0].clone();
        for p in &self.planes[1..] {
            for (o, v) in out.data.iter_mut().zip(&p.data) {
                if *v > *o {
                    *o = *v;
                }
            }
        }
        out
    }

    pub fn clamp_inplace(&mut self, lo: f32, hi: f32) {
        for p in &mut self.planes {
            p.clamp_inplace(lo, hi);
        }
    }

    /// Apply a per-sample transform to every plane.
    pub fn map_inplace(&mut self, f: impl Fn(f32) -> f32 + Copy) {
        for p in &mut self.planes {
            p.map_inplace(f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_sampling_interpolates() {
        let p = Plane::from_vec(2, 2, vec![0.0, 1.0, 2.0, 3.0]);
        assert!((p.sample_bilinear(0.5, 0.0) - 0.5).abs() < 1e-6);
        assert!((p.sample_bilinear(0.0, 0.5) - 1.0).abs() < 1e-6);
        assert!((p.sample_bilinear(0.5, 0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn focal_px_prefers_35mm_equivalent() {
        let s = ShotInfo {
            focal_length_mm: Some(16.0),
            focal_length_35mm: Some(24.0),
            sensor_width_mm: Some(23.5),
            ..Default::default()
        };
        assert!((s.focal_px(3600).unwrap() - 2400.0).abs() < 1e-9);
        let s2 = ShotInfo {
            focal_length_mm: Some(24.0),
            sensor_width_mm: Some(36.0),
            ..Default::default()
        };
        assert!((s2.focal_px(6000).unwrap() - 4000.0).abs() < 1e-9);
        assert!(ShotInfo::default().focal_px(100).is_none());
    }
}
