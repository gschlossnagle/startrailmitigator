//! Star detection: threshold the background-subtracted, matched-filtered image
//! above the local noise and label connected components.

use crate::background::BackgroundMap;
use crate::image::Plane;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectOptions {
    /// Detection threshold in units of the local noise sigma.
    pub threshold_sigma: f32,
    /// Gaussian sigma of the matched filter applied before thresholding.
    pub filter_sigma: f64,
    pub min_area: usize,
    pub max_area: usize,
    /// Maximum bounding-box extent in pixels (longer = satellite, plane, landscape).
    pub max_extent: usize,
    /// Pixels above this normalized level count as saturated.
    pub saturation_level: f32,
    /// Detections touching this many pixels of the border are dropped.
    pub border: usize,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            threshold_sigma: 4.0,
            filter_sigma: 1.0,
            min_area: 3,
            max_area: 4000,
            max_extent: 96,
            saturation_level: 0.98,
            border: 2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Detection {
    pub id: usize,
    pub x_min: usize,
    pub y_min: usize,
    pub x_max: usize,
    pub y_max: usize,
    pub area: usize,
    /// Peak background-subtracted value and its position.
    pub peak: f32,
    pub peak_x: usize,
    pub peak_y: usize,
    pub flux: f32,
    pub saturated: bool,
    /// Linear pixel indices belonging to the component.
    #[serde(skip)]
    pub pixels: Vec<u32>,
}

impl Detection {
    pub fn extent(&self) -> usize {
        (self.x_max - self.x_min + 1).max(self.y_max - self.y_min + 1)
    }
}

/// Separable Gaussian blur. Returns the blurred plane and the factor by which
/// white noise sigma is reduced (sqrt of the sum of squared kernel weights).
pub fn gaussian_blur(src: &Plane, sigma: f64) -> (Plane, f32) {
    let radius = (3.0 * sigma).ceil().max(1.0) as i64;
    let kernel: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp() as f32)
        .collect();
    let sum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|k| k / sum).collect();
    let k2: f32 = kernel.iter().map(|k| k * k).sum();
    let (w, h) = (src.width, src.height);
    let mut tmp = Plane::new(w, h);
    for y in 0..h {
        let row = src.row(y);
        let out = tmp.row_mut(y);
        for x in 0..w {
            let mut acc = 0.0;
            for (k, kv) in kernel.iter().enumerate() {
                let xx = (x as i64 + k as i64 - radius).clamp(0, w as i64 - 1) as usize;
                acc += kv * row[xx];
            }
            out[x] = acc;
        }
    }
    let mut dst = Plane::new(w, h);
    for y in 0..h {
        for (k, kv) in kernel.iter().enumerate() {
            let yy = (y as i64 + k as i64 - radius).clamp(0, h as i64 - 1) as usize;
            let src_row = tmp.row(yy);
            let out = dst.row_mut(y);
            for x in 0..w {
                out[x] += kv * src_row[x];
            }
        }
    }
    (dst, k2) // noise variance scales by k2 per pass; two passes -> k2 total for separable filter
}

/// Detect star-like components. `residual` is the background-subtracted plane
/// used for peak/flux measurement, `raw` the original plane used for the
/// saturation test, and `mask` (values > 0.5) marks pixels to ignore.
pub fn detect(
    residual: &Plane,
    raw: &Plane,
    bg: &BackgroundMap,
    mask: Option<&Plane>,
    opts: &DetectOptions,
) -> Vec<Detection> {
    let (w, h) = (residual.width, residual.height);
    let (filtered, noise_var_factor) = gaussian_blur(residual, opts.filter_sigma);
    let noise_factor = noise_var_factor.sqrt();
    let mut above = vec![false; w * h];
    for i in 0..w * h {
        let masked = mask.map(|m| m.data[i] > 0.5).unwrap_or(false);
        above[i] =
            !masked && filtered.data[i] > opts.threshold_sigma * noise_factor * bg.noise.data[i];
    }

    let mut labels = vec![0u32; w * h];
    let mut out = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut next_id = 0usize;
    for start in 0..w * h {
        if !above[start] || labels[start] != 0 {
            continue;
        }
        next_id += 1;
        let label = next_id as u32;
        labels[start] = label;
        stack.clear();
        stack.push(start);
        let mut pixels: Vec<u32> = Vec::new();
        let (mut x_min, mut y_min, mut x_max, mut y_max) = (w, h, 0, 0);
        let mut peak = f32::NEG_INFINITY;
        let (mut px, mut py) = (0, 0);
        let mut flux = 0.0;
        let mut saturated = false;
        while let Some(i) = stack.pop() {
            pixels.push(i as u32);
            let (x, y) = (i % w, i / w);
            x_min = x_min.min(x);
            y_min = y_min.min(y);
            x_max = x_max.max(x);
            y_max = y_max.max(y);
            let v = residual.data[i];
            flux += v;
            if v > peak {
                peak = v;
                px = x;
                py = y;
            }
            if raw.data[i] >= opts.saturation_level {
                saturated = true;
            }
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if above[j] && labels[j] == 0 {
                        labels[j] = label;
                        stack.push(j);
                    }
                }
            }
        }
        let area = pixels.len();
        let extent = (x_max - x_min + 1).max(y_max - y_min + 1);
        let touches_border = x_min < opts.border
            || y_min < opts.border
            || x_max + opts.border >= w
            || y_max + opts.border >= h;
        if area < opts.min_area
            || area > opts.max_area
            || extent > opts.max_extent
            || touches_border
        {
            continue;
        }
        out.push(Detection {
            id: out.len(),
            x_min,
            y_min,
            x_max,
            y_max,
            area,
            peak,
            peak_x: px,
            peak_y: py,
            flux,
            saturated,
            pixels,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::background::estimate_background;

    #[test]
    fn finds_planted_sources_and_ignores_noise() {
        let (w, h) = (200, 150);
        let mut p = Plane::filled(w, h, 0.1);
        let mut seed = 5u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 10_000) as f32 / 10_000.0
        };
        for v in &mut p.data {
            let n: f32 = (0..12).map(|_| rnd()).sum::<f32>() - 6.0;
            *v += 0.005 * n;
        }
        let planted = [(50usize, 40usize), (120, 90), (170, 20)];
        for (sx, sy) in planted {
            for dy in -2i64..=2 {
                for dx in -2i64..=2 {
                    let g = (-((dx * dx + dy * dy) as f32) / 2.0).exp();
                    let x = (sx as i64 + dx) as usize;
                    let y = (sy as i64 + dy) as usize;
                    p.set(x, y, p.get(x, y) + 0.3 * g);
                }
            }
        }
        let bg = estimate_background(&p, 32, None);
        let mut resid = p.clone();
        for i in 0..w * h {
            resid.data[i] -= bg.background.data[i];
        }
        let dets = detect(&resid, &p, &bg, None, &DetectOptions::default());
        assert_eq!(dets.len(), 3, "{dets:?}");
        for (sx, sy) in planted {
            assert!(dets.iter().any(|d| d.peak_x == sx && d.peak_y == sy));
        }
    }
}
