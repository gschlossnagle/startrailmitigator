//! Replace fitted trails with round stars.
//!
//! For each star the fitted trail model is subtracted from every channel, which
//! leaves the real sky and its noise in place. Stars the model does not explain
//! to the noise floor (saturated cores, bright PSF wings) have their footprint
//! overwritten with sky borrowed from alongside the trail, level-corrected by
//! the background map so gradients survive. Finally a round Gaussian star of the
//! trail's PSF width and peak colour is added at the chosen anchor.

use crate::background::BackgroundMap;
use crate::fit::StarFit;
use crate::image::{Image, Plane};
use crate::trail_model::{round_star, TrailModel};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum Anchor {
    /// Where the star was when the shutter opened.
    Start,
    /// Where the star was when the shutter closed.
    End,
    /// Midpoint of the trail.
    Center,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepairOptions {
    /// Trails whose (length + FWHM) / FWHM is below this are left alone.
    pub min_axis_ratio: f64,
    /// Multiplier on the round star's peak amplitude.
    pub star_gain: f32,
    pub anchor: Anchor,
    /// Normalized residual RMS above which the model is not trusted and the
    /// footprint is filled from adjacent sky instead.
    pub fallback_chi: f64,
    /// Skip stars whose fitted orientation deviates from the sky model by more
    /// than this many degrees (double stars, galaxies, hot pixel pairs).
    pub max_theta_dev_deg: f64,
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            min_axis_ratio: 1.25,
            star_gain: 1.0,
            anchor: Anchor::Start,
            fallback_chi: 2.5,
            max_theta_dev_deg: 25.0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RepairStats {
    pub repaired: usize,
    pub model_subtracted: usize,
    pub patch_filled: usize,
    pub skipped_round: usize,
    pub skipped_unconverged: usize,
    pub skipped_misaligned: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fill {
    Model,
    Patch,
}

/// Which stars get repaired and how. `pred_theta[i]` is the sky model's motion
/// direction at star `i`, used for the misalignment gate.
fn plan(
    fits: &[StarFit],
    pred_theta: &[f64],
    opts: &RepairOptions,
    stats: &mut RepairStats,
) -> Vec<Option<Fill>> {
    fits.iter()
        .zip(pred_theta)
        .map(|(f, pt)| {
            if !f.converged || f.amp_lum <= 0.0 {
                stats.skipped_unconverged += 1;
                return None;
            }
            if 1.0 + f.elongation < opts.min_axis_ratio {
                stats.skipped_round += 1;
                return None;
            }
            let dev = crate::sky::wrap_half_pi(f.geom.theta - pt)
                .abs()
                .to_degrees();
            if dev > opts.max_theta_dev_deg {
                stats.skipped_misaligned += 1;
                return None;
            }
            if f.saturated || f.chi_rms > opts.fallback_chi {
                Some(Fill::Patch)
            } else {
                Some(Fill::Model)
            }
        })
        .collect()
}

/// Deterministic pseudo-random bit from a pixel index.
#[inline]
fn hash_bit(i: usize) -> bool {
    let mut x = i as u64;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x & 1 == 1
}

/// Repair the image in place. Returns statistics and a mask of modified pixels.
pub fn repair(
    image: &mut Image,
    fits: &[StarFit],
    pred_theta: &[f64],
    bg: &BackgroundMap,
    labels: &[u32],
    opts: &RepairOptions,
) -> (RepairStats, Plane) {
    let (w, h) = (image.width(), image.height());
    let mut stats = RepairStats::default();
    let fills = plan(fits, pred_theta, opts, &mut stats);
    let mut modified = Plane::new(w, h);

    // Pass 1: subtract every fitted trail (also the wings of patch-filled stars).
    for (f, fill) in fits.iter().zip(&fills) {
        if fill.is_none() {
            continue;
        }
        let model = TrailModel::new(f.geom);
        if let Some((x0, y0, x1, y1)) = f.geom.bbox(w, h) {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let t = model.eval(x as f64, y as f64) as f32;
                    if t <= 1e-5 {
                        continue;
                    }
                    let i = y * w + x;
                    for (c, p) in image.planes.iter_mut().enumerate() {
                        p.data[i] -= f.amp_rgb[c] * t;
                    }
                    modified.data[i] = 1.0;
                }
            }
        }
    }

    // Pass 2: patch-fill footprints the model cannot be trusted on.
    for (f, fill) in fits.iter().zip(&fills) {
        if *fill != Some(Fill::Patch) {
            continue;
        }
        patch_fill(image, f, bg, labels, &mut modified);
    }

    // Pass 3: add the round stars.
    for (f, fill) in fits.iter().zip(&fills) {
        if fill.is_none() {
            continue;
        }
        let (ax, ay) = match opts.anchor {
            Anchor::Start => f.start,
            Anchor::End => f.end,
            Anchor::Center => (f.geom.x0, f.geom.y0),
        };
        let star = round_star(ax, ay, f.geom.sigma);
        if let Some((x0, y0, x1, y1)) = star.geom.bbox(w, h) {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let g = star.eval(x as f64, y as f64) as f32;
                    if g <= 1e-5 {
                        continue;
                    }
                    let i = y * w + x;
                    for (c, p) in image.planes.iter_mut().enumerate() {
                        p.data[i] += opts.star_gain * f.amp_rgb[c] * g;
                    }
                    modified.data[i] = 1.0;
                }
            }
        }
        stats.repaired += 1;
        match fill {
            Some(Fill::Model) => stats.model_subtracted += 1,
            Some(Fill::Patch) => stats.patch_filled += 1,
            None => {}
        }
    }

    image.clamp_inplace(0.0, 1.0);
    (stats, modified)
}

/// Overwrite the footprint of a trail with sky borrowed from a parallel strip
/// beside it. The strip is offset perpendicular to the trail by just over the
/// footprint's half-width, level-corrected with the background map, and the
/// side is chosen per pixel so no single side's structure is copied as a streak.
fn patch_fill(
    image: &mut Image,
    f: &StarFit,
    bg: &BackgroundMap,
    labels: &[u32],
    modified: &mut Plane,
) {
    let (w, h) = (image.width(), image.height());
    let model = TrailModel::new(f.geom);
    let noise_c = bg.noise.sample_bilinear(f.geom.x0, f.geom.y0) as f64;
    // Footprint threshold: where the trail would still be visible above the noise.
    let tau = (0.5 * noise_c / f.amp_lum.max(1e-6)).clamp(1e-4, 0.05);
    let half_width = f.geom.sigma * (2.0 * (1.0 / tau).ln()).sqrt();
    let offset = 2.0 * half_width + 1.5;
    let (dx, dy) = f.geom.direction();
    let (nx, ny) = (-dy, dx);
    let own = f.det_id as u32 + 1;
    let Some((x0, y0, x1, y1)) = f.geom.bbox(w, h) else {
        return;
    };
    // Snapshot the window so borrowed pixels are not ones already overwritten.
    let src: Vec<Plane> = image.planes.clone();
    let usable = |sx: f64, sy: f64| -> bool {
        if sx < 0.0 || sy < 0.0 || sx > w as f64 - 1.0 || sy > h as f64 - 1.0 {
            return false;
        }
        let i = sy.round() as usize * w + sx.round() as usize;
        let lbl = labels[i];
        lbl == 0 || lbl == own
    };
    for y in y0..=y1 {
        for x in x0..=x1 {
            let t = model.eval(x as f64, y as f64);
            if t <= tau {
                continue;
            }
            // Feather the outer ring of the footprint.
            let wgt = ((t / tau - 1.0) / 1.0).clamp(0.0, 1.0) as f32;
            let i = y * w + x;
            let side = if hash_bit(i) { 1.0 } else { -1.0 };
            let mut chosen = None;
            for s in [side, -side] {
                let sx = x as f64 + s * nx * offset;
                let sy = y as f64 + s * ny * offset;
                if usable(sx, sy) {
                    chosen = Some((sx, sy));
                    break;
                }
            }
            let bg_here = bg.background.get(x, y);
            for (c, p) in image.planes.iter_mut().enumerate() {
                let fill = match chosen {
                    Some((sx, sy)) => {
                        let v = src[c].sample_bilinear(sx, sy);
                        let b = bg.background.sample_bilinear(sx, sy);
                        v - b + bg_here
                    }
                    None => {
                        // No clean neighbour strip: synthesize noise around the background.
                        let n = (hash_f(i * 7 + c) - 0.5) * 3.46 * bg.noise.get(x, y);
                        bg_here + n
                    }
                };
                p.data[i] = (1.0 - wgt) * p.data[i] + wgt * fill;
            }
            modified.data[i] = 1.0;
        }
    }
}

#[inline]
fn hash_f(i: usize) -> f32 {
    let mut x = i as u64;
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 29;
    (x % 1_000_003) as f32 / 1_000_003.0
}
