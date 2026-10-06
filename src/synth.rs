//! Synthetic star-field generator with ground truth, used for testing every
//! stage of the pipeline against known trail geometry.

use crate::image::{Image, Plane, ShotInfo, SourceFormat};
use crate::sky::{Camera, SkyModel};
use crate::trail_model::{TrailGeometry, TrailModel};
use nalgebra::Vector3;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthParams {
    pub width: usize,
    pub height: usize,
    pub n_stars: usize,
    pub exposure_s: f64,
    pub focal_px: f64,
    /// Apparent rotation axis in camera coordinates (toward the south celestial pole).
    pub axis: Vector3<f64>,
    /// Mean PSF sigma in pixels.
    pub psf_sigma: f64,
    /// Fractional per-star scatter of the PSF sigma.
    pub psf_jitter: f64,
    /// Mean sky level, normalized (white = 1).
    pub sky_level: f32,
    /// Peak-to-peak linear sky gradient across the frame.
    pub sky_gradient: f32,
    /// Read noise sigma, normalized.
    pub read_noise: f32,
    /// Shot-noise coefficient: sigma = shot_noise * sqrt(value).
    pub shot_noise: f32,
    /// Peak amplitude range (log-uniform, biased faint).
    pub amp_min: f32,
    pub amp_max: f32,
    /// Number of stars forced to saturate.
    pub saturated: usize,
    pub seed: u64,
}

impl SynthParams {
    /// A crop from a 45 MP full-frame body (8256 px wide) at 14 mm for 20 s.
    /// The pixel scale is that of the real sensor, so trails have realistic
    /// lengths whatever the crop size.
    pub fn preset_wide(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            n_stars: (width * height / 8000).max(50),
            exposure_s: 20.0,
            focal_px: 14.0 / 36.0 * 8256.0,
            axis: crate::sky::nominal_axis(
                crate::sky::Hemisphere::North,
                crate::sky::Facing::S,
                25.0,
            ),
            psf_sigma: 1.1,
            psf_jitter: 0.1,
            sky_level: 0.06,
            sky_gradient: 0.03,
            read_noise: 0.0015,
            shot_noise: 0.012,
            amp_min: 0.01,
            amp_max: 3.0,
            saturated: 5,
            seed: 7,
        }
    }

    /// A crop from a 100 MP full-frame body (11648 px wide) at 25 mm for 30 s.
    /// Trails are about four times longer than [`Self::preset_wide`].
    pub fn preset_long(width: usize, height: usize) -> Self {
        Self {
            exposure_s: 30.0,
            focal_px: 25.0 / 36.0 * 11648.0,
            psf_sigma: 1.4,
            n_stars: (width * height / 6000).max(50),
            seed: 11,
            ..Self::preset_wide(width, height)
        }
    }
}

impl SynthParams {
    /// Noise-free sky value at a pixel for one channel.
    pub fn sky_value(&self, x: usize, y: usize, c: usize) -> f32 {
        const TINT: [f32; 3] = [1.0, 0.95, 1.1];
        let t = (x as f32 / self.width as f32) * 0.6 + (y as f32 / self.height as f32) * 0.4;
        (self.sky_level + self.sky_gradient * (t - 0.5)) * TINT[c.min(2)]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TruthStar {
    pub start: (f64, f64),
    pub center: (f64, f64),
    pub end: (f64, f64),
    pub theta: f64,
    pub length: f64,
    pub sigma: f64,
    /// Peak amplitude per channel (before clipping).
    pub amp: [f32; 3],
    pub saturated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Truth {
    pub params: SynthParams,
    pub model: SkyModel,
    pub stars: Vec<TruthStar>,
}

/// Render a synthetic frame. Returns the (noisy, clipped) image and the truth.
pub fn generate(params: &SynthParams) -> (Image, Truth) {
    let (w, h) = (params.width, params.height);
    let camera = Camera::new(params.focal_px, w, h);
    let model = SkyModel::new(camera, params.axis, params.exposure_s);
    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    let normal = Normal::new(0.0f64, 1.0).unwrap();

    // Sky with a gradient.
    let mut planes: Vec<Plane> = (0..3).map(|_| Plane::new(w, h)).collect();
    for y in 0..h {
        for x in 0..w {
            for (c, p) in planes.iter_mut().enumerate() {
                p.data[y * w + x] = params.sky_value(x, y, c);
            }
        }
    }

    let mut stars = Vec::with_capacity(params.n_stars);
    let margin = 4.0;
    for i in 0..params.n_stars {
        let cx = margin + rng.random::<f64>() * (w as f64 - 2.0 * margin);
        let cy = margin + rng.random::<f64>() * (h as f64 - 2.0 * margin);
        let Some(pred) = model.trail_at(cx, cy) else {
            continue;
        };
        let sigma = params.psf_sigma * (1.0 + params.psf_jitter * normal.sample(&mut rng)).max(0.5);
        let u: f32 = rng.random();
        let mut amp = params.amp_min * (params.amp_max / params.amp_min).powf(u * u);
        let forced_sat = i < params.saturated;
        if forced_sat {
            amp = amp.max(1.5 + 3.0 * rng.random::<f32>());
        }
        // Star colour: scale red and blue relative to green, keep max = amp.
        let r = 0.7 + 0.6 * rng.random::<f32>();
        let b = 0.6 + 0.7 * rng.random::<f32>();
        let m = r.max(1.0).max(b);
        let amp_rgb = [amp * r / m, amp / m, amp * b / m];
        let geom = TrailGeometry {
            x0: 0.5 * (pred.start.0 + pred.end.0),
            y0: 0.5 * (pred.start.1 + pred.end.1),
            theta: pred.theta,
            length: pred.length,
            sigma,
        };
        let tm = TrailModel::new(geom);
        for (c, p) in planes.iter_mut().enumerate() {
            tm.accumulate(p, amp_rgb[c]);
        }
        stars.push(TruthStar {
            start: pred.start,
            center: (geom.x0, geom.y0),
            end: pred.end,
            theta: pred.theta,
            length: pred.length,
            sigma,
            amp: amp_rgb,
            saturated: amp_rgb.iter().any(|a| *a >= 1.0),
        });
    }

    // Noise and clipping.
    for p in &mut planes {
        for v in &mut p.data {
            let s = (params.shot_noise * v.max(0.0).sqrt()) as f64;
            let n =
                normal.sample(&mut rng) * s + normal.sample(&mut rng) * params.read_noise as f64;
            *v = (*v + n as f32).clamp(0.0, 1.0);
        }
    }

    let mut img = Image::from_planes(planes);
    img.format = SourceFormat {
        bits: 16,
        icc_profile: None,
        linear: true,
    };
    img.shot = ShotInfo {
        exposure_s: Some(params.exposure_s),
        focal_length_35mm: Some(params.focal_px * 36.0 / w as f64),
        focal_length_mm: Some(params.focal_px * 36.0 / w as f64),
        sensor_width_mm: Some(36.0),
        ..Default::default()
    };
    (
        img,
        Truth {
            params: params.clone(),
            model,
            stars,
        },
    )
}
