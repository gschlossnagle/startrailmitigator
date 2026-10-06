//! Per-star trail fitting with Levenberg-Marquardt.
//!
//! The geometry (`x0, y0, theta, length, sigma`) plus a luminance amplitude and
//! local background are fitted on the luminance plane. The orientation and
//! length carry a prior from the global sky model whose strength grows as the
//! star gets fainter, so faint stars effectively inherit the model's geometry
//! while bright ones are free to refine it. Per-channel amplitudes are then
//! solved linearly so the star keeps its colour.

use crate::background::BackgroundMap;
use crate::detect::Detection;
use crate::image::{Image, Plane};
use crate::moments::Moments;
use crate::sky::{wrap_half_pi, TrailPrediction};
use crate::trail_model::{TrailGeometry, TrailModel};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::storage::Owned;
use nalgebra::{DMatrix, DVector, Dyn};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitOptions {
    pub max_trail_px: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub max_amp: f64,
    pub saturation_level: f32,
    /// Extra pixels of window around the model support.
    pub window_pad: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            max_trail_px: 48.0,
            sigma_min: 0.4,
            sigma_max: 8.0,
            max_amp: 50.0,
            saturation_level: 0.98,
            window_pad: 2.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarFit {
    pub det_id: usize,
    /// Fitted geometry; `theta` points along the direction of motion.
    pub geom: TrailGeometry,
    /// Position of the star when the shutter opened.
    pub start: (f64, f64),
    pub end: (f64, f64),
    pub amp_lum: f64,
    pub bg_lum: f64,
    pub amp_rgb: Vec<f32>,
    pub bg_rgb: Vec<f32>,
    /// Peak signal to noise of the trail.
    pub snr: f64,
    /// RMS of the normalized residual (data - model) / noise over the model core,
    /// worst channel. ~1 means the model explains the star to the noise floor.
    pub chi_rms: f64,
    pub saturated: bool,
    pub converged: bool,
    pub n_pixels: usize,
    /// Trail length divided by the PSF FWHM: 0 means round.
    pub elongation: f64,
}

/// Shared read-only inputs for fitting all stars of a frame.
pub struct FitInput<'a> {
    pub image: &'a Image,
    pub lum: &'a Plane,
    pub bg: &'a BackgroundMap,
    /// Component label per pixel: 0 = none, otherwise detection id + 1.
    pub labels: &'a [u32],
    pub mask: Option<&'a Plane>,
}

struct TrailProblem {
    xs: Vec<f64>,
    ys: Vec<f64>,
    data: Vec<f64>,
    inv_noise: Vec<f64>,
    p: DVector<f64>,
    prior_theta: f64,
    prior_theta_sigma: f64,
    prior_len: f64,
    prior_len_sigma: f64,
    opts: FitOptions,
}

impl TrailProblem {
    fn geom(&self) -> TrailGeometry {
        TrailGeometry {
            x0: self.p[0],
            y0: self.p[1],
            theta: self.p[2],
            length: self.p[3],
            sigma: self.p[4],
        }
    }

    fn residuals_for(&self, p: &DVector<f64>) -> DVector<f64> {
        let geom = TrailGeometry {
            x0: p[0],
            y0: p[1],
            theta: p[2],
            length: p[3],
            sigma: p[4],
        };
        let (amp, bg) = (p[5], p[6]);
        let model = TrailModel::new(geom);
        let n = self.xs.len();
        let mut r = DVector::zeros(n + 2);
        for i in 0..n {
            r[i] =
                (amp * model.eval(self.xs[i], self.ys[i]) + bg - self.data[i]) * self.inv_noise[i];
        }
        r[n] = wrap_half_pi(p[2] - self.prior_theta) / self.prior_theta_sigma;
        r[n + 1] = (p[3] - self.prior_len) / self.prior_len_sigma;
        r
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for TrailProblem {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.p = x.clone();
        self.p[3] = self.p[3].clamp(0.0, self.opts.max_trail_px);
        self.p[4] = self.p[4].clamp(self.opts.sigma_min, self.opts.sigma_max);
        self.p[5] = self.p[5].clamp(0.0, self.opts.max_amp);
    }

    fn params(&self) -> DVector<f64> {
        self.p.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        Some(self.residuals_for(&self.p))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let base = self.residuals_for(&self.p);
        let m = base.len();
        let steps = [
            0.01,
            0.01,
            0.002,
            0.02,
            0.01,
            (self.p[5].abs() * 1e-3).max(1e-6),
            1e-4,
        ];
        let mut j = DMatrix::zeros(m, 7);
        for (k, step) in steps.iter().enumerate() {
            let mut q = self.p.clone();
            q[k] += step;
            let r = self.residuals_for(&q);
            for i in 0..m {
                j[(i, k)] = (r[i] - base[i]) / step;
            }
        }
        Some(j)
    }
}

/// Fit one detected star. Returns `None` when the window has too few usable pixels.
pub fn fit_star(
    input: &FitInput,
    det: &Detection,
    moments: &Moments,
    pred: &TrailPrediction,
    opts: &FitOptions,
) -> Option<StarFit> {
    let (w, h) = (input.lum.width, input.lum.height);
    let sigma0 = moments.sigma_est.clamp(0.6, 4.0);
    let len0 = if pred.length > 0.0 {
        pred.length
    } else {
        moments.length_est
    }
    .min(opts.max_trail_px);
    let radius = 0.5 * len0 + 4.0 * sigma0 + opts.window_pad;
    let x0 = (moments.xc - radius).floor().max(0.0) as usize;
    let y0 = (moments.yc - radius).floor().max(0.0) as usize;
    let x1 = (moments.xc + radius).ceil().min(w as f64 - 1.0) as usize;
    let y1 = (moments.yc + radius).ceil().min(h as f64 - 1.0) as usize;
    let own = det.id as u32 + 1;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut data = Vec::new();
    let mut inv_noise = Vec::new();
    let mut any_saturated = false;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let i = y * w + x;
            let lbl = input.labels[i];
            if lbl != 0 && lbl != own {
                continue;
            }
            if input.mask.map(|m| m.data[i] > 0.5).unwrap_or(false) {
                continue;
            }
            let sat = input
                .image
                .planes
                .iter()
                .any(|p| p.data[i] >= opts.saturation_level);
            if sat {
                any_saturated = true;
                continue;
            }
            xs.push(x as f64);
            ys.push(y as f64);
            data.push(input.lum.data[i] as f64);
            inv_noise.push(1.0 / (input.bg.noise.data[i] as f64).max(1e-6));
        }
    }
    if xs.len() < 12 {
        return None;
    }

    let noise_c = input.bg.noise.sample_bilinear(moments.xc, moments.yc) as f64;
    let snr0 = (det.peak as f64 / noise_c.max(1e-6)).max(1.0);
    let prior_theta_sigma = (2.0f64).to_radians() * (1.0 + snr0 / 10.0);
    let prior_len_sigma = (0.1 * len0 + 0.3) * (1.0 + snr0 / 20.0);
    let bg0 = input.bg.background.sample_bilinear(moments.xc, moments.yc) as f64;

    let p0 = DVector::from_vec(vec![
        moments.xc,
        moments.yc,
        pred.theta,
        len0,
        sigma0,
        (det.peak as f64).max(1e-4),
        bg0,
    ]);
    let problem = TrailProblem {
        xs,
        ys,
        data,
        inv_noise,
        p: p0.clone(),
        prior_theta: pred.theta,
        prior_theta_sigma,
        prior_len: len0,
        prior_len_sigma,
        opts: opts.clone(),
    };
    let (problem, report) = LevenbergMarquardt::new()
        .with_tol(1e-6)
        .with_patience(60)
        .minimize(problem);
    let converged = report.termination.was_successful();
    let mut geom = problem.geom();
    let amp_lum = problem.p[5];
    let bg_lum = problem.p[6];

    // Orient theta along the predicted motion so tail_end is the start.
    let d = geom.theta - pred.theta;
    if d.cos() < 0.0 {
        geom.theta += PI;
    }
    geom.theta = geom.theta.rem_euclid(2.0 * PI);
    if geom.theta > PI {
        geom.theta -= 2.0 * PI;
    }

    // Per-channel linear amplitudes and backgrounds.
    let model = TrailModel::new(geom);
    let n = problem.xs.len();
    let tvals: Vec<f64> = (0..n)
        .map(|i| model.eval(problem.xs[i], problem.ys[i]))
        .collect();
    let s_tt: f64 = tvals.iter().map(|t| t * t).sum();
    let s_t: f64 = tvals.iter().sum();
    let nn = n as f64;
    let det_m = s_tt * nn - s_t * s_t;
    let mut amp_rgb = Vec::with_capacity(input.image.channels());
    let mut bg_rgb = Vec::with_capacity(input.image.channels());
    let mut chi_rms: f64 = 0.0;
    for plane in &input.image.planes {
        let mut s_td = 0.0;
        let mut s_d = 0.0;
        for (i, t) in tvals.iter().enumerate() {
            let v = plane.data[problem.ys[i] as usize * w + problem.xs[i] as usize] as f64;
            s_td += t * v;
            s_d += v;
        }
        let (a, b) = if det_m.abs() > 1e-12 {
            (
                (s_td * nn - s_t * s_d) / det_m,
                (s_tt * s_d - s_t * s_td) / det_m,
            )
        } else {
            (0.0, s_d / nn)
        };
        let a = a.clamp(0.0, opts.max_amp);
        amp_rgb.push(a as f32);
        bg_rgb.push(b as f32);
        // Residual over the core of the model.
        let mut acc = 0.0;
        let mut cnt = 0usize;
        for (i, t) in tvals.iter().enumerate() {
            if *t > 0.05 {
                let v = plane.data[problem.ys[i] as usize * w + problem.xs[i] as usize] as f64;
                let r = (a * t + b - v) * problem.inv_noise[i];
                acc += r * r;
                cnt += 1;
            }
        }
        if cnt > 0 {
            chi_rms = chi_rms.max((acc / cnt as f64).sqrt());
        }
    }

    Some(StarFit {
        det_id: det.id,
        geom,
        start: geom.tail_end(),
        end: geom.head_end(),
        amp_lum,
        bg_lum,
        amp_rgb,
        bg_rgb,
        snr: amp_lum / noise_c.max(1e-6),
        chi_rms,
        saturated: any_saturated || det.saturated,
        converged,
        n_pixels: n,
        elongation: geom.length / (2.3548 * geom.sigma),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::background::BackgroundMap;
    use crate::image::Plane;

    fn setup(
        geom: TrailGeometry,
        amp: f32,
        noise: f32,
        clip: bool,
    ) -> (Image, Plane, BackgroundMap, Vec<u32>, Detection, Moments) {
        let (w, h) = (80, 80);
        let bg_level = 0.05f32;
        let mut planes: Vec<Plane> = (0..3).map(|_| Plane::filled(w, h, bg_level)).collect();
        let colour = [1.0f32, 0.8, 0.6];
        let model = TrailModel::new(geom);
        for (c, p) in planes.iter_mut().enumerate() {
            model.accumulate(p, amp * colour[c]);
        }
        let mut seed = 42u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 10_000) as f32 / 10_000.0
        };
        for p in &mut planes {
            for v in &mut p.data {
                let n: f32 = (0..12).map(|_| rnd()).sum::<f32>() - 6.0;
                *v += noise * n;
                if clip {
                    *v = v.min(1.0);
                }
            }
        }
        let image = Image::from_planes(planes);
        let lum = image.luminance();
        let bg = BackgroundMap {
            background: Plane::filled(
                w,
                h,
                0.2126 * bg_level + 0.7152 * bg_level * 1.0 + 0.0722 * bg_level,
            ),
            noise: Plane::filled(w, h, noise.max(1e-4)),
            tile: 32,
        };
        let labels = vec![0u32; w * h];
        let det = Detection {
            id: 0,
            x_min: 30,
            y_min: 30,
            x_max: 50,
            y_max: 50,
            area: 50,
            peak: amp * 0.9,
            peak_x: 40,
            peak_y: 40,
            flux: 0.0,
            saturated: false,
            pixels: vec![],
        };
        // Deliberately offset the initial moments to test convergence.
        let moments = Moments {
            xc: geom.x0 + 0.6,
            yc: geom.y0 - 0.4,
            theta: geom.theta + 0.1,
            sigma_major: 2.0,
            sigma_minor: geom.sigma * 1.2,
            elongation: 2.0,
            flux: 1.0,
            length_est: geom.length * 1.3,
            sigma_est: geom.sigma * 1.2,
        };
        (image, lum, bg, labels, det, moments)
    }

    #[test]
    fn recovers_geometry_for_short_and_long_trails() {
        for (len, sigma) in [(2.0, 1.0), (6.0, 1.2), (12.0, 1.4), (30.0, 1.6)] {
            let geom = TrailGeometry {
                x0: 40.3,
                y0: 39.6,
                theta: 0.5,
                length: len,
                sigma,
            };
            let (image, lum, bg, labels, det, moments) = setup(geom, 0.4, 0.004, false);
            let input = FitInput {
                image: &image,
                lum: &lum,
                bg: &bg,
                labels: &labels,
                mask: None,
            };
            let pred = TrailPrediction {
                start: geom.tail_end(),
                end: geom.head_end(),
                theta: geom.theta + 0.05,
                length: geom.length * 1.1,
            };
            let fit = fit_star(&input, &det, &moments, &pred, &FitOptions::default()).unwrap();
            assert!(
                (fit.geom.x0 - geom.x0).abs() < 0.1,
                "len {len}: x0 {}",
                fit.geom.x0
            );
            assert!(
                (fit.geom.y0 - geom.y0).abs() < 0.1,
                "len {len}: y0 {}",
                fit.geom.y0
            );
            assert!(
                (fit.geom.length - geom.length).abs() < 0.1 * geom.length.max(3.0),
                "len {len}: L {}",
                fit.geom.length
            );
            assert!(
                (fit.geom.sigma - geom.sigma).abs() < 0.1 * geom.sigma,
                "len {len}: sigma {}",
                fit.geom.sigma
            );
            assert!(
                (fit.geom.theta - geom.theta).abs() < 0.03,
                "len {len}: theta {}",
                fit.geom.theta
            );
            // Start is the tail end along the predicted motion.
            let (sx, sy) = geom.tail_end();
            assert!(
                (fit.start.0 - sx).abs() < 0.2 && (fit.start.1 - sy).abs() < 0.2,
                "start {:?}",
                fit.start
            );
            assert!(
                (fit.amp_rgb[0] / 0.4 - 1.0).abs() < 0.05,
                "amp r {}",
                fit.amp_rgb[0]
            );
            assert!(
                (fit.amp_rgb[2] / 0.24 - 1.0).abs() < 0.05,
                "amp b {}",
                fit.amp_rgb[2]
            );
            assert!(fit.chi_rms < 1.5, "chi {}", fit.chi_rms);
            assert!(!fit.saturated);
        }
    }

    #[test]
    fn flipped_prediction_flips_start_and_end() {
        let geom = TrailGeometry {
            x0: 40.0,
            y0: 40.0,
            theta: 0.5,
            length: 10.0,
            sigma: 1.2,
        };
        let (image, lum, bg, labels, det, moments) = setup(geom, 0.4, 0.004, false);
        let input = FitInput {
            image: &image,
            lum: &lum,
            bg: &bg,
            labels: &labels,
            mask: None,
        };
        let pred = TrailPrediction {
            start: geom.head_end(),
            end: geom.tail_end(),
            theta: geom.theta + PI,
            length: geom.length,
        };
        let fit = fit_star(&input, &det, &moments, &pred, &FitOptions::default()).unwrap();
        let (hx, hy) = geom.head_end();
        assert!(
            (fit.start.0 - hx).abs() < 0.2 && (fit.start.1 - hy).abs() < 0.2,
            "start {:?}",
            fit.start
        );
    }

    #[test]
    fn saturated_trail_still_locates_geometry() {
        let geom = TrailGeometry {
            x0: 40.0,
            y0: 40.0,
            theta: -1.0,
            length: 14.0,
            sigma: 1.5,
        };
        let (image, lum, bg, labels, det, moments) = setup(geom, 3.0, 0.004, true);
        let input = FitInput {
            image: &image,
            lum: &lum,
            bg: &bg,
            labels: &labels,
            mask: None,
        };
        let pred = TrailPrediction {
            start: geom.tail_end(),
            end: geom.head_end(),
            theta: geom.theta,
            length: geom.length,
        };
        let fit = fit_star(&input, &det, &moments, &pred, &FitOptions::default()).unwrap();
        assert!(fit.saturated);
        assert!((fit.geom.x0 - geom.x0).abs() < 0.4, "x0 {}", fit.geom.x0);
        assert!((fit.geom.y0 - geom.y0).abs() < 0.4, "y0 {}", fit.geom.y0);
        assert!(
            (fit.geom.theta - geom.theta).abs() < 0.05,
            "theta {}",
            fit.geom.theta
        );
        assert!(
            (fit.geom.length - geom.length).abs() < 2.5,
            "L {}",
            fit.geom.length
        );
    }
}
