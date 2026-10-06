//! End-to-end orchestration: background, detection, measurement, sky fit,
//! per-star fits and repair.

use crate::background::{estimate_background, BackgroundMap};
use crate::detect::{detect, DetectOptions, Detection};
use crate::fit::{fit_star, FitInput, FitOptions, StarFit};
use crate::image::{Image, Plane};
use crate::moments::{measure, Moments};
use crate::repair::{repair, RepairOptions, RepairStats};
use crate::sky::{
    fit_sky, Facing, Hemisphere, SkyFit, SkyFitOptions, TrailObservation, TrailPrediction,
};
use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Params {
    pub hemisphere: Option<Hemisphere>,
    pub facing: Option<Facing>,
    pub flip_motion: bool,
    /// Overrides for metadata.
    pub exposure_s: Option<f64>,
    pub focal_px: Option<f64>,
    pub sensor_width_mm: Option<f64>,
    /// Background mesh tile size in pixels.
    pub tile: usize,
    pub max_trail_px: f64,
    /// Apply a 2.2 power law before processing (and undo after) for
    /// display-encoded TIFFs.
    pub linearize: bool,
    pub detect: DetectOptions,
    pub fit: FitOptions,
    pub repair: RepairOptions,
    /// Minimum SNR for a star to vote in the sky fit.
    pub sky_fit_min_snr: f64,
    /// Minimum moment axis ratio for a star to vote in the sky fit.
    pub sky_fit_min_elongation: f64,
    pub sky_fit_max_obs: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            hemisphere: None,
            facing: None,
            flip_motion: false,
            exposure_s: None,
            focal_px: None,
            sensor_width_mm: None,
            tile: 128,
            max_trail_px: 48.0,
            linearize: false,
            detect: DetectOptions::default(),
            fit: FitOptions::default(),
            repair: RepairOptions::default(),
            sky_fit_min_snr: 8.0,
            sky_fit_min_elongation: 1.2,
            sky_fit_max_obs: 3000,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarRecord {
    pub det: Detection,
    pub moments: Moments,
    pub prediction: TrailPrediction,
    pub fit: Option<StarFit>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub width: usize,
    pub height: usize,
    pub exposure_s: Option<f64>,
    pub focal_px: Option<f64>,
    pub sky: Option<SkyFit>,
    pub detections: usize,
    pub fitted: usize,
    pub sky_fit_votes: usize,
    /// Median (length + FWHM) / FWHM of the fitted stars.
    pub median_elongation: f64,
    /// Median moment-based major/minor axis ratio of all detections; unlike the
    /// fitted value it carries no prior from the sky model.
    pub median_moment_elongation: f64,
    pub motion_summary: String,
    pub warnings: Vec<String>,
    pub repair: Option<RepairStats>,
    pub seconds: f64,
}

pub struct Analysis {
    pub lum: Plane,
    pub bg: BackgroundMap,
    pub labels: Vec<u32>,
    pub stars: Vec<StarRecord>,
    pub sky: Option<SkyFit>,
    pub report: Report,
}

fn linearize(image: &mut Image, forward: bool) {
    let g = if forward { 2.2 } else { 1.0 / 2.2 };
    image.map_inplace(|v| v.max(0.0).powf(g));
}

/// Human-readable description of the motion the model implies at the frame centre.
fn describe_motion(sky: &SkyFit, w: usize, h: usize) -> String {
    let m = &sky.model;
    match m.trail_at(0.5 * w as f64, 0.5 * h as f64) {
        Some(p) => {
            let (dx, dy) = (p.end.0 - p.start.0, p.end.1 - p.start.1);
            let horiz = if dx.abs() >= dy.abs() {
                if dx > 0.0 {
                    "left to right"
                } else {
                    "right to left"
                }
            } else if dy > 0.0 {
                "downward"
            } else {
                "upward"
            };
            format!(
                "at the frame centre stars move {} ({:.1} px over the exposure); trails anchored at their starting end",
                horiz, p.length
            )
        }
        None => "motion at the frame centre is undefined (pole at the horizon of the projection)"
            .to_string(),
    }
}

/// Detect, measure, fit the sky model and fit every star. Does not modify the image.
pub fn analyze(image: &Image, params: &Params, mask: Option<&Plane>) -> Result<Analysis> {
    let t0 = Instant::now();
    let (w, h) = (image.width(), image.height());
    let mut warnings = Vec::new();

    let exposure_s = params.exposure_s.or(image.shot.exposure_s);
    let focal_px = params.focal_px.or_else(|| {
        let mut shot = image.shot.clone();
        if shot.sensor_width_mm.is_none() {
            shot.sensor_width_mm = params.sensor_width_mm;
        }
        shot.focal_px(w)
    });
    if exposure_s.is_none() {
        warnings
            .push("exposure time unknown: trail length scale will be fitted from the data".into());
    }
    if focal_px.is_none() {
        warnings
            .push("focal length unknown: it will be fitted from the trail direction field".into());
    }
    if params.hemisphere.is_none() {
        warnings.push("no --hemisphere given: assuming north; motion sense may be reversed".into());
    } else if params.facing.is_none() {
        warnings.push("no --facing given: motion sense resolved from hemisphere alone, which is less reliable".into());
    }

    let lum = image.luminance();
    let tile = params.tile.max((2.0 * params.max_trail_px) as usize);
    let bg = estimate_background(&lum, tile, mask);
    let mut residual = lum.clone();
    for i in 0..w * h {
        residual.data[i] -= bg.background.data[i];
    }
    let mut dopts = params.detect.clone();
    dopts.max_extent = dopts.max_extent.max((params.max_trail_px + 16.0) as usize);
    let dets = detect(&residual, &lum, &bg, mask, &dopts);
    log::info!("{} detections", dets.len());

    let mut labels = vec![0u32; w * h];
    for d in &dets {
        for &i in &d.pixels {
            labels[i as usize] = d.id as u32 + 1;
        }
    }

    let moments: Vec<Option<Moments>> = dets
        .par_iter()
        .map(|d| {
            let floor = 0.5 * bg.noise.get(d.peak_x, d.peak_y);
            measure(&residual, d, 2, floor)
        })
        .collect();

    // Votes for the sky fit.
    let mut obs: Vec<(f64, TrailObservation)> = Vec::new();
    for (d, m) in dets.iter().zip(&moments) {
        let Some(m) = m else { continue };
        let snr = d.peak as f64 / bg.noise.get(d.peak_x, d.peak_y).max(1e-6) as f64;
        if d.saturated
            || snr < params.sky_fit_min_snr
            || m.elongation < params.sky_fit_min_elongation
        {
            continue;
        }
        let weight = snr.min(100.0).sqrt() * (m.elongation - 1.0).min(3.0);
        obs.push((
            weight,
            TrailObservation {
                x: m.xc,
                y: m.yc,
                theta: m.theta,
                length: m.length_est,
                weight,
            },
        ));
    }
    obs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    obs.truncate(params.sky_fit_max_obs);
    let votes: Vec<TrailObservation> = obs.into_iter().map(|(_, o)| o).collect();
    if votes.len() < 10 {
        warnings.push(format!(
            "only {} elongated stars available for the sky fit",
            votes.len()
        ));
    }
    let sky = fit_sky(
        w,
        h,
        &votes,
        &SkyFitOptions {
            focal_px,
            exposure_s,
            hemisphere: params.hemisphere,
            facing: params.facing,
            flip: params.flip_motion,
            ..Default::default()
        },
    );
    let Some(sky) = sky else {
        bail!(
            "not enough elongated stars to fit the sky-rotation model (found {})",
            votes.len()
        );
    };
    if let Some(a) = sky.hint_agreement_deg {
        if a > 70.0 {
            warnings.push(format!(
                "fitted celestial axis is {a:.0} degrees from the hint-derived guess; check --hemisphere/--facing"
            ));
        }
    }
    if let (Some(fr), true) = (sky.fitted_rotation_rad, exposure_s.is_some()) {
        let ratio = fr / sky.model.rotation.max(1e-12);
        if !(0.6..=1.6).contains(&ratio) {
            warnings.push(format!(
                "trail lengths imply {:.0}% of the rotation expected from the exposure time; focal length or exposure metadata may be off",
                ratio * 100.0
            ));
        }
    }
    log::info!(
        "sky fit: axis {:?}, rms {:.2} deg, {}/{} inliers",
        sky.model.axis,
        sky.rms_deg,
        sky.inliers,
        sky.observations
    );

    // Per-star fits.
    let input = FitInput {
        image,
        lum: &lum,
        bg: &bg,
        labels: &labels,
        mask,
    };
    let fopts = FitOptions {
        max_trail_px: params.max_trail_px,
        ..params.fit.clone()
    };
    let stars: Vec<StarRecord> = dets
        .par_iter()
        .zip(moments.par_iter())
        .filter_map(|(d, m)| {
            let m = (*m)?;
            let prediction = sky.model.trail_at(m.xc, m.yc)?;
            let fit = fit_star(&input, d, &m, &prediction, &fopts);
            Some(StarRecord {
                det: d.clone(),
                moments: m,
                prediction,
                fit,
            })
        })
        .collect();

    let mut elong: Vec<f64> = stars
        .iter()
        .filter_map(|s| s.fit.as_ref())
        .map(|f| 1.0 + f.elongation)
        .collect();
    elong.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_elongation = elong.get(elong.len() / 2).cloned().unwrap_or(1.0);
    let mut melong: Vec<f64> = moments.iter().flatten().map(|m| m.elongation).collect();
    melong.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_moment_elongation = melong.get(melong.len() / 2).cloned().unwrap_or(1.0);

    let report = Report {
        width: w,
        height: h,
        exposure_s,
        focal_px: Some(sky.model.camera.focal_px),
        sky: Some(sky.clone()),
        detections: dets.len(),
        fitted: stars.iter().filter(|s| s.fit.is_some()).count(),
        sky_fit_votes: votes.len(),
        median_elongation,
        median_moment_elongation,
        motion_summary: describe_motion(&sky, w, h),
        warnings,
        repair: None,
        seconds: t0.elapsed().as_secs_f64(),
    };
    Ok(Analysis {
        lum,
        bg,
        labels,
        stars,
        sky: Some(sky),
        report,
    })
}

/// Analyze and repair the image in place. Returns the analysis (with repair
/// statistics in its report) and the mask of modified pixels.
pub fn fix(image: &mut Image, params: &Params, mask: Option<&Plane>) -> Result<(Analysis, Plane)> {
    let t0 = Instant::now();
    let encoded = params.linearize && !image.format.linear;
    if encoded {
        linearize(image, true);
    }
    let mut analysis = analyze(image, params, mask)?;
    let fits: Vec<StarFit> = analysis
        .stars
        .iter()
        .filter_map(|s| s.fit.clone())
        .collect();
    let pred_theta: Vec<f64> = analysis
        .stars
        .iter()
        .filter(|s| s.fit.is_some())
        .map(|s| s.prediction.theta)
        .collect();
    let (stats, modified) = repair(
        image,
        &fits,
        &pred_theta,
        &analysis.bg,
        &analysis.labels,
        &params.repair,
    );
    if encoded {
        linearize(image, false);
    }
    analysis.report.repair = Some(stats);
    analysis.report.seconds = t0.elapsed().as_secs_f64();
    Ok((analysis, modified))
}
