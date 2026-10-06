//! End-to-end tests against synthetic frames with known trail geometry.

use strm::background::estimate_background;
use strm::detect::{detect, DetectOptions};
use strm::image::Image;
use strm::moments::measure;
use strm::pipeline::{analyze, fix, Params};
use strm::sky::{Facing, Hemisphere};
use strm::synth::{generate, SynthParams, Truth};

fn params() -> Params {
    Params {
        hemisphere: Some(Hemisphere::North),
        facing: Some(Facing::S),
        ..Default::default()
    }
}

/// Moment-based axis ratios and centroids of every detection in an image.
fn measure_all(img: &Image) -> Vec<(f64, f64, f64, f32)> {
    let lum = img.luminance();
    let bg = estimate_background(&lum, 128, None);
    let mut resid = lum.clone();
    for i in 0..resid.data.len() {
        resid.data[i] -= bg.background.data[i];
    }
    let dets = detect(&resid, &lum, &bg, None, &DetectOptions::default());
    dets.iter()
        .filter_map(|d| {
            let m = measure(&resid, d, 2, 0.5 * bg.noise.get(d.peak_x, d.peak_y))?;
            Some((m.xc, m.yc, m.elongation, d.peak))
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn check_preset(sp: SynthParams, min_len_px: f64) {
    let (img, truth): (Image, Truth) = generate(&sp);
    let mut lens: Vec<f64> = truth.stars.iter().map(|s| s.length).collect();
    let typical = median(&mut lens);
    assert!(typical >= min_len_px, "preset trails too short: {typical}");

    // Sky model recovery.
    let analysis = analyze(&img, &params(), None).unwrap();
    let sky = analysis.sky.as_ref().unwrap();
    // A crop of a few degrees constrains the pole distance weakly, so judge the
    // model by what the repair uses: the predicted direction, sense and length
    // at each star.
    let mut dir_err = 0.0f64;
    let mut len_err = 0.0f64;
    let mut n = 0usize;
    for s in &truth.stars {
        let p = sky.model.trail_at(s.center.0, s.center.1).unwrap();
        let d = (p.theta - s.theta)
            .sin()
            .atan2((p.theta - s.theta).cos())
            .abs()
            .to_degrees();
        dir_err += d;
        len_err += (p.length / s.length - 1.0).abs();
        n += 1;
    }
    let (dir_err, len_err) = (dir_err / n as f64, len_err / n as f64);
    assert!(
        dir_err < 1.5,
        "mean direction error {dir_err:.2} deg (includes sense)"
    );
    assert!(len_err < 0.1, "mean length error {:.1}%", len_err * 100.0);
    assert!(sky.rms_deg < 8.0, "orientation rms {}", sky.rms_deg);

    // Before: clearly elongated. After: round.
    let before = measure_all(&img);
    let mut before_e: Vec<f64> = before.iter().map(|b| b.2).collect();
    let mut fixed = img.clone();
    let (fixed_analysis, modified) = fix(&mut fixed, &params(), None).unwrap();
    let stats = fixed_analysis.report.repair.as_ref().unwrap();
    assert!(
        stats.repaired as f64 > 0.8 * truth.stars.len() as f64,
        "repaired {} of {}",
        stats.repaired,
        truth.stars.len()
    );
    let after = measure_all(&fixed);
    let mut after_e: Vec<f64> = after.iter().map(|a| a.2).collect();
    let (mb, ma) = (median(&mut before_e), median(&mut after_e));
    assert!(mb > 1.4, "before median elongation {mb}");
    assert!(ma < 1.15, "after median elongation {ma} (before {mb})");

    // Anchoring: bright unsaturated stars should now sit at their trail's start.
    let mut checked = 0;
    let mut worst = 0.0f64;
    // Overlapping trails are not deblended yet, so only isolated stars count.
    let isolated = |s: &strm::synth::TruthStar| {
        truth.stars.iter().all(|o| {
            std::ptr::eq(o, s) || {
                let d =
                    ((o.center.0 - s.center.0).powi(2) + (o.center.1 - s.center.1).powi(2)).sqrt();
                d > 0.5 * (o.length + s.length) + 4.0 * (o.sigma + s.sigma)
            }
        })
    };
    for s in truth
        .stars
        .iter()
        .filter(|s| !s.saturated && s.amp[1] > 0.15 && isolated(s))
    {
        let (sx, sy) = s.start;
        let nearest = after
            .iter()
            .map(|a| ((a.0 - sx).powi(2) + (a.1 - sy).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        if nearest.is_finite() {
            worst = worst.max(nearest);
            checked += 1;
            assert!(
                nearest < 0.6,
                "star at {:?} (len {:.1}) ended up {nearest:.2} px from its start",
                s.start,
                s.length
            );
        }
    }
    assert!(checked >= 6, "only {checked} bright stars checked");

    // Fill quality: modified pixels away from the new round stars should match
    // the noise-free sky to within the noise.
    let (w, h) = (img.width(), img.height());
    let lum_noise = {
        let lum = img.luminance();
        estimate_background(&lum, 128, None).noise
    };
    let mut acc = 0.0f64;
    let mut acc2 = 0.0f64;
    let mut n = 0usize;
    for y in 0..h {
        for x in 0..w {
            if modified.get(x, y) < 0.5 {
                continue;
            }
            let near_star = truth.stars.iter().any(|s| {
                let d2 = (s.start.0 - x as f64).powi(2) + (s.start.1 - y as f64).powi(2);
                d2 < (4.0 * s.sigma).powi(2)
            });
            if near_star {
                continue;
            }
            for c in 0..3 {
                let r = (fixed.planes[c].get(x, y) - sp.sky_value(x, y, c)) as f64
                    / lum_noise.get(x, y) as f64;
                acc += r;
                acc2 += r * r;
                n += 1;
            }
        }
    }
    assert!(n > 1000, "too few filled pixels to judge: {n}");
    let mean = acc / n as f64;
    let rms = (acc2 / n as f64).sqrt();
    assert!(mean.abs() < 0.35, "fill bias {mean:.2} sigma");
    assert!(rms < 2.2, "fill rms {rms:.2} sigma");
}

#[test]
fn long_preset_round_trip() {
    check_preset(SynthParams::preset_long(1200, 900), 8.0);
}

#[test]
fn wide_preset_round_trip() {
    check_preset(SynthParams::preset_wide(900, 700), 2.5);
}
