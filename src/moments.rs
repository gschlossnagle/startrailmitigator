//! Intensity-weighted second moments of a detection: centroid, orientation,
//! major/minor widths, and first estimates of trail length and PSF sigma.

use crate::detect::Detection;
use crate::image::Plane;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Moments {
    pub xc: f64,
    pub yc: f64,
    /// Orientation of the major axis, radians from +x toward +y, in (-pi/2, pi/2].
    pub theta: f64,
    pub sigma_major: f64,
    pub sigma_minor: f64,
    pub elongation: f64,
    pub flux: f64,
    /// Trail length estimate from `12 * (major^2 - minor^2)`.
    pub length_est: f64,
    /// PSF sigma estimate (the minor axis).
    pub sigma_est: f64,
}

/// Measure moments over the detection's bounding box grown by `grow` pixels,
/// weighting each pixel by its background-subtracted value above `floor`.
pub fn measure(residual: &Plane, det: &Detection, grow: usize, floor: f32) -> Option<Moments> {
    let (w, h) = (residual.width, residual.height);
    let x0 = det.x_min.saturating_sub(grow);
    let y0 = det.y_min.saturating_sub(grow);
    let x1 = (det.x_max + grow).min(w - 1);
    let y1 = (det.y_max + grow).min(h - 1);
    let (mut s, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let v = (residual.get(x, y) - floor).max(0.0) as f64;
            s += v;
            sx += v * x as f64;
            sy += v * y as f64;
        }
    }
    if s <= 0.0 {
        return None;
    }
    let (xc, yc) = (sx / s, sy / s);
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let v = (residual.get(x, y) - floor).max(0.0) as f64;
            let dx = x as f64 - xc;
            let dy = y as f64 - yc;
            sxx += v * dx * dx;
            syy += v * dy * dy;
            sxy += v * dx * dy;
        }
    }
    // Add the pixel's own variance (1/12) to account for sampling.
    let cxx = sxx / s + 1.0 / 12.0;
    let cyy = syy / s + 1.0 / 12.0;
    let cxy = sxy / s;
    let tr = cxx + cyy;
    let det_c = cxx * cyy - cxy * cxy;
    let disc = (0.25 * tr * tr - det_c).max(0.0).sqrt();
    let l1 = 0.5 * tr + disc;
    let l2 = (0.5 * tr - disc).max(1e-6);
    let theta = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let sigma_major = l1.sqrt();
    let sigma_minor = l2.sqrt();
    let length_est = (12.0 * (l1 - l2)).max(0.0).sqrt();
    Some(Moments {
        xc,
        yc,
        theta,
        sigma_major,
        sigma_minor,
        elongation: sigma_major / sigma_minor,
        flux: s,
        length_est,
        sigma_est: sigma_minor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trail_model::{TrailGeometry, TrailModel};

    #[test]
    fn recovers_orientation_and_length_of_a_clean_trail() {
        let (w, h) = (64, 64);
        let mut p = Plane::new(w, h);
        let geom = TrailGeometry {
            x0: 31.3,
            y0: 30.6,
            theta: 0.6,
            length: 12.0,
            sigma: 1.2,
        };
        TrailModel::new(geom).accumulate(&mut p, 0.5);
        let det = Detection {
            id: 0,
            x_min: 20,
            y_min: 20,
            x_max: 44,
            y_max: 42,
            area: 0,
            peak: 0.5,
            peak_x: 31,
            peak_y: 31,
            flux: 0.0,
            saturated: false,
            pixels: vec![],
        };
        let m = measure(&p, &det, 2, 0.0).unwrap();
        assert!((m.xc - geom.x0).abs() < 0.05, "xc {}", m.xc);
        assert!((m.yc - geom.y0).abs() < 0.05, "yc {}", m.yc);
        assert!((m.theta - geom.theta).abs() < 0.02, "theta {}", m.theta);
        assert!(
            (m.length_est - geom.length).abs() < 1.0,
            "length {}",
            m.length_est
        );
        assert!(
            (m.sigma_est - geom.sigma).abs() < 0.15,
            "sigma {}",
            m.sigma_est
        );
        assert!(m.elongation > 2.0);
    }
}
