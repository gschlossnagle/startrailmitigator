//! Analytic model of a trailed star: a round Gaussian point-spread function
//! convolved with a line segment (Vereš et al. 2012). The model is normalized to
//! unit peak so that an amplitude multiplies it directly.

use serde::{Deserialize, Serialize};

/// Geometry of one trail in image pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrailGeometry {
    /// Centre of the trail (midpoint of the segment).
    pub x0: f64,
    pub y0: f64,
    /// Direction of the trail in radians, measured from +x toward +y (image y is down).
    pub theta: f64,
    /// Length of the segment in pixels (0 = untrailed point source).
    pub length: f64,
    /// Gaussian PSF sigma in pixels.
    pub sigma: f64,
}

impl TrailGeometry {
    /// Unit vector along the trail.
    pub fn direction(&self) -> (f64, f64) {
        (self.theta.cos(), self.theta.sin())
    }

    /// Endpoint at `-L/2` along the direction.
    pub fn tail_end(&self) -> (f64, f64) {
        let (dx, dy) = self.direction();
        (
            self.x0 - 0.5 * self.length * dx,
            self.y0 - 0.5 * self.length * dy,
        )
    }

    /// Endpoint at `+L/2` along the direction.
    pub fn head_end(&self) -> (f64, f64) {
        let (dx, dy) = self.direction();
        (
            self.x0 + 0.5 * self.length * dx,
            self.y0 + 0.5 * self.length * dy,
        )
    }

    /// Half-size of a square window that contains essentially all of the model.
    pub fn support_radius(&self) -> f64 {
        0.5 * self.length + 4.0 * self.sigma
    }

    /// Pixel bounding box (inclusive) of the model support, clipped to an image.
    pub fn bbox(&self, width: usize, height: usize) -> Option<(usize, usize, usize, usize)> {
        let r = self.support_radius().ceil();
        let x_min = (self.x0 - r).floor().max(0.0);
        let y_min = (self.y0 - r).floor().max(0.0);
        let x_max = (self.x0 + r).ceil().min(width as f64 - 1.0);
        let y_max = (self.y0 + r).ceil().min(height as f64 - 1.0);
        if x_max < x_min || y_max < y_min {
            None
        } else {
            Some((
                x_min as usize,
                y_min as usize,
                x_max as usize,
                y_max as usize,
            ))
        }
    }
}

/// Precomputed evaluator for one trail model.
#[derive(Clone, Copy, Debug)]
pub struct TrailModel {
    pub geom: TrailGeometry,
    cos_t: f64,
    sin_t: f64,
    inv_2sig2: f64,
    inv_sig_sqrt2: f64,
    half_len: f64,
    inv_peak: f64,
}

impl TrailModel {
    pub fn new(geom: TrailGeometry) -> Self {
        let sigma = geom.sigma.max(1e-3);
        let half_len = 0.5 * geom.length.max(0.0);
        let inv_sig_sqrt2 = 1.0 / (sigma * std::f64::consts::SQRT_2);
        // Peak value of the un-normalized trail function at (0, 0).
        let peak = 2.0 * libm::erf(half_len * inv_sig_sqrt2);
        let inv_peak = if peak > 1e-9 { 1.0 / peak } else { 0.0 };
        Self {
            geom,
            cos_t: geom.theta.cos(),
            sin_t: geom.theta.sin(),
            inv_2sig2: 1.0 / (2.0 * sigma * sigma),
            inv_sig_sqrt2,
            half_len,
            inv_peak,
        }
    }

    /// Model value at a pixel position, in [0, 1].
    #[inline]
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.geom.x0;
        let dy = y - self.geom.y0;
        // Rotate into trail-aligned coordinates.
        let u = dx * self.cos_t + dy * self.sin_t;
        let v = -dx * self.sin_t + dy * self.cos_t;
        let perp = (-v * v * self.inv_2sig2).exp();
        if self.inv_peak == 0.0 {
            // Degenerate length: pure Gaussian.
            return perp * (-u * u * self.inv_2sig2).exp();
        }
        let along = libm::erf((u + self.half_len) * self.inv_sig_sqrt2)
            - libm::erf((u - self.half_len) * self.inv_sig_sqrt2);
        perp * along * self.inv_peak
    }

    /// Accumulate `amp * model` into a plane over the model's support.
    pub fn accumulate(&self, plane: &mut crate::image::Plane, amp: f32) {
        if let Some((x0, y0, x1, y1)) = self.geom.bbox(plane.width, plane.height) {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let v = self.eval(x as f64, y as f64) as f32;
                    if v > 0.0 {
                        let i = plane.idx(x, y);
                        plane.data[i] += amp * v;
                    }
                }
            }
        }
    }
}

/// Unit-peak round Gaussian star model.
pub fn round_star(x0: f64, y0: f64, sigma: f64) -> TrailModel {
    TrailModel::new(TrailGeometry {
        x0,
        y0,
        theta: 0.0,
        length: 0.0,
        sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_peak_at_centre() {
        for len in [0.0, 0.5, 3.0, 20.0] {
            let m = TrailModel::new(TrailGeometry {
                x0: 10.0,
                y0: 10.0,
                theta: 0.7,
                length: len,
                sigma: 1.3,
            });
            assert!((m.eval(10.0, 10.0) - 1.0).abs() < 1e-9, "len {len}");
        }
    }

    #[test]
    fn zero_length_matches_gaussian() {
        let m = TrailModel::new(TrailGeometry {
            x0: 0.0,
            y0: 0.0,
            theta: 1.0,
            length: 0.0,
            sigma: 1.5,
        });
        let expect = (-(2.0f64 * 2.0 + 1.0) / (2.0 * 1.5 * 1.5)).exp();
        assert!((m.eval(2.0, 1.0) - expect).abs() < 1e-9);
    }

    #[test]
    fn long_trail_is_flat_along_its_axis_and_gaussian_across() {
        let m = TrailModel::new(TrailGeometry {
            x0: 0.0,
            y0: 0.0,
            theta: 0.0,
            length: 30.0,
            sigma: 1.0,
        });
        assert!((m.eval(10.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((m.eval(10.0, 1.0) - (-0.5f64).exp()).abs() < 1e-6);
        // Half-maximum at the segment end.
        assert!((m.eval(15.0, 0.0) - 0.5).abs() < 1e-6);
        assert!(m.eval(20.0, 0.0) < 1e-4);
    }

    #[test]
    fn endpoints_are_symmetric_about_centre() {
        let g = TrailGeometry {
            x0: 5.0,
            y0: 7.0,
            theta: std::f64::consts::FRAC_PI_2,
            length: 4.0,
            sigma: 1.0,
        };
        let (tx, ty) = g.tail_end();
        let (hx, hy) = g.head_end();
        assert!((tx - 5.0).abs() < 1e-9 && (ty - 5.0).abs() < 1e-9);
        assert!((hx - 5.0).abs() < 1e-9 && (hy - 9.0).abs() < 1e-9);
    }
}
