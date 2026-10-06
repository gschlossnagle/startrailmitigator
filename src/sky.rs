//! Sky-rotation model: the apparent motion of stars during an exposure, seen
//! through a pinhole camera, is a rotation about the celestial axis. Fitting that
//! axis to the orientations of all the trails in a frame predicts the direction,
//! sense and length of the trail at every pixel, which is what makes "anchor at
//! the start of the trail" well defined.
//!
//! Camera coordinates: x right, y down, z forward (into the scene). This is a
//! right-handed frame. A pixel `(px, py)` corresponds to the direction
//! `((px - cx) / f, (py - cy) / f, 1)`.
//!
//! The apparent rotation vector points toward the **south** celestial pole with
//! the right-hand rule: stars circle the north celestial pole counterclockwise as
//! seen from the ground, which is the same as a right-handed rotation about the
//! direction to the south pole.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Sidereal rotation rate of the sky in radians per second (15.04 arcsec/s).
pub const SIDEREAL_RATE: f64 = 7.292_115_9e-5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum Hemisphere {
    North,
    South,
}

/// Rough compass direction the centre of the frame points toward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[clap(rename_all = "UPPER")]
pub enum Facing {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}

impl Facing {
    /// Azimuth in degrees, clockwise from north.
    pub fn azimuth_deg(self) -> f64 {
        match self {
            Facing::N => 0.0,
            Facing::NE => 45.0,
            Facing::E => 90.0,
            Facing::SE => 135.0,
            Facing::S => 180.0,
            Facing::SW => 225.0,
            Facing::W => 270.0,
            Facing::NW => 315.0,
        }
    }
}

/// Pinhole camera intrinsics in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub focal_px: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Camera {
    pub fn new(focal_px: f64, width: usize, height: usize) -> Self {
        Self {
            focal_px,
            cx: 0.5 * (width as f64 - 1.0),
            cy: 0.5 * (height as f64 - 1.0),
        }
    }

    #[inline]
    pub fn pixel_to_dir(&self, x: f64, y: f64) -> Vector3<f64> {
        Vector3::new(
            (x - self.cx) / self.focal_px,
            (y - self.cy) / self.focal_px,
            1.0,
        )
        .normalize()
    }

    #[inline]
    pub fn dir_to_pixel(&self, d: &Vector3<f64>) -> Option<(f64, f64)> {
        if d.z <= 1e-9 {
            return None;
        }
        Some((
            self.cx + self.focal_px * d.x / d.z,
            self.cy + self.focal_px * d.y / d.z,
        ))
    }
}

/// Rotate `v` about the unit `axis` by `angle` radians (Rodrigues' formula).
pub fn rotate_about(v: &Vector3<f64>, axis: &Vector3<f64>, angle: f64) -> Vector3<f64> {
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

/// Unit vector from spherical angles in the camera frame: `az` is measured in
/// the image plane from +x toward +y, `alt` is the angle out of the image plane
/// toward +z (forward).
pub fn axis_from_angles(az: f64, alt: f64) -> Vector3<f64> {
    let (sa, ca) = alt.sin_cos();
    let (sz, cz) = az.sin_cos();
    Vector3::new(ca * cz, ca * sz, sa)
}

/// Inverse of [`axis_from_angles`]: `(az, alt)` in radians.
pub fn angles_from_axis(a: &Vector3<f64>) -> (f64, f64) {
    (a.y.atan2(a.x), a.z.clamp(-1.0, 1.0).asin())
}

/// Predicted trail at one pixel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrailPrediction {
    /// Where the star was when the shutter opened.
    pub start: (f64, f64),
    /// Where the star was when the shutter closed.
    pub end: (f64, f64),
    /// Direction of motion in radians from +x toward +y.
    pub theta: f64,
    /// Trail length in pixels.
    pub length: f64,
}

/// Full model of the sky's apparent motion in one frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkyModel {
    pub camera: Camera,
    /// Unit vector toward the south celestial pole in camera coordinates.
    pub axis: Vector3<f64>,
    /// Total apparent rotation during the exposure, in radians.
    pub rotation: f64,
}

impl SkyModel {
    pub fn new(camera: Camera, axis: Vector3<f64>, exposure_s: f64) -> Self {
        Self {
            camera,
            axis: axis.normalize(),
            rotation: SIDEREAL_RATE * exposure_s,
        }
    }

    /// Direction (unit vector, image plane) and per-radian length factor of the
    /// motion at a pixel, without the exposure applied. Length factor is the
    /// pixel displacement per radian of sky rotation.
    #[inline]
    pub fn velocity_at(&self, x: f64, y: f64) -> ((f64, f64), f64) {
        let d = self.camera.pixel_to_dir(x, y);
        let v = self.axis.cross(&d);
        // Derivative of the gnomonic projection.
        let inv_dz2 = 1.0 / (d.z * d.z);
        let vx = (v.x * d.z - d.x * v.z) * inv_dz2 * self.camera.focal_px;
        let vy = (v.y * d.z - d.y * v.z) * inv_dz2 * self.camera.focal_px;
        let n = (vx * vx + vy * vy).sqrt();
        if n < 1e-12 {
            ((1.0, 0.0), 0.0)
        } else {
            ((vx / n, vy / n), n)
        }
    }

    /// Exact start/end of the trail whose midpoint (in time) is at `(x, y)`.
    pub fn trail_at(&self, x: f64, y: f64) -> Option<TrailPrediction> {
        let d = self.camera.pixel_to_dir(x, y);
        let ds = rotate_about(&d, &self.axis, -0.5 * self.rotation);
        let de = rotate_about(&d, &self.axis, 0.5 * self.rotation);
        let start = self.camera.dir_to_pixel(&ds)?;
        let end = self.camera.dir_to_pixel(&de)?;
        let (dx, dy) = (end.0 - start.0, end.1 - start.1);
        Some(TrailPrediction {
            start,
            end,
            theta: dy.atan2(dx),
            length: (dx * dx + dy * dy).sqrt(),
        })
    }

    /// Image position of the pole the trails circle, if it projects in front of
    /// the camera (it can be far outside the frame).
    pub fn pole_pixel(&self) -> Option<(f64, f64)> {
        self.camera
            .dir_to_pixel(&self.axis)
            .or_else(|| self.camera.dir_to_pixel(&(-self.axis)))
    }
}

/// Rough guess at where the celestial axis is in camera coordinates, built from
/// the hemisphere and the compass direction of the frame centre. Only its sign
/// relative to the fitted axis is used, so it tolerates tens of degrees of error.
///
/// `pitch_deg` is how far above the horizon the frame centre points; a typical
/// astro-landscape composition is tilted up somewhat.
pub fn nominal_axis(hemisphere: Hemisphere, facing: Facing, pitch_deg: f64) -> Vector3<f64> {
    // East-North-Up frame. The south celestial pole lies at altitude -lat toward
    // south; use |lat| = 45 degrees as a middle-of-the-road guess.
    let lat = match hemisphere {
        Hemisphere::North => 45.0f64.to_radians(),
        Hemisphere::South => -45.0f64.to_radians(),
    };
    let scp = Vector3::new(0.0, -lat.cos(), -lat.sin());
    let a = facing.azimuth_deg().to_radians();
    let h = pitch_deg.to_radians();
    let forward = Vector3::new(a.sin() * h.cos(), a.cos() * h.cos(), h.sin());
    let right = Vector3::new(a.cos(), -a.sin(), 0.0);
    let down = forward.cross(&right);
    Vector3::new(scp.dot(&right), scp.dot(&down), scp.dot(&forward))
}

/// One measured trail used to fit the sky model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrailObservation {
    pub x: f64,
    pub y: f64,
    /// Orientation of the trail line in radians; only defined modulo pi.
    pub theta: f64,
    /// Measured trail length in pixels (may be noisy).
    pub length: f64,
    pub weight: f64,
}

/// How the sign of the rotation axis was decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignSource {
    /// Hemisphere plus facing direction.
    Facing,
    /// Hemisphere only; weaker.
    HemisphereOnly,
    /// No hints at all; assumed northern hemisphere and an axis pointing down the frame.
    Assumed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkyFit {
    pub model: SkyModel,
    /// Robust RMS of the orientation residuals of the inliers, in degrees.
    pub rms_deg: f64,
    pub inliers: usize,
    pub observations: usize,
    /// Total rotation angle implied by the measured trail lengths, in radians
    /// (a check against sidereal rate x exposure time).
    pub fitted_rotation_rad: Option<f64>,
    pub sign_source: SignSource,
    /// Angle in degrees between the fitted axis and the hint-derived nominal axis.
    pub hint_agreement_deg: Option<f64>,
    /// True when the focal length was fitted rather than supplied.
    pub focal_fitted: bool,
}

/// Wrap an angle difference to (-pi/2, pi/2] (orientations are mod pi).
#[inline]
pub fn wrap_half_pi(mut d: f64) -> f64 {
    d %= PI;
    if d > 0.5 * PI {
        d -= PI;
    } else if d <= -0.5 * PI {
        d += PI;
    }
    d
}

/// Huber loss on an angular residual (radians).
#[inline]
fn huber(r: f64, delta: f64) -> f64 {
    let a = r.abs();
    if a <= delta {
        0.5 * r * r
    } else {
        delta * (a - 0.5 * delta)
    }
}

fn orientation_cost(
    camera: &Camera,
    axis: &Vector3<f64>,
    obs: &[TrailObservation],
    delta: f64,
) -> f64 {
    let model = SkyModel {
        camera: *camera,
        axis: *axis,
        rotation: 0.0,
    };
    let mut cost = 0.0;
    for o in obs {
        let ((vx, vy), _) = model.velocity_at(o.x, o.y);
        let r = wrap_half_pi(vy.atan2(vx) - o.theta);
        cost += o.weight * huber(r, delta);
    }
    cost
}

/// Grid-search the axis over one hemisphere of directions (the other hemisphere
/// gives identical orientations with the opposite sign).
fn search_axis(camera: &Camera, obs: &[TrailObservation], delta: f64) -> (Vector3<f64>, f64) {
    let mut best = (Vector3::new(0.0, 1.0, 0.0), f64::INFINITY);
    let eval = |az: f64, alt: f64, best: &mut (Vector3<f64>, f64)| {
        let a = axis_from_angles(az, alt);
        let c = orientation_cost(camera, &a, obs, delta);
        if c < best.1 {
            *best = (a, c);
        }
    };
    // Coarse: 4 degree steps, alt in [0, 90].
    let step = 4.0f64.to_radians();
    let mut alt = 0.0;
    while alt <= 0.5 * PI + 1e-9 {
        let n_az = ((2.0 * PI * alt.cos() / step).ceil() as usize).max(1);
        for i in 0..n_az {
            let az = 2.0 * PI * i as f64 / n_az as f64;
            eval(az, alt, &mut best);
        }
        alt += step;
    }
    // Refine around the best with shrinking steps. Each level searches a small
    // patch of the sphere using a local tangent-plane parameterization so the
    // azimuth singularity at alt = 90 degrees does not matter.
    let mut half_span = step;
    while half_span > 0.002f64.to_radians() {
        let centre = best.0;
        let (e1, e2) = tangent_basis(&centre);
        let s = half_span / 4.0;
        let mut local_best = best;
        for i in -4..=4 {
            for j in -4..=4 {
                let a = (centre + e1 * (i as f64 * s) + e2 * (j as f64 * s)).normalize();
                let c = orientation_cost(camera, &a, obs, delta);
                if c < local_best.1 {
                    local_best = (a, c);
                }
            }
        }
        best = local_best;
        half_span /= 3.0;
    }
    best
}

fn tangent_basis(n: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let helper = if n.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let e1 = n.cross(&helper).normalize();
    let e2 = n.cross(&e1).normalize();
    (e1, e2)
}

/// Options for [`fit_sky`].
#[derive(Clone, Debug)]
pub struct SkyFitOptions {
    /// Focal length in pixels, if known. Otherwise it is searched.
    pub focal_px: Option<f64>,
    /// Exposure time in seconds, if known.
    pub exposure_s: Option<f64>,
    pub hemisphere: Option<Hemisphere>,
    pub facing: Option<Facing>,
    /// Flip the motion sense the hints would otherwise choose.
    pub flip: bool,
    /// Assumed pitch of the frame centre above the horizon for the nominal axis.
    pub pitch_deg: f64,
    /// Huber threshold for orientation residuals, in degrees.
    pub huber_deg: f64,
}

impl Default for SkyFitOptions {
    fn default() -> Self {
        Self {
            focal_px: None,
            exposure_s: None,
            hemisphere: None,
            facing: None,
            flip: false,
            pitch_deg: 20.0,
            huber_deg: 10.0,
        }
    }
}

/// Weighted median of `v` (weights must be positive).
fn weighted_median(mut v: Vec<(f64, f64)>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = v.iter().map(|p| p.1).sum();
    let mut acc = 0.0;
    for (val, w) in &v {
        acc += w;
        if acc >= 0.5 * total {
            return Some(*val);
        }
    }
    v.last().map(|p| p.0)
}

/// Fit the sky-rotation model to measured trail orientations and lengths.
pub fn fit_sky(
    width: usize,
    height: usize,
    obs: &[TrailObservation],
    opts: &SkyFitOptions,
) -> Option<SkyFit> {
    if obs.len() < 3 {
        return None;
    }
    let delta = opts.huber_deg.to_radians();
    let (camera, axis_unsigned, focal_fitted) = match opts.focal_px {
        Some(f) => {
            let cam = Camera::new(f, width, height);
            let (a, _) = search_axis(&cam, obs, delta);
            (cam, a, false)
        }
        None => {
            // Search a log-spaced range of plausible focal lengths.
            let w = width as f64;
            let mut best: Option<(Camera, Vector3<f64>, f64)> = None;
            let mut f = 0.35 * w;
            while f <= 3.0 * w {
                let cam = Camera::new(f, width, height);
                let (a, c) = search_axis(&cam, obs, delta);
                if best.as_ref().is_none_or(|b| c < b.2) {
                    best = Some((cam, a, c));
                }
                f *= 1.25;
            }
            let (cam, a, _) = best?;
            (cam, a, true)
        }
    };

    // Resolve the sign from hints.
    let (sign_source, nominal) = match (opts.hemisphere, opts.facing) {
        (Some(h), Some(fc)) => (
            SignSource::Facing,
            Some(nominal_axis(h, fc, opts.pitch_deg)),
        ),
        (Some(h), None) => {
            let y = match h {
                Hemisphere::North => 1.0,
                Hemisphere::South => -1.0,
            };
            (SignSource::HemisphereOnly, Some(Vector3::new(0.0, y, 0.0)))
        }
        (None, _) => (SignSource::Assumed, Some(Vector3::new(0.0, 1.0, 0.0))),
    };
    let mut axis = axis_unsigned;
    let mut agreement = None;
    if let Some(n) = nominal {
        let n = n.normalize();
        if axis.dot(&n) < 0.0 {
            axis = -axis;
        }
        agreement = Some(axis.dot(&n).clamp(-1.0, 1.0).acos().to_degrees());
    }
    if opts.flip {
        axis = -axis;
    }

    // Rotation angle implied by the measured lengths.
    let unit_model = SkyModel {
        camera,
        axis,
        rotation: 0.0,
    };
    let ratios: Vec<(f64, f64)> = obs
        .iter()
        .filter(|o| o.length > 0.0)
        .map(|o| {
            let (_, per_rad) = unit_model.velocity_at(o.x, o.y);
            (o.length / per_rad.max(1e-9), o.weight)
        })
        .collect();
    let fitted_scale = weighted_median(ratios);

    let rotation = match opts.exposure_s {
        Some(t) => SIDEREAL_RATE * t,
        None => fitted_scale.unwrap_or(0.0),
    };

    // Residual statistics.
    let mut resid: Vec<f64> = obs
        .iter()
        .map(|o| {
            let ((vx, vy), _) = unit_model.velocity_at(o.x, o.y);
            wrap_half_pi(vy.atan2(vx) - o.theta).abs()
        })
        .collect();
    resid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let inliers = resid.iter().filter(|r| **r <= 2.0 * delta).count();
    let rms = if inliers > 0 {
        (resid.iter().take(inliers).map(|r| r * r).sum::<f64>() / inliers as f64).sqrt()
    } else {
        f64::NAN
    };

    Some(SkyFit {
        model: SkyModel {
            camera,
            axis,
            rotation,
        },
        rms_deg: rms.to_degrees(),
        inliers,
        observations: obs.len(),
        fitted_rotation_rad: fitted_scale,
        sign_source,
        hint_agreement_deg: agreement,
        focal_fitted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn north_hemisphere_facing_south_moves_left_to_right() {
        // Frame centre on the celestial equator region; stars move east to west,
        // which facing south is left to right in the image.
        let axis = nominal_axis(Hemisphere::North, Facing::S, 0.0);
        let cam = Camera::new(3000.0, 6000, 4000);
        let m = SkyModel::new(cam, axis, 20.0);
        let p = m.trail_at(cam.cx, cam.cy).unwrap();
        assert!(p.end.0 > p.start.0, "expected motion toward +x: {p:?}");
        assert!(approx(p.end.1, p.start.1, 1e-6));
        // 20 s at 15 arcsec/s = 300 arcsec; at 3000 px focal that is
        // 300/206265*3000 = 4.36 px for a star 45 degrees from the pole: times sin(45).
        let expect = SIDEREAL_RATE * 20.0 * 3000.0 * (45.0f64).to_radians().sin();
        assert!(
            approx(p.length, expect, 0.01),
            "len {} vs {}",
            p.length,
            expect
        );
    }

    #[test]
    fn north_hemisphere_facing_north_below_pole_moves_left_to_right() {
        // Below Polaris, counterclockwise circumpolar motion goes west to east,
        // which facing north is left to right.
        let axis = nominal_axis(Hemisphere::North, Facing::N, 0.0);
        let cam = Camera::new(3000.0, 6000, 4000);
        let m = SkyModel::new(cam, axis, 20.0);
        let p = m.trail_at(cam.cx, cam.cy).unwrap();
        assert!(p.end.0 > p.start.0, "{p:?}");
    }

    #[test]
    fn south_hemisphere_facing_north_moves_right_to_left() {
        // Facing north from the south, east is to the right; stars move east to west.
        let axis = nominal_axis(Hemisphere::South, Facing::N, 0.0);
        let cam = Camera::new(3000.0, 6000, 4000);
        let m = SkyModel::new(cam, axis, 20.0);
        let p = m.trail_at(cam.cx, cam.cy).unwrap();
        assert!(p.end.0 < p.start.0, "{p:?}");
    }

    #[test]
    fn facing_east_stars_rise() {
        // Facing east with a level camera: stars move upward (toward -y).
        let axis = nominal_axis(Hemisphere::North, Facing::E, 0.0);
        let cam = Camera::new(3000.0, 6000, 4000);
        let m = SkyModel::new(cam, axis, 20.0);
        let p = m.trail_at(cam.cx, cam.cy).unwrap();
        assert!(p.end.1 < p.start.1, "{p:?}");
    }

    #[test]
    fn velocity_direction_matches_exact_endpoints() {
        let axis = axis_from_angles(0.3, 0.9);
        let cam = Camera::new(2000.0, 3000, 2000);
        let m = SkyModel::new(cam, axis, 30.0);
        for (x, y) in [(10.0, 10.0), (2990.0, 1990.0), (1500.0, 100.0)] {
            let p = m.trail_at(x, y).unwrap();
            let ((vx, vy), per_rad) = m.velocity_at(x, y);
            assert!(approx(wrap_half_pi(vy.atan2(vx) - p.theta), 0.0, 1e-4));
            assert!(approx(per_rad * m.rotation, p.length, 0.05));
        }
    }

    fn synthetic_obs(model: &SkyModel, n: usize, noise_deg: f64) -> Vec<TrailObservation> {
        let mut obs = Vec::new();
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 1_000_000) as f64 / 1_000_000.0
        };
        let w = model.camera.cx * 2.0;
        let h = model.camera.cy * 2.0;
        for _ in 0..n {
            let x = rnd() * w;
            let y = rnd() * h;
            let p = model.trail_at(x, y).unwrap();
            let noise = (rnd() - 0.5) * 2.0 * noise_deg.to_radians();
            obs.push(TrailObservation {
                x,
                y,
                theta: p.theta + noise,
                length: p.length,
                weight: 1.0,
            });
        }
        obs
    }

    #[test]
    fn fit_recovers_axis_and_sign_for_all_facings() {
        let cam = Camera::new(4500.0, 6000, 4000);
        for hemi in [Hemisphere::North, Hemisphere::South] {
            for facing in [
                Facing::N,
                Facing::NE,
                Facing::E,
                Facing::SE,
                Facing::S,
                Facing::SW,
                Facing::W,
                Facing::NW,
            ] {
                // True geometry uses a different latitude and pitch than the
                // nominal guess to prove that rough hints suffice.
                let lat = match hemi {
                    Hemisphere::North => 38.0f64,
                    Hemisphere::South => -33.0,
                }
                .to_radians();
                let scp = Vector3::new(0.0, -lat.cos(), -lat.sin());
                let a = (facing.azimuth_deg() + 25.0).to_radians();
                let hh = 35.0f64.to_radians();
                let forward = Vector3::new(a.sin() * hh.cos(), a.cos() * hh.cos(), hh.sin());
                let right = Vector3::new(a.cos(), -a.sin(), 0.0);
                let down = forward.cross(&right);
                let true_axis = Vector3::new(scp.dot(&right), scp.dot(&down), scp.dot(&forward));
                let truth = SkyModel::new(cam, true_axis, 25.0);
                let obs = synthetic_obs(&truth, 400, 3.0);
                let fit = fit_sky(
                    6000,
                    4000,
                    &obs,
                    &SkyFitOptions {
                        focal_px: Some(4500.0),
                        exposure_s: Some(25.0),
                        hemisphere: Some(hemi),
                        facing: Some(facing),
                        ..Default::default()
                    },
                )
                .unwrap();
                let err = fit
                    .model
                    .axis
                    .dot(&truth.axis)
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees();
                assert!(
                    err < 1.0,
                    "{hemi:?} {facing:?}: axis error {err} deg, fit {fit:?}"
                );
            }
        }
    }

    #[test]
    fn fit_recovers_focal_length_when_unknown() {
        let cam = Camera::new(2600.0, 6000, 4000);
        let truth = SkyModel::new(cam, nominal_axis(Hemisphere::North, Facing::SW, 25.0), 20.0);
        let obs = synthetic_obs(&truth, 600, 2.0);
        let fit = fit_sky(
            6000,
            4000,
            &obs,
            &SkyFitOptions {
                focal_px: None,
                exposure_s: Some(20.0),
                hemisphere: Some(Hemisphere::North),
                facing: Some(Facing::SW),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(fit.focal_fitted);
        let err = fit
            .model
            .axis
            .dot(&truth.axis)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        assert!(err < 3.0, "axis error {err}");
        let fr = fit.model.camera.focal_px / 2600.0;
        assert!(fr > 0.7 && fr < 1.4, "focal ratio {fr}");
        let rot = fit.fitted_rotation_rad.unwrap();
        assert!(
            (rot / truth.rotation - 1.0).abs() < 0.3,
            "rotation {rot} vs {}",
            truth.rotation
        );
    }
}
