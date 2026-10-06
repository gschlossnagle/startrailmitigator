//! Mesh-based background and noise estimation (SExtractor style): sigma-clipped
//! median and MAD per tile, median-filtered across tiles, bilinearly
//! interpolated to full resolution.

use crate::image::Plane;

#[derive(Clone, Debug)]
pub struct BackgroundMap {
    pub background: Plane,
    /// Per-pixel noise sigma estimate.
    pub noise: Plane,
    pub tile: usize,
}

/// Sigma-clipped median and robust sigma (1.4826 * MAD) of a sample.
pub fn clipped_median_sigma(values: &mut Vec<f32>, iterations: usize) -> Option<(f32, f32)> {
    if values.len() < 8 {
        return None;
    }
    let mut lo = f32::NEG_INFINITY;
    let mut hi = f32::INFINITY;
    let mut result = None;
    for _ in 0..=iterations {
        values.retain(|v| *v >= lo && *v <= hi);
        if values.len() < 8 {
            break;
        }
        let med = median_inplace(values);
        let mut dev: Vec<f32> = values.iter().map(|v| (v - med).abs()).collect();
        let mad = median_inplace(&mut dev);
        let sigma = (1.4826 * mad).max(1e-7);
        result = Some((med, sigma));
        lo = med - 3.0 * sigma;
        hi = med + 3.0 * sigma;
    }
    result
}

fn median_inplace(v: &mut [f32]) -> f32 {
    let n = v.len();
    let mid = n / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let m = *m;
    if n % 2 == 0 {
        let lower = v[..mid].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        0.5 * (m + lower)
    } else {
        m
    }
}

/// Estimate background and noise maps. `mask`, when given, marks pixels to
/// ignore with values > 0.5 (e.g. foreground landscape).
pub fn estimate_background(plane: &Plane, tile: usize, mask: Option<&Plane>) -> BackgroundMap {
    let (w, h) = (plane.width, plane.height);
    let tile = tile.max(16);
    let nx = w.div_ceil(tile);
    let ny = h.div_ceil(tile);
    let mut bg_tiles = vec![f32::NAN; nx * ny];
    let mut sd_tiles = vec![f32::NAN; nx * ny];
    let mut buf: Vec<f32> = Vec::with_capacity(tile * tile);
    for ty in 0..ny {
        for tx in 0..nx {
            buf.clear();
            let x0 = tx * tile;
            let y0 = ty * tile;
            let x1 = (x0 + tile).min(w);
            let y1 = (y0 + tile).min(h);
            for y in y0..y1 {
                for x in x0..x1 {
                    if let Some(m) = mask {
                        if m.get(x, y) > 0.5 {
                            continue;
                        }
                    }
                    buf.push(plane.get(x, y));
                }
            }
            if let Some((m, s)) = clipped_median_sigma(&mut buf, 3) {
                bg_tiles[ty * nx + tx] = m;
                sd_tiles[ty * nx + tx] = s;
            }
        }
    }

    fill_nans(&mut bg_tiles, nx, ny);
    fill_nans(&mut sd_tiles, nx, ny);
    let bg_tiles = median3x3(&bg_tiles, nx, ny);
    let sd_tiles = median3x3(&sd_tiles, nx, ny);

    let background = interpolate(&bg_tiles, nx, ny, tile, w, h);
    let noise = interpolate(&sd_tiles, nx, ny, tile, w, h);
    BackgroundMap {
        background,
        noise,
        tile,
    }
}

fn fill_nans(t: &mut [f32], nx: usize, ny: usize) {
    let valid: Vec<f32> = t.iter().cloned().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        for v in t.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut global = valid.clone();
    let gmed = median_inplace(&mut global);
    // Iteratively fill from finite neighbours; anything left gets the global median.
    for _ in 0..(nx + ny) {
        let snapshot = t.to_vec();
        let mut changed = false;
        for y in 0..ny {
            for x in 0..nx {
                if snapshot[y * nx + x].is_finite() {
                    continue;
                }
                let mut acc = 0.0;
                let mut n = 0;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                        if xx < 0 || yy < 0 || xx >= nx as i64 || yy >= ny as i64 {
                            continue;
                        }
                        let v = snapshot[yy as usize * nx + xx as usize];
                        if v.is_finite() {
                            acc += v;
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    t[y * nx + x] = acc / n as f32;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for v in t.iter_mut() {
        if !v.is_finite() {
            *v = gmed;
        }
    }
}

fn median3x3(t: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut out = vec![0.0; nx * ny];
    let mut buf = Vec::with_capacity(9);
    for y in 0..ny {
        for x in 0..nx {
            buf.clear();
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx >= nx as i64 || yy >= ny as i64 {
                        continue;
                    }
                    buf.push(t[yy as usize * nx + xx as usize]);
                }
            }
            out[y * nx + x] = median_inplace(&mut buf);
        }
    }
    out
}

fn interpolate(t: &[f32], nx: usize, ny: usize, tile: usize, w: usize, h: usize) -> Plane {
    let mut out = Plane::new(w, h);
    let half = 0.5 * tile as f32;
    for y in 0..h {
        let fy = ((y as f32 + 0.5 - half) / tile as f32).clamp(0.0, (ny - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(ny - 1);
        let wy = fy - y0 as f32;
        for x in 0..w {
            let fx = ((x as f32 + 0.5 - half) / tile as f32).clamp(0.0, (nx - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(nx - 1);
            let wx = fx - x0 as f32;
            let a = t[y0 * nx + x0];
            let b = t[y0 * nx + x1];
            let c = t[y1 * nx + x0];
            let d = t[y1 * nx + x1];
            let top = a + (b - a) * wx;
            let bot = c + (d - c) * wx;
            out.data[y * w + x] = top + (bot - top) * wy;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_flat_background_and_noise() {
        let (w, h) = (256, 192);
        let mut p = Plane::new(w, h);
        let mut seed = 99u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 10_000) as f32 / 10_000.0
        };
        for y in 0..h {
            for x in 0..w {
                // Approximate Gaussian noise with sigma ~0.01 via sum of uniforms.
                let n: f32 = (0..12).map(|_| rnd()).sum::<f32>() - 6.0;
                p.set(x, y, 0.1 + 0.001 * x as f32 / w as f32 + 0.01 * n);
            }
        }
        // Plant some bright stars that must not bias the estimate.
        for i in 0..40 {
            let x = (i * 37) % w;
            let y = (i * 53) % h;
            p.set(x, y, 1.0);
        }
        let bg = estimate_background(&p, 32, None);
        for y in (0..h).step_by(17) {
            for x in (0..w).step_by(19) {
                let v = bg.background.get(x, y);
                assert!((v - 0.1).abs() < 0.004, "bg {v} at {x},{y}");
                let s = bg.noise.get(x, y);
                assert!(s > 0.007 && s < 0.013, "noise {s}");
            }
        }
    }
}
