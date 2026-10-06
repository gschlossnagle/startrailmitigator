//! Diagnostic images: a stretched luminance overlay with detections, fitted
//! trails, start markers and a grid of motion arrows.

use crate::image::{Image, Plane};
use crate::pipeline::Analysis;
use anyhow::Result;
use image::{Rgb, RgbImage};
use std::path::Path;

fn stretch(v: f32, bg: f32, noise: f32) -> u8 {
    // asinh stretch around the background, a few sigma to mid grey.
    let x = (v - bg) / (noise.max(1e-5) * 20.0);
    let y = 0.25 + 0.5 * (x.asinh() / 4.0f32.asinh());
    (y.clamp(0.0, 1.0) * 255.0) as u8
}

fn line(img: &mut RgbImage, x0: f64, y0: f64, x1: f64, y1: f64, colour: Rgb<u8>) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let n = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize).max(1);
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let x = (x0 + t * (x1 - x0)).round() as i64;
        let y = (y0 + t * (y1 - y0)).round() as i64;
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, colour);
        }
    }
}

fn dot(img: &mut RgbImage, x: f64, y: f64, r: i64, colour: Rgb<u8>) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (cx, cy) = (x.round() as i64, y.round() as i64);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let (px, py) = (cx + dx, cy + dy);
                if px >= 0 && py >= 0 && px < w && py < h {
                    img.put_pixel(px as u32, py as u32, colour);
                }
            }
        }
    }
}

/// Write `overlay.png` (annotated frame) and `modified.png` (repaired pixels).
pub fn write_debug(dir: &Path, analysis: &Analysis, modified: Option<&Plane>) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let lum = &analysis.lum;
    let (w, h) = (lum.width as u32, lum.height as u32);
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let g = stretch(
                lum.data[i],
                analysis.bg.background.data[i],
                analysis.bg.noise.data[i],
            );
            img.put_pixel(x, y, Rgb([g, g, g]));
        }
    }
    let blue = Rgb([80, 120, 255]);
    let green = Rgb([60, 230, 60]);
    let red = Rgb([255, 50, 50]);
    let yellow = Rgb([255, 220, 0]);
    for s in &analysis.stars {
        let d = &s.det;
        let (x0, y0, x1, y1) = (
            d.x_min as f64,
            d.y_min as f64,
            d.x_max as f64,
            d.y_max as f64,
        );
        line(&mut img, x0, y0, x1, y0, blue);
        line(&mut img, x1, y0, x1, y1, blue);
        line(&mut img, x1, y1, x0, y1, blue);
        line(&mut img, x0, y1, x0, y0, blue);
        if let Some(f) = &s.fit {
            line(&mut img, f.start.0, f.start.1, f.end.0, f.end.1, green);
            dot(&mut img, f.start.0, f.start.1, 1, red);
        }
    }
    if let Some(sky) = &analysis.sky {
        let step = (w.max(h) / 12).max(40) as f64;
        let mut y = step * 0.5;
        while y < h as f64 {
            let mut x = step * 0.5;
            while x < w as f64 {
                if let Some(p) = sky.model.trail_at(x, y) {
                    let (dx, dy) = (p.end.0 - p.start.0, p.end.1 - p.start.1);
                    let n = (dx * dx + dy * dy).sqrt().max(1e-9);
                    let len = step * 0.3;
                    let (ux, uy) = (dx / n, dy / n);
                    let (ex, ey) = (x + ux * len, y + uy * len);
                    line(&mut img, x, y, ex, ey, yellow);
                    // Arrow head.
                    line(
                        &mut img,
                        ex,
                        ey,
                        ex - (ux - uy * 0.6) * 6.0,
                        ey - (uy + ux * 0.6) * 6.0,
                        yellow,
                    );
                    line(
                        &mut img,
                        ex,
                        ey,
                        ex - (ux + uy * 0.6) * 6.0,
                        ey - (uy - ux * 0.6) * 6.0,
                        yellow,
                    );
                }
                x += step;
            }
            y += step;
        }
    }
    img.save(dir.join("overlay.png"))?;
    if let Some(m) = modified {
        let mut mi = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if m.data[(y * w + x) as usize] > 0.5 {
                    255
                } else {
                    0
                };
                mi.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        mi.save(dir.join("modified.png"))?;
    }
    Ok(())
}

/// Write `crops.png`: before/after pairs of the brightest repaired stars,
/// magnified, with the same stretch on both sides. A red tick above each
/// "after" tile marks the fitted start position.
pub fn write_crops(
    dir: &Path,
    before: &Image,
    after: &Image,
    analysis: &Analysis,
    n: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let (w, h) = (before.width(), before.height());
    let half: i64 = 14;
    let zoom: u32 = 6;
    let tile = (2 * half as u32 + 1) * zoom;
    let per_row = 4;
    let mut stars: Vec<&crate::pipeline::StarRecord> = analysis
        .stars
        .iter()
        .filter(|s| s.fit.is_some())
        .filter(|s| {
            let (x, y) = (s.moments.xc as i64, s.moments.yc as i64);
            x > half + 1 && y > half + 1 && x < w as i64 - half - 2 && y < h as i64 - half - 2
        })
        .collect();
    stars.sort_by(|a, b| {
        let fa = a.fit.as_ref().unwrap().amp_lum;
        let fb = b.fit.as_ref().unwrap().amp_lum;
        fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
    });
    // Mix of bright and typical stars: half from the top, half spread over the rest.
    let mut picked: Vec<&crate::pipeline::StarRecord> = stars.iter().take(n / 2).cloned().collect();
    let rest = &stars[(n / 2).min(stars.len())..];
    if !rest.is_empty() {
        let step = (rest.len() / (n - n / 2)).max(1);
        picked.extend(rest.iter().step_by(step).take(n - n / 2).cloned());
    }
    if picked.is_empty() {
        return Ok(());
    }
    let rows = picked.len().div_ceil(per_row);
    let gap = 4;
    let out_w = per_row as u32 * (2 * tile + 3 * gap);
    let out_h = rows as u32 * (tile + gap + 6);
    let mut img = RgbImage::from_pixel(out_w, out_h, Rgb([30, 30, 30]));
    for (k, s) in picked.iter().enumerate() {
        let f = s.fit.as_ref().unwrap();
        let (cx, cy) = (s.moments.xc.round() as i64, s.moments.yc.round() as i64);
        let i = (cy as usize) * w + cx as usize;
        let bg = analysis.bg.background.data[i];
        let noise = analysis.bg.noise.data[i];
        let lo = bg - 2.0 * noise;
        let hi = (bg + f.amp_lum as f32 * 1.1).max(lo + 1e-4);
        let col = (k % per_row) as u32;
        let row = (k / per_row) as u32;
        let ox = col * (2 * tile + 3 * gap) + gap;
        let oy = row * (tile + gap + 6) + 6;
        for (which, src) in [before, after].iter().enumerate() {
            let tx = ox + which as u32 * (tile + gap);
            for dy in -half..=half {
                for dx in -half..=half {
                    let (x, y) = ((cx + dx) as usize, (cy + dy) as usize);
                    let px = y * w + x;
                    let mut rgb = [0u8; 3];
                    for (c, out) in rgb.iter_mut().enumerate() {
                        let v = src.planes[c.min(src.channels() - 1)].data[px];
                        let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0).powf(0.5);
                        *out = (t * 255.0) as u8;
                    }
                    for zy in 0..zoom {
                        for zx in 0..zoom {
                            img.put_pixel(
                                tx + (dx + half) as u32 * zoom + zx,
                                oy + (dy + half) as u32 * zoom + zy,
                                Rgb(rgb),
                            );
                        }
                    }
                }
            }
        }
        // Start marker above the "after" tile.
        let sx = ox + tile + gap + ((f.start.0 - cx as f64 + half as f64) * zoom as f64) as u32;
        for yy in 0..5 {
            if sx < out_w {
                img.put_pixel(sx, oy - 6 + yy, Rgb([255, 40, 40]));
            }
        }
    }
    img.save(dir.join("crops.png"))?;
    Ok(())
}
