use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use strm::image::Plane;
use strm::pipeline::{self, Params};
use strm::repair::Anchor;
use strm::sky::{Facing, Hemisphere};
use strm::synth::{self, SynthParams};

#[derive(Parser)]
#[command(
    name = "strm",
    version,
    about = "Star trail mitigator: round off trailed stars in astro-landscape frames"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// Verbose logging.
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Repair an image and write the result.
    Fix {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        opts: ProcessArgs,
    },
    /// Detect and fit trails without writing an image.
    Analyze {
        input: PathBuf,
        #[command(flatten)]
        opts: ProcessArgs,
    },
    /// Render a synthetic star field with ground truth.
    Synth {
        #[arg(short, long)]
        output: PathBuf,
        /// JSON file for the ground truth.
        #[arg(long)]
        truth: Option<PathBuf>,
        /// "wide" (45 MP, 14 mm, 20 s) or "long" (100 MP, 25 mm, 30 s), scaled to the size.
        #[arg(long, default_value = "long")]
        preset: String,
        #[arg(long, default_value_t = 2000)]
        width: usize,
        #[arg(long, default_value_t = 1400)]
        height: usize,
        #[arg(long)]
        exposure: Option<f64>,
        #[arg(long)]
        focal_length_px: Option<f64>,
        #[arg(long)]
        psf_sigma: Option<f64>,
        #[arg(long)]
        stars: Option<usize>,
        #[arg(long)]
        saturated: Option<usize>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, value_enum)]
        hemisphere: Option<Hemisphere>,
        #[arg(long, value_enum)]
        facing: Option<Facing>,
    },
}

#[derive(Args, Clone)]
struct ProcessArgs {
    /// Hemisphere the photo was taken from. Decides the direction of motion.
    #[arg(long, value_enum)]
    hemisphere: Option<Hemisphere>,
    /// Rough compass direction the frame centre points toward (N, NE, E, ...).
    #[arg(long, value_enum)]
    facing: Option<Facing>,
    /// Reverse the motion sense the hints would otherwise choose.
    #[arg(long)]
    flip_motion: bool,
    /// Exposure time in seconds (overrides EXIF).
    #[arg(long)]
    exposure: Option<f64>,
    /// Focal length in pixels (overrides EXIF-derived value).
    #[arg(long)]
    focal_length_px: Option<f64>,
    /// Sensor width in mm, to turn an EXIF focal length into pixels.
    #[arg(long)]
    sensor_width: Option<f64>,
    /// Detection threshold in noise sigmas.
    #[arg(long, default_value_t = 4.0)]
    threshold: f32,
    /// Leave stars alone whose axis ratio (length + width) / width is below this.
    #[arg(long, default_value_t = 1.25)]
    min_elongation: f64,
    /// Longest trail the tool should consider, in pixels.
    #[arg(long, default_value_t = 48.0)]
    max_trail_px: f64,
    /// Which end of the trail the round star is placed at.
    #[arg(long, value_enum, default_value_t = Anchor::Start)]
    anchor: Anchor,
    /// Multiplier on the rebuilt star's brightness.
    #[arg(long, default_value_t = 1.0)]
    star_gain: f32,
    /// Sky mask image (PNG or TIFF): white = sky, black = ignore (landscape).
    #[arg(long)]
    mask: Option<PathBuf>,
    /// Process display-encoded TIFFs in linear light (power 2.2).
    #[arg(long)]
    linear: bool,
    /// Directory for overlay.png and modified.png diagnostics.
    #[arg(long)]
    debug_dir: Option<PathBuf>,
    /// Write a JSON report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

impl ProcessArgs {
    fn params(&self) -> Params {
        let mut p = Params {
            hemisphere: self.hemisphere,
            facing: self.facing,
            flip_motion: self.flip_motion,
            exposure_s: self.exposure,
            focal_px: self.focal_length_px,
            sensor_width_mm: self.sensor_width,
            max_trail_px: self.max_trail_px,
            linearize: self.linear,
            ..Default::default()
        };
        p.detect.threshold_sigma = self.threshold;
        p.repair.min_axis_ratio = self.min_elongation;
        p.repair.anchor = self.anchor;
        p.repair.star_gain = self.star_gain;
        p
    }
}

fn load_mask(path: &Path, w: usize, h: usize) -> Result<Plane> {
    let is_tiff = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "tif" | "tiff"))
        .unwrap_or(false);
    let plane = if is_tiff {
        let img = strm::io::load(path)?;
        img.luminance()
    } else {
        let img = image::open(path)
            .with_context(|| format!("reading mask {}", path.display()))?
            .to_luma8();
        let (mw, mh) = (img.width() as usize, img.height() as usize);
        Plane::from_vec(
            mw,
            mh,
            img.into_raw()
                .into_iter()
                .map(|v| v as f32 / 255.0)
                .collect(),
        )
    };
    if plane.width != w || plane.height != h {
        anyhow::bail!(
            "mask is {}x{} but the image is {}x{}",
            plane.width,
            plane.height,
            w,
            h
        );
    }
    // Our convention internally: >0.5 means ignore. The mask file uses white = sky.
    let mut m = plane;
    m.map_inplace(|v| 1.0 - v);
    Ok(m)
}

fn print_report(r: &pipeline::Report) {
    println!("image: {}x{}", r.width, r.height);
    if let Some(e) = r.exposure_s {
        println!("exposure: {e:.2} s");
    }
    if let Some(f) = r.focal_px {
        println!("focal length: {f:.0} px");
    }
    println!(
        "detections: {}, fitted: {}, sky-fit votes: {}",
        r.detections, r.fitted, r.sky_fit_votes
    );
    if let Some(s) = &r.sky {
        println!(
            "sky fit: rms {:.2} deg over {}/{} inliers; sign from {:?}{}",
            s.rms_deg,
            s.inliers,
            s.observations,
            s.sign_source,
            s.hint_agreement_deg
                .map(|a| format!(", {a:.0} deg from the hint guess"))
                .unwrap_or_default()
        );
        if let Some(fr) = s.fitted_rotation_rad {
            println!(
                "rotation from trail lengths: {:.1} s equivalent (model uses {:.1} s)",
                fr / strm::sky::SIDEREAL_RATE,
                s.model.rotation / strm::sky::SIDEREAL_RATE
            );
        }
    }
    println!(
        "median axis ratio: {:.2} (fitted), {:.2} (moments)",
        r.median_elongation, r.median_moment_elongation
    );
    println!("motion: {}", r.motion_summary);
    if let Some(rep) = &r.repair {
        println!(
            "repaired {} stars ({} model-subtracted, {} patch-filled); skipped {} round, {} unconverged, {} misaligned",
            rep.repaired,
            rep.model_subtracted,
            rep.patch_filled,
            rep.skipped_round,
            rep.skipped_unconverged,
            rep.skipped_misaligned
        );
    }
    for w in &r.warnings {
        println!("warning: {w}");
    }
    println!("elapsed: {:.1} s", r.seconds);
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_default_env()
        .filter_level(if cli.verbose {
            log::LevelFilter::Info
        } else {
            log::LevelFilter::Warn
        })
        .init();
    match cli.cmd {
        Cmd::Fix {
            input,
            output,
            opts,
        } => {
            let mut img = strm::io::load(&input)?;
            let mask = opts
                .mask
                .as_ref()
                .map(|m| load_mask(m, img.width(), img.height()))
                .transpose()?;
            let params = opts.params();
            let before = if opts.debug_dir.is_some() {
                img.clone()
            } else {
                strm::image::Image::new(1, 1, 1)
            };
            let (analysis, modified) = pipeline::fix(&mut img, &params, mask.as_ref())?;
            strm::io::save(&output, &img)?;
            println!("wrote {}", output.display());
            print_report(&analysis.report);
            if let Some(dir) = &opts.debug_dir {
                strm::debug::write_debug(dir, &analysis, Some(&modified))?;
                strm::debug::write_crops(dir, &before, &img, &analysis, 24)?;
                println!("diagnostics in {}", dir.display());
            }
            if let Some(path) = &opts.report {
                std::fs::write(path, serde_json::to_string_pretty(&analysis.report)?)?;
            }
        }
        Cmd::Analyze { input, opts } => {
            let img = strm::io::load(&input)?;
            let mask = opts
                .mask
                .as_ref()
                .map(|m| load_mask(m, img.width(), img.height()))
                .transpose()?;
            let params = opts.params();
            let analysis = pipeline::analyze(&img, &params, mask.as_ref())?;
            print_report(&analysis.report);
            if let Some(dir) = &opts.debug_dir {
                strm::debug::write_debug(dir, &analysis, None)?;
                println!("diagnostics in {}", dir.display());
            }
            if let Some(path) = &opts.report {
                std::fs::write(path, serde_json::to_string_pretty(&analysis.report)?)?;
            }
        }
        Cmd::Synth {
            output,
            truth,
            preset,
            width,
            height,
            exposure,
            focal_length_px,
            psf_sigma,
            stars,
            saturated,
            seed,
            hemisphere,
            facing,
        } => {
            let mut p = match preset.as_str() {
                "wide" => SynthParams::preset_wide(width, height),
                "long" => SynthParams::preset_long(width, height),
                other => anyhow::bail!("unknown preset {other}; use wide or long"),
            };
            if let Some(v) = exposure {
                p.exposure_s = v;
            }
            if let Some(v) = focal_length_px {
                p.focal_px = v;
            }
            if let Some(v) = psf_sigma {
                p.psf_sigma = v;
            }
            if let Some(v) = stars {
                p.n_stars = v;
            }
            if let Some(v) = saturated {
                p.saturated = v;
            }
            if let Some(v) = seed {
                p.seed = v;
            }
            if hemisphere.is_some() || facing.is_some() {
                p.axis = strm::sky::nominal_axis(
                    hemisphere.unwrap_or(Hemisphere::North),
                    facing.unwrap_or(Facing::S),
                    25.0,
                );
            }
            let (img, t) = synth::generate(&p);
            strm::io::save(&output, &img)?;
            println!(
                "wrote {} ({} stars, {:.1} px typical trail)",
                output.display(),
                t.stars.len(),
                {
                    let mut l: Vec<f64> = t.stars.iter().map(|s| s.length).collect();
                    l.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    l.get(l.len() / 2).cloned().unwrap_or(0.0)
                }
            );
            if let Some(tp) = truth {
                std::fs::write(&tp, serde_json::to_string_pretty(&t)?)?;
                println!("truth in {}", tp.display());
            }
        }
    }
    Ok(())
}
