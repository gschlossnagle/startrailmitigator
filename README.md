# startrailmitigator

Eliminates or mitigates star trailing in wide-angle astro-landscape photographs.

Modern high-resolution sensors show visible star elongation even in exposures that honor the
500 or NPF rule. This tool finds each elongated star, works out which end of the trail is the
*start* of the exposure, and replaces the trail with a round star whose diameter equals the
trail's width, anchored at that start. The removed part of the trail is filled with sky that
matches the surrounding background.

Status: working proof of concept. Rust core library plus a `strm` command-line tool that reads
8/16-bit TIFF and camera RAW files and writes 16-bit TIFF. Tested end to end against synthetic
frames with known trail geometry; real-image tuning is in progress (drop test frames in
`samples/`, see `samples/README.md`).

## Build

```
cargo build --release
./target/release/strm --help
```

## Usage

```
# Repair a frame shot from the northern hemisphere, camera pointed roughly south.
strm fix IMG_1234.NEF -o IMG_1234_round.tif --hemisphere north --facing S

# Lightroom hands external editors a 16-bit TIFF; process it the same way.
strm fix IMG_1234.tif -o IMG_1234_round.tif --hemisphere north --facing S --linear

# Inspect what the tool would do without writing an image.
strm analyze IMG_1234.NEF --hemisphere north --facing S --debug-dir dbg/ --report report.json

# Make a synthetic test frame (100 MP, 25 mm, 30 s crop) with ground truth.
strm synth -o field.tif --truth truth.json --preset long --width 2000 --height 1400
```

Key options for `fix` and `analyze`:

| Option | Meaning |
|---|---|
| `--hemisphere north\|south` | Which hemisphere the photo was taken from. Required for a reliable motion sense; assumed north with a warning when missing. |
| `--facing N\|NE\|E\|SE\|S\|SW\|W\|NW` | Rough compass direction of the frame centre. Only needs to be within a few tens of degrees. |
| `--flip-motion` | Reverse the motion sense if the overlay shows it chose wrong. |
| `--anchor start\|end\|center` | Which end of the trail the round star is placed at. Default `start`. |
| `--exposure SEC`, `--focal-length-px PX`, `--sensor-width MM` | Override or supplement EXIF. Without them the trail scale and focal length are fitted from the data. |
| `--max-trail-px` | Longest trail to consider (default 48). |
| `--min-elongation` | Leave stars alone whose (length + width) / width is below this (default 1.25). |
| `--threshold` | Detection threshold in noise sigmas (default 4). |
| `--star-gain` | Multiplier on the rebuilt stars' brightness (default 1). |
| `--mask FILE` | PNG/TIFF sky mask, white = sky, black = ignore (landscape). |
| `--linear` | Process display-encoded TIFFs in linear light (power 2.2). RAW input is already linear. |
| `--debug-dir DIR` | Writes `overlay.png` (detections, fitted trails, start markers, motion arrows), `modified.png` and `crops.png` (before/after magnified crops). |
| `--report FILE` | JSON report with the fitted sky model, counts and warnings. |

The tool prints the motion it assumed in words ("stars move left to right ... anchored at
their starting end"). Check it against the arrows in `overlay.png` the first time you process
a new shooting setup.

## How it works

1. **Background and noise maps** from a sigma-clipped mesh (tiles of at least twice the
   maximum trail length), so Milky Way gradients and vignetting are followed.
2. **Detection** by thresholding the matched-filtered, background-subtracted luminance above
   the local noise; connected components are measured with intensity-weighted moments for
   centroid, orientation, major/minor widths.
3. **Sky-rotation model**: the apparent motion of every star is a rotation about the
   celestial axis seen through a pinhole camera. The axis direction is fitted to all the
   measured trail orientations (grid search plus robust refinement, focal length fitted too
   when EXIF lacks it). Orientations alone cannot tell the sense of motion, which is the one
   bit the `--hemisphere` and `--facing` hints resolve. The model then predicts direction,
   sense and length of the trail at every pixel.
4. **Per-star fit** of an analytic trailed-PSF model (a Gaussian convolved with a line
   segment, Vereš et al. 2012) by Levenberg-Marquardt on luminance, with a prior toward the
   sky model that strengthens for faint stars. Per-channel amplitudes are solved linearly so
   each star keeps its colour. Saturated pixels are masked.
5. **Repair**: the fitted trail is subtracted from every channel (leaving the real sky and
   its noise), and a round Gaussian star of the trail's PSF width and peak colour is added at
   the start of the trail. Where the model does not explain the star to the noise floor
   (saturated cores, PSF wings), the footprint is instead overwritten with sky borrowed from a
   strip alongside the trail, level-corrected with the background map so gradients survive.

Expected trail lengths: sky drift is 15 arcsec/s, so a 45 MP full-frame body at 14 mm for 20 s
trails ~5 px, and a 100 MP body at 25 mm for 30 s trails ~19 px. The analytic model works
at sub-pixel precision, which matters for the shorter case.

## Testing

```
cargo test
```

Unit tests cover the model, geometry, I/O round trips, background, detection and the fit.
`tests/synthetic.rs` renders frames at both scales, runs the full pipeline, and checks that
the predicted trail direction and length match the truth, that stars come out round, that
bright stars sit within half a pixel of their true starting position, and that the filled
pixels match the noise-free sky to within the noise.

## Known limitations

- Overlapping trails are not deblended; a faint star inside a bright neighbour's trail is
  removed along with it.
- Lens distortion is not modelled. Strong wide-angle distortion would show up as a radial
  pattern in the orientation residuals of the sky fit.
- Output TIFFs carry the ICC profile and the EXIF fields the tool uses (exposure, focal
  length, camera), not the full EXIF block.
- RAW decoding uses rawler's demosaic; output is scene-linear with sRGB primaries, which a
  raw editor will want to tone-map.

## Roadmap

- Lightroom Classic plugin: a Lua external-editor plugin that runs `strm fix` on the TIFF
  Lightroom hands over.
- Photoshop: expose the core over a C ABI for a filter plugin.
- Deblending of merged detections, radial distortion term, optional flux-conserving star
  brightness, lower memory footprint for 100 MP frames.
