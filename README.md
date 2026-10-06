# startrailmitigator

Eliminates or mitigates star trailing in wide-angle astro-landscape photographs.

Modern high-resolution sensors show visible star elongation even in exposures that honor the
500 or NPF rule. This tool finds each elongated star, works out which end of the trail is the
*start* of the exposure, and replaces the trail with a round star whose diameter equals the
trail's width, anchored at that start. The removed part of the trail is filled with sky that
matches the surrounding background.

Status: proof of concept in progress. Rust core library plus a `strm` command-line tool that
reads 8/16-bit TIFF and camera RAW files and writes 16-bit TIFF.

## Planned usage

```
strm fix IN -o OUT.tif --hemisphere north --facing S
strm analyze IN --hemisphere north --facing S --report report.json --debug-dir dbg/
strm synth -o field.tif --truth truth.json
```

`--hemisphere` and `--facing` (N, NE, E, SE, S, SW, W, NW) only need to be roughly right; they
resolve which direction the sky was moving so that trails are anchored consistently.
Exposure time and focal length are read from EXIF when present.

## How it works

1. Background and noise maps from a sigma-clipped mesh.
2. Star detection by thresholding above the local noise; moments give each blob's orientation and length.
3. A global sky-rotation model (rotation about the celestial axis seen through a pinhole camera) is fit
   to all the trail orientations, which predicts trail direction, length, and sense everywhere in the frame.
4. Each trail is fit with an analytic trailed-PSF model (a Gaussian convolved with a line segment).
5. The fitted trail is subtracted, a round star of the same peak brightness and color is added at the
   trail's start, and any residue from bright stars is filled with sky borrowed from alongside the trail.

## Samples

Put test images in `samples/` (see `samples/README.md`); large files go through Git LFS.

## Roadmap

- Lightroom Classic plugin: a Lua external-editor plugin that runs `strm fix` on the TIFF Lightroom hands over.
- Photoshop: expose the core over a C ABI for a filter plugin.
