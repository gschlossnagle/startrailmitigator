# Sample images

Drop real astro-landscape frames here for testing. Any of these work:

- 16-bit TIFF (what Lightroom hands to an external editor)
- RAW: DNG, NEF, CR2/CR3, ARW, RAF, ORF, RW2, IIQ, 3FR/FFF

Large image files in this directory are tracked with Git LFS (see `.gitattributes`).
Run `git lfs install` once on your machine before adding files, then `git add` as usual.
GitHub rejects single files over 100 MB that are not in LFS.

## Per-image notes

Add a line per file to `samples/NOTES.md` (create it if missing) with whatever you know:

```
2025-08-14_milkyway_arch.NEF  hemisphere=north facing=S exposure=25s lens=24mm camera=100MP body
```

`hemisphere` and `facing` (N, NE, E, SE, S, SW, W, NW) are what the tool uses to decide which end of
each trail is the *start* of the exposure. Exposure and focal length are normally read from EXIF, so
only note them if the file has been stripped of metadata.

Outputs written while testing go in `samples/out/`, which is git-ignored.
