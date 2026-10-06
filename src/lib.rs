//! Star Trail Mitigator core library.
//!
//! Detects elongated (trailed) stars in wide-angle astro-landscape frames, fits a
//! global sky-rotation model to decide which end of each trail is the *start* of the
//! exposure, and replaces each trail with a round star of the trail's width anchored
//! at that start, filling the removed trail with matching sky background.
