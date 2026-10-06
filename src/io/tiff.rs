//! TIFF reading and writing via the `tiff` crate.

use crate::image::{Image, Plane, SourceFormat};
use anyhow::{anyhow, bail, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{colortype, TiffEncoder};
use tiff::tags::Tag;
use tiff::ColorType;

pub fn load_tiff(path: &Path) -> Result<Image> {
    let file = File::open(path)?;
    let mut dec =
        Decoder::new(BufReader::new(file))?.with_limits(tiff::decoder::Limits::unlimited());
    let (w, h) = dec.dimensions()?;
    let (width, height) = (w as usize, h as usize);
    let color = dec.colortype()?;
    let channels = match color {
        ColorType::Gray(_) => 1,
        ColorType::RGB(_) => 3,
        ColorType::RGBA(_) => 4,
        other => bail!("unsupported TIFF color type {other:?}"),
    };
    let icc = dec.get_tag_u8_vec(Tag::IccProfile).ok();
    let (planes, bits) = match dec.read_image()? {
        DecodingResult::U8(buf) => (deinterleave(&buf, width, height, channels, 1.0 / 255.0), 8),
        DecodingResult::U16(buf) => (
            deinterleave(&buf, width, height, channels, 1.0 / 65535.0),
            16,
        ),
        DecodingResult::F32(buf) => (deinterleave(&buf, width, height, channels, 1.0), 32),
        other => bail!(
            "unsupported TIFF sample format {:?}",
            std::mem::discriminant(&other)
        ),
    };
    // Drop alpha; keep 1 or 3 planes.
    let planes: Vec<Plane> = planes.into_iter().take(3).collect();
    let mut img = Image::from_planes(planes);
    img.format = SourceFormat {
        bits,
        icc_profile: icc,
        linear: false,
    };
    img.shot = super::exif::read_exif(path).unwrap_or_default();
    Ok(img)
}

fn deinterleave<T: Copy + Into<f32>>(
    buf: &[T],
    width: usize,
    height: usize,
    channels: usize,
    scale: f32,
) -> Vec<Plane> {
    let n = width * height;
    let mut planes: Vec<Plane> = (0..channels).map(|_| Plane::new(width, height)).collect();
    for i in 0..n {
        for (c, plane) in planes.iter_mut().enumerate() {
            plane.data[i] = buf[i * channels + c].into() * scale;
        }
    }
    planes
}

pub fn save_tiff(path: &Path, image: &Image) -> Result<()> {
    let (w, h, c) = (image.width(), image.height(), image.channels());
    if c != 1 && c != 3 {
        bail!("can only write 1- or 3-channel images, got {c}");
    }
    let file = File::create(path)?;
    let mut enc = TiffEncoder::new(BufWriter::new(file))?;
    let n = w * h;
    match (image.format.bits, c) {
        (8, 1) => {
            let data: Vec<u8> = (0..n).map(|i| to_u8(image.planes[0].data[i])).collect();
            let mut img = enc.new_image::<colortype::Gray8>(w as u32, h as u32)?;
            write_icc(img.encoder(), image)?;
            img.write_data(&data)?;
        }
        (8, 3) => {
            let mut data = Vec::with_capacity(n * 3);
            for i in 0..n {
                for p in &image.planes {
                    data.push(to_u8(p.data[i]));
                }
            }
            let mut img = enc.new_image::<colortype::RGB8>(w as u32, h as u32)?;
            write_icc(img.encoder(), image)?;
            img.write_data(&data)?;
        }
        (_, 1) => {
            let data: Vec<u16> = (0..n).map(|i| to_u16(image.planes[0].data[i])).collect();
            let mut img = enc.new_image::<colortype::Gray16>(w as u32, h as u32)?;
            write_icc(img.encoder(), image)?;
            img.write_data(&data)?;
        }
        (_, 3) => {
            let mut data = Vec::with_capacity(n * 3);
            for i in 0..n {
                for p in &image.planes {
                    data.push(to_u16(p.data[i]));
                }
            }
            let mut img = enc.new_image::<colortype::RGB16>(w as u32, h as u32)?;
            write_icc(img.encoder(), image)?;
            img.write_data(&data)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn write_icc<W: std::io::Write + std::io::Seek, K: tiff::encoder::TiffKind>(
    dir: &mut tiff::encoder::DirectoryEncoder<'_, W, K>,
    image: &Image,
) -> Result<()> {
    if let Some(icc) = &image.format.icc_profile {
        dir.write_tag(Tag::IccProfile, &icc[..])
            .map_err(|e| anyhow!("writing ICC profile: {e}"))?;
    }
    Ok(())
}

#[inline]
fn to_u16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb16_round_trip_is_lossless_and_keeps_icc() {
        let dir = std::env::temp_dir().join(format!("strm-tiff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.tif");
        let (w, h) = (7, 5);
        let mut img = Image::new(w, h, 3);
        for c in 0..3 {
            for i in 0..w * h {
                img.planes[c].data[i] = ((i * 7 + c * 1000) % 65536) as f32 / 65535.0;
            }
        }
        img.format.icc_profile = Some(vec![1, 2, 3, 4, 5]);
        save_tiff(&path, &img).unwrap();
        let back = load_tiff(&path).unwrap();
        assert_eq!(back.width(), w);
        assert_eq!(back.height(), h);
        assert_eq!(back.channels(), 3);
        assert_eq!(back.format.bits, 16);
        assert_eq!(back.format.icc_profile, Some(vec![1, 2, 3, 4, 5]));
        for c in 0..3 {
            for i in 0..w * h {
                assert!((back.planes[c].data[i] - img.planes[c].data[i]).abs() < 1e-6);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gray8_round_trip() {
        let dir = std::env::temp_dir().join(format!("strm-tiff8-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g8.tif");
        let mut img = Image::new(4, 3, 1);
        for (i, v) in img.planes[0].data.iter_mut().enumerate() {
            *v = (i * 20) as f32 / 255.0;
        }
        img.format.bits = 8;
        save_tiff(&path, &img).unwrap();
        let back = load_tiff(&path).unwrap();
        assert_eq!(back.format.bits, 8);
        for i in 0..12 {
            assert!((back.planes[0].data[i] - img.planes[0].data[i]).abs() < 1e-6);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
