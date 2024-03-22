use std::time::SystemTime;

use image::io::Reader as ImageReader;
use ndarray::Ix2;
use wfc::{data::{Flips, Rotations}, impls::image::ImageParams};

// TODO: reexport needed types from image crate?

fn main() {
    let image = ImageReader::open("/home/al/files/github/wfc/examples/simple/sample.png")
        .expect("couldn't find image")
        .decode()
        .expect("failed to decode the image");

    let image = image.into_rgb8();

    let params = ImageParams {
        image,
        win_size: 3
    };

    let mut tile_set = params.tile_set();
    let mut wave = tile_set
        .with_rots()
        .with_flips()
        .wave(Ix2(10, 10));

    let t0 = SystemTime::now();
    wave.collapse(None);
    let t1 = SystemTime::now();

    println!("collapsed in {:?}", t1.duration_since(t0).unwrap());
}
