use std::env;
use std::path::PathBuf;
use std::time::SystemTime;

use image::ImageBuffer;
use ndarray::Ix2;

use wfc::impls::image::ImageParams;
use wfc::traits::Flips;
use wfc::traits::Recover;
use wfc::traits::Rotations;

fn main() {
    let args: Vec<String> = env::args().collect();
    let img_path = args
        .get(1)
        .map(PathBuf::from)
        .expect("a sample path is required!");

    let win_size = args
        .get(2)
        .map(|s| s.parse::<usize>().expect("valid number expected"))
        .expect("a window size is required!");

    let image = image::open(img_path).expect("image not found");

    let params = ImageParams { image, win_size };
    let mut tileset = params.tileset();
    tileset.with_rots().with_flips();

    let mut wave = tileset.wave(Ix2(70, 70));

    let t0 = SystemTime::now();
    wave.collapse(None);
    let t1 = SystemTime::now();

    let image: ImageBuffer<_, _> = wave.recover();
    image.save("wave.png").expect("failed to save image");

    println!("collapsed in {:?}", t1.duration_since(t0).unwrap());
}
