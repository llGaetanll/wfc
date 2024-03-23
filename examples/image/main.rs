use std::path::PathBuf;
use std::time::SystemTime;

use ndarray::Ix2;

use wfc::data::Flips;
use wfc::data::Rotations;
use wfc::impls::image::ImageParams;

fn main() {
    let img_path = PathBuf::from("examples/image/sample.png");
    let image = image::open(img_path).expect("image not found");
    let image = image.into_rgb8();

    let params = ImageParams { image, win_size: 3 };
    let mut tile_set = params.tile_set();
    let mut wave = tile_set.with_rots().with_flips().wave(Ix2(10, 10));

    let t0 = SystemTime::now();
    wave.collapse(None);
    let t1 = SystemTime::now();

    println!("collapsed in {:?}", t1.duration_since(t0).unwrap());
}
