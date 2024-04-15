use std::path::PathBuf;
use std::time::SystemTime;

use ndarray::Ix2;

use wfc::impls::image::ImageParams;
use wfc::traits::Flips;
use wfc::traits::Rotations;
use wfc::ext::ndarray::ArrayToImageExt;

fn main() {
    let img_path = PathBuf::from("examples/image/sample.png");
    let image = image::open(img_path).expect("image not found");

    let params = ImageParams { image, win_size: 3 };
    let mut tileset = params.tileset();
    tileset.with_rots().with_flips();

    let mut wave = tileset.wave(Ix2(60, 60));

    let t0 = SystemTime::now();
    wave.collapse(None);
    let t1 = SystemTime::now();

    let image = wave.recover().to_image().unwrap();
    image.save("wave.png").expect("failed to save image");

    println!("collapsed in {:?}", t1.duration_since(t0).unwrap());
}
