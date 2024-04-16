use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use image::Pixel;
use image::ImageBuffer;
use ndarray::Ix2;

use wfc::data::TileSet;
use wfc::traits::{Flips, Recover, Rotations};

type Img<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;

fn scale_image<P: Pixel>(image: Img<P>, scale: u32) -> Img<P> {
    let (width, height) = image.dimensions();
    let mut res: Img<P> = ImageBuffer::new(width * scale, height * scale);

    for i in 0..(width * scale) {
        for j in 0..(height * scale) {
            let pixel = image.get_pixel(i / scale, j / scale).to_owned();
            res.put_pixel(i, j, pixel);
        }
    }

    res
}

fn main() {
    let tiles_path = PathBuf::from("examples/tiles/tileset/");
    let images: Vec<_> = fs::read_dir(tiles_path)
        .expect("tileset directory not found")
        .filter_map(|file| {
            let file = file.ok();
            file.and_then(|file| image::open(file.path()).ok())
        })
        .collect();

    let mut tileset = TileSet::from_images(images);
    tileset.with_rots().with_flips();
    let mut wave = tileset.wave(Ix2(20, 20));

    let t0 = SystemTime::now();
    wave.collapse(None);
    let t1 = SystemTime::now();

    println!("collapsed in {:?}", t1.duration_since(t0).unwrap());

    println!("scaling image");
    let image: ImageBuffer<_, _> = wave.recover();
    let image = scale_image(image, 3); // resize the image
    image.save("wave.png").expect("failed to save image");
}
