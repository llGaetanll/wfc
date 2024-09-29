use std::env;
use std::path::PathBuf;
use std::time::SystemTime;

use image::ImageBuffer;
use image::Pixel;
use wfc::prelude::*;
use wfc::wave::traits::ParWave;

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

    let params = ImageParams::<_, Flat>::new_flat(image, win_size);
    let mut tileset = params.tileset();
    tileset.with_rots().with_flips();

    let mut wave = ImageWave::init(&mut tileset, (10, 10));

    let mut rng = rand::thread_rng();

    let t0 = SystemTime::now();
    let image = wave.collapse_parallel(&mut rng);
    let t1 = SystemTime::now();

    let image = scale_image(image, 10); // resize the image
    image.save("wave.png").expect("failed to save image");

    println!("collapsed in {:?}", t1.duration_since(t0).unwrap());
}
