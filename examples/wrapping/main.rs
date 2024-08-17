use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use image::DynamicImage;
use image::ImageBuffer;
use image::Pixel;
use image::Rgba;
use wfc::prelude::*;
use wfc::surface::Surface;
use wfc::wave::Wave;

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

fn build_wave<S: Surface<2>>(
    images: Vec<DynamicImage>,
    shape: (usize, usize),
) -> ImageWave<Rgba<u8>, S> {
    let mut tileset = TileSet::from_images(images);
    tileset.with_rots().with_flips();

    Wave::init(&mut tileset, shape)
}

fn main() {
    let tiles_path = PathBuf::from("examples/samples/tiles/");
    let images: Vec<_> = fs::read_dir(tiles_path)
        .expect("tileset directory not found")
        .filter_map(|file| {
            let file = file.ok();
            file.and_then(|file| image::open(file.path()).ok())
        })
        .collect();

    let shape = (3, 3);
    let mut rng = rand::thread_rng();

    {
        let mut wave = build_wave::<Flat>(images.clone(), shape);

        let t0 = SystemTime::now();
        let image = wave.collapse(&mut rng);
        let t1 = SystemTime::now();

        println!(
            "collapsed flat wave in {:?}",
            t1.duration_since(t0).unwrap()
        );

        println!("scaling image");
        let image = scale_image(image, 10); // resize the image
        image.save("flat.png").expect("failed to save image");
    }

    {
        let mut wave = build_wave::<Torus>(images.clone(), shape);

        let t0 = SystemTime::now();
        let image = wave.collapse(&mut rng);
        let t1 = SystemTime::now();

        println!(
            "collapsed torus wave in {:?}",
            t1.duration_since(t0).unwrap()
        );

        println!("scaling image");
        let image = scale_image(image, 10); // resize the image
        image.save("torus.png").expect("failed to save image");
    }

    {
        let mut wave = build_wave::<ProjectivePlane>(images.clone(), shape);

        let t0 = SystemTime::now();
        let image = wave.collapse(&mut rng);
        let t1 = SystemTime::now();

        println!(
            "collapsed projective plane wave in {:?}",
            t1.duration_since(t0).unwrap()
        );

        println!("scaling image");
        let image = scale_image(image, 10); // resize the image
        image
            .save("projective-plane.png")
            .expect("failed to save image");
    }

    {
        let mut wave = build_wave::<KleinBottle>(images.clone(), shape);

        let t0 = SystemTime::now();
        let image = wave.collapse(&mut rng);
        let t1 = SystemTime::now();

        println!(
            "collapsed klein bottle wave in {:?}",
            t1.duration_since(t0).unwrap()
        );

        println!("scaling image");
        let image = scale_image(image, 10); // resize the image
        image
            .save("klein-bottle.png")
            .expect("failed to save image");
    }
}
