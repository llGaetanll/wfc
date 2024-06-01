use std::error::Error;
use std::path::PathBuf;

use ndarray::Ix2;

use sdl2::pixels::PixelFormatEnum;
use sdl2::surface::Surface;
use wfc::prelude::*;

const WIDTH: usize = 200;
const HEIGHT: usize = 200;
const SCALE: usize = 5;

fn main() -> Result<(), Box<dyn Error>> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window(
            "Wave Function Collapse",
            (SCALE * WIDTH) as u32,
            (SCALE * HEIGHT) as u32,
        )
        .position_centered()
        .build()?;

    let mut canvas = window
        .into_canvas()
        .target_texture()
        .present_vsync()
        .build()?;

    let texture_creator = canvas.texture_creator();

    let img_path = PathBuf::from("examples/live/sample.png");
    let win_size = 3;

    let image = image::open(img_path).expect("image not found");

    let params = ImageParams::<_, Flat>::new_flat(image.to_rgb8(), win_size);
    let mut tileset = params.tileset();
    tileset.with_rots().with_flips();

    let mut wave = ImageWave::init(&mut tileset, Ix2(WIDTH, HEIGHT));

    let mut rng = rand::thread_rng();

    let image = wave
        .attach(Box::new(move |res| {
            canvas.clear();

            let mut flat_pixels = res.into_raw();
            let surface = Surface::from_data(
                &mut flat_pixels,
                (WIDTH * win_size) as u32,     // width of the texture
                (HEIGHT * win_size) as u32,    // height of the texture
                (WIDTH * win_size) as u32 * 3, // this is the number of channels for each pixel
                PixelFormatEnum::RGB24,
            ).unwrap();

            // create a texture from the surface
            let texture = texture_creator
                .create_texture_from_surface(surface).unwrap();

            canvas.copy(&texture, None, None).expect("failed to draw");

            canvas.present();
        }))
        .collapse_parallel(&mut rng);

    image.save("wave.png").expect("failed to save image");

    Ok(())
}
