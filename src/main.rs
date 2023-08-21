use ndarray::Ix2;
use regex::Regex;
use std::env;
use std::path::Path;
use std::rc::Rc;
use wfc::traits::SdlView;

mod wfc;

fn main() -> Result<(), String> {
    let args: Vec<_> = env::args().collect();

    // first argument is the sample image path
    let img_path = Path::new(&args[1]);

    // second argument is the size of the sliding window
    let window_size = args[2].parse::<usize>().unwrap();

    // third argument is a string of the form "[width]x[height]" that specifies the output image size
    // with width and height in terms of window size
    let re = Regex::new(r"^(\d+)x(\d+)$").unwrap();
    let caps = re.captures(&args[3]).unwrap();

    // extract the width and height of the output image in terms of window size
    let width: usize = caps[1].parse().expect("Invalid width");
    let height: usize = caps[2].parse().expect("Invalid height");

    let sdl_context = sdl2::init()?;

    let sample_image = wfc::sample::from_image(&img_path)?;

    // perform sliding windows on our sample
    let tiles = sample_image.window(window_size);

    // tiles are reference counted. this isn't exactly what we want but it works for now
    let tiles: Vec<Rc<_>> = tiles.into_iter().map(|tile| Rc::new(tile)).collect();

    // create a WxH wave of tiles from our sample image
    let wave = wfc::wave::Wave::new(tiles, Ix2(width, height))?;

    sample_image.show(&sdl_context)?;
    // wave.show(&sdl_context)?;

    // wave.wave[[0, 0]].show(&sdl_context)?;

    Ok(())
}
