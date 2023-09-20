use ndarray::Ix2;
use regex::Regex;
use std::env;
use std::path::Path;
use std::time::SystemTime;
use wfc::sample;

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

    let sample = sample::from_image(&img_path, window_size, false, true)?;
    let wave = crate::wfc::wave::from_sample(&sample, Ix2(width, height))?;

    let t0 = SystemTime::now();
    wave.collapse(None);
    let t1 = SystemTime::now();

    println!("Collapse took {:?}", t1.duration_since(t0).unwrap());

    wave.show(&sdl_context)?;

    Ok(())
}
