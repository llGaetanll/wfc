// use ndarray::{s, Array3, Ix2, SliceInfo, SliceInfoElem};
use ndarray::Ix2;
use regex::Regex;
// use std::borrow::Borrow;
use std::env;
use std::path::Path;
use std::rc::Rc;
// use wfc::traits::Pixelize;

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

    let sample_image = wfc::sample::from_image(&img_path)?;

    // perform sliding windows on our sample
    let tiles = sample_image.window(window_size);

    // tiles are reference counted. this isn't exactly what we want but it works for now
    let tiles: Vec<Rc<_>> = tiles.into_iter().map(|tile| Rc::new(tile)).collect();

    // create a WxH wave of tiles from our sample image

    let wave = wfc::wave::Wave::new(tiles, Ix2(width, height))?;

    // wave does not live long enough
    // wave.test();

    // wave.collapse(Ix2(3, 4));

    // let wavetile = wave.wave[[0, 0]].borrow();
    // wavetile.pixels();

    /* NEEDED FUNCTIONS
    also a callback might be good for any change to the wave function so that we can draw its updates
    but maybe its possible to listen for changes? idrk yet

    // test collapse a tile

    // propagate the wave
    wave.propagate();
    */
    Ok(())
}

/*
let arr =
    Array::from_shape_vec((3, 3, 3), (0..27).map(|x| x as u8).collect::<Vec<_>>()).unwrap();
// let windows = arr.windows((2, 2, 2));

let dim: Vec<Ix> = vec![2, 2, 2]; // doesn't work with this because ndarray knows the dims of `arr` (and if the vector did not have length 3, it would fail)
let dim: [usize; 3] = [2; 3]; // because this always has length 3, it cannot fail and so it works
let windows = arr.windows(dim);
*/

/*
let sdl_context = sdl2::init()?;

let path_str = img_path.to_string_lossy();
println!("img path: {path_str}, window size: {window_size}, output width: {width}, output height: {height}");

// create a sample image struct
let sample_image = wfc::SampleImage::new(&img_path)?;

// create a list of tiles from the sample image
let tiles = wfc::window(sample_image, window_size);

let n = tiles.len();
println!("Made {n} different tiles");

let mut wave = wfc::Wave::new(&tiles, width, height, window_size);

wave.collapse(1, 1);

wave.show(&sdl_context)?;
 */
