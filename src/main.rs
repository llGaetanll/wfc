use ndarray::{Array2, Array5, ArrayBase, ArrayD, Ix2, Ix5, IxDyn, NdIndex, OwnedRepr};
use regex::Regex;
use std::env;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread;
use wfc::traits::SdlView;

use rayon::prelude::*;

use crate::wfc::tile::Tile;

mod wfc;

fn main() -> Result<(), String> {
    /*
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

    // sample_image.show(&sdl_context)?;
    wave.show(&sdl_context)?;

    // wave.wave[[0, 0]].show(&sdl_context)?;
    */

    let px: [u8; 3] = [0, 0, 0];
    let arr: ArrayD<[u8; 3]> = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![px; 6]).unwrap();

    let thing = Tile::new(arr.view());

    let lst: Array5<i32> = Array5::zeros((2, 3, 3, 4, 6));

    let shape = lst.shape();
    let ndim = lst.ndim();
    let strides = lst.strides();

    println!(
        "shape: {:?}, ndim: {:?}, strides {:?}",
        shape, ndim, strides
    );

    Ok(())
}
