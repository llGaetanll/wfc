use std::{path::PathBuf, time::SystemTime};
use wfc::traits::SdlTexture;
use ndarray::Ix2;

fn main() {
    let sdl_context = sdl2::init().expect("failed to init sdl2 context");

    let sample_path = PathBuf::from("examples/simple/sample.png");

    let bitmap = wfc::from_image(&sample_path, 3, true, true)
        .expect("failed to create bitmap");

    let tileset = bitmap.tile_set();

    let t0 = SystemTime::now();

    let mut wave = tileset.wave(Ix2(10, 10));

    wave.collapse(None);

    let t1 = SystemTime::now();

    wave.show(&sdl_context, "Wave", 20)
        .expect("failed to display wave");

    println!("wave collapsed in {:?}", t1.duration_since(t0).unwrap())
}
