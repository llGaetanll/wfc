use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use ndarray::Ix2;
use wfc::data::Flips;
use wfc::data::Rotations;
use wfc::data::TileSet;

fn main() {
    let tiles_path = PathBuf::from("examples/tiles/tileset/");
    let images: Vec<_> = fs::read_dir(tiles_path)
        .expect("tileset directory not found")
        .filter_map(|file| {
            let file = file.ok();
            file.and_then(|file| image::open(file.path()).ok())
        })
        .collect();

    let mut tile_set = TileSet::from_images(images);
    let mut wave = tile_set.with_rots().with_flips().wave(Ix2(10, 10));

    let t0 = SystemTime::now();
    wave.collapse(None);
    let t1 = SystemTime::now();

    println!("collapsed in {:?}", t1.duration_since(t0).unwrap());
}
