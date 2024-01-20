use ndarray::Ix2;
use std::path::PathBuf;

fn main() {
    // let sdl_context = sdl2::init().expect("failed to init sdl2 context");

    let sample_path = PathBuf::from("/home/al/files/github/wfc/examples/simple/sample.png");

    let bitmap = wfc::from_image(&sample_path, 3, true, true).expect("failed to create bitmap");

    let tileset = bitmap.tile_set();

    // let t0 = SystemTime::now();

    let mut wave = tileset.wave(Ix2(10, 10));

    // let t1 = SystemTime::now();

    wave.collapse(None);

    // prevent rust from optimizing away the wave (so that we can benchmark it!)
    let _ = wave;

    // let t2 = SystemTime::now();

    // wave.show(&sdl_context, "Wave", 20)
    //     .expect("failed to display wave");

    // println!("sampled in {:?}", t1.duration_since(t0).unwrap());
    // println!("collapsed in {:?}", t2.duration_since(t1).unwrap());
}
