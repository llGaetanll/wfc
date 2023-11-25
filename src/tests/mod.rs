use std::path::PathBuf;

use env_logger;
use ndarray::Ix2;

use crate::types::DimN;

const TESTS_PATH: [&str; 3] = [".", "src", "tests"];
const ASSETS_DIR: &str = "assets";

fn img_path(img: &str) -> PathBuf {
    let tests_path: PathBuf = TESTS_PATH.iter().collect();
    let assets_path = tests_path.join(ASSETS_DIR);

    assets_path.join(img)
}

struct Params<const N: usize> {
    img_path: PathBuf,
    window_size: usize,
    wave_dims: DimN<N>,
}

fn init_logger() {
    let _ = env_logger::builder()
        // Include all events in tests
        .filter_level(log::LevelFilter::max())
        // Ensure events are captured by `cargo test`
        .is_test(true)
        // Ignore errors initializing the logger if tests race to configure it
        .try_init();
}

#[test]
fn simple() {
    init_logger();

    let params = Params {
        img_path: img_path("sample.png"),
        window_size: 3,
        wave_dims: Ix2(5, 5),
    };

    let sdl_context = sdl2::init().expect("failed to init sdl2 context");

    let bitmap = crate::from_image(&params.img_path, params.window_size, true, true)
        .expect("failed to create bitmap");
    let tileset = bitmap.tile_set();

    let mut wave = tileset.wave(params.wave_dims);
    wave.collapse(None);

    wave.show(&sdl_context).expect("failed to display wave");

    assert_eq!(false, true)
}
