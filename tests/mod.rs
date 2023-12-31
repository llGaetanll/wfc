use std::path::PathBuf;

use wfc::traits::SdlTexture;

use ndarray::Ix2;

const TESTS_PATH: [&str; 2] = [".", "tests"];
const ASSETS_DIR: &str = "assets";

fn img_path(img: &str) -> PathBuf {
    let tests_path: PathBuf = TESTS_PATH.iter().collect();
    let assets_path = tests_path.join(ASSETS_DIR);

    assets_path.join(img)
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

    let sdl_context = sdl2::init().expect("failed to init sdl2 context");

    let bitmap = wfc::from_image(&img_path("sample.png"), 3, true, true)
        .expect("failed to create bitmap");
    let tileset = bitmap.tile_set();

    let mut wave = tileset.wave(Ix2(10, 10));

    wave.collapse(None);
    wave.show(&sdl_context, "Wave", 20)
        .expect("failed to display wave");

    assert_eq!(false, true)
}
