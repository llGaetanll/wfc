mod ext;
mod types;
mod util;

pub mod data;
pub mod impls;
pub mod surface;
pub mod traits;

mod bitset;

mod tile;
mod wavetile;

pub mod prelude;
pub mod wave;

// re-export rand so downstream crates don't have to think as hard about matching `wfc`'s version
pub use data::TileSet;
#[cfg(feature = "image")]
pub use image;
pub use ndarray;
pub use rand;
#[cfg(feature = "sdl")]
pub use sdl2;
