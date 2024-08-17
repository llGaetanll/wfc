pub use crate::data::TileSet;
pub use crate::impls::image::ImageParams;
pub use crate::impls::image::ImageTileSet;
pub use crate::impls::image::ImageWave;
pub use crate::surface::Flat;
pub use crate::surface::FlatWave;
pub use crate::traits::Flips;
pub use crate::traits::Merge;
pub use crate::traits::Rotations;
pub use crate::traits::WaveTileable;
#[cfg(feature = "parallel")]
pub use crate::wave::traits::ParWave;
pub use crate::wave::traits::Wave as _;
pub use crate::wave::traits::WaveBase;

#[cfg(feature = "wrapping")]
mod wrapping {
    pub use crate::surface::wrapping::KleinBottle;
    pub use crate::surface::wrapping::KleinWave;
    pub use crate::surface::wrapping::ProjectivePlane;
    pub use crate::surface::wrapping::ProjectiveWave;
    pub use crate::surface::wrapping::Torus;
    pub use crate::surface::wrapping::TorusWave;
}

#[cfg(feature = "image")]
pub use image;
pub use rand;
#[cfg(feature = "sdl")]
pub use sdl2;
#[cfg(feature = "wrapping")]
pub use wrapping::*;
