pub use crate::data::TileSet;

pub use crate::traits::Flips;
pub use crate::traits::Rotations;
pub use crate::traits::Merge;
pub use crate::traits::WaveTileable;

pub use crate::wave::traits::WaveBase;
pub use crate::wave::traits::Wave;

#[cfg(feature = "parallel")]
pub use crate::wave::traits::ParWave;

pub use crate::impls::image::ImageWave;
pub use crate::impls::image::ImageParams;
pub use crate::impls::image::ImageTileSet;

pub use crate::surface::Flat;
pub use crate::surface::FlatWave;

#[cfg(feature = "wrapping")]
mod wrapping {
    pub use crate::surface::wrapping::Torus;
    pub use crate::surface::wrapping::TorusWave;
    pub use crate::surface::wrapping::ProjectivePlane;
    pub use crate::surface::wrapping::ProjectiveWave;
    pub use crate::surface::wrapping::KleinBottle;
    pub use crate::surface::wrapping::KleinWave;
}

#[cfg(feature = "wrapping")]
pub use wrapping::*;

pub use rand;
pub use ndarray;
