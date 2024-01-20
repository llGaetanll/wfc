use ndarray::Dim;
use std::fmt::Debug;

pub type TileHash = u64;
pub type DimN<const N: usize> = Dim<[usize; N]>;

pub type BoundaryHash = u64;

// this is effectively represented as [u8; 3]  in memory, but lets us implement a foreign trait on
// it
#[repr(transparent)]
#[derive(Clone, Copy, Hash, Debug)]
pub struct Pixel([u8; 3]);

impl From<[u8; 3]> for Pixel {
    fn from(value: [u8; 3]) -> Self {
        Pixel(value)
    }
}

impl From<Pixel> for [u8; 3] {
    fn from(value: Pixel) -> Self {
        value.0
    }
}

impl IntoIterator for Pixel {
    type Item = u8;
    type IntoIter = std::array::IntoIter<Self::Item, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
