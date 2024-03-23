use ndarray::Dimension;

use std::marker::PhantomData;

use crate::bitset::BitSet;
use crate::traits::BoundaryHash;
use crate::types::DimN;
use crate::types::TileID;

pub struct Tile<'a, T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    data: PhantomData<&'a T>,

    /// The index of the data of the `Tile` in the associated `TileSet`.
    /// This is how we reconstruct the data from the `Wave`
    pub id: TileID,

    pub hashes: BitSet,
    pub shape: usize,
}

impl<'a, T, const N: usize> Tile<'a, T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub fn new(id: TileID, hashes: BitSet, shape: usize) -> Self {
        Tile {
            data: PhantomData,
            id,
            hashes,
            shape,
        }
    }
}
