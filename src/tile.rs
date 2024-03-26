use ndarray::Dimension;
use ndarray::NdIndex;

use std::marker::PhantomData;

use crate::bitset::BitSet;
use crate::data::TileSet;
use crate::traits::BoundaryHash;
use crate::traits::Recover;
use crate::types::DimN;
use crate::types::TileID;

pub struct Tile<'a, T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    /// The `Tile` does not actually own any data. Instead, the data of the `Wave` is entirely
    /// owned by `TileSet`.
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

impl<'a, T, const N: usize> Recover<'a, T, N> for Tile<'a, T, N>
where
    T: BoundaryHash<N> + Clone,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Recover the `T` for type `Tile<'a, T, N>`.
    fn recover(&'a self, tileset: &'a TileSet<'a, T, N>) -> T {
        tileset.data[self.id].clone()
    }
}
