use ndarray::Dimension;
use ndarray::NdIndex;

use crate::bitset::BitSet;
use crate::traits::BoundaryHash;
use crate::types::DimN;

pub struct Tile<T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    data: T,

    pub hashes: BitSet,
    pub shape: usize,
}

impl<T, const N: usize> Tile<T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub fn new(data: T, hashes: BitSet, shape: usize) -> Self {
        Tile {
            data,
            hashes,
            shape,
        }
    }
}

impl<T, const N: usize> Tile<T, N>
where
    T: BoundaryHash<N> + Clone,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Recover the `T` for type `Tile<T, N>`.
    pub fn recover(&self) -> T {
        self.data.clone()
    }
}
