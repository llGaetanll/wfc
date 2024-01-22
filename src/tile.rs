use ndarray::Dimension;

use std::hash::Hash;
use std::marker::PhantomData;

use crate::bitset::BitSet;
use crate::types::DimN;
use crate::types::TileHash;

pub struct Tile<'a, T, const N: usize>
where
    T: Hash,
    DimN<N>: Dimension,
{
    data: PhantomData<&'a T>,

    pub id: TileHash,

    pub hashes: BitSet,
    pub shape: usize,
}

impl<'a, T, const N: usize> Tile<'a, T, N>
where
    T: Hash,
    DimN<N>: Dimension,
{
    pub fn new(id: TileHash, hashes: BitSet, shape: usize) -> Self {
        Tile {
            data: PhantomData,
            id,
            hashes,
            shape,
        }
    }
}
