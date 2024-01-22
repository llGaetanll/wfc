use ndarray::Dimension;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use std::hash::Hash;
use std::marker::PhantomData;

use crate::bitset::BitSet;
use crate::types::DimN;
use crate::types::TileHash;

/// A `Tile` is a view into our Sample
/// `N` is the dimension of each tile
/// `T is the type of each element
///
/// Note: It would be nice to restrict the type of `hashes` so that the size of
/// the vector is exactly inline with `N`, but I don't know how I could do that
///
/// Note: all axes of the dynamic array are the same size.
pub struct Tile<'a, T, const N: usize>
where
    T: Hash,
    DimN<N>: Dimension,
{
    data: PhantomData<&'a T>,

    /// The hash of the current tile. This is computed from the Type that the
    /// tile references. If two tiles have the same data, they will have
    /// the same hash, no matter the type.
    pub id: TileHash,

    pub hashes: BitSet,
    pub shape: usize,
}

impl<'a, T, const N: usize> Tile<'a, T, N>
where
    T: Hash,
    DimN<N>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    /// Create a new tile from an ndarray
    pub fn new(id: TileHash, hashes: BitSet, shape: usize) -> Self {
        Tile {
            data: PhantomData,
            id,
            hashes,
            shape,
        }
    }
}
