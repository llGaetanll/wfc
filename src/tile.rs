use crate::bitset::BitSet;
use crate::traits::BoundaryHash;
use crate::traits::Recover;

pub struct Tile<T, const N: usize> {
    data: T,

    pub hashes: BitSet,
    pub shape: usize,
}

impl<T, const N: usize> Tile<T, N>
where
    T: BoundaryHash<N>,
{
    pub fn new(data: T, hashes: BitSet, shape: usize) -> Self {
        Tile {
            data,
            hashes,
            shape,
        }
    }
}

impl<T, const N: usize> Recover<T> for Tile<T, N>
where
    T: Clone,
{
    type Inner = T;

    /// Recover the `T` for type `Tile<T, N>`.
    fn recover(&self) -> T {
        self.data.to_owned()
    }
}
