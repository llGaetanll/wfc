use crate::bitset::BitSet;
use crate::traits::BoundaryHash;
use crate::traits::Recover;

pub struct Tile<T, const N: usize>
where
    T: BoundaryHash<N>,
{
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

impl<T, const N: usize> Recover<T, T, N> for Tile<T, N>
where
    T: BoundaryHash<N> + Clone,
{
    /// Recover the `T` for type `Tile<T, N>`.
    fn recover(&self) -> T {
        self.data.to_owned()
    }
}
