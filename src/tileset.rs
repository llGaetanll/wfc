use std::fmt::Debug;
use std::hash::Hash;
use std::pin::Pin;

use ndarray::Dimension;
use ndarray::NdIndex;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use crate::tile::Tile;
use crate::types::BoundaryHash;
use crate::types::DimN;
use crate::wave::Wave;

pub struct TileSet<'a, T, const N: usize>
where
    T: Hash,
    DimN<N>: Dimension,
{
    // all unique hashes. Order matters
    pub hashes: Vec<BoundaryHash>,
    pub num_hashes: usize,

    // tiles are views into the bitmap
    pub tiles_lr: Vec<Tile<'a, T, N>>,
    pub tiles_rl: Vec<Tile<'a, T, N>>,

    pub num_tiles: usize,
    pub tile_size: usize,
}

impl<'a, T, const N: usize> TileSet<'a, T, N>
where
    T: Hash + Send + Sync + Clone + Debug,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    pub fn wave(&'a self, shape: DimN<N>) -> Pin<Box<Wave<'a, T, N>>> {
        Wave::new(shape, self)
    }
}
