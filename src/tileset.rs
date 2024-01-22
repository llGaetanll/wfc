use std::hash::Hash;

use ndarray::Dimension;

use crate::tile::Tile;
use crate::types::BoundaryHash;
use crate::types::DimN;

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
