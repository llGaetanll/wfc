use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::pin::Pin;

use ndarray::Dimension;
use ndarray::NdIndex;

use crate::bitset::BitSet;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::types::DimN;
use crate::wave::Wave;
use crate::wavetile::WaveTile;

pub struct TileSet<'a, T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub data: Vec<T>,

    pub tiles_lr: Vec<Tile<'a, T, N>>,
    pub tiles_rl: Vec<Tile<'a, T, N>>,

    // sidelength of the tile
    tile_size: usize,
}

impl<'a, T, const N: usize> TileSet<'a, T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub fn new(data: Vec<T>, tile_size: usize) -> Self {
        let mut tileset = TileSet {
            data,
            tiles_lr: vec![],
            tiles_rl: vec![],
            tile_size,
        };

        tileset.compute_hashes();

        tileset
    }

    // compute the list of unique hashes of the `TileSet`
    pub fn compute_hashes(&mut self) {}
}

pub trait Rotations<'a, T, const N: usize>
where
    T: Hash + Clone,
    DimN<N>: Dimension,
{
    fn with_rots(&'a mut self) -> &'a mut Self;
}

pub trait Flips<'a, T, const N: usize>
where
    T: Hash + Clone,
    DimN<N>: Dimension,
{
    fn with_flips(&'a mut self) -> &'a mut Self;
}

impl<'a, T, const N: usize> TileSet<'a, T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    // FIXME: exposing DimN<N> leaks ndarray API
    pub fn wave(&'a mut self, shape: DimN<N>) -> Pin<Box<Wave<'_, T, N>>> {
        let mut hash_index: usize = 0;
        let mut unique_hashes: HashMap<u64, usize> = HashMap::new();
        let mut tile_hashes = Vec::with_capacity(self.data.len());

        // iterate through all the tiles in the bitmap
        for tile in &self.data {
            let hashes = tile.boundary_hashes();
            tile_hashes.push(hashes);

            // add all unique hashes to an index map
            for &hash in hashes.iter().flat_map(|h| h.iter()) {
                if let Entry::Vacant(v) = unique_hashes.entry(hash) {
                    v.insert(hash_index);
                    hash_index += 1;
                }
            }
        }

        let num_tiles = self.data.len();
        let num_hashes = unique_hashes.len();

        for (i, hashes) in tile_hashes.iter().enumerate() {
            let mut hashes_lr = BitSet::zeros(2 * N * num_hashes);
            let mut hashes_rl = BitSet::zeros(2 * N * num_hashes);

            // TODO: update this comment to explain left right vs right left
            // This is where we define the mapping in our tile's bitset. Some explanation is in
            // order.
            //
            // Suppose we have a 2D Tile. Then, we have two axes and `N = 2`. For each axis, we
            // have a left and right side (and so we have a left and right hash) which
            // corresponds to a unique hash number in `unique_hashes`.
            //
            // This is the layout of the tile's bitset:
            //
            //   [Axis 1 left] | [Axis 1 right] | [Axis 2 left] | [Axis 2 right]
            //
            // where each block represents `num_hashes` bits, and only one bit in each block is
            // flipped on. Namely, this is `index_left` for the left blocks, and `index_right`
            // for the right blocks.
            for (i, [left, right]) in hashes.iter().enumerate() {
                let index_left = unique_hashes[left];
                let index_right = unique_hashes[right];

                hashes_lr.on((2 * i) * num_hashes + index_left);
                hashes_lr.on((2 * i + 1) * num_hashes + index_right);

                hashes_rl.on((2 * i + 1) * num_hashes + index_left);
                hashes_rl.on((2 * i) * num_hashes + index_right);
            }

            self.tiles_lr.push(Tile::new(i, hashes_lr, self.tile_size));
            self.tiles_rl.push(Tile::new(i, hashes_rl, self.tile_size));
        }

        Wave::new(shape, num_hashes, &self.tiles_lr, &self.tiles_rl, num_tiles)
    }
}

pub trait GoBack<'a, T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    fn go_back(&'a self, tileset: &'a TileSet<'a, T, N>) -> &'a T;
}

impl<'a, T, const N: usize> GoBack<'a, T, N> for Wave<'a, T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Convert back from a `Wave<T>` to `T`
    fn go_back(&'a self, tileset: &'a TileSet<'a, T, N>) -> &'a T {
        todo!()
    }
}

impl<'a, T, const N: usize> GoBack<'a, T, N> for WaveTile<'a, T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    fn go_back(&'a self, tileset: &'a TileSet<'a, T, N>) -> &'a T {
        todo!()
    }
}

impl<'a, T, const N: usize> GoBack<'a, T, N> for Tile<'a, T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    fn go_back(&'a self, tileset: &'a TileSet<'a, T, N>) -> &'a T {
        &tileset.data[self.id]
    }
}
