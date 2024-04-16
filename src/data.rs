use std::collections::hash_map::Entry;
use std::collections::HashMap;

use ndarray::Dimension;
use ndarray::NdIndex;

use crate::bitset::BitSet;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::traits::Stitch;
use crate::types::DimN;
use crate::wave::Wave;

pub struct TileSet<T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub data: Vec<T>,

    pub tiles_lr: Vec<Tile<T, N>>,
    pub tiles_rl: Vec<Tile<T, N>>,

    pub tile_size: usize,
}

impl<T, const N: usize> TileSet<T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub fn new(data: Vec<T>, tile_size: usize) -> Self {
        TileSet {
            data,
            tiles_lr: vec![],
            tiles_rl: vec![],
            tile_size,
        }
    }
}

impl<T, const N: usize> TileSet<T, N>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    pub fn wave(&mut self, shape: DimN<N>) -> Wave<T, N> {
        let mut hash_index: usize = 0;

        // key: hash, value: index into `tile_hashes`
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

        let num_hashes = unique_hashes.len();

        for (hashes, data) in tile_hashes.iter().zip(&self.data) {
            let mut hashes_lr = BitSet::zeros(2 * N * num_hashes);
            let mut hashes_rl = BitSet::zeros(2 * N * num_hashes);

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
            //
            // There is still one thing unexplained, and that is the use of two bitsets for two
            // sets of tiles: `tiles_lr`, and `tiles_rl`. When we propagate the Wave, each WaveTile
            // looks at its neighbors' BitSets to update its own. More specifically: for any given
            // axis, it might look at its left wall and compare it with the right wall of its left
            // neighbor, on that axis. But this requires an akward shift in the BitSet of either
            // our current WaveTile, or its neighbor's. Better is to alternate the layouts of every
            // other WaveTile's bitset. Much like a chessboard, all the black squares might have
            // the alignment described above, while the white squares have their lefts and rights
            // swapped. This eliminates the need for any bitshifting whatsoever in the WaveTiles at
            // collapse time, and it is perhaps the single most impactful speed improvement to the
            // algorithm.
            for (i, [left, right]) in hashes.iter().enumerate() {
                let index_left = unique_hashes[left];
                let index_right = unique_hashes[right];

                hashes_lr.on((2 * i) * num_hashes + index_left);
                hashes_lr.on((2 * i + 1) * num_hashes + index_right);

                hashes_rl.on((2 * i + 1) * num_hashes + index_left);
                hashes_rl.on((2 * i) * num_hashes + index_right);
            }

            self.tiles_lr
                .push(Tile::new(data.clone(), hashes_lr, self.tile_size));
            self.tiles_rl
                .push(Tile::new(data.clone(), hashes_rl, self.tile_size));
        }

        Wave::new(&self.tiles_lr, &self.tiles_rl, shape, num_hashes)
    }
}
