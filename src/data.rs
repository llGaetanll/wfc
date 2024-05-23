use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::marker::PhantomData;

use ndarray::Dimension;
use ndarray::NdIndex;

use crate::bitset::BitSet;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Stitch;
use crate::traits::Surface;
use crate::traits::WaveBase;
use crate::traits::WaveTile;
// use crate::traits::WaveTile;
use crate::types::DimN;
use crate::wave::Wave;
use crate::Recover;

pub struct TileSet<Inner, Outer, const N: usize>
where
    Inner: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    _outer: PhantomData<Outer>,
    pub data: Vec<Inner>,

    tiles: Vec<Tile<Inner, N>>,
    co_tiles: Vec<Tile<Inner, N>>,

    tile_size: usize,
    num_hashes: usize,
}

impl<Inner, Outer, const N: usize> TileSet<Inner, Outer, N>
where
    Inner: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub fn new(data: Vec<Inner>, tile_size: usize) -> Self {
        TileSet {
            _outer: PhantomData,
            data,
            tiles: vec![],
            co_tiles: vec![],
            tile_size,
            num_hashes: 0,
        }
    }
}

impl<Inner, Outer, const N: usize> TileSet<Inner, Outer, N>
where
    // Inner: Clone + BoundaryHash<N> + Stitch<N, T = Inner> + Recover<Outer, Inner = Inner>,
    Inner: WaveTile<Inner, Outer, N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    pub fn wave<S>(&mut self, shape: DimN<N>) -> Wave<Inner, Outer, S, N>
    where
        S: Surface<N>,
    {
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

        self.num_hashes = unique_hashes.len();

        for (hashes, data) in tile_hashes.iter().zip(&self.data) {
            let mut bitset = BitSet::zeros(2 * N * self.num_hashes);
            let mut co_bitset = BitSet::zeros(2 * N * self.num_hashes);

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

                bitset.on((2 * i) * self.num_hashes + index_left);
                bitset.on((2 * i + 1) * self.num_hashes + index_right);

                co_bitset.on((2 * i + 1) * self.num_hashes + index_left);
                co_bitset.on((2 * i) * self.num_hashes + index_right);
            }

            self.tiles
                .push(Tile::new(data.clone(), bitset, self.tile_size));
            self.co_tiles
                .push(Tile::new(data.clone(), co_bitset, self.tile_size));
        }

        Wave::init(self, shape)
    }
}
