use std::array::from_fn;
use std::fmt::Debug;

use rand::Rng;
use rand::RngCore;

use crate::bitset;
use crate::bitset::BitSet;
use crate::bitset::BitSlice;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::traits::Recover;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::util::partition_in_place;

#[derive(Debug)]
pub enum WaveTileError {
    OutOfTiles,
    RollbackOOB,
}

pub type NeighborWaveTiles<T, const N: usize> = [[Option<*mut WaveTile<T, N>>; 2]; N];
pub type Iter = usize;
pub type Index = usize;

pub struct WaveTile<T, const N: usize>
where
    T: BoundaryHash<N>,
{
    pub parity: usize, // either 0 or 1
    pub hashes: BitSet,
    pub neighbors: NeighborWaveTiles<T, N>,
    pub neighbor_hashes: [[*const BitSlice; 2]; N],

    pub tiles: Vec<*const Tile<T, N>>,
    pub entropy: usize,
    pub index: WfcNdIndex<N>,

    // (iter, index)
    filtered_tile_indices: Vec<(Iter, Index)>,
    start_index: Index, // cached for speed

    alt_bitset: BitSet,      // to avoid allocs at runtime
    masks: [[BitSet; 2]; N], // NOTE: depends on parity
}

impl<T, const N: usize> WaveTile<T, N>
where
    T: BoundaryHash<N>,
{
    /// Create a new `WaveTile`
    pub fn new(
        tiles: Vec<*const Tile<T, N>>,
        index: WfcNdIndex<N>,
        num_hashes: usize,
        parity: usize,
        temp_ptr: *const BitSlice,
    ) -> Self {
        let entropy = tiles.len();

        let masks = bitset::gen_bitmasks(num_hashes, parity);

        let mut wavetile = WaveTile {
            parity,
            hashes: BitSet::zeros(2 * N * num_hashes),
            neighbor_hashes: [[temp_ptr; 2]; N],
            neighbors: [[None; 2]; N],

            tiles,
            entropy,
            index,

            // avoid reallocs during collapse
            filtered_tile_indices: Vec::with_capacity(entropy),
            start_index: 0,

            alt_bitset: BitSet::zeros(2 * N * num_hashes),
            masks,
        };

        wavetile.update_hashes();

        wavetile
    }

    /// Collapses a `WaveTile` to one of its possible tiles, at random.
    pub fn collapse<R>(&mut self, rng: &mut R, iter: usize) -> NeighborWaveTiles<T, N>
    where
        R: RngCore + ?Sized,
    {
        assert!(self.entropy > 1, "called collapse on a collapsed WaveTile!");

        let n = self.tiles.len();

        let idx = rng.gen_range(self.start_index..n);

        self.filtered_tile_indices.push((iter, n - 1));
        self.start_index = n - 1;

        self.tiles.swap(idx, n - 1);

        self.update_hashes();
        self.entropy = 1;

        self.neighbors
    }

    /// Update the `possible_tiles` of the current `WaveTile`
    pub fn update(&mut self, iter: usize) -> Result<NeighborWaveTiles<T, N>, WaveTileError> {
        if self.entropy == 1 {
            return Ok(from_fn(|_| from_fn(|_| None)));
        }

        // new hashes are computed
        self.gen_alt_wavetile_bitset();
        self.hashes.intersect(&self.alt_bitset);

        let i = partition_in_place(&mut self.tiles[self.start_index..], |tile| {
            // SAFETY: `Vec` size is fixed before collapse
            let tile = unsafe { &**tile };

            !tile.hashes.is_subset(&self.hashes)
        });

        self.start_index += i;

        self.filtered_tile_indices.push((iter, self.start_index));
        self.entropy = self.tiles.len() - self.start_index;

        self.update_hashes();

        if self.entropy == 0 {
            Err(WaveTileError::OutOfTiles)
        } else {
            Ok(self.neighbors)
        }
    }

    /// Rollback a `WaveTile` one step. This is called when some `WaveTile` runs out of options for
    /// possible `Tile`s
    pub fn rollback(&mut self, n: usize, current_iter: usize) {
        // TODO: there are invariants on this vector that we are not taking advantage of. It's
        // probably possible to write this slightly faster
        self.filtered_tile_indices
            .retain(|(iter, _)| current_iter - iter > n);

        // the number of invalidated tiles
        let n = self
            .filtered_tile_indices
            .last()
            .map(|&(_, i)| i)
            .unwrap_or(0);

        self.start_index = n;
        self.entropy = self.tiles.len() - n;
        self.update_hashes();
    }

    // Compute a `BitSet` composed of the neighbor's `BitSlice`s
    // NOTE: hot function
    fn gen_alt_wavetile_bitset(&mut self) {
        self.alt_bitset.zero();

        for ([hash_left, hash_right], [mask_left, mask_right]) in
            self.neighbor_hashes.iter().zip(self.masks.iter())
        {
            let right_hashes = {
                // SAFETY: `Vec`s are not resized during collapse and so don't move
                let hashes = unsafe { &**hash_right };

                let mut res = mask_right.clone(); // cheap
                res.intersect(hashes);

                res
            };

            let left_hashes = {
                // SAFETY: `Vec`s are not resized during collapse and so don't move
                let hashes = unsafe { &**hash_left };

                let mut res = mask_left.clone();
                res.intersect(hashes);

                res
            };

            self.alt_bitset.union(&right_hashes).union(&left_hashes);
        }
    }

    /// Given a list of `tile`s that this WaveTile can be, this precomputes the list of valid
    /// hashes for each of its borders. This is used to speed up the wave propagation algorithm.
    fn update_hashes(&mut self) {
        // reset own bitsets
        self.hashes.zero();

        for tile in &self.tiles[self.start_index..] {
            // SAFETY: `Vec` size is fixed
            let tile = unsafe { &**tile };

            self.hashes.union(&tile.hashes);
        }
    }
}

impl<T, const N: usize> Recover<T, T, N> for WaveTile<T, N>
where
    T: BoundaryHash<N> + Clone + Merge,
{
    /// Recovers the `T` for type `WaveTile<T, N>`. Note that `T` must be `Merge`.
    ///
    /// In the future, this `Merge` requirement may be relaxed to only non-collapsed `WaveTile`s.
    /// This is a temporary limitation of the API. TODO
    fn recover(&self) -> T {
        let ts: Vec<T> = self.tiles[self.start_index..]
            .iter()
            .map(|&tile| {
                // SAFETY: `Vec` size is fixed
                let tile = unsafe { &*tile };

                tile.recover()
            })
            .collect();

        T::merge(&ts)
    }
}
