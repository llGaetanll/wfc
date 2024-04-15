use ndarray::Dimension;
use ndarray::NdIndex;
use rand::Rng;

use crate::bitset::BitSet;
use crate::bitset::BitSlice;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::types::DimN;

#[derive(Debug)]
pub enum WaveTileError {
    OutOfTiles,
    RollbackOOB,
}

pub struct WaveTile<T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub hashes: BitSet,
    pub neighbor_hashes: [[Option<*const BitSlice>; 2]; N],
    pub entropy: usize,

    pub possible_tiles: Vec<*const Tile<T, N>>,
    filtered_tiles: Vec<*const Tile<T, N>>,

    // (iter, index)
    filtered_tile_indices: Vec<(usize, usize)>,

    num_hashes: usize,
    pub parity: usize, // either 0 or 1
}

impl<T, const N: usize> WaveTile<T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    /// Create a new `WaveTile`
    pub fn new(tiles: Vec<*const Tile<T, N>>, num_hashes: usize, parity: usize) -> Self {
        let entropy = tiles.len();

        let mut wavetile = WaveTile {
            hashes: BitSet::zeros(2 * N * num_hashes),
            neighbor_hashes: [[None; 2]; N],
            entropy,

            possible_tiles: tiles,
            // avoid reallocs during collapse
            filtered_tiles: Vec::with_capacity(entropy),
            filtered_tile_indices: Vec::with_capacity(entropy),

            num_hashes,
            parity,
        };

        wavetile.update_hashes();

        wavetile
    }

    /// Collapses a `WaveTile` to one of its possible tiles, at random.
    pub fn collapse(&mut self, iter: usize) {
        assert!(self.entropy > 1, "called collapse on a collapsed WaveTile!");

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.entropy);

        self.filtered_tile_indices.push((iter, self.filtered_tiles.len()));

        let tile = self.possible_tiles.swap_remove(idx);

        // move al possible tiles to filtered tiles
        self.filtered_tiles.append(&mut self.possible_tiles);
        self.possible_tiles.push(tile);

        self.update_hashes();

        self.entropy = 1;
    }

    /// Update the `possible_tiles` of the current `WaveTile`
    pub fn update(&mut self, iter: usize) -> Result<(), WaveTileError> {
        if self.entropy == 1 {
            return Ok(());
        }

        // new hashes are computed
        let alt_wavetile_bitset = self.get_alt_wavetile_bitset();
        self.hashes.intersect(&alt_wavetile_bitset);

        let filtered_tiles = self.possible_tiles.iter().filter(|tile| {
            // SAFETY: `Vec` size is fixed before collapse
            let tile = unsafe { &***tile };

            !tile.hashes.is_subset(&self.hashes)
        });

        self.filtered_tile_indices
            .push((iter, self.filtered_tiles.len()));
        self.filtered_tiles.extend(filtered_tiles);
        self.possible_tiles.retain(|tile| {
            // SAFETY: `Vec` size is fixed before collapse
            let tile = unsafe { &**tile };

            tile.hashes.is_subset(&self.hashes)
        });

        self.entropy = self.possible_tiles.len();

        // NOTE: mutates self's hashes
        self.update_hashes();

        if self.entropy == 0 {
            Err(WaveTileError::OutOfTiles)
        } else {
            Ok(())
        }
    }

    /// Rollback a `WaveTile` one step. This is called when some `WaveTile` runs out of options for
    /// possible `Tile`s
    pub fn rollback(&mut self, n: usize, current_iter: usize) {
        let l = self.filtered_tile_indices.len();
        let num_invalids = self
            .filtered_tile_indices
            .iter()
            .rev()
            .take_while(|(iter, _)| current_iter - iter <= n)
            .count();

        if num_invalids == 0 {
            return
        }

        let (_, i) = self.filtered_tile_indices[l - num_invalids];
        self.filtered_tile_indices.truncate(l - num_invalids);

        let rollback_tiles = self.filtered_tiles.drain(i..);
        self.possible_tiles.extend(rollback_tiles);

        self.entropy = self.possible_tiles.len();
        self.update_hashes();
    }

    // Compute a `BitSet` composed of the neighbor's `BitSlice`s
    fn get_alt_wavetile_bitset(&self) -> BitSet {
        let mut bitset = BitSet::zeros(2 * N * self.num_hashes);

        let odd = self.parity;
        let even = 1 - self.parity;

        for (axis, [left, right]) in self.neighbor_hashes.iter().enumerate() {
            let right_hashes = match right {
                Some(hashes) => {
                    // SAFETY: `Vec`s are not resized during collapse and so don't move
                    let hashes = unsafe { &**hashes };
                    hashes.mask((2 * axis + even) * self.num_hashes, self.num_hashes)
                }
                None => {
                    let ones = BitSet::ones(2 * N * self.num_hashes);
                    ones.mask((2 * axis + even) * self.num_hashes, self.num_hashes)
                }
            };

            let left_hashes = match left {
                Some(hashes) => {
                    // SAFETY: `Vec`s are not resized during collapse and so don't move
                    let hashes = unsafe { &**hashes };
                    hashes.mask((2 * axis + odd) * self.num_hashes, self.num_hashes)
                }
                None => {
                    let ones = BitSet::ones(2 * N * self.num_hashes);
                    ones.mask((2 * axis + odd) * self.num_hashes, self.num_hashes)
                }
            };

            bitset.union(&right_hashes).union(&left_hashes);
        }

        bitset
    }

    /// Given a list of `tile`s that this WaveTile can be, this precomputes the list of valid
    /// hashes for each of its borders. This is used to speed up the wave propagation algorithm.
    fn update_hashes(&mut self) {
        // reset own bitsets
        self.hashes.zero();

        // new bitsets is union of possible tiles
        for tile in &self.possible_tiles {
            // SAFETY: `Vec` size is fixed
            let tile = unsafe { &**tile };

            self.hashes.union(&tile.hashes);
        }
    }
}

impl<T, const N: usize> WaveTile<T, N>
where
    T: BoundaryHash<N> + Clone + Merge,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Recovers the `T` for type `WaveTile<T, N>`. Note that `T` must be `Merge`.
    ///
    /// In the future, this `Merge` requirement may be relaxed to only non-collapsed `WaveTile`s.
    /// This is a temporary limitation of the API. TODO
    pub fn recover(&self) -> T {
        let ts: Vec<T> = self
            .possible_tiles
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
