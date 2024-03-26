use std::marker::PhantomPinned;

use ndarray::Dimension;
use ndarray::NdIndex;

use rand::Rng;

use crate::bitset::BitSet;
use crate::bitset::BitSlice;
use crate::data::TileSet;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::traits::Recover;
use crate::types::DimN;

pub struct WaveTile<'a, T, const N: usize>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    pub hashes: BitSet,
    pub neighbor_hashes: [[Option<*const BitSlice>; 2]; N],
    pub entropy: usize,

    possible_tiles: Vec<&'a Tile<'a, T, N>>,
    filtered_tiles: Vec<&'a Tile<'a, T, N>>,
    filtered_tile_indices: Vec<usize>,

    num_hashes: usize,
    parity: usize, // either 0 or 1

    _pin: PhantomPinned,
}

impl<'a, T, const N: usize> WaveTile<'a, T, N>
where
    T: BoundaryHash<N>,
    DimN<N>: Dimension,
{
    /// Create a new `WaveTile`
    pub fn new(
        tiles: Vec<&'a Tile<'a, T, N>>,
        hashes: BitSet,
        num_hashes: usize,
        parity: usize,
    ) -> Self {
        let entropy = tiles.len();

        WaveTile {
            possible_tiles: tiles,
            // avoid reallocs at runtime
            filtered_tiles: Vec::with_capacity(entropy),
            filtered_tile_indices: Vec::with_capacity(entropy),
            num_hashes,
            hashes,
            neighbor_hashes: [[None; 2]; N],
            entropy,
            parity,
            _pin: PhantomPinned,
        }
    }

    /// Collapses a `WaveTile` to one of its possible tiles, at random.
    pub fn collapse(&mut self) -> Option<()> {
        if self.entropy < 1 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.entropy);

        let filtered_tiles = self
            .possible_tiles
            .iter()
            .enumerate()
            .filter_map(|(i, tile)| if i != idx { Some(tile) } else { None });

        self.filtered_tile_indices.push(self.filtered_tiles.len());
        self.filtered_tiles.extend(filtered_tiles);

        let tile: &Tile<'a, T, N> = self.possible_tiles[idx];
        self.possible_tiles.clear();
        self.possible_tiles.push(tile);

        self.update_hashes();

        self.entropy = 1;

        Some(())
    }

    /// Update the `possible_tiles` of the current `WaveTile`
    pub fn update(&mut self) {
        if self.entropy < 2 {
            return;
        }

        let hashes: BitSet = BitSet::new();

        // owned copy of the self.hashes
        let mut hashes = std::mem::replace(&mut self.hashes, hashes);

        // prepare the correct wavetile hash
        let mut alt_wavetile_bitset = BitSet::zeros(2 * N * self.num_hashes);

        let odd = self.parity;
        let even = 1 - self.parity;

        for (axis, [left, right]) in self.neighbor_hashes.iter().enumerate() {
            let right_hashes = match right {
                Some(hashes) => {
                    // SAFETY: Wave is pinned
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
                    // SAFETY: Wave is pinned
                    let hashes = unsafe { &**hashes };
                    hashes.mask((2 * axis + odd) * self.num_hashes, self.num_hashes)
                }
                None => {
                    let ones = BitSet::ones(2 * N * self.num_hashes);
                    ones.mask((2 * axis + odd) * self.num_hashes, self.num_hashes)
                }
            };

            alt_wavetile_bitset.union(&right_hashes).union(&left_hashes);
        }

        hashes.intersect(&alt_wavetile_bitset);

        // new hashes are computed
        self.hashes = hashes;

        let filtered_tiles = self
            .possible_tiles
            .iter()
            .filter(|tile| !tile.hashes.is_subset(&self.hashes));

        self.filtered_tile_indices.push(self.filtered_tiles.len());
        self.filtered_tiles.extend(filtered_tiles);
        self.possible_tiles
            .retain(|tile| tile.hashes.is_subset(&self.hashes));

        self.entropy = self.possible_tiles.len();

        // NOTE: mutates self's hashes
        self.update_hashes();
    }

    /// Given a list of `tile`s that this WaveTile can be, this precomputes the list of valid
    /// hashes for each of its borders. This is used to speed up the wave propagation algorithm.
    fn update_hashes(&mut self) {
        // reset own bitsets
        self.hashes.zero();

        // new bitsets is union of possible tiles
        for tile in &self.possible_tiles {
            self.hashes.union(&tile.hashes);
        }
    }
}

impl<'a, T, const N: usize> Recover<'a, T, N> for WaveTile<'a, T, N>
where
    T: BoundaryHash<N> + Clone + Merge,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Recovers the `T` for type `WaveTile<'a, T, N>`. Note that `T` must be `Merge`.
    ///
    /// In the future, this `Merge` requirement may be relaxed to only non-collapsed `WaveTile`s.
    /// This is a temporary limitation of the API. TODO
    fn recover(&'a self, tileset: &'a TileSet<'a, T, N>) -> T {
        let ts: Vec<T> = self
            .possible_tiles
            .iter()
            .map(|&tile| tile.recover(tileset))
            .collect();

        T::merge(&ts)
    }
}
