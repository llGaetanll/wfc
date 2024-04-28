use core::hash::Hash;

use std::array::from_fn;
use std::fmt::Debug;
use std::ops::Deref;
use std::ops::DerefMut;

use rand::Rng;

use crate::bitset::BitSet;
use crate::bitset::BitSlice;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::traits::Recover;

#[derive(Debug)]
pub enum WaveTileError {
    OutOfTiles,
    RollbackOOB,
}

pub type NeighborWaveTiles<T, const N: usize> = [[Option<WaveTilePtr<T, N>>; 2]; N];

pub struct WaveTile<T, const N: usize>
where
    T: BoundaryHash<N>,
{
    pub hashes: BitSet,
    pub neighbor_hashes: [[Option<*const BitSlice>; 2]; N],
    pub neighbors: NeighborWaveTiles<T, N>,

    pub tiles: Vec<*const Tile<T, N>>,
    pub entropy: usize,
    pub index: [usize; N],

    // (iter, index)
    filtered_tile_indices: Vec<(usize, usize)>,
    start_index: usize, // cached for speed

    num_hashes: usize,
    pub parity: usize, // either 0 or 1
}

impl<T, const N: usize> WaveTile<T, N>
where
    T: BoundaryHash<N>,
{
    /// Create a new `WaveTile`
    pub fn new(
        tiles: Vec<*const Tile<T, N>>,
        index: [usize; N],
        num_hashes: usize,
        parity: usize,
    ) -> Self {
        let entropy = tiles.len();

        let mut wavetile = WaveTile {
            hashes: BitSet::zeros(2 * N * num_hashes),
            neighbor_hashes: [[None; 2]; N],
            neighbors: [[None; 2]; N],

            tiles,
            entropy,

            index,

            // avoid reallocs during collapse
            filtered_tile_indices: Vec::with_capacity(entropy),
            start_index: 0,

            num_hashes,
            parity,
        };

        wavetile.update_hashes();

        wavetile
    }

    pub fn collapse2(&mut self, iter: usize) -> NeighborWaveTiles<T, N> {
        assert!(self.entropy > 1, "called collapse on a collapsed WaveTile!");

        let n = self.tiles.len();

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(self.start_index..n);

        self.filtered_tile_indices.push((iter, n - 1));
        self.start_index = n - 1;

        self.tiles.swap(idx, n - 1);

        self.update_hashes();
        self.entropy = 1;

        self.neighbors
    }

    pub fn update2(&mut self, iter: usize) -> Result<NeighborWaveTiles<T, N>, WaveTileError> {
        if self.entropy == 1 {
            return Ok(from_fn(|_| from_fn(|_| None)));
        }

        // new hashes are computed
        let alt_wavetile_bitset = self.get_alt_wavetile_bitset();
        self.hashes.intersect(&alt_wavetile_bitset);

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

    /// Collapses a `WaveTile` to one of its possible tiles, at random.
    pub fn collapse(&mut self, iter: usize) {
        assert!(self.entropy > 1, "called collapse on a collapsed WaveTile!");

        let n = self.tiles.len();

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(self.start_index..n);

        self.filtered_tile_indices.push((iter, n - 1));
        self.start_index = n - 1;

        self.tiles.swap(idx, n - 1);

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
            Ok(())
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
    // TODO: hot function
    fn get_alt_wavetile_bitset(&self) -> BitSet {
        let mut bitset = BitSet::zeros(2 * N * self.num_hashes);

        let odd = self.parity;
        let even = 1 - self.parity;

        for (axis, [left, right]) in self.neighbor_hashes.iter().enumerate() {
            let right_hashes = match right {
                Some(hashes) => {
                    // SAFETY: `Vec`s are not resized during collapse and so don't move
                    let hashes = unsafe { &**hashes };

                    // TODO: cache these masks?
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

unsafe impl<T, const N: usize> Send for WaveTile<T, N> where T: BoundaryHash<N> {}
unsafe impl<T, const N: usize> Sync for WaveTile<T, N> where T: BoundaryHash<N> {}



#[repr(transparent)]
pub struct WaveTilePtr<T: BoundaryHash<N>, const N: usize>(*mut WaveTile<T, N>);

impl<T, const N: usize> WaveTilePtr<T, N>
where T: BoundaryHash<N>
{
    pub fn from(ptr: *mut WaveTile<T, N>) -> Self {
        Self(ptr)
    }
}

impl<T, const N: usize> Deref for WaveTilePtr<T, N> 
where T: BoundaryHash<N>
{
    type Target = WaveTile<T, N>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<T, const N: usize> DerefMut for WaveTilePtr<T, N> 
where T: BoundaryHash<N>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

impl<T, const N: usize> Debug for WaveTilePtr<T, N>
where T: BoundaryHash<N>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

unsafe impl<T, const N: usize> Send for WaveTilePtr<T, N>
where T: BoundaryHash<N> {}

unsafe impl<T, const N: usize> Sync for WaveTilePtr<T, N>
where T: BoundaryHash<N> {}

impl<T, const N: usize> Clone for WaveTilePtr<T, N>
where T: BoundaryHash<N> {
    fn clone(&self) -> Self {
        WaveTilePtr::from(self.0)
    }
}

impl<T, const N: usize> Copy for WaveTilePtr<T, N>
where T: BoundaryHash<N> {}

impl<T, const N: usize> Hash for WaveTilePtr<T, N>
where T: BoundaryHash<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl<T, const N: usize> PartialEq for WaveTilePtr<T, N>
where T: BoundaryHash<N> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, const N: usize> Eq for WaveTilePtr<T, N>
where T: BoundaryHash<N> {}

/// Takes `&mut [T]` and a predicate `P: FnMut(&T) -> bool` and partitions the list according to
/// the predicate. Swaps elements of the list such that all elements satisfying `P` appear before
/// any element not satisfying `P`.
///
/// The index `i` returned by the function always points at the first element for which `P` is false.
/// Note that if `P` is trivial, then `i = |list|` points outside the list.
pub fn partition_in_place<T, P>(list: &mut [T], mut predicate: P) -> usize
where
    P: FnMut(&T) -> bool,
{
    if list.is_empty() {
        return 0;
    }

    let (mut lo, mut hi) = (0, list.len() - 1);

    while lo < hi {
        if predicate(&list[lo]) {
            lo += 1;
            continue;
        }

        if !predicate(&list[hi]) {
            hi -= 1;
            continue;
        }

        list.swap(lo, hi);
        lo += 1;
        hi -= 1;
    }

    if predicate(&list[lo]) {
        lo + 1
    } else {
        lo
    }
}
