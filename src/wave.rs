use std::hash::Hash;
use std::pin::Pin;

use ndarray::Array;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::NdIndex;

use crate::bitset::BitSet;
use crate::tile::Tile;
use crate::tileset::TileSet;
use crate::types::DimN;
use crate::wavetile::WaveTile;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::ext::ndarray::WaveArrayExt;

/// A `Wave` is an `N` dimensional array of `WaveTile`s.
///
/// `N` is the dimension of the wave, as well as the dimension of each element of each `WaveTile`.
/// `T` is the type of the element of the wave. All `WaveTile`s for a `Wave` hold the same type.
pub struct Wave<'a, T, const N: usize>
where
    T: Hash,
    DimN<N>: Dimension, // ensures that [usize; N] is a Dimension implemented by ndarray
    [usize; N]: NdIndex<DimN<N>>, // ensures that any [usize; N] is a valid index into the nd array
{
    wave: Array<WaveTile<'a, T, N>, DimN<N>>,

    // cached to speed up propagate
    min_entropy: (usize, [usize; N]),
    max_entropy: (usize, [usize; N]),
}

impl<'a, T, const N: usize> Wave<'a, T, N>
where
    T: Hash,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    pub fn new(shape: DimN<N>, tile_set: &'a TileSet<'a, T, N>) -> Pin<Box<Self>> {
        let tiles_lr: Vec<&'a Tile<'a, T, N>> = tile_set.tiles_lr.iter().collect();
        let tiles_rl: Vec<&'a Tile<'a, T, N>> = tile_set.tiles_rl.iter().collect();

        let bitset_lr: Vec<&BitSet> = tiles_lr.iter().map(|tile| &tile.hashes).collect();
        let bitset_rl: Vec<&BitSet> = tiles_rl.iter().map(|tile| &tile.hashes).collect();

        let wavetile_bitset_lr: BitSet = Self::merge_tile_bitsets(bitset_lr, tile_set.num_hashes);
        let wavetile_bitset_rl: BitSet = Self::merge_tile_bitsets(bitset_rl, tile_set.num_hashes);

        let wave = Wave {
            min_entropy: (tile_set.num_tiles, [0; N]),
            max_entropy: (tile_set.num_tiles, [0; N]),

            wave: Array::from_shape_fn(shape, |i| {
                let mut parity: usize = i.into_dimension().as_array_view().sum();
                parity %= 2;

                // NOTE: this parity trick *does not* work for any wrapping wave. Maybe encode this
                // into the type of the wave?
                if parity % 2 == 0 {
                    WaveTile::new(
                        tiles_lr.clone(),
                        wavetile_bitset_lr.clone(),
                        tile_set.num_hashes,
                        parity,
                    )
                } else {
                    WaveTile::new(
                        tiles_rl.clone(),
                        wavetile_bitset_rl.clone(),
                        tile_set.num_hashes,
                        parity,
                    )
                }
            }),
        };

        // we must put our wave in a pinned box to ensure that its contents are not moved.
        let mut wave = Box::pin(wave);

        // for each WaveTile, we need to get a list of pointers to its neighbors. This is why we
        // pinned the box.

        let get_wavetile_neighbor_bitsets = |index: usize| -> [[Option<*const BitSet>; 2]; N] {
            let index = wave.wave.get_nd_index(index);
            let neighbor_indices = wave.wave.get_index_neighbors(index);

            let neighbor_bitsets: [[Option<*const BitSet>; 2]; N] = neighbor_indices
                .iter()
                .map(|[left, right]| {
                    [
                        left.map(|index| {
                            let wavetile_left = &wave.wave[index];
                            let bitset: *const BitSet = &wavetile_left.hashes;
                            bitset
                        }),
                        right.map(|index| {
                            let wavetile_right = &wave.wave[index];
                            let bitset: *const BitSet = &wavetile_right.hashes;
                            bitset
                        }),
                    ]
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            neighbor_bitsets
        };

        // the complete list of all neighbor bitset pointers for each wavetile of the wave
        let wavetile_bitsets_neighbors: Vec<[[Option<*const BitSet>; 2]; N]> = (0..wave.wave.len())
            .map(get_wavetile_neighbor_bitsets)
            .collect();

        // 2. Assign the pointers
        // NOTE: We HAD to do this in two steps.
        //
        //      The first was to get the pointers to the neighbors of each WaveTile. We needed an
        //      immutable reference to wave.wave to do this, since we needed to traverse it.
        //
        //      The second is to mutate the wave, which we are doing now. Remember that, without
        //      interior mutability, we can't mutate a WaveTile in the Wave without mutating the
        //      Wave itself. But this requires a mutable reference to wave.wave, hence it must be
        //      done independently from the pointer collection step.
        for (wavetile, neighbor_bitsets) in wave
            .wave
            .iter_mut()
            .zip(wavetile_bitsets_neighbors.into_iter())
        {
            wavetile.neighbor_hashes = neighbor_bitsets;
        }

        wave
    }

    /// Collapses the wave
    pub fn collapse(&mut self, starting_index: Option<WfcNdIndex<N>>) {
        // if the wave is fully collapsed
        if self.max_entropy.0 < 2 {
            return;
        };

        // get starting index
        let index = match starting_index {
            Some(index) => index,
            None => self.min_entropy.1,
        };

        // collapse the starting tile
        {
            let wavetile = &mut self.wave[index];
            wavetile.collapse();
        }

        self.propagate(index);

        // handle the rest of the wave
        loop {
            if self.max_entropy.0 < 2 {
                break;
            };

            {
                let index = self.min_entropy.1;
                let wavetile = &mut self.wave[index];
                wavetile.collapse();
            }

            self.propagate(index);
        }
    }

    // Computes the total union of the list of bit sets in `tile_hashes`.
    fn merge_tile_bitsets(tile_bitsets: Vec<&BitSet>, num_hashes: usize) -> BitSet {
        let mut bitset = BitSet::zeros(2 * N * num_hashes);

        for tile_bitset in tile_bitsets {
            bitset.union(tile_bitset);
        }

        bitset
    }

    /// Propagates the wave from an index `start`
    /// TODO:
    ///  - stop propagation if neighboring tiles haven't updated
    fn propagate(&mut self, start: WfcNdIndex<N>) {
        let (mut min_entropy, mut min_idx) = (usize::MAX, self.min_entropy.1);
        let (mut max_entropy, mut max_idx) = (0, self.max_entropy.1);

        let mut upd_entropy = |entropy, index| {
            if entropy > 1 && entropy < min_entropy {
                min_entropy = entropy;
                min_idx = index;
            }

            if entropy > max_entropy {
                max_entropy = entropy;
                max_idx = index;
            }
        };

        let index_groups = self.wave.get_index_groups(start);
        for index_group in index_groups.into_iter() {
            for index in index_group.into_iter() {
                // TODO: catch if a wavetile has no more options
                // at this stage instead of on collapse?
                self.wave[index].update();

                // update entropy bounds
                let entropy = self.wave[index].entropy;
                upd_entropy(entropy, index);
            }
        }

        // update entropy
        self.min_entropy = (min_entropy, min_idx);
        self.max_entropy = (max_entropy, max_idx);
    }
}
