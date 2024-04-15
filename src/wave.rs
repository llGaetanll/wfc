use ndarray::Array;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::NdIndex;

use crate::bitset::BitSlice;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::traits::Stitch;
use crate::types::DimN;
use crate::wavetile::WaveTile;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::ext::ndarray::WaveArrayExt;

pub struct Wave<T, const N: usize>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    pub wave: Array<WaveTile<T, N>, DimN<N>>,

    // cached to speed up propagate
    min_entropy: (usize, [usize; N]),
    max_entropy: (usize, [usize; N]),
}

// TODO: maybe encode whether the `Wave` has collapsed as part of the type?
impl<T, const N: usize> Wave<T, N>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    pub fn new(
        tiles_lr: &[Tile<T, N>],
        tiles_rl: &[Tile<T, N>],
        shape: DimN<N>,
        num_hashes: usize,
    ) -> Self {
        let num_tiles = tiles_lr.len();

        let tiles_lr: Vec<*const Tile<T, N>> = tiles_lr
            .iter()
            .map(|tile| tile as *const Tile<T, N>)
            .collect();
        let tiles_rl: Vec<*const Tile<T, N>> = tiles_rl
            .iter()
            .map(|tile| tile as *const Tile<T, N>)
            .collect();

        let mut wave = Wave {
            wave: Array::from_shape_fn(shape, |i| {
                let mut parity: usize = i.into_dimension().as_array_view().sum();
                parity %= 2;

                if parity == 0 {
                    WaveTile::new(tiles_lr.clone(), num_hashes, parity)
                } else {
                    WaveTile::new(tiles_rl.clone(), num_hashes, parity)
                }
            }),

            min_entropy: (num_tiles, [0; N]),
            max_entropy: (num_tiles, [0; N]),
        };

        // for each WaveTile, we need to get a list of pointers to its neighbors. This is why we
        // pinned the box.
        let get_wavetile_neighbor_bitsets = |index: usize| -> [[Option<*const BitSlice>; 2]; N] {
            let index = wave.wave.get_nd_index(index);
            let neighbor_indices = wave.wave.get_index_neighbors(index);

            let neighbor_bitsets: [[Option<*const BitSlice>; 2]; N] = neighbor_indices
                .iter()
                .map(|[left, right]| {
                    [
                        left.map(|index| {
                            let wavetile_left = &wave.wave[index];
                            let hashes: &BitSlice = &wavetile_left.hashes;
                            let hashes: *const BitSlice = hashes;
                            hashes
                        }),
                        right.map(|index| {
                            let wavetile_right = &wave.wave[index];
                            let hashes: &BitSlice = &wavetile_right.hashes;
                            let hashes: *const BitSlice = hashes;
                            hashes
                        }),
                    ]
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            neighbor_bitsets
        };

        // the complete list of all neighbor bitset pointers for each wavetile of the wave
        let wavetile_bitsets_neighbors: Vec<[[Option<*const BitSlice>; 2]; N]> =
            (0..wave.wave.len())
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
        {
            for (wavetile, neighbor_bitsets) in wave
                .wave
                .iter_mut()
                .zip(wavetile_bitsets_neighbors.into_iter())
            {
                wavetile.neighbor_hashes = neighbor_bitsets;
            }
        }

        wave
    }

    /// Collapse the `Wave`
    pub fn collapse(&mut self, starting_index: Option<WfcNdIndex<N>>) /* -> Option<T> */
    {
        // if the wave is fully collapsed
        if self.max_entropy.0 < 2 {
            return; // Some(self.recover(tileset));
        };

        // get starting index
        let mut index = match starting_index {
            Some(index) => index,
            None => self.min_entropy.1,
        };

        // handle the rest of the wave
        for i in 0.. {
            if self.max_entropy.0 < 2 {
                return;
            };

            let wavetile = &mut self.wave[index];

            // since the sets acted on by `collapse` and `propagate` are disjoint, it's ok to make
            // these the same iteration.
            wavetile.collapse(i);
            self.propagate(index, i);

            index = self.min_entropy.1;
        }
    }

    /// Propagates the wave from an index `start`
    /// TODO:
    ///  - stop propagation if neighboring tiles haven't updated
    fn propagate(&mut self, start: WfcNdIndex<N>, iter: usize) {
        let mut min_entropy = (usize::MAX, self.min_entropy.1);
        let mut max_entropy = (0, self.max_entropy.1);

        let mut upd_entropy = |entropy, index| {
            if entropy > 1 && entropy < min_entropy.0 {
                min_entropy.0 = entropy;
                min_entropy.1 = index;
            }

            if entropy > max_entropy.0 {
                max_entropy.0 = entropy;
                max_entropy.1 = index;
            }
        };

        let index_groups = self.wave.get_index_groups(start);
        for index_group in index_groups.into_iter() {
            for index in index_group.into_iter() {
                match self.wave[index].update(iter) {
                    Ok(_) => {
                        // update entropy bounds
                        let entropy = self.wave[index].entropy;
                        upd_entropy(entropy, index);
                    }
                    Err(_) => {
                        self.rollback(iter);
                        return;
                    }
                };
            }
        }

        // update entropy
        self.min_entropy = min_entropy;
        self.max_entropy = max_entropy;
    }

    /// Rollback the `Wave`. This is used when a `WaveTile` has ran out of possible `Tile`s.
    fn rollback(&mut self, iter: usize) {
        let (mut min_entropy, mut min_idx) = (usize::MAX, 0);
        let (mut max_entropy, mut max_idx) = (0, 0);

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

        for (i, wavetile) in self.wave.iter_mut().enumerate() {
            wavetile.rollback(3, iter);
            upd_entropy(wavetile.entropy, i);
        }

        // update entropy
        self.min_entropy = (min_entropy, self.wave.get_nd_index(min_idx));
        self.max_entropy = (max_entropy, self.wave.get_nd_index(max_idx));
    }
}

impl<T, const N: usize> Wave<T, N>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Recovers the `T` from type `Wave<'a, T, N>`. Note that `T` must be `Merge` and `Stitch`.
    ///
    /// In the future, this `Merge` requirement may be relaxed to only non-collapsed `Wave`s. This
    /// is a temporary limitation of the API. TODO
    pub fn recover(&self) -> T {
        let ts: Vec<T> = self.wave.iter().map(|wt| wt.recover()).collect();

        let dim = self.wave.raw_dim();

        let array = Array::from_shape_vec(dim, ts).unwrap();

        T::stitch(&array)
    }
}
