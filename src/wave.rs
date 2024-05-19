use std::collections::HashSet;

use ndarray::Array;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::NdIndex;
use rand::RngCore;

use crate::bitset::BitSet;
use crate::bitset::BitSlice;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::traits::Recover;
use crate::traits::Stitch;
use crate::types::DimN;
use crate::util::manhattan_dist;
use crate::wavetile::WaveTile;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::ext::ndarray::WaveArrayExt;
use crate::wavetile::WaveTileError;

pub struct Wave<T, const N: usize>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    WfcNdIndex<N>: NdIndex<DimN<N>>,
{
    pub wave: Array<WaveTile<T, N>, DimN<N>>,

    work: Vec<HashSet<*mut WaveTile<T, N>>>,
    ones: BitSet
}

// TODO: maybe encode whether the `Wave` has collapsed as part of the type?
impl<T, const N: usize> Wave<T, N>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    WfcNdIndex<N>: NdIndex<DimN<N>>,
{
    pub fn new(
        tiles_lr: &[Tile<T, N>],
        tiles_rl: &[Tile<T, N>],
        shape: DimN<N>,
        num_hashes: usize,
    ) -> Self {
        let tiles_lr: Vec<*const Tile<T, N>> = tiles_lr
            .iter()
            .map(|tile| tile as *const Tile<T, N>)
            .collect();
        let tiles_rl: Vec<*const Tile<T, N>> = tiles_rl
            .iter()
            .map(|tile| tile as *const Tile<T, N>)
            .collect();

        let dummy_bitset = BitSet::new();
        let temp_ptr = {
            let ptr: &BitSlice = &dummy_bitset;
            ptr as *const BitSlice
        };

        let wave = Array::from_shape_fn(shape, |i| {
            let i = i.into_dimension();
            let i = i.as_array_view();

            let mut parity: usize = i.sum();
            parity %= 2;

            let index: WfcNdIndex<N> = i
                .as_slice()
                .expect("not in standard order!")
                .try_into()
                .unwrap();

            if parity == 0 {
                WaveTile::new(tiles_lr.clone(), index, num_hashes, parity, temp_ptr)
            } else {
                WaveTile::new(tiles_rl.clone(), index, num_hashes, parity, temp_ptr)
            }
        });

        // this is the maximal distance between any two points in our array
        let max_man_dist = wave.max_manhattan_dist();

        let mut wave = Wave {
            wave,

            work: vec![HashSet::new(); max_man_dist],

            ones: BitSet::ones(2 * N * num_hashes)
        };

        let n = wave.wave.len();

        let get_wavetile_neighbors = |index: usize| -> [[Option<*mut WaveTile<T, N>>; 2]; N] {
            let index = wave.wave.get_nd_index(index);
            let neighbor_indices = wave.wave.get_index_neighbors(index);

            neighbor_indices
                .iter()
                .map(|[l, r]| {
                    [
                        l.map(|index| &mut wave.wave[index] as *mut WaveTile<T, N>),
                        r.map(|index| &mut wave.wave[index] as *mut WaveTile<T, N>),
                    ]
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };

        let get_neighbor_hashes =
            |wts: [[Option<*mut WaveTile<T, N>>; 2]; N]| -> [[*const BitSlice; 2]; N] {
                wts.map(|[l, r]| {
                    [
                        l.map(|wt| {
                            // SAFETY: WaveTile array size is fixed, so it is not moved.
                            let wt = unsafe { &*wt };
                            let hashes: &BitSlice = &wt.hashes;
                            hashes as *const BitSlice
                        }).unwrap_or({
                            let ones: &BitSlice = &wave.ones;
                            ones as *const BitSlice
                        }),
                        r.map(|wt| {
                            // SAFETY: ibid
                            let wt = unsafe { &*wt };
                            let hashes: &BitSlice = &wt.hashes;
                            hashes as *const BitSlice
                        }).unwrap_or({
                            let ones: &BitSlice = &wave.ones;
                            ones as *const BitSlice
                        }),
                    ]
                })
            };

        let wavetile_neighbors: Vec<[[Option<*mut WaveTile<T, N>>; 2]; N]> =
            (0..n).map(get_wavetile_neighbors).collect();

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
            for (wavetile, neighbor_wavetiles) in
                wave.wave.iter_mut().zip(wavetile_neighbors.into_iter())
            {
                let hashes = get_neighbor_hashes(neighbor_wavetiles);

                wavetile.neighbors = neighbor_wavetiles;

                // this is where we overwrite the temporary pointer
                wavetile.neighbor_hashes = hashes;
            }
        }

        wave
    }

    /// Collapse the [`Wave`].
    pub fn collapse<R>(&mut self, rng: &mut R) 
    where R: RngCore + ?Sized
    {
        for iter in 0.. {
            let [(_min, min_idx), (max, _max_idx)] = self.get_entropy();
            if max < 2 {
                break;
            }

            let wt_min = &mut self.wave[min_idx];

            let next = wt_min.collapse(rng, iter);
            self.work[0].extend(next.into_iter().flat_map(|axis| axis.into_iter()).flatten()); // all distance 1
            if self.propagate(iter, min_idx).is_err() {
                self.rollback(iter);
            }
        }
    }

    fn propagate(&mut self, iter: usize, index: WfcNdIndex<N>) -> Result<(), WaveTileError> {
        for d in 0.. {
            let mut next_work = HashSet::new();
            let work = self.work[d].drain();

            for wt in work {
                // SAFETY: `self.wave`'s size is unchanged during collapse
                let wt: &mut WaveTile<T, N> = unsafe { &mut *wt };

                let next = wt
                    .update(iter)?
                    .into_iter()
                    .flat_map(|axis| axis.into_iter())
                    .filter_map(|wt| {
                        wt.filter(|&wt| {
                            // SAFETY: `self.wave`'s size is unchanged during collapse
                            let wt: &mut WaveTile<T, N> = unsafe { &mut *wt };

                            manhattan_dist(index, wt.index) == d + 2
                        })
                    });

                next_work.extend(next);
            }

            // the wave stops as soon as no more work is required
            if next_work.is_empty() {
                break;
            }

            // all work of distance d
            self.work[d + 1] = next_work;
        }

        Ok(())
    }

    /// Rollback the [`Wave`]. This is used when a [`WaveTile`] has ran out of possible [`Tile`]s.
    fn rollback(&mut self, iter: usize) {
        // println!("rolling back wave");

        for wavetile in &mut self.wave {
            wavetile.rollback(3, iter);
        }
    }

    fn get_entropy(&mut self) -> [(usize, WfcNdIndex<N>); 2] {
        self.wave
            .iter()
            .fold([(usize::MAX, [0; N]), (0, [0; N])], |acc, wt| {
                let [mut min_entropy, mut max_entropy] = acc;

                if wt.entropy > 1 && wt.entropy < min_entropy.0 {
                    min_entropy.0 = wt.entropy;
                    min_entropy.1 = wt.index;
                }

                if wt.entropy > max_entropy.0 {
                    max_entropy.0 = wt.entropy;
                    max_entropy.1 = wt.index;
                }

                [min_entropy, max_entropy]
            })
    }
}

impl<T, const N: usize> Recover<T, T, N> for Wave<T, N>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    WfcNdIndex<N>: NdIndex<DimN<N>>,
{
    /// Recovers the `T` from type [`Wave`]. Note that `T` must be [`Merge`] and [`Stitch`].
    ///
    /// In the future, this [`Merge`] requirement may be relaxed to only non-collapsed [`Wave`]s. This
    /// is a temporary limitation of the API. TODO
    fn recover(&self) -> T {
        let ts: Vec<T> = self.wave.iter().map(|wt| wt.recover()).collect();

        let dim = self.wave.raw_dim();

        let array = Array::from_shape_vec(dim, ts).unwrap();

        T::stitch(&array)
    }
}
