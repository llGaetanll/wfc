use std::collections::HashSet;

use ndarray::parallel::prelude::*;
use ndarray::Array;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::NdIndex;

use rayon::prelude::*;

use crate::bitset::BitSlice;
use crate::tile::Tile;
use crate::traits::BoundaryHash;
use crate::traits::Merge;
use crate::traits::Recover;
use crate::traits::Stitch;
use crate::types::DimN;
use crate::wavetile::WaveTile;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::ext::ndarray::WaveArrayExt;
use crate::wavetile::WaveTileError;
use crate::wavetile::WaveTilePtr;

pub struct Wave<T, const N: usize>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    pub wave: Array<WaveTile<T, N>, DimN<N>>,

    work: Vec<HashSet<WaveTilePtr<T, N>>>,
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

        let wave = Array::from_shape_fn(shape, |i| {
            let i = i.into_dimension();
            let i = i.as_array_view();

            let mut parity: usize = i.sum();
            parity %= 2;

            let index: [usize; N] = i
                .as_slice()
                .expect("not in standard order!")
                .try_into()
                .unwrap();

            if parity == 0 {
                WaveTile::new(tiles_lr.clone(), index, num_hashes, parity)
            } else {
                WaveTile::new(tiles_rl.clone(), index, num_hashes, parity)
            }
        });

        // this is the maximal distance between any two points in our array
        let max_man_dist = wave.max_manhattan_dist();

        let mut wave = Wave {
            wave,

            work: vec![HashSet::new(); max_man_dist],
        };

        // wave.entropy = wave.wave.iter_mut().map(|wt| wt as *mut WaveTile<T, N>).collect();

        let n = wave.wave.len();

        let get_wavetile_neighbors = |index: usize| -> [[Option<WaveTilePtr<T, N>>; 2]; N] {
            let index = wave.wave.get_nd_index(index);
            let neighbor_indices = wave.wave.get_index_neighbors(index);

            neighbor_indices
                .iter()
                .map(|[l, r]| {
                    [
                        l.map(|index| {
                            WaveTilePtr::from(&mut wave.wave[index] as *mut WaveTile<T, N>)
                        }),
                        r.map(|index| {
                            WaveTilePtr::from(&mut wave.wave[index] as *mut WaveTile<T, N>)
                        }),
                    ]
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };

        let get_neighbor_hashes =
            |wts: [[Option<WaveTilePtr<T, N>>; 2]; N]| -> [[Option<*const BitSlice>; 2]; N] {
                wts.map(|[l, r]| {
                    [
                        l.map(|wt| {
                            // SAFETY: WaveTile array size is fixed, so it is not moved.
                            let wt = unsafe { &*wt };
                            let hashes: &BitSlice = &wt.hashes;
                            hashes as *const BitSlice
                        }),
                        r.map(|wt| {
                            // SAFETY: ibid
                            let wt = unsafe { &*wt };
                            let hashes: &BitSlice = &wt.hashes;
                            hashes as *const BitSlice
                        }),
                    ]
                })
            };

        let wavetile_neighbors: Vec<[[Option<WaveTilePtr<T, N>>; 2]; N]> =
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
                wavetile.neighbor_hashes = hashes;
            }
        }

        wave
    }

    pub fn collapse2(&mut self) {
        for iter in 0.. {
            let (wt_min_idx, wt_max_idx) = self.get_entropy();
            let wt_max = &self.wave[wt_max_idx];

            if wt_max.entropy < 2 {
                break;
            }

            let wt_min = &mut self.wave[wt_min_idx];

            let next = wt_min.collapse2(0);
            self.work[0].extend(next.into_iter().flat_map(|axis| axis.into_iter()).flatten()); // all distance 1
            if self.propagate2(iter, wt_min_idx).is_err() {
                self.rollback(iter);
            }
        }
    }

    fn get_entropy(&mut self) -> ([usize; N], [usize; N]) {
        let id = [(usize::MAX, [0; N]), (0, [0; N])];

        let [(_, wt_min), (_, wt_max)] = self
            .wave
            .par_iter_mut()
            .map(|wavetile| (wavetile.entropy, wavetile.index))
            .fold(
                || id,
                |acc, (entropy, index)| {
                    let [mut min_entropy, mut max_entropy] = acc;

                    if entropy > 1 && entropy < min_entropy.0 {
                        min_entropy.0 = entropy;
                        min_entropy.1 = index;
                    }

                    if entropy > max_entropy.0 {
                        max_entropy.0 = entropy;
                        max_entropy.1 = index;
                    }

                    [min_entropy, max_entropy]
                },
            )
            .reduce(
                || id,
                |a, b| {
                    let mut entropy_range = a;

                    if b[0].0 < a[0].0 {
                        entropy_range[0] = b[0];
                    }

                    if b[1].0 > a[1].0 {
                        entropy_range[1] = b[1];
                    }

                    entropy_range
                },
            );

        (wt_min, wt_max)
    }

    pub fn propagate2(&mut self, iter: usize, index: [usize; N]) -> Result<(), WaveTileError> {
        for d in 0.. {
            // process these in parallel
            let next_work = self.work[d]
                .par_drain()
                .flat_map(|mut wt| {
                    wt.update2(iter)
                        .expect("oops")
                        .into_par_iter()
                        .flat_map(|axis| axis.into_par_iter())
                        .filter_map(|wt| {
                            wt.filter(|&wt| {
                                manhattan_dist(index, wt.index) == d + 2
                            })
                        })
                })
                .collect::<HashSet<_>>();

            // the wave stops as soon as no more work is required
            if next_work.is_empty() {
                break;
            }

            // all work of distance d
            self.work[d + 1] = next_work;
        }

        Ok(())
    }

    /// Collapse the `Wave`
    pub fn collapse(&mut self, starting_index: Option<WfcNdIndex<N>>) /* -> Option<T> */
    {
        /*
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
        */
    }

    /// Propagates the wave from an index `start`
    /// TODO:
    ///  - stop propagation if neighboring tiles haven't updated
    fn propagate(&mut self, start: WfcNdIndex<N>, iter: usize) {
        /*
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
        for index_group in index_groups {
            for index in index_group {
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
        */
    }

    /// Rollback the `Wave`. This is used when a `WaveTile` has ran out of possible `Tile`s.
    fn rollback(&mut self, iter: usize) {
        println!("rolling back wave");

        self.wave.par_iter_mut().for_each(|wavetile| {
            wavetile.rollback(3, iter);
        });
    }
}

impl<T, const N: usize> Recover<T, T, N> for Wave<T, N>
where
    T: BoundaryHash<N> + Clone + Merge + Stitch<T, N>,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Recovers the `T` from type `Wave<T, N>`. Note that `T` must be `Merge` and `Stitch`.
    ///
    /// In the future, this `Merge` requirement may be relaxed to only non-collapsed `Wave`s. This
    /// is a temporary limitation of the API. TODO
    fn recover(&self) -> T {
        let ts: Vec<T> = self.wave.iter().map(|wt| wt.recover()).collect();

        let dim = self.wave.raw_dim();

        let array = Array::from_shape_vec(dim, ts).unwrap();

        T::stitch(&array)
    }
}

// compute the manhattan distance between two points
fn manhattan_dist<const N: usize>(a: [usize; N], b: [usize; N]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| ((a as isize) - (b as isize)).unsigned_abs())
        .sum()
}
