use std::collections::HashSet;
use std::marker::PhantomData;

use ndarray::Array;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::NdIndex;

use crate::bitset::BitSet;
use crate::bitset::BitSlice;
use crate::ext::ndarray::WaveArrayExt;
use crate::surface::Surface;
use crate::traits::Merge;
use crate::traits::Recover;
use crate::traits::WaveTileable;
use crate::types::DimN;
use crate::wavetile::WaveTile;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::TileSet;

pub use traits::WaveBase;

pub struct Wave<Inner, Outer, S, const N: usize>
where
    Inner: WaveTileable<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: Surface<N>,
{
    pub wave: Array<WaveTile<Inner, N>, DimN<N>>,
    work: Vec<HashSet<*mut WaveTile<Inner, N>>>,
    ones: BitSet,

    _outer: PhantomData<Outer>,
    _s: PhantomData<S>,
}

impl<Inner, Outer, S, const N: usize> WaveBase<Inner, Outer, S, N> for Wave<Inner, Outer, S, N>
where
    Inner: WaveTileable<Inner, Outer, N>,
    DimN<N>: Dimension,
    WfcNdIndex<N>: NdIndex<DimN<N>>,
    S: Surface<N>,
{
    // TODO: remove ndarray from public facing API
    fn init(tileset: &TileSet<Inner, Outer, S, N>, shape: DimN<N>) -> Self {
        let (tiles, co_tiles) = tileset.get_tile_ptrs();
        let num_hashes = tileset.num_hashes;
        let dummy = BitSet::new();
        let dummy_ptr: *const BitSlice = &dummy as &BitSlice;

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

            let tiles = if parity == 0 {
                tiles.clone()
            } else {
                co_tiles.clone()
            };

            WaveTile::new(tiles, index, num_hashes, parity, dummy_ptr)
        });

        // TODO: use shape to compute this instead
        let max_man_dist = wave.max_manhattan_dist();

        let mut wave = Wave {
            wave,
            work: vec![HashSet::new(); max_man_dist],
            ones: BitSet::ones(2 * N * num_hashes),
            _outer: PhantomData::<Outer>,
            _s: PhantomData::<S>,
        };

        let n = wave.wave.len();

        let get_wavetile_neighbors = |index: usize| -> [[Option<*mut WaveTile<Inner, N>>; 2]; N] {
            let index = wave.wave.get_nd_index(index);
            let neighbor_indices = wave.wave.get_index_neighbors(index);

            neighbor_indices
                .iter()
                .map(|[l, r]| {
                    [
                        l.map(|index| &mut wave.wave[index] as *mut WaveTile<Inner, N>),
                        r.map(|index| &mut wave.wave[index] as *mut WaveTile<Inner, N>),
                    ]
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };

        let get_neighbor_hashes =
            |wts: [[Option<*mut WaveTile<Inner, N>>; 2]; N]| -> [[*const BitSlice; 2]; N] {
                wts.map(|[l, r]| {
                    [
                        l.map(|wt| {
                            // SAFETY: WaveTile array size is fixed, so it is not moved.
                            let wt = unsafe { &*wt };
                            &wt.hashes as &BitSlice
                        })
                        .unwrap_or(&wave.ones) as *const BitSlice,
                        r.map(|wt| {
                            // SAFETY: WaveTile array size is fixed, so it is not moved.
                            let wt = unsafe { &*wt };
                            &wt.hashes as &BitSlice
                        })
                        .unwrap_or(&wave.ones) as *const BitSlice,
                    ]
                })
            };

        let wavetile_neighbors: Vec<[[Option<*mut WaveTile<Inner, N>>; 2]; N]> =
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
}

impl<Inner, Outer, S, const N: usize> Recover<Outer> for Wave<Inner, Outer, S, N>
where
    Inner: Merge + WaveTileable<Inner, Outer, N>,
    S: Surface<N>,
    DimN<N>: Dimension,
{
    type Inner = Inner;

    /// Recovers the `Outer` type for type `Wave<Inner, Outer, S, N>`.
    ///
    /// In the future, this `Merge` requirement may be relaxed to only non-collapsed `WaveTile`s.
    /// This is a temporary limitation of the API. TODO
    fn recover(&self) -> Outer {
        let ts = self.wave.iter().map(|wt| wt.recover()).collect();

        let dim = self.wave.raw_dim();

        let array = Array::from_shape_vec(dim, ts).unwrap();

        Inner::stitch(&array).recover()
    }
}

pub mod traits {
    use std::collections::HashSet;

    use ndarray::{Dimension, NdIndex};
    use rand::RngCore;

    use crate::data::TileSet;
    use crate::ext::ndarray::NdIndex as WfcNdIndex;
    use crate::surface::Surface;
    use crate::traits::Recover;
    use crate::traits::WaveTileable;
    use crate::types::DimN;
    use crate::util::manhattan_dist;
    use crate::wavetile::{WaveTile, WaveTileError};

    pub trait WaveBase<Inner, Outer, S, const N: usize>
    where
        Inner: WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        S: Surface<N>,
    {
        fn init(tileset: &TileSet<Inner, Outer, S, N>, shape: DimN<N>) -> Self;
    }

    pub trait Wave<Inner, Outer, S, const N: usize>:
        WaveBase<Inner, Outer, S, N> + private::Fns<N>
    where
        Inner: WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        S: Surface<N>,
    {
        fn collapse<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized;
    }

    pub trait ParWave<Inner, Outer, S, const N: usize>:
        WaveBase<Inner, Outer, S, N> + private::Fns<N> + private::ParFns<N>
    where
        Inner: Send + Sync + WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        S: Surface<N>,
    {
        fn collapse_parallel<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized;
    }

    impl<Inner, Outer, S, const N: usize> Wave<Inner, Outer, S, N> for super::Wave<Inner, Outer, S, N>
    where
        super::Wave<Inner, Outer, S, N>: Recover<Outer, Inner = Inner>,
        Inner: WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        WfcNdIndex<N>: NdIndex<DimN<N>>,
        S: Surface<N>,
    {
        fn collapse<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized,
        {
            for iter in 0.. {
                let [(_min, min_idx), (max, _max_idx)] =
                    <Self as private::Fns<N>>::get_entropy(self);
                if max < 2 {
                    break;
                }

                let wt_min = &mut self.wave[min_idx];

                let next = wt_min.collapse(rng, iter);
                self.work[0].extend(next.into_iter().flat_map(|axis| axis.into_iter()).flatten()); // all distance 1
                if <Self as private::Fns<N>>::propagate(self, iter, min_idx).is_err() {
                    <Self as private::Fns<N>>::rollback(self, iter);
                }
            }

            <Self as Recover<Outer>>::recover(self)
        }
    }

    impl<Inner, Outer, S, const N: usize> ParWave<Inner, Outer, S, N>
        for super::Wave<Inner, Outer, S, N>
    where
        Inner: Send + Sync + WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        WfcNdIndex<N>: NdIndex<DimN<N>>,
        S: Surface<N>,
    {
        fn collapse_parallel<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized,
        {
            todo!()
        }
    }

    mod private {
        use crate::{ext::ndarray::NdIndex as WfcNdIndex, wavetile::WaveTileError};

        pub trait Fns<const N: usize> {
            fn propagate(&mut self, iter: usize, index: WfcNdIndex<N>)
                -> Result<(), WaveTileError>;
            fn rollback(&mut self, iter: usize);
            fn get_entropy(&self) -> [(usize, WfcNdIndex<N>); 2];
        }

        pub trait ParFns<const N: usize> {
            fn propagate_parallel(&mut self);
            fn rollback_parallel(&mut self);
        }
    }

    impl<Inner, Outer, S, const N: usize> private::Fns<N> for super::Wave<Inner, Outer, S, N>
    where
        Inner: WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        S: Surface<N>,
    {
        // TODO: maybe iter should be part of the wave? That way, we don't have to pass it through
        // everything
        fn propagate(&mut self, iter: usize, index: WfcNdIndex<N>) -> Result<(), WaveTileError> {
            for d in 0.. {
                let mut next_work = HashSet::new();
                let work = self.work[d].drain();

                for wt in work {
                    // SAFETY: `self.wave`'s size is unchanged during collapse
                    let wt: &mut WaveTile<Inner, N> = unsafe { &mut *wt };

                    let next = wt
                        .update(iter)?
                        .into_iter()
                        .flat_map(|axis| axis.into_iter())
                        .filter_map(|wt| {
                            wt.filter(|&wt| {
                                // SAFETY: `self.wave`'s size is unchanged during collapse
                                let wt: &mut WaveTile<Inner, N> = unsafe { &mut *wt };

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

        fn rollback(&mut self, iter: usize) {
            for wavetile in &mut self.wave {
                // TODO: enable rollback of any number of iterations
                wavetile.rollback(3, iter);
            }
        }

        fn get_entropy(&self) -> [(usize, WfcNdIndex<N>); 2] {
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

    impl<Inner, Outer, S, const N: usize> private::ParFns<N> for super::Wave<Inner, Outer, S, N>
    where
        Inner: WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        S: Surface<N>,
    {
        fn propagate_parallel(&mut self) {
            todo!()
        }

        fn rollback_parallel(&mut self) {
            todo!()
        }
    }
}
