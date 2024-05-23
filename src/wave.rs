use std::collections::HashSet;
use std::marker::PhantomData;

use ndarray::Array;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::NdIndex;
use rand::RngCore;

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
    fn init(tileset: &TileSet<Inner, Outer, N>, shape: DimN<N>) -> Self {
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
                            let hashes: &BitSlice = &wt.hashes;
                            hashes as *const BitSlice
                        })
                        .unwrap_or({
                            let ones: &BitSlice = &wave.ones;
                            ones as *const BitSlice
                        }),
                        r.map(|wt| {
                            // SAFETY: ibid
                            let wt = unsafe { &*wt };
                            let hashes: &BitSlice = &wt.hashes;
                            hashes as *const BitSlice
                        })
                        .unwrap_or({
                            let ones: &BitSlice = &wave.ones;
                            ones as *const BitSlice
                        }),
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

impl<Inner, Outer, S, const N: usize> traits::Wave<Inner, Outer, S, N> for Wave<Inner, Outer, S, N>
where
    Inner: WaveTileable<Inner, Outer, N>,
    DimN<N>: Dimension,
    WfcNdIndex<N>: NdIndex<DimN<N>>,
    S: Surface<N>,
{
    fn collapse<R>(&mut self, rng: &mut R) -> Outer
    where
        R: RngCore + ?Sized,
    {
        todo!()
    }
}

impl<Inner, Outer, S, const N: usize> traits::ParWave<Inner, Outer, S, N>
    for Wave<Inner, Outer, S, N>
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

impl<T, S, const N: usize> Recover<T> for Wave<T, T, S, N>
where
    T: Merge + WaveTileable<T, T, N>,
    S: Surface<N>,
    DimN<N>: Dimension,
{
    type Inner = T;

    /// Recovers the `T` for type `Wave<T, T, S, N>`
    ///
    /// In the future, this `Merge` requirement may be relaxed to only non-collapsed `WaveTile`s.
    /// This is a temporary limitation of the API. TODO
    fn recover(&self) -> T {
        let ts = self.wave.iter().map(|wt| wt.recover()).collect();

        let dim = self.wave.raw_dim();

        let array = Array::from_shape_vec(dim, ts).unwrap();

        T::stitch(&array)
    }
}

pub mod traits {
    use ndarray::Dimension;
    use rand::RngCore;

    use crate::surface::Surface;
    use crate::traits::WaveTileable;
    use crate::types::DimN;
    use crate::TileSet;

    pub trait WaveBase<Inner, Outer, S, const N: usize>
    where
        Inner: WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        S: Surface<N>,
    {
        fn init(tileset: &TileSet<Inner, Outer, N>, shape: DimN<N>) -> Self;
    }

    pub trait Wave<Inner, Outer, S, const N: usize>:
        WaveBase<Inner, Outer, S, N> + private::Fns
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
        WaveBase<Inner, Outer, S, N> + private::ParFns
    where
        Inner: Send + Sync + WaveTileable<Inner, Outer, N>,
        DimN<N>: Dimension,
        S: Surface<N>,
    {
        fn collapse_parallel<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized;
    }

    mod private {
        pub trait Fns {
            fn propagate(&mut self);
            fn rollback(&mut self);
            fn get_entropy(&self);
        }

        pub trait ParFns {
            fn propagate_parallel(&mut self);
            fn rollback_parallel(&mut self);
            fn get_entropy(&self);
        }
    }

    impl<Inner, Outer, S, const N: usize> private::Fns for super::Wave<Inner, Outer, S, N>
    where
        Inner: WaveTileable<Inner, Outer, N>,
        S: Surface<N>,
    {
        fn propagate(&mut self) {
            todo!()
        }

        fn rollback(&mut self) {
            todo!()
        }

        fn get_entropy(&self) {
            todo!()
        }
    }

    impl<Inner, Outer, S, const N: usize> private::ParFns for super::Wave<Inner, Outer, S, N>
    where
        Inner: WaveTileable<Inner, Outer, N>,
        S: Surface<N>,
    {
        fn propagate_parallel(&mut self) {
            todo!()
        }

        fn rollback_parallel(&mut self) {
            todo!()
        }

        fn get_entropy(&self) {
            todo!()
        }
    }
}
