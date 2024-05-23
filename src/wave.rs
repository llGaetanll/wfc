#![allow(unused_imports)]

use std::collections::HashSet;
use std::marker::PhantomData;

use ndarray::Array;
use ndarray::Dimension;
use rand::RngCore;

use crate::bitset::BitSet;
use crate::traits;
use crate::traits::BoundaryHash;
use crate::traits::Flat;
use crate::traits::KleinBottle;
use crate::traits::Merge;
use crate::traits::ProjectivePlane;
use crate::traits::Stitch;
use crate::traits::Torus;
use crate::types::DimN;
use crate::wavetile::WaveTile;
use crate::Recover;

pub type FlatWave<Inner, Outer, const N: usize> = Wave<Inner, Outer, Flat, N>;
pub type TorusWave<Inner, Outer> = Wave<Inner, Outer, Torus, 2>;
pub type ProjectiveWave<Inner, Outer> = Wave<Inner, Outer, ProjectivePlane, 2>;
pub type KleinWave<Inner, Outer> = Wave<Inner, Outer, KleinBottle, 2>;

pub struct Wave<Inner, Outer, S, const N: usize>
where
    Inner: traits::WaveTile<Inner, Outer, N>,
    S: traits::Surface<N>,
{
    pub wave: Array<WaveTile<Inner, N>, DimN<N>>,
    work: Vec<HashSet<*mut WaveTile<Inner, N>>>,
    ones: BitSet,

    _outer: PhantomData<Outer>,
    _s: PhantomData<S>,
}

impl<Inner, Outer, S, const N: usize> traits::WaveBase<Inner, Outer, S, N>
    for Wave<Inner, Outer, S, N>
where
    Inner: traits::WaveTile<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: traits::Surface<N>,
{
    fn init(tileset: &crate::TileSet<Inner, Outer, N>, shape: DimN<N>) -> Self {
        todo!()
    }
}

impl<Inner, Outer, S, const N: usize> traits::Wave<Inner, Outer, S, N> for Wave<Inner, Outer, S, N>
where
    Inner: traits::WaveTile<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: traits::Surface<N>,
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
    Inner: Send + Sync + traits::WaveTile<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: traits::Surface<N>,
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
    T: Merge + traits::WaveTile<T, T, N>,
    S: traits::Surface<N>,
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
