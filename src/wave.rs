#![allow(unused_imports)]

use std::collections::HashSet;
use std::marker::PhantomData;

use ndarray::Array;
use ndarray::Dimension;
use rand::RngCore;

use crate::bitset::BitSet;
use crate::traits;
use crate::traits::Flat;
use crate::traits::KleinBottle;
use crate::traits::ProjectivePlane;
use crate::traits::Torus;
use crate::types::DimN;
use crate::wavetile::WaveTile;

pub type FlatWave<Inner, Outer, const N: usize> = Wave<Inner, Outer, Flat, N>;
pub type TorusWave<Inner, Outer> = Wave<Inner, Outer, Torus, 2>;
pub type ProjectiveWave<Inner, Outer> = Wave<Inner, Outer, ProjectivePlane, 2>;
pub type KleinWave<Inner, Outer> = Wave<Inner, Outer, KleinBottle, 2>;

pub struct Wave<Inner, Outer, S, const N: usize>
where
    Inner: traits::WaveTile<Inner, Outer, N>,
    S: traits::Surface<N>,
{
    wave: Array<WaveTile<Inner, N>, DimN<N>>,
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
    fn collase<R>(&mut self, rng: &mut R) -> Outer
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
    fn collase_parallel<R>(&mut self, rng: &mut R) -> Outer
    where
        R: RngCore + ?Sized,
    {
        todo!()
    }
}
