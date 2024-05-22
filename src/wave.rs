#![allow(unused_imports)]

use std::collections::HashSet;
use std::marker::PhantomData;

use ndarray::Array;
use ndarray::Dimension;

use crate::bitset::BitSet;
use crate::traits;
use crate::traits::Flat;
use crate::traits::KleinBottle;
use crate::traits::ProjectivePlane;
use crate::traits::Torus;
use crate::traits::WrappingSurface;
use crate::types::DimN;
use crate::wavetile::WaveTile;

pub type TorusWave<Inner, Outer> = Wave<Inner, Outer, Torus, 2>;
pub type ProjectiveWave<Inner, Outer> = Wave<Inner, Outer, ProjectivePlane, 2>;
pub type KleinWave<Inner, Outer> = Wave<Inner, Outer, KleinBottle, 2>;

pub struct Wave<Inner, Outer, S, const N: usize>
where
    Inner: traits::WaveTile<Inner, Outer, N>,
    S: traits::Surface,
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
    S: traits::Surface,
    DimN<N>: Dimension,
{
    fn init(tileset: &crate::TileSet<Inner, Outer, N>, shape: DimN<N>) -> Self {
        todo!()
    }
}

pub mod flat {
    use std::collections::HashSet;
    use std::marker::PhantomData;

    use ndarray::Array;
    use ndarray::Dimension;
    use ndarray::IntoDimension;
    use ndarray::NdIndex;
    use rand::RngCore;

    use crate::bitset::BitSet;
    use crate::bitset::BitSlice;
    use crate::tile::Tile;
    use crate::traits;
    use crate::traits::Flat;
    use crate::traits::KleinBottle;
    use crate::traits::ProjectivePlane;
    use crate::traits::Torus;
    use crate::traits::WrappingSurface;
    use crate::types::DimN;
    use crate::util::manhattan_dist;
    use crate::wavetile::WaveTile;

    use crate::ext::ndarray::NdIndex as WfcNdIndex;
    use crate::ext::ndarray::WaveArrayExt;
    use crate::wavetile::WaveTileError;

    use super::Wave;

    impl<Inner, Outer, const N: usize> traits::Wave<Inner, Outer, Flat, N> for Wave<Inner, Outer, Flat, N>
    where
        Inner: traits::WaveTile<Inner, Outer, N>,
        DimN<N>: Dimension,
    {
        fn collase<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized,
        {
            todo!()
        }
    }

    impl<Inner, Outer, const N: usize> traits::ParWave<Inner, Outer, Flat, N> for Wave<Inner, Outer, Flat, N>
    where
        Inner: Send + Sync + traits::WaveTile<Inner, Outer, N>,
        DimN<N>: Dimension,
    {
        fn collase_parallel<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized,
        {
            todo!()
        }
    }
}

pub mod wrapping {
    use std::collections::HashSet;
    use std::marker::PhantomData;

    use ndarray::Array;
    use ndarray::Dimension;
    use ndarray::IntoDimension;
    use ndarray::NdIndex;
    use rand::RngCore;

    use crate::bitset::BitSet;
    use crate::bitset::BitSlice;
    use crate::tile::Tile;
    use crate::traits;
    use crate::traits::Flat;
    use crate::traits::KleinBottle;
    use crate::traits::ProjectivePlane;
    use crate::traits::Torus;
    use crate::traits::WrappingSurface;
    use crate::types::DimN;
    use crate::util::manhattan_dist;
    use crate::wavetile::WaveTile;

    use crate::ext::ndarray::NdIndex as WfcNdIndex;
    use crate::ext::ndarray::WaveArrayExt;
    use crate::wavetile::WaveTileError;

    use super::Wave;

    impl<Inner, Outer, S, const N: usize> traits::Wave<Inner, Outer, S, N> for Wave<Inner, Outer, S, N>
    where
        Inner: traits::WaveTile<Inner, Outer, N>,
        S: traits::WrappingSurface<N>,
        DimN<N>: Dimension,
    {
        fn collase<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized,
        {
            todo!()
        }
    }

    impl<Inner, Outer, S, const N: usize> traits::ParWave<Inner, Outer, S, N> for Wave<Inner, Outer, S, N>
    where
        Inner: Send + Sync + traits::WaveTile<Inner, Outer, N>,
        S: traits::WrappingSurface<N>,
        DimN<N>: Dimension,
    {
        fn collase_parallel<R>(&mut self, rng: &mut R) -> Outer
        where
            R: RngCore + ?Sized,
        {
            todo!()
        }
    }
}
