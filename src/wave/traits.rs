use std::collections::HashSet;

use ndarray::{Dimension, NdIndex};
use rand::RngCore;

use crate::data::TileSet;
use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::surface::Surface;
use crate::traits::{Recover, WaveTileable};
use crate::types::DimN;
use crate::util::manhattan_dist;
use crate::wavetile::{WaveTile, WaveTileError};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use self::private::ParFns;

pub trait WaveBase<Inner, Outer, S, const N: usize>
where
    Inner: WaveTileable<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: Surface<N>,
{
    fn init(tileset: &mut TileSet<Inner, Outer, S, N>, shape: DimN<N>) -> Self;
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

#[cfg(feature = "parallel")]
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
            let [(_min, min_idx), (max, _max_idx)] = <Self as private::Fns<N>>::get_entropy(self);
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

#[cfg(feature = "parallel")]
impl<Inner, Outer, S, const N: usize> ParWave<Inner, Outer, S, N>
    for super::Wave<Inner, Outer, S, N>
where
    super::Wave<Inner, Outer, S, N>: Recover<Outer, Inner = Inner>,
    Inner: Send + Sync + WaveTileable<Inner, Outer, N>,
    DimN<N>: Dimension,
    WfcNdIndex<N>: NdIndex<DimN<N>>,
    S: Surface<N>,
{
    fn collapse_parallel<R>(&mut self, rng: &mut R) -> Outer
    where
        R: RngCore + ?Sized,
    {
        for iter in 0.. {
            let [(_min, min_idx), (max, _max_idx)] = <Self as private::Fns<N>>::get_entropy(self);
            if max < 2 {
                break;
            }

            let wt_min = &mut self.wave[min_idx];

            let next = wt_min.collapse(rng, iter);
            self.work[0].extend(next.into_iter().flat_map(|axis| axis.into_iter()).flatten()); // all distance 1
            if <Self as private::Fns<N>>::propagate(self, iter, min_idx).is_err() {
                self.rollback_parallel(iter);
            }
        }

        <Self as Recover<Outer>>::recover(self)
    }
}

mod private {
    use crate::{ext::ndarray::NdIndex as WfcNdIndex, wavetile::WaveTileError};

    pub trait Fns<const N: usize> {
        fn propagate(&mut self, iter: usize, index: WfcNdIndex<N>) -> Result<(), WaveTileError>;
        fn rollback(&mut self, iter: usize);
        fn get_entropy(&self) -> [(usize, WfcNdIndex<N>); 2];
    }

    #[cfg(feature = "parallel")]
    pub trait ParFns<const N: usize> {
        fn propagate_parallel(
            &mut self,
            iter: usize,
            index: WfcNdIndex<N>,
        ) -> Result<(), WaveTileError>;
        fn rollback_parallel(&mut self, iter: usize);
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

            for mut wt in work {
                // SAFETY: `self.wave`'s size is unchanged during collapse
                let wt: &mut WaveTile<Inner, N> = &mut wt;

                let next = wt
                    .update(iter)?
                    .into_iter()
                    .flat_map(|axis| axis.into_iter())
                    .filter_map(|wt| {
                        wt.filter(|&wt| {
                            // SAFETY: `self.wave`'s size is unchanged during collapse
                            let wt: &WaveTile<Inner, N> = &wt;

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

#[cfg(feature = "parallel")]
impl<Inner, Outer, S, const N: usize> private::ParFns<N> for super::Wave<Inner, Outer, S, N>
where
    Inner: Send + Sync + WaveTileable<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: Surface<N>,
{
    fn propagate_parallel(
        &mut self,
        iter: usize,
        index: WfcNdIndex<N>,
    ) -> Result<(), WaveTileError> {
        for d in 0.. {
            // process these in parallel
            let next_work = self.work[d]
                .par_drain()
                .flat_map(|mut wt| {
                    wt.update(iter)
                        .expect("oops")
                        .into_par_iter()
                        .flat_map(|axis| axis.into_par_iter())
                        .filter_map(|wt| wt.filter(|&wt| manhattan_dist(index, wt.index) == d + 2))
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

    fn rollback_parallel(&mut self, iter: usize) {
        self.wave.par_iter_mut().for_each(|wavetile| {
            wavetile.rollback(3, iter);
        });
    }
}
