use std::fmt::Debug;
use std::hash::Hash;
use std::time::SystemTime;

use log::info;

use ndarray::Array;
use ndarray::Array2;
use ndarray::Dimension;
use ndarray::NdIndex;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use bit_set::BitSet;

use crate::ext::ndarray::ManhattanIter;
use crate::tile::Tile;
use crate::tileset::TileSet;

use super::traits::Pixel;
use super::traits::SdlTexture;
use super::types;
use super::types::DimN;
use super::wavetile::WaveTile;

/// A `Wave` is a `D` dimensional array of `WaveTile`s.
///
/// `D` is the dimension of the wave, as well as the dimension of each element of each `WaveTile`.
/// `T` is the type of the element of the wave. All `WaveTile`s for a `Wave` hold the same type.
pub struct Wave<'a, T, const N: usize>
where
    T: Hash + Sync + std::fmt::Debug,
    DimN<N>: Dimension, // ensures that [usize; N] is a Dimension implemented by ndarray
    [usize; N]: NdIndex<DimN<N>>, // ensures that any [usize; N] is a valid index into the nd array

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    pub wave: Array<WaveTile<'a, T, N>, DimN<N>>,

    // used to quickly compute diffs on wavetiles that have changed
    pub diffs: Array<bool, DimN<N>>,

    // cached to speed up propagate
    min_entropy: (usize, [usize; N]),
    max_entropy: (usize, [usize; N]),

    shape: DimN<N>,
    tile_size: usize,
}

impl<'a, T, const N: usize> Wave<'a, T, N>
where
    T: Hash + Sync + Send + std::fmt::Debug,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    pub fn new(shape: DimN<N>, tile_set: &'a TileSet<'a, T, N>) -> Self {
        let tiles = &tile_set.tiles;

        let tile_refs: Vec<&'a Tile<'a, T, N>> = tiles.iter().map(|tile| tile).collect();
        let tile_hashes: Vec<&[[BitSet; 2]; N]> = tiles.iter().map(|tile| &tile.hashes).collect();

        let num_hashes = tile_set.hashes.len();
        let wavetile_hashes = Self::merge_tile_bitsets(tile_hashes, num_hashes);

        let num_tiles = tiles.len();

        // the size of each tile
        let tile_size = tiles.first().unwrap().shape;

        Wave {
            min_entropy: (num_tiles, [0; N]),
            max_entropy: (num_tiles, [0; N]),

            wave: Array::from_shape_fn(shape, |_| {
                WaveTile::new(tile_refs.clone(), wavetile_hashes.clone(), num_hashes)
            }),
            diffs: Array::from_shape_fn(shape, |_| false),

            shape,
            tile_size
        }
    }

    /// Collapses the wave
    pub fn collapse(&mut self, starting_index: Option<[usize; N]>) {
        let t0 = SystemTime::now();

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
            info!("collapsing {:?}", index);
            let wavetile = &mut self.wave[index];
            wavetile.collapse();
            self.diffs[index] = true;
        }

        self.propagate(&index);

        // handle the rest of the wave
        loop {
            if self.max_entropy.0 < 2 {
                break;
            };

            {
                // let (index, _) = self.min_entropy();
                let index = self.min_entropy.1;
                info!("collapsing {:?}", index);
                let wavetile = &mut self.wave[index];
                wavetile.collapse();
                self.diffs[index] = true;
            }

            self.propagate(&index);
        }

        let t1 = SystemTime::now();

        info!("Collapsed wave in {:?}", t1.duration_since(t0).unwrap());
    }

    // Computes the total union of the list of bit sets in `tile_hashes`.
    fn merge_tile_bitsets(
        tile_hashes: Vec<&[[BitSet; 2]; N]>,
        capacity: usize,
    ) -> [[BitSet; 2]; N] {
        let mut hashes: [[BitSet; 2]; N] =
            vec![vec![BitSet::with_capacity(capacity); 2].try_into().unwrap(); N]
                .try_into()
                .unwrap();

        let or_bitsets = |main: &mut [[BitSet; 2]; N], other: &[[BitSet; 2]; N]| {
            main.iter_mut().zip(other.iter()).for_each(
                |([main_left, main_right], [other_left, other_right])| {
                    main_left.union_with(other_left);
                    main_right.union_with(other_right);
                },
            )
        };

        for tile_hash in tile_hashes {
            or_bitsets(&mut hashes, tile_hash);
        }

        hashes
    }

    /// Propagates the wave from an index `start`
    /// TODO:
    ///  - stop propagation if neighboring tiles haven't updated
    ///  - currently, a wavetile looks at all of its neighbors, but it only needs to look at a
    ///  subset of those: the ones closer to the center of the wave.
    fn propagate(&mut self, start: &[usize; N]) {
        let t0 = SystemTime::now();
        let mut skips = 0;

        let index_groups = self.get_index_groups(start);

        // println!("{:?}", self.diffs);

        let (mut min_entropy, mut min_idx) = (usize::MAX, self.min_entropy.1.clone());
        let (mut max_entropy, mut max_idx) = (0, self.max_entropy.1.clone());

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

        let it = ManhattanIter::new(start, &self.wave);

        // for index_group in it {
        //     for index in index_group {
        //         let neighbor_wavetiles = self.wave.neighbors(&index);
        //         let neighbor_hashes: [[Option<BitSet>; 2]; N] = neighbor_wavetiles
        //             .iter()
        //             .enumerate()
        //             .map(|(axis, [left, right])| {
        //                 [
        //                     left.map(|left| {
        //                         let left: &[BitSet; 2] = &left.hashes[axis];
        //                         left[1].clone()
        //                     }),
        //                     right.map(|right| {
        //                         let right: &[BitSet; 2] = &right.hashes[axis];
        //                         right[0].clone()
        //                     }),
        //                 ]
        //             })
        //             .collect::<Vec<_>>()
        //             .try_into()
        //             .unwrap();
        //
        //         self.diffs[index] = self.wave[index].update(neighbor_hashes);
        //
        //         // update entropy bounds
        //         let entropy = self.wave[index].entropy;
        //         upd_entropy(entropy, index);
        //     }
        // }

        for index_group in index_groups.into_iter() {
            for index in index_group.into_iter() {
                let neighbor_indices = self.compute_neighbors(&index);

                let skip = !neighbor_indices.iter().any(|(left, right)| {
                    left.map_or(false, |idx| self.diffs[idx])
                        || right.map_or(false, |idx| self.diffs[idx])
                });

                if skip {
                    // if we get to skip the current tile, it's elligible for entropy check
                    let entropy = self.wave[index].entropy;
                    upd_entropy(entropy, index);

                    skips += 1;
                    continue;
                }

                let neighbor_indices = self.compute_neighbors(&index);

                // debug!(
                //     "Propagate [{:?}] - neighbor indices: {:?}",
                //     index, neighbor_indices
                // );

                // note that if a `WaveTile` is on the edge of the wave, it might not have all its
                // neighbors. This also depends on the wrapping logic chosen
                let neighbor_hashes: [[Option<BitSet>; 2]; N] = neighbor_indices
                    .into_iter()
                    .enumerate()
                    .map(|(axis, (left, right))| {
                        let left = left
                            // .filter(|&idx| self.diffs[idx])
                            .map(|idx| {
                                let wavetile = &self.wave[idx];
                                let hashes: &[BitSet; 2] = &wavetile.hashes[axis];
                                hashes[1].clone()
                            });

                        let right = right
                            // .filter(|&idx| self.diffs[idx])
                            .map(|idx| {
                                let wavetile = &self.wave[idx];
                                let hashes: &[BitSet; 2] = &wavetile.hashes[axis];
                                hashes[0].clone()
                            });

                        [left, right]
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                // debug!(
                //     "Propagate [{:?}] - neighbor hashes: {:?}",
                //     index, neighbor_hashes
                // );

                // debug!("updating {:?}", index);

                // TODO: catch if a wavetile has no more options at this stage instead of on
                // collapse?
                self.diffs[index] = self.wave[index].update(neighbor_hashes);

                // update entropy bounds
                let entropy = self.wave[index].entropy;
                upd_entropy(entropy, index);
            }
        }

        // update entropy
        self.min_entropy = (min_entropy, min_idx);
        self.max_entropy = (max_entropy, max_idx);

        // reset diffs
        Array::fill(&mut self.diffs, false);

        let t1 = SystemTime::now();

        info!(
            "Propagate in {:?}. Total skips: {}",
            t1.duration_since(t0).unwrap(),
            skips
        );
    }

    /// TODO: doesn't really belong in wave..
    /// compute an nd index from a given index and the local strides
    fn compute_nd_index(flat_index: usize, strides: &[isize]) -> [usize; N] {
        let mut nd_index: Vec<usize> = Vec::new();

        strides.iter().fold(flat_index, |idx_remain, &stride| {
            let stride = stride as usize;
            nd_index.push(idx_remain / stride);
            idx_remain % stride
        });

        // safe since we fold on stride and |stride| == N
        nd_index.try_into().unwrap()
    }

    /// Given a `starting_index`, this function computes a list A where
    /// A[i] is a list of ND indices B, where for all b in B
    ///
    ///    manhattan_dist(b, starting_index) = i
    ///
    /// This is used to be able to parallelize the wave propagation step.
    fn get_index_groups(&self, starting_index: &[usize; N]) -> Vec<Vec<[usize; N]>> {
        let strides = self.wave.strides();

        // compute manhattan distance between `start` and `index`
        let manhattan_dist = |index: &[usize; N]| -> usize {
            starting_index
                .iter()
                .zip(index.iter())
                .map(|(&a, &b)| ((a as isize) - (b as isize)).abs() as usize)
                .sum()
        };

        // compute the tile farthest away from the starting index
        // this gives us the number of index groups B are in our wave given the `starting_index`
        let max_manhattan_dist = self
            .wave
            .iter()
            .enumerate()
            .map(|(i, _)| Self::compute_nd_index(i, strides))
            .map(|nd_index| manhattan_dist(&nd_index))
            .max()
            .unwrap();

        let mut index_groups: Vec<Vec<[usize; N]>> = vec![Vec::new(); max_manhattan_dist];

        for (index, _) in self.wave.iter().enumerate() {
            let nd_index = Self::compute_nd_index(index, strides);
            let dist = manhattan_dist(&nd_index);

            if dist == 0 {
                continue;
            }

            index_groups[dist - 1].push(nd_index);
        }

        index_groups
    }

    /// For a given `index`, computes all the neighbors of that index
    fn compute_neighbors(
        &self,
        index: &[usize; N],
    ) -> [(Option<[usize; N]>, Option<[usize; N]>); N] {
        // what's amazing here is that we always have as many neighbor pairs as we have dimensions!
        let mut neighbors: [(Option<[usize; N]>, Option<[usize; N]>); N] = [(None, None); N];

        let shape = self.wave.shape();
        let ndim = self.wave.ndim();

        for axis in 0..ndim {
            // min and max indices on the current axis
            let (min_bd, max_bd) = (0, shape[axis]);
            let range = min_bd..max_bd;

            // compute the left and right neighbors on that axis
            let mut axis_neighbors: [Option<[usize; N]>; 2] = [None; 2];
            for (i, delta) in [-1isize, 1isize].iter().enumerate() {
                let axis_index = ((index[axis] as isize) + delta) as usize;

                let mut axis_neighbor: Option<[usize; N]> = None;

                if range.contains(&axis_index) {
                    let mut neighbor_axis_index = index.clone();
                    neighbor_axis_index[axis] = axis_index;
                    axis_neighbor = Some(neighbor_axis_index);
                }

                axis_neighbors[i] = axis_neighbor;
            }

            let axis_neighbors: (Option<[usize; N]>, Option<[usize; N]>) = axis_neighbors.into();
            neighbors[axis] = axis_neighbors;
        }

        neighbors
    }
}

impl<'a> Pixel for Wave<'a, types::Pixel, 2> {
    fn dims(&self) -> (usize, usize) {
        // the width and height of a wave can differ
        let (width, height) = self.shape.into_pattern();
        let tile_size = self.tile_size;

        (width * tile_size, height * tile_size)
    }

    fn pixels(&self) -> ndarray::Array2<types::Pixel> {
        let (width, height) = self.dims();
        let tile_size = self.tile_size;

        let mut pixels: Array2<types::Pixel> =
            Array2::from_shape_fn((width, height), |_| {
                [0; 3].into()
            });

        for ((i, j), wavetile) in self.wave.indexed_iter() {
            let wt_pixels = wavetile.pixels();

            for ((k, l), pixel) in wt_pixels.indexed_iter() {
                let (x, y) = (i * tile_size + k, j * tile_size + l);
                pixels[[x, y]] = *pixel;
            }
        }

        pixels
    }
}

// use default implementation
impl<'a> SdlTexture for Wave<'a, types::Pixel, 2> {}

impl<'a> crate::out::img::Image for Wave<'a, types::Pixel, 2> {}

impl<'a, T, const N: usize> Debug for Wave<'a, T, N>
where
    T: Hash + Sync + Send + std::fmt::Debug,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.wave)
    }
}
