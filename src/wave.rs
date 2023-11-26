use std::fmt::Debug;
use std::hash::Hash;
use std::time::SystemTime;

use log::debug;
use ndarray::Array;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::NdIndex;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use sdl2::render::Texture;

use bit_set::BitSet;

use crate::tile::Tile;
use crate::tileset::TileSet;

use super::traits::SdlTexture;
use super::traits::Pixel;
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
    Dim<[usize; N]>: Dimension, // ensures that [usize; N] is a Dimension implemented by ndarray
    [usize; N]: NdIndex<Dim<[usize; N]>>, // ensures that any [usize; N] is a valid index into the nd array

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    pub wave: Array<WaveTile<'a, T, N>, Dim<[usize; N]>>,

    // used to quickly compute diffs on wavetiles that have changed
    pub diffs: Array<bool, DimN<N>>,

    pub shape: DimN<N>,
}

impl<'a, T, const N: usize> Wave<'a, T, N>
where
    T: Hash + Sync + Send + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
    [usize; N]: NdIndex<Dim<[usize; N]>>,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    pub fn new(shape: DimN<N>, tile_set: &'a TileSet<'a, T, N>) -> Self {
        let tiles = &tile_set.tiles;

        let tile_refs: Vec<&'a Tile<'a, T, N>> = tiles.iter().map(|tile| tile).collect();
        let tile_hashes: Vec<&[[BitSet; 2]; N]> = tiles.iter().map(|tile| &tile.hashes).collect();

        let num_hashes = tile_set.hashes.len();
        let wavetile_hashes = Self::merge_tile_bitsets(tile_hashes, num_hashes);

        Wave {
            wave: Array::from_shape_fn(shape, |_| {
                WaveTile::new(tile_refs.clone(), wavetile_hashes.clone(), num_hashes)
            }),
            diffs: Array::from_shape_fn(shape, |_| false),
            shape,
        }
    }

    /// Collapses the wave
    pub fn collapse(&mut self, starting_index: Option<[usize; N]>) {
        // scope this in order to avoid hanging
        let max_entropy = {
            let (_, wavetile) = self.max_entropy();
            wavetile.entropy
        };

        // if the wave is fully collapsed
        if max_entropy < 2 {
            return;
        };

        // get starting index
        let index = match starting_index {
            Some(index) => index,
            None => {
                let (index, _) = self.min_entropy();
                index
            }
        };

        // collapse the starting tile
        {
            debug!("collapsing {:?}", index);
            let wavetile = &mut self.wave[index];
            wavetile.collapse();
            self.diffs[index] = true;
        }

        self.propagate(&index);

        // handle the rest of the wave
        loop {
            let max_entropy = {
                // TODO: compute max & min entropy at propagate?
                let (_, wavetile) = self.max_entropy();
                wavetile.entropy
            };

            if max_entropy < 2 {
                break;
            };

            {
                let (index, _) = self.min_entropy();
                debug!("Collapsing {:?}", index);
                let wavetile = &mut self.wave[index];
                wavetile.collapse();
                self.diffs[index] = true;
            }

            self.propagate(&index);
        }
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

    fn min_entropy(&self) -> ([usize; N], &WaveTile<'a, T, N>) {
        let strides = self.wave.strides();

        let res = self
            .wave
            .iter()
            .enumerate()
            .filter(|&(_, wavetile)| {
                // want non-collapsed tiles
                let non_collapsed = wavetile.entropy > 1;

                non_collapsed
            })
            .map(|(flat_index, wavetile)| {
                let nd_index = Self::compute_nd_index(flat_index, strides);

                (nd_index, wavetile)
            })
            .min_by_key(|&(_, wavetile)| {
                let entropy = wavetile.entropy;

                entropy
            })
            .unwrap();

        res
    }

    fn max_entropy(&self) -> ([usize; N], &WaveTile<'a, T, N>) {
        let strides = self.wave.strides();

        let res = self
            .wave
            .iter()
            .enumerate()
            .map(|(flat_index, wavetile)| {
                let nd_index = Self::compute_nd_index(flat_index, strides);

                (nd_index, wavetile)
            })
            .max_by_key(|&(_, wavetile)| {
                let entropy = wavetile.entropy;

                entropy
            })
            .unwrap();

        res
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

        for index_group in index_groups.into_iter() {
            for index in index_group.into_iter() {
                let neighbor_indices = self.compute_neighbors(&index);

                let skip = !neighbor_indices.iter().any(|(left, right)| {
                    left.map_or(false, |idx| self.diffs[idx])
                        || right.map_or(false, |idx| self.diffs[idx])
                });

                // if skip {
                //     continue;
                // }

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

                debug!("updating {:?}", index);

                // TODO: catch if a wavetile has no more options at this stage instead of on
                // collapse?
                self.diffs[index] = self.wave[index].update(neighbor_hashes);
            }
        }

        // reset diffs
        Array::fill(&mut self.diffs, false);

        let t1 = SystemTime::now();

        // println!(
        //     "Propagate in {:?}. Total skips: {}",
        //     t1.duration_since(t0).unwrap(),
        //     skips
        // );
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

/*
pub fn from_tiles<'a, T, const N: usize>(
    tiles: Vec<&'a Tile<'a, T, N>>,
    shape: DimN<N>,
) -> Result<Wave<'a, T, N>, String>
where
    T: Hash + Sync + Send + std::fmt::Debug,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<DimN<N>>,
{
    // let tiles: Vec<_> = sample.0.iter().map(|data| Tile::new(data.view())).collect();
    let n = tiles.len();

    println!("{n} tiles");

    // create a wave of tiles of the appropriate shape from our sample image
    Wave::new(shape, tiles)
}
*/

impl<'a> Wave<'a, types::Pixel, 2> {
    pub fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
        const TILE_SIZE: usize = 50;

        // get the width and height of the wave window output
        let [width, height]: [usize; 2] = self.wave.shape().try_into().unwrap();
        let (win_width, win_height) = (width * TILE_SIZE, height * TILE_SIZE);

        // init sdl
        let video_subsystem = sdl_context.video()?;
        let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;
        let window = video_subsystem
            .window(
                "Wave Function Collapse",
                win_width as u32,
                win_height as u32,
            )
            .position_centered()
            .build()
            .map_err(|e| e.to_string())?;

        let mut canvas = window
            .into_canvas()
            .software()
            .build()
            .map_err(|e| e.to_string())?;

        let texture_creator = canvas.texture_creator();

        // thread::spawn(|| for updates in rx {});

        // turn each tile into a texture to be loaded on the sdl canvas
        let tile_textures = self
            .wave
            .iter()
            .map(|wavetile| wavetile.texture(&texture_creator))
            .collect::<Vec<Result<Texture, String>>>();

        // load every texture into the canvas, side by side
        for (i, tile_texture) in tile_textures.iter().enumerate() {
            canvas.copy(
                tile_texture.as_ref().expect("texture error"),
                None,
                Rect::new(
                    (i as i32 % width as i32) * TILE_SIZE as i32,
                    (i as i32 / width as i32) * TILE_SIZE as i32,
                    TILE_SIZE as u32,
                    TILE_SIZE as u32,
                ),
            )?;
        }

        canvas.present();

        'mainloop: loop {
            for event in sdl_context.event_pump()?.poll_iter() {
                match event {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Option::Some(Keycode::Escape),
                        ..
                    } => break 'mainloop,
                    _ => {}
                }
            }

            // sleep 1s to not overwhelm system resources
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        Ok(())
    }
}

impl<'a, T, const N: usize> Debug for Wave<'a, T, N>
where
    T: Hash + Sync + Send + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
    [usize; N]: NdIndex<Dim<[usize; N]>>,

    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.wave)
    }
}

impl<'a> Wave<'a, types::Pixel, 2> {
    fn to_img(&mut self, index: usize)
    where
        [usize; 2]: NdIndex<DimN<2>>,
    {
        let wavetiles = &self.wave;

        let (width, height) = self.shape.into_pattern();
        let scaling: usize = 30;

        let mut imgbuf =
            image::ImageBuffer::new((width * scaling) as u32, (height * scaling) as u32);

        for (_, wavetile) in wavetiles.iter().enumerate() {
            let pixels = wavetile.pixels();

            // Iterate over the coordinates and pixels of the image
            for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
                let x = x as usize / scaling;
                let y = y as usize / scaling;

                let px = pixels.get([x, y]).unwrap().to_owned();
                *pixel = image::Rgb(px.into());
            }
        }

        imgbuf
            .save(format!("./assets/wave/wave-{index}.png"))
            .unwrap();
    }
}
