use std::borrow::Borrow;
use std::cell::{Ref, RefCell};
use std::cmp::min_by_key;
use std::collections::HashSet;
use std::fmt::Debug;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::Receiver;
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread;
use std::time::SystemTime;

use ndarray::{s, ArcArray, Array, Dim, Dimension, NdIndex, SliceArg, SliceInfo, SliceInfoElem};

use rayon::prelude::*;

use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use sdl2::render::Texture as SdlTexture;

use crate::wfc::types::BoundaryHash;

use super::sample::{self, Sample};
use super::tile::{DimN, Tile};
use super::traits::Hashable;
use super::traits::SdlTexturable;
use super::types::Pixel;
use super::wavetile::WaveTile;

/// A `Wave` is a `D` dimensional array of `WaveTile`s.
///
/// `D` is the dimension of the wave, as well as the dimension of each element of each `WaveTile`.
/// `T` is the type of the element of the wave. All `WaveTile`s for a `Wave` hold the same type.
pub struct Wave<'a, T, const N: usize>
where
    T: Hashable + Sync + std::fmt::Debug,
    Dim<[usize; N]>: Dimension, // ensures that [usize; N] is a Dimension implemented by ndarray
    [usize; N]: NdIndex<Dim<[usize; N]>>, // ensures that any [usize; N] is a valid index into the nd array

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    pub wave: ArcArray<RwLock<WaveTile<'a, T, N>>, Dim<[usize; N]>>,

    // used to quickly compute diffs on wavetiles that have changed
    diffs: ArcArray<RwLock<bool>, DimN<N>>,

    shape: DimN<N>,
}

impl<'a, T, const N: usize> Wave<'a, T, N>
where
    T: Hashable + Sync + Send + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
    [usize; N]: NdIndex<Dim<[usize; N]>>,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    fn new(tiles: Vec<Tile<'a, T, N>>, shape: DimN<N>) -> Result<Self, String> {
        // tiles are reference counted
        let tiles = tiles
            .into_iter()
            .map(|tile| Arc::new(tile))
            .collect::<Vec<_>>();

        let wave = ArcArray::from_shape_fn(shape, |_| RwLock::new(WaveTile::new(&tiles)));
        let diffs = ArcArray::from_shape_fn(shape, |_| RwLock::new(false));

        Ok(Wave { wave, diffs, shape })
    }

    fn min_entropy(&self) -> ([usize; N], &RwLock<WaveTile<'a, T, N>>) {
        let strides = self.wave.strides();

        let res = self
            .wave
            .iter()
            .enumerate()
            .filter(|&(_, wavetile)| {
                let wavetile = wavetile.read().expect("thread hung");

                // want non-collapsed tiles
                let non_collapsed = wavetile.entropy > 1;

                non_collapsed
            })
            .map(|(flat_index, wavetile)| {
                let nd_index = Self::compute_nd_index(flat_index, strides);

                (nd_index, wavetile)
            })
            .min_by_key(|&(_, wavetile)| {
                let wavetile = wavetile.read().expect("thread hung");

                let entropy = wavetile.entropy;

                entropy
            })
            .unwrap();

        res
    }

    fn max_entropy(&self) -> ([usize; N], &RwLock<WaveTile<'a, T, N>>) {
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
                let wavetile = wavetile.read().expect("thread hung");

                let entropy = wavetile.entropy;

                entropy
            })
            .unwrap();

        res
    }

    /// Collapses the wave
    pub fn collapse(&mut self, starting_index: Option<[usize; N]>) {
        // scope this in order to avoid hanging
        let max_entropy = {
            let (_, wavetile) = self.max_entropy();
            let wavetile = wavetile.read().expect("thread hung");
            wavetile.entropy
        };

        // if the wave is fully collapsed
        if max_entropy < 2 {
            return;
        };

        // get starting index
        let (index, wavetile) = match starting_index {
            Some(index) => (index, &self.wave[index]),
            None => self.min_entropy(),
        };

        // collapse the starting tile
        {
            let mut wavetile = wavetile
                .write()
                .expect("thread hung upon wavetile write attempt");

            wavetile.collapse();

            let mut diff = self.diffs[index].write().unwrap();
            *diff = true;
        }

        self.propagate(&index);

        // handle the rest of the wave
        loop {
            // reset diffs
            self.diffs = ArcArray::from_shape_fn(self.shape, |_| RwLock::new(false));

            let (_, wavetile) = self.max_entropy();
            let max_entropy = {
                let wavetile = wavetile
                    .read()
                    .expect("thread hung upon wavetile read attempt");
                wavetile.entropy
            };

            if max_entropy < 2 {
                break;
            };

            let (index, wavetile) = self.min_entropy();
            {
                let mut wavetile = wavetile
                    .write()
                    .expect("thread hung upon wavetile write attempt");

                wavetile.collapse();

                let mut diff = self.diffs[index]
                    .write()
                    .expect("thread hung upon diff write attempt");
                *diff = true;
            }

            self.propagate(&index);
        }
    }

    /// Propagates the wave from an index `start`
    /// TODO:
    ///  - stop propagation if neighboring tiles haven't updated
    ///  - currently, a wavetile looks at all of its neighbors, but it only needs to look at a
    ///  subset of those: the ones closer to the center of the wave.
    pub fn propagate(&self, start: &[usize; N]) {
        let t0 = SystemTime::now();
        let mut total_skips = Arc::new(Mutex::new(0));

        let index_groups = self.get_index_groups(start);

        for index_group in index_groups.into_iter() {
            // all tiles in an index group can be performed at the same time
            index_group.par_iter().for_each(|&index| {
                let total_skips = total_skips.clone();

                // note that if a `WaveTile` is on the edge of the wave, it might not have all its
                // neighbors. This also depends on the wrapping logic chosen
                let neighbors = self.compute_neighbors(&index);
                let mut wavetile_neighbors: [(
                    Option<HashSet<BoundaryHash>>,
                    Option<HashSet<BoundaryHash>>,
                ); N] = vec![(None, None); N].try_into().unwrap();

                let mut num_skips = 0;

                // from the neighbor indices, we construct a list of the neighbors hashes
                for (axis_index, index_pair) in neighbors.into_iter().enumerate() {
                    // NOTE: the right hashset comes from the left tile because of how we compare
                    // borders
                    let (left_index, right_index) = index_pair;

                    let right_hashset = match left_index {
                        Some(neighbor_index) => {
                            let diff = self.diffs[neighbor_index]
                                .read()
                                .expect("thread hung while attempting to read diff");

                            // check if the neighbor has changed
                            match *diff {
                                true => {
                                    let wavetile = self.wave[neighbor_index]
                                        .read()
                                        .expect("thread hung while attempting to read wavetile");

                                    let hashes = wavetile.hashes.get(axis_index).unwrap();
                                    let hashes = hashes.1.clone();

                                    Some(hashes)
                                }
                                false => {
                                    num_skips += 1;
                                    None
                                }
                            }
                        }
                        None => {
                            num_skips += 1;

                            None
                        }
                    };

                    let left_hashset = match right_index {
                        Some(neighbor_index) => {
                            let diff = self.diffs[neighbor_index]
                                .read()
                                .expect("thread hung while attempting to read diff");

                            match *diff {
                                true => {
                                    let wavetile = self.wave[neighbor_index]
                                        .read()
                                        .expect("thread hung while attempting to read wavetile");

                                    let hashes = wavetile.hashes.get(axis_index).unwrap();
                                    let hashes = hashes.0.clone();

                                    Some(hashes)
                                }
                                false => {
                                    num_skips += 1;
                                    None
                                }
                            }
                        }
                        None => {
                            num_skips += 1;
                            None
                        }
                    };

                    wavetile_neighbors[axis_index] = (right_hashset, left_hashset);
                }

                let mut total_skips = total_skips.lock().unwrap();
                *total_skips += num_skips;

                // TODO: catch if a wavetile has no more options at this stage instead of on
                // collapse?
                let lock = &self.wave[index];
                let mut wavetile = lock.write().expect("thread hung");
                let wavetile_changed = wavetile.update(wavetile_neighbors);

                // if statement so that we don't read unless we have to
                if wavetile_changed {
                    let mut diff = self.diffs[index].write().expect("thread hung");
                    *diff = true;
                }
            });
        }

        let t1 = SystemTime::now();
        let total_skips = total_skips.lock().unwrap();

        println!(
            "Propagate in {:?}. Total skips: {}",
            t1.duration_since(t0).unwrap(),
            *total_skips
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

pub fn from_sample<'a, T, const N: usize>(
    sample: &'a Sample<T, N>,
    shape: DimN<N>,
) -> Result<Wave<'a, T, N>, String>
where
    T: Hashable + Sync + Send + std::fmt::Debug,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<DimN<N>>,
{
    let tiles = sample.tiles(); // sample.window(window_size);

    let n = tiles.len();
    println!("{n} tiles");

    // create a wave of tiles of the appropriate shape from our sample image
    Wave::new(tiles, shape)
}

impl<'a> Wave<'a, Pixel, 2> {
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
            .map(|wavetile| {
                let wavetile = wavetile.read().expect("thread hung");

                wavetile.texture(&texture_creator)
            })
            .collect::<Vec<Result<SdlTexture, String>>>();

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
    T: Hashable + Sync + Send + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
    [usize; N]: NdIndex<Dim<[usize; N]>>,

    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.wave)
    }
}
