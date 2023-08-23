use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use ndarray::{s, Array, Dim, Dimension, Ix2, IxDyn, NdIndex, SliceArg, SliceInfo, SliceInfoElem};

use rayon::prelude::*;

use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use sdl2::render::Texture as SdlTexture;

use super::tile::Tile;
use super::traits::{Hashable, SdlTexturable, SdlView};
use super::types::Pixel;
use super::wavetile::WaveTile;

// TODO: use ndarray NdIndex<D> type if possible, this is a hack
type Index = Vec<usize>;

/// A `Wave` is a `D` dimensional array of `WaveTile`s.
///
/// `D` is the dimension of the wave, as well as the dimension of each element of each `WaveTile`.
/// `T` is the type of the element of the wave. All `WaveTile`s for a `Wave` hold the same type.
pub struct Wave<'a, T, D>
where
    D: Dimension + Sized,
    T: Hashable,
{
    pub wave: Array<WaveTile<'a, T, D>, D>,
}

impl<'a, T, D> Wave<'a, T, D>
where
    D: Dimension + Sized,
    T: Hashable,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, D, <D as Dimension>::Smaller>: SliceArg<D>,
{
    pub fn new(tiles: Vec<Rc<Tile<'a, T, D>>>, dim: D) -> Result<Self, String> {
        let wave = Array::from_shape_fn(dim, |_| WaveTile::new(&tiles).unwrap());

        Ok(Wave { wave })
    }

    pub fn test(&self, idx: Box<dyn NdIndex<D>>) {}

    // TODO: index type isn't great. Should be [usize; N] where N is ndim
    pub fn collapse(&self, start: Vec<usize>) {
        let strides = self.wave.strides();

        // compute an nd index from a given index and the local strides
        let compute_ndindex = |idx: usize| -> Index {
            let mut nd_index: Index = Vec::new();

            strides.iter().fold(idx, |idx_remain, &stride| {
                let stride = stride as usize;
                nd_index.push(idx_remain / stride);
                idx_remain % stride
            });

            nd_index
        };

        // compute manhattan distance between `start` and `index`
        let manhattan_dist = |index: &Index| -> usize {
            start
                .iter()
                .zip(index.iter())
                .map(|(&a, &b)| ((a as isize) - (b as isize)) as usize)
                .sum()
        };

        let max_manhattan_dist = self
            .wave
            .iter()
            .enumerate()
            .map(|(i, _)| compute_ndindex(i))
            .map(|nd_index| manhattan_dist(&nd_index))
            .max()
            .unwrap();

        let mut wavetile_groups: Vec<Vec<&WaveTile<'a, T, D>>> =
            vec![Vec::new(); max_manhattan_dist];

        self.wave
            .iter()
            .enumerate()
            .map(|(i, wavetile)| {
                let nd_index = compute_ndindex(i);
                (manhattan_dist(&nd_index), wavetile)
            })
            .for_each(|(dist, wavetile)| {
                wavetile_groups[dist].push(wavetile);
            });

        // &self.wave[IxDyn(&[1, 2])];

        for wavetile_group in wavetile_groups.into_iter() {
            // wavetile_group.into_par_iter().map(|idx| {});
        }
    }

    // TODO: index type isn't great. Should be [usize; N] where N is ndim
    fn neighbors<A>(array: &Array<A, D>, index: &Vec<usize>) -> Vec<Vec<usize>> {
        let mut neighbors: Vec<Vec<usize>> = Vec::new();

        let shape = array.shape();
        let ndim = array.ndim();

        for d in 1..ndim {
            let axis = d - 1;
            let (min_bd, max_bd) = (0, shape[axis]);

            for delta in [-1, 1].into_iter() {
                let axis_index = index[d] + (delta as usize);

                let range = min_bd..=max_bd;

                // TODO: enable spherical/kleinbottle wrapping via enum
                if range.contains(&axis_index) {
                    let mut neighbor_index = index.clone();
                    neighbor_index[d] = axis_index;

                    neighbors.push(neighbor_index);
                }
            }
        }

        neighbors
    }

    pub fn shape(&self) -> D {
        self.wave.raw_dim()
    }
}

pub fn from_tiles<'a, T, D>(tiles: Vec<Tile<'a, T, D>>, dim: D) -> Result<Wave<'a, T, D>, String>
where
    T: Hashable,
    D: Dimension + Sized,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, D, <D as Dimension>::Smaller>: SliceArg<D>,
{
    let tiles: Vec<Rc<_>> = tiles.into_iter().map(|tile| Rc::new(tile)).collect();

    Wave::new(tiles, dim)
}

impl<'a> SdlView for Wave<'a, Pixel, Ix2> {
    fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
        const TILE_SIZE: usize = 200;

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

        // turn each tile into a texture to be loaded on the sdl canvas
        let tile_textures = self
            .wave
            .iter()
            .map(|wavetile| wavetile.texture(&texture_creator))
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
