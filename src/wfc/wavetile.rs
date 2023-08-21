use ndarray::{Array2, Dimension, Ix2, SliceArg, SliceInfo, SliceInfoElem};
use rand::Rng;
use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use std::{cell::RefCell, collections::HashSet, rc::Rc};

use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{Texture, TextureCreator};
use sdl2::surface::Surface;
use sdl2::video::WindowContext;

use super::tile::Tile;
use super::traits::{Hashable, Pixelizable, SdlTexturable, SdlView};
use super::types::{BoundaryHash, Pixel};

/// A `WaveTile` is a list of `Tile`s in superposition
/// `T` is the type of each element of the tile
/// `D` is the dimension of each tile
pub struct WaveTile<'a, T, D>
where
    T: Hashable,
{
    /// The list of possible tiles that the WaveTile can be
    ///
    /// For each tile, we store a unsigned integer which is initialized as 0. If
    /// a tile is no longer possible, this number is incremented to 1. In every
    /// subsequent pass, if a number i > 0, it is again incremented. this allows
    /// us to reverse the operation.
    possible_tiles: RefCell<Vec<(Rc<Tile<'a, T, D>>, usize)>>,

    /// list of memoized hashes derived from `possible_tiles`
    hashes: Vec<(HashSet<BoundaryHash>, HashSet<BoundaryHash>)>,

    /// computed as the number of valid tiles
    entropy: usize,

    /// The shape of the WaveTile
    shape: usize,
}

impl<'a, T, D> WaveTile<'a, T, D>
where
    D: Dimension,
    T: Hashable,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, D, <D as Dimension>::Smaller>: SliceArg<D>,
{
    /***
     * Create a new WaveTile from a list of tiles
     */
    pub fn new(tiles: &Vec<Rc<Tile<'a, T, D>>>) -> Result<Self, String> {
        // the shape of the wavetile is defined by the shape of the first tile
        // if the remaining tiles are not of the same shape, we return an error
        let shape = tiles.get(0).unwrap().shape();

        let tiles = tiles
            .iter()
            .map(|tile| (Rc::clone(tile), 0))
            .collect::<Vec<(Rc<Tile<T, D>>, usize)>>();

        // TODO: assert that all tiles are the same shape
        Ok(WaveTile {
            possible_tiles: RefCell::new(tiles),

            hashes: Self::derive_hashes(&tiles),

            entropy: Self::compute_entropy(&tiles),

            // this is the shape of the WaveTile
            shape,
        })
    }

    fn compute_entropy(tiles: &Vec<(Rc<Tile<'a, T, D>>, usize)>) -> usize {
        tiles.iter().filter(|(_, n)| *n == 0).count()
    }

    fn derive_hashes(
        tiles: &Vec<Rc<Tile<'a, T, D>>>,
    ) -> Vec<(HashSet<BoundaryHash>, HashSet<BoundaryHash>)> {
        // number of dimensions
        // FIXME: this is a hack
        let n = tiles.get(0).unwrap().hashes.len();

        let mut hashes: Vec<(HashSet<BoundaryHash>, HashSet<BoundaryHash>)> =
            vec![(HashSet::new(), HashSet::new()); n];

        // TODO: probably painfully slow
        for tile in tiles.iter() {
            for (j, axis_hash) in tile.hashes.iter().enumerate() {
                let (h1, h2) = axis_hash;
                let (hs1, hs2) = &mut hashes[j];

                // NOTE: cloning a u64 is fast, but maybe we don't need to
                hs1.insert(h1.clone());
                hs2.insert(h2.clone());
            }
        }

        hashes
    }

    // FIXME: logic incorrect
    pub fn collapse(&self) {
        let n = self.possible_tiles.borrow().len();
        let rand_idx = rand::thread_rng().gen_range(0..n);

        // collapse the tile list to a single tile
        self.possible_tiles
            .borrow_mut()
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| *i != rand_idx)
            .for_each(|(_, &mut (_, ref mut count))| *count += 1);
    }

    /***
     * Update the `possible_tiles` of the current `WaveTile` given a list of neighbors.
     */
    pub fn update(&self, neighbors: Vec<(&WaveTile<'a, T, D>, &WaveTile<'a, T, D>)>) {
        // number of dimensions
        // FIXME: this is a hack
        let n = neighbors.len();

        let mut possible_hashes: Vec<(HashSet<BoundaryHash>, HashSet<BoundaryHash>)> =
            vec![(HashSet::new(), HashSet::new()); n];

        // 1. compute possible hashes of the wavetile for each axis
        for (i, (wt1, wt2)) in neighbors.iter().enumerate() {
            let my_hashes: &(HashSet<BoundaryHash>, HashSet<BoundaryHash>) = &self.hashes[i];

            let neighbor_hashes_1: &(HashSet<BoundaryHash>, HashSet<BoundaryHash>) = &wt1.hashes[i];
            let neighbor_hashes_2: &(HashSet<BoundaryHash>, HashSet<BoundaryHash>) = &wt2.hashes[i];

            let left_intersection: HashSet<BoundaryHash> = my_hashes
                .0
                .intersection(&neighbor_hashes_1.1)
                .map(|hash| hash.clone())
                .collect();

            let right_intersection: HashSet<BoundaryHash> = my_hashes
                .1
                .intersection(&neighbor_hashes_2.0)
                .map(|hash| hash.clone())
                .collect();

            possible_hashes[i].0.extend(&left_intersection);
            possible_hashes[i].1.extend(&right_intersection);
        }

        // 2. for each tile, iterate over each axis, if any of its hashes are NOT in the
        //    possible_hashes, filter out the tile.
        for (tile, count) in self.possible_tiles.borrow_mut().iter_mut() {
            let invalid = tile
                .hashes
                .iter()
                .enumerate()
                .any(|(i, (left_hash, right_hash))| {
                    let (left, right) = &possible_hashes[i];

                    return !left.contains(left_hash) || !right.contains(right_hash);
                });

            if *count > 0 || invalid {
                *count += 1;
            }
        }
    }
}

impl<'a> Pixelizable for WaveTile<'a, Pixel, Ix2> {
    fn pixels(&self) -> Array2<Pixel> {
        // notice that a single number represents the size of the tile, no
        // matter the dimension. This is because it is enforced that all axes of
        // the tile be the same size.
        let size = self.shape;

        let possible_tiles = self.possible_tiles.borrow();
        let valid_tiles = possible_tiles.iter().filter(|(_, n)| *n == 0);
        let num_tiles = valid_tiles.clone().count();

        let wavetile_pixels = Array2::from_shape_vec(
            (size, size),
            valid_tiles
                .fold(
                    {
                        let acc: Vec<[f64; 3]> = vec![[0., 0., 0.]; size * size];
                        acc
                    },
                    |acc, (tile, _)| {
                        acc.iter()
                            .zip(tile.pixels().into_iter())
                            .map(|(acc_px, tile_px)| {
                                acc_px
                                    .iter()
                                    .zip(tile_px.into_iter())
                                    .map(|(acc_chan, tile_chan)| {
                                        acc_chan + ((tile_chan as f64) / (num_tiles as f64))
                                    })
                                    .collect::<Vec<f64>>()
                                    .try_into()
                                    .unwrap()
                            })
                            .collect()
                    },
                )
                .into_iter()
                .map(|px| {
                    px.into_iter()
                        .map(|c| c as u8)
                        .collect::<Vec<u8>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<[u8; 3]>>(),
        )
        .unwrap();

        wavetile_pixels
    }
}

impl<'a> SdlTexturable for WaveTile<'a, Pixel, Ix2> {
    /***
     * Create a texture object for the current wavetile
     */
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String> {
        let size = self.shape;

        // we need to flatten the list of pixels to turn it into a texture
        let mut flat_pixels: Vec<u8> = self
            .pixels()
            .into_iter()
            .flat_map(|pixel| pixel.into_iter().map(|p| p))
            .collect();

        // create a surface from the flat pixels vector
        let surface = Surface::from_data(
            &mut flat_pixels,
            size as u32,     // width of the texture
            size as u32,     // height of the texture
            size as u32 * 3, // this is the number of channels for each pixel
            PixelFormatEnum::RGB24,
        )
        .map_err(|e| e.to_string())?;

        // create a texture from the surface
        texture_creator
            .create_texture_from_surface(&surface)
            .map_err(|e| e.to_string())
    }
}

impl<'a> SdlView for WaveTile<'a, Pixel, Ix2> {
    fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
        const WIN_SIZE: u32 = 100;

        let video_subsystem = sdl_context.video()?;
        let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;
        let window = video_subsystem
            .window("WaveTile View", WIN_SIZE, WIN_SIZE)
            .position_centered()
            .build()
            .map_err(|e| e.to_string())?;

        let mut canvas = window
            .into_canvas()
            .software()
            .build()
            .map_err(|e| e.to_string())?;

        let texture_creator = canvas.texture_creator();

        let texture = &self.texture(&texture_creator)?;

        canvas.copy(texture, None, Rect::new(0, 0, WIN_SIZE, WIN_SIZE))?;

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

            // sleep to not overwhelm system resources
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        Ok(())
    }
}
