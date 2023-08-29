use std::borrow::Borrow;
use std::fmt::Debug;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::SystemTime;
use std::{cell::RefCell, collections::HashSet, rc::Rc};

use ndarray::{Array2, Dim, Dimension, Ix2, NdIndex, SliceArg, SliceInfo, SliceInfoElem};

use rand::seq::SliceRandom;
use rand::Rng;

use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{Texture, TextureCreator};
use sdl2::surface::Surface;
use sdl2::video::WindowContext;

use super::tile::Tile;
use super::traits::{Hashable, Pixelizable, SdlTexturable};
use super::types::{BoundaryHash, Pixel};

/// A `WaveTile` is a list of `Tile`s in superposition
/// `T` is the type of each element of the tile
/// `D` is the dimension of each tile
pub struct WaveTile<'a, T, const N: usize>
where
    T: Hashable,
    Dim<[usize; N]>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    /// The list of possible tiles that the WaveTile can be
    ///
    /// For each tile, we store a unsigned integer which is initialized as 0. If
    /// a tile is no longer possible, this number is incremented to 1. In every
    /// subsequent pass, if a number i > 0, it is again incremented. this allows
    /// us to reverse the operation.
    possible_tiles: Vec<(Arc<Tile<'a, T, N>>, usize)>,

    /// list of memoized hashes derived from `possible_tiles`
    /// TODO: might not need to be public
    pub hashes: [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N],

    /// computed as the number of valid tiles
    pub entropy: usize,

    /// The shape of the WaveTile
    shape: usize,
}

impl<'a, T, const N: usize> WaveTile<'a, T, N>
where
    T: Hashable + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    /***
     * Create a new WaveTile from a list of tiles
     */
    pub fn new(tiles: &Vec<Arc<Tile<'a, T, N>>>) -> Self {
        // the shape of the wavetile is defined by the shape of the first tile
        // if the remaining tiles are not of the same shape, we return an error
        let shape = tiles.get(0).unwrap().shape();

        // TODO: assert that all tiles are the same shape

        let tiles = tiles
            .iter()
            .map(|tile| (Arc::clone(tile), 0))
            .collect::<Vec<(Arc<Tile<T, N>>, usize)>>();

        let hashes = Self::derive_hashes(&tiles);
        let entropy = Self::compute_entropy(&tiles);

        WaveTile {
            possible_tiles: tiles,
            hashes,
            entropy,
            shape,
        }
    }

    fn compute_entropy(tiles: &Vec<(Arc<Tile<'a, T, N>>, usize)>) -> usize {
        tiles.iter().filter(|(_, n)| *n == 0).count()
    }

    /// Given a list of `tile`s that this WaveTile can be, this precomputes the list of valid
    /// hashes for each of its borders. This is used to speed up the wave propagation algorithm.
    fn derive_hashes(
        tiles: &Vec<(Arc<Tile<'a, T, N>>, usize)>,
    ) -> [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N] {
        let possible_tiles: Vec<&Arc<Tile<'a, T, N>>> = tiles
            .into_iter()
            .filter(|(_, i)| *i == 0)
            .map(|(tile, _)| tile)
            .collect();

        let mut hashes: Vec<(HashSet<BoundaryHash>, HashSet<BoundaryHash>)> =
            vec![(HashSet::new(), HashSet::new()); N];

        // TODO: probably painfully slow?
        for tile in possible_tiles.iter() {
            for (j, axis_hash) in tile.hashes.iter().enumerate() {
                let (h1, h2) = axis_hash;
                let (hs1, hs2) = &mut hashes[j];

                // NOTE: cloning a u64 is fast, but maybe we don't need to
                hs1.insert(h1.clone());
                hs2.insert(h2.clone());
            }
        }

        hashes.try_into().unwrap()
    }

    /// Collapses a `WaveTile` to one of its possible tiles, at random.
    pub fn collapse(&mut self) -> Result<(), ()> {
        let mut rng = rand::thread_rng();

        let available_indices: Vec<usize> = self
            .possible_tiles
            .iter()
            .enumerate()
            .filter(|(_, (_, n))| *n == 0)
            .map(|(i, _)| i)
            .collect();

        match available_indices.choose(&mut rng) {
            Some(&rand_idx) => {
                // collapse the tile list to a single tile
                self.possible_tiles
                    .iter_mut()
                    .enumerate()
                    .filter(|(i, _)| *i != rand_idx)
                    .for_each(|(_, &mut (_, ref mut count))| *count += 1);

                // update the hashes for the current tile
                // TODO: code kinda gross
                // TODO: it would be nice if the hashes were updated automatically whenever the
                // possible_tiles are changed.
                self.hashes = Self::derive_hashes(&self.possible_tiles);

                self.entropy = 1;

                Ok(())
            }
            None => {
                println!("No more options!");
                Err(())
            }
        }
    }

    /// Update the `possible_tiles` of the current `WaveTile` given a list of neighbors.
    pub fn update(
        &mut self,
        neighbor_hashes: [(Option<HashSet<BoundaryHash>>, Option<HashSet<BoundaryHash>>); N],
    ) {
        let mut possible_hashes: [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N] =
            vec![(HashSet::new(), HashSet::new()); N]
                .try_into()
                .unwrap();

        // 1. compute possible hashes of the wavetile for each axis
        for (axis_index, axis_hash_pair) in neighbor_hashes.into_iter().enumerate() {
            // TODO: zip instead
            let (self_left_hash, self_right_hash): &(HashSet<BoundaryHash>, HashSet<BoundaryHash>) =
                &self.hashes.get(axis_index).unwrap();

            // NOTE: PURPOSEFULLY REVERSED.
            //
            // This is because the wall that connects us to our LEFT neighbor is ITS RIGHT wall.
            let (neighbor_right_hashes, neighbor_left_hashes) = axis_hash_pair;

            // TODO: use references?
            let mut valid_hashes: (HashSet<BoundaryHash>, HashSet<BoundaryHash>) =
                (HashSet::default(), HashSet::default());

            valid_hashes.0 = match neighbor_right_hashes {
                Some(neighbor_right_hashes) => {
                    // TODO: temp fix until backtracking works
                    if neighbor_right_hashes.is_empty() {
                        self_left_hash.clone()
                    } else {
                        let intersection = self_left_hash
                            .intersection(&neighbor_right_hashes)
                            .map(|h| h.clone())
                            .collect::<HashSet<BoundaryHash>>();

                        intersection
                    }
                }
                None => self_left_hash.clone(), // TODO: expensive?
            };

            valid_hashes.1 = match neighbor_left_hashes {
                Some(neighbor_left_hashes) => {
                    // TODO: temp fix until backtracking works
                    if neighbor_left_hashes.is_empty() {
                        self_right_hash.clone()
                    } else {
                        let intersection = self_right_hash
                            .intersection(&neighbor_left_hashes)
                            .map(|h| h.clone())
                            .collect::<HashSet<BoundaryHash>>();

                        intersection
                    }
                }
                None => self_right_hash.clone(), // TODO: expensive?
            };

            possible_hashes[axis_index] = valid_hashes;
        }

        // 2. for each tile, iterate over each axis, if any of its hashes are NOT in the
        //    possible_hashes, filter out the tile.
        for (tile, count) in self.possible_tiles.iter_mut() {
            let mut invalid = false;
            for (axis_index, axis_hashes) in tile.hashes.iter().enumerate() {
                let (wavetile_left, wavetile_right) = &possible_hashes[axis_index];
                let (tile_left, tile_right) = axis_hashes;

                if (!wavetile_left.contains(tile_left)) || (!wavetile_right.contains(tile_right)) {
                    invalid = true;
                    break;
                }
            }

            if *count > 0 || invalid {
                *count += 1; // TODO: short circuit on count > 0 is an easy optimization
            }
        }

        // 3. Update the hashes
        self.hashes = Self::derive_hashes(&self.possible_tiles);

        // 4. Update the entropy
        self.entropy = Self::compute_entropy(&self.possible_tiles);
    }
}

impl<'a> Pixelizable for WaveTile<'a, Pixel, 2> {
    fn pixels(&self) -> Array2<Pixel> {
        // notice that a single number represents the size of the tile, no
        // matter the dimension. This is because it is enforced that all axes of
        // the tile be the same size.
        let size = self.shape;

        let valid_tiles = self.possible_tiles.iter().filter(|(_, n)| *n == 0);
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

impl<'a> SdlTexturable for WaveTile<'a, Pixel, 2> {
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

pub struct TileUpdate<'a, T, const N: usize>
where
    T: Hashable + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
    [usize; N]: NdIndex<Dim<[usize; N]>>,
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    index: [usize; N],
    wavetile: WaveTile<'a, T, N>,
}

impl<'a> WaveTile<'a, Pixel, 2> {
    fn show<Upd: Send + Sync>(
        &self,
        sdl_context: &sdl2::Sdl,
        rx: Receiver<Upd>,
    ) -> Result<(), String> {
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

impl<'a, T, const N: usize> Debug for WaveTile<'a, T, N>
where
    T: Hashable + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.possible_tiles)
    }
}
