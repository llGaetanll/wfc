use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::mpsc::Receiver;

use ndarray::{Array2, Dim, Dimension, NdIndex, SliceArg, SliceInfo, SliceInfoElem};

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
    possible_tiles: Vec<&'a Tile<'a, T, N>>,
    filtered_tiles: Vec<Vec<&'a Tile<'a, T, N>>>,

    /// list of memoized hashes derived from `possible_tiles`
    /// TODO: might not need to be public
    pub hashes: [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N],

    /// computed as the number of valid tiles
    pub entropy: usize,
    pub shape: usize,
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
     * TODO: assert that all tiles are the same shape
     */
    pub fn new(tiles: Vec<&'a Tile<'a, T, N>>) -> Self {
        let hashes = Self::derive_hashes(&tiles);
        let entropy = tiles.len();

        let shape = tiles[0].shape();

        WaveTile {
            possible_tiles: tiles,
            filtered_tiles: Vec::new(),
            hashes,
            entropy,
            shape,
        }
    }

    /// Given a list of `tile`s that this WaveTile can be, this precomputes the list of valid
    /// hashes for each of its borders. This is used to speed up the wave propagation algorithm.
    fn derive_hashes(
        tiles: &Vec<&'a Tile<'a, T, N>>,
    ) -> [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N] {
        let mut hashes: [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N] =
            vec![(HashSet::new(), HashSet::new()); N]
                .try_into()
                .unwrap();

        for tile in tiles.iter() {
            for (j, axis_hash) in tile.hashes.iter().enumerate() {
                let (h1, h2) = axis_hash.clone();
                let (hs1, hs2) = &mut hashes[j];

                hs1.insert(h1);
                hs2.insert(h2);
            }
        }

        hashes
    }

    /// Collapses a `WaveTile` to one of its possible tiles, at random.
    pub fn collapse(&mut self) -> Option<()> {
        if self.entropy < 1 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.entropy);

        let valid_tile = self.possible_tiles.get(idx).unwrap().to_owned();
        let invalid_tiles = self
            .possible_tiles
            .drain(..)
            .enumerate()
            .filter_map(|(i, tile)| if i != idx { Some(tile) } else { None })
            .collect();

        self.possible_tiles = vec![valid_tile];
        self.filtered_tiles.push(invalid_tiles);

        self.hashes = Self::derive_hashes(&self.possible_tiles);
        self.entropy = 1;

        Some(())
    }

    /// Update the `possible_tiles` of the current `WaveTile` given a list of neighbors.
    ///
    /// returns whether the wavetile has changed
    pub fn update(
        &mut self,
        neighbor_hashes: [(Option<HashSet<BoundaryHash>>, Option<HashSet<BoundaryHash>>); N],
    ) -> bool {
        let same_neighbors = neighbor_hashes
            .iter()
            .all(|(left, right)| left.is_none() && right.is_none());

        if self.entropy < 2 {
            return true;
        }

        if same_neighbors {
            return true;
        }

        let hashes: [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N] =
            vec![(HashSet::new(), HashSet::new()); N]
                .try_into()
                .unwrap();

        // owned copy of the self.hashes
        let hashes = std::mem::replace(&mut self.hashes, hashes);

        let new_hashes: [(HashSet<BoundaryHash>, HashSet<BoundaryHash>); N] = hashes
            .into_iter()
            .zip(neighbor_hashes.into_iter())
            .map(
                |((wavetile_left, wavetile_right), (neighbor_right, neighbor_left))| {
                    let left = match neighbor_right {
                        Some(hashes) => wavetile_left
                            .intersection(&hashes)
                            .map(|hash| hash.to_owned())
                            .collect::<HashSet<BoundaryHash>>(),
                        None => wavetile_left,
                    };

                    let right = match neighbor_left {
                        Some(hashes) => wavetile_right
                            .intersection(&hashes)
                            .map(|hash| hash.to_owned())
                            .collect::<HashSet<BoundaryHash>>(),
                        None => wavetile_right,
                    };

                    (left, right)
                },
            )
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // 2. for each tile, iterate over each axis, if any of its hashes are NOT in the
        //    possible_hashes, filter out the tile.
        let (valid_tiles, invalid_tiles): (Vec<_>, Vec<_>) = self
            .possible_tiles
            .drain(..) // allows us to effectively take ownership of the vector
            .partition(|&tile| {
                tile.hashes.iter().zip(new_hashes.iter()).all(
                    |((tile_left, tile_right), (wavetile_left, wavetile_right))| {
                        wavetile_left.contains(tile_left) && wavetile_right.contains(tile_right)
                    },
                )
            });

        let old_entropy = self.entropy;
        let new_entropy = valid_tiles.len();

        self.entropy = new_entropy;
        self.hashes = Self::derive_hashes(&valid_tiles);
        self.possible_tiles = valid_tiles;
        self.filtered_tiles.push(invalid_tiles);

        old_entropy != new_entropy;

        true
    }
}

impl<'a> Pixelizable for WaveTile<'a, Pixel, 2> {
    fn pixels(&self) -> Array2<Pixel> {
        // notice that a single number represents the size of the tile, no
        // matter the dimension. This is because it is enforced that all axes of
        // the tile be the same size.
        let size = self.shape;

        let valid_tiles = self.possible_tiles.iter();
        let num_tiles = valid_tiles.clone().count();

        let wavetile_pixels = Array2::from_shape_vec(
            (size, size),
            valid_tiles
                .fold(
                    {
                        let acc: Vec<[f64; 3]> = vec![[0., 0., 0.]; size * size];
                        acc
                    },
                    |acc, tile| {
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
