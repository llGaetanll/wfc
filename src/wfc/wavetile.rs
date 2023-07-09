use ndarray::{Array2, ArrayView2, Dimension, Ix2, SliceArg, SliceInfo, SliceInfoElem};
use std::cmp::max;
use std::rc::Rc;
use std::ptr;

use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{Texture as SdlTexture, TextureCreator};
use sdl2::surface::Surface;
use sdl2::video::WindowContext;

use super::tile::Tile;
use super::traits::{Hashable, Pixelize, SdlView};
use super::types::Pixel;

use rand::seq::SliceRandom;

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
    possible_tiles: Vec<(Rc<Tile<'a, T, D>>, usize)>,

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

    // issue here is that the reference to the tiles vector is shorter than 'a
    // that shouldn't be an issue though since 'a is inside a Reference Counter
    // but it is, since lifetimes must be known at compile time
    pub fn new(tiles: &Vec<Rc<Tile<'a, T, D>>>) -> Result<Self, String> {
        // the shape of the wavetile is defined by the shape of the first tile
        // if the remaining tiles are not of the same shape, we return an error
        let shape = tiles.get(0).unwrap().shape();

        // TODO: assert that all tiles are the same shape
        Ok(WaveTile {
            possible_tiles: tiles
                .iter()
                .map(|tile| (Rc::clone(tile), 0))
                .collect::<Vec<(Rc<Tile<'a, T, D>>, usize)>>(),

            // this is the shape of the WaveTile
            shape,
        })
    }

    // pub fn collapse(&mut self) {
    //     let possible_tiles: Vec<&Rc<Tile<'a, T, D>>> = self.possible_tiles.iter().map(|(tile, _)| tile).collect();
    //
    //     let mut rng = rand::thread_rng();
    //
    //     // pick a random value in the array
    //     // since the array is never empty, we can unwrap safely
    //     let random_tile: &Rc<Tile<'a, T, D>> = possible_tiles.choose(&mut rng).unwrap();
    //
    //     // TODO: to collapse the tile is to set the tile list to only contain the chosen tile
    //     // self.possible_tiles = vec![tile.clone()];
    //     self.possible_tiles.iter_mut().filter(|(tile, _)| !ptr::eq(tile, random_tile)).for_each(|&mut (_, ref mut count)| *count += 1);
    // }
}

impl<'a> Pixelize<'a> for WaveTile<'a, Pixel, Ix2> {
    fn pixels(&'a self) -> ArrayView2<'a, Pixel> {
        // notice that a single number represents the size of the tile, no
        // matter the dimension. This is because it is enforced that all axes of
        // the tile be the same size.
        let size = self.shape;

        // a list of tiles as vectors of pixels
        let pixel_tiles = self
            .possible_tiles
            .iter()
            .filter(|(_, n)| *n == 0) // we only consider the tiles that are valid in this WaveTile
            .map(|(tile, _)| tile.pixels());

        /*
        let pixels = pixel_tiles
            // we create a cumulative sum of every image, pixel channel-wise
            .fold(
                {
                    // our accumulator is a vector of u64 to avoid overflows
                    // let acc = Array2::zeros(ndarray::Dim(shape)).map(|_| [0; 3]);
                    let acc: Vec<[u64; 3]> = vec![[0; 3]; size * size];
                    acc
                },
                |acc, img| {
                    // for each image, we add pixel channel-wise to the accumulator
                    acc.iter()
                        .zip(img.iter())
                        // add the accumulator and the current image channel-wise
                        .map(|(p1, p2)| -> [u64; 3] {
                            // since p2 comes from the original image, each of its
                            // pixels is still of type [u8; 3], so we convert it
                            [
                                p1[0] + p2[0] as u64,
                                p1[1] + p2[1] as u64,
                                p1[2] + p2[2] as u64,
                            ]
                        })
                        // collect the new image, this the new iterator
                        .collect::<Vec<[u64; 3]>>()
                },
            )
            .iter()
            // for every pixel in the new image sum, we divide by the number of tiles to get the average
            .map(|[r, g, b]| {
                // number of possible tiles in the current WaveTile
                // since we don't want to divide by zero, it's possible for
                // WaveTiles to have no options, we divide by 1 instead
                let n: u64 = max(self.possible_tiles.len() as u64, 1);

                [(*r / n) as u8, (*g / n) as u8, (*b / n) as u8]
            })
            .collect::<Vec<[u8; 3]>>();
        */

        todo!()
    }
}

/*
impl Display for WaveTile<'_, Pixel, Ix2> {
    fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
        todo!()
    }

    fn texture<'a>(
        &self,
        texture_creator: &'a TextureCreator<WindowContext>,
    ) -> Result<SdlTexture<'a>, String> {
        // the size of a wavetile is the size of each of its tiles (which are all the same size, and are all square, cubic, etc...)
        // let size = ;

        // a list of tiles as vectors of pixels
        let pixel_tiles = self
            .possible_tiles
            .iter()
            .filter(|(tile, n)| *n == 0) // we only consider the tiles that are valid in this wavetile
            .map(|(tile, _)| tile.get_pixels());

        let pixels = pixel_tiles
            // we create a cumulative sum of every image, pixel channel-wise
            .fold(
                {
                    // our accumulator is a vector of u64 to avoid overflows
                    let acc: Vec<[u64; 3]> = vec![[0; 3]; size * size];
                    acc
                },
                |acc, img| {
                    // for each image, we add pixel channel-wise to the accumulator
                    acc.iter()
                        .zip(img.iter())
                        // add the accumulator and the current image channel-wise
                        .map(|(p1, p2)| -> [u64; 3] {
                            // since p2 comes from the original image, each of its
                            // pixels is still of type [u8; 3], so we convert it
                            [
                                p1[0] + p2[0] as u64,
                                p1[1] + p2[1] as u64,
                                p1[2] + p2[2] as u64,
                            ]
                        })
                        // collect the new image, this the new iterator
                        .collect::<Vec<[u64; 3]>>()
                },
            )
            .iter()
            // for every pixel in the new image sum, we divide by the number of tiles to get the average
            .map(|[r, g, b]| {
                // number of possible tiles in the current wavetile
                // since we don't want to divide by zero, it's possible for
                // wavetiles to have no options, we divide by 1 instead
                let n: u64 = max(self.possible_tiles.len() as u64, 1);

                [(*r / n) as u8, (*g / n) as u8, (*b / n) as u8]
            })
            .collect::<Vec<[u8; 3]>>();

        todo!()

        /*
        // we need to flatten the list of pixels to turn it into a texture
        let mut flat_pixels: Vec<u8> = pixels
            .into_iter()
            .flat_map(
                // consume the pixel channels, allows us to iterate over u8 instead of &u8
                |pixel| pixel.into_iter(),
            )
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

            */
    }
}
 */
