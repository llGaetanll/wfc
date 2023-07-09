use image::{GenericImageView, Pixel as ImgPixel};
use ndarray::{Array, Dimension, Ix2, SliceArg, SliceInfo, SliceInfoElem};
use std::path::Path;

use super::tile::Tile;
use super::traits::Hashable;
use super::types::Pixel;

/// We use an ndarray of type A and dimension D to store our data
/// `A` is the dimension of the Sample, `T` is the type of each element
pub struct Sample<T, D>
where
    T: Hashable,
{
    data: Array<T, D>,
}

/***
 * Creates a `Sample` object containing an image represented as a 2D array
 * of `Pixel` elements.
 */
pub fn from_image(path: &Path) -> Result<Sample<Pixel, Ix2>, String> {
    // open the sample image
    let img = image::open(path).map_err(|e| e.to_string())?;
    let (width, height) = img.dimensions();

    let pixels: Array<Pixel, Ix2> = img
        .pixels()
        .map(|p| p.2.to_rgb().0)
        .collect::<Array<_, _>>()
        .into_shape((width as usize, height as usize))
        .unwrap();

    Ok(Sample { data: pixels })
}

impl<T, D> Sample<T, D>
where
    D: Dimension,
    T: Hashable,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, D, <D as Dimension>::Smaller>: SliceArg<D>,
{
    /***
     * Return a vector of tiles by performing sliding windows.
     *
     * `size`: the side length of the window.
     */
    pub fn window<'a>(&'a self, size: usize) -> Vec<Tile<'a, T, D>> {
        let n = self.data.ndim();

        // create a cubic window of side length `size`
        let mut win_size = self.data.raw_dim();
        for i in 0..n {
            win_size[i] = size;
        }

        let tiles: Vec<Tile<T, D>> = self
            .data
            .windows(win_size)
            .into_iter()
            .map(|window| Tile::new(window))
            .collect();

        tiles
    }
}
