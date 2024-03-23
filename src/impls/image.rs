use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

use image::ImageBuffer;
use image::Pixel;

use ndarray::Array2;

use crate::data::Flips;
use crate::data::Rotations;
use crate::data::TileSet;

use crate::ext::image::ImageToArrayExt;
use crate::ext::ndarray::ArrayTransformations;

pub struct ImageParams<P: Pixel> {
    pub image: ImageBuffer<P, Vec<P::Subpixel>>,
    pub win_size: usize,
}

impl<Sp, P> ImageParams<P>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash,
{
    /// Construct a `TileSet` from an `image` and `win_size`.
    pub fn tile_set(&self) -> TileSet<Array2<P>, 2> {
        let image = &self.image;

        let pixels: Array2<P> = image.to_array().unwrap();

        let window = [self.win_size; 2];

        // complete list of unique tiles
        let mut tiles: HashMap<u64, Array2<P>> = HashMap::new();

        for window in pixels.windows(window) {
            let tile = window.to_owned();

            // hash the tile
            let mut hasher = DefaultHasher::new();
            tile.hash(&mut hasher);
            let hash = hasher.finish();

            tiles.insert(hash, tile);
        }

        let data: Vec<_> = tiles.into_values().collect();

        TileSet::new(data, self.win_size)
    }
}

impl<'a, Sp, P> TileSet<'a, Array2<P>, 2>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash,
{
    /// Constructs a `TileSet` from a list of images.
    ///
    /// Panics if any of the following are true:
    /// - The list of images is empty
    /// - The images are non-square
    /// - The images are not all the same size
    pub fn from_images(tiles: Vec<ImageBuffer<P, Vec<Sp>>>) -> Self {
        assert!(!tiles.is_empty(), "tiles list is empty");

        let (width, height) = tiles[0].dimensions();
        assert!(width == height, "tiles must be square");

        let tile_size = width as usize;

        for tile in &tiles {
            let (width, height) = tile.dimensions();

            assert!(width == height, "tiles must be square");
            assert!(
                tile_size == width as usize,
                "tile are not all the same size"
            );
        }

        let arrays: Vec<_> = tiles
            .into_iter()
            .map(|tile| tile.to_array().unwrap())
            .collect();

        TileSet::new(arrays, tile_size)
    }
}

impl<'a, Sp, P> Rotations<'a, Array2<P>, 2> for TileSet<'a, Array2<P>, 2>
where
    Sp: Clone + Hash,
    P: Pixel<Subpixel = Sp> + Hash,
{
    fn with_rots(&'a mut self) -> &mut TileSet<'a, Array2<P>, 2> {
        let new_tiles: Vec<Array2<P>> = self
            .data
            .iter()
            .flat_map(|tile| tile.rotations().into_iter())
            .collect();

        self.data = new_tiles;

        self.compute_hashes();

        self
    }
}

impl<'a, Sp, P> Flips<'a, Array2<P>, 2> for TileSet<'a, Array2<P>, 2>
where
    Sp: Clone + Hash,
    P: Pixel<Subpixel = Sp> + Hash,
{
    fn with_flips(&'a mut self) -> &'a mut TileSet<'a, Array2<P>, 2> {
        let new_tiles: Vec<_> = self
            .data
            .iter()
            .flat_map(|tile| tile.flips().into_iter())
            .collect();

        self.data = new_tiles;

        self.compute_hashes();

        self
    }
}
