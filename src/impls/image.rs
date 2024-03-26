use std::collections::hash_map::DefaultHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

use image::GenericImage;
use image::Pixel;

use image::Primitive;
use image::Rgb;
use image::Rgba;
use ndarray::Array2;

use crate::data::Flips;
use crate::data::Rotations;
use crate::data::TileSet;

use crate::ext::image::ImageToArrayExt;
use crate::ext::ndarray::ArrayTransformations;
use crate::traits::Merge;

pub struct ImageParams<I: GenericImage> {
    /// The image to pass to the `TileSet`
    pub image: I,

    /// The sidelength of the sliding window to perform on the image
    pub win_size: usize,
}

impl<Sp, P, I> ImageParams<I>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash,
    I: GenericImage<Pixel = P>,
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
    pub fn from_images<I>(tiles: Vec<I>) -> Self
    where
        I: GenericImage<Pixel = P>,
    {
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
        let mut tiles: HashMap<u64, Array2<P>> = HashMap::new();

        for tile in &self.data {
            for rotation in tile.rotations() {
                // hash the tile
                let mut hasher = DefaultHasher::new();
                rotation.hash(&mut hasher);
                let hash = hasher.finish();

                // if the tile is not already present in the map, add it
                if let Entry::Vacant(v) = tiles.entry(hash) {
                    v.insert(rotation);
                }
            }
        }

        self.data = tiles.into_values().collect::<Vec<_>>();

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
        let mut tiles: HashMap<u64, Array2<P>> = HashMap::new();

        for tile in &self.data {
            for rotation in tile.flips() {
                // hash the tile
                let mut hasher = DefaultHasher::new();
                rotation.hash(&mut hasher);
                let hash = hasher.finish();

                // if the tile is not already present in the map, add it
                if let Entry::Vacant(v) = tiles.entry(hash) {
                    v.insert(rotation);
                }
            }
        }

        self.data = tiles.into_values().collect::<Vec<_>>();

        self.compute_hashes();

        self
    }
}

impl<T: Primitive> Merge for Rgb<T> {
    /// Computes the average Pixel
    fn merge(xs: &[Self]) -> Self
    where
        Self: Sized + Clone,
    {
        let n = xs.len() as f64;

        let px: [T; 3] = xs
            .iter() // iterate over all pixels
            .fold([0., 0., 0.], |acc, px| {
                let [r1, g1, b1] = acc;
                let [r2, g2, b2]: [f64; 3] = px.0.map(|c|
                        // this won't fail since `T` is `Num`
                        c.to_f64().unwrap());

                // to avoid arithmetic overflow, divide first, add later
                [r1 + (r2 / n), g1 + (g2 / n), b1 + (b2 / n)]
            })
            .map(|c|
                // convert back into our original primitive
                // this also won't fail
                T::from(c).unwrap());

        Rgb::from(px)
    }
}

impl<T: Primitive> Merge for Rgba<T> {
    /// Computes the average Pixel
    fn merge(xs: &[Self]) -> Self
    where
        Self: Sized + Clone,
    {
        let n = xs.len() as f64;

        let px: [T; 4] = xs
            .iter() // iterate over all pixels
            .fold([0., 0., 0., 0.], |acc, px| {
                let [r1, g1, b1, a1] = acc;
                let [r2, g2, b2, a2]: [f64; 4] = px.0.map(|c|
                        // this won't fail since `T` is `Num`
                        c.to_f64().unwrap());

                // to avoid arithmetic overflow, divide first, add later
                [r1 + (r2 / n), g1 + (g2 / n), b1 + (b2 / n), a1 + (a2 / n)]
            })
            .map(|c|
                // convert back into our original primitive
                // this also won't fail
                T::from(c).unwrap());

        Rgba::from(px)
    }
}
