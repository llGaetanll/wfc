use std::collections::hash_map::DefaultHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

use image::GenericImage;
use image::ImageBuffer;
use image::Pixel;

use image::Primitive;
use image::Rgb;
use image::Rgba;
use ndarray::Array2;

use crate::data::TileSet;
use crate::ext::image::ImageToArrayExt;
use crate::ext::ndarray::ArrayTransformations;
use crate::traits::Flat;
use crate::traits::Flips;
use crate::traits::Merge;
use crate::traits::Rotations;
use crate::wave::Wave;

pub type ImageWave<P> = Wave<Array2<P>, ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>, Flat, 2>;
pub type ImageTileSet<P, I> = TileSet<Array2<P>, I, 2>;

pub struct ImageParams<I: GenericImage> {
    pub image: I,
    pub win_size: usize,
}

impl<Sp, P, I> ImageParams<I>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash + Merge,
    I: GenericImage<Pixel = P>,
{
    /// Construct a `TileSet` from an `image` and `win_size`.
    pub fn tileset(&self) -> ImageTileSet<P, I> {
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

impl<Sp, P, I> ImageTileSet<P, I>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash + Merge,
    I: GenericImage<Pixel = P>,
{
    /// Constructs a `TileSet` from a list of images.
    ///
    /// Panics if any of the following are true:
    /// - The list of images is empty
    /// - The images are non-square
    /// - The images are not all the same size
    pub fn from_images(tiles: Vec<I>) -> Self {
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

impl<T: Hash + Clone, Outer> Rotations<2> for TileSet<Array2<T>, Outer, 2> {
    type T = Array2<T>;

    fn with_rots(&mut self) -> &mut Self {
        let mut tiles: HashMap<u64, Array2<T>> = HashMap::new();

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

        self
    }
}

impl<T: Hash + Clone, Outer> Flips<2> for TileSet<Array2<T>, Outer, 2> {
    type T = Array2<T>;

    fn with_flips(&mut self) -> &mut Self {
        let mut tiles: HashMap<u64, Array2<T>> = HashMap::new();

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
