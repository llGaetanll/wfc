use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;

use image::GenericImage;
use image::ImageBuffer;
use image::Pixel;
use image::Primitive;
use image::Rgb;
use image::Rgba;

use ndarray::Array2;

use crate::data::TileSet;
use crate::ext::image::ImageToArrayExt;
use crate::ext::ndarray::ArrayToImageExt;
use crate::surface::Flat;
use crate::surface::FlatWave;
use crate::surface::Surface;
use crate::traits::Merge;
use crate::traits::Recover;

pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;
pub type ImageWave<P> = FlatWave<Array2<P>, Image<P>, 2>;
pub type ImageTileSet<P, S> = TileSet<Array2<P>, Image<P>, S, 2>;

pub struct ImageParams<I: GenericImage, S: Surface<2>> {
    pub image: I,
    pub win_size: usize,
    _s: PhantomData<S>,
}

impl<Sp, P, I, S> ImageParams<I, S>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash + Merge,
    I: GenericImage<Pixel = P>,
    S: Surface<2>,
{
    pub fn new_flat(image: I, win_size: usize) -> ImageParams<I, Flat> {
        ImageParams {
            image,
            win_size,
            _s: PhantomData::<Flat>,
        }
    }

    /// Construct a `TileSet` from an `image` and `win_size`.
    pub fn tileset(&self) -> ImageTileSet<P, S> {
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

#[cfg(feature = "wrapping")]
impl<Sp, P, I, S> ImageParams<I, S>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash + Merge,
    I: GenericImage<Pixel = P>,
    S: Surface<2>,
{
    pub fn new_projective_plane(image: I, win_size: usize) -> ImageParams<I, Flat> {
        todo!()
    }

    pub fn new_torus(image: I, win_size: usize) -> ImageParams<I, Flat> {
        todo!()
    }

    pub fn new_klein_bottle(image: I, win_size: usize) -> ImageParams<I, Flat> {
        todo!()
    }
}

impl<Sp, P, S> ImageTileSet<P, S>
where
    Sp: Hash,
    P: Pixel<Subpixel = Sp> + Hash + Merge,
    S: Surface<2>,
{
    /// Constructs a `TileSet` from a list of images.
    ///
    /// Panics if any of the following are true:
    /// - The list of images is empty
    /// - The images are non-square
    /// - The images are not all the same size
    pub fn from_images<I>(tiles: Vec<I>) -> ImageTileSet<P, S>
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

impl<P: Pixel> Recover<Image<P>> for Array2<P> {
    type Inner = Array2<P>;

    fn recover(&self) -> Image<P> {
        self.clone().to_image().unwrap()
    }
}
