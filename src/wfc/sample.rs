use image::{GenericImageView, Pixel as ImgPixel};
use ndarray::{Array, ArrayView, Dim, Dimension, Ix2, NdIndex, SliceArg, SliceInfo, SliceInfoElem};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use super::tile::{DimN, Tile};
use super::traits::Hashable;
use super::types::Pixel;

/// We use an ndarray of type A and dimension D to store our data
/// `N` is the dimension of the Sample, `T` is the type of each element
///
/// Samples contain a list of tiles
pub struct Sample<T, const N: usize>(pub Vec<Array<T, DimN<N>>>)
where
    T: Hashable,
    DimN<N>: Dimension;

// TODO: use better types to represent rotations and flips?
pub fn from_image(
    path: &Path,
    win_size: usize,
    rotations: bool,
    flips: bool,
) -> Result<Sample<Pixel, 2>, String> {
    // open the sample image
    let img = image::open(path).map_err(|e| e.to_string())?;

    println!("img info: {:?}", img.dimensions());
    let (width, height) = img.dimensions();

    let pixels: Array<Pixel, Ix2> = img
        .pixels()
        .map(|p| p.2.to_rgb().0)
        .collect::<Array<_, _>>()
        .into_shape((width as usize, height as usize))
        .unwrap();

    // create a cubic window of side length `size`
    let dim = [win_size; 2];

    // complete list of unique tiles
    // let mut tiles: HashMap<u64, Array<Pixel, Ix2>> = HashMap::new();
    let mut tiles = Vec::new();

    for window in pixels.windows(dim).into_iter() {
        // let flips = tile_flips(window);
        let rots = tile_rots(window);
        // let flips = flips.into_iter().map(|tile| (hash(&tile), tile));

        tiles.extend(rots);
    }

    // let tiles = tiles.into_values().collect();
    let sample = Sample(tiles);

    // sample.tiles_to_img();

    Ok(sample)
}

fn hash<T, const N: usize>(view: &Array<T, DimN<N>>) -> u64
where
    T: Hash,
    DimN<N>: Dimension,
{
    let mut s = DefaultHasher::new();
    view.hash(&mut s);
    s.finish()
}

// only 2D for now
fn tile_flips<T>(view: ArrayView<T, Ix2>) -> Vec<Array<T, Ix2>>
where
    T: Clone,
{
    let mut flips = vec![];
    flips.reserve(2);

    // width of tile
    let size = *view.shape().first().unwrap();

    for axis in 0..2 {
        let flip: Array<T, Ix2> = Array::from_shape_fn(view.dim(), |idx| {
            let mut idx: [usize; 2] = idx.into();
            idx[axis] = size - idx[axis] - 1;

            view[idx].to_owned()
        });

        flips.push(flip);
    }

    flips
}

// only in 2D for now
fn tile_rots<T>(view: ArrayView<T, Ix2>) -> Vec<Array<T, Ix2>>
where
    T: Clone,
{
    // width of tile
    let n = *view.shape().first().unwrap();
    let view_t = view.t();

    let r1: Array<T, Ix2> = view.to_owned();
    let r2: Array<T, Ix2> =
        Array::from_shape_fn(view_t.dim(), |(i, j)| view_t[[n - i - 1, j]].to_owned());
    let r3: Array<T, Ix2> =
        Array::from_shape_fn(view.dim(), |(i, j)| view[[n - i - 1, n - j - 1]].to_owned());
    let r4: Array<T, Ix2> =
        Array::from_shape_fn(view_t.dim(), |(i, j)| view_t[[i, n - j - 1]].to_owned());

    vec![r1, r2, r3, r4]
}

impl<T, const N: usize> Sample<T, N>
where
    T: Hashable,
    DimN<N>: Dimension,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    /***
     * Returns the tiles from the sample
     */
    pub fn tiles(&self) -> Vec<Tile<T, N>> {
        self.0.iter().map(|tile| Tile::new(tile.view())).collect()
    }
}

impl Sample<Pixel, 2> {
    fn tiles_to_img(&self)
    where
        [usize; 2]: NdIndex<DimN<2>>,
    {
        let tiles = &self.0;

        let [width, height]: [usize; 2] = self.0.first().unwrap().shape().try_into().unwrap();
        let scaling: usize = 30;

        for (i, tile) in tiles.iter().enumerate() {
            let mut imgbuf =
                image::ImageBuffer::new((width * scaling) as u32, (height * scaling) as u32);

            // Iterate over the coordinates and pixels of the image
            for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
                let x = x as usize / scaling;
                let y = y as usize / scaling;

                let px = tile.get([x, y]).unwrap().to_owned();
                *pixel = image::Rgb(px);
            }

            imgbuf.save(format!("./assets/tiles/tile-{i}.png")).unwrap();
        }
    }
}

/*
impl Pixelizable for Sample<Pixel, 2> {
    fn pixels(&self) -> ndarray::Array2<Pixel> {
        self.0.to_owned()
    }
}

impl SdlTexturable for Sample<Pixel, 2> {
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String> {
        let [width, height]: [usize; 2] = self.0.shape().try_into().unwrap();

        let mut flat_pixels: Vec<u8> = self
            .pixels()
            .into_iter()
            .flat_map(|pixel| pixel.into_iter().map(|p| p))
            .collect();

        let surface = Surface::from_data(
            &mut flat_pixels,
            width as u32,     // width of the texture
            height as u32,    // height of the texture
            width as u32 * 3, // this is the number of channels for each pixel
            PixelFormatEnum::RGB24,
        )
        .map_err(|e| e.to_string())?;

        // create a texture from the surface
        texture_creator
            .create_texture_from_surface(surface)
            .map_err(|e| e.to_string())
    }
}

impl Sample<Pixel, 2> {
    pub fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
        let [width, height]: [usize; 2] = self.0.shape().try_into().unwrap();

        let video_subsystem = sdl_context.video()?;
        let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;
        let window = video_subsystem
            .window("Tile View", 50 * width as u32, 50 * height as u32) // TODO: cleanup
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

        canvas.copy(
            texture,
            None,
            Rect::new(0, 0, 50 * width as u32, 50 * height as u32), // TODO: cleanup
        )?;

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
*/
