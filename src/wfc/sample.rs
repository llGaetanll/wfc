use image::{GenericImageView, Pixel as ImgPixel};
use ndarray::{Array, Dimension, Ix2, SliceArg, SliceInfo, SliceInfoElem};
use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{Texture, TextureCreator};
use sdl2::surface::Surface;
use sdl2::video::WindowContext;
use std::path::Path;

use super::tile::Tile;
use super::traits::{Hashable, Pixelizable, SdlTexturable, SdlView};
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
    pub fn window(&self, size: usize) -> Vec<Tile<T, D>> {
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

impl Pixelizable for Sample<Pixel, Ix2> {
    fn pixels(&self) -> ndarray::Array2<Pixel> {
        self.data.to_owned()
    }
}

impl SdlTexturable for Sample<Pixel, Ix2> {
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String> {
        let [width, height]: [usize; 2] = self.data.shape().try_into().unwrap();

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

impl SdlView for Sample<Pixel, Ix2> {
    fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
        let [width, height]: [usize; 2] = self.data.shape().try_into().unwrap();

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
