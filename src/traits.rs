use ndarray::Array2;

use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{Texture, TextureCreator};
use sdl2::surface::Surface;
use sdl2::video::WindowContext;

use crate::types;

pub trait Pixel {
    fn shape(&self) -> usize;

    // Note that pixels needs to return an owned copy because, even though tiles don't need to own
    // their pixels, wavetiles have to compute theirs, and so must return an owned copy.
    fn pixels(&self) -> Array2<types::Pixel>;
}

/***
* Can create a texture from the object
*/
pub trait SdlTexture: Pixel {
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String> {
        let size = self.shape();

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
