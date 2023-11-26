use ndarray::{Array2, ArrayView2};

use sdl2::render::{Texture, TextureCreator};
use sdl2::video::WindowContext;

use crate::types;

/***
* Returns an array of pixels
*/
pub trait Pixel {
    fn pixels(&self) -> Array2<types::Pixel>;
}

/***
* Can create a texture from the object
*/
pub trait SdlTexture {
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String>;
}
