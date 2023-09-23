use ndarray::Array2;

use sdl2::render::{Texture, TextureCreator};
use sdl2::video::WindowContext;

use super::types::Pixel;

/***
* Returns an array of pixels
*/
pub trait Pixelizable {
    fn pixels(&self) -> Array2<Pixel>;
}

/***
* Can call show on the object to display a window
*/
/*
pub trait SdlView {
    type Updates;

    /***
     * Opens a window displaying the object
     */
    fn show(&self, sdl_context: &sdl2::Sdl, rx: Receiver<Self::Updates>) -> Result<(), String>;
}
*/

/***
* Can create a texture from the object
*/
pub trait SdlTexturable {
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String>;
}

pub trait Hashable {
    fn hash(&self) -> u64;
}
