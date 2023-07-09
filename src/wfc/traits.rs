use ndarray::ArrayView2;

use sdl2::render::{Texture as SdlTexture, TextureCreator};
use sdl2::video::WindowContext;

use super::types::Pixel;

pub trait Pixelize<'a> {
    fn pixels(&'a self) -> ArrayView2<'a, Pixel>;
}

pub trait SdlView {
    fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String>;

    fn texture<'a>(
        &self,
        texture_creator: &'a TextureCreator<WindowContext>,
    ) -> Result<SdlTexture<'a>, String>;
}

// TODO
pub trait Image {}

pub trait Hashable {
    fn hash(&self) -> u64;
}
