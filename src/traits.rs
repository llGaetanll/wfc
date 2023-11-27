use ndarray::Array2;

use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::{Texture, TextureCreator};
use sdl2::surface::Surface;
use sdl2::video::WindowContext;

use crate::types;

pub trait Pixel {
    fn dims(&self) -> (usize, usize);

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
        let (width, height) = self.dims();

        // we need to flatten the list of pixels to turn it into a texture
        let mut flat_pixels: Vec<u8> = self
            .pixels()
            .into_iter()
            .flat_map(|pixel| pixel.into_iter().map(|p| p))
            .collect();

        // create a surface from the flat pixels vector
        let surface = Surface::from_data(
            &mut flat_pixels,
            width as u32,    // width of the texture
            height as u32,   // height of the texture
            height as u32 * 3,
            PixelFormatEnum::RGB24,
        )
        .map_err(|e| e.to_string())?;

        // create a texture from the surface
        texture_creator
            .create_texture_from_surface(&surface)
            .map_err(|e| e.to_string())
    }

    fn show(&self, sdl_context: &sdl2::Sdl, title: &str, scaling: u32) -> Result<(), String> {
        let (width, height) = self.dims();

        // NOTE: this can fail, but it won't for any reasonable value of width and height
        let (width, height) = (width as u32 * scaling, height as u32 * scaling);

        let video_subsystem = sdl_context.video()?;
        let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;
        let window = video_subsystem
            .window(title, width, height)
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

        canvas.copy(texture, None, Rect::new(0, 0, width, height))?;
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
