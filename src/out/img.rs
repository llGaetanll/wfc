use image::{ImageBuffer, Rgb};

use crate::traits::Pixel;

pub trait Image: Pixel {
    fn to_img(&self, scaling: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = self.dims();
        let pixels = self.pixels();

        let mut imgbuf =
            image::ImageBuffer::new((width * scaling) as u32, (height * scaling) as u32);

        // Iterate over the coordinates and pixels of the image
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            let x = x as usize / scaling;
            let y = y as usize / scaling;

            let px = pixels.get([x, y]).unwrap().to_owned();
            *pixel = image::Rgb(px.into());
        }

        imgbuf
    }
}
