use image::GenericImage;

use image::Pixel;
use ndarray::Array2;
use ndarray::ShapeError;

/// Convert from an image to an array
pub trait ImageToArrayExt<P, I>
where
    P: Pixel,
    I: GenericImage<Pixel = P>,
{
    fn to_array(&self) -> Result<Array2<P>, ShapeError>;
}

impl<P, I> ImageToArrayExt<P, I> for I
where
    P: Pixel,
    I: GenericImage<Pixel = P>,
{
    /// Convert a GenericImage into an Array of 2 dimensions
    fn to_array(&self) -> Result<Array2<P>, ShapeError> {
        let (w, h) = self.dimensions();

        let pixels: Vec<P> = self.pixels().map(|(_, _, px)| px).collect();

        Array2::from_shape_vec((w as usize, h as usize), pixels)
    }
}
