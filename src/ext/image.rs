use image::GenericImage;
use image::GenericImageView;

use ndarray::Array2;
use ndarray::ShapeError;

/// Convert from an image to an array
pub trait ImageToArrayExt<I>
where
    I: GenericImage,
{
    fn to_array(&self) -> Result<Array2<<I as GenericImageView>::Pixel>, ShapeError>;
}

impl<I> ImageToArrayExt<I> for I
where
    I: GenericImage,
{
    /// Convert a GenericImage into an Array of 2 dimensions
    fn to_array(&self) -> Result<Array2<<I as GenericImageView>::Pixel>, ShapeError> {
        let (w, h) = self.dimensions();

        let pixels: Vec<<I as GenericImageView>::Pixel> =
            self.pixels().map(|(_, _, px)| px).collect();

        Array2::from_shape_vec((w as usize, h as usize), pixels)
    }
}
