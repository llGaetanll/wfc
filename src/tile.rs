use ndarray::Array2;
use ndarray::ArrayView;
use ndarray::ArrayView2;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::Texture;
use sdl2::render::TextureCreator;
use sdl2::surface::Surface;
use sdl2::video::WindowContext;

use bit_set::BitSet;

use std::fmt::Debug;
use std::hash::Hash;

use crate::ext::ndarray as ndarray_ext;
use ndarray_ext::ArrayHash;

use crate::traits::Pixel;
use crate::traits::SdlTexture;
use crate::types;
use crate::types::DimN;
use crate::types::TileHash;

/// A `Tile` is a view into our Sample
/// `D` is the dimension of each tile
/// `T is the type of each element
///
/// Note: It would be nice to restrict the type of `hashes` so that the size of
/// the vector is exactly inline with `D`, but I don't know how I could do that
///
/// Note: all axes of the dynamic array are the same size.
pub struct Tile<'a, T, const N: usize>
where
    T: Hash,
    DimN<N>: Dimension,
{
    /// The data of the Tile. Note that the tile does not own its data.
    data: ArrayView<'a, T, DimN<N>>,

    /// The hash of the current tile. This is computed from the Type that the
    /// tile references. If two tiles have the same data, they will have
    /// the same hash, no matter the type.
    pub id: TileHash,

    /// The hash of each side of the tile.
    /// Each tuple represents opposite sides on an axis of the tile.
    pub hashes: [[BitSet; 2]; N],
}

impl<'a, T, const N: usize> Tile<'a, T, N>
where
    T: Hash,
    DimN<N>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    /***
     * Create a new tile from an ndarray
     */
    pub fn new(data: ArrayView<'a, T, DimN<N>>, hashes: [[BitSet; 2]; N]) -> Self {
        Tile {
            data,
            id: ArrayHash::hash(&data),
            hashes,
        }
    }

    /***
     * Return the shape of the tile
     *
     * Note: It is enforced that all axes of the tile be the same size.
     */
    pub fn shape(&self) -> usize {
        self.data.shape()[0]
    }
}

impl<'a> Pixel for Tile<'a, types::Pixel, 2> {
    /***
     * Returns a pixel vector representation of the current tile
     */
    fn pixels(&self) -> Array2<types::Pixel> {
        self.data.to_owned()
    }
}

impl<'a> SdlTexture for Tile<'a, types::Pixel, 2> {
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String> {
        let size = self.shape();

        let mut flat_pixels: Vec<u8> = self
            .pixels()
            .into_iter()
            .flat_map(|pixel| pixel.into_iter().map(|p| p))
            .collect();

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
            .create_texture_from_surface(surface)
            .map_err(|e| e.to_string())
    }
}

impl<'a> Tile<'a, types::Pixel, 2> {
    pub fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
        const WIN_SIZE: u32 = 100;

        let video_subsystem = sdl_context.video()?;
        let _image_context = sdl2::image::init(InitFlag::PNG | InitFlag::JPG)?;
        let window = video_subsystem
            .window("Tile View", WIN_SIZE, WIN_SIZE)
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

        canvas.copy(texture, None, Rect::new(0, 0, WIN_SIZE, WIN_SIZE))?;

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

impl<'a, T, const N: usize> Debug for Tile<'a, T, N>
where
    T: Hash + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}
