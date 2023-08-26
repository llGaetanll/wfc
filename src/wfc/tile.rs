use ndarray::{Array2, ArrayView, Dim, Dimension, SliceArg, SliceInfo, SliceInfoElem};
use sdl2::event::Event;
use sdl2::image::InitFlag;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::Texture;
use sdl2::render::TextureCreator;
use sdl2::surface::Surface;
use sdl2::video::WindowContext;

use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use super::traits::Hashable;
use super::traits::Pixelizable;
use super::traits::SdlTexturable;
use super::types::BoundaryHash;
use super::types::Pixel;

pub type TileHash = u64;

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
    T: Hashable,
    Dim<[usize; N]>: Dimension,
{
    /// The data of the Tile. Note that the tile does not own its data.
    data: ArrayView<'a, T, Dim<[usize; N]>>,

    /// The hash of the current tile. This is computed from the Type that the
    /// tile references. If two tiles have the same data, they will have
    /// the same hash, no matter the type.
    pub id: TileHash,

    /// The hash of each side of the tile.
    /// Each tuple represents opposite sides on an axis of the tile.
    pub hashes: [(BoundaryHash, BoundaryHash); N],
}

impl<'a, T, const N: usize> Tile<'a, T, N>
where
    T: Hashable,
    Dim<[usize; N]>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    /***
     * Create a new tile from an ndarray
     */
    pub fn new(data: ArrayView<'a, T, Dim<[usize; N]>>) -> Self {
        let id = Self::compute_id(&data);
        let hashes = Self::compute_hashes(&data);

        Tile { data, id, hashes }
    }

    /***
     * Return the shape of the tile
     *
     * Note: It is enforced that all axes of the tile be the same size.
     */
    pub fn shape(&self) -> usize {
        self.data.shape()[0]
    }

    /***
     * Compute the hash of the tile
     */
    fn compute_id(data: &ArrayView<'a, T, Dim<[usize; N]>>) -> TileHash {
        // we iterate through each element of the tile and hash it
        // it's important to note that the individual hashes of each element
        // cannot collide. Hasher must ensure this

        // NOTE: parallelize this? maybe not, it's too deep in the call stack
        let hashes: Vec<u64> = data.iter().map(|el| el.hash()).collect();

        // TODO: speed test this
        // hash the list of hashes into one final hash for the whole tile
        let mut s = DefaultHasher::new();
        hashes.hash(&mut s);
        s.finish()
    }

    /***
     * Compute the boundary hashes of the tile
     */
    fn compute_hashes(
        data: &ArrayView<'a, T, Dim<[usize; N]>>,
    ) -> [(BoundaryHash, BoundaryHash); N] {
        let mut b_hashes: [(BoundaryHash, BoundaryHash); N] = [(0, 0); N];
        // = Vec::new();

        let boundary_slice = SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        };

        // helper to slice the array on a particular axis
        // FIXME: switch to using arrays instead of vecs for free speedup
        let make_slice = |i: usize, side: isize| {
            let mut slice_info = vec![boundary_slice; N];
            slice_info[i] = SliceInfoElem::Index(side);

            SliceInfo::<_, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>::try_from(
                slice_info,
            )
            .unwrap()
        };

        // helper to hash a vector of hashes
        let hash_vec = |hashes: Vec<u64>| {
            let mut s = DefaultHasher::new();
            hashes.hash(&mut s);
            s.finish()
        };

        // for each dimension
        for i in 0..N {
            // front slice of the current axis
            let front = make_slice(i, 0);

            // back slice of the current axis
            let back = make_slice(i, -1);

            // slice the array on each axis
            let front_slice = data.slice(front);
            let back_slice = data.slice(back);

            // flatten the boundary into a vector of hashes
            let front_hashes: Vec<u64> = front_slice.iter().map(|el| el.hash()).collect();
            let back_hashes: Vec<u64> = back_slice.iter().map(|el| el.hash()).collect();

            // hash the vector of hashes
            let front_hash = hash_vec(front_hashes);
            let back_hash = hash_vec(back_hashes);

            // add to hash list
            b_hashes[i] = (front_hash, back_hash);
        }

        b_hashes
    }
}

impl<'a> Pixelizable for Tile<'a, Pixel, 2> {
    /***
     * Returns a pixel vector representation of the current tile
     */
    fn pixels(&self) -> Array2<Pixel> {
        self.data.to_owned()
    }
}

impl<'a> SdlTexturable for Tile<'a, Pixel, 2> {
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

impl<'a> Tile<'a, Pixel, 2> {
    fn show(&self, sdl_context: &sdl2::Sdl) -> Result<(), String> {
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
    T: Hashable + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}
