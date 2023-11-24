use ndarray::Array;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::ArrayView;
use ndarray::Data;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use sdl2::render::{Texture, TextureCreator};
use sdl2::video::WindowContext;

use std::collections::hash_map::DefaultHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

use std::hash::Hash;
use std::hash::Hasher;

use super::types::BoundaryHash;
use super::types::DimN;
use super::types::Pixel;
use super::types::TileHash;

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

pub trait TileArrayTransformationsExt<T> {
    fn rotations(&self) -> Vec<Array<T, Ix2>>;

    fn flips(&self) -> Vec<Array<T, Ix2>>;
}

pub trait TileArrayHashExt<const N: usize> {
    fn hash(&self) -> TileHash;
}

impl<S, const N: usize> TileArrayHashExt<N> for ArrayBase<S, DimN<N>>
where
    S: Data,
    S::Elem: Hash,
    DimN<N>: Dimension,
{
    fn hash(&self) -> TileHash {
        let mut s = DefaultHasher::new();

        <ArrayBase<S, DimN<N>> as Hash>::hash(&self, &mut s);

        s.finish()
    }
}

pub trait TileArrayBoundaryHashExt<const N: usize>
where
    DimN<N>: Dimension,
{
    fn boundary_hashes(&self) -> [[BoundaryHash; 2]; N];
}

impl<S, const N: usize> TileArrayBoundaryHashExt<N> for ArrayBase<S, DimN<N>>
where
    S: Data,
    S::Elem: Hash,
    DimN<N>: Dimension,

    // this ensures that whatever dimension we're in can be sliced into a smaller one
    // we need this to extract the boundary of the current tile so that we can hash it
    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    fn boundary_hashes(&self) -> [[BoundaryHash; 2]; N] {
        let boundary_slice = SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        };

        // helper to slice the array on a particular axis
        let make_slice = |i: usize, side: isize| {
            let mut slice_info = vec![boundary_slice; N];
            slice_info[i] = SliceInfoElem::Index(side);

            SliceInfo::<_, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>::try_from(
                slice_info,
            )
            .unwrap()
        };

        let hash_slice = |slice: ArrayView<S::Elem, _>| -> BoundaryHash {
            let mut s = DefaultHasher::new();
            slice.iter().for_each(|el| el.hash(&mut s));
            s.finish()
        };

        // maps the boundary hash to an index
        let mut hash_index: usize = 0;
        let mut unique_hashes: HashMap<BoundaryHash, usize> = HashMap::new();
        let mut tile_hashes: [[BoundaryHash; 2]; N] = [[0; 2]; N];

        // for each axis
        for n in 0..N {
            // front slice of the current axis
            let front = make_slice(n, 0);

            // back slice of the current axis
            let back = make_slice(n, -1);

            // slice the array on each axis
            let front_slice = self.slice(front);
            let back_slice = self.slice(back);

            // hash each slice
            let front_hash = hash_slice(front_slice);
            let back_hash = hash_slice(back_slice);

            // if the hash isn't already in the map, add it
            match unique_hashes.entry(front_hash) {
                Entry::Vacant(v) => {
                    v.insert(hash_index);
                    hash_index += 1;
                }
                Entry::Occupied(_) => {}
            }

            match unique_hashes.entry(back_hash) {
                Entry::Vacant(v) => {
                    v.insert(hash_index);
                    hash_index += 1;
                }
                Entry::Occupied(_) => {}
            }

            // add to hash list
            tile_hashes[n] = [front_hash, back_hash];
        }

        tile_hashes
    }
}

impl<'a, T> TileArrayTransformationsExt<T> for ArrayView<'a, T, Ix2>
where
    T: Clone,
{
    fn rotations(&self) -> Vec<Array<T, Ix2>> {
        // side length of tile
        let n = self.shape().first().unwrap();
        let view_t = self.t();

        let r1: Array<T, Ix2> = self.to_owned();
        let r2: Array<T, Ix2> =
            Array::from_shape_fn(view_t.dim(), |(i, j)| view_t[[n - i - 1, j]].to_owned());
        let r3: Array<T, Ix2> =
            Array::from_shape_fn(self.dim(), |(i, j)| self[[n - i - 1, n - j - 1]].to_owned());
        let r4: Array<T, Ix2> =
            Array::from_shape_fn(view_t.dim(), |(i, j)| view_t[[i, n - j - 1]].to_owned());

        vec![r1, r2, r3, r4]
    }

    fn flips(&self) -> Vec<Array<T, Ix2>> {
        // sidelength of tile
        let n = self.shape().first().unwrap();

        let horizontal = Array::from_shape_fn(self.dim(), |(i, j)| self[[n - i - 1, j]].to_owned());
        let vertical = Array::from_shape_fn(self.dim(), |(i, j)| self[[i, n - j - 1]].to_owned());

        vec![horizontal, vertical]
    }
}

/***
* Can create a texture from the object
*/
pub trait SdlTexturable {
    fn texture<'b>(
        &self,
        texture_creator: &'b TextureCreator<WindowContext>,
    ) -> Result<Texture<'b>, String>;
}

// pub trait Hash {
//     fn hash(&self) -> u64;
// }
