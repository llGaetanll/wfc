use ndarray::{ArrayView, ArrayView2, Dimension, Ix2, SliceArg, SliceInfo, SliceInfoElem};

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::traits::Hashable;
use super::traits::Pixelize;
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
pub struct Tile<'a, T, D>
where
    T: Hashable,
{
    /// The data of the Tile. Note that the tile does not own its data.
    data: ArrayView<'a, T, D>,

    /// The hash of the current tile. This is computed from the Type that the
    /// tile references. If two tiles have the same data, they will have
    /// the same hash, no matter the type.
    id: TileHash,

    /// The hash of each side of the tile.
    /// Each tuple represents opposite sides on an axis of the tile.
    hashes: Vec<(BoundaryHash, BoundaryHash)>,
}

impl<'a, T, D> Tile<'a, T, D>
where
    D: Dimension,
    T: Hashable,

    // ensures that `D` is such that `SliceInfo` implements the `SliceArg` type of it.
    SliceInfo<Vec<SliceInfoElem>, D, <D as Dimension>::Smaller>: SliceArg<D>,
{
    /***
     * Create a new tile from an ndarray
     */
    pub fn new(data: ArrayView<'a, T, D>) -> Self {
        let id = Self::get_id(&data);
        let hashes = Self::get_hashes(&data);

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
    fn get_id(data: &ArrayView<'a, T, D>) -> TileHash {
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
    fn get_hashes(data: &ArrayView<'a, T, D>) -> Vec<(BoundaryHash, BoundaryHash)> {
        let mut b_hashes: Vec<(BoundaryHash, BoundaryHash)> = Vec::new();

        let boundary_slice = SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        };

        let n = data.ndim();

        // helper to slice the array on a particular axis
        let make_slice = |i: usize, side: isize| {
            let mut slice_info = vec![boundary_slice; n];
            slice_info[i] = SliceInfoElem::Index(side);

            SliceInfo::<_, D, D::Smaller>::try_from(slice_info).unwrap()
        };

        // helper to hash a vector of hashes
        let hash_vec = |hashes: Vec<u64>| {
            let mut s = DefaultHasher::new();
            hashes.hash(&mut s);
            s.finish()
        };

        // for each dimension
        for i in 0..n {
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
            b_hashes.push((front_hash, back_hash))
        }

        b_hashes
    }
}

impl<'a> Pixelize<'a> for Tile<'a, Pixel, Ix2> {
    /***
     * Returns a pixel vector representation of the current tile
     */
    fn pixels(&'a self) -> ArrayView2<Pixel> {
        self.data
    }
}
