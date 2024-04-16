use std::array::from_fn;
use std::collections::hash_map::DefaultHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hasher;

use core::hash::Hash;

use ndarray::Array;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::ArrayView;
use ndarray::Axis;
use ndarray::Data;
use ndarray::Dimension;
use ndarray::IntoDimension;
use ndarray::NdIndex;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use crate::types::DimN;

pub trait Rotations<T, const N: usize>
where
    T: Hash + Clone,
    DimN<N>: Dimension,
{
    fn with_rots(&mut self) -> &mut Self;
}

pub trait Flips<T, const N: usize>
where
    T: Hash + Clone,
    DimN<N>: Dimension,
{
    fn with_flips(&mut self) -> &mut Self;
}

/// A trait to characterize types with a `Hash` boundary
pub trait BoundaryHash<const N: usize>: Hash {
    fn boundary_hashes(&self) -> [[u64; 2]; N];
}

impl<S, const N: usize> BoundaryHash<N> for ArrayBase<S, DimN<N>>
where
    S: Data,
    S::Elem: Hash,
    DimN<N>: Dimension,

    // this ensures that whatever dimension we're in can be sliced into a smaller one
    // we need this to extract the boundary of the current tile so that we can hash it
    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    fn boundary_hashes(&self) -> [[u64; 2]; N] {
        let boundary_slice = SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        };

        // helper to slice the array on a particular axis
        let make_slice = |i: usize, side: isize| {
            let mut slice_info = vec![boundary_slice; N];
            slice_info[i] = SliceInfoElem::Index(side);

            SliceInfo::<_, DimN<N>, <DimN<N> as Dimension>::Smaller>::try_from(slice_info).unwrap()
        };

        let hash_slice = |slice: ArrayView<S::Elem, _>| -> u64 {
            let mut s = DefaultHasher::new();
            slice.iter().for_each(|el| el.hash(&mut s));
            s.finish()
        };

        // maps the boundary hash to an index
        let mut hash_index: usize = 0;
        let mut unique_hashes: HashMap<u64, usize> = HashMap::new();
        let tile_hashes: [[u64; 2]; N] = from_fn(|axis| {
            // front slice of the current axis
            let front = make_slice(axis, 0);

            // back slice of the current axis
            let back = make_slice(axis, -1);

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

            [front_hash, back_hash]
        });

        tile_hashes
    }
}

pub trait Merge {
    fn merge(xs: &[Self]) -> Self
    where
        Self: Sized + Clone;
}

// NOTE: only for owned arrays.
impl<T, const N: usize> Merge for Array<T, DimN<N>>
where
    T: Merge + Clone,
    DimN<N>: Dimension,
{
    /// Merge `Array`s of elements. Assumes that all input arrays are the same shape.
    ///
    /// NOTE: requires an allocation.
    ///
    /// Panics if any of the following are true:
    /// - `xs` is empty
    /// - array shapes don't all match
    fn merge(xs: &[Self]) -> Self
    where
        Self: Sized + Clone,
    {
        assert!(
            !xs.is_empty(),
            "list of arrays cannot be empty, nothing to merge"
        );

        let same_shape = xs.iter().all(|arr| xs[0].shape() == arr.shape());
        assert!(same_shape, "arrays are not all the same shape");

        let n = xs[0].len();
        let m = xs.len();

        // requires an allocation
        let v: Vec<T> = xs
            .iter()
            .flat_map(|arr| {
                // flatten the array
                arr.as_standard_layout().into_iter()
            })
            .collect();

        let mut array = Array2::from_shape_vec((m, n), v).unwrap(); // won't fail
        array.swap_axes(0, 1); // transpose in place

        let ts: Vec<T> = array
            .axis_iter(Axis(0))
            .map(|row| {
                // I don't think this will fail though
                let row = row.as_standard_layout();
                let slice = row.as_slice().expect("was not in standard order!");

                // merge pixel-wise
                T::merge(slice)
            })
            .collect();

        let dim = xs[0].raw_dim();
        let array: Array<T, DimN<N>> = Array::from_shape_vec(dim, ts).expect("failed to shape vec");

        array
    }
}

// A trait for types `T` where it is possible to stitch arrays of `T` back into a single `T`.
//
// NOTE: It is not clear to me at the moment whether the output type should also be `T`. In every
// scenario that I can imagine, it is, but maybe this is not always the case.
pub trait Stitch<T, const N: usize> {
    fn stitch(xs: &Array<T, DimN<N>>) -> T;
}

// NOTE: Only for owned arrays.
impl<T, const N: usize> Stitch<Array<T, DimN<N>>, N> for Array<T, DimN<N>>
where
    T: Clone,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    /// Stitch an `Array` of `Array`s back into a single `Array`. Panics if any of the following
    /// are true.
    ///
    /// Requires an allocation.
    ///
    /// - `xs` is empty
    /// - The inner arrays are not all "N-square" (meaning every coordinate of their shape is the same)
    /// - The shapes of the inner `Array`s don't match
    fn stitch(xs: &Array<Array<T, DimN<N>>, DimN<N>>) -> Array<T, DimN<N>> {
        assert!(
            !xs.is_empty(),
            "list of arrays cannot be empty, nothing to stitch"
        );

        let first = xs.first().unwrap();
        let shape = first.shape();
        let sl = shape[0];
        let shape = [sl; N];

        // inner arrays need to be "square"
        assert!(
            xs.iter().all(|arr| shape == arr.shape()),
            "arrays are not all the N-square"
        );

        // compute the dimension of the outer array
        let mut dim = xs.raw_dim();
        for i in 0..N {
            dim[i] *= sl;
        }

        Array::from_shape_fn(dim, |idx| {
            // we need to turn our array index into a [usize; N]
            let idx = idx.into_dimension();
            let idx = idx.as_array_view();

            // to index into our input, we need to do two things. First, we need to pick out the
            // correct outer array from our input, and then pick out the correct inner element from
            // *it*.
            //
            // To select the correct outer array, we divide our coordinates by the sidelength of
            // the inner arrays (here called `sl`.) Then, to correctly index into that array, we
            // mod by this sidelength, and we have our element.

            // it seems the compiler optimizes away this allocation
            let outer_index: [usize; N] = idx
                .iter()
                .map(|i| i / sl)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let inner_index: [usize; N] = idx
                .iter()
                .map(|i| i % sl)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            let array: &Array<T, DimN<N>> = &xs[outer_index];
            let elem: &T = &array[inner_index];

            elem.clone()
        })
    }
}

/// Recover the `T`. This is used in the wave to convert back to the original type (i.e. produce
/// the full picture output.)
///
/// Note that `T` need be `Clone`. This is because wave function collapse may naturally use a
/// `Tile` more than once, and so may need to access the underlying data (`T`) more than once as
/// well. This however has no impact on performance during collapse.
pub trait Recover<T, U, const N: usize>
where
    T: BoundaryHash<N> + Clone,
{
    fn recover(&self) -> U;
}
