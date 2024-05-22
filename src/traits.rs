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
use rand::RngCore;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::types::DimN;
use crate::TileSet;

/// A type `T` implements [`Rotations`] if it can make sense for `T` to be rotated.
pub trait Rotations<const N: usize>
where
    DimN<N>: Dimension,
{
    type T: Hash + Clone;

    fn with_rots(&mut self) -> &mut Self;
}

/// A type `T` implements [`Flips`] if it can make sense for `T` to be flipped.
pub trait Flips<const N: usize>
where
    DimN<N>: Dimension,
{
    type T: Hash + Clone;

    fn with_flips(&mut self) -> &mut Self;
}

/// A trait to characterize types with a [`Hash`] boundary.
pub trait BoundaryHash<const N: usize>: Hash {
    fn boundary_hashes(&self) -> [[u64; 2]; N];
}

impl<S, const N: usize> BoundaryHash<N> for ArrayBase<S, DimN<N>>
where
    S: Data,
    S::Elem: Hash,
    DimN<N>: Dimension,

    // This ensures that whatever dimension we're in can be sliced into a smaller one.
    // We need this to extract the boundary of the current tile, so that we can hash it
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

/// A type `T` implements [`Merge`] if it can make sense for `[T]` to be merged into a single `T`.
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
    /// Merge [`Array`]s of elements. Assumes that all input arrays are the same shape.
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

/// A trait for types `T` where it is possible to stitch arrays of `T` back into a single `T`.
pub trait Stitch<const N: usize> {
    type T;

    fn stitch(xs: &Array<Self::T, DimN<N>>) -> Self::T;
}

// NOTE: Only for owned arrays.
impl<T, const N: usize> Stitch<N> for Array<T, DimN<N>>
where
    T: Clone,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,
{
    type T = Array<T, DimN<N>>;

    /// Stitch an [`Array`] of [`Array`]s back into a single `Array`. Panics if any of the following
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

/// Recover the `Input`. This is used in the wave to convert back to the original type (i.e. produce
/// the full picture output.)
///
/// Note that `Input` need be [`Clone`]. This is because wave function collapse may naturally use a
/// tile more than once, and so may need to access the underlying data (`Input`) more than once as
/// well. This however has no impact on performance during collapse.

// Note that `Outer` is a generic type parameter, but input is an associated type. This is a
// design deicision to make the API more ergonomic. Indeed all three of `Wave`, `WaveTile`, and
// `Tile` impl `Recover` from `T` to `T`. However, it is often the case that `Wave`'s `Outer` type
// may differ from its `Input` type. For instance, images are internally represented as 2D arrays
// of pixels, but when we recover a `Wave` of this type, we don't want a 2D array of pixels as
// output, we want an image! This need for multiple `Outer` type implementations of `Recover`
// requires us to lift `Outer` from a mere associated type to a divine generic argument.
pub trait Recover<Outer, const N: usize> {
    type Inner: BoundaryHash<N> + Clone;

    fn recover(&self) -> Outer;
}

/// A simple wrapper trait for types which can be used as tiles of a [`Wave`]. This trait is
/// primarily to avoid repeating the underlying trait bounds all over the crate.
pub trait WaveTile<Inner, Outer, const N: usize>:
    Clone + BoundaryHash<N> + Stitch<N, T = Inner> + Recover<Outer, N, Inner = Inner>
{
}

pub struct Flat;
pub struct Torus;
pub struct ProjectivePlane;
pub struct KleinBottle;

pub trait Surface<const N: usize> {
    fn neighborhood(shape: [usize; N], i: WfcNdIndex<N>) -> [[Option<WfcNdIndex<N>>; 2]; N];
}

impl<const N: usize> Surface<N> for Flat {
    fn neighborhood(shape: [usize; N], i: WfcNdIndex<N>) -> [[Option<WfcNdIndex<N>>; 2]; N] {
        from_fn(|axis| {
            let left = if i[axis] == 0 {
                None
            } else {
                let mut left = i;
                left[axis] -= 1;
                Some(left)
            };

            let right = if i[axis] == shape[axis] - 1 {
                None
            } else {
                let mut right = i;
                right[axis] += 1;
                Some(right)
            };

            [left, right]
        })
    }
}

impl Surface<2> for Torus {
    fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
        todo!()
    }
}

impl Surface<2> for ProjectivePlane {
    fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
        todo!()
    }
}

impl Surface<2> for KleinBottle {
    fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
        todo!()
    }
}

pub trait WaveBase<Inner, Outer, S, const N: usize>
where
    Inner: WaveTile<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: Surface<N>,
{
    fn init(tileset: &TileSet<Inner, Outer, N>, shape: DimN<N>) -> Self;
}

pub trait Wave<Inner, Outer, S, const N: usize>: WaveBase<Inner, Outer, S, N>
where
    Inner: WaveTile<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: Surface<N>,
{
    fn collase<R>(&mut self, rng: &mut R) -> Outer
    where
        R: RngCore + ?Sized;
}

pub trait ParWave<Inner, Outer, S, const N: usize>: WaveBase<Inner, Outer, S, N>
where
    Inner: Send + Sync + WaveTile<Inner, Outer, N>,
    DimN<N>: Dimension,
    S: Surface<N>,
{
    fn collase_parallel<R>(&mut self, rng: &mut R) -> Outer
    where
        R: RngCore + ?Sized;
}
