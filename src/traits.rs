use std::array::from_fn;
use std::collections::hash_map::DefaultHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hasher;

use core::hash::Hash;

use ndarray::ArrayBase;
use ndarray::ArrayView;
use ndarray::Data;
use ndarray::DataMut;
use ndarray::Dimension;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use crate::types::DimN;

/// A trait to characterize types with a `Hash` boundary
pub trait BoundaryHash<const N: usize>: Hash
where
    DimN<N>: Dimension,
{
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

pub trait MergeMut {
    fn merge(&mut self, other: &Self);
}

impl<E, S, D> MergeMut for ArrayBase<S, D>
where
    E: MergeMut + Clone,
    S: DataMut<Elem = E>,
    D: Dimension
{
    fn merge(&mut self, other: &ArrayBase<S, D>) {
        assert!(self.shape() == other.shape(), "cannot merge arrays of different shapes");

        self.zip_mut_with(other, |a, b| a.merge(b));
    }
}
