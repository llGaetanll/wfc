use std::array::from_fn;
use std::collections::hash_map::DefaultHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

use ndarray::Array;
use ndarray::ArrayBase;
use ndarray::ArrayView;
use ndarray::Data;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::NdIndex;
use ndarray::RawData;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use crate::types::BoundaryHash;
use crate::types::DimN;
use crate::types::TileHash;

// Extensions to the ndarray crate needed for wfc.

pub trait ArrayTransformations<T> {
    fn rotations(&self) -> Vec<Array<T, Ix2>>;

    fn flips(&self) -> Vec<Array<T, Ix2>>;
}

// Only 2D for now
impl<'a, T> ArrayTransformations<T> for ArrayView<'a, T, Ix2>
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

pub trait ArrayHash<const N: usize> {
    fn hash(&self) -> TileHash;
}

impl<S, const N: usize> ArrayHash<N> for ArrayBase<S, DimN<N>>
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

pub trait ArrayBoundaryHash<const N: usize>
where
    DimN<N>: Dimension,
{
    fn boundary_hashes(&self) -> [[BoundaryHash; 2]; N];
}

impl<S, const N: usize> ArrayBoundaryHash<N> for ArrayBase<S, DimN<N>>
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

            SliceInfo::<_, DimN<N>, <DimN<N> as Dimension>::Smaller>::try_from(slice_info).unwrap()
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

pub trait ArrayNeighbors<'a, S, const N: usize>
where
    S: Data,
    DimN<N>: Dimension,
{
    fn neighbors(&'a self, index: [usize; N]) -> [[Option<&'a S::Elem>; 2]; N];
}

impl<'a, S, const N: usize> ArrayNeighbors<'a, S, N> for ArrayBase<S, DimN<N>>
where
    S: Data,
    DimN<N>: Dimension,

    [usize; N]: NdIndex<DimN<N>>,
{
    fn neighbors(&'a self, index: [usize; N]) -> [[Option<&'a S::Elem>; 2]; N] {
        let mut neighbors: [[Option<&'a S::Elem>; 2]; N] = from_fn(|_| [None; 2]);

        let shape = self.shape();

        // TODO: allow more wrapping techniques for indices
        for d in 0..N {
            let left = if index[d] > 0 {
                let mut index = index.clone();
                index[d] -= 1;
                Some(&self[index])
            } else {
                None
            };

            let right = if index[d] < shape[d] - 1 {
                let mut index = index.clone();
                index[d] += 1;
                Some(&self[index])
            } else {
                None
            };

            neighbors[d] = [left, right];
        }

        neighbors
    }
}

/// An iterator over an `A: ArrayBase<S, D>` that takes a starting point `start` and produces
/// elements of type `Vec<S::Elem>` where index `i` is all entries of the array with Manhattan
/// distance `i` from the starting point `start`. Used internally in propagate.
pub struct ManhattanIter<'a, S, const N: usize>
where
    S: Data + RawData,
    DimN<N>: Dimension,
{
    dist: usize,
    max_dist: usize,
    start: [usize; N],
    data: &'a ArrayBase<S, DimN<N>>,
}

impl<'a, S, const N: usize> ManhattanIter<'a, S, N>
where
    S: 'a + Data + RawData,
    DimN<N>: Dimension,
{
    pub fn new(start: [usize; N], data: &'a ArrayBase<S, DimN<N>>) -> Self {
        let max_dist = Self::compute_max_dist(start, data.shape());

        ManhattanIter {
            start,
            max_dist,
            dist: 0,
            data,
        }
    }

    fn compute_max_dist(start: [usize; N], shape: &[usize]) -> usize {
        assert_eq!(shape.len(), N);

        // find all corner coordinates of the array
        let mut corners: Vec<Vec<usize>> = vec![vec![0; N]];
        for (axis, s) in shape.iter().enumerate() {
            let mut new_corners = corners.clone();

            for corner in new_corners.iter_mut() {
                (*corner)[axis] = *s;
            }

            corners.extend(new_corners.into_iter());
        }

        let dist = |p: &[usize; N]| Self::dist(&start, p);

        // find the farthest corner from the starting point
        let far_corner = corners
            .into_iter()
            .map(|corner| corner.try_into().unwrap()) // type casting
            .max_by_key(dist)
            .unwrap(); // farthest corner is always defined

        dist(&far_corner)
    }

    // compute the manhattan distance between two points
    fn dist(a: &[usize; N], b: &[usize; N]) -> usize {
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| if ai > bi { ai - bi } else { bi - ai })
            .sum()
    }
}

impl<'a, S, const N: usize> Iterator for ManhattanIter<'a, S, N>
where
    S: Data + RawData,
    DimN<N>: Dimension,

    // ensure that we can index into an N dimensional array using a [usize; N]
    [usize; N]: NdIndex<DimN<N>>,
{
    type Item = Vec<&'a S::Elem>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.dist > self.max_dist {
            return None;
        }

        let elements = util::man_dist(self.start, self.dist)
            .into_iter()
            .map(|i| &self.data[i])
            .collect();

        self.dist += 1;

        Some(elements)
    }
}

impl<'a, S, const N: usize> ExactSizeIterator for ManhattanIter<'a, S, N>
where
    S: Data + RawData,
    DimN<N>: Dimension,

    // ensure that we can index into an N dimensional array using a [usize; N]
    [usize; N]: NdIndex<DimN<N>>,
{
    fn len(&self) -> usize {
        self.max_dist
    }
}

mod util {
    pub fn man_dist<const N: usize>(p: [usize; N], dist: usize) -> Vec<[usize; N]> {
        // k is the index of p
        fn rec<const N: usize>(p: [isize; N], dist: isize, k: usize) -> Vec<[isize; N]> {
            if dist == 0 {
                return vec![p];
            }

            // the last coordinate is different. We have 0 degrees of freedom
            if k == N - 1 {
                let mut left = p.clone();
                let mut right = p.clone();

                left[k] -= dist;
                right[k] += dist;

                return vec![left, right];
            }

            let mut indices = Vec::new();
            for d in (-dist)..=dist {
                let mut p = p.clone();

                p[k] += d;

                indices.extend(rec(p, dist - d.abs(), k + 1).iter())
            }

            indices
        }

        // NOTE: we cast these to isize which is technically fallible
        let p: [isize; N] = p
            .iter()
            .map(|&x| x as isize)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let dist = dist as isize;

        let cast_point = |p: [isize; N]| -> Option<[usize; N]> {
            let mut res: [usize; N] = [0; N];

            for (i, &x) in p.iter().enumerate() {
                if x < 0 {
                    return None;
                }

                // this does not fail
                res[i] = x as usize;
            }

            Some(res)
        };

        rec(p, dist, 0).into_iter().filter_map(cast_point).collect()
    }
}
