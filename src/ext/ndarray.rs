use crate::types::DimN;
use ndarray::ArrayBase;
use std::array::from_fn;

pub type FlatIndex = usize;
pub type NdIndex<const N: usize> = [usize; N];
pub type NeighborIndices<const N: usize> = [[Option<NdIndex<N>>; 2]; N];

pub trait WaveArrayExt<const N: usize> {
    fn get_nd_index(&self, flat_index: FlatIndex) -> NdIndex<N>;
    fn get_index_groups(&self, start: NdIndex<N>) -> Vec<Vec<NdIndex<N>>>;
    fn get_index_neighbors(&self, index: NdIndex<N>) -> NeighborIndices<N>;
}

impl<S, const N: usize> WaveArrayExt<N> for ArrayBase<S, DimN<N>>
where
    DimN<N>: ndarray::Dimension,
    S: ndarray::Data,
{
    /// Converts a flat index into an NdIndex
    fn get_nd_index(&self, flat_index: FlatIndex) -> NdIndex<N> {
        let strides = self.strides();

        let mut nd_index: [usize; N] = [0; N];
        let mut idx_remain = flat_index;

        // safe because |strides| == N
        unsafe {
            for (i, stride) in strides.iter().enumerate() {
                let stride = *stride as usize;
                let idx = nd_index.get_unchecked_mut(i); // will not fail
                *idx = idx_remain / stride;
                idx_remain %= stride;
            }
        }

        nd_index
    }

    /// Computes all the Manhattan distance groups
    fn get_index_groups(&self, start: NdIndex<N>) -> Vec<Vec<NdIndex<N>>> {
        // compute manhattan distance between `start` and `index`
        let manhattan_dist = |index: NdIndex<N>| -> usize {
            start
                .iter()
                .zip(index.iter())
                .map(|(&a, &b)| ((a as isize) - (b as isize)).unsigned_abs())
                .sum()
        };

        // compute the tile farthest away from the starting index
        // this gives us the number of index groups B are in our wave given the `starting_index`
        let max_manhattan_dist = self
            .iter()
            .enumerate()
            .map(|(i, _)| manhattan_dist(self.get_nd_index(i)))
            .max()
            .unwrap();

        let mut index_groups: Vec<Vec<NdIndex<N>>> = vec![Vec::new(); max_manhattan_dist];

        for (index, _) in self.iter().enumerate() {
            let nd_index = self.get_nd_index(index);
            let dist = manhattan_dist(nd_index);

            if dist == 0 {
                continue;
            }

            index_groups[dist - 1].push(nd_index);
        }

        index_groups
    }

    /// For a given NdIndex, returns the list of all the neighbors of that index
    fn get_index_neighbors(&self, index: NdIndex<N>) -> NeighborIndices<N> {
        let shape = self.shape();

        from_fn(|axis| {
            let left = if index[axis] == 0 {
                None
            } else {
                let mut left = index;
                left[axis] -= 1;
                Some(left)
            };

            let right = if index[axis] == shape[axis] - 1 {
                None
            } else {
                let mut right = index;
                right[axis] += 1;
                Some(right)
            };

            [left, right]
        })
    }
}
