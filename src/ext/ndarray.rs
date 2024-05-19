use std::array::from_fn;

use image::ImageBuffer;
use image::Pixel;

use ndarray::Array;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::Dimension;

use crate::types::DimN;

pub type FlatIndex = usize;
pub type NdIndex<const N: usize> = [usize; N];
pub type NeighborIndices<const N: usize> = [[Option<NdIndex<N>>; 2]; N];

pub trait WaveArrayExt<const N: usize> {
    fn max_manhattan_dist(&self) -> usize;
    fn get_nd_index(&self, flat_index: FlatIndex) -> NdIndex<N>;
    fn get_index_neighbors(&self, index: NdIndex<N>) -> NeighborIndices<N>;
}

impl<S, const N: usize> WaveArrayExt<N> for ArrayBase<S, DimN<N>>
where
    DimN<N>: Dimension,
    S: Data,
{
    /// Find the maximum manhattan distance between any two points of the array.
    fn max_manhattan_dist(&self) -> usize {
        let shape = self.shape();
        shape.iter().sum::<usize>() - N
    }

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

/// Convert from an array to an image
pub trait ArrayToImageExt<P>
where
    P: Pixel,
{
    fn to_image(self) -> Option<ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>>;
}

impl<P> ArrayToImageExt<P> for Array2<P>
where
    P: Pixel,
{
    fn to_image(self) -> Option<ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>> {
        let s = self.shape();
        let (w, h) = (s[0], s[1]);
        let c = P::CHANNEL_COUNT as usize;

        let data = self.into_raw_vec();

        let mut subpixel_data: Vec<P::Subpixel> = Vec::with_capacity(w * h * c);
        for pixel in data {
            subpixel_data.extend(pixel.channels().iter());
        }

        ImageBuffer::from_vec(w as u32, h as u32, subpixel_data)
    }
}

pub trait ArrayTransformations<T> {
    fn rotations(&self) -> Vec<Array2<T>>;

    fn flips(&self) -> Vec<Array2<T>>;
}

impl<T> ArrayTransformations<T> for Array2<T>
where
    T: Clone,
{
    fn rotations(&self) -> Vec<Array2<T>> {
        // side length of tile
        let n = self.shape().first().unwrap();
        let view_t = self.t();

        let r1: Array2<T> = self.clone();
        let r2: Array2<T> =
            Array::from_shape_fn(view_t.dim(), |(i, j)| view_t[[n - i - 1, j]].to_owned());
        let r3: Array2<T> =
            Array::from_shape_fn(self.dim(), |(i, j)| self[[n - i - 1, n - j - 1]].to_owned());
        let r4: Array2<T> =
            Array::from_shape_fn(view_t.dim(), |(i, j)| view_t[[i, n - j - 1]].to_owned());

        vec![r1, r2, r3, r4]
    }

    fn flips(&self) -> Vec<Array2<T>> {
        // sidelength of tile
        let n = self.shape().first().unwrap();

        let horizontal = Array::from_shape_fn(self.dim(), |(i, j)| self[[n - i - 1, j]].to_owned());
        let vertical = Array::from_shape_fn(self.dim(), |(i, j)| self[[i, n - j - 1]].to_owned());

        vec![horizontal, vertical]
    }
}
