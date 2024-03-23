use std::array::from_fn;
use std::mem;

use image::ImageBuffer;
use image::Pixel;

use ndarray::Array;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::Dimension;

use crate::types::DimN;

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
    DimN<N>: Dimension,
    S: Data,
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

/// Convert from an array to an image
pub trait ArrayToImageExt<P>
where
    P: Pixel,
{
    fn to_image(self) -> Option<ImageBuffer<P, Vec<P::Subpixel>>>;
}

impl<P> ArrayToImageExt<P> for Array2<P>
where
    P: Pixel + 'static, // FIXME: can't be static, this will leak
{
    fn to_image(self) -> Option<ImageBuffer<P, Vec<P::Subpixel>>>
    {
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

/// Convert from a 2D array of `Pixel`s to a 3D array of `Subpixel`s
pub trait ArraySubpixelsExt<P>
where
    P: Pixel,
{
    fn to_subpixels(self) -> Array3<P::Subpixel>;
}

impl<P> ArraySubpixelsExt<P> for Array2<P>
where P: Pixel
{
    /// Convert an array of `Pixel`s to `Subpixel`s 
    fn to_subpixels(self) -> Array3<P::Subpixel> {
        let s = self.shape();
        let (w, h) = (s[0], s[1]);
        let c = P::CHANNEL_COUNT as usize;

        let data = self.into_raw_vec();

        // SAFETY: the input array of subpixels is contiguous in memory
        let subpixel_data = unsafe {
            let ptr = data.as_ptr() as *mut P;
            let length = data.len();
            let capacity = data.capacity();

            mem::transmute::<Vec<P>, Vec<P::Subpixel>>(Vec::from_raw_parts(ptr, length * c, capacity * c))
        };

        Array3::from_shape_vec((w, h, c), subpixel_data).unwrap()
    }
}

/// Convert from a 3D array of `Subpixel`s to a 2D array of `Pixel`s
pub trait ArrayPixelsExt<P>
where
    P: Pixel,
{
    fn to_pixels(self) -> Array2<P>;
}

impl<P> ArrayPixelsExt<P> for Array3<P::Subpixel>
where P: Pixel
{
    /// Convert an array of `Subpixel`s to `Pixel`s 
    fn to_pixels(self) -> Array2<P> {
        let s = self.shape();
        let (w, h) = (s[0], s[1]);
        let c = P::CHANNEL_COUNT as usize;

        let data = self.into_raw_vec();

        // SAFETY: the input array of pixels is contiguous in memory
        let pixel_data = unsafe {
            let ptr = data.as_ptr() as *mut P::Subpixel;
            let length = data.len();
            let capacity = data.capacity();

            mem::transmute::<Vec<P::Subpixel>, Vec<P>>(Vec::from_raw_parts(ptr, length.div_ceil(c), capacity.div_ceil(c)))
        };

        Array2::from_shape_vec((w, h), pixel_data).unwrap()
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
