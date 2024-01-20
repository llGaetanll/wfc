use std::fmt::Debug;
use std::hash::Hash;

use ndarray::Array2;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use rand::Rng;

use crate::bitset::BitSet;
use crate::tile::Tile;
use crate::traits::Pixel;
use crate::traits::SdlTexture;
use crate::types;
use crate::types::DimN;

/// A `WaveTile` is a list of `Tile`s in superposition
/// `T` is the type of each element of the tile
/// `D` is the dimension of each tile
pub struct WaveTile<'a, T, const N: usize>
where
    T: Hash,
    Dim<[usize; N]>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    possible_tiles: Vec<&'a Tile<'a, T, N>>,
    filtered_tiles: Vec<&'a Tile<'a, T, N>>,
    filtered_tile_indices: Vec<usize>,

    num_hashes: usize,
    pub hashes: BitSet,

    /// An optional pointer to the bitset of each neighbor. We need `Option` because a WaveTile on
    /// the edge of the wave may not have all of its neighbors.
    pub neighbor_hashes: [[Option<*const BitSet>; 2]; N],

    /// computed as the number of valid tiles
    pub entropy: usize,
    pub shape: usize,

    // either 0 or 1
    pub parity: usize,
}

impl<'a, T, const N: usize> WaveTile<'a, T, N>
where
    T: Hash + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    /// Create a new `WaveTile` from a list of tiles
    pub fn new(
        tiles: Vec<&'a Tile<'a, T, N>>,
        hashes: BitSet,
        num_hashes: usize,
        shape: usize,
        parity: usize,
    ) -> Self {
        let entropy = tiles.len();

        WaveTile {
            possible_tiles: tiles,
            // avoid reallocs at runtime
            filtered_tiles: Vec::with_capacity(entropy),
            filtered_tile_indices: Vec::with_capacity(entropy),
            num_hashes,
            hashes,
            neighbor_hashes: [[None; 2]; N],
            entropy,
            shape,
            parity,
        }
    }

    /// Collapses a `WaveTile` to one of its possible tiles, at random.
    pub fn collapse(&mut self) -> Option<()> {
        if self.entropy < 1 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.entropy);

        let filtered_tiles = self
            .possible_tiles
            .iter()
            .enumerate()
            .filter_map(|(i, tile)| if i != idx { Some(tile) } else { None });

        self.filtered_tile_indices.push(self.filtered_tiles.len());
        self.filtered_tiles.extend(filtered_tiles);

        let tile: &Tile<'a, T, N> = self.possible_tiles[idx];
        self.possible_tiles.clear();
        self.possible_tiles.push(tile);

        self.update_hashes();

        self.entropy = 1;

        Some(())
    }

    /// Update the `possible_tiles` of the current `WaveTile`
    pub fn update(&mut self) {
        if self.entropy < 2 {
            return;
        }

        let hashes: BitSet = BitSet::new();

        // owned copy of the self.hashes
        let mut hashes = std::mem::replace(&mut self.hashes, hashes);

        // prepare the correct wavetile hash.
        let mut alt_wavetile_bitset = BitSet::zeros(2 * N * self.num_hashes);

        let odd = self.parity;
        let even = 1 - self.parity;

        for (axis, [left, right]) in self.neighbor_hashes.iter().enumerate() {
            let right_hashes = match right {
                Some(hashes) => {
                    let hashes = unsafe { &**hashes };
                    hashes.mask((2 * axis + even) * self.num_hashes, self.num_hashes)
                }
                None => {
                    let ones = BitSet::ones(2 * N * self.num_hashes);
                    ones.mask((2 * axis + even) * self.num_hashes, self.num_hashes)
                }
            };

            let left_hashes = match left {
                Some(hashes) => {
                    let hashes = unsafe { &**hashes };
                    hashes.mask((2 * axis + odd) * self.num_hashes, self.num_hashes)
                }
                None => {
                    let ones = BitSet::ones(2 * N * self.num_hashes);
                    ones.mask((2 * axis + odd) * self.num_hashes, self.num_hashes)
                }
            };

            alt_wavetile_bitset.union(&right_hashes).union(&left_hashes);
        }

        hashes.intersect(&alt_wavetile_bitset);

        // new hashes are computed
        self.hashes = hashes;

        // debug!("wavetile hashes after intersection:\n\t{:?}", self.hashes);

        let filtered_tiles = self
            .possible_tiles
            .iter()
            .filter(|tile| !tile.hashes.is_subset(&self.hashes));

        self.filtered_tile_indices.push(self.filtered_tiles.len());
        self.filtered_tiles.extend(filtered_tiles);
        self.possible_tiles
            .retain(|tile| tile.hashes.is_subset(&self.hashes));

        self.entropy = self.possible_tiles.len();

        // NOTE: mutates self's hashes
        self.update_hashes();
    }

    /// Given a list of `tile`s that this WaveTile can be, this precomputes the list of valid
    /// hashes for each of its borders. This is used to speed up the wave propagation algorithm.
    fn update_hashes(&mut self) {
        // reset own bitsets
        self.hashes.zero();

        // new bitsets is union of possible tiles
        for tile in &self.possible_tiles {
            self.hashes.union(&tile.hashes);
        }
    }
}

impl<'a> Pixel for WaveTile<'a, types::Pixel, 2> {
    fn dims(&self) -> (usize, usize) {
        // wavetiles are squares
        (self.shape, self.shape)
    }

    fn pixels(&self) -> Array2<types::Pixel> {
        // notice that a single number represents the size of the tile, no
        // matter the dimension. This is because it is enforced that all axes of
        // the tile be the same size.
        let size = self.shape;
        let num_tiles = self.entropy;

        // merge all tiles into a single one, channel-wise
        let mut pixels: Vec<[f64; 3]> = vec![[0., 0., 0.]; size * size];
        for tile in &self.possible_tiles {
            let tile_pixels = tile.pixels();

            for (pixel, tile_pixel) in pixels.iter_mut().zip(tile_pixels.into_iter()) {
                for (acc_c, c) in pixel.iter_mut().zip(tile_pixel.into_iter()) {
                    // since division distributes over addition we can do the division in here and
                    // avoid overflow at the cost of a higher (although negligible) floating point
                    // error.
                    *acc_c += (c as f64) / (num_tiles as f64);
                }
            }
        }

        let as_u8 = |px: [f64; 3]| -> types::Pixel {
            let pixel: [u8; 3] = px
                .into_iter()
                .map(|c| c as u8)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            pixel.into()
        };

        // convert back into u8
        let pixels: Vec<types::Pixel> = pixels.into_iter().map(as_u8).collect();

        Array2::from_shape_vec((size, size), pixels).expect("WaveTile pixel conversion failed")
    }
}

// use default implementation
impl<'a> SdlTexture for WaveTile<'a, types::Pixel, 2> {}

impl<'a> crate::out::img::Image for WaveTile<'a, types::Pixel, 2> {}

impl<'a, T, const N: usize> Debug for WaveTile<'a, T, N>
where
    T: Hash + std::fmt::Debug,
    Dim<[usize; N]>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, Dim<[usize; N]>, <Dim<[usize; N]> as Dimension>::Smaller>:
        SliceArg<Dim<[usize; N]>>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.possible_tiles)
    }
}
