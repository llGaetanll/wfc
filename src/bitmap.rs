use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;

use image::GenericImageView;
use image::Pixel as ImgPixel;

use log::debug;

use ndarray::Array;
use ndarray::Array3;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use crate::bitset::BitSet;
use crate::ext::ndarray as ndarray_ext;
use crate::tile::Tile;
use crate::tileset::TileSet;
use crate::types::BoundaryHash;
use crate::types::DimN;
use crate::types::Pixel;

use ndarray_ext::ArrayBoundaryHash;
use ndarray_ext::ArrayHash;
use ndarray_ext::ArrayTransformations;

pub struct BitMap<T, const N: usize>
where
    DimN<N>: Dimension,
{
    data: Box<Array<T, <DimN<N> as Dimension>::Larger>>,
    num_tiles: usize,
}

pub fn from_image(
    path: &Path,
    win_size: usize,
    with_rotations: bool,
    with_flips: bool,
) -> Result<BitMap<Pixel, 2>, String> {
    debug!(
        "Creating bitmap for image: {:?}. rots: {}, flips: {}",
        path, with_rotations, with_flips
    );

    // open the sample image
    let img = image::open(path).map_err(|e| e.to_string())?;
    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;

    let pixels: Array<Pixel, Ix2> = img
        .pixels()
        .map(|p| p.2.to_rgb().0.into())
        .collect::<Array<_, _>>()
        .into_shape((width, height))
        .unwrap();

    // create a square window of side length `win_size`
    let dim = [win_size; 2];

    // complete list of unique tiles
    let mut tiles: HashMap<u64, Array<Pixel, Ix2>> = HashMap::new();

    // TODO: allow rotations of flips and flips of rotations
    for window in pixels.windows(dim) {
        if with_flips {
            let flips = window
                .flips()
                .into_iter()
                .map(|tile| (ArrayHash::hash(&tile), tile));

            tiles.extend(flips);
        }

        if with_rotations {
            let rotations = window
                .rotations()
                .into_iter()
                .map(|tile| (ArrayHash::hash(&tile), tile));

            tiles.extend(rotations);
        }
    }

    let num_tiles = tiles.len();

    debug!("Bitmap dims: {num_tiles} x {win_size} x {win_size}");

    // Create a bitmap from the tiles array.
    let bitmap = Array3::from_shape_vec(
        (num_tiles, win_size, win_size),
        tiles.into_values().flatten().collect(),
    )
    .map_err(|e| e.to_string())?;

    Ok(BitMap {
        data: Box::new(bitmap), // TODO: check if an array is already on the heap
        num_tiles,
    })
}

impl<T, const N: usize> BitMap<T, N>
where
    T: Hash,
    DimN<N>: Dimension,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    pub fn tile_set<'a>(&'a self) -> TileSet<'a, T, N>
    where
        // we need this bound because our bitmap's dimension is larger than the tiles so when we
        // extract the tiles out, the dimension lowers again. This bound is effectively trivial but
        // the type checker can't know this
        <DimN<N> as Dimension>::Larger: Dimension<Smaller = DimN<N>>,
    {
        // construct a list of unique hashes each with an index accessible in constant time.
        let mut hash_index: usize = 0;
        let mut unique_hashes: HashMap<BoundaryHash, usize> = HashMap::new();
        let mut tile_hashes = Vec::with_capacity(self.num_tiles);

        // iterate through all the tiles in the bitmap
        for tile in self.data.axis_iter(Axis(0)) {
            let hashes = tile.boundary_hashes();
            tile_hashes.push(hashes);

            // add all unique hashes to an index map
            for &hash in hashes.iter().flat_map(|h| h.iter()) {
                if let Entry::Vacant(v) = unique_hashes.entry(hash) {
                    v.insert(hash_index);
                    hash_index += 1;
                }
            }
        }

        // transform the tiles into a list of pairs
        let num_hashes = unique_hashes.len();

        // debug!("{} unique hashes", num_hashes);
        debug!("Creating bitsets of size {}", 2 * N * num_hashes);

        let tiles: Vec<Tile<'a, T, N>> = self
            .data
            .axis_iter(Axis(0))
            .zip(tile_hashes.iter())
            .map(|(view, hashes)| {
                let mut tile_hashes = BitSet::zeros(2 * N * num_hashes);

                // This is where we define the mapping in our tile's bitset. Some explanation is in
                // order.
                //
                // Suppose we have a 2D Tile. Then, we have two axes and `N = 2`. For each axis, we
                // have a left and right side (and so we have a left and right hash) which
                // corresponds to a unique hash number in `unique_hashes`.
                //
                // This is the layout of the tile's bitset:
                //
                //   [Axis 1 left] | [Axis 1 right] | [Axis 2 left] | [Axis 2 right]
                //
                // where each block represents `num_hashes` bits, and only one bit in each block is
                // flipped on. Namely, this is `index_left` for the left blocks, and `index_right`
                // for the right blocks.
                for (i, [left, right]) in hashes.iter().enumerate() {
                    let index_left = unique_hashes[left];
                    let index_right = unique_hashes[right];

                    tile_hashes.on((2 * i) * num_hashes + index_left);
                    tile_hashes.on((2 * i + 1) * num_hashes + index_right);
                }

                Tile::new(view, tile_hashes)
            })
            .collect();

        let hashes: Vec<BoundaryHash> = unique_hashes.into_keys().collect();

        TileSet { hashes, tiles }
    }
}
