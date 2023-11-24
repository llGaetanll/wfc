use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::path::Path;

use bit_set::BitSet;

use image::GenericImageView;
use image::Pixel as ImgPixel;

use ndarray::Array;
use ndarray::Array3;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::NdIndex;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use super::tile::Tile;
use super::traits::TileArrayBoundaryHashExt;
use super::traits::TileArrayHashExt;
use super::traits::TileArrayTransformationsExt;
use super::types::BoundaryHash;
use super::types::DimN;
use super::types::Pixel;
use super::wave::Wave;
use super::wavetile::WaveTile;

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
    // open the sample image
    let img = image::open(path).map_err(|e| e.to_string())?;
    let (width, height) = img.dimensions();
    let width = width as usize;
    let height = height as usize;

    let pixels: Array<Pixel, Ix2> = img
        .pixels()
        .map(|p| p.2.to_rgb().0.into())
        .collect::<Array<_, _>>()
        .into_shape((width as usize, height as usize))
        .unwrap();

    // create a cubic window of side length `size`
    let dim = [win_size; 2];

    // complete list of unique tiles
    let mut tiles: HashMap<u64, Array<Pixel, Ix2>> = HashMap::new();

    for window in pixels.windows(dim).into_iter() {
        // if with_flips {
        //     let flips = tile_flips(window)
        //         .into_iter()
        //         .map(|tile| (hash(&tile), tile));
        //
        //     tiles.extend(rotations);
        // }

        if with_rotations {
            let rotations = window
                .rotations()
                .into_iter()
                .map(|tile| (TileArrayHashExt::hash(&tile.view()), tile));

            tiles.extend(rotations);
        }
    }

    let num_tiles = tiles.len();

    println!("Created a {num_tiles} x {win_size} x {win_size} bitmap");

    // Create a bitmap from the tiles array.
    let bitmap = Array3::from_shape_vec(
        (num_tiles, win_size, win_size),
        tiles.into_values().flatten().collect(),
    )
    .map_err(|e| e.to_string())?;

    Ok(BitMap {
        data: Box::new(bitmap),
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
                match unique_hashes.entry(hash) {
                    Entry::Vacant(v) => {
                        v.insert(hash_index);
                        hash_index += 1;
                    }
                    _ => {}
                }
            }
        }

        // transform the tiles into a list of pairs
        let num_hashes = unique_hashes.len();
        let tiles: Vec<Tile<'a, T, N>> = self
            .data
            .axis_iter(Axis(0))
            .zip(tile_hashes.iter())
            .map(|(view, hashes)| {
                let mut hash_index: [[BitSet; 2]; N] =
                    vec![
                        vec![BitSet::with_capacity(num_hashes); 2]
                            .try_into()
                            .unwrap();
                        2
                    ]
                    .try_into()
                    .unwrap();

                let it = hashes.iter().zip(hash_index.iter_mut());
                for ([hash_left, hash_right], [bset_left, bset_right]) in it {
                    bset_left.insert(unique_hashes[hash_left]);
                    bset_right.insert(unique_hashes[hash_right]);
                }

                Tile::new(view, hash_index)
            })
            .collect();

        let hashes: Vec<BoundaryHash> = unique_hashes.into_keys().collect();

        TileSet { hashes, tiles }
    }
}

pub struct TileSet<'a, T, const N: usize>
where
    T: Hash,
    DimN<N>: Dimension,
{
    // all unique hashes. Order matters
    pub hashes: Vec<BoundaryHash>,

    // tiles are views into the bitmap
    pub tiles: Vec<Tile<'a, T, N>>,
}

impl<'a, T, const N: usize> TileSet<'a, T, N>
where
    T: Hash + Send + Sync + Clone + Debug,
    DimN<N>: Dimension,
    [usize; N]: NdIndex<DimN<N>>,

    SliceInfo<Vec<SliceInfoElem>, DimN<N>, <DimN<N> as Dimension>::Smaller>: SliceArg<DimN<N>>,
{
    pub fn wave(&'a self, shape: DimN<N>) -> Wave<'a, T, N> {
        let num_hashes = self.hashes.len();

        // computes the complete OR of all the boundarysets for each side of each axis in each
        // dimension
        let wavetile_hashes = self.tiles.iter()
            .map(|tile| &tile.hashes)
            .fold(
            {
                let init: [[BitSet; 2]; N] = vec![
                    vec![BitSet::with_capacity(num_hashes); 2]
                        .try_into()
                        .unwrap();
                    N
                ]
                .try_into()
                .unwrap();

                init
            },
            |acc, bset| {
                acc.iter()
                    .zip(bset.iter())
                    // TODO: inneficient?
                    .map(|([acc_l, acc_r], [l, r])| {
                        [
                            BitSet::from_iter(acc_l.union(&l)),
                            BitSet::from_iter(acc_r.union(&r)),
                        ]
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            },
        );

        let tile_refs: Vec<&'a Tile<'a, T, N>> = self.tiles.iter().map(|tile| tile).collect();

        Wave {
            wave: Array::from_shape_fn(shape, |_| {
                WaveTile::new(tile_refs.clone(), wavetile_hashes.clone(), num_hashes)
            }),
            diffs: Array::from_shape_fn(shape, |_| false),
            shape,
        }
    }
}
