use std::fmt::Debug;
use std::hash::Hash;

use bit_set::BitSet;

use ndarray::Array;
use ndarray::Dimension;
use ndarray::NdIndex;
use ndarray::SliceArg;
use ndarray::SliceInfo;
use ndarray::SliceInfoElem;

use crate::tile::Tile;
use crate::types::BoundaryHash;
use crate::types::DimN;
use crate::wave::Wave;
use crate::wavetile::WaveTile;

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
