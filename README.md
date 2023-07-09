## Types

```rs
pub struct Tile<'a, T, D> where T: Hashable,
{
    /// The data of the Tile. Note that the tile does not own its data.
    data: ArrayView<'a, T, D>,

    /// The hash of the current tile. This is computed from the Type that the
    /// tile references. If two tiles have the same data, they will have
    /// the same hash, no matter the type.
    id: TileHash,

    /// The hash of each side of the tile.
    /// Each tuple represents opposite sides on an axis of the tile.
    hashes: Vec<(BoundaryHash, BoundaryHash)>,
}

pub struct WaveTile<'a, T, D>
where
    T: Hashable,
{
    /// The list of possible tiles that the WaveTile can be
    ///
    /// For each tile, we store a unsigned integer which is initialized as 0. If
    /// a tile is no longer possible, this number is incremented to 1. In every
    /// subsequent pass, if a number i > 0, it is again incremented. this allows
    /// us to reverse the operation.
    possible_tiles: Vec<(Rc<Tile<'a, T, D>>, usize)>,

    /// The shape of the WaveTile
    shape: usize,
}

pub struct Wave<'a, T, D>
where
    T: Hashable,
{
    // we keep an array of RefCell WaveTiles in order to interiorly mutate the possible tiles of each WaveTiles
    pub wave: Array<RefCell<WaveTile<'a, T, D>>, D>,
}
```
