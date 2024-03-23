use ndarray::Dim;

pub type DimN<const N: usize> = Dim<[usize; N]>;

/// The index of the data of a `Tile`, inside of the `TileSet`
pub type TileID = usize;
