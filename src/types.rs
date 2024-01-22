use ndarray::Dim;

pub type DimN<const N: usize> = Dim<[usize; N]>;

pub type TileHash = u64;
pub type BoundaryHash = u64;
