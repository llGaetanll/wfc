use ndarray::Dim;
use ndarray::Array;

pub type DimN<const N: usize> = Dim<[usize; N]>;

pub struct Cache<Inner, const N: usize> {
    pub entropies: Array<usize, DimN<N>>,
    pub cache: Array<Inner, DimN<N>>
}
