use std::array::from_fn;

use crate::ext::ndarray::NdIndex as WfcNdIndex;
use crate::wave::Wave;

pub struct Flat;

pub type FlatWave<Inner, Outer, const N: usize> = Wave<Inner, Outer, Flat, N>;

pub trait Surface<const N: usize> {
    fn neighborhood(shape: [usize; N], i: WfcNdIndex<N>) -> [[Option<WfcNdIndex<N>>; 2]; N];
}

impl<const N: usize> Surface<N> for Flat {
    fn neighborhood(shape: [usize; N], i: WfcNdIndex<N>) -> [[Option<WfcNdIndex<N>>; 2]; N] {
        from_fn(|axis| {
            let left = if i[axis] == 0 {
                None
            } else {
                let mut left = i;
                left[axis] -= 1;
                Some(left)
            };

            let right = if i[axis] == shape[axis] - 1 {
                None
            } else {
                let mut right = i;
                right[axis] += 1;
                Some(right)
            };

            [left, right]
        })
    }
}

#[cfg(feature = "wrapping")]
pub mod wrapping {
    use super::Surface;
    use crate::ext::ndarray::NdIndex as WfcNdIndex;
    use crate::wave::Wave;

    pub trait WrappingSurface<const N: usize> {
        fn neighborhood(shape: [usize; N], i: WfcNdIndex<N>) -> [[WfcNdIndex<N>; 2]; N];
    }

    impl<T, const N: usize> Surface<N> for T
    where
        T: WrappingSurface<N>,
    {
        fn neighborhood(shape: [usize; N], i: WfcNdIndex<N>) -> [[Option<WfcNdIndex<N>>; 2]; N] {
            Self::neighborhood(shape, i).map(|[l, r]| [Some(l), Some(r)])
        }
    }

    pub struct Torus;
    pub struct ProjectivePlane;
    pub struct KleinBottle;

    pub type TorusWave<Inner, Outer> = Wave<Inner, Outer, Torus, 2>;
    pub type ProjectiveWave<Inner, Outer> = Wave<Inner, Outer, ProjectivePlane, 2>;
    pub type KleinWave<Inner, Outer> = Wave<Inner, Outer, KleinBottle, 2>;

    impl WrappingSurface<2> for Torus {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[WfcNdIndex<2>; 2]; 2] {
            let [x, y] = i;
            let [n, m] = shape;

            [
                [[(x + n - 1) % n, y], [(x + 1) % n, y]],
                [[x, (y + m - 1) % m], [x, (y + 1) % m]],
            ]
        }
    }

    impl WrappingSurface<2> for ProjectivePlane {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[WfcNdIndex<2>; 2]; 2] {
            let [x, y] = i;
            let [n, m] = shape;

            let xs = if x == 0 {
                [[n - 1, m - y - 1], [1, y]]
            } else if x == n - 1 {
                [[x - 1, y], [0, m - y - 1]]
            } else {
                [[x - 1, y], [x + 1, y]]
            };

            let ys = if y == 0 {
                [[n - x - 1, m - 1], [x, 1]]
            } else if y == m - 1 {
                [[x, y - 1], [n - x - 1, 0]]
            } else {
                [[x, y - 1], [x, y + 1]]
            };

            [xs, ys]
        }
    }

    impl WrappingSurface<2> for KleinBottle {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[WfcNdIndex<2>; 2]; 2] {
            let [x, y] = i;
            let [n, m] = shape;

            let xs = [[(x + n - 1) % n, y], [(x + 1) % n, y]];

            let ys = if y == 0 {
                [[n - x - 1, m - 1], [x, y + 1]]
            } else if y == m - 1 {
                [[x, y - 1], [n - x - 1, 0]]
            } else {
                [[x, y - 1], [x, y + 1]]
            };

            [xs, ys]
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::KleinBottle;
    use crate::prelude::ProjectivePlane;
    use crate::prelude::Torus;

    use super::wrapping::WrappingSurface;

    const SHAPE: [usize; 2] = [3, 3];

    fn test_neighborhood<S: WrappingSurface<2>>(idx: [usize; 2], exp: [[[usize; 2]; 2]; 2]) {
        let res = S::neighborhood(SHAPE, idx);

        assert_eq!(exp, res)
    }

    #[test]
    fn torus() {
        test_neighborhood::<Torus>([0, 0], [[[2, 0], [1, 0]], [[0, 2], [0, 1]]]);
        test_neighborhood::<Torus>([0, 2], [[[2, 2], [1, 2]], [[0, 1], [0, 0]]]);
        test_neighborhood::<Torus>([1, 1], [[[0, 1], [2, 1]], [[1, 0], [1, 2]]]);
        test_neighborhood::<Torus>([2, 0], [[[1, 0], [0, 0]], [[2, 2], [2, 1]]]);
        test_neighborhood::<Torus>([0, 2], [[[2, 2], [1, 2]], [[0, 1], [0, 0]]]);
    }

    #[test]
    fn projective_plane() {
        test_neighborhood::<ProjectivePlane>([0, 0], [[[2, 2], [1, 0]], [[2, 2], [0, 1]]]);
        test_neighborhood::<ProjectivePlane>([0, 2], [[[2, 0], [1, 2]], [[0, 1], [2, 0]]]);
        test_neighborhood::<ProjectivePlane>([1, 1], [[[0, 1], [2, 1]], [[1, 0], [1, 2]]]);
        test_neighborhood::<ProjectivePlane>([2, 0], [[[1, 0], [0, 2]], [[0, 2], [2, 1]]]);
        test_neighborhood::<ProjectivePlane>([0, 2], [[[2, 0], [1, 2]], [[0, 1], [2, 0]]]);
    }

    #[test]
    fn klein_bottle() {
        test_neighborhood::<KleinBottle>([0, 0], [[[2, 0], [1, 0]], [[2, 2], [0, 1]]]);
        test_neighborhood::<KleinBottle>([0, 2], [[[2, 2], [1, 2]], [[0, 1], [2, 0]]]);
        test_neighborhood::<KleinBottle>([1, 1], [[[0, 1], [2, 1]], [[1, 0], [1, 2]]]);
        test_neighborhood::<KleinBottle>([2, 0], [[[1, 0], [0, 0]], [[0, 2], [2, 1]]]);
        test_neighborhood::<KleinBottle>([0, 2], [[[2, 2], [1, 2]], [[0, 1], [2, 0]]]);
    }
}
