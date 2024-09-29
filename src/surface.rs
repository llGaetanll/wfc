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

    pub struct Torus;
    pub struct ProjectivePlane;
    pub struct KleinBottle;

    pub type TorusWave<Inner, Outer> = Wave<Inner, Outer, Torus, 2>;
    pub type ProjectiveWave<Inner, Outer> = Wave<Inner, Outer, ProjectivePlane, 2>;
    pub type KleinWave<Inner, Outer> = Wave<Inner, Outer, KleinBottle, 2>;

    impl Surface<2> for Torus {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
            let [x, y] = i;
            let [n, m] = shape;

            [
                [Some([(x + n - 1) % n, y]), Some([(x + 1) % n, y])],
                [Some([x, (y + m - 1) % m]), Some([x, (y + 1) % m])],
            ]
        }
    }

    impl Surface<2> for ProjectivePlane {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
            from_wrap_info([[false, true], [true, false]], shape, i)
        }
    }

    impl Surface<2> for KleinBottle {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
            from_wrap_info([[false, true], [false, false]], shape, i)
        }
    }

    #[inline]
    fn from_wrap_info(
        info: [[bool; 2]; 2],
        shape: [usize; 2],
        i: WfcNdIndex<2>,
    ) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
        info.iter()
            .enumerate()
            .map(|(axis, &[l, r])| [wrap(axis, shape, i, -1, l), wrap(axis, shape, i, 1, r)])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[inline]
    fn wrap<const N: usize>(
        axis: usize,
        shape: [usize; N],
        i: WfcNdIndex<N>,
        d: isize,
        rev: bool,
    ) -> Option<WfcNdIndex<N>> {
        let mut index = i;
        let n = shape[axis];

        index[axis] = (index[axis] as isize + d + n as isize) as usize % n;

        if rev && !(0isize..n as isize).contains(&(index[axis] as isize + d)) {
            index[axis] = n - index[axis];
        }

        Some(index)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::Torus;

    use super::Surface;

    const SHAPE: [usize; 2] = [3, 3];

    fn unwrap_res(res: [[Option<[usize; 2]>; 2]; 2]) -> [[[usize; 2]; 2]; 2] {
        res.map(|[l, r]| [l.unwrap(), r.unwrap()])
    }

    fn test_neighborhood<S: Surface<2>>(idx: [usize; 2], exp: [[[usize; 2]; 2]; 2]) {
        let res = S::neighborhood(SHAPE, idx);

        assert_eq!(exp, unwrap_res(res))
    }

    #[test]
    fn torus() {
        test_neighborhood::<Torus>([0, 0], [[[2, 0], [1, 0]], [[0, 2], [0, 1]]]);
        test_neighborhood::<Torus>([0, 2], [[[2, 2], [1, 2]], [[0, 1], [0, 0]]]);
        test_neighborhood::<Torus>([1, 1], [[[0, 1], [2, 1]], [[1, 0], [1, 2]]]);
        test_neighborhood::<Torus>([2, 0], [[[1, 0], [0, 0]], [[2, 2], [2, 1]]]);
        test_neighborhood::<Torus>([0, 2], [[[2, 2], [1, 2]], [[0, 1], [0, 0]]]);
    }
}
