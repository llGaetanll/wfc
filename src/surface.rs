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
    use crate::ext::ndarray::NdIndex as WfcNdIndex;
    use crate::wave::Wave;

    use super::Surface;

    pub struct Torus;
    pub struct ProjectivePlane;
    pub struct KleinBottle;

    pub type TorusWave<Inner, Outer> = Wave<Inner, Outer, Torus, 2>;
    pub type ProjectiveWave<Inner, Outer> = Wave<Inner, Outer, ProjectivePlane, 2>;
    pub type KleinWave<Inner, Outer> = Wave<Inner, Outer, KleinBottle, 2>;

    impl Surface<2> for Torus {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
            todo!()
        }
    }

    impl Surface<2> for ProjectivePlane {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
            todo!()
        }
    }

    impl Surface<2> for KleinBottle {
        fn neighborhood(shape: [usize; 2], i: WfcNdIndex<2>) -> [[Option<WfcNdIndex<2>>; 2]; 2] {
            todo!()
        }
    }
}
