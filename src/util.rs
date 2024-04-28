use crate::ext::ndarray::NdIndex as WfcNdIndex;

/// Takes `&mut [T]` and a predicate `P: FnMut(&T) -> bool` and partitions the list according to
/// the predicate. Swaps elements of the list such that all elements satisfying `P` appear before
/// any element not satisfying `P`.
///
/// The index `i` returned by the function always points at the first element for which `P` is false.
/// Note that if `P` is trivial, then `i = |list|` points outside the list.
pub fn partition_in_place<T, P>(list: &mut [T], mut predicate: P) -> usize
where
    P: FnMut(&T) -> bool,
{
    if list.is_empty() {
        return 0;
    }

    let (mut lo, mut hi) = (0, list.len() - 1);

    while lo < hi {
        if predicate(&list[lo]) {
            lo += 1;
            continue;
        }

        if !predicate(&list[hi]) {
            hi -= 1;
            continue;
        }

        list.swap(lo, hi);
        lo += 1;
        hi -= 1;
    }

    if predicate(&list[lo]) {
        lo + 1
    } else {
        lo
    }
}

/// Compute the manhattan distance between two points
pub fn manhattan_dist<const N: usize>(a: WfcNdIndex<N>, b: WfcNdIndex<N>) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| ((a as isize) - (b as isize)).unsigned_abs())
        .sum()
}
