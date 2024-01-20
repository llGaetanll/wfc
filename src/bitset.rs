use core::fmt::Debug;

/// Wrapper in case ever needs to be changed
pub type BitsType = Vec<usize>;

pub const WORD_SIZE: usize = std::mem::size_of::<usize>() * 8;

/// A Bitset representation, specialized for wfc
#[repr(transparent)]
#[derive(Default, Clone)]
pub struct BitSet {
    bits: BitsType,
}

fn create_mask(capacity: usize, size: usize, offset: usize) -> BitSet {
    let a = offset / WORD_SIZE;
    let b = (offset + size) / WORD_SIZE;

    let start_bit = offset % WORD_SIZE;
    let end_bit = (offset + size) % WORD_SIZE;

    let mut mask: Vec<usize> = vec![0usize; capacity];

    for chunk in mask.iter_mut().take(b + 1).skip(a) {
        *chunk = usize::MAX;
    }

    mask[a] >>= start_bit;
    mask[b] &= usize::MAX
        .checked_shl((WORD_SIZE - end_bit) as u32)
        .unwrap_or(0);

    BitSet { bits: mask }
}

// TODO: allow operations to return operators for easy chaining of operations
impl BitSet {
    /// Initialize a new, empty `BitSet`
    pub fn new() -> Self {
        Self::default()
    }

    /// Initialize a new `BitSet` with a given `capacity`.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bits: Vec::with_capacity(capacity),
        }
    }

    /// Initialize a new BitSet of a given `capacity`, and fill it with zeros. The actual capacity
    /// allocated is the smallest `usize` multiple larger than `capacity`.
    pub fn zeros(capacity: usize) -> Self {
        let n = (capacity + WORD_SIZE - 1) / WORD_SIZE;

        Self {
            bits: vec![usize::MIN; n],
        }
    }

    /// Initialize a new BitSet of a given `capacity`, and fill it with ones. The actual capacity
    /// allocated is the smallest `usize` multiple larger than `capacity`.
    pub fn ones(capacity: usize) -> Self {
        let n = (capacity + WORD_SIZE - 1) / WORD_SIZE;

        Self {
            bits: vec![usize::MAX; n],
        }
    }

    /// Sets the bit at the given `index` on.
    /// Returns `&mut self` to allow for operation chaining.
    pub fn on(&mut self, index: usize) -> &mut Self {
        let i = index / WORD_SIZE;
        let j = index % WORD_SIZE;

        let mask = 1usize << (WORD_SIZE - j - 1);
        self.bits[i] |= mask;

        self
    }

    /// Sets the bit at the given `index` off.
    /// Returns `&mut self` to allow for operation chaining.
    pub fn off(&mut self, index: usize) -> &mut Self {
        let i = index / WORD_SIZE;
        let j = index % WORD_SIZE;

        let mask = 1usize << (WORD_SIZE - j - 1);
        self.bits[i] &= !mask;

        self
    }

    /// Compute the intersection of two bitsets while mutating `self`.
    /// Returns `&mut self` to allow for operation chaining.
    pub fn intersect(&mut self, other: &Self) -> &mut Self {
        self.bits
            .iter_mut()
            .zip(other.bits.iter())
            .for_each(|(l, r)| *l &= r);

        self
    }

    /// Compute the union of two bitsets while mutating `self`.
    /// Returns `&mut self` to allow for operation chaining.
    pub fn union(&mut self, other: &Self) -> &mut Self {
        self.bits
            .iter_mut()
            .zip(other.bits.iter())
            .for_each(|(l, r)| *l |= r);

        self
    }

    /// Computes whether `self` is a subset of `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .all(|(l, r)| *l & *r == *l)
    }

    /// Computes whether `self` is a superset of `other`.
    pub fn is_superset(&self, other: &Self) -> bool {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .all(|(l, r)| *l & *r == *r)
    }

    /// Create a contiguous mask of `self`. Note that `offset + size` should never exceed the size
    /// of `self`.
    pub fn mask(&self, offset: usize, size: usize) -> BitSet {
        let n = self.bits.len();

        assert!(
            n <= size + offset,
            "tried to mask with size: {size}, offset: {offset}, but length is {n}"
        );

        let mut mask = create_mask(n, size, offset);
        mask.intersect(self);

        mask
    }

    /// Zeros out the bitset, effectively resetting it.
    pub fn zero(&mut self) {
        self.bits.fill(0usize);
    }

    /// Negates `self`, producing a copy.
    pub fn negate(&self) -> Self {
        Self {
            bits: self.bits.iter().map(|n| !n).collect(),
        }
    }

    /// Rotates `self` left by `nbits`.
    pub fn rotate_left(&mut self, nbits: usize) {
        let rem = nbits % WORD_SIZE;
        let rot = nbits / WORD_SIZE;
        let rot_ceil = (nbits + WORD_SIZE - 1) / WORD_SIZE;

        let mask: usize = !(1usize.rotate_left((WORD_SIZE - rem) as u32) - 1);

        let mut rem_bits: Vec<usize> = self
            .bits
            .iter()
            .map(|word| (*word & mask).rotate_left(rem as u32))
            .collect();

        rem_bits.rotate_left(rot_ceil);

        self.bits.iter_mut().for_each(|word| *word <<= nbits);
        self.bits.rotate_left(rot);
        self.bits
            .iter_mut()
            .zip(rem_bits)
            .for_each(|(word, rem)| *word |= rem);
    }

    /// Rotates `self` right by `nbits`.
    pub fn rotate_right(&mut self, nbits: usize) {
        let rem = nbits % WORD_SIZE;
        let rot = nbits / WORD_SIZE;
        let rot_ceil = (nbits + WORD_SIZE - 1) / WORD_SIZE;

        let mask: usize = 1usize.rotate_left(rem as u32) - 1;

        let mut rem_bits: Vec<usize> = self
            .bits
            .iter()
            .map(|word| (*word & mask).rotate_right(rem as u32))
            .collect();

        rem_bits.rotate_right(rot_ceil);

        self.bits.iter_mut().for_each(|word| *word >>= nbits);
        self.bits.rotate_right(rot);
        self.bits
            .iter_mut()
            .zip(rem_bits)
            .for_each(|(word, rem)| *word |= rem);
    }
}

impl Debug for BitSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for n in &self.bits {
            write!(f, "{:064b}", *n)?;
        }

        Ok(())
    }
}

/// Functions which don't really belong with the more elementary ones above
impl BitSet {
    pub fn axis_mask(&self, num_hashes: usize) -> Self {
        if num_hashes <= WORD_SIZE {
            self.axis_mask_small(num_hashes)
        } else {
            self.axis_mask_large(num_hashes)
        }
    }

    fn axis_mask_small(&self, num_hashes: usize) -> Self {
        let mut bit_pattern = 0usize;

        let num_patterns = (WORD_SIZE + 2 * num_hashes - 1) / (2 * num_hashes);
        for _ in 0..num_patterns {
            bit_pattern = bit_pattern
                .checked_shr(2u32 * num_hashes as u32)
                .unwrap_or(0);

            bit_pattern |= usize::MAX
                .checked_shl((WORD_SIZE - num_hashes) as u32)
                .unwrap_or(0);
        }

        let num_words = self.bits.len();
        let mut mask = vec![bit_pattern; num_words];

        for (i, word) in mask.iter_mut().enumerate() {
            let lshift = (i * WORD_SIZE) % (2 * num_hashes);

            *word = word.checked_shl(lshift as u32).unwrap_or(0);

            *word |= bit_pattern
                .checked_shr((2 * num_hashes - lshift) as u32)
                .unwrap_or(0);
        }

        Self { bits: mask }
    }

    fn axis_mask_large(&self, num_hashes: usize) -> BitSet {
        let num_words = self.bits.len();

        let mut mask = vec![0usize; num_words];
        let num_bits = num_words * WORD_SIZE;

        let k = (num_bits + 2 * num_hashes - 1) / (2 * num_hashes);

        for i in 0..k {
            let start_bit_number = 2 * num_hashes * i;
            let end_bit_number = 2 * num_hashes * i + num_hashes;

            let start_bit_index = start_bit_number / WORD_SIZE;
            let start_bit_offset = start_bit_number % WORD_SIZE;

            let end_bit_index = end_bit_number / WORD_SIZE;
            let end_bit_offset = end_bit_number % WORD_SIZE;

            mask[start_bit_index] = usize::MAX.checked_shr(start_bit_offset as u32).unwrap_or(0);

            // FIXME: this check wont be necessary for our values
            if end_bit_index < num_words {
                mask[end_bit_index] = usize::MAX
                    .checked_shl((WORD_SIZE - end_bit_offset) as u32)
                    .unwrap_or(0);
            }

            // fill in any completely full words
            mask.iter_mut()
                .take(end_bit_index.min(num_words)) // FIXME: remove this min too
                .skip(start_bit_index + 1)
                .for_each(|word| *word = usize::MAX);
        }

        BitSet { bits: mask }
    }
}

impl BitSet {
    pub fn swap_axes(&mut self, num_hashes: usize) {
        let mut left_mask = Self::axis_mask(self, num_hashes);
        let right_mask = left_mask.negate();

        left_mask.intersect(self);
        self.intersect(&right_mask);

        left_mask.rotate_right(num_hashes);
        self.rotate_left(num_hashes);

        self.union(&left_mask);
    }
}
