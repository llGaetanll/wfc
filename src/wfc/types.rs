use super::traits::Hashable;

pub type Pixel = [u8; 3];
pub type BoundaryHash = u64;

impl Hashable for Pixel {
    /// Produce a non-colliding hash for a pixel
    fn hash(&self) -> u64 {
        self[0] as u64 * 256_u64 * 256_u64 + self[1] as u64 * 256_u64 + self[2] as u64
    }
}
