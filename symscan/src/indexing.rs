/// Index stored in the low half of a [`VariantIndexPair`].
///
/// Only [`u32`] (within / cached) and [`CrossIndex`] (across) are used.
pub trait VariantIndex: Copy + Send + Sync {
    /// Bits stored in the low half of a [`VariantIndexPair`] (for [`CrossIndex`], includes the type bit).
    fn index_bits(self) -> u32;

    fn from_index_bits(bits: u32) -> Self;

    /// String index with any type tag cleared — what goes into convergent-index output.
    fn string_index(self) -> u32;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CrossIndex(u32);

impl CrossIndex {
    const TYPE_MASK: u32 = 1 << 31;
    const VALUE_MASK: u32 = !Self::TYPE_MASK;
    pub const MAX: usize = (1 << 31) - 1;

    pub fn from(value: u32, is_ref: bool) -> Self {
        debug_assert_ne!(value & Self::TYPE_MASK, Self::TYPE_MASK);

        if is_ref {
            Self(value | Self::TYPE_MASK)
        } else {
            Self(value)
        }
    }

    pub fn is_ref(&self) -> bool {
        self.0 & Self::TYPE_MASK == Self::TYPE_MASK
    }

    pub fn get_value(&self) -> u32 {
        self.0 & Self::VALUE_MASK
    }

    pub fn bits(self) -> u32 {
        self.0
    }

    pub fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
}

impl VariantIndex for CrossIndex {
    #[inline(always)]
    fn index_bits(self) -> u32 {
        self.bits()
    }

    #[inline(always)]
    fn from_index_bits(bits: u32) -> Self {
        CrossIndex::from_bits(bits)
    }

    #[inline(always)]
    fn string_index(self) -> u32 {
        self.get_value()
    }
}

impl VariantIndex for u32 {
    #[inline(always)]
    fn index_bits(self) -> u32 {
        self
    }

    #[inline(always)]
    fn from_index_bits(bits: u32) -> Self {
        bits
    }

    #[inline(always)]
    fn string_index(self) -> u32 {
        self
    }
}

/// One VIP array element: a 32-bit variant hash in the high half and a packed index
/// ([`u32`] or [`CrossIndex`]) in the low half.
///
/// Derived [`Ord`] depends on the hash living in the high half so that sorting by the
/// packed word groups equal hashes together (and, for [`CrossIndex`], keeps query
/// indices before reference indices within a group). The high half is a 32-bit foldhash
/// truncation; collisions only ever add candidates that the distance check already filters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct VariantIndexPair(u64);

impl VariantIndexPair {
    pub const NUM_BUCKETS: usize = 1 << Self::BUCKET_BITS;

    const BUCKET_BITS: u32 = 8;
    const BUCKET_SHIFT: u32 = u64::BITS - Self::BUCKET_BITS;

    #[inline(always)]
    pub fn new(variant_hash: u32, index_bits: u32) -> Self {
        Self(((variant_hash as u64) << 32) | index_bits as u64)
    }

    #[inline(always)]
    pub fn from_index(variant_hash: u32, index: impl VariantIndex) -> Self {
        Self::new(variant_hash, index.index_bits())
    }

    /// High half: 32-bit foldhash truncation of the deletion variant.
    #[inline(always)]
    pub fn variant_hash(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Low half: packed index bits ([`u32`] verbatim, or [`CrossIndex`] including its type bit).
    #[inline(always)]
    pub fn index_bits(self) -> u32 {
        self.0 as u32
    }

    /// Top [`Self::BUCKET_BITS`] of the variant hash — the MSD radix bucket.
    #[inline(always)]
    pub fn bucket(self) -> usize {
        (self.0 >> Self::BUCKET_SHIFT) as usize
    }
}
