//! SymScan enables extremely fast discovery of pairs of similar strings within and across large
//! collections.
//!
//! SymScan is a variation on the [symmetric deletion
//! ](https://seekstorm.com/blog/1000x-spelling-correction/) algorithm that is optimised for
//! bulk-searching similar strings within one or across two large string collections at once (e.g.
//! searching for similar protein sequences among a collection of 10M). The key algorithmic
//! difference between SymScan and traditional symmetric deletion is the use of a [sort-merge
//! join](https://en.wikipedia.org/wiki/Sort-merge_join) approach in place of hashmaps to discover
//! input strings that share common deletion variants. This sort-and-scan approach trades off an
//! additional factor of O(log N) (with N the total number of strings being compared) in expected
//! time complexity for improved cache locality and effective parallelization, and ends up being
//! much faster for the above use case. Parallelization is handled using the
//! [rayon](https://docs.rs/rayon/latest/rayon/) crate internally.
//!
//! SymScan provides separate implementations for [Levenshtein edit
//! distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and [Hamming
//! distance](https://en.wikipedia.org/wiki/Hamming_distance). See [`get_neighbors_within`] /
//! [`get_hamming_neighbors_within`] and [`get_neighbors_across`] / [`get_hamming_neighbors_across`]
//! for details on the API.
//!
//! Even for our intended use case of discovering pairs of similar strings from large collections,
//! it is sometimes useful to memoize the deletion variant computations for at least one side of the
//! query (e.g. reference-side memoization when making repeated queries against a very large
//! reference collection with relatively smaller query collections). For such cases, the library
//! also provides the [`CachedRef`] / [`CachedRefHamming`] structs.

use foldhash::fast::FixedState;
use hashbrown::HashMap;
use rapidfuzz::distance::{hamming, levenshtein};
use rayon::prelude::*;
use std::fmt::Display;
use std::hash::{BuildHasher, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::{ptr, str};
use utils::{CrossIndex, MaxDistance};

/// Used to specify the source of certain [`Error`] variants.
#[derive(Debug)]
pub enum InputType {
    Query,
    Reference,
}

impl Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            InputType::Query => "query",
            InputType::Reference => "reference",
        };
        write!(f, "{}", text)
    }
}

/// Symscan error variants.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An input collection contained references to at least one non-ASCII string.
    #[error("non-ASCII input currently unsupported ('{offending_string}' at {offending_idx})")]
    NonAsciiInput {
        input_type: InputType,
        offending_idx: usize,
        offending_string: String,
    },

    /// An input collection contained more than the maximum allowed number of strings.
    ///
    /// In most cases, the maximum allowed length is [4,294,967,295](u32::MAX). This is because
    /// internal computations use [`u32`]s to encode string indices. The exception is when calling
    /// [`get_neighbors_across`], where the maximum is instead 2,147,483,647 ((2^31)-1) due to the
    /// fact that one of the 32 bits is reserved for distinguishing between indexes of the `query`
    /// slice and the `reference` slice.
    #[error("{input_type} must not hold more than {limit} elements, got {got}")]
    TooManyStrings {
        input_type: InputType,
        got: usize,
        limit: usize,
    },

    /// The `max_distance` function / method parameter was set to [255](u8::MAX).
    ///
    /// This results in an error because that value is reserved for encoding when pairs exceed the
    /// threshold distance during internal computations.
    #[error("max_distance is capped at {limit}, got {illegal}", limit = u8::MAX - 1, illegal = u8::MAX)]
    MaxDistCapped,

    /// The `max_distance` method parameter was set to a value greater than that given when
    /// constructing [`CachedRef`] being queried.
    ///
    /// This results in an error because the `max_distance` given at [`CachedRef`] construction
    /// time determines how many `reference` string deletion variants are generated and cached in
    /// the struct. A cache containing deletion variants to a depth of X cannot support symscan
    /// queries with `max_distance` > X.
    #[error("CachedRef instance not compatible with max_distance above {limit}, got {got}")]
    MaxDistTooLargeForCache { got: u8, limit: u8 },
}

mod utils {
    use super::Error;

    #[derive(Clone, Copy, PartialEq, PartialOrd)]
    pub struct MaxDistance(u8);

    impl MaxDistance {
        pub fn as_u8(&self) -> u8 {
            self.0
        }

        pub fn as_usize(&self) -> usize {
            self.0 as usize
        }
    }

    impl TryFrom<u8> for MaxDistance {
        type Error = Error;

        fn try_from(value: u8) -> Result<Self, Self::Error> {
            if value == u8::MAX {
                Err(Error::MaxDistCapped)
            } else {
                Ok(Self(value))
            }
        }
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
}

/// Multiplier for spreading a 32-bit key over 64 bits: the odd integer nearest `2^64 / phi`.
///
/// Multiplying by an odd constant is a bijection modulo 2^64, and this constant's bit pattern is
/// dense enough that carries propagate every input bit into the high bits of the product. Known
/// as Knuth's multiplicative hashing, or Fibonacci hashing.
const GOLDEN_RATIO_64: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Default)]
struct VariantHasher(u64);

impl Hasher for VariantHasher {
    fn write(&mut self, bytes: &[u8]) {
        unreachable!("hasher only designed for u32 variant hashes, got {bytes:?}");
    }

    /// Spread the 32-bit variant hash across all 64 bits.
    ///
    /// hashbrown picks the bucket from the low bits of the hash and derives its SIMD control byte
    /// from the top 7, so a 32-bit key has to occupy both ends. Zero-extending would pin every
    /// control byte to zero; shifting into the high half would pin every key to bucket zero.
    fn write_u32(&mut self, i: u32) {
        self.0 = (i as u64).wrapping_mul(GOLDEN_RATIO_64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Default)]
struct VariantHasherBuilder;

impl BuildHasher for VariantHasherBuilder {
    type Hasher = VariantHasher;

    fn build_hasher(&self) -> Self::Hasher {
        VariantHasher::default()
    }
}

struct Span {
    start: usize,
    len: usize,
}

impl Span {
    fn new(start: usize, len: usize) -> Self {
        Span { start, len }
    }

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn as_range(&self) -> Range<usize> {
        self.start..self.start + self.len
    }
}

/// Collection of string pairs that lie within the specified Levenshtein edit distance threshold.
///
/// This is what is returned via the [`Ok`] variant from [`get_neighbors_within`],
/// [`get_neighbors_across`], and related methods in [`CachedRef`]. [`row`](NeighborPairs::row) and
/// [`col`](NeighborPairs::col) contain the indices of the neighbor string pairs, and
/// [`dists`](NeighborPairs::dists) contains the Levenshtein distances between the corresponding
/// pairs.
///
/// # A note on double-counting pairs
///
/// When returning the results of [`get_neighbors_within`] / [`CachedRef::get_neighbors_within`],
/// string pairs _**ARE NOT**_ double-counted. As seen in the
/// [examples](get_neighbors_within#examples), each pair is represented once where the
/// [`row`](NeighborPairs::row) index is always less than the [`col`](NeighborPairs::col) index. In
/// other words, if you were to interpret the [`NeighborPairs`] in these situations as a sparse
/// matrix, only the lower triangle will be filled.
#[derive(Debug, PartialEq)]
pub struct NeighborPairs {
    /// Indices of strings in the input `query` slice that have neighbors.
    pub row: Vec<u32>,

    /// Indices of neighbor strings. When computing neighbor pairs across separate `query` and
    /// `reference` slices, then `query[row[i]]` and `reference[col[i]]` are neighbors. When
    /// computing neighbor pairs within a single `query` slice, `query[row[i]]` and `query[col[i]]`
    /// are neighbors.
    pub col: Vec<u32>,

    /// Edit distances between neighbor string pairs. When computing neighbor pairs across separate
    /// `query` and `reference` slices, then `Levenshtein(query[row[i]], reference[col[i]]) ==
    /// dists[i]`. When computing neighbor pairs within a single `query` slice,
    /// `Levenshtein(query[row[i]], query[col[i]]) == dists[i]`.
    pub dists: Vec<u8>,
}

impl NeighborPairs {
    /// The number of neighboring string pairs detected.
    pub fn len(&self) -> usize {
        self.row.len()
    }

    /// Returns true if no neighbors were detected.
    pub fn is_empty(&self) -> bool {
        self.row.is_empty()
    }
}

/// Private zero-cost strategy distinguishing Levenshtein vs Hamming pipelines.
trait Metric: Copy + Send + Sync + 'static {
    fn count_oneshot<S: AsRef<str>>(strings: &[S], max_distance: MaxDistance) -> Vec<usize>;

    fn write_oneshot_rawidx<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    );

    fn write_oneshot_ci<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        is_ref: bool,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    );

    fn write_cached_rawidx<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    );

    fn distance(a: &str, b: &str, cutoff: usize) -> u8;
}

#[derive(Clone, Copy)]
struct Levenshtein;

#[derive(Clone, Copy)]
struct Hamming;

impl Metric for Levenshtein {
    #[inline(always)]
    fn count_oneshot<S: AsRef<str>>(strings: &[S], max_distance: MaxDistance) -> Vec<usize> {
        get_num_del_vars_per_string_up_to(strings, max_distance)
    }

    #[inline(always)]
    fn write_oneshot_rawidx<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    ) {
        write_vi_pairs_true_deletions(
            input,
            input_idx,
            max_deletions,
            chunk,
            hash_builder,
            scratch,
        );
    }

    #[inline(always)]
    fn write_oneshot_ci<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        is_ref: bool,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    ) {
        write_vi_pairs_true_deletions(
            input,
            CrossIndex::from(input_idx, is_ref),
            max_deletions,
            chunk,
            hash_builder,
            scratch,
        );
    }

    #[inline(always)]
    fn write_cached_rawidx<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    ) {
        write_vi_pairs_true_deletions(
            input,
            input_idx,
            max_deletions,
            chunk,
            hash_builder,
            scratch,
        );
    }

    #[inline(always)]
    fn distance(a: &str, b: &str, cutoff: usize) -> u8 {
        match levenshtein::distance_with_args(
            a.bytes(),
            b.bytes(),
            &levenshtein::Args::default().score_cutoff(cutoff),
        ) {
            None => u8::MAX,
            Some(dist) => dist as u8,
        }
    }
}

impl Metric for Hamming {
    #[inline(always)]
    fn count_oneshot<S: AsRef<str>>(strings: &[S], max_distance: MaxDistance) -> Vec<usize> {
        get_num_del_vars_per_string_at(strings, max_distance)
    }

    #[inline(always)]
    fn write_oneshot_rawidx<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    ) {
        write_vi_pairs_exact_null(
            input,
            input_idx,
            max_deletions,
            chunk,
            hash_builder,
            scratch,
        );
    }

    #[inline(always)]
    fn write_oneshot_ci<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        is_ref: bool,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    ) {
        write_vi_pairs_exact_null(
            input,
            CrossIndex::from(input_idx, is_ref),
            max_deletions,
            chunk,
            hash_builder,
            scratch,
        );
    }

    #[inline(always)]
    fn write_cached_rawidx<H: BuildHasher>(
        input: &str,
        input_idx: u32,
        max_deletions: MaxDistance,
        chunk: &mut [MaybeUninit<VariantIndexPair>],
        hash_builder: &H,
        scratch: &mut Vec<u8>,
    ) {
        write_vi_pairs_up_to_null(
            input,
            input_idx,
            max_deletions,
            chunk,
            hash_builder,
            scratch,
        );
    }

    #[inline(always)]
    fn distance(a: &str, b: &str, cutoff: usize) -> u8 {
        match hamming::distance_with_args(
            a.bytes(),
            b.bytes(),
            &hamming::Args::default().score_cutoff(cutoff),
        ) {
            Ok(Some(dist)) => dist as u8,
            _ => u8::MAX,
        }
    }
}

/// Shared memoized deletion-variant store used by [`CachedRef`] and [`CachedRefHamming`].
struct CachedStore<M: Metric> {
    str_store: Vec<u8>,
    str_spans: Vec<Span>,
    index_store: Vec<u32>,
    variant_map: HashMap<u32, Span, VariantHasherBuilder>,
    max_distance: MaxDistance,
    _metric: PhantomData<M>,
}

impl<M: Metric> CachedStore<M> {
    fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
        if reference.len() > u32::MAX as usize {
            return Err(Error::TooManyStrings {
                input_type: InputType::Reference,
                got: reference.len(),
                limit: u32::MAX as usize,
            });
        }
        let max_distance = MaxDistance::try_from(max_distance)?;
        check_strings_ascii(reference, InputType::Reference)?;

        let (str_store, str_spans) = {
            let strlens: Vec<_> = reference.iter().map(|s| s.as_ref().len()).collect();

            let mut str_store_uninit = prealloc_maybeuninit_vec(strlens.iter().sum());
            let str_spans = get_disjoint_spans(&strlens);
            let str_store_chunks = get_disjoint_chunks_mut(&strlens, &mut str_store_uninit[..]);

            reference
                .par_iter()
                .zip(str_store_chunks.into_par_iter())
                .for_each(|(s, chunk)| {
                    debug_assert_eq!(s.as_ref().len(), chunk.len());
                    unsafe {
                        ptr::copy_nonoverlapping(
                            s.as_ref().as_ptr(),
                            chunk.as_mut_ptr() as *mut u8,
                            s.as_ref().len(),
                        )
                    };
                });

            let str_store = unsafe { cast_to_initialised_vec(str_store_uninit) };

            (str_store, str_spans)
        };

        let hash_builder = FixedState::default();

        let (index_store, convergence_groups) = {
            let num_vars_per_string = get_num_del_vars_per_string_up_to(reference, max_distance);

            let mut variant_index_pairs_uninit =
                prealloc_maybeuninit_vec::<VariantIndexPair>(num_vars_per_string.iter().sum());
            let vip_chunks =
                get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

            reference
                .par_iter()
                .zip(vip_chunks.into_par_iter())
                .enumerate()
                .for_each_init(Vec::new, |scratch, (idx, (s, chunk))| {
                    M::write_cached_rawidx(
                        s.as_ref(),
                        idx as u32,
                        max_distance,
                        chunk,
                        &hash_builder,
                        scratch,
                    );
                });

            let variant_index_pairs =
                unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };

            // Every group is kept, including singletons: the variant map must hold every reference
            // variant for cross queries to find it.
            collect_convergent_indices::<u32, _>(variant_index_pairs, |group| {
                let len = distinct(group).count();
                Some((len, (group[0].variant_hash(), len)))
            })
        };

        let mut variant_map =
            HashMap::with_capacity_and_hasher(convergence_groups.len(), VariantHasherBuilder);
        let mut cursor = 0;

        for (v_hash, len) in convergence_groups {
            variant_map.entry(v_hash).insert(Span::new(cursor, len));
            cursor += len;
        }

        debug_assert_eq!(cursor, index_store.len());

        Ok(CachedStore {
            str_store,
            str_spans,
            index_store,
            variant_map,
            max_distance,
            _metric: PhantomData,
        })
    }

    fn get_neighbors_within(&self, max_distance: u8) -> Result<NeighborPairs, Error> {
        let max_distance = MaxDistance::try_from(max_distance)?;
        if max_distance > self.max_distance {
            return Err(Error::MaxDistTooLargeForCache {
                got: max_distance.as_u8(),
                limit: self.max_distance.as_u8(),
            });
        }

        let mut convergent_indices = Vec::with_capacity(self.variant_map.len());
        self.variant_map.iter().for_each(|(_, span)| {
            if span.len() == 1 {
                return;
            }
            convergent_indices.push(self.get_convergent_indices_from_span(span));
        });

        let candidates = get_hit_candidates_within(&convergent_indices);
        let dists = self.compute_dists_fully_cached(&candidates, self, max_distance);

        Ok(validate_and_collect_hits(candidates, dists, max_distance))
    }

    fn get_neighbors_across(
        &self,
        query: &[impl AsRef<str> + Sync],
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        let max_distance = MaxDistance::try_from(max_distance)?;
        if max_distance > self.max_distance {
            return Err(Error::MaxDistTooLargeForCache {
                got: max_distance.as_u8(),
                limit: self.max_distance.as_u8(),
            });
        }
        if query.len() > u32::MAX as usize {
            return Err(Error::TooManyStrings {
                input_type: InputType::Query,
                got: query.len(),
                limit: u32::MAX as usize,
            });
        }
        check_strings_ascii(query, InputType::Query)?;

        let (q_idx_store, convergence_groups) = {
            let num_vars_per_string = M::count_oneshot(query, max_distance);

            let mut variant_index_pairs_uninit =
                prealloc_maybeuninit_vec(num_vars_per_string.iter().sum());
            let vip_chunks =
                get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

            let hash_builder = FixedState::default();

            query
                .par_iter()
                .zip(vip_chunks.into_par_iter())
                .enumerate()
                .for_each_init(Vec::new, |scratch, (idx, (s, chunk))| {
                    M::write_oneshot_rawidx(
                        s.as_ref(),
                        idx as u32,
                        max_distance,
                        chunk,
                        &hash_builder,
                        scratch,
                    );
                });

            let variant_index_pairs =
                unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };

            collect_convergent_indices::<u32, _>(variant_index_pairs, |group| {
                let span = self.variant_map.get(&group[0].variant_hash())?;
                let len_q = distinct(group).count();
                Some((len_q, (len_q, self.get_convergent_indices_from_span(span))))
            })
        };

        let mut cursor = 0;
        let convergence_groups: Vec<_> = convergence_groups
            .into_iter()
            .map(|(len_q, r_indices)| {
                let group = (&q_idx_store[cursor..cursor + len_q], r_indices);
                cursor += len_q;
                group
            })
            .collect();

        debug_assert_eq!(cursor, q_idx_store.len());

        let candidates = get_hit_candidates_across(&convergence_groups);
        let dists = self.compute_dists_partially_cached(&candidates, query, max_distance);

        Ok(validate_and_collect_hits(candidates, dists, max_distance))
    }

    fn get_neighbors_across_cached(
        &self,
        query: &Self,
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        let max_distance = MaxDistance::try_from(max_distance)?;
        if max_distance > self.max_distance {
            return Err(Error::MaxDistTooLargeForCache {
                got: max_distance.as_u8(),
                limit: self.max_distance.as_u8(),
            });
        }
        if max_distance > query.max_distance {
            return Err(Error::MaxDistTooLargeForCache {
                got: max_distance.as_u8(),
                limit: query.max_distance.as_u8(),
            });
        }

        let convergence_groups = if query.variant_map.len() < self.variant_map.len() {
            let mut num_convergence_groups = 0;

            query.variant_map.iter().for_each(|(variant, _)| {
                if self.variant_map.get(variant).is_some() {
                    num_convergence_groups += 1;
                }
            });

            let mut convergence_groups = Vec::with_capacity(num_convergence_groups);

            query.variant_map.iter().for_each(|(variant, span_q)| {
                if let Some(span_r) = self.variant_map.get(variant) {
                    convergence_groups.push((
                        query.get_convergent_indices_from_span(span_q),
                        self.get_convergent_indices_from_span(span_r),
                    ));
                }
            });

            convergence_groups
        } else {
            let mut num_convergence_groups = 0;

            self.variant_map.iter().for_each(|(variant, _)| {
                if query.variant_map.get(variant).is_some() {
                    num_convergence_groups += 1;
                }
            });

            let mut convergence_groups = Vec::with_capacity(num_convergence_groups);

            self.variant_map.iter().for_each(|(variant, span_r)| {
                if let Some(span_q) = query.variant_map.get(variant) {
                    convergence_groups.push((
                        query.get_convergent_indices_from_span(span_q),
                        self.get_convergent_indices_from_span(span_r),
                    ));
                }
            });

            convergence_groups
        };

        let candidates = get_hit_candidates_across(&convergence_groups);
        let dists = self.compute_dists_fully_cached(&candidates, query, max_distance);

        Ok(validate_and_collect_hits(candidates, dists, max_distance))
    }

    #[inline(always)]
    fn get_convergent_indices_from_span(&self, span: &Span) -> &[u32] {
        &self.index_store[span.as_range()]
    }

    #[inline(always)]
    fn get_str_at_index(&self, i: usize) -> &str {
        unsafe { str::from_utf8_unchecked(&self.str_store[self.str_spans[i].as_range()]) }
    }

    fn compute_dists_partially_cached(
        &self,
        hit_candidates: &[(u32, u32)],
        query: &[impl AsRef<str> + Sync],
        max_distance: MaxDistance,
    ) -> Vec<u8> {
        hit_candidates
            .par_iter()
            .map(|&(idx_query, idx_reference)| {
                M::distance(
                    query[idx_query as usize].as_ref(),
                    self.get_str_at_index(idx_reference as usize),
                    max_distance.as_usize(),
                )
            })
            .collect()
    }

    fn compute_dists_fully_cached(
        &self,
        hit_candidates: &[(u32, u32)],
        query: &Self,
        max_distance: MaxDistance,
    ) -> Vec<u8> {
        hit_candidates
            .par_iter()
            .map(|&(idx_query, idx_reference)| {
                M::distance(
                    query.get_str_at_index(idx_query as usize),
                    self.get_str_at_index(idx_reference as usize),
                    max_distance.as_usize(),
                )
            })
            .collect()
    }
}

/// A struct for memoizing the deletion variant calculations for a string collection.
///
/// When [constructed](CachedRef::new), [`CachedRef`] precomputes and stores the deletion variants
/// for the supplied `reference` strings as a hashmap. This significantly speeds up subsequent
/// queries against the reference, at the cost of spending extra time to construct the hashmap.
/// This is useful for use-cases where you want to repeatedly query the same reference, especially
/// if the reference is very large. However, for one-off computations, the pure functions
/// [`get_neighbors_within`] and [`get_neighbors_across`] are faster.
///
/// **Note** that [`CachedRef`] instances constructed with `max_distance` set to X can only support
/// queries with `max_distance` less than or equal to X.
///
/// **Note** when interpreting the index order of returned [`NeighborPairs`], the string collection
/// specified at construction is considered the _reference_, and any string collections specified
/// during subsequent query calls are considered the _query_.
///
/// # Examples
///
/// ```
/// use symscan::{CachedRef, NeighborPairs};
///
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let cached = CachedRef::new(&reference, 2).unwrap();
///
/// let NeighborPairs { row, col, dists } = cached
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 1)
///     .unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = cached
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 2)
///     .unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2, 2]);
/// assert_eq!(col,   vec![3, 3, 2, 3]);
/// assert_eq!(dists, vec![2, 1, 2, 0]);
/// ```
pub struct CachedRef {
    store: CachedStore<Levenshtein>,
}

impl CachedRef {
    /// Construct a new [`CachedRef`] instance.
    pub fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
        Ok(CachedRef {
            store: CachedStore::new(reference, max_distance)?,
        })
    }

    /// The memoized equivalent of [`get_neighbors_within`].
    pub fn get_neighbors_within(&self, max_distance: u8) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_within(max_distance)
    }

    /// The memoized equivalent of [`get_neighbors_across`].
    pub fn get_neighbors_across(
        &self,
        query: &[impl AsRef<str> + Sync],
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_across(query, max_distance)
    }

    /// Equivalent to [`CachedRef::get_neighbors_across`], where the query is also a [`CachedRef`]
    /// instance.
    pub fn get_neighbors_across_cached(
        &self,
        query: &Self,
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store
            .get_neighbors_across_cached(&query.store, max_distance)
    }
}

/// A version of [`CachedRef`] but for Hamming distance instead of Levenshtein distance.
///
/// # Examples
///
/// ```
/// use symscan::{CachedRefHamming, NeighborPairs};
///
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let cached_hamming = CachedRefHamming::new(&reference, 2).unwrap();
///
/// let NeighborPairs { row, col, dists } = cached_hamming
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 1)
///     .unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = cached_hamming
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 2)
///     .unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2]);
/// assert_eq!(col,   vec![3, 3, 3]);
/// assert_eq!(dists, vec![2, 1, 0]);
/// ```
pub struct CachedRefHamming {
    store: CachedStore<Hamming>,
}

impl CachedRefHamming {
    /// Construct a new [`CachedRefHamming`] instance.
    pub fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
        Ok(CachedRefHamming {
            store: CachedStore::new(reference, max_distance)?,
        })
    }

    /// The memoized equivalent of [`get_hamming_neighbors_within`].
    pub fn get_neighbors_within(&self, max_distance: u8) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_within(max_distance)
    }

    /// The memoized equivalent of [`get_hamming_neighbors_across`].
    pub fn get_neighbors_across(
        &self,
        query: &[impl AsRef<str> + Sync],
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_across(query, max_distance)
    }

    /// Equivalent to [`CachedRefHamming::get_neighbors_across`], where the query is also a
    /// [`CachedRefHamming`] instance.
    pub fn get_neighbors_across_cached(
        &self,
        query: &Self,
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store
            .get_neighbors_across_cached(&query.store, max_distance)
    }
}

/// Detect string pairs within an input collection that lie within a threshold Levenshtein edit
/// distance.
///
/// The function considers all possible combinations (not permutations, [read
/// more](NeighborPairs#a-note-on-double-counting-pairs)) of string pairs from `query`, and returns
/// all those where the two strings are no more than `max_distance` Levenshtein edit distance units
/// apart.
///
/// # Errors
///
/// Currently, the crate only supports ASCII input. The function will [`Err`] with
/// [`Error::NonAsciiInput`] if `query` contains any non-ASCII data.
///
/// There are some hard limits on the sizes of the input arguments (see [`Error::TooManyStrings`],
/// [`Error::MaxDistCapped`]). Note however that in practice, runtime or memory usage is almost
/// certainly the limiting factor instead.
///
/// # Examples
///
/// ```
/// use symscan::{get_neighbors_within, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let NeighborPairs { row, col, dists } = get_neighbors_within(&query, 1).unwrap();
///
/// assert_eq!(row,   vec![0, 0, 1]);
/// assert_eq!(col,   vec![1, 3, 2]);
/// assert_eq!(dists, vec![1, 1, 1]);
///
/// let NeighborPairs { row, col, dists } = get_neighbors_within(&query, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 0, 0, 1, 1]);
/// assert_eq!(col,   vec![1, 2, 3, 2, 3]);
/// assert_eq!(dists, vec![1, 2, 1, 1, 2]);
/// ```
pub fn get_neighbors_within(
    query: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    get_neighbors_within_impl::<Levenshtein>(query, max_distance)
}

/// Detect string pairs across two input collections that lie within a threshold Levenshtein edit
/// distance.
///
/// The function considers all string pairs in the cartesian product of `query` and `reference`,
/// and returns all those where the two strings are no more than `max_distance` Levenshtein edit
/// distance units apart.
///
/// # Errors
///
/// Currently, the crate only supports ASCII input. The function will [`Err`] with
/// [`Error::NonAsciiInput`] if `query` or `reference` contain any non-ASCII data.
///
/// There are some hard limits on the sizes of the input arguments (see [`Error::TooManyStrings`],
/// [`Error::MaxDistCapped`]). Note however that in practice, runtime or memory usage is almost
/// certainly the limiting factor instead.
///
/// # Examples
///
/// ```
/// use symscan::{get_neighbors_across, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let NeighborPairs { row, col, dists } = get_neighbors_across(&query, &reference, 1).unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = get_neighbors_across(&query, &reference, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2, 2]);
/// assert_eq!(col,   vec![3, 3, 2, 3]);
/// assert_eq!(dists, vec![2, 1, 2, 0]);
/// ```
pub fn get_neighbors_across(
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    get_neighbors_across_impl::<Levenshtein>(query, reference, max_distance)
}

/// A version of [`get_neighbors_within`] which uses Hamming distance instead of Levenshtein
/// distance.
///
/// # Examples
///
/// ```
/// use symscan::{get_hamming_neighbors_within, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_within(&query, 1).unwrap();
///
/// assert_eq!(row,   vec![0, 1]);
/// assert_eq!(col,   vec![1, 2]);
/// assert_eq!(dists, vec![1, 1]);
///
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_within(&query, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 0, 1]);
/// assert_eq!(col,   vec![1, 2, 2]);
/// assert_eq!(dists, vec![1, 2, 1]);
/// ```
pub fn get_hamming_neighbors_within(
    query: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    get_neighbors_within_impl::<Hamming>(query, max_distance)
}

/// A version of [`get_neighbors_across`] which uses Hamming distance instead of Levenshtein
/// distance.
///
/// # Examples
///
/// ```
/// use symscan::{get_hamming_neighbors_across, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_across(&query, &reference, 1).unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_across(&query, &reference, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2]);
/// assert_eq!(col,   vec![3, 3, 3]);
/// assert_eq!(dists, vec![2, 1, 0]);
/// ```
pub fn get_hamming_neighbors_across(
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    get_neighbors_across_impl::<Hamming>(query, reference, max_distance)
}

fn get_neighbors_within_impl<M: Metric>(
    query: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    if query.len() > u32::MAX as usize {
        return Err(Error::TooManyStrings {
            input_type: InputType::Query,
            got: query.len(),
            limit: u32::MAX as usize,
        });
    }
    let max_distance = MaxDistance::try_from(max_distance)?;
    check_strings_ascii(query, InputType::Query)?;

    let (convergent_indices, group_sizes) = {
        let num_vars_per_string = M::count_oneshot(query, max_distance);

        let mut variant_index_pairs_uninit =
            prealloc_maybeuninit_vec(num_vars_per_string.iter().sum());
        let vip_chunks =
            get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

        let hash_builder = FixedState::default();

        query
            .par_iter()
            .zip(vip_chunks.into_par_iter())
            .enumerate()
            .for_each_init(Vec::new, |scratch, (idx, (s, chunk))| {
                M::write_oneshot_rawidx(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    chunk,
                    &hash_builder,
                    scratch,
                );
            });

        let variant_index_pairs = unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };
        // Payload is the group size; both halves of (num_indices, payload) are the same value.
        collect_convergent_indices::<u32, _>(variant_index_pairs, |group| {
            let len = distinct(group).count();
            (len > 1).then_some((len, len))
        })
    };

    let convergent_chunks = get_convergent_chunks(&group_sizes, &convergent_indices[..]);
    let candidates = get_hit_candidates_within(&convergent_chunks);
    let dists = compute_dists::<M>(&candidates, query, query, max_distance);

    Ok(validate_and_collect_hits(candidates, dists, max_distance))
}

fn get_neighbors_across_impl<M: Metric>(
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    if query.len() > CrossIndex::MAX {
        return Err(Error::TooManyStrings {
            input_type: InputType::Query,
            got: query.len(),
            limit: CrossIndex::MAX,
        });
    }
    if reference.len() > CrossIndex::MAX {
        return Err(Error::TooManyStrings {
            input_type: InputType::Reference,
            got: reference.len(),
            limit: CrossIndex::MAX,
        });
    }
    let max_distance = MaxDistance::try_from(max_distance)?;
    check_strings_ascii(query, InputType::Query)?;
    check_strings_ascii(reference, InputType::Reference)?;

    let (convergent_indices, group_sizes) = {
        let num_del_variants_q = M::count_oneshot(query, max_distance);
        let num_del_variants_r = M::count_oneshot(reference, max_distance);

        let total_capacity =
            num_del_variants_q.iter().sum::<usize>() + num_del_variants_r.iter().sum::<usize>();
        let mut variant_index_pairs_uninit = prealloc_maybeuninit_vec(total_capacity);
        let (vip_chunks_q, vip_chunks_r) = get_disjoint_chunks_mut_cross(
            &num_del_variants_q,
            &num_del_variants_r,
            &mut variant_index_pairs_uninit[..],
        );

        let hash_builder = FixedState::default();

        query
            .par_iter()
            .zip(vip_chunks_q.into_par_iter())
            .enumerate()
            .for_each_init(Vec::new, |scratch, (idx, (s, chunk))| {
                M::write_oneshot_ci(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    false,
                    chunk,
                    &hash_builder,
                    scratch,
                );
            });
        reference
            .par_iter()
            .zip(vip_chunks_r.into_par_iter())
            .enumerate()
            .for_each_init(Vec::new, |scratch, (idx, (s, chunk))| {
                M::write_oneshot_ci(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    true,
                    chunk,
                    &hash_builder,
                    scratch,
                );
            });

        let variant_index_pairs = unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };
        collect_convergent_indices::<CrossIndex, _>(variant_index_pairs, |group| {
            let (len_q, len_r) = distinct(group).fold((0, 0), |(q, r), word| {
                if CrossIndex::from_index_bits(word.index_bits()).is_ref() {
                    (q, r + 1)
                } else {
                    (q + 1, r)
                }
            });
            (len_q > 0 && len_r > 0).then_some((len_q + len_r, (len_q, len_r)))
        })
    };

    let convergent_chunks = get_convergent_chunks_cross(&group_sizes, &convergent_indices[..]);
    let candidates = get_hit_candidates_across(&convergent_chunks);
    let dists = compute_dists::<M>(&candidates, query, reference, max_distance);

    Ok(validate_and_collect_hits(candidates, dists, max_distance))
}

fn check_strings_ascii(strings: &[impl AsRef<str>], input_type: InputType) -> Result<(), Error> {
    for (idx, s) in strings.iter().enumerate() {
        if !s.as_ref().is_ascii() {
            return Err(Error::NonAsciiInput {
                input_type,
                offending_idx: idx,
                offending_string: s.as_ref().to_string(),
            });
        }
    }
    Ok(())
}

/// Compute the total number of deletion variants up to a certain number of maximum deletions.
fn get_num_del_vars_per_string_up_to(
    strings: &[impl AsRef<str>],
    max_distance: MaxDistance,
) -> Vec<usize> {
    strings
        .iter()
        .map(|s| {
            let mut num_vars = 0;
            for k in 0..=max_distance.as_u8() {
                if k as usize > s.as_ref().len() {
                    break;
                }
                num_vars += get_num_k_combs(s.as_ref().len(), k);
            }
            num_vars
        })
        .collect()
}

/// Compute the total number of deletion variants per input string at exactly some number of
/// deletions.
fn get_num_del_vars_per_string_at(
    strings: &[impl AsRef<str>],
    max_distance: MaxDistance,
) -> Vec<usize> {
    strings
        .iter()
        .map(|s| {
            if max_distance.as_usize() >= s.as_ref().len() {
                1
            } else {
                get_num_k_combs(s.as_ref().len(), max_distance.as_u8())
            }
        })
        .collect()
}

fn get_num_k_combs(n: usize, k: u8) -> usize {
    debug_assert!(n > 0);
    debug_assert!(n >= k as usize);

    if k == 0 {
        return 1;
    }

    let num_subsamples: usize = (n - k as usize + 1..=n).product();
    let subsample_perms: usize = (1..=k as usize).product();

    num_subsamples / subsample_perms
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
struct VariantIndexPair(u64);

impl VariantIndexPair {
    const BUCKET_BITS: u32 = 8;
    const NUM_BUCKETS: usize = 1 << Self::BUCKET_BITS;
    const BUCKET_SHIFT: u32 = u64::BITS - Self::BUCKET_BITS;

    #[inline(always)]
    fn new(variant_hash: u32, index_bits: u32) -> Self {
        Self(((variant_hash as u64) << 32) | index_bits as u64)
    }

    #[inline(always)]
    fn from_index(variant_hash: u32, index: impl VariantIndex) -> Self {
        Self::new(variant_hash, index.index_bits())
    }

    /// High half: 32-bit foldhash truncation of the deletion variant.
    #[inline(always)]
    fn variant_hash(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Low half: packed index bits ([`u32`] verbatim, or [`CrossIndex`] including its type bit).
    #[inline(always)]
    fn index_bits(self) -> u32 {
        self.0 as u32
    }

    /// Top [`Self::BUCKET_BITS`] of the variant hash — the MSD radix bucket.
    #[inline(always)]
    fn bucket(self) -> usize {
        (self.0 >> Self::BUCKET_SHIFT) as usize
    }
}

/// Index stored in the low half of a [`VariantIndexPair`].
///
/// Only [`u32`] (within / cached) and [`CrossIndex`] (across) are used.
trait VariantIndex: Copy + Send + Sync {
    /// Bits stored in the low half of a [`VariantIndexPair`] (for [`CrossIndex`], includes the type bit).
    fn index_bits(self) -> u32;

    fn from_index_bits(bits: u32) -> Self;

    /// String index with any type tag cleared — what goes into convergent-index output.
    fn string_index(self) -> u32;
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

/// Invoke `f` once for every lexicographic combination of `k` distinct indices from `0..n`.
///
/// Special-cases `k == 1` and `k == 2` with tight nested loops. Larger `k` uses an in-place
/// combination stepper over a stack buffer (no per-combination heap allocation).
fn for_each_combination(n: usize, k: usize, mut f: impl FnMut(&[usize])) {
    if k > n {
        return;
    }
    if k == 0 {
        f(&[]);
        return;
    }

    match k {
        1 => {
            for i in 0..n {
                f(&[i]);
            }
        }
        2 => {
            for i in 0..n {
                for j in (i + 1)..n {
                    f(&[i, j]);
                }
            }
        }
        _ => {
            // max_distance is capped at u8::MAX - 1, so k fits in this stack buffer.
            debug_assert!(k < 256);
            let mut indices = [0usize; 256];
            for i in 0..k {
                indices[i] = i;
            }
            loop {
                f(&indices[..k]);

                let mut i = k;
                loop {
                    if i == 0 {
                        return;
                    }
                    i -= 1;
                    if indices[i] < n - k + i {
                        indices[i] += 1;
                        for j in (i + 1)..k {
                            indices[j] = indices[j - 1] + 1;
                        }
                        break;
                    }
                }
            }
        }
    }
}

/// Generate deletion variants by dropping deleted characters (Levenshtein / true deletions), for
/// depths `0..=max_deletions`.
fn write_vi_pairs_true_deletions<I: VariantIndex, H: BuildHasher>(
    input: &str,
    index: I,
    max_deletions: MaxDistance,
    chunk: &mut [MaybeUninit<VariantIndexPair>],
    hash_builder: &H,
    scratch: &mut Vec<u8>,
) {
    let input_length = input.len();
    let input_bytes = input.as_bytes();

    chunk[0].write(VariantIndexPair::from_index(
        hash_string(input, hash_builder),
        index,
    ));

    let mut variant_idx = 1;
    scratch.reserve(input_length);
    for num_deletions in 1..=max_deletions.as_u8() {
        let k = num_deletions as usize;
        if k > input_length {
            break;
        }

        for_each_combination(input_length, k, |deletion_indices| {
            scratch.clear();
            let mut offset = 0;

            for &idx in deletion_indices {
                scratch.extend_from_slice(&input_bytes[offset..idx]);
                offset = idx + 1;
            }
            scratch.extend_from_slice(&input_bytes[offset..input_length]);

            chunk[variant_idx].write(VariantIndexPair::from_index(
                hash_string(&*scratch, hash_builder),
                index,
            ));
            variant_idx += 1;
        });
    }
}

/// Generate deletion variants at exactly `max_deletions` with null-character placeholders (Hamming
/// one-shot path).
fn write_vi_pairs_exact_null<I: VariantIndex, H: BuildHasher>(
    input: &str,
    index: I,
    max_deletions: MaxDistance,
    chunk: &mut [MaybeUninit<VariantIndexPair>],
    hash_builder: &H,
    scratch: &mut Vec<u8>,
) {
    const NULL_CHARACTER: u8 = u8::MAX;
    let input_length = input.len();
    let input_bytes = input.as_bytes();
    scratch.reserve(input_length);

    if max_deletions.as_usize() >= input_length {
        scratch.clear();
        scratch.resize(input_length, NULL_CHARACTER);
        chunk[0].write(VariantIndexPair::from_index(
            hash_string(&*scratch, hash_builder),
            index,
        ));
        return;
    }

    let mut variant_idx = 0;
    for_each_combination(input_length, max_deletions.as_usize(), |deletion_indices| {
        scratch.clear();
        let mut cursor = 0;

        for &idx in deletion_indices {
            scratch.extend_from_slice(&input_bytes[cursor..idx]);
            scratch.push(NULL_CHARACTER);
            cursor = idx + 1;
        }
        scratch.extend_from_slice(&input_bytes[cursor..input_length]);

        chunk[variant_idx].write(VariantIndexPair::from_index(
            hash_string(&*scratch, hash_builder),
            index,
        ));
        variant_idx += 1;
    });
}

/// Generate deletion variants with null-character placeholders for depths `0..=max_deletions`
/// (Hamming cached construction path).
fn write_vi_pairs_up_to_null<I: VariantIndex, H: BuildHasher>(
    input: &str,
    index: I,
    max_deletions: MaxDistance,
    chunk: &mut [MaybeUninit<VariantIndexPair>],
    hash_builder: &H,
    scratch: &mut Vec<u8>,
) {
    const NULL_CHARACTER: u8 = u8::MAX;
    let input_length = input.len();
    let input_bytes = input.as_bytes();

    chunk[0].write(VariantIndexPair::from_index(
        hash_string(input, hash_builder),
        index,
    ));

    let mut variant_idx = 1;
    scratch.reserve(input_length);
    for num_deletions in 1..=max_deletions.as_u8() {
        let k = num_deletions as usize;
        if k > input_length {
            break;
        }

        for_each_combination(input_length, k, |deletion_indices| {
            scratch.clear();
            let mut cursor = 0;

            for &idx in deletion_indices {
                scratch.extend_from_slice(&input_bytes[cursor..idx]);
                scratch.push(NULL_CHARACTER);
                cursor = idx + 1;
            }
            scratch.extend_from_slice(&input_bytes[cursor..input_length]);

            chunk[variant_idx].write(VariantIndexPair::from_index(
                hash_string(&*scratch, hash_builder),
                index,
            ));
            variant_idx += 1;
        });
    }
}

fn hash_string(s: impl AsRef<[u8]>, hash_builder: &impl BuildHasher) -> u32 {
    let mut hasher = hash_builder.build_hasher();
    hasher.write(s.as_ref());
    (hasher.finish() >> 32) as u32
}

fn prealloc_maybeuninit_vec<T>(total_capacity: usize) -> Vec<MaybeUninit<T>> {
    let mut v: Vec<MaybeUninit<T>> = Vec::with_capacity(total_capacity);
    unsafe { v.set_len(total_capacity) };
    v
}

unsafe fn cast_to_initialised_vec<T>(mut input: Vec<MaybeUninit<T>>) -> Vec<T> {
    let ptr = input.as_mut_ptr() as *mut T;
    let len = input.len();
    let cap = input.capacity();
    std::mem::forget(input);
    Vec::from_raw_parts(ptr, len, cap)
}

fn get_disjoint_spans(span_lens: &[usize]) -> Vec<Span> {
    let mut spans = Vec::with_capacity(span_lens.len());
    let mut cursor = 0;
    for &n in span_lens {
        spans.push(Span::new(cursor, n));
        cursor += n;
    }
    spans
}

fn get_disjoint_chunks_mut<'a, T>(
    chunk_lens: &[usize],
    mut backing_memory: &'a mut [T],
) -> Vec<&'a mut [T]> {
    let mut chunks = Vec::with_capacity(chunk_lens.len());
    for &n in chunk_lens {
        let (chunk, rest) = backing_memory.split_at_mut(n);
        chunks.push(chunk);
        backing_memory = rest;
    }

    debug_assert_eq!(backing_memory.len(), 0);

    chunks
}

/// Similar to get_disjoint_chunks_mut but for cross-set queries. Takes two chunk length slices and
/// generates two chunk vectors.
fn get_disjoint_chunks_mut_cross<'a, T>(
    chunk_lens_a: &[usize],
    chunk_lens_b: &[usize],
    mut backing_memory: &'a mut [T],
) -> (Vec<&'a mut [T]>, Vec<&'a mut [T]>) {
    let mut chunks_a = Vec::with_capacity(chunk_lens_a.len());
    for &n in chunk_lens_a {
        let (chunk, rest) = backing_memory.split_at_mut(n);
        chunks_a.push(chunk);
        backing_memory = rest;
    }

    let mut chunks_b = Vec::with_capacity(chunk_lens_b.len());
    for &n in chunk_lens_b {
        let (chunk, rest) = backing_memory.split_at_mut(n);
        chunks_b.push(chunk);
        backing_memory = rest;
    }

    debug_assert_eq!(backing_memory.len(), 0);

    (chunks_a, chunks_b)
}

/// The runs of equal variant hashes in a sorted VIP slice.
#[inline]
fn groups(vip: &[VariantIndexPair]) -> impl Iterator<Item = &[VariantIndexPair]> {
    vip.chunk_by(|a, b| a.variant_hash() == b.variant_hash())
}

/// Entries of one group with adjacent duplicates skipped, replacing the removed `Vec::dedup`.
#[inline]
fn distinct(group: &[VariantIndexPair]) -> impl Iterator<Item = VariantIndexPair> + '_ {
    group
        .chunk_by(|a, b| a.index_bits() == b.index_bits())
        .map(|run| run[0])
}

/// Sort `vip` by scattering entries into buckets keyed on the top bits of the variant hash,
/// then sorting each bucket in parallel.
///
/// Produces the same order as `par_sort_unstable`, but every pass is parallel. Also returns the
/// bucket boundaries: equal hashes share a bucket, so these never fall inside a hash group.
fn bucket_sort(vip: Vec<VariantIndexPair>) -> (Vec<VariantIndexPair>, Vec<usize>) {
    let n = vip.len();
    if n == 0 {
        return (vip, vec![0, 0]);
    }

    let num_chunks = (rayon::current_num_threads() * 4).min(n);
    let chunk_size = n.div_ceil(num_chunks);
    let src_chunks: Vec<&[VariantIndexPair]> = vip.chunks(chunk_size).collect();

    let histograms: Vec<[usize; VariantIndexPair::NUM_BUCKETS]> = src_chunks
        .par_iter()
        .map(|chunk| {
            let mut hist = [0usize; VariantIndexPair::NUM_BUCKETS];
            for &word in *chunk {
                hist[word.bucket()] += 1;
            }
            hist
        })
        .collect();

    // Bucket-major destinations: for each bucket, consecutive slices for each source chunk.
    // `bounds` marks the start of every bucket plus a final n.
    let mut dest_uninit = prealloc_maybeuninit_vec::<VariantIndexPair>(n);
    let mut dest_slices: Vec<Vec<&mut [MaybeUninit<VariantIndexPair>]>> = (0..src_chunks.len())
        .map(|_| Vec::with_capacity(VariantIndexPair::NUM_BUCKETS))
        .collect();
    let mut bounds = Vec::with_capacity(VariantIndexPair::NUM_BUCKETS + 1);
    bounds.push(0);

    let mut bounds_cursor = 0;
    let mut rest: &mut [MaybeUninit<VariantIndexPair>] = &mut dest_uninit[..];
    for bucket in 0..VariantIndexPair::NUM_BUCKETS {
        for (chunk_i, hist) in histograms.iter().enumerate() {
            let len = hist[bucket];
            let (slice, next) = rest.split_at_mut(len);
            dest_slices[chunk_i].push(slice);
            rest = next;
        }
        let bucket_len: usize = histograms.iter().map(|h| h[bucket]).sum();
        bounds_cursor += bucket_len;
        bounds.push(bounds_cursor);
    }
    debug_assert!(rest.is_empty());

    src_chunks
        .par_iter()
        .zip(dest_slices.into_par_iter())
        .for_each(|(src, mut dests)| {
            let mut cursors = [0usize; VariantIndexPair::NUM_BUCKETS];
            for &word in *src {
                let b = word.bucket();
                dests[b][cursors[b]].write(word);
                cursors[b] += 1;
            }
            for b in 0..VariantIndexPair::NUM_BUCKETS {
                debug_assert_eq!(cursors[b], dests[b].len());
            }
        });

    let mut sorted = unsafe { cast_to_initialised_vec(dest_uninit) };

    let bucket_lens: Vec<usize> = bounds.windows(2).map(|w| w[1] - w[0]).collect();
    let bucket_chunks = get_disjoint_chunks_mut(&bucket_lens, &mut sorted[..]);
    bucket_chunks.into_par_iter().for_each(|bucket| {
        // TODO: better to do radix sort here?
        bucket.sort_unstable();
    });

    (sorted, bounds)
}

/// Sort the variant-index pairs, then collect the indices of every convergent group in parallel.
///
/// `describe` returns `(num_indices, payload)` for a kept group, or `None` to skip it. It sees the
/// group with duplicates still present, so it must count through [`distinct`].
fn collect_convergent_indices<I: VariantIndex, Payload: Copy + Send>(
    variant_index_pairs: Vec<VariantIndexPair>,
    describe: impl Fn(&[VariantIndexPair]) -> Option<(usize, Payload)> + Send + Sync,
) -> (Vec<u32>, Vec<Payload>) {
    let (variant_index_pairs, bounds) = bucket_sort(variant_index_pairs);
    let chunks: Vec<_> = bounds
        .windows(2)
        .map(|w| &variant_index_pairs[w[0]..w[1]])
        .collect();

    let counts: Vec<(usize, usize)> = chunks
        .par_iter()
        .map(|chunk| {
            groups(chunk)
                .filter_map(&describe)
                .fold((0, 0), |(n_idx, n_grp), (n, _)| (n_idx + n, n_grp + 1))
        })
        .collect();

    let index_counts: Vec<usize> = counts.iter().map(|&(n, _)| n).collect();
    let group_counts: Vec<usize> = counts.iter().map(|&(_, n)| n).collect();

    let mut indices_uninit = prealloc_maybeuninit_vec(index_counts.iter().sum());
    let mut payloads_uninit = prealloc_maybeuninit_vec(group_counts.iter().sum());
    let index_chunks = get_disjoint_chunks_mut(&index_counts, &mut indices_uninit[..]);
    let payload_chunks = get_disjoint_chunks_mut(&group_counts, &mut payloads_uninit[..]);

    chunks
        .par_iter()
        .zip(index_chunks.into_par_iter())
        .zip(payload_chunks.into_par_iter())
        .for_each(|((chunk, out_indices), out_payloads)| {
            let mut i = 0;
            let mut g = 0;
            for group in groups(chunk) {
                let Some((_, payload)) = describe(group) else {
                    continue;
                };
                for word in distinct(group) {
                    out_indices[i].write(I::from_index_bits(word.index_bits()).string_index());
                    i += 1;
                }
                out_payloads[g].write(payload);
                g += 1;
            }
            debug_assert_eq!(i, out_indices.len());
            debug_assert_eq!(g, out_payloads.len());
        });

    unsafe {
        (
            cast_to_initialised_vec(indices_uninit),
            cast_to_initialised_vec(payloads_uninit),
        )
    }
}

/// Given a contiguous slice of indices and a slice of sizes that demarcate chunks of indices that
/// converge to the same deletion variant, return a vector of slices where each slice groups
/// together indices of strings that converge to the same deletion variant.
fn get_convergent_chunks<'a, T>(
    conv_group_sizes: &[usize],
    mut convergent_indices: &'a [T],
) -> Vec<&'a [T]> {
    let mut conv_chunks = Vec::with_capacity(conv_group_sizes.len());
    for &n in conv_group_sizes {
        let (chunk, rest) = convergent_indices.split_at(n);
        conv_chunks.push(chunk);
        convergent_indices = rest;
    }

    debug_assert_eq!(convergent_indices.len(), 0);

    conv_chunks
}

/// Similar to get_convergent_chunks but for cross-set queries, where the elements in the output
/// vector are two-tuples of slices, the first slice of the convergent indices from the query set,
/// and the second slice of convergent indices from the reference set.
fn get_convergent_chunks_cross<'a, T>(
    conv_group_sizes: &[(usize, usize)],
    mut convergent_indices: &'a [T],
) -> Vec<(&'a [T], &'a [T])> {
    let mut conv_chunks = Vec::with_capacity(conv_group_sizes.len());
    for &(n_q, n_r) in conv_group_sizes {
        let (chunk_q, rest) = convergent_indices.split_at(n_q);
        let (chunk_r, rest) = rest.split_at(n_r);
        conv_chunks.push((chunk_q, chunk_r));
        convergent_indices = rest;
    }

    debug_assert_eq!(convergent_indices.len(), 0);

    conv_chunks
}

fn get_hit_candidates_within(convergent_indices: &[impl AsRef<[u32]> + Sync]) -> Vec<(u32, u32)> {
    let num_hit_candidates: Vec<_> = convergent_indices
        .iter()
        .map(|indices| get_num_k_combs(indices.as_ref().len(), 2))
        .collect();
    let total_capacity = num_hit_candidates.iter().sum();

    let mut hit_candidates_uninit = prealloc_maybeuninit_vec(total_capacity);
    let hc_chunks = get_disjoint_chunks_mut(&num_hit_candidates, &mut hit_candidates_uninit);

    convergent_indices
        .par_iter()
        .zip(hc_chunks.into_par_iter())
        .for_each(|(indices, chunk)| {
            let indices = indices.as_ref();
            let mut i = 0;
            for a in 0..indices.len() {
                for b in (a + 1)..indices.len() {
                    chunk[i].write((indices[a], indices[b]));
                    i += 1;
                }
            }
            debug_assert_eq!(i, chunk.len());
        });

    let mut hit_candidates = unsafe { cast_to_initialised_vec(hit_candidates_uninit) };

    // TODO: use new bucket-based parallel sorting here too? (and equivalent for _across variant)
    hit_candidates.par_sort_unstable();
    hit_candidates.dedup();

    hit_candidates
}

fn get_hit_candidates_across<T, U>(convergent_indices: &[(T, U)]) -> Vec<(u32, u32)>
where
    T: AsRef<[u32]> + Sync,
    U: AsRef<[u32]> + Sync,
{
    let num_hit_candidates: Vec<_> = convergent_indices
        .iter()
        .map(|(qi, ri)| qi.as_ref().len() * ri.as_ref().len())
        .collect();
    let total_capacity = num_hit_candidates.iter().sum();

    let mut hit_candidates_uninit = prealloc_maybeuninit_vec(total_capacity);
    let hc_chunks = get_disjoint_chunks_mut(&num_hit_candidates, &mut hit_candidates_uninit);

    convergent_indices
        .par_iter()
        .zip(hc_chunks.into_par_iter())
        .for_each(|((indices_q, indices_r), chunk)| {
            let indices_q = indices_q.as_ref();
            let indices_r = indices_r.as_ref();
            let mut i = 0;
            for &q in indices_q {
                for &r in indices_r {
                    chunk[i].write((q, r));
                    i += 1;
                }
            }
            debug_assert_eq!(i, chunk.len());
        });

    let mut hit_candidates = unsafe { cast_to_initialised_vec(hit_candidates_uninit) };

    hit_candidates.par_sort_unstable();
    hit_candidates.dedup();

    hit_candidates
}

fn compute_dists<M: Metric>(
    hit_candidates: &[(u32, u32)],
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: MaxDistance,
) -> Vec<u8> {
    hit_candidates
        .par_iter()
        .map(|&(idx_query, idx_reference)| {
            M::distance(
                query[idx_query as usize].as_ref(),
                reference[idx_reference as usize].as_ref(),
                max_distance.as_usize(),
            )
        })
        .collect()
}

/// Examine and double check hits to see if they are real, then collect into a tuple of vectors.
fn validate_and_collect_hits(
    hit_candidates: Vec<(u32, u32)>,
    dists: Vec<u8>,
    max_distance: MaxDistance,
) -> NeighborPairs {
    let mut qi_filtered = Vec::with_capacity(dists.len());
    let mut ri_filtered = Vec::with_capacity(dists.len());
    let mut dists_filtered = Vec::with_capacity(dists.len());

    for ((qi, ri), d) in hit_candidates.into_iter().zip(dists) {
        if d > max_distance.as_u8() {
            continue;
        }
        qi_filtered.push(qi);
        ri_filtered.push(ri);
        dists_filtered.push(d);
    }

    qi_filtered.shrink_to_fit();
    ri_filtered.shrink_to_fit();
    dists_filtered.shrink_to_fit();

    NeighborPairs {
        row: qi_filtered,
        col: ri_filtered,
        dists: dists_filtered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, BufRead, Cursor};

    // component tests

    fn pack(hash: u32, idx: u32) -> VariantIndexPair {
        VariantIndexPair::new(hash, idx)
    }

    fn pack_ci(hash: u32, ci: CrossIndex) -> VariantIndexPair {
        VariantIndexPair::from_index(hash, ci)
    }

    fn assert_bucket_bounds_ok(vip: &[VariantIndexPair], bounds: &[usize]) {
        assert_eq!(*bounds.first().unwrap(), 0);
        assert_eq!(*bounds.last().unwrap(), vip.len());
        for window in bounds.windows(2) {
            assert!(window[0] <= window[1]);
        }
        for &b in &bounds[1..bounds.len().saturating_sub(1)] {
            if b == 0 || b >= vip.len() {
                continue;
            }
            assert_ne!(
                vip[b - 1].variant_hash(),
                vip[b].variant_hash(),
                "boundary {b} splits a hash group"
            );
        }
    }

    #[test]
    fn test_variant_index_pair_layout() {
        assert_eq!(std::mem::size_of::<VariantIndexPair>(), 8);
        assert_eq!(std::mem::align_of::<VariantIndexPair>(), 8);
    }

    #[test]
    fn test_bucket_sort_matches_par_sort() {
        let assert_matches = |vip: Vec<VariantIndexPair>| {
            let mut expected = vip.clone();
            expected.par_sort_unstable();
            let (got, bounds) = bucket_sort(vip.clone());
            assert_eq!(got, expected);
            assert_bucket_bounds_ok(&got, &bounds);
        };

        assert_matches(vec![]);
        assert_matches(vec![pack(1, 0)]);
        assert_matches((0..50).map(|i| pack(42, i)).collect());
        assert_matches(vec![pack(0, 0), pack(0, 1), pack(0, 2)]);
        assert_matches(vec![
            pack(0, 0),
            pack(0, 1),
            pack(u32::MAX, 2),
            pack(u32::MAX, 3),
        ]);
        assert_matches(vec![
            pack(1, 0),
            pack(1, 0),
            pack(1, 1),
            pack(2, 2),
            pack(2, 2),
        ]);

        // Only bucket 0 and only bucket 255.
        assert_matches(vec![pack(0x00_12_34_56, 0), pack(0x00_ab_cd_ef, 1)]);
        assert_matches(vec![pack(0xff_12_34_56, 0), pack(0xff_ab_cd_ef, 1)]);

        // Pseudorandom via inline xorshift (no rand dependency).
        let mut state = 0x1234_5678_u64;
        let mut vip = Vec::with_capacity(200);
        for i in 0..200u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vip.push(pack((state >> 32) as u32, i));
        }
        assert_matches(vip);

        // CrossIndex flavour: type bit in the low half must survive the sort order.
        let q = |i| CrossIndex::from(i, false);
        let r = |i| CrossIndex::from(i, true);
        let vip = vec![
            pack_ci(5, r(1)),
            pack_ci(5, q(3)),
            pack_ci(5, q(1)),
            pack_ci(1, r(0)),
            pack_ci(1, q(0)),
        ];
        let mut expected = vip.clone();
        expected.par_sort_unstable();
        let (got, _) = bucket_sort(vip);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_bucket_bounds_never_split_a_hash_group() {
        let cases: Vec<Vec<VariantIndexPair>> = vec![
            vec![],
            vec![pack(1, 0)],
            (0..50).map(|i| pack(42, i)).collect(),
            {
                let mut vip = Vec::new();
                for i in 0..20u32 {
                    vip.push(pack(1, i));
                }
                for i in 0..10u32 {
                    vip.push(pack(2, i));
                }
                for i in 0..5u32 {
                    vip.push(pack(3, i));
                }
                vip
            },
        ];
        for vip in cases {
            let (sorted, bounds) = bucket_sort(vip);
            assert_bucket_bounds_ok(&sorted, &bounds);
            if !sorted.is_empty() {
                assert_eq!(bounds.len(), VariantIndexPair::NUM_BUCKETS + 1);
            }
        }
    }

    #[test]
    fn test_bucket_sort_orders_query_before_reference() {
        let q = |i| CrossIndex::from(i, false);
        let r = |i| CrossIndex::from(i, true);
        let vip = vec![
            pack_ci(7, r(2)),
            pack_ci(7, q(5)),
            pack_ci(7, r(1)),
            pack_ci(7, q(0)),
            pack_ci(3, r(0)),
            pack_ci(3, q(1)),
        ];
        let (sorted, _) = bucket_sort(vip);

        for group in groups(&sorted) {
            let mut seen_ref = false;
            for word in distinct(group) {
                let ci = CrossIndex::from_index_bits(word.index_bits());
                if ci.is_ref() {
                    seen_ref = true;
                } else {
                    assert!(!seen_ref, "query index after reference within a hash group");
                }
            }
        }
    }

    #[test]
    fn test_collect_convergent_indices() {
        let within = |vip: Vec<VariantIndexPair>| {
            collect_convergent_indices::<u32, _>(vip, |group| {
                let len = distinct(group).count();
                (len > 1).then_some((len, len))
            })
        };
        let cross = |vip: Vec<VariantIndexPair>| {
            collect_convergent_indices::<CrossIndex, _>(vip, |group| {
                let (len_q, len_r) = distinct(group).fold((0, 0), |(q, r), word| {
                    if CrossIndex::from_index_bits(word.index_bits()).is_ref() {
                        (q, r + 1)
                    } else {
                        (q + 1, r)
                    }
                });
                (len_q > 0 && len_r > 0).then_some((len_q + len_r, (len_q, len_r)))
            })
        };

        // Singletons dropped; one multi-index group kept; duplicates collapsed.
        assert_eq!(
            within(vec![pack(1, 0), pack(2, 1), pack(3, 2)]),
            (vec![], vec![])
        );
        assert_eq!(
            within(vec![pack(1, 0), pack(1, 1), pack(1, 2)]),
            (vec![0, 1, 2], vec![3])
        );
        assert_eq!(
            within(vec![
                pack(1, 0),
                pack(1, 0),
                pack(1, 1),
                pack(2, 2),
                pack(3, 3),
                pack(3, 4),
                pack(3, 4),
                pack(4, 5),
            ]),
            (vec![0, 1, 3, 4], vec![2, 2])
        );
        // Unsorted input is sorted first.
        assert_eq!(
            within(vec![
                pack(3, 1),
                pack(1, 0),
                pack(3, 0),
                pack(2, 2),
                pack(1, 1)
            ]),
            (vec![0, 1, 0, 1], vec![2, 2])
        );
        assert_eq!(within(vec![]), (vec![], vec![]));

        // Same-side-only groups dropped; cross group keeps query then ref indices.
        let q = |i| CrossIndex::from(i, false);
        let r = |i| CrossIndex::from(i, true);
        assert_eq!(
            cross(vec![pack_ci(1, q(0)), pack_ci(1, q(1))]),
            (vec![], vec![])
        );
        assert_eq!(
            cross(vec![pack_ci(1, r(0)), pack_ci(1, r(1))]),
            (vec![], vec![])
        );
        assert_eq!(
            cross(vec![
                pack_ci(1, q(0)),
                pack_ci(1, q(0)),
                pack_ci(1, q(1)),
                pack_ci(1, r(0)),
                pack_ci(1, r(2)),
                pack_ci(1, r(2)),
                pack_ci(2, q(3)),
            ]),
            (vec![0, 1, 0, 2], vec![(2, 2)])
        );
        assert_eq!(cross(vec![]), (vec![], vec![]));
    }

    #[test]
    fn test_nck() {
        let cases = [(5, 2, 10), (5, 5, 1), (5, 0, 1)];
        for (n, k, expected) in cases {
            let result = get_num_k_combs(n, k);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_for_each_combination() {
        let mut ours = Vec::new();
        for_each_combination(5, 3, |idxs| ours.push(idxs.to_vec()));
        let expected = vec![
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [0, 2, 3],
            [0, 2, 4],
            [0, 3, 4],
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 4],
            [2, 3, 4],
        ];
        assert_eq!(ours, expected);

        // Also cover k > n (should yield nothing).
        let mut ours = Vec::new();
        for_each_combination(5, 6, |idxs| ours.push(idxs.to_vec()));
        assert!(ours.is_empty());
    }

    #[test]
    fn test_exact_null_full_deletions_hashes_null_bytes() {
        let input = "ab";
        let max_distance = MaxDistance::try_from(2).expect("legal");
        let mut chunk = prealloc_maybeuninit_vec(1);
        let hash_builder = FixedState::default();
        let mut scratch = Vec::new();

        write_vi_pairs_exact_null(
            input,
            0u32,
            max_distance,
            &mut chunk,
            &hash_builder,
            &mut scratch,
        );

        let pairs = unsafe { cast_to_initialised_vec(chunk) };
        let expected = hash_string([u8::MAX, u8::MAX], &hash_builder);
        assert_eq!(pairs[0].variant_hash(), expected);
        assert_eq!(pairs[0].index_bits(), 0);
    }

    #[test]
    fn test_get_num_del_vars_per_string() {
        let strings = ["foo".to_string(), "bar".to_string(), "baz".to_string()];
        let result =
            get_num_del_vars_per_string_up_to(&strings, MaxDistance::try_from(1).expect("legal"));
        assert_eq!(result, vec![4, 4, 4]);
    }

    const TEST_QUERY: [&str; 5] = ["fizz", "fuzz", "buzz", "izzy", "lofi"];
    const TEST_REF: [&str; 3] = ["file", "tofu", "fizz"];

    fn pair_combinations(n: u32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                out.push((a, b));
            }
        }
        out
    }

    fn cartesian_product(n: u32, m: u32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for a in 0..n {
            for b in 0..m {
                out.push((a, b));
            }
        }
        out
    }

    #[test]
    fn test_compute_dists() {
        let cases = [
            (
                pair_combinations(5),
                &TEST_QUERY[..],
                MaxDistance::try_from(1).expect("legal"),
                vec![1, 255, 255, 255, 1, 255, 255, 255, 255, 255],
            ),
            (
                pair_combinations(5),
                &TEST_QUERY[..],
                MaxDistance::try_from(2).expect("legal"),
                vec![1, 2, 2, 255, 1, 255, 255, 255, 255, 255],
            ),
            (
                cartesian_product(5, 3),
                &TEST_REF[..],
                MaxDistance::try_from(1).expect("legal"),
                vec![
                    255, 255, 0, 255, 255, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                ],
            ),
            (
                cartesian_product(5, 3),
                &TEST_REF[..],
                MaxDistance::try_from(2).expect("legal"),
                vec![
                    2, 255, 0, 255, 255, 1, 255, 255, 2, 255, 255, 2, 255, 2, 255,
                ],
            ),
        ];

        for (candidates, reference, mdist, expected) in cases {
            let results = compute_dists::<Levenshtein>(&candidates, &TEST_QUERY, reference, mdist);
            assert_eq!(results, expected);
        }
    }

    #[test]
    fn test_get_true_hits() {
        let cases = [
            (
                pair_combinations(5),
                vec![1, 255, 255, 255, 1, 255, 255, 255, 255, 255],
                MaxDistance::try_from(1).expect("legal"),
                NeighborPairs {
                    row: vec![0, 1],
                    col: vec![1, 2],
                    dists: vec![1, 1],
                },
            ),
            (
                pair_combinations(5),
                vec![1, 2, 2, 255, 1, 255, 255, 255, 255, 255],
                MaxDistance::try_from(2).expect("legal"),
                NeighborPairs {
                    row: vec![0, 0, 0, 1],
                    col: vec![1, 2, 3, 2],
                    dists: vec![1, 2, 2, 1],
                },
            ),
        ];

        for (candidates, dists, mdist, expected) in cases {
            let result = validate_and_collect_hits(candidates, dists, mdist);
            assert_eq!(result, expected);
        }
    }

    // testing on real world data

    static CDR3_Q_BYTES: &[u8] = include_bytes!("../../test_files/cdr3b_10k_a.txt");
    static CDR3_R_BYTES: &[u8] = include_bytes!("../../test_files/cdr3b_10k_b.txt");
    static EXPECTED_BYTES_WITHIN_1: &[u8] = include_bytes!("../../test_files/results_10k_a.txt");
    static EXPECTED_BYTES_WITHIN_2: &[u8] = include_bytes!("../../test_files/results_10k_a_d2.txt");
    static EXPECTED_BYTES_CROSS_1: &[u8] = include_bytes!("../../test_files/results_10k_cross.txt");
    static EXPECTED_BYTES_CROSS_2: &[u8] =
        include_bytes!("../../test_files/results_10k_cross_d2.txt");
    static EXPECTED_BYTES_HAMMING_WITHIN_1: &[u8] =
        include_bytes!("../../test_files/results_10k_a_hamming.txt");
    static EXPECTED_BYTES_HAMMING_WITHIN_2: &[u8] =
        include_bytes!("../../test_files/results_10k_a_hamming_d2.txt");
    static EXPECTED_BYTES_HAMMING_CROSS_1: &[u8] =
        include_bytes!("../../test_files/results_10k_cross_hamming.txt");
    static EXPECTED_BYTES_HAMMING_CROSS_2: &[u8] =
        include_bytes!("../../test_files/results_10k_cross_hamming_d2.txt");

    fn bytes_as_ascii_lines(bytes: &[u8]) -> Vec<String> {
        Cursor::new(bytes)
            .lines()
            .collect::<io::Result<Vec<String>>>()
            .expect("test files have valid lines")
    }

    fn bytes_as_neighbour_pairs(bytes: &[u8]) -> NeighborPairs {
        let mut i = Vec::new();
        let mut j = Vec::new();
        let mut dists = Vec::new();

        Cursor::new(bytes).lines().for_each(|v| {
            let line = v.expect("test files have valid lines");
            let triplet: Vec<_> = line.split(",").collect();
            i.push(
                triplet[0]
                    .parse::<u32>()
                    .expect("test files have int triplets")
                    - 1,
            );
            j.push(
                triplet[1]
                    .parse::<u32>()
                    .expect("test files have int triplets")
                    - 1,
            );
            dists.push(
                triplet[2]
                    .parse::<u8>()
                    .expect("test files have int triplets"),
            );
        });

        NeighborPairs {
            row: i,
            col: j,
            dists,
        }
    }

    #[test]
    fn test_within() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);

        let hits = get_neighbors_within(&query, 1).expect("short input");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_1));

        let hits = get_neighbors_within(&query, 2).expect("short input");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_2));
    }

    #[test]
    fn test_cross() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);

        let hits = get_neighbors_across(&query, &reference, 1).expect("valid inputs");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_1));

        let hits = get_neighbors_across(&query, &reference, 2).expect("valid inputs");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_2));
    }

    #[test]
    fn test_hamming_within() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);

        let hits = get_hamming_neighbors_within(&query, 1).expect("short input");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_1)
        );

        let hits = get_hamming_neighbors_within(&query, 2).expect("short input");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_2)
        );
    }

    #[test]
    fn test_hamming_cross() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);

        let hits = get_hamming_neighbors_across(&query, &reference, 1).expect("valid inputs");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_1)
        );

        let hits = get_hamming_neighbors_across(&query, &reference, 2).expect("valid inputs");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_2)
        );
    }

    #[test]
    fn test_within_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let cached = CachedRef::new(&query, 2).expect("short input");

        let hits = cached.get_neighbors_within(1).expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_1));

        let hits = cached.get_neighbors_within(2).expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_2));
    }

    #[test]
    fn test_hamming_within_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let cached_hamming = CachedRefHamming::new(&query, 2).expect("short input");

        let hits = cached_hamming
            .get_neighbors_within(1)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_1)
        );

        let hits = cached_hamming
            .get_neighbors_within(2)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_2)
        );
    }

    #[test]
    fn test_cross_partially_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached = CachedRef::new(&reference, 2).expect("short input");

        let hits = cached
            .get_neighbors_across(&query, 1)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_1));

        let hits = cached
            .get_neighbors_across(&query, 2)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_2));
    }

    #[test]
    fn test_cross_fully_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached_query = CachedRef::new(&query, 2).expect("short input");
        let cached_reference = CachedRef::new(&reference, 2).expect("short input");

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 1)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_1));

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 2)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_2));
    }

    #[test]
    fn test_hamming_cross_partially_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached_hamming = CachedRefHamming::new(&reference, 2).expect("short input");

        let hits = cached_hamming
            .get_neighbors_across(&query, 1)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_1)
        );

        let hits = cached_hamming
            .get_neighbors_across(&query, 2)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_2)
        );
    }

    #[test]
    fn test_hamming_cross_fully_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached_query = CachedRefHamming::new(&query, 2).expect("short input");
        let cached_reference = CachedRefHamming::new(&reference, 2).expect("short input");

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 1)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_1)
        );

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 2)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_2)
        );
    }
}
