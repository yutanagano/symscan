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
use itertools::Itertools;
use rapidfuzz::distance::{hamming, levenshtein};
use rayon::prelude::*;
use std::fmt::Display;
use std::hash::{BuildHasher, Hasher};
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
    }
}

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn write(&mut self, bytes: &[u8]) {
        unreachable!("hasher only designed for u64, got {bytes:?}");
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Default)]
struct IdentityHasherBuilder;

impl BuildHasher for IdentityHasherBuilder {
    type Hasher = IdentityHasher;

    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher::default()
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
    str_store: Vec<u8>,
    str_spans: Vec<Span>,
    index_store: Vec<u32>,
    variant_map: HashMap<u64, Span, IdentityHasherBuilder>,
    max_distance: MaxDistance,
}

impl CachedRef {
    /// Construct a new [`CachedRef`] instance.
    pub fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
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
            let strlens = reference.iter().map(|s| s.as_ref().len()).collect_vec();

            let mut str_store_uninit = prealloc_maybeuninit_vec(strlens.iter().sum());
            let str_spans = get_disjoint_spans(&strlens);
            let str_store_chunks = get_disjoint_chunks_mut(&strlens, &mut str_store_uninit[..]);

            reference
                .par_iter()
                .zip(str_store_chunks.into_par_iter())
                .with_min_len(100000)
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
                prealloc_maybeuninit_vec::<(u64, u32)>(num_vars_per_string.iter().sum());
            let vip_chunks =
                get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

            reference
                .par_iter()
                .zip(vip_chunks.into_par_iter())
                .enumerate()
                .with_min_len(100000)
                .for_each(|(idx, (s, chunk))| {
                    write_vi_pairs_rawidx(
                        s.as_ref(),
                        idx as u32,
                        max_distance,
                        chunk,
                        &hash_builder,
                    );
                });

            let mut variant_index_pairs =
                unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };

            variant_index_pairs.par_sort_unstable();
            variant_index_pairs.dedup();

            let mut total_num_convergent_indices = 0;
            let mut num_convergence_groups = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    total_num_convergent_indices += chunk.len();
                    num_convergence_groups += 1;
                });

            let mut convergent_indices = Vec::with_capacity(total_num_convergent_indices);
            let mut convergence_groups = Vec::with_capacity(num_convergence_groups);
            let mut cursor = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    convergent_indices.extend(chunk.iter().map(|&(_, i)| i));
                    convergence_groups.push((chunk[0].0, Span::new(cursor, chunk.len())));
                    cursor += chunk.len();
                });

            debug_assert_eq!(cursor, convergent_indices.len());

            (convergent_indices, convergence_groups)
        };

        let mut variant_map =
            HashMap::with_capacity_and_hasher(convergence_groups.len(), IdentityHasherBuilder);

        for (v_hash, index_range) in convergence_groups {
            variant_map.entry(v_hash).insert(index_range);
        }

        Ok(CachedRef {
            str_store,
            str_spans,
            index_store,
            variant_map,
            max_distance,
        })
    }

    /// The memoized equivalent of [`get_neighbors_within`].
    pub fn get_neighbors_within(&self, max_distance: u8) -> Result<NeighborPairs, Error> {
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

    /// The memoized equivalent of [`get_neighbors_across`].
    pub fn get_neighbors_across(
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
            let num_vars_per_string = get_num_del_vars_per_string_up_to(query, max_distance);

            let mut variant_index_pairs_uninit =
                prealloc_maybeuninit_vec(num_vars_per_string.iter().sum());
            let vip_chunks =
                get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

            let hash_builder = FixedState::default();

            query
                .par_iter()
                .zip(vip_chunks.into_par_iter())
                .enumerate()
                .with_min_len(100000)
                .for_each(|(idx, (s, chunk))| {
                    write_vi_pairs_rawidx(
                        s.as_ref(),
                        idx as u32,
                        max_distance,
                        chunk,
                        &hash_builder,
                    );
                });

            let mut variant_index_pairs =
                unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };

            variant_index_pairs.par_sort_unstable();
            variant_index_pairs.dedup();

            let mut total_num_convergent_q_indices = 0;
            let mut num_convergence_groups = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    let variant = &chunk[0].0;
                    if self.variant_map.get(variant).is_some() {
                        total_num_convergent_q_indices += chunk.len();
                        num_convergence_groups += 1;
                    }
                });

            let mut q_idx_store = Vec::with_capacity(total_num_convergent_q_indices);
            let mut convergence_groups = Vec::with_capacity(num_convergence_groups);
            let mut cursor = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    let variant = &chunk[0].0;
                    if let Some(span) = self.variant_map.get(variant) {
                        q_idx_store.extend(chunk.iter().map(|&(_, i)| i));
                        convergence_groups.push((
                            cursor..cursor + chunk.len(),
                            self.get_convergent_indices_from_span(span),
                        ));
                        cursor += chunk.len();
                    }
                });

            (q_idx_store, convergence_groups)
        };

        let convergence_groups = convergence_groups
            .into_iter()
            .map(|(r, s)| (&q_idx_store[r], s))
            .collect_vec();

        let candidates = get_hit_candidates_across(&convergence_groups);
        let dists = self.compute_dists_partially_cached(&candidates, query, max_distance);

        Ok(validate_and_collect_hits(candidates, dists, max_distance))
    }

    /// Equivalent to [`CachedRef::get_neighbors_across`], where the query is also a [`CachedRef`]
    /// instance.
    pub fn get_neighbors_across_cached(
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
            .with_min_len(100000)
            .map(|&(idx_query, idx_reference)| {
                let dist = {
                    match levenshtein::distance_with_args(
                        query[idx_query as usize].as_ref().bytes(),
                        self.get_str_at_index(idx_reference as usize).bytes(),
                        &levenshtein::Args::default().score_cutoff(max_distance.as_usize()),
                    ) {
                        None => u8::MAX,
                        Some(dist) => dist as u8,
                    }
                };

                dist
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
            .with_min_len(100000)
            .map(|&(idx_query, idx_reference)| {
                let dist = {
                    match levenshtein::distance_with_args(
                        query.get_str_at_index(idx_query as usize).bytes(),
                        self.get_str_at_index(idx_reference as usize).bytes(),
                        &levenshtein::Args::default().score_cutoff(max_distance.as_usize()),
                    ) {
                        None => u8::MAX,
                        Some(dist) => dist as u8,
                    }
                };

                dist
            })
            .collect()
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
    str_store: Vec<u8>,
    str_spans: Vec<Span>,
    index_store: Vec<u32>,
    variant_map: HashMap<u64, Span, IdentityHasherBuilder>,
    max_distance: MaxDistance,
}

impl CachedRefHamming {
    /// Construct a new [`CachedRefHamming`] instance.
    pub fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
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
            let strlens = reference.iter().map(|s| s.as_ref().len()).collect_vec();

            let mut str_store_uninit = prealloc_maybeuninit_vec(strlens.iter().sum());
            let str_spans = get_disjoint_spans(&strlens);
            let str_store_chunks = get_disjoint_chunks_mut(&strlens, &mut str_store_uninit[..]);

            reference
                .par_iter()
                .zip(str_store_chunks.into_par_iter())
                .with_min_len(100000)
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
                prealloc_maybeuninit_vec::<(u64, u32)>(num_vars_per_string.iter().sum());
            let vip_chunks =
                get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

            reference
                .par_iter()
                .zip(vip_chunks.into_par_iter())
                .enumerate()
                .with_min_len(100000)
                .for_each(|(idx, (s, chunk))| {
                    write_vi_pairs_cached_hamming(
                        s.as_ref(),
                        idx as u32,
                        max_distance,
                        chunk,
                        &hash_builder,
                    );
                });

            let mut variant_index_pairs =
                unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };

            variant_index_pairs.par_sort_unstable();
            variant_index_pairs.dedup();

            let mut total_num_convergent_indices = 0;
            let mut num_convergence_groups = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    total_num_convergent_indices += chunk.len();
                    num_convergence_groups += 1;
                });

            let mut convergent_indices = Vec::with_capacity(total_num_convergent_indices);
            let mut convergence_groups = Vec::with_capacity(num_convergence_groups);
            let mut cursor = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    convergent_indices.extend(chunk.iter().map(|&(_, i)| i));
                    convergence_groups.push((chunk[0].0, Span::new(cursor, chunk.len())));
                    cursor += chunk.len();
                });

            debug_assert_eq!(cursor, convergent_indices.len());

            (convergent_indices, convergence_groups)
        };

        let mut variant_map =
            HashMap::with_capacity_and_hasher(convergence_groups.len(), IdentityHasherBuilder);

        for (v_hash, index_range) in convergence_groups {
            variant_map.entry(v_hash).insert(index_range);
        }

        Ok(CachedRefHamming {
            str_store,
            str_spans,
            index_store,
            variant_map,
            max_distance,
        })
    }

    /// The memoized equivalent of [`get_hamming_neighbors_within`].
    pub fn get_neighbors_within(&self, max_distance: u8) -> Result<NeighborPairs, Error> {
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

    /// The memoized equivalent of [`get_hamming_neighbors_across`].
    pub fn get_neighbors_across(
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
            let num_vars_per_string = get_num_del_vars_per_string_at(query, max_distance);

            let mut variant_index_pairs_uninit =
                prealloc_maybeuninit_vec(num_vars_per_string.iter().sum());
            let vip_chunks =
                get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

            let hash_builder = FixedState::default();

            query
                .par_iter()
                .zip(vip_chunks.into_par_iter())
                .enumerate()
                .with_min_len(100000)
                .for_each(|(idx, (s, chunk))| {
                    write_vi_pairs_rawidx_hamming(
                        s.as_ref(),
                        idx as u32,
                        max_distance,
                        chunk,
                        &hash_builder,
                    );
                });

            let mut variant_index_pairs =
                unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };

            variant_index_pairs.par_sort_unstable();
            variant_index_pairs.dedup();

            let mut total_num_convergent_q_indices = 0;
            let mut num_convergence_groups = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    let variant = &chunk[0].0;
                    if self.variant_map.get(variant).is_some() {
                        total_num_convergent_q_indices += chunk.len();
                        num_convergence_groups += 1;
                    }
                });

            let mut q_idx_store = Vec::with_capacity(total_num_convergent_q_indices);
            let mut convergence_groups = Vec::with_capacity(num_convergence_groups);
            let mut cursor = 0;

            variant_index_pairs
                .chunk_by(|(v1, _), (v2, _)| v1 == v2)
                .for_each(|chunk| {
                    let variant = &chunk[0].0;
                    if let Some(span) = self.variant_map.get(variant) {
                        q_idx_store.extend(chunk.iter().map(|&(_, i)| i));
                        convergence_groups.push((
                            cursor..cursor + chunk.len(),
                            self.get_convergent_indices_from_span(span),
                        ));
                        cursor += chunk.len();
                    }
                });

            (q_idx_store, convergence_groups)
        };

        let convergence_groups = convergence_groups
            .into_iter()
            .map(|(r, s)| (&q_idx_store[r], s))
            .collect_vec();

        let candidates = get_hit_candidates_across(&convergence_groups);
        let dists = self.compute_dists_partially_cached(&candidates, query, max_distance);

        Ok(validate_and_collect_hits(candidates, dists, max_distance))
    }

    /// Equivalent to [`CachedRefHamming::get_neighbors_across`], where the query is also a
    /// [`CachedRefHamming`] instance.
    pub fn get_neighbors_across_cached(
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
            .with_min_len(100000)
            .map(|&(idx_query, idx_reference)| {
                debug_assert_eq!(
                    query[idx_query as usize].as_ref().len(),
                    self.get_str_at_index(idx_reference as usize).len()
                );

                match unsafe {
                    hamming::distance_with_args(
                        query[idx_query as usize].as_ref().bytes(),
                        self.get_str_at_index(idx_reference as usize).bytes(),
                        &hamming::Args::default().score_cutoff(max_distance.as_usize()),
                    )
                    .unwrap_unchecked()
                } {
                    None => u8::MAX,
                    Some(dist) => dist as u8,
                }
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
            .with_min_len(100000)
            .map(|&(idx_query, idx_reference)| {
                debug_assert_eq!(
                    query.get_str_at_index(idx_query as usize).len(),
                    self.get_str_at_index(idx_reference as usize).len()
                );

                match unsafe {
                    hamming::distance_with_args(
                        query.get_str_at_index(idx_query as usize).bytes(),
                        self.get_str_at_index(idx_reference as usize).bytes(),
                        &hamming::Args::default().score_cutoff(max_distance.as_usize()),
                    )
                    .unwrap_unchecked()
                } {
                    None => u8::MAX,
                    Some(dist) => dist as u8,
                }
            })
            .collect()
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
        let num_vars_per_string = get_num_del_vars_per_string_up_to(query, max_distance);

        let mut variant_index_pairs_uninit =
            prealloc_maybeuninit_vec(num_vars_per_string.iter().sum());
        let vip_chunks =
            get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

        let hash_builder = FixedState::default();

        query
            .par_iter()
            .zip(vip_chunks.into_par_iter())
            .enumerate()
            .with_min_len(100000)
            .for_each(|(idx, (s, chunk))| {
                write_vi_pairs_rawidx(s.as_ref(), idx as u32, max_distance, chunk, &hash_builder);
            });

        let variant_index_pairs = unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };
        collect_convergent_indices(variant_index_pairs)
    };

    let convergent_chunks = get_convergent_chunks(&group_sizes, &convergent_indices[..]);
    let candidates = get_hit_candidates_within(&convergent_chunks);
    let dists = compute_dists_levenshtein(&candidates, query, query, max_distance);

    Ok(validate_and_collect_hits(candidates, dists, max_distance))
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
        let num_del_variants_q = get_num_del_vars_per_string_up_to(query, max_distance);
        let num_del_variants_r = get_num_del_vars_per_string_up_to(reference, max_distance);

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
            .with_min_len(100000)
            .for_each(|(idx, (s, chunk))| {
                write_vi_pairs_ci(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    false,
                    chunk,
                    &hash_builder,
                );
            });
        reference
            .par_iter()
            .zip(vip_chunks_r.into_par_iter())
            .enumerate()
            .with_min_len(100000)
            .for_each(|(idx, (s, chunk))| {
                write_vi_pairs_ci(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    true,
                    chunk,
                    &hash_builder,
                );
            });

        let variant_index_pairs = unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };
        collect_convergent_indices_cross(variant_index_pairs)
    };

    let convergent_chunks = get_convergent_chunks_cross(&group_sizes, &convergent_indices[..]);
    let candidates = get_hit_candidates_across(&convergent_chunks);
    let dists = compute_dists_levenshtein(&candidates, query, reference, max_distance);

    Ok(validate_and_collect_hits(candidates, dists, max_distance))
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
        let num_vars_per_string = get_num_del_vars_per_string_at(query, max_distance);

        let mut variant_index_pairs_uninit =
            prealloc_maybeuninit_vec(num_vars_per_string.iter().sum());
        let vip_chunks =
            get_disjoint_chunks_mut(&num_vars_per_string, &mut variant_index_pairs_uninit[..]);

        let hash_builder = FixedState::default();

        query
            .par_iter()
            .zip(vip_chunks.into_par_iter())
            .enumerate()
            .with_min_len(100000)
            .for_each(|(idx, (s, chunk))| {
                write_vi_pairs_rawidx_hamming(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    chunk,
                    &hash_builder,
                );
            });

        let variant_index_pairs = unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };
        collect_convergent_indices(variant_index_pairs)
    };

    let convergent_chunks = get_convergent_chunks(&group_sizes, &convergent_indices[..]);
    let candidates = get_hit_candidates_within(&convergent_chunks);
    let dists = compute_dists_hamming(&candidates, query, query, max_distance);

    Ok(validate_and_collect_hits(candidates, dists, max_distance))
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
        let num_del_variants_q = get_num_del_vars_per_string_at(query, max_distance);
        let num_del_variants_r = get_num_del_vars_per_string_at(reference, max_distance);

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
            .with_min_len(100000)
            .for_each(|(idx, (s, chunk))| {
                write_vi_pairs_ci_hamming(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    false,
                    chunk,
                    &hash_builder,
                );
            });
        reference
            .par_iter()
            .zip(vip_chunks_r.into_par_iter())
            .enumerate()
            .with_min_len(100000)
            .for_each(|(idx, (s, chunk))| {
                write_vi_pairs_ci_hamming(
                    s.as_ref(),
                    idx as u32,
                    max_distance,
                    true,
                    chunk,
                    &hash_builder,
                );
            });

        let variant_index_pairs = unsafe { cast_to_initialised_vec(variant_index_pairs_uninit) };
        collect_convergent_indices_cross(variant_index_pairs)
    };

    let convergent_chunks = get_convergent_chunks_cross(&group_sizes, &convergent_indices[..]);
    let candidates = get_hit_candidates_across(&convergent_chunks);
    let dists = compute_dists_hamming(&candidates, query, reference, max_distance);

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
        .collect_vec()
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
        .collect_vec()
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

/// Given an input string and its index in the original input vector, generate all possible strings
/// after making at most max_deletions single-character deletions, compute their hash, and write
/// them into the slots in the provided chunk, as 2-tuples (hash, input_idx).
fn write_vi_pairs_rawidx(
    input: &str,
    input_idx: u32,
    max_deletions: MaxDistance,
    chunk: &mut [MaybeUninit<(u64, u32)>],
    hash_builder: &impl BuildHasher,
) {
    let input_length = input.len();

    chunk[0].write((hash_string(input, hash_builder), input_idx));

    let mut variant_idx = 1;
    let mut variant_buffer = Vec::with_capacity(input_length);
    for num_deletions in 1..=max_deletions.as_u8() {
        if num_deletions as usize > input_length {
            break;
        }

        for deletion_indices in (0..input_length).combinations(num_deletions as usize) {
            variant_buffer.clear();
            let mut offset = 0;

            for idx in deletion_indices {
                variant_buffer.extend_from_slice(&input.as_bytes()[offset..idx]);
                offset = idx + 1;
            }
            variant_buffer.extend_from_slice(&input.as_bytes()[offset..input_length]);

            chunk[variant_idx].write((hash_string(&variant_buffer, hash_builder), input_idx));
            variant_idx += 1;
        }
    }
}

/// Similar to write_vi_pairs_rawidx but with the indices wrapped in CrossIndex.
fn write_vi_pairs_ci(
    input: &str,
    input_idx: u32,
    max_deletions: MaxDistance,
    is_ref: bool,
    chunk: &mut [MaybeUninit<(u64, CrossIndex)>],
    hash_builder: &impl BuildHasher,
) {
    let input_length = input.len();

    chunk[0].write((
        hash_string(input, hash_builder),
        CrossIndex::from(input_idx, is_ref),
    ));

    let mut variant_idx = 1;
    let mut variant_buffer = Vec::with_capacity(input_length);
    for num_deletions in 1..=max_deletions.as_u8() {
        if num_deletions as usize > input_length {
            break;
        }

        for deletion_indices in (0..input_length).combinations(num_deletions as usize) {
            variant_buffer.clear();
            let mut offset = 0;

            for idx in deletion_indices {
                variant_buffer.extend_from_slice(&input.as_bytes()[offset..idx]);
                offset = idx + 1;
            }
            variant_buffer.extend_from_slice(&input.as_bytes()[offset..input_length]);

            chunk[variant_idx].write((
                hash_string(&variant_buffer, hash_builder),
                CrossIndex::from(input_idx, is_ref),
            ));
            variant_idx += 1;
        }
    }
}

/// Equivalent to write_vi_pairs_rawidx but for Hamming, where we only generate deletions of exactly
/// max_deletions characters.
fn write_vi_pairs_rawidx_hamming(
    input: &str,
    input_idx: u32,
    max_deletions: MaxDistance,
    chunk: &mut [MaybeUninit<(u64, u32)>],
    hash_builder: &impl BuildHasher,
) {
    const NULL_CHARACTER: u8 = u8::MAX;
    let input_length = input.len();
    let mut variant_buffer = Vec::with_capacity(input_length);

    if max_deletions.as_usize() >= input_length {
        variant_buffer.fill(NULL_CHARACTER);
        chunk[0].write((hash_string(variant_buffer, hash_builder), input_idx));
        return;
    }

    for (variant_idx, deletion_indices) in (0..input_length)
        .combinations(max_deletions.as_usize())
        .enumerate()
    {
        variant_buffer.clear();
        let mut cursor = 0;

        for idx in deletion_indices {
            variant_buffer.extend_from_slice(&input.as_bytes()[cursor..idx]);
            variant_buffer.push(NULL_CHARACTER);
            cursor = idx + 1;
        }
        variant_buffer.extend_from_slice(&input.as_bytes()[cursor..input_length]);

        chunk[variant_idx].write((hash_string(&variant_buffer, hash_builder), input_idx));
    }
}

/// Equivalent to write_vi_pairs_ci but for Hamming distance instead of Levenshtein.
fn write_vi_pairs_ci_hamming(
    input: &str,
    input_idx: u32,
    max_deletions: MaxDistance,
    is_ref: bool,
    chunk: &mut [MaybeUninit<(u64, CrossIndex)>],
    hash_builder: &impl BuildHasher,
) {
    const NULL_CHARACTER: u8 = u8::MAX;
    let input_length = input.len();
    let mut variant_buffer = Vec::with_capacity(input_length);

    if max_deletions.as_usize() >= input_length {
        variant_buffer.fill(NULL_CHARACTER);
        chunk[0].write((
            hash_string(variant_buffer, hash_builder),
            CrossIndex::from(input_idx, is_ref),
        ));
        return;
    }

    for (variant_idx, deletion_indices) in (0..input_length)
        .combinations(max_deletions.as_usize())
        .enumerate()
    {
        variant_buffer.clear();
        let mut cursor = 0;

        for idx in deletion_indices {
            variant_buffer.extend_from_slice(&input.as_bytes()[cursor..idx]);
            variant_buffer.push(NULL_CHARACTER);
            cursor = idx + 1;
        }
        variant_buffer.extend_from_slice(&input.as_bytes()[cursor..input_length]);

        chunk[variant_idx].write((
            hash_string(&variant_buffer, hash_builder),
            CrossIndex::from(input_idx, is_ref),
        ));
    }
}

/// Similar to write_vi_pairs_rawidx_hamming but for CachedRefHamming, where we store deletion
/// variants up to max_deletions deletions instead of only exactly max_deletions deletions.
fn write_vi_pairs_cached_hamming(
    input: &str,
    input_idx: u32,
    max_deletions: MaxDistance,
    chunk: &mut [MaybeUninit<(u64, u32)>],
    hash_builder: &impl BuildHasher,
) {
    const NULL_CHARACTER: u8 = u8::MAX;
    let input_length = input.len();

    chunk[0].write((hash_string(input, hash_builder), input_idx));

    let mut variant_idx = 1;
    let mut variant_buffer = Vec::with_capacity(input_length);
    for num_deletions in 1..=max_deletions.as_u8() {
        if num_deletions as usize > input_length {
            break;
        }

        for deletion_indices in (0..input_length).combinations(num_deletions as usize) {
            variant_buffer.clear();
            let mut cursor = 0;

            for idx in deletion_indices {
                variant_buffer.extend_from_slice(&input.as_bytes()[cursor..idx]);
                variant_buffer.push(NULL_CHARACTER);
                cursor = idx + 1;
            }
            variant_buffer.extend_from_slice(&input.as_bytes()[cursor..input_length]);

            chunk[variant_idx].write((hash_string(&variant_buffer, hash_builder), input_idx));
            variant_idx += 1;
        }
    }
}

fn hash_string(s: impl AsRef<[u8]>, hash_builder: &impl BuildHasher) -> u64 {
    let mut hasher = hash_builder.build_hasher();
    hasher.write(s.as_ref());
    hasher.finish()
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

fn collect_convergent_indices(mut variant_index_pairs: Vec<(u64, u32)>) -> (Vec<u32>, Vec<usize>) {
    variant_index_pairs.par_sort_unstable();
    variant_index_pairs.dedup();

    let mut total_num_convergent_indices = 0;
    let mut num_convergence_groups = 0;

    variant_index_pairs
        .chunk_by(|(v1, _), (v2, _)| v1 == v2)
        .filter(|chunk| chunk.len() > 1)
        .for_each(|chunk| {
            total_num_convergent_indices += chunk.len();
            num_convergence_groups += 1;
        });

    let mut convergent_indices = Vec::with_capacity(total_num_convergent_indices);
    let mut convergence_group_sizes = Vec::with_capacity(num_convergence_groups);

    variant_index_pairs
        .chunk_by(|(v1, _), (v2, _)| v1 == v2)
        .filter(|chunk| chunk.len() > 1)
        .for_each(|chunk| {
            convergent_indices.extend(chunk.iter().map(|&(_, i)| i));
            convergence_group_sizes.push(chunk.len());
        });

    (convergent_indices, convergence_group_sizes)
}

fn collect_convergent_indices_cross(
    mut variant_index_pairs: Vec<(u64, CrossIndex)>,
) -> (Vec<u32>, Vec<(usize, usize)>) {
    variant_index_pairs.par_sort_unstable();
    variant_index_pairs.dedup();

    let mut total_num_convergent_indices = 0;
    let mut num_convergence_groups = 0;

    variant_index_pairs
        .chunk_by(|(v1, _), (v2, _)| v1 == v2)
        .filter(|chunk| chunk.len() > 1)
        .for_each(|chunk| {
            total_num_convergent_indices += chunk.len();
            num_convergence_groups += 1;
        });

    let mut convergent_indices = Vec::with_capacity(total_num_convergent_indices);
    let mut convergence_group_sizes = Vec::with_capacity(num_convergence_groups);

    variant_index_pairs
        .chunk_by(|(v1, _), (v2, _)| v1 == v2)
        .filter(|chunk| chunk.len() > 1)
        .map(|chunk| {
            let len_q = chunk.iter().filter(|(_, ci)| !ci.is_ref()).count();
            let len_r = chunk.iter().filter(|(_, ci)| ci.is_ref()).count();
            (chunk, len_q, len_r)
        })
        .filter(|(_, len_q, len_r)| len_q * len_r > 0)
        .for_each(|(chunk, len_q, len_r)| {
            convergent_indices.extend(
                chunk
                    .iter()
                    .filter(|(_, ci)| !ci.is_ref())
                    .map(|&(_, ci)| ci.get_value()),
            );
            convergent_indices.extend(
                chunk
                    .iter()
                    .filter(|(_, ci)| ci.is_ref())
                    .map(|&(_, ci)| ci.get_value()),
            );

            convergence_group_sizes.push((len_q, len_r));
        });

    (convergent_indices, convergence_group_sizes)
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
    let num_hit_candidates = convergent_indices
        .iter()
        .map(|indices| get_num_k_combs(indices.as_ref().len(), 2))
        .collect_vec();
    let total_capacity = num_hit_candidates.iter().sum();

    let mut hit_candidates_uninit = prealloc_maybeuninit_vec(total_capacity);
    let hc_chunks = get_disjoint_chunks_mut(&num_hit_candidates, &mut hit_candidates_uninit);

    convergent_indices
        .par_iter()
        .zip(hc_chunks.into_par_iter())
        .with_min_len(100000)
        .for_each(|(indices, chunk)| {
            for (i, candidate) in indices
                .as_ref()
                .iter()
                .copied()
                .tuple_combinations()
                .enumerate()
            {
                chunk[i].write(candidate);
            }
        });

    let mut hit_candidates = unsafe { cast_to_initialised_vec(hit_candidates_uninit) };

    hit_candidates.par_sort_unstable();
    hit_candidates.dedup();

    hit_candidates
}

fn get_hit_candidates_across<T, U>(convergent_indices: &[(T, U)]) -> Vec<(u32, u32)>
where
    T: AsRef<[u32]> + Sync,
    U: AsRef<[u32]> + Sync,
{
    let num_hit_candidates = convergent_indices
        .iter()
        .map(|(qi, ri)| qi.as_ref().len() * ri.as_ref().len())
        .collect_vec();
    let total_capacity = num_hit_candidates.iter().sum();

    let mut hit_candidates_uninit = prealloc_maybeuninit_vec(total_capacity);
    let hc_chunks = get_disjoint_chunks_mut(&num_hit_candidates, &mut hit_candidates_uninit);

    convergent_indices
        .par_iter()
        .zip(hc_chunks.into_par_iter())
        .with_min_len(100000)
        .for_each(|((indices_q, indices_r), chunk)| {
            for (i, candidate) in indices_q
                .as_ref()
                .iter()
                .copied()
                .cartesian_product(indices_r.as_ref().iter().copied())
                .enumerate()
            {
                chunk[i].write(candidate);
            }
        });

    let mut hit_candidates = unsafe { cast_to_initialised_vec(hit_candidates_uninit) };

    hit_candidates.par_sort_unstable();
    hit_candidates.dedup();

    hit_candidates
}

fn compute_dists_levenshtein(
    hit_candidates: &[(u32, u32)],
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: MaxDistance,
) -> Vec<u8> {
    hit_candidates
        .par_iter()
        .with_min_len(100000)
        .map(|&(idx_query, idx_reference)| {
            match levenshtein::distance_with_args(
                query[idx_query as usize].as_ref().bytes(),
                reference[idx_reference as usize].as_ref().bytes(),
                &levenshtein::Args::default().score_cutoff(max_distance.as_usize()),
            ) {
                None => u8::MAX,
                Some(dist) => dist as u8,
            }
        })
        .collect()
}

fn compute_dists_hamming(
    hit_candidates: &[(u32, u32)],
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: MaxDistance,
) -> Vec<u8> {
    hit_candidates
        .par_iter()
        .with_min_len(100000)
        .map(|&(idx_query, idx_reference)| {
            debug_assert!(
                query[idx_query as usize].as_ref().len()
                    == reference[idx_reference as usize].as_ref().len()
            );

            match unsafe {
                hamming::distance_with_args(
                    query[idx_query as usize].as_ref().bytes(),
                    reference[idx_reference as usize].as_ref().bytes(),
                    &hamming::Args::default().score_cutoff(max_distance.as_usize()),
                )
                .unwrap_unchecked()
            } {
                None => u8::MAX,
                Some(dist) => dist as u8,
            }
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

    #[test]
    fn test_nck() {
        let cases = [(5, 2, 10), (5, 5, 1), (5, 0, 1)];
        for (n, k, expected) in cases {
            let result = get_num_k_combs(n, k);
            assert_eq!(result, expected);
        }
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

    #[test]
    fn test_compute_dists() {
        let cases = [
            (
                (0..5).tuple_combinations().collect_vec(),
                &TEST_QUERY[..],
                MaxDistance::try_from(1).expect("legal"),
                vec![1, 255, 255, 255, 1, 255, 255, 255, 255, 255],
            ),
            (
                (0..5).tuple_combinations().collect_vec(),
                &TEST_QUERY[..],
                MaxDistance::try_from(2).expect("legal"),
                vec![1, 2, 2, 255, 1, 255, 255, 255, 255, 255],
            ),
            (
                (0..5).cartesian_product(0..3).collect_vec(),
                &TEST_REF[..],
                MaxDistance::try_from(1).expect("legal"),
                vec![
                    255, 255, 0, 255, 255, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                ],
            ),
            (
                (0..5).cartesian_product(0..3).collect_vec(),
                &TEST_REF[..],
                MaxDistance::try_from(2).expect("legal"),
                vec![
                    2, 255, 0, 255, 255, 1, 255, 255, 2, 255, 255, 2, 255, 2, 255,
                ],
            ),
        ];

        for (candidates, reference, mdist, expected) in cases {
            let results = compute_dists_levenshtein(&candidates, &TEST_QUERY, reference, mdist);
            assert_eq!(results, expected);
        }
    }

    #[test]
    fn test_get_true_hits() {
        let cases = [
            (
                (0..5).tuple_combinations().collect_vec(),
                vec![1, 255, 255, 255, 1, 255, 255, 255, 255, 255],
                MaxDistance::try_from(1).expect("legal"),
                NeighborPairs {
                    row: vec![0, 1],
                    col: vec![1, 2],
                    dists: vec![1, 1],
                },
            ),
            (
                (0..5).tuple_combinations().collect_vec(),
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
            let triplet = line.split(",").collect_vec();
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
