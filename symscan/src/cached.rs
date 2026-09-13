use std::{
    hash::{BuildHasher, Hasher},
    marker::PhantomData,
    ops::Range,
    ptr,
};

use foldhash::fast::FixedState;
use hashbrown::HashMap;
use rayon::prelude::*;

use crate::{
    core::{self, MaxDistance},
    indexing::VariantIndexPair,
    metric::{self, Metric},
    Error, InputType, NeighborPairs,
};

/// Multiplier for spreading a 32-bit key over 64 bits: the odd integer nearest `2^64 / phi`.
///
/// Multiplying by an odd constant is a bijection modulo 2^64, and this constant's bit pattern is
/// dense enough that carries propagate every input bit into the high bits of the product. Known
/// as Knuth's multiplicative hashing, or Fibonacci hashing.
const GOLDEN_RATIO_64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Shared memoized deletion-variant store used by [`CachedRef`] and [`CachedRefHamming`].
pub struct CachedStore<M: Metric> {
    str_store: Vec<u8>,
    str_spans: Vec<Span>,
    index_store: Vec<u32>,
    variant_map: HashMap<u32, Span, VariantHasherBuilder>,
    max_distance: MaxDistance,
    _metric: PhantomData<M>,
}

impl<M: Metric> CachedStore<M> {
    pub fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
        if reference.len() > u32::MAX as usize {
            return Err(Error::TooManyStrings {
                input_type: InputType::Reference,
                got: reference.len(),
                limit: u32::MAX as usize,
            });
        }
        let max_distance = MaxDistance::try_from(max_distance)?;
        core::check_strings_ascii(reference, InputType::Reference)?;

        let (str_store, str_spans) = {
            let strlens: Vec<_> = reference.iter().map(|s| s.as_ref().len()).collect();

            let mut str_store_uninit = core::prealloc_maybeuninit_vec(strlens.iter().sum());
            let str_spans = get_disjoint_spans(&strlens);
            let str_store_chunks =
                core::get_disjoint_chunks_mut(&strlens, &mut str_store_uninit[..]);

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

            let str_store = unsafe { core::cast_to_initialised_vec(str_store_uninit) };

            (str_store, str_spans)
        };

        let hash_builder = FixedState::default();

        let (index_store, convergence_groups) = {
            let num_vars_per_string =
                metric::get_num_del_vars_per_string_up_to(reference, max_distance);

            let mut variant_index_pairs_uninit = core::prealloc_maybeuninit_vec::<VariantIndexPair>(
                num_vars_per_string.iter().sum(),
            );
            let vip_chunks = core::get_disjoint_chunks_mut(
                &num_vars_per_string,
                &mut variant_index_pairs_uninit[..],
            );

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
                unsafe { core::cast_to_initialised_vec(variant_index_pairs_uninit) };

            // Every group is kept, including singletons: the variant map must hold every reference
            // variant for cross queries to find it.
            core::collect_convergent_indices::<u32, _>(variant_index_pairs, |group| {
                let len = core::distinct(group).count();
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

        let candidates = core::get_hit_candidates_within(&convergent_indices);
        let dists = self.compute_dists_fully_cached(&candidates, self, max_distance);

        Ok(core::validate_and_collect_hits(
            candidates,
            dists,
            max_distance,
        ))
    }

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
        core::check_strings_ascii(query, InputType::Query)?;

        let (q_idx_store, convergence_groups) = {
            let num_vars_per_string = M::count_oneshot(query, max_distance);

            let mut variant_index_pairs_uninit =
                core::prealloc_maybeuninit_vec(num_vars_per_string.iter().sum());
            let vip_chunks = core::get_disjoint_chunks_mut(
                &num_vars_per_string,
                &mut variant_index_pairs_uninit[..],
            );

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
                unsafe { core::cast_to_initialised_vec(variant_index_pairs_uninit) };

            core::collect_convergent_indices::<u32, _>(variant_index_pairs, |group| {
                let span = self.variant_map.get(&group[0].variant_hash())?;
                let len_q = core::distinct(group).count();
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

        let candidates = core::get_hit_candidates_across(&convergence_groups);
        let dists = self.compute_dists_partially_cached(&candidates, query, max_distance);

        Ok(core::validate_and_collect_hits(
            candidates,
            dists,
            max_distance,
        ))
    }

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

        let candidates = core::get_hit_candidates_across(&convergence_groups);
        let dists = self.compute_dists_fully_cached(&candidates, query, max_distance);

        Ok(core::validate_and_collect_hits(
            candidates,
            dists,
            max_distance,
        ))
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

fn get_disjoint_spans(span_lens: &[usize]) -> Vec<Span> {
    let mut spans = Vec::with_capacity(span_lens.len());
    let mut cursor = 0;
    for &n in span_lens {
        spans.push(Span::new(cursor, n));
        cursor += n;
    }
    spans
}
