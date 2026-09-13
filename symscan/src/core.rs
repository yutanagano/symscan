use std::mem::MaybeUninit;

use rayon::prelude::*;

use crate::{
    indexing::{VariantIndex, VariantIndexPair},
    metric::Metric,
    Error, InputType, NeighborPairs,
};

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

pub fn check_strings_ascii(
    strings: &[impl AsRef<str>],
    input_type: InputType,
) -> Result<(), Error> {
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

pub fn prealloc_maybeuninit_vec<T>(total_capacity: usize) -> Vec<MaybeUninit<T>> {
    let mut v: Vec<MaybeUninit<T>> = Vec::with_capacity(total_capacity);
    unsafe { v.set_len(total_capacity) };
    v
}

pub unsafe fn cast_to_initialised_vec<T>(mut input: Vec<MaybeUninit<T>>) -> Vec<T> {
    let ptr = input.as_mut_ptr() as *mut T;
    let len = input.len();
    let cap = input.capacity();
    std::mem::forget(input);
    Vec::from_raw_parts(ptr, len, cap)
}

pub fn get_disjoint_chunks_mut<'a, T>(
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
pub fn get_disjoint_chunks_mut_cross<'a, T>(
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

/// Entries of one group with adjacent duplicates skipped, replacing the removed `Vec::dedup`.
#[inline]
pub fn distinct(group: &[VariantIndexPair]) -> impl Iterator<Item = VariantIndexPair> + '_ {
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
pub fn collect_convergent_indices<I: VariantIndex, Payload: Copy + Send>(
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
pub fn get_convergent_chunks<'a, T>(
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
pub fn get_convergent_chunks_cross<'a, T>(
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

pub fn get_hit_candidates_within(
    convergent_indices: &[impl AsRef<[u32]> + Sync],
) -> Vec<(u32, u32)> {
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

pub fn get_hit_candidates_across<T, U>(convergent_indices: &[(T, U)]) -> Vec<(u32, u32)>
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

pub fn compute_dists<M: Metric>(
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
pub fn validate_and_collect_hits(
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

pub fn get_num_k_combs(n: usize, k: u8) -> usize {
    debug_assert!(n > 0);
    debug_assert!(n >= k as usize);

    if k == 0 {
        return 1;
    }

    let num_subsamples: usize = (n - k as usize + 1..=n).product();
    let subsample_perms: usize = (1..=k as usize).product();

    num_subsamples / subsample_perms
}

/// The runs of equal variant hashes in a sorted VIP slice.
#[inline]
fn groups(vip: &[VariantIndexPair]) -> impl Iterator<Item = &[VariantIndexPair]> {
    vip.chunk_by(|a, b| a.variant_hash() == b.variant_hash())
}

#[cfg(test)]
mod tests {
    use crate::{indexing::CrossIndex, metric::Levenshtein};

    use super::*;

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
        assert_matches(vec![VariantIndexPair::new(1, 0)]);
        assert_matches((0..50).map(|i| VariantIndexPair::new(42, i)).collect());
        assert_matches(vec![
            VariantIndexPair::new(0, 0),
            VariantIndexPair::new(0, 1),
            VariantIndexPair::new(0, 2),
        ]);
        assert_matches(vec![
            VariantIndexPair::new(0, 0),
            VariantIndexPair::new(0, 1),
            VariantIndexPair::new(u32::MAX, 2),
            VariantIndexPair::new(u32::MAX, 3),
        ]);
        assert_matches(vec![
            VariantIndexPair::new(1, 0),
            VariantIndexPair::new(1, 0),
            VariantIndexPair::new(1, 1),
            VariantIndexPair::new(2, 2),
            VariantIndexPair::new(2, 2),
        ]);

        // Only bucket 0 and only bucket 255.
        assert_matches(vec![
            VariantIndexPair::new(0x00_12_34_56, 0),
            VariantIndexPair::new(0x00_ab_cd_ef, 1),
        ]);
        assert_matches(vec![
            VariantIndexPair::new(0xff_12_34_56, 0),
            VariantIndexPair::new(0xff_ab_cd_ef, 1),
        ]);

        // Pseudorandom via inline xorshift (no rand dependency).
        let mut state = 0x1234_5678_u64;
        let mut vip = Vec::with_capacity(200);
        for i in 0..200u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vip.push(VariantIndexPair::new((state >> 32) as u32, i));
        }
        assert_matches(vip);

        // CrossIndex flavour: type bit in the low half must survive the sort order.
        let q = |i| CrossIndex::from(i, false);
        let r = |i| CrossIndex::from(i, true);
        let vip = vec![
            VariantIndexPair::from_index(5, r(1)),
            VariantIndexPair::from_index(5, q(3)),
            VariantIndexPair::from_index(5, q(1)),
            VariantIndexPair::from_index(1, r(0)),
            VariantIndexPair::from_index(1, q(0)),
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
            vec![VariantIndexPair::new(1, 0)],
            (0..50).map(|i| VariantIndexPair::new(42, i)).collect(),
            {
                let mut vip = Vec::new();
                for i in 0..20u32 {
                    vip.push(VariantIndexPair::new(1, i));
                }
                for i in 0..10u32 {
                    vip.push(VariantIndexPair::new(2, i));
                }
                for i in 0..5u32 {
                    vip.push(VariantIndexPair::new(3, i));
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
            VariantIndexPair::from_index(7, r(2)),
            VariantIndexPair::from_index(7, q(5)),
            VariantIndexPair::from_index(7, r(1)),
            VariantIndexPair::from_index(7, q(0)),
            VariantIndexPair::from_index(3, r(0)),
            VariantIndexPair::from_index(3, q(1)),
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
            within(vec![
                VariantIndexPair::new(1, 0),
                VariantIndexPair::new(2, 1),
                VariantIndexPair::new(3, 2)
            ]),
            (vec![], vec![])
        );
        assert_eq!(
            within(vec![
                VariantIndexPair::new(1, 0),
                VariantIndexPair::new(1, 1),
                VariantIndexPair::new(1, 2)
            ]),
            (vec![0, 1, 2], vec![3])
        );
        assert_eq!(
            within(vec![
                VariantIndexPair::new(1, 0),
                VariantIndexPair::new(1, 0),
                VariantIndexPair::new(1, 1),
                VariantIndexPair::new(2, 2),
                VariantIndexPair::new(3, 3),
                VariantIndexPair::new(3, 4),
                VariantIndexPair::new(3, 4),
                VariantIndexPair::new(4, 5),
            ]),
            (vec![0, 1, 3, 4], vec![2, 2])
        );
        // Unsorted input is sorted first.
        assert_eq!(
            within(vec![
                VariantIndexPair::new(3, 1),
                VariantIndexPair::new(1, 0),
                VariantIndexPair::new(3, 0),
                VariantIndexPair::new(2, 2),
                VariantIndexPair::new(1, 1)
            ]),
            (vec![0, 1, 0, 1], vec![2, 2])
        );
        assert_eq!(within(vec![]), (vec![], vec![]));

        // Same-side-only groups dropped; cross group keeps query then ref indices.
        let q = |i| CrossIndex::from(i, false);
        let r = |i| CrossIndex::from(i, true);
        assert_eq!(
            cross(vec![
                VariantIndexPair::from_index(1, q(0)),
                VariantIndexPair::from_index(1, q(1))
            ]),
            (vec![], vec![])
        );
        assert_eq!(
            cross(vec![
                VariantIndexPair::from_index(1, r(0)),
                VariantIndexPair::from_index(1, r(1))
            ]),
            (vec![], vec![])
        );
        assert_eq!(
            cross(vec![
                VariantIndexPair::from_index(1, q(0)),
                VariantIndexPair::from_index(1, q(0)),
                VariantIndexPair::from_index(1, q(1)),
                VariantIndexPair::from_index(1, r(0)),
                VariantIndexPair::from_index(1, r(2)),
                VariantIndexPair::from_index(1, r(2)),
                VariantIndexPair::from_index(2, q(3)),
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
}
