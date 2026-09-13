use std::{
    hash::{BuildHasher, Hasher},
    mem::MaybeUninit,
};

use rapidfuzz::distance::{hamming, levenshtein};

use crate::{
    core::{self, MaxDistance},
    indexing::{CrossIndex, VariantIndex, VariantIndexPair},
};

/// Private zero-cost strategy distinguishing Levenshtein vs Hamming pipelines.
pub trait Metric: Copy + Send + Sync + 'static {
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
pub struct Levenshtein;

#[derive(Clone, Copy)]
pub struct Hamming;

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

/// Compute the total number of deletion variants up to a certain number of maximum deletions.
pub fn get_num_del_vars_per_string_up_to(
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
                num_vars += core::get_num_k_combs(s.as_ref().len(), k);
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
                core::get_num_k_combs(s.as_ref().len(), max_distance.as_u8())
            }
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use foldhash::fast::FixedState;

    use super::*;

    #[test]
    fn test_exact_null_full_deletions_hashes_null_bytes() {
        let input = "ab";
        let max_distance = MaxDistance::try_from(2).expect("legal");
        let mut chunk = core::prealloc_maybeuninit_vec(1);
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

        let pairs = unsafe { core::cast_to_initialised_vec(chunk) };
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
}
