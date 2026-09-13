use foldhash::fast::FixedState;
use rayon::prelude::*;

use crate::{
    core::{self, MaxDistance},
    indexing::{CrossIndex, VariantIndex},
    metric::Metric,
    Error, InputType, NeighborPairs,
};

pub fn get_neighbors_within_impl<M: Metric>(
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
    core::check_strings_ascii(query, InputType::Query)?;

    let (convergent_indices, group_sizes) = {
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
        // Payload is the group size; both halves of (num_indices, payload) are the same value.
        core::collect_convergent_indices::<u32, _>(variant_index_pairs, |group| {
            let len = core::distinct(group).count();
            (len > 1).then_some((len, len))
        })
    };

    let convergent_chunks = core::get_convergent_chunks(&group_sizes, &convergent_indices[..]);
    let candidates = core::get_hit_candidates_within(&convergent_chunks);
    let dists = core::compute_dists::<M>(&candidates, query, query, max_distance);

    Ok(core::validate_and_collect_hits(
        candidates,
        dists,
        max_distance,
    ))
}

pub fn get_neighbors_across_impl<M: Metric>(
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
    core::check_strings_ascii(query, InputType::Query)?;
    core::check_strings_ascii(reference, InputType::Reference)?;

    let (convergent_indices, group_sizes) = {
        let num_del_variants_q = M::count_oneshot(query, max_distance);
        let num_del_variants_r = M::count_oneshot(reference, max_distance);

        let total_capacity =
            num_del_variants_q.iter().sum::<usize>() + num_del_variants_r.iter().sum::<usize>();
        let mut variant_index_pairs_uninit = core::prealloc_maybeuninit_vec(total_capacity);
        let (vip_chunks_q, vip_chunks_r) = core::get_disjoint_chunks_mut_cross(
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

        let variant_index_pairs =
            unsafe { core::cast_to_initialised_vec(variant_index_pairs_uninit) };
        core::collect_convergent_indices::<CrossIndex, _>(variant_index_pairs, |group| {
            let (len_q, len_r) = core::distinct(group).fold((0, 0), |(q, r), word| {
                if CrossIndex::from_index_bits(word.index_bits()).is_ref() {
                    (q, r + 1)
                } else {
                    (q + 1, r)
                }
            });
            (len_q > 0 && len_r > 0).then_some((len_q + len_r, (len_q, len_r)))
        })
    };

    let convergent_chunks =
        core::get_convergent_chunks_cross(&group_sizes, &convergent_indices[..]);
    let candidates = core::get_hit_candidates_across(&convergent_chunks);
    let dists = core::compute_dists::<M>(&candidates, query, reference, max_distance);

    Ok(core::validate_and_collect_hits(
        candidates,
        dists,
        max_distance,
    ))
}
