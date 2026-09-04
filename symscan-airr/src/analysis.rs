use std::cmp::{max, min};

use itertools::Itertools;

use crate::parsing::AirrData;

/// Packed representation of symmetric squared matrix.
///
/// Represented as the upper triangle + diagonal using a flattened data array. The indexing is
/// column-major.
pub struct SymmetricMatrix<T: Clone + Default> {
    vals: Vec<T>,
}

impl<T: Clone + Default> SymmetricMatrix<T> {
    fn new(side_len: usize) -> Self {
        let packed_size = side_len * (side_len + 1) / 2;
        Self {
            vals: vec![T::default(); packed_size],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> &T {
        let i = min(row, col);
        let j = max(row, col);
        let idx = i + (j * (j + 1) / 2);
        &self.vals[idx]
    }

    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let i = min(row, col);
        let j = max(row, col);
        let idx = i + (j * (j + 1) / 2);
        &mut self.vals[idx]
    }
}

pub fn compute_overlap_matrix(data: &AirrData) -> Result<SymmetricMatrix<u64>, symscan::Error> {
    let neighbor_pairs = symscan::get_neighbors_within(data.interned_junctions.uniques(), 2)?;

    let mut ovl_mat = SymmetricMatrix::new(data.interned_repertoires.len());

    for (jid_1, jid_2) in neighbor_pairs.row.iter().zip(neighbor_pairs.col.iter()) {
        for (dc_1, dc_2) in data
            .dup_counts
            .for_junuction_id(*jid_1)
            .cartesian_product(data.dup_counts.for_junuction_id(*jid_2))
        {
            let factor = dc_1.duplicate_count as u64 * dc_2.duplicate_count as u64;
            *ovl_mat.get_mut(dc_1.repertoire_id as usize, dc_2.repertoire_id as usize) +=
                match dc_1.repertoire_id == dc_2.repertoire_id {
                    // diagonal needs to be added twice the amount to account for the fact that we
                    // are adding the outer product and its transpose
                    true => 2 * factor,

                    // off-diagonal is updated simultaneously with mirrored cell according to the
                    // SymmetricMatrix implementation
                    false => factor,
                }
        }
    }

    for junction_id in 0..data.interned_junctions.len() as u32 {
        for (dc_1, dc_2) in data
            .dup_counts
            .for_junuction_id(junction_id)
            .combinations_with_replacement(2)
            .map(|p| (p[0], p[1]))
        {
            *ovl_mat.get_mut(dc_1.repertoire_id as usize, dc_2.repertoire_id as usize) +=
                dc_1.duplicate_count as u64 * dc_2.duplicate_count as u64;
        }
    }

    Ok(ovl_mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing;

    static MOCK_AIRR_TSV: &[u8] = include_bytes!("../../test_files/mock_airr.tsv");

    #[test]
    fn test_compute_overlap_matrix() {
        let parsed = parsing::parse_airr_tsv(MOCK_AIRR_TSV).expect("should parse valid tsv");
        let ovl_mat = compute_overlap_matrix(&parsed).expect("should not be any symscan errors");

        assert_eq!(ovl_mat.vals, vec![10, 8, 14]);
    }
}
