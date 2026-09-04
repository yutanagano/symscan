use std::{
    cmp::{max, min},
    ops::{Index, IndexMut},
};

use itertools::Itertools;

use crate::parsing::AirrData;

pub trait Matrix<T: Clone + Default>: Index<(usize, usize), Output = T> {}

/// Flat representation of a general (dense) matrix.
///
/// Represented using a flattened array with col-major indexing. A new matrix is filled with the
/// default value for the item type.
pub struct DenseMatrix<T: Clone + Default> {
    num_rows: usize,
    num_cols: usize,
    vals: Vec<T>,
}

impl<T: Clone + Default> DenseMatrix<T> {
    fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            vals: vec![T::default(); num_rows * num_cols],
        }
    }

    fn flat_index(&self, index: (usize, usize)) -> usize {
        let (row, col) = index;
        if row >= self.num_rows || col >= self.num_cols {
            panic!("index ({row}, {col}) out of range");
        };
        row + col * self.num_rows
    }
}

impl<T: Clone + Default> Index<(usize, usize)> for DenseMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let idx = self.flat_index(index);
        &self.vals[idx]
    }
}

impl<T: Clone + Default> IndexMut<(usize, usize)> for DenseMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.flat_index(index);
        &mut self.vals[idx]
    }
}

impl<T: Clone + Default> Matrix<T> for DenseMatrix<T> {}

/// Packed representation of symmetric squared matrix.
///
/// Represented as the upper triangle + diagonal using a flattened data array. The indexing is
/// column-major. A new matrix is filled with the default value for the item type.
pub struct SymmetricMatrix<T: Clone + Default> {
    side_len: usize,
    vals: Vec<T>,
}

impl<T: Clone + Default> SymmetricMatrix<T> {
    fn new(side_len: usize) -> Self {
        let packed_size = side_len * (side_len + 1) / 2;
        Self {
            side_len,
            vals: vec![T::default(); packed_size],
        }
    }

    fn to_flat_index(&self, index: (usize, usize)) -> usize {
        if index.0 >= self.side_len || index.1 >= self.side_len {
            panic!("index ({}, {}) out of range", index.0, index.1);
        }

        let row = min(index.0, index.1);
        let col = max(index.0, index.1);
        row + (col * (col + 1) / 2)
    }
}

impl<T: Clone + Default> Index<(usize, usize)> for SymmetricMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let idx = self.to_flat_index(index);
        &self.vals[idx]
    }
}

impl<T: Clone + Default> IndexMut<(usize, usize)> for SymmetricMatrix<T> {
    /// Note that since this is a packed representation, mutating the entry for (i, j) will also
    /// effectively mutate the value of (j, i)
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.to_flat_index(index);
        &mut self.vals[idx]
    }
}

impl<T: Clone + Default> Matrix<T> for SymmetricMatrix<T> {}

pub fn compute_overlap_matrix_within(
    data: &AirrData,
    max_distance: u8,
) -> Result<SymmetricMatrix<u64>, symscan::Error> {
    let neighbor_pairs =
        symscan::get_neighbors_within(data.interned_junctions.uniques(), max_distance)?;

    let mut ovl_mat = SymmetricMatrix::new(data.interned_repertoires.len());

    for (jid_1, jid_2) in neighbor_pairs.row.iter().zip(neighbor_pairs.col.iter()) {
        for (dc_1, dc_2) in data
            .dup_counts
            .for_junuction_id(*jid_1)
            .cartesian_product(data.dup_counts.for_junuction_id(*jid_2))
        {
            let factor = dc_1.duplicate_count as u64 * dc_2.duplicate_count as u64;

            // Here, we need to add the outer product of the duplicate counts vector for junction 1
            // and that of junction 2 _twice_ - we add the outer product, then its transpose as
            // well. Now, because of the packed representation, mutations to the off-diagonal are
            // automatically reflected on both sides of the diagonal, so there is no need to mutate
            // both (i,j) and (j,i). However, the diagonal cells are unique, so to emulate the
            // addition of both outer products, the diagonal must have the result of the outer
            // product added twice.
            ovl_mat[(dc_1.repertoire_id as usize, dc_2.repertoire_id as usize)] +=
                match dc_1.repertoire_id == dc_2.repertoire_id {
                    true => 2 * factor,
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
            ovl_mat[(dc_1.repertoire_id as usize, dc_2.repertoire_id as usize)] +=
                dc_1.duplicate_count as u64 * dc_2.duplicate_count as u64;
        }
    }

    Ok(ovl_mat)
}

pub fn compute_overlap_matrix_across(
    data_query: &AirrData,
    data_ref: &AirrData,
    max_distance: u8,
) -> Result<DenseMatrix<u64>, symscan::Error> {
    let neighbor_pairs = symscan::get_neighbors_across(
        data_query.interned_junctions.uniques(),
        data_ref.interned_junctions.uniques(),
        max_distance,
    )?;

    let mut ovl_mat = DenseMatrix::new(
        data_query.interned_repertoires.len(),
        data_ref.interned_repertoires.len(),
    );

    for (jid_q, jid_r) in neighbor_pairs.row.iter().zip(neighbor_pairs.col.iter()) {
        for (dc_q, dc_r) in data_query
            .dup_counts
            .for_junuction_id(*jid_q)
            .cartesian_product(data_ref.dup_counts.for_junuction_id(*jid_r))
        {
            ovl_mat[(dc_q.repertoire_id as usize, dc_r.repertoire_id as usize)] +=
                dc_q.duplicate_count as u64 * dc_r.duplicate_count as u64;
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
        let ovl_mat =
            compute_overlap_matrix_within(&parsed, 2).expect("should not be any symscan errors");

        assert_eq!(ovl_mat.vals, vec![10, 8, 14]);
    }
}
