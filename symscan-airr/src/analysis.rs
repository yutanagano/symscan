use std::{
    cmp::{max, min},
    ops::{Add, AddAssign, Index, IndexMut},
};

use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::parsing::AirrData;

pub trait Matrix<T: FieldLike>: Index<(usize, usize), Output = T> {}
pub trait FieldLike: Copy + Default + Add + AddAssign {}
impl<T> FieldLike for T where T: Copy + Default + Add + AddAssign {}

/// Flat representation of a general (dense) matrix.
///
/// Represented using a flattened array with col-major indexing. A new matrix is filled with the
/// default value for the item type.
pub struct DenseMatrix<T: FieldLike> {
    num_rows: usize,
    num_cols: usize,
    vals: Vec<T>,
}

impl<T: FieldLike> DenseMatrix<T> {
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

impl<T: FieldLike> Index<(usize, usize)> for DenseMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let idx = self.flat_index(index);
        &self.vals[idx]
    }
}

impl<T: FieldLike> IndexMut<(usize, usize)> for DenseMatrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.flat_index(index);
        &mut self.vals[idx]
    }
}

impl<T: FieldLike> AddAssign for DenseMatrix<T> {
    fn add_assign(&mut self, rhs: Self) {
        if self.num_rows != rhs.num_rows || self.num_cols != rhs.num_cols {
            panic!("cannot add matrices of different shapes")
        };
        for (cell_accum, cell_summand) in self.vals.iter_mut().zip(rhs.vals.iter()) {
            *cell_accum += *cell_summand
        }
    }
}

impl<T: FieldLike> Add for DenseMatrix<T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: FieldLike> Matrix<T> for DenseMatrix<T> {}

/// Packed representation of symmetric squared matrix.
///
/// Represented as the upper triangle + diagonal using a flattened data array. The indexing is
/// column-major. A new matrix is filled with the default value for the item type.
pub struct SymmetricMatrix<T: FieldLike> {
    side_len: usize,
    vals: Vec<T>,
}

impl<T: FieldLike> SymmetricMatrix<T> {
    fn new(side_len: usize) -> Self {
        let packed_size = side_len * (side_len + 1) / 2;
        Self {
            side_len,
            vals: vec![T::default(); packed_size],
        }
    }

    fn flat_index(&self, index: (usize, usize)) -> usize {
        if index.0 >= self.side_len || index.1 >= self.side_len {
            panic!("index ({}, {}) out of range", index.0, index.1);
        }

        let row = min(index.0, index.1);
        let col = max(index.0, index.1);
        row + (col * (col + 1) / 2)
    }
}

impl<T: FieldLike> Index<(usize, usize)> for SymmetricMatrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let idx = self.flat_index(index);
        &self.vals[idx]
    }
}

impl<T: FieldLike> IndexMut<(usize, usize)> for SymmetricMatrix<T> {
    /// Note that since this is a packed representation, mutating the entry for (i, j) will also
    /// effectively mutate the value of (j, i)
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let idx = self.flat_index(index);
        &mut self.vals[idx]
    }
}

impl<T: FieldLike> AddAssign for SymmetricMatrix<T> {
    fn add_assign(&mut self, rhs: Self) {
        if self.side_len != rhs.side_len {
            panic!("cannot add matrices of different shapes")
        };
        for (cell_accum, cell_summand) in self.vals.iter_mut().zip(rhs.vals.iter()) {
            *cell_accum += *cell_summand
        }
    }
}

impl<T: FieldLike> Add for SymmetricMatrix<T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<T: FieldLike> Matrix<T> for SymmetricMatrix<T> {}

pub fn compute_overlap_matrix_within(
    data: &AirrData,
    max_distance: u8,
    hamming: bool,
) -> Result<SymmetricMatrix<u64>, symscan::Error> {
    let junction_seqs: Vec<&str> = data.interned_junctions.iter().collect();
    let neighbor_pairs = match hamming {
        true => symscan::get_hamming_neighbors_within(&junction_seqs, max_distance),
        false => symscan::get_neighbors_within(&junction_seqs, max_distance),
    }?;

    let num_reps = data.interned_repertoires.len();
    let overlap_matrix = neighbor_pairs
        .row
        .par_iter()
        .zip(neighbor_pairs.col.par_iter())
        .fold(
            || SymmetricMatrix::new(num_reps),
            |mut ovl_mat, (&jid_1, &jid_2)| {
                for (dc_1, dc_2) in data
                    .dup_counts
                    .for_junuction_id(jid_1)
                    .cartesian_product(data.dup_counts.for_junuction_id(jid_2))
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
                        if dc_1.repertoire_id == dc_2.repertoire_id {
                            2 * factor
                        } else {
                            factor
                        }
                }

                ovl_mat
            },
        )
        .reduce(
            || SymmetricMatrix::new(num_reps),
            |accum, summand| accum + summand,
        );

    let junction_ids: Vec<u32> = (0..data.interned_junctions.len() as u32).collect();
    let overlap_matrix_self_term = junction_ids
        .par_iter()
        .fold(
            || SymmetricMatrix::new(num_reps),
            |mut ovl_mat, &jid| {
                for (dc_1, dc_2) in data
                    .dup_counts
                    .for_junuction_id(jid)
                    .combinations_with_replacement(2)
                    .map(|p| (p[0], p[1]))
                {
                    ovl_mat[(dc_1.repertoire_id as usize, dc_2.repertoire_id as usize)] +=
                        dc_1.duplicate_count as u64 * dc_2.duplicate_count as u64;
                }
                ovl_mat
            },
        )
        .reduce(
            || SymmetricMatrix::new(num_reps),
            |accum, summand| accum + summand,
        );

    Ok(overlap_matrix + overlap_matrix_self_term)
}

pub fn compute_overlap_matrix_across(
    data_query: &AirrData,
    data_ref: &AirrData,
    max_distance: u8,
    hamming: bool,
) -> Result<DenseMatrix<u64>, symscan::Error> {
    let seqs_query: Vec<&str> = data_query.interned_junctions.iter().collect();
    let seqs_ref: Vec<&str> = data_ref.interned_junctions.iter().collect();
    let neighbor_pairs = match hamming {
        true => symscan::get_hamming_neighbors_across(&seqs_query, &seqs_ref, max_distance),
        false => symscan::get_neighbors_across(&seqs_query, &seqs_ref, max_distance),
    }?;

    let num_reps_q = data_query.interned_repertoires.len();
    let num_reps_r = data_ref.interned_repertoires.len();
    let overlap_matrix = neighbor_pairs
        .row
        .par_iter()
        .zip(neighbor_pairs.col.par_iter())
        .fold(
            || DenseMatrix::new(num_reps_q, num_reps_r),
            |mut ovl_mat, (&jid_q, &jid_r)| {
                for (dc_q, dc_r) in data_query
                    .dup_counts
                    .for_junuction_id(jid_q)
                    .cartesian_product(data_ref.dup_counts.for_junuction_id(jid_r))
                {
                    ovl_mat[(dc_q.repertoire_id as usize, dc_r.repertoire_id as usize)] +=
                        dc_q.duplicate_count as u64 * dc_r.duplicate_count as u64;
                }
                ovl_mat
            },
        )
        .reduce(
            || DenseMatrix::new(num_reps_q, num_reps_r),
            |accum, summand| accum + summand,
        );

    Ok(overlap_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing;

    static MOCK_AIRR_TSV: &[u8] = include_bytes!("../../test_files/mock_airr.tsv");

    #[test]
    fn test_compute_overlap_matrix() {
        let parsed =
            parsing::parse_airr_tsv(MOCK_AIRR_TSV, false, None).expect("should parse valid tsv");
        let ovl_mat = compute_overlap_matrix_within(&parsed, 2, false)
            .expect("should not be any symscan errors");

        assert_eq!(ovl_mat.vals, vec![10, 8, 14]);
    }
}
