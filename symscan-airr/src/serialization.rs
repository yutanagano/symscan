use std::io::{self, Write};

use itertools::Itertools;

use crate::{
    analysis::{Matrix, SymmetricMatrix},
    parsing::AirrData,
};

/// Serialize a matrix in triplet tsv format, where every index is explicitly specified as a row.
///
/// This will lead to rows with duplicate information if the underlying overlap matrix is
/// symmetrical.
pub fn om_as_triplet_tsv_full(
    overlap_matrix: &impl Matrix<u64>,
    context_query: &AirrData,
    context_ref: &AirrData,
    writer: &mut impl Write,
) -> io::Result<()> {
    let mut repnames_query: Vec<&str> = context_query.interned_repertoires.iter().collect();
    let mut repnames_ref: Vec<&str> = context_ref.interned_repertoires.iter().collect();
    repnames_query.sort_unstable();
    repnames_ref.sort_unstable();

    for (repname_q, repname_r) in repnames_query.iter().cartesian_product(repnames_ref.iter()) {
        let repid_q = context_query
            .interned_repertoires
            .get_id(repname_q)
            .expect("valid repname should have id");
        let repid_r = context_ref
            .interned_repertoires
            .get_id(repname_r)
            .expect("valid repname should have id");
        let overlap = overlap_matrix[(repid_q as usize, repid_r as usize)];

        writeln!(writer, "{repname_q}\t{repname_r}\t{overlap}")?;
    }

    Ok(())
}

/// Serialize a matrix in triplet tsv format, where only the upper triangle and diagonal are
/// explicitly specified.
///
/// This is only available for symmetric matrices.
pub fn om_as_triplet_tsv_upper(
    overlap_matrix: &SymmetricMatrix<u64>,
    context: &AirrData,
    writer: &mut impl Write,
) -> io::Result<()> {
    let mut repnames: Vec<&str> = context.interned_repertoires.iter().collect();
    repnames.sort_unstable();

    for (repname_1, repname_2) in repnames
        .iter()
        .combinations_with_replacement(2)
        .map(|p| (p[0], p[1]))
    {
        let repid_1 = context
            .interned_repertoires
            .get_id(repname_1)
            .expect("valid repname should have id");
        let repid_2 = context
            .interned_repertoires
            .get_id(repname_2)
            .expect("valid repname should have id");
        let overlap = overlap_matrix[(repid_1 as usize, repid_2 as usize)];

        writeln!(writer, "{repname_1}\t{repname_2}\t{overlap}")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{analysis, parsing};

    static MOCK_AIRR_TSV: &[u8] = include_bytes!("../../test_files/mock_airr.tsv");

    #[test]
    fn test_om_as_triplet_tsv_ut() {
        let parsed =
            parsing::parse_airr_tsv(MOCK_AIRR_TSV, false, None).expect("should parse valid tsv");
        let ovl_mat = analysis::compute_overlap_matrix_within(&parsed, 2, false)
            .expect("should not be any symscan errors");
        let mut output = Vec::new();
        om_as_triplet_tsv_upper(&ovl_mat, &parsed, &mut output).expect("write should not fail");

        let expected = "a\ta\t10\na\tb\t8\nb\tb\t14\n";

        assert_eq!(output, expected.as_bytes());
    }
}
