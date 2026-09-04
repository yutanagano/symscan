use std::io::{Result, Write};

use itertools::Itertools;

use crate::{analysis::SymmetricMatrix, parsing::AirrData};

pub fn write_overlap_matrix_as_tsv(
    overlap_matrix: &SymmetricMatrix<u64>,
    context: &AirrData,
    writer: &mut impl Write,
) -> Result<()> {
    let mut repertoire_names = context.interned_repertoires.uniques().to_vec();
    repertoire_names.sort_unstable();

    for (repname_1, repname_2) in repertoire_names
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
        let overlap = overlap_matrix.get(repid_1 as usize, repid_2 as usize);

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
    fn test_write_overlap_matrix_as_tsv() {
        let parsed = parsing::parse_airr_tsv(MOCK_AIRR_TSV).expect("should parse valid tsv");
        let ovl_mat =
            analysis::compute_overlap_matrix(&parsed).expect("should not be any symscan errors");
        let mut output = Vec::new();
        write_overlap_matrix_as_tsv(&ovl_mat, &parsed, &mut output).expect("write should not fail");

        let expected = "a\ta\t10\na\tb\t8\nb\tb\t14\n";

        assert_eq!(output, expected.as_bytes());
    }
}
