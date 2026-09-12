use std::{
    char::TryFromCharError,
    hash::BuildHasher,
    io::{BufRead, Read},
};

use csv::{ByteRecord, Reader, ReaderBuilder};
use foldhash::fast::FixedState;
use hashbrown::{hash_table::Entry, HashMap, HashTable};

use crate::ParsingOpts;

const STRING_HASHER: FixedState = FixedState::with_seed(0);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("file separator must be ASCII: {0}")]
    TryFromChar(#[from] TryFromCharError),
    #[error("{0}")]
    InvalidCsv(#[from] csv::Error),
    #[error("missing column {0}")]
    MissingColumn(String),
    #[error("non-ASCII junction on line {line}")]
    JunctionNotAscii { line: u64 },
    #[error("non-UTF8 bytes on line {line}")]
    ValueNotUtf8 { line: u64 },
    #[error("invalid duplicate_count {val} on line {line}")]
    InvalidDuplicateCount { line: u64, val: String },
    #[error("string interner has reached maximum capacity ({}), please try again with a smaller input size", u32::MAX)]
    OverCapacity,
}

/// Structured representation of data from an AIRR TSV file.
pub struct AirrData {
    pub interned_junctions: InternedStrings,
    pub interned_repertoires: InternedStrings,
    pub dup_counts: DupCountCsr,
}

/// A struct for interning strings.
pub struct InternedStrings {
    /// Contiguous buffer array that stores the backing memory for the interned strings.
    buffer: Vec<u8>,

    /// Location of each interned string's data on the buffer vector.
    ///
    /// Has length N+1 where N is the number of unique interned strings. The memory for the kth
    /// string is stored in buffer[offsets[k]..offsets[k+1]].
    offsets: Vec<usize>,

    /// Used to map strings to their IDs.
    ids: HashTable<u32>,
}

impl InternedStrings {
    fn with_capacity(cap: usize) -> Self {
        let buffer = Vec::with_capacity(cap.saturating_mul(20));
        let mut offsets = Vec::with_capacity(cap + 1);
        let ids = HashTable::with_capacity(cap);
        offsets.push(0);

        Self {
            buffer,
            offsets,
            ids,
        }
    }

    /// The number of unique interned strings.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The set of unique interned strings.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        (0..self.len() as u32).map(|id| self.str_at(id))
    }

    pub fn get_id(&self, s: &str) -> Option<u32> {
        let hash = STRING_HASHER.hash_one(s);
        self.ids.find(hash, |&id| self.str_at(id) == s).copied()
    }

    fn str_at(&self, id: u32) -> &str {
        let i_start = self.offsets[id as usize];
        let i_end = self.offsets[id as usize + 1];
        unsafe { str::from_utf8_unchecked(&self.buffer[i_start..i_end]) }
    }

    /// Given a string, either get its ID if already assigned, and otherwise assign one and return.
    fn get_or_make_id(&mut self, s: &str) -> Result<u32, Error> {
        let hash = STRING_HASHER.hash_one(s);

        let Self {
            buffer,
            offsets,
            ids,
        } = self;
        let str_at = |id: u32| {
            let i_start = offsets[id as usize];
            let i_end = offsets[id as usize + 1];
            unsafe { str::from_utf8_unchecked(&buffer[i_start..i_end]) }
        };

        match ids.entry(
            hash,
            |&id| str_at(id) == s,
            |&id| STRING_HASHER.hash_one(str_at(id)),
        ) {
            Entry::Occupied(e) => Ok(*e.get()),
            Entry::Vacant(e) => {
                let id = u32::try_from(offsets.len() - 1).map_err(|_| Error::OverCapacity)?;
                buffer.extend_from_slice(s.as_bytes());
                offsets.push(buffer.len());
                e.insert(id);
                Ok(id)
            }
        }
    }
}

pub struct DupCountCsr {
    dup_counts: Vec<u32>,
    rep_ids: Vec<u32>,
    junction_ranges: Vec<u32>,
}

impl DupCountCsr {
    fn from_coo(coo: HashMap<PackedSeqId, u32>, num_rows: usize) -> Self {
        let mut junction_ranges = vec![0u32; num_rows + 1];
        for k in coo.keys() {
            junction_ranges[(k.junction_id() as usize) + 1] += 1;
        }
        for i in 0..num_rows {
            junction_ranges[i + 1] += junction_ranges[i];
        }

        let mut dup_counts = vec![0u32; coo.len()];
        let mut rep_ids = vec![0u32; coo.len()];
        let mut junction_cursors = junction_ranges.clone();
        for (k, dup_count) in coo {
            let junction_idx = &mut junction_cursors[k.junction_id() as usize];
            dup_counts[*junction_idx as usize] = dup_count;
            rep_ids[*junction_idx as usize] = k.repertoire_id();
            *junction_idx += 1;
        }

        Self {
            dup_counts,
            rep_ids,
            junction_ranges,
        }
    }

    pub fn for_junuction_id(
        &self,
        id: u32,
    ) -> impl Iterator<Item = DupCountEntry> + use<'_> + Clone {
        let junction_range = self.junction_ranges[id as usize] as usize
            ..self.junction_ranges[id as usize + 1] as usize;
        let rep_ids = &self.rep_ids[junction_range.clone()];
        let dup_counts = &self.dup_counts[junction_range];

        rep_ids
            .iter()
            .copied()
            .zip(dup_counts.iter().copied())
            .map(|(rep_id, dup_count)| DupCountEntry {
                repertoire_id: rep_id,
                duplicate_count: dup_count,
            })
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct PackedSeqId(u64);

impl PackedSeqId {
    fn from_parts(junction_id: u32, repertoire_id: u32) -> Self {
        let internal = ((junction_id as u64) << 32) | (repertoire_id as u64);
        Self(internal)
    }

    fn junction_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    fn repertoire_id(&self) -> u32 {
        (self.0 & 0xFFFFFFFF) as u32
    }
}

#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone, Copy)]
pub struct DupCountEntry {
    pub repertoire_id: u32,
    pub duplicate_count: u32,
}

struct ColIndices {
    junction: usize,
    duplicate_count: usize,
    repertoire: usize,
    locus: Option<usize>,
}

impl ColIndices {
    fn try_from(tsv: &mut Reader<impl Read>, opts: &ParsingOpts) -> Result<Self, Error> {
        let headers = tsv.headers()?;

        let get_col_index = |colname: &str| {
            headers
                .iter()
                .position(|h| h == colname)
                .ok_or(Error::MissingColumn(colname.to_string()))
        };
        let junction = get_col_index(&opts.junction_col)?;
        let duplicate_count = get_col_index(&opts.count_col)?;
        let repertoire = get_col_index(&opts.repertoire_col)?;
        let locus = if opts.locus.is_some() {
            Some(get_col_index(&opts.locus_col)?)
        } else {
            None
        };

        Ok(Self {
            junction,
            duplicate_count,
            repertoire,
            locus,
        })
    }
}

/// Parses AIRR-compliant TSV input.
///
/// The function accepts a reader as input, from which the TSV will be parsed. The function should
/// first verify that the data from the reader is indeed an AIRR-compliant TSV containing at least
/// all of the following columns:
///
/// - junction_aa OR cdr3_aa
/// - dulpicate_count
/// - repertoire_id
///
/// Then, the function should parse the input TSV data and collect it into an interned collection of
/// AIR sequences, an interned collection of repertoires, and a sparse CSR representation of a
/// duplicate-count matrix. The shape of the duplicate-count matrix should be n_unique_AIRs x n_reps
/// (i.e. every row corresponds to a unique AIR sequence). Each row contains data representing the
/// duplicate count of a unique sequence in each of the input repertoires.
pub fn parse_airr_tsv(
    in_stream: impl BufRead,
    num_rows_hint: Option<u32>,
    parsing_opts: &ParsingOpts,
) -> Result<AirrData, Error> {
    let mut tsv_reader = ReaderBuilder::new()
        .delimiter(parsing_opts.sep.try_into()?)
        .from_reader(in_stream);

    let size_hint = num_rows_hint.unwrap_or(0) as usize;
    let mut interned_junctions = InternedStrings::with_capacity(size_hint);
    let mut interned_repertoires = InternedStrings::with_capacity(0);
    let mut dup_count_coo: HashMap<PackedSeqId, u32> = HashMap::with_capacity(size_hint);

    let col_indices = ColIndices::try_from(&mut tsv_reader, parsing_opts)?;
    let mut record = ByteRecord::new();
    while tsv_reader.read_byte_record(&mut record)? {
        // If locus is set, ignore records with unmatching loci
        if let Some(target_locus) = parsing_opts.locus.as_deref() {
            let record_locus = record
                .get(
                    col_indices
                        .locus
                        .expect("locus column should not be None if locus is set"),
                )
                .expect("idx should not be out of range");
            if record_locus != target_locus.as_bytes() {
                continue;
            }
        }

        // If for any record the junction, repertoire, or duplicate count are missing, skip the
        // record since we don't have enough information.

        let junction = record
            .get(col_indices.junction)
            .expect("idx should not be out of range");
        if junction.is_empty() {
            continue;
        }
        if !junction.is_ascii() {
            return Err(Error::JunctionNotAscii {
                line: record
                    .position()
                    .expect("record should have position set on read")
                    .line(),
            });
        }
        // Already checked junction is ASCII above
        let junction_id =
            interned_junctions.get_or_make_id(unsafe { str::from_utf8_unchecked(junction) })?;

        let repertoire = record
            .get(col_indices.repertoire)
            .expect("idx should not be out of range");
        if repertoire.is_empty() {
            continue;
        }
        let repertoire = str::from_utf8(repertoire).map_err(|_| Error::ValueNotUtf8 {
            line: record
                .position()
                .expect("record should have position set on read")
                .line(),
        })?;
        let repertoire_id = interned_repertoires.get_or_make_id(repertoire)?;

        let duplicate_count = record
            .get(col_indices.duplicate_count)
            .expect("idx should not be out of range");
        if duplicate_count.is_empty() {
            continue;
        }
        let duplicate_count = str::from_utf8(duplicate_count).map_err(|_| Error::ValueNotUtf8 {
            line: record
                .position()
                .expect("record should have position set on read")
                .line(),
        })?;
        let Ok(duplicate_count) = duplicate_count.parse::<u32>() else {
            return Err(Error::InvalidDuplicateCount {
                line: record
                    .position()
                    .expect("record should have position set on read")
                    .line(),
                val: duplicate_count.to_string(),
            });
        };

        let coo_key = PackedSeqId::from_parts(junction_id, repertoire_id);
        *dup_count_coo.entry(coo_key).or_insert(0) += duplicate_count;
    }

    // assert_eq!(dup_count_coo.len(), 9);
    let dup_counts = DupCountCsr::from_coo(dup_count_coo, interned_junctions.len());
    Ok(AirrData {
        interned_junctions,
        interned_repertoires,
        dup_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    static MOCK_AIRR_TSV: &[u8] = include_bytes!("../../test_files/mock_airr.tsv");
    static MOCK_AIRR_CUSTOM_COLS: &[u8] =
        include_bytes!("../../test_files/mock_airr_custom_cols.tsv");
    static MOCK_AIRR_CSV: &[u8] = include_bytes!("../../test_files/mock_airr.csv");

    #[test]
    fn test_col_indices() {
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(MOCK_AIRR_TSV);
        let col_indices = ColIndices::try_from(&mut reader, &ParsingOpts::default()).unwrap();

        assert_eq!(col_indices.junction, 0);
        assert_eq!(col_indices.duplicate_count, 1);
        assert_eq!(col_indices.repertoire, 2);
    }

    #[test]
    fn test_custom_cols() {
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(MOCK_AIRR_CUSTOM_COLS);
        let mut opts = ParsingOpts::default();
        opts.junction_col = "foo".to_string();
        opts.count_col = "bar".to_string();
        opts.repertoire_col = "baz".to_string();

        let col_indices = ColIndices::try_from(&mut reader, &opts)
            .expect("custom columns should be named correctly");

        assert_eq!(col_indices.junction, 0);
        assert_eq!(col_indices.duplicate_count, 1);
        assert_eq!(col_indices.repertoire, 2);
    }

    #[test]
    fn test_parse_tsv() {
        let parsed = parse_airr_tsv(MOCK_AIRR_TSV, None, &ParsingOpts::default())
            .expect("should parse valid tsv");

        assert_eq!(parsed.interned_junctions.len(), 9);
        assert_eq!(parsed.interned_repertoires.len(), 2);

        assert_eq!(
            parsed
                .interned_junctions
                .get_id("CAVSTSGGSYIPTF")
                .expect("should not be at capacity"),
            2
        );
        assert_eq!(
            parsed
                .interned_repertoires
                .get_id("a")
                .expect("should not be at capacity"),
            0
        );
        assert_eq!(
            parsed
                .interned_repertoires
                .get_id("b")
                .expect("should not be at capacity"),
            1
        );

        let mut dup_counts_id_1: Vec<DupCountEntry> =
            parsed.dup_counts.for_junuction_id(2).collect();
        dup_counts_id_1.sort();
        assert_eq!(
            dup_counts_id_1,
            vec![
                DupCountEntry {
                    repertoire_id: 0,
                    duplicate_count: 1
                },
                DupCountEntry {
                    repertoire_id: 1,
                    duplicate_count: 1
                }
            ]
        );
    }

    #[test]
    fn test_custom_sep() {
        let mut opts = ParsingOpts::default();
        opts.sep = ',';

        let parsed = parse_airr_tsv(MOCK_AIRR_CSV, None, &opts).expect("should parse valid csv");

        assert_eq!(parsed.interned_junctions.len(), 9);
        assert_eq!(parsed.interned_repertoires.len(), 2);
    }
}
