use std::{
    hash::BuildHasher,
    io::{BufRead, Read},
};

use csv::{Reader, ReaderBuilder, StringRecord};
use foldhash::fast::FixedState;
use hashbrown::{hash_table::Entry, HashMap, HashTable};

const STRING_HASHER: FixedState = FixedState::with_seed(0);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    InvalidCsv(csv::Error),
    #[error("missing column {0}")]
    MissingColumn(String),
    #[error("non-ASCII junction {seq} on line {line}")]
    JunctionNotAscii { line: u64, seq: String },
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
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            offsets: vec![0],
            ids: HashTable::new(),
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
    jctn: usize,
    rprt: usize,
    dpct: usize,
}

impl ColIndices {
    fn try_from(tsv: &mut Reader<impl Read>, use_cdr3_col: bool) -> Result<Self, Error> {
        let headers = tsv.headers().map_err(Error::InvalidCsv)?;
        let get_col_index = |colname: &str| {
            headers
                .iter()
                .position(|h| h == colname)
                .ok_or(Error::MissingColumn(colname.to_string()))
        };
        let jctn = match use_cdr3_col {
            true => get_col_index("cdr3_aa"),
            false => get_col_index("junction_aa"),
        }?;
        let rprt = get_col_index("repertoire_id")?;
        let dpct = get_col_index("duplicate_count")?;

        Ok(Self { jctn, rprt, dpct })
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
pub fn parse_airr_tsv(in_stream: impl BufRead, use_cdr3_col: bool) -> Result<AirrData, Error> {
    let mut reader = ReaderBuilder::new().delimiter(b'\t').from_reader(in_stream);

    let mut interned_junctions = InternedStrings::new();
    let mut interned_repertoires = InternedStrings::new();
    let mut dup_count_coo: HashMap<PackedSeqId, u32> = HashMap::new();

    let col_indices = ColIndices::try_from(&mut reader, use_cdr3_col)?;
    let mut record = StringRecord::new();
    while reader.read_record(&mut record).map_err(Error::InvalidCsv)? {
        // If for any record the junction, repertoire, or duplicate count are missing, skip the
        // record since we don't have enough information.

        let junction = record
            .get(col_indices.jctn)
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
                seq: junction.to_string(),
            });
        }
        let junction_id = interned_junctions.get_or_make_id(junction)?;

        let repertoire = record
            .get(col_indices.rprt)
            .expect("idx should not be out of range");
        if repertoire.is_empty() {
            continue;
        }
        let repertoire_id = interned_repertoires.get_or_make_id(repertoire)?;

        let duplicate_count = record
            .get(col_indices.dpct)
            .expect("idx should not be out of range");
        if duplicate_count.is_empty() {
            continue;
        }
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

    #[test]
    fn test_col_indices() {
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(MOCK_AIRR_TSV);
        let col_indices = ColIndices::try_from(&mut reader, false).unwrap();

        assert_eq!(col_indices.jctn, 0);
        assert_eq!(col_indices.rprt, 2);
    }

    #[test]
    fn test_use_cdr3_col() {
        let mut reader = ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(MOCK_AIRR_TSV);
        let result = ColIndices::try_from(&mut reader, true);

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_tsv() {
        let parsed = parse_airr_tsv(MOCK_AIRR_TSV, false).expect("should parse valid tsv");

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
}
