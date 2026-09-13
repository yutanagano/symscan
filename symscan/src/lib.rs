//! SymScan enables extremely fast discovery of pairs of similar strings within and across large
//! collections.
//!
//! SymScan is a variation on the [symmetric deletion
//! ](https://seekstorm.com/blog/1000x-spelling-correction/) algorithm that is optimised for
//! bulk-searching similar strings within one or across two large string collections at once (e.g.
//! searching for similar protein sequences among a collection of 10M). The key algorithmic
//! difference between SymScan and traditional symmetric deletion is the use of a [sort-merge
//! join](https://en.wikipedia.org/wiki/Sort-merge_join) approach in place of hashmaps to discover
//! input strings that share common deletion variants. This sort-and-scan approach trades off an
//! additional factor of O(log N) (with N the total number of strings being compared) in expected
//! time complexity for improved cache locality and effective parallelization, and ends up being
//! much faster for the above use case. Parallelization is handled using the
//! [rayon](https://docs.rs/rayon/latest/rayon/) crate internally.
//!
//! SymScan provides separate implementations for [Levenshtein edit
//! distance](https://en.wikipedia.org/wiki/Levenshtein_distance) and [Hamming
//! distance](https://en.wikipedia.org/wiki/Hamming_distance). See [`get_neighbors_within`] /
//! [`get_hamming_neighbors_within`] and [`get_neighbors_across`] / [`get_hamming_neighbors_across`]
//! for details on the API.
//!
//! Even for our intended use case of discovering pairs of similar strings from large collections,
//! it is sometimes useful to memoize the deletion variant computations for at least one side of the
//! query (e.g. reference-side memoization when making repeated queries against a very large
//! reference collection with relatively smaller query collections). For such cases, the library
//! also provides the [`CachedRef`] / [`CachedRefHamming`] structs.

use std::fmt::Display;

use crate::{
    cached::CachedStore,
    metric::{Hamming, Levenshtein},
};

mod cached;
mod core;
mod functional;
mod indexing;
mod metric;

/// Symscan error variants.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An input collection contained references to at least one non-ASCII string.
    #[error("non-ASCII input currently unsupported ('{offending_string}' at {offending_idx})")]
    NonAsciiInput {
        input_type: InputType,
        offending_idx: usize,
        offending_string: String,
    },

    /// An input collection contained more than the maximum allowed number of strings.
    ///
    /// In most cases, the maximum allowed length is [4,294,967,295](u32::MAX). This is because
    /// internal computations use [`u32`]s to encode string indices. The exception is when calling
    /// [`get_neighbors_across`], where the maximum is instead 2,147,483,647 ((2^31)-1) due to the
    /// fact that one of the 32 bits is reserved for distinguishing between indexes of the `query`
    /// slice and the `reference` slice.
    #[error("{input_type} must not hold more than {limit} elements, got {got}")]
    TooManyStrings {
        input_type: InputType,
        got: usize,
        limit: usize,
    },

    /// The `max_distance` function / method parameter was set to [255](u8::MAX).
    ///
    /// This results in an error because that value is reserved for encoding when pairs exceed the
    /// threshold distance during internal computations.
    #[error("max_distance is capped at {limit}, got {illegal}", limit = u8::MAX - 1, illegal = u8::MAX)]
    MaxDistCapped,

    /// The `max_distance` method parameter was set to a value greater than that given when
    /// constructing [`CachedRef`] being queried.
    ///
    /// This results in an error because the `max_distance` given at [`CachedRef`] construction
    /// time determines how many `reference` string deletion variants are generated and cached in
    /// the struct. A cache containing deletion variants to a depth of X cannot support symscan
    /// queries with `max_distance` > X.
    #[error("CachedRef instance not compatible with max_distance above {limit}, got {got}")]
    MaxDistTooLargeForCache { got: u8, limit: u8 },
}

/// Used to specify the source of certain [`Error`] variants.
#[derive(Debug)]
pub enum InputType {
    Query,
    Reference,
}

impl Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            InputType::Query => "query",
            InputType::Reference => "reference",
        };
        write!(f, "{}", text)
    }
}

/// Collection of string pairs that lie within the specified Levenshtein edit distance threshold.
///
/// This is what is returned via the [`Ok`] variant from [`get_neighbors_within`],
/// [`get_neighbors_across`], and related methods in [`CachedRef`]. [`row`](NeighborPairs::row) and
/// [`col`](NeighborPairs::col) contain the indices of the neighbor string pairs, and
/// [`dists`](NeighborPairs::dists) contains the Levenshtein distances between the corresponding
/// pairs.
///
/// # A note on double-counting pairs
///
/// When returning the results of [`get_neighbors_within`] / [`CachedRef::get_neighbors_within`],
/// string pairs _**ARE NOT**_ double-counted. As seen in the
/// [examples](get_neighbors_within#examples), each pair is represented once where the
/// [`row`](NeighborPairs::row) index is always less than the [`col`](NeighborPairs::col) index. In
/// other words, if you were to interpret the [`NeighborPairs`] in these situations as an unpacked
/// coordinate representation of a sparse matrix, then only the strictly upper triangle can be
/// filled.
#[derive(Debug, PartialEq)]
pub struct NeighborPairs {
    /// Indices of strings in the input `query` slice that have neighbors.
    pub row: Vec<u32>,

    /// Indices of neighbor strings. When computing neighbor pairs across separate `query` and
    /// `reference` slices, then `query[row[i]]` and `reference[col[i]]` are neighbors. When
    /// computing neighbor pairs within a single `query` slice, `query[row[i]]` and `query[col[i]]`
    /// are neighbors.
    pub col: Vec<u32>,

    /// Edit distances between neighbor string pairs. When computing neighbor pairs across separate
    /// `query` and `reference` slices, then `Levenshtein(query[row[i]], reference[col[i]]) ==
    /// dists[i]`. When computing neighbor pairs within a single `query` slice,
    /// `Levenshtein(query[row[i]], query[col[i]]) == dists[i]`.
    pub dists: Vec<u8>,
}

impl NeighborPairs {
    /// The number of neighboring string pairs detected.
    pub fn len(&self) -> usize {
        self.row.len()
    }

    /// Returns true if no neighbors were detected.
    pub fn is_empty(&self) -> bool {
        self.row.is_empty()
    }
}

/// A struct for memoizing the deletion variant calculations for a string collection.
///
/// When [constructed](CachedRef::new), [`CachedRef`] precomputes and stores the deletion variants
/// for the supplied `reference` strings as a hashmap. This significantly speeds up subsequent
/// queries against the reference, at the cost of spending extra time to construct the hashmap.
/// This is useful for use-cases where you want to repeatedly query the same reference, especially
/// if the reference is very large. However, for one-off computations, the pure functions
/// [`get_neighbors_within`] and [`get_neighbors_across`] are faster.
///
/// **Note** that [`CachedRef`] instances constructed with `max_distance` set to X can only support
/// queries with `max_distance` less than or equal to X.
///
/// **Note** when interpreting the index order of returned [`NeighborPairs`], the string collection
/// specified at construction is considered the _reference_, and any string collections specified
/// during subsequent query calls are considered the _query_.
///
/// # Examples
///
/// ```
/// use symscan::{CachedRef, NeighborPairs};
///
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let cached = CachedRef::new(&reference, 2).unwrap();
///
/// let NeighborPairs { row, col, dists } = cached
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 1)
///     .unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = cached
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 2)
///     .unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2, 2]);
/// assert_eq!(col,   vec![3, 3, 2, 3]);
/// assert_eq!(dists, vec![2, 1, 2, 0]);
/// ```
pub struct CachedRef {
    store: CachedStore<Levenshtein>,
}

impl CachedRef {
    /// Construct a new [`CachedRef`] instance.
    pub fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
        Ok(CachedRef {
            store: CachedStore::new(reference, max_distance)?,
        })
    }

    /// The memoized equivalent of [`get_neighbors_within`].
    pub fn get_neighbors_within(&self, max_distance: u8) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_within(max_distance)
    }

    /// The memoized equivalent of [`get_neighbors_across`].
    pub fn get_neighbors_across(
        &self,
        query: &[impl AsRef<str> + Sync],
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_across(query, max_distance)
    }

    /// Equivalent to [`CachedRef::get_neighbors_across`], where the query is also a [`CachedRef`]
    /// instance.
    pub fn get_neighbors_across_cached(
        &self,
        query: &Self,
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store
            .get_neighbors_across_cached(&query.store, max_distance)
    }
}

/// A version of [`CachedRef`] but for Hamming distance instead of Levenshtein distance.
///
/// # Examples
///
/// ```
/// use symscan::{CachedRefHamming, NeighborPairs};
///
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let cached_hamming = CachedRefHamming::new(&reference, 2).unwrap();
///
/// let NeighborPairs { row, col, dists } = cached_hamming
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 1)
///     .unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = cached_hamming
///     .get_neighbors_across(&["fizz", "fuzz", "buzz", "fizzy"], 2)
///     .unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2]);
/// assert_eq!(col,   vec![3, 3, 3]);
/// assert_eq!(dists, vec![2, 1, 0]);
/// ```
pub struct CachedRefHamming {
    store: CachedStore<Hamming>,
}

impl CachedRefHamming {
    /// Construct a new [`CachedRefHamming`] instance.
    pub fn new(reference: &[impl AsRef<str> + Sync], max_distance: u8) -> Result<Self, Error> {
        Ok(CachedRefHamming {
            store: CachedStore::new(reference, max_distance)?,
        })
    }

    /// The memoized equivalent of [`get_hamming_neighbors_within`].
    pub fn get_neighbors_within(&self, max_distance: u8) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_within(max_distance)
    }

    /// The memoized equivalent of [`get_hamming_neighbors_across`].
    pub fn get_neighbors_across(
        &self,
        query: &[impl AsRef<str> + Sync],
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store.get_neighbors_across(query, max_distance)
    }

    /// Equivalent to [`CachedRefHamming::get_neighbors_across`], where the query is also a
    /// [`CachedRefHamming`] instance.
    pub fn get_neighbors_across_cached(
        &self,
        query: &Self,
        max_distance: u8,
    ) -> Result<NeighborPairs, Error> {
        self.store
            .get_neighbors_across_cached(&query.store, max_distance)
    }
}

/// Detect string pairs within an input collection that lie within a threshold Levenshtein edit
/// distance.
///
/// The function considers all possible combinations (not permutations, [read
/// more](NeighborPairs#a-note-on-double-counting-pairs)) of string pairs from `query`, and returns
/// all those where the two strings are no more than `max_distance` Levenshtein edit distance units
/// apart.
///
/// # Errors
///
/// Currently, the crate only supports ASCII input. The function will [`Err`] with
/// [`Error::NonAsciiInput`] if `query` contains any non-ASCII data.
///
/// There are some hard limits on the sizes of the input arguments (see [`Error::TooManyStrings`],
/// [`Error::MaxDistCapped`]). Note however that in practice, runtime or memory usage is almost
/// certainly the limiting factor instead.
///
/// # Examples
///
/// ```
/// use symscan::{get_neighbors_within, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let NeighborPairs { row, col, dists } = get_neighbors_within(&query, 1).unwrap();
///
/// assert_eq!(row,   vec![0, 0, 1]);
/// assert_eq!(col,   vec![1, 3, 2]);
/// assert_eq!(dists, vec![1, 1, 1]);
///
/// let NeighborPairs { row, col, dists } = get_neighbors_within(&query, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 0, 0, 1, 1]);
/// assert_eq!(col,   vec![1, 2, 3, 2, 3]);
/// assert_eq!(dists, vec![1, 2, 1, 1, 2]);
/// ```
pub fn get_neighbors_within(
    query: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    functional::get_neighbors_within_impl::<Levenshtein>(query, max_distance)
}

/// Detect string pairs across two input collections that lie within a threshold Levenshtein edit
/// distance.
///
/// The function considers all string pairs in the cartesian product of `query` and `reference`,
/// and returns all those where the two strings are no more than `max_distance` Levenshtein edit
/// distance units apart.
///
/// # Errors
///
/// Currently, the crate only supports ASCII input. The function will [`Err`] with
/// [`Error::NonAsciiInput`] if `query` or `reference` contain any non-ASCII data.
///
/// There are some hard limits on the sizes of the input arguments (see [`Error::TooManyStrings`],
/// [`Error::MaxDistCapped`]). Note however that in practice, runtime or memory usage is almost
/// certainly the limiting factor instead.
///
/// # Examples
///
/// ```
/// use symscan::{get_neighbors_across, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let NeighborPairs { row, col, dists } = get_neighbors_across(&query, &reference, 1).unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = get_neighbors_across(&query, &reference, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2, 2]);
/// assert_eq!(col,   vec![3, 3, 2, 3]);
/// assert_eq!(dists, vec![2, 1, 2, 0]);
/// ```
pub fn get_neighbors_across(
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    functional::get_neighbors_across_impl::<Levenshtein>(query, reference, max_distance)
}

/// A version of [`get_neighbors_within`] which uses Hamming distance instead of Levenshtein
/// distance.
///
/// # Examples
///
/// ```
/// use symscan::{get_hamming_neighbors_within, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_within(&query, 1).unwrap();
///
/// assert_eq!(row,   vec![0, 1]);
/// assert_eq!(col,   vec![1, 2]);
/// assert_eq!(dists, vec![1, 1]);
///
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_within(&query, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 0, 1]);
/// assert_eq!(col,   vec![1, 2, 2]);
/// assert_eq!(dists, vec![1, 2, 1]);
/// ```
pub fn get_hamming_neighbors_within(
    query: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    functional::get_neighbors_within_impl::<Hamming>(query, max_distance)
}

/// A version of [`get_neighbors_across`] which uses Hamming distance instead of Levenshtein
/// distance.
///
/// # Examples
///
/// ```
/// use symscan::{get_hamming_neighbors_across, NeighborPairs};
///
/// let query = ["fizz", "fuzz", "buzz", "fizzy"];
/// let reference = ["foo", "bar", "baz", "buzz"];
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_across(&query, &reference, 1).unwrap();
///
/// assert_eq!(row,   vec![1, 2]);
/// assert_eq!(col,   vec![3, 3]);
/// assert_eq!(dists, vec![1, 0]);
///
/// let NeighborPairs { row, col, dists } = get_hamming_neighbors_across(&query, &reference, 2).unwrap();
///
/// assert_eq!(row,   vec![0, 1, 2]);
/// assert_eq!(col,   vec![3, 3, 3]);
/// assert_eq!(dists, vec![2, 1, 0]);
/// ```
pub fn get_hamming_neighbors_across(
    query: &[impl AsRef<str> + Sync],
    reference: &[impl AsRef<str> + Sync],
    max_distance: u8,
) -> Result<NeighborPairs, Error> {
    functional::get_neighbors_across_impl::<Hamming>(query, reference, max_distance)
}

#[cfg(test)]
mod tests {
    use std::io::{self, BufRead, Cursor};

    use super::*;

    static CDR3_Q_BYTES: &[u8] = include_bytes!("../../test_files/cdr3b_10k_a.txt");
    static CDR3_R_BYTES: &[u8] = include_bytes!("../../test_files/cdr3b_10k_b.txt");
    static EXPECTED_BYTES_WITHIN_1: &[u8] = include_bytes!("../../test_files/results_10k_a.txt");
    static EXPECTED_BYTES_WITHIN_2: &[u8] = include_bytes!("../../test_files/results_10k_a_d2.txt");
    static EXPECTED_BYTES_CROSS_1: &[u8] = include_bytes!("../../test_files/results_10k_cross.txt");
    static EXPECTED_BYTES_CROSS_2: &[u8] =
        include_bytes!("../../test_files/results_10k_cross_d2.txt");
    static EXPECTED_BYTES_HAMMING_WITHIN_1: &[u8] =
        include_bytes!("../../test_files/results_10k_a_hamming.txt");
    static EXPECTED_BYTES_HAMMING_WITHIN_2: &[u8] =
        include_bytes!("../../test_files/results_10k_a_hamming_d2.txt");
    static EXPECTED_BYTES_HAMMING_CROSS_1: &[u8] =
        include_bytes!("../../test_files/results_10k_cross_hamming.txt");
    static EXPECTED_BYTES_HAMMING_CROSS_2: &[u8] =
        include_bytes!("../../test_files/results_10k_cross_hamming_d2.txt");

    fn bytes_as_ascii_lines(bytes: &[u8]) -> Vec<String> {
        Cursor::new(bytes)
            .lines()
            .collect::<io::Result<Vec<String>>>()
            .expect("test files have valid lines")
    }

    fn bytes_as_neighbour_pairs(bytes: &[u8]) -> NeighborPairs {
        let mut i = Vec::new();
        let mut j = Vec::new();
        let mut dists = Vec::new();

        Cursor::new(bytes).lines().for_each(|v| {
            let line = v.expect("test files have valid lines");
            let triplet: Vec<_> = line.split(",").collect();
            i.push(
                triplet[0]
                    .parse::<u32>()
                    .expect("test files have int triplets")
                    - 1,
            );
            j.push(
                triplet[1]
                    .parse::<u32>()
                    .expect("test files have int triplets")
                    - 1,
            );
            dists.push(
                triplet[2]
                    .parse::<u8>()
                    .expect("test files have int triplets"),
            );
        });

        NeighborPairs {
            row: i,
            col: j,
            dists,
        }
    }

    #[test]
    fn test_within() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);

        let hits = get_neighbors_within(&query, 1).expect("short input");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_1));

        let hits = get_neighbors_within(&query, 2).expect("short input");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_2));
    }

    #[test]
    fn test_cross() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);

        let hits = get_neighbors_across(&query, &reference, 1).expect("valid inputs");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_1));

        let hits = get_neighbors_across(&query, &reference, 2).expect("valid inputs");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_2));
    }

    #[test]
    fn test_hamming_within() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);

        let hits = get_hamming_neighbors_within(&query, 1).expect("short input");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_1)
        );

        let hits = get_hamming_neighbors_within(&query, 2).expect("short input");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_2)
        );
    }

    #[test]
    fn test_hamming_cross() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);

        let hits = get_hamming_neighbors_across(&query, &reference, 1).expect("valid inputs");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_1)
        );

        let hits = get_hamming_neighbors_across(&query, &reference, 2).expect("valid inputs");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_2)
        );
    }

    #[test]
    fn test_within_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let cached = CachedRef::new(&query, 2).expect("short input");

        let hits = cached.get_neighbors_within(1).expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_1));

        let hits = cached.get_neighbors_within(2).expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_WITHIN_2));
    }

    #[test]
    fn test_hamming_within_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let cached_hamming = CachedRefHamming::new(&query, 2).expect("short input");

        let hits = cached_hamming
            .get_neighbors_within(1)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_1)
        );

        let hits = cached_hamming
            .get_neighbors_within(2)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_WITHIN_2)
        );
    }

    #[test]
    fn test_cross_partially_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached = CachedRef::new(&reference, 2).expect("short input");

        let hits = cached
            .get_neighbors_across(&query, 1)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_1));

        let hits = cached
            .get_neighbors_across(&query, 2)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_2));
    }

    #[test]
    fn test_cross_fully_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached_query = CachedRef::new(&query, 2).expect("short input");
        let cached_reference = CachedRef::new(&reference, 2).expect("short input");

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 1)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_1));

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 2)
            .expect("legal max distance");
        assert_eq!(hits, bytes_as_neighbour_pairs(EXPECTED_BYTES_CROSS_2));
    }

    #[test]
    fn test_hamming_cross_partially_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached_hamming = CachedRefHamming::new(&reference, 2).expect("short input");

        let hits = cached_hamming
            .get_neighbors_across(&query, 1)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_1)
        );

        let hits = cached_hamming
            .get_neighbors_across(&query, 2)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_2)
        );
    }

    #[test]
    fn test_hamming_cross_fully_cached() {
        let query = bytes_as_ascii_lines(CDR3_Q_BYTES);
        let reference = bytes_as_ascii_lines(CDR3_R_BYTES);
        let cached_query = CachedRefHamming::new(&query, 2).expect("short input");
        let cached_reference = CachedRefHamming::new(&reference, 2).expect("short input");

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 1)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_1)
        );

        let hits = cached_reference
            .get_neighbors_across_cached(&cached_query, 2)
            .expect("legal max distance");
        assert_eq!(
            hits,
            bytes_as_neighbour_pairs(EXPECTED_BYTES_HAMMING_CROSS_2)
        );
    }
}
