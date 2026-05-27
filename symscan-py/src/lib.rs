use numpy::IntoPyArray;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyString, PyTuple},
};
use symscan::NeighborPairs;

enum CachedRefInternal {
    Levenshtein(symscan::CachedRef),
    Hamming(symscan::CachedRefHamming),
}

impl CachedRefInternal {
    fn new(
        reference: &[impl AsRef<str> + Sync],
        max_distance: u8,
        distance_type: &str,
    ) -> PyResult<Self> {
        match distance_type {
            "levenshtein" => Ok(Self::Levenshtein(
                symscan::CachedRef::new(reference, max_distance)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )),
            "hamming" => Ok(Self::Hamming(
                symscan::CachedRefHamming::new(reference, max_distance)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )),
            _ => Err(PyValueError::new_err(
                "distance_type must be either levenshtein or hamming",
            )),
        }
    }

    fn get_neighbors_within(&self, max_distance: u8) -> PyResult<NeighborPairs> {
        match self {
            Self::Levenshtein(x) => x.get_neighbors_within(max_distance),
            Self::Hamming(x) => x.get_neighbors_within(max_distance),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_neighbors_across(
        &self,
        query: &[impl AsRef<str> + Sync],
        max_distance: u8,
    ) -> PyResult<NeighborPairs> {
        match self {
            Self::Levenshtein(x) => x.get_neighbors_across(query, max_distance),
            Self::Hamming(x) => x.get_neighbors_across(query, max_distance),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn get_neighbors_across_cached(
        &self,
        query: &Self,
        max_distance: u8,
    ) -> PyResult<NeighborPairs> {
        match self {
            Self::Levenshtein(x) => match query {
                Self::Levenshtein(y) => x
                    .get_neighbors_across_cached(y, max_distance)
                    .map_err(|e| PyValueError::new_err(e.to_string())),
                _ => Err(PyTypeError::new_err(
                    "CachedRefHamming cannot be used to query CachedRef",
                )),
            },
            Self::Hamming(x) => match query {
                Self::Hamming(y) => x
                    .get_neighbors_across_cached(y, max_distance)
                    .map_err(|e| PyValueError::new_err(e.to_string())),
                _ => Err(PyTypeError::new_err(
                    "CachedRef cannot be used to query CachedRefHamming",
                )),
            },
        }
    }
}

/// A class for memoizing the deletion variant calculations for a string collection.
///
/// When constructed, the CachedRef instance precomputes and stores the deletion variants for the
/// supplied `reference` strings as a hashmap. This significantly speeds up subsequent queries
/// against the reference, at the upfront cost of spending extra time to construct the hashmap.
/// This is useful for use-cases where you want to repeatedly query the same reference, especially
/// if the reference is very large. However, for one-off computations, the pure functions
/// :py:func:`~symscan.get_neighbors_within` and :py:func:`~symscan.get_neighbors_across` are
/// faster.
///
/// .. note::
///     When interpreting the index order of method return values, the string collection specified
///     at construction is considered the `reference`, and any string collections specified during
///     subsequent query calls are considered the `query`.
///
/// Parameters
/// ----------
/// reference
/// max_distance
///     The maximum edit distance that this CachedRef instance will be able to support in future
///     queries.
/// distance_type
///     The string metric to use when measuring distance between input strings. When using Hamming
///     distance, strings of different lengths will never be considered neighbors.
#[pyclass]
struct CachedRef {
    internal: CachedRefInternal,
}

#[pymethods]
impl CachedRef {
    #[new]
    #[pyo3(signature = (reference, max_distance = 1, distance_type = "levenshtein"))]
    fn new(reference: &Bound<PyAny>, max_distance: u8, distance_type: &str) -> PyResult<Self> {
        check_distance_type(distance_type)?;
        let ref_handles = get_pystring_handles(reference)?;
        let ref_views = get_str_refs(&ref_handles)?;

        let internal = CachedRefInternal::new(&ref_views, max_distance, distance_type)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(CachedRef { internal })
    }

    /// The memoized equivalent of :py:func:`~symscan.get_neighbors_within`.
    ///
    /// Parameters
    /// ----------
    /// max_distance : int, default=1
    ///     The maximum edit distance at which strings are considered neighbours. This must not be
    ///     greater than the `max_distance` specified when constructing the caller instance.
    ///
    /// Returns
    /// -------
    /// row : ndarray of shape (N,), dtype=uint32
    ///     Indices of strings in the cached reference that have neighbors.
    ///
    /// col : ndarray of shape (N,), dtype=uint32
    ///     Indices of neighbor strings (i.e. ``reference[row[i]]`` and ``reference[col[i]]`` are
    ///     neighbors).
    ///
    /// dists : ndarray of shape (N,), dtype=uint8
    ///     Edit distances between neighbors (i.e. ``Levenshtein(reference[row[i]],
    ///     reference[col[i]]) = dists[i]``).
    ///
    /// Examples
    /// --------
    /// Look for pairs of similar strings within a string collection.
    ///
    /// >>> import symscan
    /// >>> cached = symscan.CachedRef(["fizz", "fuzz", "buzz"])
    /// >>> (row, col, dists) = cached.get_neighbors_within()
    /// >>> row
    /// array([0, 1], dtype=uint32)
    /// >>> col
    /// array([1, 2], dtype=uint32)
    /// >>> dists
    /// array([1, 1], dtype=uint8)
    #[pyo3(signature = (max_distance = 1))]
    fn get_neighbors_within<'py>(
        &self,
        py: Python<'py>,
        max_distance: u8,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let symscan::NeighborPairs { row, col, dists } =
            self.internal.get_neighbors_within(max_distance)?;

        PyTuple::new(
            py,
            [
                row.into_pyarray(py).as_any(),
                col.into_pyarray(py).as_any(),
                dists.into_pyarray(py).as_any(),
            ],
        )
    }

    /// The memoized equivalent of :py:func:`~symscan.get_neighbors_across`.
    ///
    /// Parameters
    /// ----------
    /// query : iterable of str or CachedRef
    /// max_distance : int, default=1
    ///     The maximum edit distance at which strings are considered neighbours.
    ///
    /// Returns
    /// -------
    /// row : ndarray of shape (N,), dtype=uint32
    ///     Indices of strings in the query that have neighbors.
    ///
    /// col : ndarray of shape (N,), dtype=uint32
    ///     Indices of neighbor strings (i.e. ``query[row[i]]`` and ``reference[col[i]]`` are
    ///     neighbors).
    ///
    /// dists : ndarray of shape (N,), dtype=uint8
    ///     Edit distances between neighbors (i.e. ``Levenshtein(query[row[i]], reference[col[i]]) =
    ///     dists[i]``).
    ///
    /// Examples
    /// --------
    /// Look for pairs of similar strings across the cached reference and a query collection.
    ///
    /// >>> import symscan
    /// >>> cached = symscan.CachedRef(["fooo", "barr", "bazz", "buzz"])
    /// >>> (row, col, dists) = cached.get_neighbors_across(["fizz", "fuzz", "buzz"])
    /// >>> row
    /// array([1, 2, 2], dtype=uint32)
    /// >>> col
    /// array([3, 2, 3], dtype=uint32)
    /// >>> dists
    /// array([1, 1, 0], dtype=uint8)
    ///
    /// It is possible to use a CachedRef instance as the query collection as well.
    ///
    /// >>> cached_query = symscan.CachedRef(["fizz", "fuzz", "buzz"])
    /// >>> (row, col, dists) = cached.get_neighbors_across(cached_query)
    /// >>> row
    /// array([1, 2, 2], dtype=uint32)
    /// >>> col
    /// array([3, 2, 3], dtype=uint32)
    /// >>> dists
    /// array([1, 1, 0], dtype=uint8)
    #[pyo3(signature = (query, max_distance = 1))]
    fn get_neighbors_across<'py>(
        &self,
        py: Python<'py>,
        query: Bound<'py, PyAny>,
        max_distance: u8,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let symscan::NeighborPairs { row, col, dists } = {
            if let Ok(cached) = query.cast::<CachedRef>() {
                self.internal
                    .get_neighbors_across_cached(&cached.borrow().internal, max_distance)?
            } else if let Ok(iterable) = query.try_iter() {
                let query_handles = get_pystring_handles(&iterable)?;
                let query_views = get_str_refs(&query_handles)?;
                self.internal
                    .get_neighbors_across(&query_views, max_distance)?
            } else {
                let type_name = query
                    .get_type()
                    .name()
                    .map(|pys| pys.to_string())
                    .unwrap_or("UNKNOWN".to_string());
                return Err(PyValueError::new_err(format!(
                        "query must be either an iterable of str or CachedRef or None, got '{type_name}'",
                    )));
            }
        };

        PyTuple::new(
            py,
            [
                row.into_pyarray(py).as_any(),
                col.into_pyarray(py).as_any(),
                dists.into_pyarray(py).as_any(),
            ],
        )
    }
}

/// Detect string pairs within an input collection that lie within a threshold edit distance.
///
/// The function considers all possible combinations of string pairs from `query`, and returns all
/// those where the two strings are no more than `max_distance` Levenshtein edit distance units
/// apart.
///
/// .. important::
///
///     This function **DOES NOT** double-count string pairs. As seen in the examples below, each
///     pair is represented once where the `row` index is always less than the `col` index. In
///     other words, if you were to interpret the output as a sparse matrix, only the lower
///     triangle will be filled.
///
/// Parameters
/// ----------
/// query : iterable of str
/// max_distance : int, default=1
///     The maximum edit distance at which strings are considered neighbours.
///
/// Returns
/// -------
/// row : ndarray of shape (N,), dtype=uint32
///     Indices of strings in the query that have neighbors.
///
/// col : ndarray of shape (N,), dtype=uint32
///     Indices of neighbor strings (i.e. ``query[row[i]]`` and ``query[col[i]]`` are neighbors).
///
/// dists : ndarray of shape (N,), dtype=uint8
///     Edit distances between neighbors (i.e. ``Levenshtein(query[row[i]], query[col[i]]) =
///     dists[i]``).
///
/// Examples
/// --------
/// Look for pairs of similar strings within a string collection. Note how string pairs are not
/// double-counted, each is counted once such that the `row` coordinate is always less than the
/// `col` coordinate.
///
/// >>> import symscan
/// >>> (row, col, dists) = symscan.get_neighbors_within(["fizz", "fuzz", "buzz"])
/// >>> row
/// array([0, 1], dtype=uint32)
/// >>> col
/// array([1, 2], dtype=uint32)
/// >>> dists
/// array([1, 1], dtype=uint8)
///
/// To increase the threshold at which string pairs are considered similar, set `max_distance`.
///
/// >>> (row, col, dists) = symscan.get_neighbors_within(["fizz", "fuzz", "buzz"], max_distance=2)
/// >>> row
/// array([0, 0, 1], dtype=uint32)
/// >>> col
/// array([1, 2, 2], dtype=uint32)
/// >>> dists
/// array([1, 2, 1], dtype=uint8)
#[pyfunction]
#[pyo3(signature = (query, max_distance = 1))]
fn get_neighbors_within<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    max_distance: u8,
) -> PyResult<Bound<'py, PyTuple>> {
    let query_handles = get_pystring_handles(query)?;
    let query_views = get_str_refs(&query_handles)?;

    let symscan::NeighborPairs { row, col, dists } =
        symscan::get_neighbors_within(&query_views, max_distance)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

    PyTuple::new(
        py,
        [
            row.into_pyarray(py).as_any(),
            col.into_pyarray(py).as_any(),
            dists.into_pyarray(py).as_any(),
        ],
    )
}

/// Detect string pairs across two input collections that lie within a threshold edit distance.
///
/// The function considers all string pairs in the cartesian product of `query` and `reference`,
/// and returns all those where the two strings are no more than `max_distance` Levenshtein edit
/// distance units apart.
///
/// Parameters
/// ----------
/// query : iterable of str
/// reference : iterable of str
/// max_distance : int, default=1
///     The maximum edit distance at which strings are considered neighbors.
///
/// Returns
/// -------
/// row : ndarray of shape (N,), dtype=uint32
///     Indices of strings in the query that have neighbors.
///
/// col : ndarray of shape (N,), dtype=uint32
///     Indices of neighbor strings (i.e. ``query[row[i]]`` and ``reference[col[i]]`` are
///     neighbors).
///
/// dists : ndarray of shape (N,), dtype=uint8
///     Edit distances between neighbors (i.e. ``Levenshtein(query[row[i]], reference[col[i]]) =
///     dists[i]``).
///
/// Examples
/// --------
/// Look for pairs of similar strings across two collections.
///
/// >>> (row, col, dists) = symscan.get_neighbors_across(["fizz", "fuzz", "buzz"], ["fooo", "barr", "bazz", "buzz"])
/// >>> row
/// array([1, 2, 2], dtype=uint32)
/// >>> col
/// array([3, 2, 3], dtype=uint32)
/// >>> dists
/// array([1, 1, 0], dtype=uint8)
///
/// To increase the threshold at which string pairs are considered similar, set `max_distance`.
///
/// >>> (row, col, dists) = symscan.get_neighbors_across(["fizz", "fuzz", "buzz"], ["fooo", "barr", "bazz", "buzz"], max_distance=2)
/// >>> row
/// array([0, 0, 1, 1, 2, 2], dtype=uint32)
/// >>> col
/// array([2, 3, 2, 3, 2, 3], dtype=uint32)
/// >>> dists
/// array([2, 2, 2, 1, 1, 0], dtype=uint8)
#[pyfunction]
#[pyo3(signature = (query, reference, max_distance = 1))]
fn get_neighbors_across<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    reference: Bound<'py, PyAny>,
    max_distance: u8,
) -> PyResult<Bound<'py, PyTuple>> {
    let query_handles = get_pystring_handles(query)?;
    let query_views = get_str_refs(&query_handles)?;
    let ref_handles = get_pystring_handles(&reference)?;
    let ref_views = get_str_refs(&ref_handles)?;

    let symscan::NeighborPairs { row, col, dists } = {
        symscan::get_neighbors_across(&query_views, &ref_views, max_distance)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    PyTuple::new(
        py,
        [
            row.into_pyarray(py).as_any(),
            col.into_pyarray(py).as_any(),
            dists.into_pyarray(py).as_any(),
        ],
    )
}

fn check_distance_type(distance_type: &str) -> PyResult<()> {
    match distance_type {
        "levenshtein" | "hamming" => Ok(()),
        _ => Err(PyValueError::new_err(
            "distance_type must be either levenshtein or hamming",
        )),
    }
}

fn get_pystring_handles<'py>(input: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyString>>> {
    if input.cast::<PyString>().is_ok() {
        Err(PyValueError::new_err("expected iterable of str, got str"))
    } else {
        input
            .try_iter()?
            .map(|v| v?.cast_into::<PyString>().map_err(PyErr::from))
            .collect::<PyResult<Vec<_>>>()
    }
}

fn get_str_refs<'py>(input: &'py [Bound<'py, PyString>]) -> PyResult<Vec<&'py str>> {
    input
        .iter()
        .map(|v| v.to_str())
        .collect::<PyResult<Vec<_>>>()
}

/// Fast discovery of similar strings in bulk
#[pymodule(name = "symscan")]
fn symscan_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_neighbors_within, m)?)?;
    m.add_function(wrap_pyfunction!(get_neighbors_across, m)?)?;
    m.add_class::<CachedRef>()?;
    Ok(())
}
