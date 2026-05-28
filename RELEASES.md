## Version 0.8.0

- Implement optimised Hamming distance versions of the functions as well as a
  cached Hamming implementation (as opposed to the default Levenshtein distance
  versions)
- Start using stable Python ABI for Python bindings (locked at v >= 3.10 for
  now)

## Version 0.7.2

- (Fix) Executable name is fixed to "symscan" from "symscan-cli"
- Clean up README

## Version 0.7.1

- Bugfixes with license packaging

## Version 0.7.0

- Reduce unnecessary allocations when computing string variant hashes
- Obtain references to Python strings at the PyO3 border instead of allocating and copying new Rust Strings
- Refactor code to publish core code as Rust library crate
- Refactor Python bindings to have more explicit, descriptive function names

## Version 0.6.1

- Faster Levenshtein distance calculations making use of max distance cutoff
- Preallocation for backing memory of CachedSymdel
- Direct exposure of Rust functions to Python
- Use of 32 bit indexes for string sets instead of 64 bits

## Version 0.6.0

- Change in return types of Python API:
  - Previously returned Python list of tuples of integers
  - Now returns three Numpy arrays
- Greatly improved runtime and memory usage
- More reliable freeing of OS memory after finishing computations

## Version 0.5.1

- Performance improvements
- Fixed bug that prevented users from specifying a `max_distance` greater than
  the length of the shortest string in the input sets

## Version 0.5.0

- Implement a memoized implementation of symdel for use in the Python package

## Version 0.4.0

- Add Python bindings
- Add documentation page
- Remove previous restriction where only input strings of length < 255
  characters were allowed
- Build CLI and Python package for all major platforms (ARM platforms newly
  added)

## Version 0.3.0

- Add option to make output 0-indexed instead of the default 1-indexed
- Support new installation methods:
  - homebrew
  - shell scripts

## Version 0.2.0

- Implement cross-search feature, symscan can now look for pairs of similar
  strings across two inputs
- Further performance optimisations

## Version 0.1.0

- First working prototype
- Fast detection of similar strings within one input, which is read from
  standard input
