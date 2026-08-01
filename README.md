# SymScan

[![Crates.io](https://img.shields.io/crates/v/symscan.svg)](https://crates.io/crates/symscan)
[![PyPI](https://img.shields.io/pypi/v/symscan.svg)](https://pypi.org/project/symscan/)
[![Docs](https://readthedocs.org/projects/symscan/badge/?version=latest)](https://symscan.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#licensing)

### Check out the [documentation page](https://symscan.readthedocs.io).

**SymScan** enables extremely fast discovery of pairs of similar strings within
and across large collections.

SymScan is a variation on the [symmetric deletion
](https://seekstorm.com/blog/1000x-spelling-correction/) algorithm that is
optimised for bulk-searching similar strings within one or across two large
string collections at once (e.g. searching for similar protein sequences among
a collection of 10M). The key algorithmic difference between SymScan and
traditional symmetric deletion is the use of a [sort-merge
join](https://en.wikipedia.org/wiki/Sort-merge_join) approach in place of hash
maps to discover input strings that share common deletion variants. This
sort-and-scan approach trades off an additional factor of O(log N) (with N the
total number of strings being compared) in expected time complexity for
improved cache locality and effective parallelization, and ends up being much
faster for the above use case.

## Installing

### CLI

```sh
brew install yutanagano/tap/symscan-cli
```

Or via installation from source
```sh
git clone https://github.com/yutanagano/symscan.git
cd symscan
cargo install --path symscan-cli
```

### Rust library

```sh
cargo add symscan
```

### Python package

```sh
pip install symscan
```

## Quick start

### CLI

SymScan takes in a list of strings (one per line) via stdin, and returns which ones are within one Levenshtein edit (the default) of each other. Each output line is
`<line 1>,<line 2>,<edit distance>` (1-indexed):

```sh
$ echo $'fizz\nfuzz\nbuzz\nfizzy' | symscan
1,2,1
1,4,1
2,3,1
```

See the [CLI docs](https://symscan.readthedocs.io/en/latest/cli.html) for
options like `-d` (max distance), `-z` (0-indexed output), `--hamming`, and
searching across two files.

### Rust

```rust
use symscan::{get_neighbors_within, NeighborPairs};

let query = ["fizz", "fuzz", "buzz", "fizzy"];
let NeighborPairs { row, col, dists } = get_neighbors_within(&query, 1).unwrap();

assert_eq!(row,   vec![0, 0, 1]);
assert_eq!(col,   vec![1, 3, 2]);
assert_eq!(dists, vec![1, 1, 1]);
```

See the [crate docs](https://docs.rs/symscan/latest/symscan/) for
searching across two collections and the memoized `CachedRef` API.

### Python

```python
>>> import symscan
>>> row, col, dists = symscan.get_neighbors_within(["fizz", "fuzz", "buzz", "fizzy"])
>>> row
array([0, 0, 1], dtype=uint32)
>>> col
array([1, 3, 2], dtype=uint32)
>>> dists
array([1, 1, 1], dtype=uint8)
```

See the [Python docs](https://symscan.readthedocs.io/en/latest/py.html)
for searching across two collections and the memoized `CachedRef` API.

## Licensing

SymScan is dual-licensed under the MIT and Apache 2.0 licenses. Unless
explicitly stated otherwise, any contribution submitted by you, as defined in
the Apache license, shall be dual-licensed as above, without any additional
terms and conditions.
