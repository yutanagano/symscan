SymScan
=======

**SymScan** enables extremely fast discovery of pairs of similar strings within
and across large collections. It is available as a :doc:`CLI tool <cli>`, an
:doc:`AIRR repertoire comparison tool <airr>`, a :doc:`Rust library <rust>`,
and a :doc:`Python package <py>`.

SymScan is a variation on the `symmetric deletion
<https://seekstorm.com/blog/1000x-spelling-correction/>`_ algorithm that is
optimised for bulk-searching similar strings within one or across two large
string collections at once (e.g. searching for similar protein sequences among
a collection of 10M). The key algorithmic difference between SymScan and
traditional symmetric deletion is the use of a `sort-merge join
<https://en.wikipedia.org/wiki/Sort-merge_join>`_ approach in place of hash
maps to discover input strings that share common deletion variants. This
sort-and-scan approach trades off an additional factor of O(log N) (with N the
total number of strings being compared) in expected time complexity for
improved cache locality and effective parallelization, and ends up being much
faster for the above use case.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   cli
   airr
   rust
   py
   citing
