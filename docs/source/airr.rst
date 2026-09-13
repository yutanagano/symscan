AIRR CLI Application
====================

Installation
------------

.. code-block:: console

   $ brew install yutanagano/tap/symscan-airr

Or install from crates.io (useful on Windows and anywhere Homebrew is
unavailable):

.. code-block:: console

   $ cargo install symscan-airr

You can also directly download precompiled binaries from the project `releases
page <https://github.com/yutanagano/symscan/releases>`_.


Usage
-----

.. tip::

   You can also view symscan-airr's inline help text with ``symscan-airr --help``.

``symscan-airr`` measures overlap between adaptive immune receptor repertoires
(AIRRs), powered by the SymScan neighbour-search algorithm. Give it an
`AIRR-compliant <https://docs.airr-community.org/en/stable/datarep/rearrangements.html>`_
TSV of rearrangements (or stream one on stdin), and it reports how much each
pair of repertoires overlaps.

Overlap is the total number of AIR pairs between two repertoires whose junction
sequences fall within the similarity threshold, weighted by
``duplicate_count``. By default the threshold is one Levenshtein edit.

A minimal example is below. First write a small AIRR TSV, then pass it as a
file argument (or pipe it on stdin — both are equivalent):

.. code-block:: console

   $ cat > example.tsv <<'EOF'
   junction_aa	duplicate_count	repertoire_id
   CAVSTSGGSYIPTF	1	a
   CAVHASGGSYIPTF	1	a
   CAVSTSGGSYIPTF	1	b
   CAVRLSGGSYIPTF	2	b
   EOF
   $ symscan-airr example.tsv
   a	a	2
   a	b	1
   b	b	5
   $ < example.tsv symscan-airr
   a	a	2
   a	b	1
   b	b	5

Each output line is a tab-separated triplet
``<repertoire_a>``, ``<repertoire_b>``, ``<overlap>`` written to standard
output (no header). Within a single input file, only the upper triangle and
diagonal are emitted, so each unordered repertoire pair appears once and
self-vs-self scores are included.

To write the result to a file:

.. code-block:: console

   $ symscan-airr example.tsv > overlap.tsv

Input
.....

The input must be a delimited table with a header row. By default the
separator is a tab and the required columns are:

- ``junction_aa`` — junction (CDR3) amino-acid sequence used for similarity
  search
- ``duplicate_count`` — abundance / clone count for that rearrangement
- ``repertoire_id`` — repertoire or sample identifier

Extra columns are ignored. Rows with an empty junction, duplicate count, or
repertoire ID are skipped. Non-ASCII junctions and unparseable duplicate
counts are errors.

Options
.......

To look for junction pairs that are at most ``<k>`` edits away from each
other, pass the option ``-d <k>``:

.. code-block:: console

   $ symscan-airr -d 2 example.tsv
   a	a	4
   a	b	6
   b	b	9

If you want to limit the neighbour search to substitutions only, set
``--hamming``. Junctions of different lengths are never neighbours under
Hamming distance.

If your table mixes loci (for example TRA and TRB), restrict the analysis with
``--locus``:

.. code-block:: console

   $ symscan-airr --locus TRB rearrangements.tsv > overlap.tsv

By default the program reads every row where the junction, duplicate count,
and repertoire ID are set, regardless of locus. The locus column (default
name ``locus``) is only required when ``--locus`` is set; override the column
name with ``--locus-col`` if needed.

Non-standard column names can be remapped:

.. code-block:: console

   $ symscan-airr --junction-col cdr3_aa --count-col umi_count \
       --repertoire-col sample_id rearrangements.tsv > overlap.tsv

For CSV (or any other delimiter), pass ``-s`` / ``--sep``:

.. code-block:: console

   $ symscan-airr -s ',' rearrangements.csv > overlap.tsv

Control parallelism with ``-n`` / ``--num-threads`` (default: one thread per
CPU core).

Compare repertoires across two files
....................................

To score every repertoire in a query file against every repertoire in a
reference file (without comparing repertoires from the same file to each
other):

.. code-block:: console

   $ symscan-airr cohort_a.tsv cohort_b.tsv > cross_overlap.tsv

The output is the full cartesian product of query repertoire names and
reference repertoire names.
