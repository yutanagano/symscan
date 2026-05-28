CLI Application
===============

Installation
------------

.. code-block:: console

   $ brew install yutanagano/tap/symscan-cli

You can also directly download precompiled binaries from the project `releases
page <https://github.com/yutanagano/symscan/releases>`_.


Usage
-----

.. tip:: 

   You can also view symscan's inline help text with ``symscan --help``.

Give symscan a list of strings, and it will tell you which ones are similar. By
default, it will detect which strings are within one (Levenshtein) edit
distance away from one another. Symscan reads its standard input stream and
considers each line (delineated by newline characters) a separate string. A
minimal example is below:

.. code-block:: console

   $ echo $'fizz\nfuzz\nbuzz\nfizzy' | symscan
   1,2,1
   1,4,1
   2,3,1

As you can see, symscan outputs its result in plaintext to standard output.
Each line in its output corresponds to a pair of similar strings that is
detected. The first two numbers in each line is the (1-indexed) line numbers
corresponding to the two similar input strings. The third and final number is
the number of edits separating the two strings.

Options
.......

To look for string pairs that are at most ``<k>`` edits away from each other,
pass the option ``-d <k>``:

.. code-block:: console

   $ echo $'fizz\nfuzz\nbuzz\nfizzy' | symscan -d 2
   1,2,1
   1,3,2
   1,4,1
   2,3,1
   2,4,2

If you want the output to have 0-indexed line numbers as opposed to 1-indexed,
pass the option ``-z``:

.. code-block:: console

   $ echo $'fizz\nfuzz\nbuzz' | symscan -d 2 -z
   0,1,1
   0,2,2
   0,3,1
   1,2,1
   1,3,2

If you want to limit your neighbor search to only allow substitutions, you can
set the ``--hamming`` option. Note that in this case, input strings with
different character lengths will never be considered neighbors.

.. code-block:: console

   $ echo $'fizz\nfuzz\nbuzz' | symscan -d 2 -z --hamming
   0,1,1
   0,2,2
   1,2,1

Read from and write to files
............................

To read input from ``input.txt`` and write to ``output.txt``:

.. code-block:: console

   $ symscan input.txt > output.txt

or

.. code-block:: console

   $ < input.txt symscan > output.txt

or

.. code-block:: console

   $ cat input.txt | symscan > output.txt

Look for pairs across two string sets
.....................................

To look strictly for strings in ``set_a.txt`` that are similar to strings in
``set_b.txt`` (ignores pairs within sets):

.. code-block:: console

   $ symscan set_a.txt set_b.txt > output.txt
