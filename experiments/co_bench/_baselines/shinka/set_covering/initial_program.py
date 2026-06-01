"""Minimal seed program for the ShinkaEvolve baseline on co_bench / set_covering.

The ShinkaEvolve runner edits only the region between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END; the rest of the file stays fixed.

The CO-Bench Set covering evaluator calls ``solve(**instance)`` where each
instance provides ``m`` (rows), ``n`` (columns), ``costs`` (per-column cost),
and ``row_cover`` (per row, the 1-indexed columns that cover it), and expects a
dict ``{"selected_columns": [1-indexed columns]}``. Selecting every column
``list(range(1, n + 1))`` covers every row and scores > 0.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def solve(n, **kwargs):
    # Trivial valid baseline: select all columns (covers every row).
    return {"selected_columns": list(range(1, n + 1))}
# EVOLVE-BLOCK-END
