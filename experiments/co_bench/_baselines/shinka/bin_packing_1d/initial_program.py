"""Minimal seed program for the ShinkaEvolve baseline on co_bench / bin_packing_1d.

The ShinkaEvolve runner edits only the region between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END; the rest of the file stays fixed.

The CO-Bench one-dimensional bin packing evaluator calls ``solve(**instance)``
where each instance provides ``id, bin_capacity, num_items, items, best_known``,
and expects a dict ``{"num_bins": int, "bins": [[1-based item indices], ...]}``.
Putting one item per bin is always feasible (no bin can exceed capacity and every
item appears exactly once), so it scores > 0.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def solve(num_items, **kwargs):
    # Trivial valid baseline: one item per bin.
    return {"num_bins": num_items, "bins": [[i] for i in range(1, num_items + 1)]}
# EVOLVE-BLOCK-END
