"""Minimal seed program for the ShinkaEvolve baseline on co_bench / tsp.

The ShinkaEvolve runner edits only the region between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END; the rest of the file stays fixed.

The CO-Bench TSP evaluator calls ``solve(**instance)`` where each instance
provides ``nodes=[(x, y), ...]`` (and ``label_tour``), and expects a dict
``{"tour": [node indices]}``. The identity tour ``list(range(len(nodes)))``
is a valid Hamiltonian cycle and scores > 0.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def solve(nodes, **kwargs):
    # Trivial valid baseline: identity tour visiting nodes in input order.
    return {"tour": list(range(len(nodes)))}
# EVOLVE-BLOCK-END
