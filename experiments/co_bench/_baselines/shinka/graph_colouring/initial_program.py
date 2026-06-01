"""Minimal seed program for the ShinkaEvolve baseline on co_bench / graph_colouring.

The ShinkaEvolve runner edits only the region between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END; the rest of the file stays fixed.

The CO-Bench Graph colouring evaluator calls ``solve(**instance)`` where each
instance provides ``n`` (the number of vertices), ``edges`` (a list of (u, v)
tuples), and ``adjacency`` (a dict mapping each vertex 1..n to the set of its
neighbours). ``solve`` must return a FLAT dict mapping every vertex_id (1..n)
to a positive-integer colour — i.e. ``{vertex: colour, ...}`` directly, NOT
wrapped under any key such as ``{"colors": ...}``. Assigning every vertex its
own distinct colour (``{v: v}``) yields zero conflicts, so it is a valid
colouring and scores > 0 (though it uses the maximum number of colours).
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def solve(n, **kwargs):
    # Trivial valid baseline: give every vertex its own distinct colour (no conflicts).
    return {v: v for v in range(1, n + 1)}
# EVOLVE-BLOCK-END
