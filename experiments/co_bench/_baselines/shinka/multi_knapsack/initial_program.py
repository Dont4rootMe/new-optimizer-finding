"""Minimal seed program for the ShinkaEvolve baseline on co_bench / multi_knapsack.

The ShinkaEvolve runner edits only the region between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END; the rest of the file stays fixed.

The CO-Bench Multidimensional knapsack evaluator calls ``solve(**instance)``
where each instance provides ``n`` (number of decision variables), ``m``
(number of constraints), ``p`` (profit coefficients, length n), ``r`` (m lists
of length n giving per-constraint resource use), and ``b`` (capacities, length
m), and expects a dict ``{"x": [0/1, ...]}`` of length n. The greedy
profit-density baseline below only adds an item if it fits ALL constraints, so
it is always feasible and scores > 0.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def solve(n, m, p, r, b, **kwargs):
    # Greedy-by-profit-density feasible baseline.
    x = [0] * n
    cap = [float(bi) for bi in b]
    order = sorted(range(n), key=lambda j: p[j] / (1.0 + sum(r[i][j] for i in range(m))), reverse=True)
    for j in order:
        if all(r[i][j] <= cap[i] for i in range(m)):
            x[j] = 1
            for i in range(m):
                cap[i] -= r[i][j]
    return {"x": x}
# EVOLVE-BLOCK-END
