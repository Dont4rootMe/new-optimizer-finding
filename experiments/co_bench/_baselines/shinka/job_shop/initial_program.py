"""Minimal seed program for the ShinkaEvolve baseline on co_bench / job_shop.

The ShinkaEvolve runner edits only the region between EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END; the rest of the file stays fixed.

The CO-Bench Job shop scheduling evaluator calls ``solve(**instance)`` where
each instance provides ``n_jobs``, ``n_machines``, ``times`` (n_jobs x
n_machines processing times) and ``machines`` (n_jobs x n_machines machine ids,
1-indexed), and expects a dict ``{"start_times": [[...]]}`` of non-negative
integer start times (n_jobs x n_machines). A fully serial schedule, in which no
two operations ever overlap, satisfies both the job-sequential and
machine-non-overlap constraints and scores > 0.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def solve(n_jobs, n_machines, times, machines, **kwargs):
    # Trivial valid baseline: schedule every operation strictly serially.
    start_times = [[0] * n_machines for _ in range(n_jobs)]
    clock = 0
    for i in range(n_jobs):
        for j in range(n_machines):
            start_times[i][j] = clock
            clock += times[i][j]
    return {"start_times": start_times}
# EVOLVE-BLOCK-END
