"""CO-Bench combinatorial-optimization experiment family.

Task-blind organism evolution against the CO-Bench benchmark
(https://github.com/sunnweiwei/CO-Bench). Each organism is a single
``solve(**kwargs) -> dict`` candidate scored by CO-Bench's per-task
evaluator. See ``AGENTS.md`` for the contract and how to add a task.
"""
