"""Working seed for the ShinkaEvolve baseline on awtf2025_heuristic.

ShinkaEvolve only edits the region between ``EVOLVE-BLOCK-START`` and
``EVOLVE-BLOCK-END``. Everything outside the block is immutable, so the
``solve_case`` signature stays fixed and the import surface the host
evaluator (``GroupCommandsAndWallPlanningExperiment``) relies on is
preserved.

The seed implements a simple but valid baseline: each robot becomes its
own group, no extra walls are added, and the program emits axis-aligned
greedy individual moves (``i <robot_idx> <U|D|L|R>``) that step every
robot toward its target while respecting the input walls and current
occupancy. The output therefore parses cleanly through
``experiments/awtf2025_heuristic/_runtime/validation.py`` and produces a
non-zero score on every case — giving the ShinkaEvolve search a real
gradient to climb from instead of an empty string the LLM has to
reverse-engineer.

This file lives under ``_baselines/shinka/`` and is intentionally not
shared with the main evolutionary loop.
"""

from __future__ import annotations


# EVOLVE-BLOCK-START
def solve_case(input_text: str) -> str:
    """Return one valid contest output for the awtf2025 grid-robot task.

    Output layout (whitespace-separated tokens, official format):

    1. ``n`` rows of vertical-wall additions, each ``n-1`` chars of ``0/1``.
    2. ``n-1`` rows of horizontal-wall additions, each ``n`` chars of
       ``0/1``.
    3. ``k`` group ids, one per robot, each in ``[0, k)``.
    4. Zero or more operations, each three tokens
       ``<op_type> <target_id> <direction>``. ``op_type`` is ``i`` for an
       individual move or ``g`` for a group move; ``direction`` is one of
       ``U D L R``. Total operation count is capped at ``k * n * n``.

    Score (smaller is better):
        ``len(operations) + 100 * sum(Manhattan(final, target))``

    Strategy implemented here:

    * Add no walls.
    * Place each robot in its own group (group ``i`` = robot ``i``).
    * Repeatedly walk every still-misplaced robot one step along the
      axis with the larger remaining gap, falling back to the other
      axis when the first is blocked. Movement is simulated against
      walls + occupancy, mirroring the evaluator's rules.
    * Stop when no robot can move (deadlock) or when the operation
      budget is exhausted.
    """

    lines = [line.strip() for line in input_text.splitlines() if line.strip()]
    if not lines:
        return ""

    n, k = map(int, lines[0].split())

    starts: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    for robot_idx in range(k):
        sx, sy, tx, ty = map(int, lines[1 + robot_idx].split())
        starts.append((sx, sy))
        targets.append((tx, ty))

    # Input walls — used to gate moves during simulation.
    wall_v_in = [lines[1 + k + row] for row in range(n)]
    wall_h_in = [lines[1 + k + n + row] for row in range(n - 1)]

    # Output walls — keep zeros so we add no walls of our own (the
    # evaluator OR-merges these with the input walls).
    out_walls_v = ["0" * (n - 1) for _ in range(n)]
    out_walls_h = ["0" * n for _ in range(n - 1)]

    groups = list(range(k))  # robot i -> group i

    DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

    positions = list(starts)
    occupied = [[False] * n for _ in range(n)]
    for (x_coord, y_coord) in positions:
        occupied[x_coord][y_coord] = True

    def can_move(x_coord: int, y_coord: int, direction: str) -> bool:
        delta_x, delta_y = DELTAS[direction]
        next_x, next_y = x_coord + delta_x, y_coord + delta_y
        if not (0 <= next_x < n and 0 <= next_y < n):
            return False
        if occupied[next_x][next_y]:
            return False
        if delta_x == 0:
            wall_col = min(y_coord, next_y)
            if wall_v_in[x_coord][wall_col] == "1":
                return False
        else:
            wall_row = min(x_coord, next_x)
            if wall_h_in[wall_row][y_coord] == "1":
                return False
        return True

    operations: list[tuple[str, int, str]] = []
    budget = k * n * n

    progress = True
    while progress and len(operations) < budget:
        progress = False
        for robot_idx in range(k):
            if len(operations) >= budget:
                break
            x_coord, y_coord = positions[robot_idx]
            target_x, target_y = targets[robot_idx]
            if (x_coord, y_coord) == (target_x, target_y):
                continue

            preferred: list[str] = []
            if abs(target_x - x_coord) >= abs(target_y - y_coord):
                if target_x > x_coord:
                    preferred.append("D")
                elif target_x < x_coord:
                    preferred.append("U")
                if target_y > y_coord:
                    preferred.append("R")
                elif target_y < y_coord:
                    preferred.append("L")
            else:
                if target_y > y_coord:
                    preferred.append("R")
                elif target_y < y_coord:
                    preferred.append("L")
                if target_x > x_coord:
                    preferred.append("D")
                elif target_x < x_coord:
                    preferred.append("U")

            for direction in preferred:
                if can_move(x_coord, y_coord, direction):
                    delta_x, delta_y = DELTAS[direction]
                    next_x, next_y = x_coord + delta_x, y_coord + delta_y
                    occupied[x_coord][y_coord] = False
                    occupied[next_x][next_y] = True
                    positions[robot_idx] = (next_x, next_y)
                    operations.append(("i", robot_idx, direction))
                    progress = True
                    break

    tokens: list[str] = []
    tokens.extend(out_walls_v)
    tokens.extend(out_walls_h)
    tokens.extend(str(group_id) for group_id in groups)
    for op_type, target_id, direction in operations:
        tokens.extend((op_type, str(target_id), direction))
    return "\n".join(tokens)
# EVOLVE-BLOCK-END
