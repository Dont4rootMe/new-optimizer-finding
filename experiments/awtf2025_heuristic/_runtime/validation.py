"""Validation and scoring helpers for the awtf2025 heuristic task."""

from __future__ import annotations

import json
import signal
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass(slots=True)
class ContestInput:
    n: int
    k: int
    starts: list[tuple[int, int]]
    targets: list[tuple[int, int]]
    wall_v: list[list[bool]]
    wall_h: list[list[bool]]


@dataclass(slots=True)
class ContestOutput:
    wall_v: list[list[bool]]
    wall_h: list[list[bool]]
    groups: list[int]
    operations: list[tuple[str, int, str]]


class CaseTimeoutError(TimeoutError):
    """Raised when one candidate exceeds the configured soft case timeout."""


def parse_input(input_text: str) -> ContestInput:
    """Parse one official awtf2025 input instance."""

    lines = [line.strip() for line in str(input_text).splitlines() if line.strip()]
    if not lines:
        raise ValueError("Input text must not be empty.")

    try:
        n, k = map(int, lines[0].split())
    except ValueError as exc:
        raise ValueError(f"Invalid first input line: {lines[0]!r}") from exc

    expected_lines = 1 + k + n + (n - 1)
    if len(lines) != expected_lines:
        raise ValueError(f"Expected {expected_lines} non-empty input lines, got {len(lines)}.")

    starts: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    for idx in range(k):
        try:
            sx, sy, tx, ty = map(int, lines[1 + idx].split())
        except ValueError as exc:
            raise ValueError(f"Invalid robot line {idx}: {lines[1 + idx]!r}") from exc
        starts.append((sx, sy))
        targets.append((tx, ty))

    wall_v = [_parse_binary_row(lines[1 + k + idx], n - 1, f"wall_v[{idx}]") for idx in range(n)]
    wall_h = [_parse_binary_row(lines[1 + k + n + idx], n, f"wall_h[{idx}]") for idx in range(n - 1)]
    return ContestInput(n=n, k=k, starts=starts, targets=targets, wall_v=wall_v, wall_h=wall_h)


def _parse_binary_row(text: str, expected_len: int, label: str) -> list[bool]:
    if len(text) != expected_len:
        raise ValueError(f"{label} must have length {expected_len}, got {len(text)}.")
    if any(ch not in {"0", "1"} for ch in text):
        raise ValueError(f"{label} must contain only '0' and '1'.")
    return [ch == "1" for ch in text]


def parse_output(contest_input: ContestInput, output_text: str) -> ContestOutput:
    """Parse one contestant output using the official wall-merging rules."""

    tokens = str(output_text).split()
    cursor = 0

    wall_v = deepcopy(contest_input.wall_v)
    wall_h = deepcopy(contest_input.wall_h)

    for row_idx in range(contest_input.n):
        token, cursor = _read_token(tokens, cursor)
        if len(token) != contest_input.n - 1:
            raise ValueError("Illegal output format for vertical walls.")
        for col_idx, ch in enumerate(token):
            if ch not in {"0", "1"}:
                raise ValueError(f"Invalid character in vertical walls: {ch!r}")
            wall_v[row_idx][col_idx] = wall_v[row_idx][col_idx] or ch == "1"

    for row_idx in range(contest_input.n - 1):
        token, cursor = _read_token(tokens, cursor)
        if len(token) != contest_input.n:
            raise ValueError("Illegal output format for horizontal walls.")
        for col_idx, ch in enumerate(token):
            if ch not in {"0", "1"}:
                raise ValueError(f"Invalid character in horizontal walls: {ch!r}")
            wall_h[row_idx][col_idx] = wall_h[row_idx][col_idx] or ch == "1"

    groups: list[int] = []
    for robot_idx in range(contest_input.k):
        token, cursor = _read_token(tokens, cursor)
        try:
            group_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid group id for robot {robot_idx}: {token!r}") from exc
        if not 0 <= group_id < contest_input.k:
            raise ValueError(f"Group id out of range for robot {robot_idx}: {group_id}")
        groups.append(group_id)

    operations: list[tuple[str, int, str]] = []
    move_cap = contest_input.k * contest_input.n * contest_input.n
    while cursor < len(tokens):
        op_type, cursor = _read_token(tokens, cursor)
        if op_type not in {"g", "i"}:
            raise ValueError(f"Invalid operation type: {op_type!r}")
        target_token, cursor = _read_token(tokens, cursor)
        try:
            target_id = int(target_token)
        except ValueError as exc:
            raise ValueError(f"Invalid operation target: {target_token!r}") from exc
        if not 0 <= target_id < contest_input.k:
            raise ValueError(f"Operation target out of range: {target_id}")
        direction, cursor = _read_token(tokens, cursor)
        if direction not in _MOVE_DELTAS:
            raise ValueError(f"Invalid direction: {direction!r}")
        operations.append((op_type, target_id, direction))
        if len(operations) > move_cap:
            raise ValueError("Too many moves.")

    return ContestOutput(wall_v=wall_v, wall_h=wall_h, groups=groups, operations=operations)


def _read_token(tokens: list[str], cursor: int) -> tuple[str, int]:
    if cursor >= len(tokens):
        raise ValueError("Unexpected EOF in contestant output.")
    return tokens[cursor], cursor + 1


def compute_absolute_score(contest_input: ContestInput, contest_output: ContestOutput) -> tuple[int, list[tuple[int, int]]]:
    """Simulate the official move rules and return `(absolute_score, final_positions)`."""

    positions = list(contest_input.starts)
    occupied = [[False for _ in range(contest_input.n)] for _ in range(contest_input.n)]
    for x_coord, y_coord in positions:
        occupied[x_coord][y_coord] = True

    grouped: list[list[int]] = [[] for _ in range(contest_input.k)]
    for robot_idx, group_id in enumerate(contest_output.groups):
        grouped[group_id].append(robot_idx)

    for op_type, target_id, direction in contest_output.operations:
        if op_type == "g":
            robots = list(grouped[target_id])
        else:
            robots = [target_id]
        robots.sort(key=lambda robot_idx: _movement_order_key(positions[robot_idx], direction))
        dx, dy = _MOVE_DELTAS[direction]
        for robot_idx in robots:
            x_coord, y_coord = positions[robot_idx]
            next_x = x_coord + dx
            next_y = y_coord + dy
            if not (0 <= next_x < contest_input.n and 0 <= next_y < contest_input.n):
                continue
            if occupied[next_x][next_y]:
                continue
            if dx == 0:
                if contest_output.wall_v[x_coord][min(y_coord, next_y)]:
                    continue
            else:
                if contest_output.wall_h[min(x_coord, next_x)][y_coord]:
                    continue
            positions[robot_idx] = (next_x, next_y)
            occupied[x_coord][y_coord] = False
            occupied[next_x][next_y] = True

    score = len(contest_output.operations)
    for robot_idx, (x_coord, y_coord) in enumerate(positions):
        target_x, target_y = contest_input.targets[robot_idx]
        score += 100 * (abs(x_coord - target_x) + abs(y_coord - target_y))
    return score, positions


def _movement_order_key(position: tuple[int, int], direction: str) -> int:
    x_coord, y_coord = position
    if direction == "U":
        return x_coord
    if direction == "D":
        return -x_coord
    if direction == "L":
        return y_coord
    return -y_coord


@contextmanager
def soft_timeout(seconds: float | None):
    """Raise `CaseTimeoutError` if the block exceeds the given timeout."""

    if seconds is None or seconds <= 0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        start = time.perf_counter()
        yield
        if time.perf_counter() - start > seconds:
            raise CaseTimeoutError(f"Candidate exceeded soft timeout of {seconds:.3f} seconds.")
        return
    if not hasattr(signal, "setitimer"):
        start = time.perf_counter()
        yield
        if time.perf_counter() - start > seconds:
            raise CaseTimeoutError(f"Candidate exceeded soft timeout of {seconds:.3f} seconds.")
        return

    def _handle_timeout(_signum: int, _frame: Any) -> None:
        raise CaseTimeoutError(f"Candidate exceeded soft timeout of {seconds:.3f} seconds.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_case_paths(corpus_dir: str | Path, case_ids: list[int]) -> list[tuple[str, Path]]:
    """Resolve configured case ids to zero-padded input paths."""

    root = Path(corpus_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Configured corpus_dir was not found: {root}")

    resolved: list[tuple[str, Path]] = []
    for case_id in case_ids:
        case_name = f"{int(case_id):04d}"
        case_path = root / f"{case_name}.txt"
        if not case_path.exists():
            raise FileNotFoundError(f"Configured case was not found: {case_path}")
        resolved.append((case_name, case_path))
    return resolved


def evaluate_case(
    *,
    case_name: str,
    case_path: Path,
    solve_case,
    per_case_soft_time_limit_sec: float | None,
) -> dict[str, Any]:
    """Run one candidate on one corpus case and return structured metrics."""

    input_text = case_path.read_text(encoding="utf-8")
    contest_input = parse_input(input_text)
    input_header = input_text.splitlines()[0].strip() if input_text.splitlines() else ""
    started_at = time.perf_counter()
    try:
        with soft_timeout(per_case_soft_time_limit_sec):
            output_text = solve_case(input_text)
    except Exception as exc:
        raise RuntimeError(
            f"solve_case failed on case {case_name} for input header {input_header!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    elapsed_sec = time.perf_counter() - started_at
    if not isinstance(output_text, str):
        raise TypeError(f"solve_case(...) must return str, got {type(output_text).__name__}.")

    try:
        contest_output = parse_output(contest_input, output_text)
    except Exception as exc:
        raise ValueError(f"Illegal output on case {case_name}: {exc}") from exc
    try:
        absolute_score, final_positions = compute_absolute_score(contest_input, contest_output)
    except Exception as exc:
        raise RuntimeError(f"Simulation failed on case {case_name}: {type(exc).__name__}: {exc}") from exc
    return {
        "case_id": case_name,
        "absolute_score": int(absolute_score),
        "elapsed_sec": float(elapsed_sec),
        "num_operations": int(len(contest_output.operations)),
        "final_positions": [list(position) for position in final_positions],
    }


def save_extra_artifact(
    *,
    organism_dir: str,
    experiment_name: str,
    payload: dict[str, Any],
) -> str:
    """Persist case-level diagnostics into an opaque JSON artifact."""

    out_path = Path(organism_dir) / "results" / f"{experiment_name}_extra.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out_path.resolve())
