"""ShinkaEvolve evaluator adapter for awtf2025_heuristic / group_commands_and_wall_planning.

ShinkaEvolve launches this file as a subprocess:

    python <this_file> --program_path <gen_X/main.py> --results_dir <gen_X/results>

Python's default subprocess behaviour only puts the script's parent
directory on ``sys.path``, not the project root, so importing
``src.baselines.shinka._evaluator`` would otherwise fail. The
``sys.path`` fix below makes the import resolve. Everything else is
delegated to the shared CLI entrypoint in ``_evaluator.py`` so the
two ``evaluate.py`` files (awtf2025 + circle_packing) stay tiny and
identical.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ``experiments/awtf2025_heuristic/_baselines/shinka/evaluate.py`` →
# parents[4] = project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.shinka._evaluator import (  # noqa: E402
    evaluate_with_host_experiment,
    run_cli,
)


def main(program_path: str, results_dir: str) -> dict[str, Any]:
    """Direct-call entrypoint kept for tests and ad-hoc CLI invocations."""

    return evaluate_with_host_experiment(
        program_path=program_path, results_dir=results_dir
    )


if __name__ == "__main__":
    raise SystemExit(run_cli())
