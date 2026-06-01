"""ShinkaEvolve evaluator adapter for circle_packing_shinka / unit_square_26.

See ``experiments/awtf2025_heuristic/_baselines/shinka/evaluate.py`` for
the full description of why this file needs an ``if __name__ ==
'__main__':`` block and a ``sys.path`` adjustment. The CLI behaviour
lives in ``src/baselines/shinka/_evaluator.run_cli`` so both adapters
stay tiny and identical.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ``experiments/circle_packing_shinka/_baselines/shinka/evaluate.py`` →
# parents[4] = project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.shinka._evaluator import (  # noqa: E402
    evaluate_with_host_experiment,
    run_cli,
)


def main(program_path: str, results_dir: str) -> dict[str, Any]:
    return evaluate_with_host_experiment(
        program_path=program_path, results_dir=results_dir
    )


if __name__ == "__main__":
    raise SystemExit(run_cli())
