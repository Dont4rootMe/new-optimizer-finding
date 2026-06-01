"""Task-blind CO-Bench experiment evaluator.

Hydra instantiates :class:`CoBenchExperimentEvaluator` via ``_target_``. The
host runner then calls ``evaluate_organism(organism_dir, cfg)`` and expects a
report dict with a ``score`` key (higher = better). The candidate lives at
``<organism_dir>/implementation.py`` and must define ``solve(**kwargs) -> dict``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from experiments.co_bench._runtime.candidate_loader import load_solve
from experiments.co_bench._runtime.cobench_bridge import import_cobench
from experiments.co_bench._runtime.errors import OptionalDependencyError

# Cap persisted feedback strings so reports stay small.
_FEEDBACK_CHAR_CAP = 4096


def _truncate(text: Any, cap: int = _FEEDBACK_CHAR_CAP) -> str:
    """Coerce to ``str`` and truncate to ``cap`` characters."""

    if text is None:
        return ""
    string = str(text)
    if len(string) <= cap:
        return string
    return string[:cap] + f"... [truncated {len(string) - cap} chars]"


class CoBenchExperimentEvaluator:
    """Evaluate a ``solve(**kwargs)`` organism against one CO-Bench task."""

    def __init__(self, co_bench_task: str, **_: Any) -> None:
        self.co_bench_task = str(co_bench_task)

    def evaluate_organism(self, organism_dir: str | None, cfg: DictConfig) -> dict[str, Any]:
        # 1. Resolve the organism dir (fallback to cfg.runtime.organism_dir).
        runtime_cfg = cfg.get("runtime")
        resolved_organism_dir = organism_dir
        if not resolved_organism_dir and runtime_cfg is not None:
            resolved_organism_dir = runtime_cfg.get("organism_dir")
        if not resolved_organism_dir:
            raise ValueError(
                "CO-Bench evaluation requires organism_dir pointing to implementation.py."
            )

        organism_path = Path(str(resolved_organism_dir)).expanduser().resolve()
        implementation_path = organism_path / "implementation.py"

        # 2. Fast, clear-error pre-validation of the solve(**kwargs) contract.
        _solve, module_path = load_solve(str(implementation_path))
        code_str = implementation_path.read_text(encoding="utf-8")

        # 3. Bridge to CO-Bench (raises OptionalDependencyError if unavailable).
        evaluation = import_cobench()

        # 4. Resolve src_dir to an ABSOLUTE path. CO-Bench re-imports eval_func
        #    from data.config_path inside spawned subprocesses, so a relative
        #    src_dir would break once the worker changes directory.
        data_cfg = cfg.get("data")
        configured_src_dir = None
        if data_cfg is not None:
            configured_src_dir = data_cfg.get("cobench_src_dir")
        if not configured_src_dir:
            configured_src_dir = os.path.join(str(cfg.paths.data_root), "co-bench")
        abs_src_dir = os.path.abspath(str(configured_src_dir))

        # 5. Load the task. Missing task dir / config.py -> optional-dependency.
        task_config = Path(abs_src_dir) / self.co_bench_task / "config.py"
        if not task_config.is_file():
            raise OptionalDependencyError(
                "co_bench",
                f"CO-Bench task data for '{self.co_bench_task}' not found at "
                f"'{task_config}'. Download the dataset (e.g. run "
                "scripts/bootstrap_cobench.sh, which uses huggingface_hub to "
                "fetch it into ${AIFS_DATA_ROOT:-./data}/co-bench), or set "
                "data.cobench_src_dir to a directory containing the per-task "
                "folders.",
            )
        data = evaluation.get_data(self.co_bench_task, src_dir=abs_src_dir)

        # 6. Evaluate the candidate code string.
        evaluator = evaluation.Evaluator(
            data,
            timeout=int(cfg.get("timeout", 10)),
            cpu_num=cfg.get("cpu_num", None),
            feedback_length=int(cfg.get("feedback_length", 64)),
        )
        feedback = evaluator.evaluate(code_str)

        # 7. Map CO-Bench feedback to the host report shape.
        dev_score = feedback.dev_score
        dev_value = None if dev_score is None else float(dev_score)
        if dev_value is None or math.isnan(dev_value):
            status = "failed"
            score = None
        else:
            status = "ok"
            score = dev_value

        test_score = feedback.test_score
        test_value = None if test_score is None else float(test_score)

        return {
            "status": status,
            "score": score,
            "objective_name": "cobench_dev_score",
            "objective_direction": "max",
            "objective_last": score,
            "test_score": test_value,
            "co_bench_task": self.co_bench_task,
            "dev_feedback": _truncate(feedback.dev_feedback),
            "test_feedback": _truncate(feedback.test_feedback),
            "candidate_module_path": str(module_path),
        }
