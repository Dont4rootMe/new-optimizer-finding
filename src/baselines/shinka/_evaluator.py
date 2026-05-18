"""Shared helpers for ShinkaEvolve evaluator adapters."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT_ENV = "SHINKA_BASELINE_PROJECT_ROOT"
CONFIG_NAME_ENV = "SHINKA_BASELINE_CONFIG"


def _project_root() -> Path:
    value = os.environ.get(PROJECT_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"Environment variable {PROJECT_ROOT_ENV} is not set; the ShinkaEvolve "
            "evaluator adapter expects it to be exported by run_shinka_baseline.sh."
        )
    return Path(value).expanduser().resolve()


def _config_name() -> str:
    value = os.environ.get(CONFIG_NAME_ENV)
    if not value:
        raise RuntimeError(
            f"Environment variable {CONFIG_NAME_ENV} is not set; the ShinkaEvolve "
            "evaluator adapter expects it to be exported by run_shinka_baseline.sh."
        )
    return value


def _compose_cfg() -> DictConfig:
    root = _project_root()
    with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
        cfg = compose(config_name=_config_name())
    return cfg


def _select_experiment_cfg(cfg: DictConfig) -> DictConfig:
    experiments = cfg.get("experiments")
    if experiments is None:
        raise RuntimeError("Composed config has no `experiments` block.")
    keys = list(experiments.keys())
    if len(keys) != 1:
        raise RuntimeError(
            f"ShinkaEvolve baseline expects exactly one experiment under `experiments:`, "
            f"got {len(keys)}: {keys}."
        )
    return experiments[keys[0]]


def _stage_organism_dir(program_path: Path) -> tuple[Path, Path | None]:
    """Place program_path as `<organism_dir>/implementation.py`.

    Returns (organism_dir, cleanup_dir). cleanup_dir is non-None only when we
    created a tmp dir, so the caller can remove it on exit.
    """

    program_path = program_path.expanduser().resolve()
    if program_path.name == "implementation.py":
        return program_path.parent, None
    tmp_dir = Path(tempfile.mkdtemp(prefix="shinka_eval_"))
    shutil.copy2(program_path, tmp_dir / "implementation.py")
    return tmp_dir, tmp_dir


def _normalize_combined_score(result: dict[str, Any]) -> float:
    score = result.get("score")
    if score is None:
        raise RuntimeError(f"Experiment evaluator returned no 'score' field: keys={list(result)}")
    return float(score)


def evaluate_with_host_experiment(program_path: str, results_dir: str) -> dict[str, Any]:
    """Run the host experiment evaluator on ``program_path`` and emit shinka files.

    ShinkaEvolve does NOT consume the return value of this function — it
    reads two files written to ``results_dir`` after the evaluator subprocess
    exits:

      * ``results_dir/correct.json``  → ``{"correct": bool, "error": str|None}``
      * ``results_dir/metrics.json``  → ``{"combined_score": float, ...}``

    (See ``shinka.utils.general.load_results``.) Until this adapter writes
    those files explicitly the runner falls back to its defaults
    (``correct=False, score=0.0``), which is exactly the symptom of the
    first 75-gen ShinkaEvolve run: every program — including a known-good
    seed scoring 21 318 locally — was logged as ``correct=False score=0.0``
    and the database showed 0/78 correct programs.

    The fix is to call :func:`shinka.core.wrap_eval.save_json_results`
    after the host evaluator returns. ``correct=True`` whenever the host
    evaluator completes without raising; ``correct=False`` with a
    traceback string on any exception. The returned dict is kept (and
    still includes the ``combined_score`` + ``public`` view used by tests
    and ad-hoc CLI invocations).
    """

    from shinka.core.wrap_eval import save_json_results

    results_dir_resolved = str(Path(results_dir).expanduser().resolve())
    metrics: dict[str, Any] = {"combined_score": 0.0}
    public: dict[str, Any] = {}
    correct = False
    error_msg: str | None = None
    result: dict[str, Any] | None = None

    organism_dir, cleanup_dir = _stage_organism_dir(Path(program_path))
    try:
        try:
            cfg = _compose_cfg()
            experiment_cfg = _select_experiment_cfg(cfg)
            experiment = instantiate(experiment_cfg, _recursive_=False)
            result = experiment.evaluate_organism(
                organism_dir=str(organism_dir), cfg=experiment_cfg
            )
        except BaseException as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    if result is not None:
        try:
            combined_score = _normalize_combined_score(result)
        except Exception as exc:  # noqa: BLE001
            combined_score = 0.0
            correct = False
            error_msg = (error_msg or "") + f"\nscore-extraction failed: {exc}"
        else:
            correct = True
            metrics["combined_score"] = combined_score
            public = {
                "objective_name": result.get("objective_name"),
                "objective_direction": result.get("objective_direction"),
                "objective_last": result.get("objective_last"),
                "status": result.get("status"),
            }
            if "num_cases" in result:
                public["num_cases"] = result["num_cases"]
                public["mean_absolute_score"] = result.get("mean_absolute_score")
                public["total_absolute_score"] = result.get("total_absolute_score")
            if "num_circles" in result:
                public["num_circles"] = result["num_circles"]
                public["reported_sum_of_radii"] = result.get("reported_sum_of_radii")
            metrics["public"] = public

    # Write the two files the ShinkaEvolve runner actually reads. This
    # is the load-bearing line that flips ``correct=False score=0.0`` to
    # the real outcome.
    try:
        save_json_results(
            results_dir_resolved,
            metrics,
            correct=correct,
            error=error_msg,
        )
    except Exception:  # noqa: BLE001
        # Last-ditch: if shinka isn't importable in the subprocess for
        # any reason, still write the files by hand so the runner can
        # see the result.
        os.makedirs(results_dir_resolved, exist_ok=True)
        with open(os.path.join(results_dir_resolved, "correct.json"), "w") as fh:
            json.dump({"correct": correct, "error": error_msg}, fh, indent=4)
        with open(os.path.join(results_dir_resolved, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=4)

    return {
        "combined_score": metrics["combined_score"],
        "public": public,
        "private": {"results_dir": results_dir_resolved},
        "correct": correct,
        "error": error_msg,
    }


def dump_cfg_for_debug(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg)


def run_cli() -> int:
    """CLI entrypoint invoked by the ShinkaEvolve scheduler subprocess.

    ShinkaEvolve 0.0.6 (``shinka.launch.scheduler.JobScheduler._build_command``)
    launches every evaluator job as:

        python <eval_program_path> --program_path <main.py> --results_dir <gen_X/results>

    It then reads ``<results_dir>/correct.json`` and
    ``<results_dir>/metrics.json`` after the subprocess exits (see
    ``shinka.utils.general.load_results``). The per-experiment
    ``evaluate.py`` files therefore must do two things on every
    invocation: (1) parse those two CLI flags and (2) actually call
    :func:`evaluate_with_host_experiment`, which is the one that writes
    the result files. Stashing that wiring here keeps the
    ``evaluate.py`` adapters tiny and identical across experiments.
    """

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="ShinkaEvolve evaluator subprocess for awtf2025/circle_packing baselines."
    )
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args, _unknown = parser.parse_known_args()
    try:
        evaluate_with_host_experiment(
            program_path=args.program_path,
            results_dir=args.results_dir,
        )
    except BaseException as exc:  # noqa: BLE001
        # ``evaluate_with_host_experiment`` already writes ``correct=False``
        # + the traceback to ``correct.json`` on any internal failure, but
        # if even *that* helper raises (e.g. shinka unimportable), surface
        # the traceback on stderr so the launcher's job_log.err is non-empty
        # and the failure is visible in post-mortem logs.
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0
