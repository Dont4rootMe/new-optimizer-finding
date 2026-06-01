"""Adapter-contract tests for the CO-Bench experiment family.

CO-Bench is driven by ONE top-level preset, ``conf/config_co-bench.yaml``. The
active task is chosen by the field ``experiments.co_bench.CO_BENCH_TASK`` (an
UPPER-CASE identifier), not by per-task presets. These tests therefore always
compose ``config_co-bench`` with ``overrides=["experiments.co_bench.CO_BENCH_TASK=<ID>"]``.

They avoid any heavy optional dependency (torch/numpy/matplotlib) and exercise
the lightweight contract surfaces:

* the ``mode=smoke`` optional-dependency skip path (no CO-Bench checkout), and
* the ``solve(**kwargs) -> dict`` candidate loader rules.

A final, dataset-coupled test asserts each selected ``co_bench_task`` string is
a real CO-Bench dataset task, but only when the checkout and data are present;
otherwise it skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]

# UPPER-CASE CO_BENCH_TASK identifiers accepted by config_co-bench, mapped to the
# lowercase slug that drives the seed/prompt/output paths.
CO_BENCH_IDENTIFIERS = (
    "TSP",
    "BIN_PACKING_1D",
    "MULTI_KNAPSACK",
    "SET_COVERING",
    "GRAPH_COLOURING",
    "JOB_SHOP",
)


def _write_module(tmp_path: Path, source: str) -> str:
    """Write ``source`` to ``tmp_path/implementation.py`` and return its path."""

    module_path = tmp_path / "implementation.py"
    module_path.write_text(source, encoding="utf-8")
    return str(module_path)


def test_cobench_smoke_skips_without_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``mode=smoke`` with a missing CO-Bench checkout -> ``status="skipped"``.

    The adapter raises ``OptionalDependencyError`` from ``import_cobench`` when
    ``COBENCH_ROOT`` points at a nonexistent directory, and the host runner
    turns that into a skipped report in smoke mode.
    """

    # Trivial but contract-valid candidate so load_solve() passes and execution
    # reaches the CO-Bench bridge (which is where the skip is raised).
    organism_dir = tmp_path / "organism"
    organism_dir.mkdir()
    _write_module(organism_dir, "def solve(**kwargs):\n    return {}\n")

    # Force the optional-dependency path: a bad checkout root, and reset the
    # bridge's cached ``evaluation`` module so a prior successful import in this
    # process cannot short-circuit the env check.
    monkeypatch.setenv("COBENCH_ROOT", "/nonexistent/path")
    import experiments.co_bench._runtime.cobench_bridge as bridge

    monkeypatch.setattr(bridge, "_EVALUATION_MODULE", None)

    runs_root = tmp_path / "runs"
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_co-bench",
            overrides=[
                "mode=smoke",
                f"+organism_dir={organism_dir}",
                f"paths.runs_root={runs_root}",
            ],
        )

    from src.validate.runner import ExperimentRunner

    results = ExperimentRunner(cfg).run()

    assert len(results) == 1
    # The runner iterates cfg.experiments by KEY; the static key is ``co_bench``
    # (the slug ``tsp`` is the task's name field, not the experiment key).
    assert results[0]["experiment_name"] == "co_bench"
    assert results[0]["status"] == "skipped"
    assert results[0]["score"] is None


def test_load_solve_accepts_and_rejects(tmp_path: Path) -> None:
    """``load_solve`` enforces the CO-Bench ``solve(**kwargs)`` contract.

    Faithful to ``candidate_loader.load_solve``:
      * ACCEPT a ``**kwargs`` VAR_KEYWORD parameter (CO-Bench calls
        ``solve(**instance)``);
      * ACCEPT a required positional-or-keyword param alongside ``**kwargs``
        (it is filled from the instance by keyword);
      * REJECT a module with no callable ``solve`` (AttributeError);
      * REJECT a ``solve`` with no ``**kwargs`` (TypeError) — instance keys
        could not be tolerated;
      * REJECT a required positional-only param (TypeError) — it can never be
        filled by keyword.
    """

    from experiments.co_bench._runtime.candidate_loader import load_solve

    # --- ACCEPT cases -------------------------------------------------------
    accept_kwargs = tmp_path / "accept_kwargs"
    accept_kwargs.mkdir()
    solve, module_path = load_solve(
        _write_module(accept_kwargs, "def solve(**kwargs):\n    return {}\n")
    )
    assert callable(solve)
    assert module_path == (accept_kwargs / "implementation.py").resolve()

    accept_required = tmp_path / "accept_required"
    accept_required.mkdir()
    solve, _ = load_solve(
        _write_module(accept_required, "def solve(nodes, **kwargs):\n    return {}\n")
    )
    assert callable(solve)

    # --- REJECT: no solve defined ------------------------------------------
    no_solve = tmp_path / "no_solve"
    no_solve.mkdir()
    with pytest.raises(AttributeError):
        load_solve(_write_module(no_solve, "x = 1\n"))

    # --- REJECT: solve without **kwargs ------------------------------------
    no_kwargs = tmp_path / "no_kwargs"
    no_kwargs.mkdir()
    with pytest.raises(TypeError):
        load_solve(_write_module(no_kwargs, "def solve(x):\n    return {}\n"))

    # --- REJECT: required positional-only parameter ------------------------
    positional_only = tmp_path / "positional_only"
    positional_only.mkdir()
    with pytest.raises(TypeError):
        load_solve(_write_module(positional_only, "def solve(x, /):\n    return {}\n"))


def _compose_task(identifier: str):
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        return compose(
            config_name="config_co-bench",
            overrides=[f"experiments.co_bench.CO_BENCH_TASK={identifier}"],
        )


@pytest.mark.parametrize("identifier", CO_BENCH_IDENTIFIERS)
def test_cobench_task_strings_match_dataset(identifier: str) -> None:
    """Each selected ``co_bench_task`` is a real CO-Bench dataset task.

    Robust: skips when the CO-Bench checkout or the per-task dataset folders are
    absent. The dataset "task set" is the set of per-task folders that contain a
    ``config.py`` under the resolved ``cobench_src_dir`` (this is exactly the
    layout the adapter validates before evaluating).
    """

    try:
        from experiments.co_bench._runtime.cobench_bridge import import_cobench
        from experiments.co_bench._runtime.errors import OptionalDependencyError
    except Exception:  # pragma: no cover - defensive
        pytest.skip("CO-Bench runtime bridge not importable")

    try:
        import_cobench()
    except OptionalDependencyError:
        pytest.skip("CO-Bench checkout/solver deps not available")

    cfg = _compose_task(identifier)
    exp_cfg = cfg.experiments.co_bench

    data_cfg = exp_cfg.get("data")
    src_dir = None
    if data_cfg is not None:
        src_dir = data_cfg.get("cobench_src_dir")
    if not src_dir:
        src_dir = Path(ROOT) / "data" / "co-bench"
    src_path = Path(str(src_dir))
    if not src_path.is_dir():
        pytest.skip(f"CO-Bench dataset directory not found: {src_path}")

    dataset_tasks = {
        config.parent.name for config in src_path.glob("*/config.py")
    }
    if not dataset_tasks:
        pytest.skip(f"No CO-Bench task folders with config.py under {src_path}")

    task = exp_cfg.co_bench_task
    assert task in dataset_tasks, (
        f"co_bench_task {task!r} for identifier {identifier!r} not in dataset "
        f"task set {sorted(dataset_tasks)}"
    )
