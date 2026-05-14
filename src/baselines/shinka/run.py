"""ShinkaEvolve baseline entry-point.

Reads a Hydra config (`baselines/<experiment>` by convention), pulls the
`shinka_evolve:` block + the project's `evolver.llm.route_weights`/`api_platforms.*`
sections, builds the appropriate ShinkaEvolve dataclasses, and runs the loop.

Invoked from `scripts/run_shinka_baseline.sh` after Ollama instances are up.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.baselines.shinka._evaluator import CONFIG_NAME_ENV, PROJECT_ROOT_ENV
from src.baselines.shinka._models import build_shinka_model_urls


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ShinkaEvolve baseline using a Hydra config.")
    parser.add_argument("--config-name", required=True, help="Hydra config name (e.g. baselines/awtf2025_heuristic).")
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Repository root that contains conf/ (defaults to cwd).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides forwarded to compose().",
    )
    return parser.parse_args(argv)


def _resolve_path(value: str | os.PathLike[str], *, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _import_shinka() -> dict[str, Any]:
    """Import ShinkaEvolve lazily and surface a friendly error if it's missing."""

    try:
        from shinka.core import EvolutionConfig, ShinkaEvolveRunner  # type: ignore
        from shinka.database import DatabaseConfig  # type: ignore
        from shinka.launch import LocalJobConfig  # type: ignore
    except ImportError as err:
        raise SystemExit(
            "ShinkaEvolve is not installed. Install it with:\n"
            "    pip install -e \".[shinka_baseline]\"\n"
            "or follow the upstream guide at https://github.com/SakanaAI/ShinkaEvolve.\n"
            f"(import error: {err})"
        ) from err

    return {
        "EvolutionConfig": EvolutionConfig,
        "ShinkaEvolveRunner": ShinkaEvolveRunner,
        "DatabaseConfig": DatabaseConfig,
        "LocalJobConfig": LocalJobConfig,
    }


def _build_evolution_config(
    *,
    EvolutionConfig: Any,
    shinka_block: DictConfig,
    project_root: Path,
    llm_models: list[str],
    results_dir: Path,
) -> Any:
    init_program_path = _resolve_path(shinka_block.initial_program_path, root=project_root)
    if not init_program_path.exists():
        raise FileNotFoundError(f"initial_program_path does not exist: {init_program_path}")

    extra = shinka_block.get("extra_runner_kwargs") or {}
    extra_dict = OmegaConf.to_container(extra, resolve=True) if isinstance(extra, DictConfig) else dict(extra)

    # ``cost_aware_coef`` was a top-level field on ``EvolutionConfig`` up to
    # shinka-evolve 0.0.5 and moved into ``llm_dynamic_selection_kwargs`` in
    # 0.0.6. Keep accepting the old yaml key for ergonomic backward compat
    # and route it to the new sub-mapping; if the user has an old shinka
    # installed the fallback path below restores the legacy call shape.
    selection_kwargs: dict[str, Any] = {}
    cost_aware_coef = shinka_block.get("cost_aware_coef")
    if cost_aware_coef is not None:
        selection_kwargs["cost_aware_coef"] = float(cost_aware_coef)

    kwargs: dict[str, Any] = {
        "init_program_path": str(init_program_path),
        "num_generations": int(shinka_block.num_generations),
        "results_dir": str(results_dir),
        "llm_models": llm_models,
        "llm_dynamic_selection": str(shinka_block.get("llm_dynamic_selection") or "ucb1"),
    }
    if selection_kwargs:
        kwargs["llm_dynamic_selection_kwargs"] = selection_kwargs
    # ``EvolutionConfig.embedding_model`` defaults to OpenAI's
    # ``text-embedding-3-small``, and shinka 0.0.6's runner refuses to
    # start unless every referenced model's env var is set (it raises
    # "missing OPENAI_API_KEY" up front). The 0.0.6 runner skips the
    # embedding-model env-var check when ``embedding_model`` is falsy,
    # so we always pass through whatever the yaml has — including
    # ``null`` — instead of swallowing the None and letting the upstream
    # OpenAI default leak in.
    embedding_model = shinka_block.get("embedding_model")
    kwargs["embedding_model"] = (
        str(embedding_model) if embedding_model else None
    )
    kwargs.update(extra_dict)

    try:
        return EvolutionConfig(**kwargs)
    except TypeError as exc:
        # Legacy shinka (<=0.0.5) flattens cost_aware_coef back onto the
        # top-level. Detect that specific shape and retry once instead of
        # forcing every user to upgrade in lockstep with our local pin.
        unrecognised = "llm_dynamic_selection_kwargs"
        if (
            unrecognised in str(exc)
            and "llm_dynamic_selection_kwargs" in kwargs
        ):
            legacy_kwargs = dict(kwargs)
            legacy_selection_kwargs = legacy_kwargs.pop(
                "llm_dynamic_selection_kwargs"
            )
            if "cost_aware_coef" in legacy_selection_kwargs:
                legacy_kwargs["cost_aware_coef"] = float(
                    legacy_selection_kwargs["cost_aware_coef"]
                )
            return EvolutionConfig(**legacy_kwargs)
        raise


def _build_local_job_config(
    *,
    LocalJobConfig: Any,
    shinka_block: DictConfig,
    project_root: Path,
) -> Any:
    eval_program_path = _resolve_path(shinka_block.evaluate_program_path, root=project_root)
    if not eval_program_path.exists():
        raise FileNotFoundError(f"evaluate_program_path does not exist: {eval_program_path}")

    # ``max_parallel_jobs`` moved off ``LocalJobConfig`` in shinka-evolve
    # 0.0.6 — it is now ``max_evaluation_jobs`` on ``ShinkaEvolveRunner``.
    # Try the modern signature first; on TypeError fall back to the legacy
    # call shape so anyone still pinned to 0.0.4/0.0.5 keeps working.
    try:
        return LocalJobConfig(eval_program_path=str(eval_program_path))
    except TypeError:
        return LocalJobConfig(
            eval_program_path=str(eval_program_path),
            max_parallel_jobs=int(shinka_block.get("max_parallel_jobs") or 1),
        )


def _build_database_config(
    *,
    DatabaseConfig: Any,
    shinka_block: DictConfig,
    results_dir: Path,
) -> Any:
    return DatabaseConfig(
        db_path=str(results_dir / "shinka.db"),
        num_islands=int(shinka_block.get("num_islands") or 1),
    )


def _populate_runtime_env(*, project_root: Path, config_name: str) -> None:
    os.environ[PROJECT_ROOT_ENV] = str(project_root)
    os.environ[CONFIG_NAME_ENV] = config_name
    os.environ.setdefault("LOCAL_OPENAI_API_KEY", "local")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    project_root = Path(args.project_root).expanduser().resolve()

    with initialize_config_dir(config_dir=str(project_root / "conf"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))

    shinka_block = cfg.get("shinka_evolve")
    if shinka_block is None:
        raise SystemExit(
            f"Config '{args.config_name}' has no top-level `shinka_evolve:` block; "
            "this is required for the ShinkaEvolve baseline."
        )

    if not bool(shinka_block.get("enabled", True)):
        print("[shinka_baseline] shinka_evolve.enabled=false; nothing to do.", file=sys.stderr)
        return 0

    explicit_models = shinka_block.get("llm_models")
    if explicit_models:
        llm_models = [str(item) for item in explicit_models]
    else:
        llm_models = build_shinka_model_urls(cfg)
    print(f"[shinka_baseline] llm_models = {llm_models}", file=sys.stderr, flush=True)

    results_dir = _resolve_path(shinka_block.results_dir, root=project_root)
    results_dir.mkdir(parents=True, exist_ok=True)

    _populate_runtime_env(project_root=project_root, config_name=args.config_name)

    shinka = _import_shinka()
    evo_cfg = _build_evolution_config(
        EvolutionConfig=shinka["EvolutionConfig"],
        shinka_block=shinka_block,
        project_root=project_root,
        llm_models=llm_models,
        results_dir=results_dir,
    )
    job_cfg = _build_local_job_config(
        LocalJobConfig=shinka["LocalJobConfig"],
        shinka_block=shinka_block,
        project_root=project_root,
    )
    db_cfg = _build_database_config(
        DatabaseConfig=shinka["DatabaseConfig"],
        shinka_block=shinka_block,
        results_dir=results_dir,
    )

    print(f"[shinka_baseline] results_dir = {results_dir}", file=sys.stderr, flush=True)
    if is_dataclass(evo_cfg):
        print(f"[shinka_baseline] EvolutionConfig: {asdict(evo_cfg)}", file=sys.stderr, flush=True)

    # In shinka-evolve 0.0.6 the evaluator-concurrency knob moved from
    # ``LocalJobConfig.max_parallel_jobs`` to
    # ``ShinkaEvolveRunner.max_evaluation_jobs``. Keep accepting the old
    # yaml key (``shinka_evolve.max_parallel_jobs``) and route it here;
    # older runners that don't take this kwarg fall back via TypeError.
    max_parallel_jobs = int(shinka_block.get("max_parallel_jobs") or 1)
    runner_kwargs: dict[str, Any] = {
        "evo_config": evo_cfg,
        "job_config": job_cfg,
        "db_config": db_cfg,
        "max_evaluation_jobs": max_parallel_jobs,
    }
    max_proposal_jobs = shinka_block.get("max_proposal_jobs")
    if max_proposal_jobs is not None:
        runner_kwargs["max_proposal_jobs"] = int(max_proposal_jobs)
    try:
        runner = shinka["ShinkaEvolveRunner"](**runner_kwargs)
    except TypeError:
        # Pre-0.0.6 runner doesn't take ``max_evaluation_jobs``; the
        # concurrency knob lived on LocalJobConfig instead, which we
        # already populated in the legacy fallback above.
        legacy_kwargs = {
            k: v for k, v in runner_kwargs.items()
            if k not in {"max_evaluation_jobs", "max_proposal_jobs"}
        }
        runner = shinka["ShinkaEvolveRunner"](**legacy_kwargs)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
