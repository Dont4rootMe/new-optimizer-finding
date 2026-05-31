"""ShinkaEvolve baseline entry-point.

Reads a Hydra config (`baselines/<experiment>` by convention), pulls the
`shinka_evolve:` block + the project's `evolver.llm.route_weights`/`api_platforms.*`
sections, builds the appropriate ShinkaEvolve dataclasses, and runs the loop.

Invoked from `scripts/run_shinka_baseline.sh` after Ollama instances are up.
"""

from __future__ import annotations

import argparse
import functools
import inspect
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


class _TokenBudgetExceeded(RuntimeError):
    """Raised inside the wrapped LLM client when a capped model hits its budget."""

    def __init__(self, model_name: str, spent: int, limit: int) -> None:
        super().__init__(
            f"per-model token budget exceeded: model={model_name} spent={spent} limit={limit}"
        )
        self.model_name = model_name
        self.spent = spent
        self.limit = limit


def _normalize_model_name(name: str) -> str:
    """Reduce a shinka model identifier to the bare model id.

    Shinka reports ``QueryResult.model_name`` as the bare ``provider_model_id``
    (e.g. ``qwen3.5:122b``), but defensively strip the ``local/<model>@<url>``
    wrapper in case a future version surfaces the full URL form so caps still
    match.
    """

    text = str(name).strip()
    if text.startswith("local/"):
        text = text[len("local/"):]
    if "@" in text:
        text = text.split("@", 1)[0]
    return text


def _parse_token_caps(shinka_block: DictConfig) -> dict[str, int]:
    """Parse ``shinka_evolve.max_tokens_per_model`` into ``{model: int}``.

    Keyed by the model id Shinka reports (the bare ``provider_model_id``, e.g.
    ``qwen3.5:122b`` / ``gemma4:31b`` / ``qwen3.5:35b``). A value of ``false`` /
    ``null`` / ``-1`` disables the cap for that model. Absent or all-``false``
    (the shipped default) returns ``{}`` — no token stop is installed and the
    baseline runs exactly as before.
    """

    raw = shinka_block.get("max_tokens_per_model")
    if raw is None:
        return {}
    container = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)
    if not isinstance(container, dict):
        raise SystemExit(
            "shinka_evolve.max_tokens_per_model must be a mapping of model -> int|false."
        )
    caps: dict[str, int] = {}
    for model, value in container.items():
        if value is None or value is False:
            continue
        if isinstance(value, bool):  # only ``True`` reaches here
            raise SystemExit(
                f"shinka_evolve.max_tokens_per_model[{model}] must be a non-negative integer "
                "or false, got true."
            )
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "false", "none", "null"}:
                continue
            value = text
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"shinka_evolve.max_tokens_per_model[{model}] must be a non-negative integer or false."
            ) from exc
        if limit == -1:
            continue
        if limit < 0:
            raise SystemExit(
                f"shinka_evolve.max_tokens_per_model[{model}] must be a non-negative integer or false."
            )
        caps[_normalize_model_name(str(model))] = limit
    return caps


def _install_shinka_token_budget_guard(runner: Any, caps: dict[str, int]) -> bool:
    """Best-effort per-model token cap for the (opaque, external) Shinka runner.

    ShinkaEvolve has no public per-model token budget hook — its native
    ``max_api_costs`` is a single global *cost* budget, and cost is 0 for local
    Ollama models. So we wrap the runner's LLM client query method(s) to tally
    ``input_tokens + output_tokens (+ thinking_tokens)`` per ``model_name`` from
    each returned ``QueryResult``. When a capped model reaches its budget we set
    the runner's ``should_stop`` event (Shinka's own stop mechanism) and raise
    :class:`_TokenBudgetExceeded` to abort the in-flight call.

    Returns ``True`` when the guard was installed. On any structural mismatch
    (Shinka internals differ across versions and the package is not installed
    locally to verify) it logs a loud warning and returns ``False`` rather than
    crashing the baseline — the run proceeds without the cap.
    """

    if not caps:
        return False
    client = getattr(runner, "llm", None)
    if client is None:
        print(
            "[shinka_baseline] WARNING: could not locate runner.llm; per-model token cap "
            "NOT enforced for this run.",
            file=sys.stderr,
            flush=True,
        )
        return False
    candidate_methods = [
        name
        for name in ("query", "batch_query", "async_query", "aquery", "query_batch")
        if callable(getattr(client, name, None))
    ]
    if not candidate_methods:
        print(
            "[shinka_baseline] WARNING: no known query method on runner.llm; per-model token "
            "cap NOT enforced for this run.",
            file=sys.stderr,
            flush=True,
        )
        return False

    spent: dict[str, int] = {}

    def _as_results(ret: Any) -> list[Any]:
        if ret is None:
            return []
        if isinstance(ret, (list, tuple)):
            return list(ret)
        return [ret]

    def _account(ret: Any) -> None:
        for result in _as_results(ret):
            model = getattr(result, "model_name", None)
            if model is None:
                continue
            key = _normalize_model_name(str(model))
            tokens = 0
            for attr in ("input_tokens", "output_tokens", "thinking_tokens"):
                value = getattr(result, attr, 0) or 0
                try:
                    tokens += int(value)
                except (TypeError, ValueError):
                    continue
            spent[key] = spent.get(key, 0) + tokens
            cap = caps.get(key)
            if cap is not None and spent[key] >= cap:
                stop_event = getattr(runner, "should_stop", None)
                if stop_event is not None and hasattr(stop_event, "set"):
                    try:
                        stop_event.set()
                    except Exception:  # noqa: BLE001
                        pass
                raise _TokenBudgetExceeded(key, spent[key], cap)

    for name in candidate_methods:
        original = getattr(client, name)
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def _async_wrapper(*args: Any, __original: Any = original, **kwargs: Any) -> Any:
                ret = await __original(*args, **kwargs)
                _account(ret)
                return ret

            setattr(client, name, _async_wrapper)
        else:

            @functools.wraps(original)
            def _sync_wrapper(*args: Any, __original: Any = original, **kwargs: Any) -> Any:
                ret = __original(*args, **kwargs)
                _account(ret)
                return ret

            setattr(client, name, _sync_wrapper)

    print(
        f"[shinka_baseline] per-model token cap active: {caps} (wrapped {candidate_methods})",
        file=sys.stderr,
        flush=True,
    )
    return True


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

    # ``EvolutionConfig.task_sys_msg`` is the system prompt the LLM
    # receives on every patch attempt. The upstream default is a
    # generic "improve the program" message, which leaves the model to
    # reverse-engineer the task from the failing program — that's why
    # the first run of this baseline produced 0/78 correct programs.
    # Two opt-in ways to override it, isolated to the baseline tree so
    # the main evolutionary loop is untouched:
    #   * ``shinka_evolve.task_sys_msg`` — inline string
    #   * ``shinka_evolve.task_sys_msg_path`` — path to a file (resolved
    #     against the project root) whose contents become the message
    # When both are set the inline message wins. When neither is set
    # the upstream default applies.
    task_sys_msg_inline = shinka_block.get("task_sys_msg")
    task_sys_msg_path = shinka_block.get("task_sys_msg_path")
    if task_sys_msg_inline:
        kwargs["task_sys_msg"] = str(task_sys_msg_inline)
    elif task_sys_msg_path:
        resolved = _resolve_path(task_sys_msg_path, root=project_root)
        if not resolved.exists():
            raise FileNotFoundError(
                f"shinka_evolve.task_sys_msg_path does not exist: {resolved}"
            )
        kwargs["task_sys_msg"] = resolved.read_text(encoding="utf-8")

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

    # Optional per-model token-budget stop (mirrors evolver.max_tokens_per_model
    # for the main loop). Inert by default — only installs the guard when the
    # yaml sets a real integer cap for at least one model.
    token_caps = _parse_token_caps(shinka_block)
    if token_caps:
        _install_shinka_token_budget_guard(runner, token_caps)
    try:
        runner.run()
    except _TokenBudgetExceeded as exc:
        print(f"[shinka_baseline] stopping run: {exc}", file=sys.stderr, flush=True)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
