"""Machine-readable task catalog and repository readiness audit.

The Hydra experiment nodes remain the executable source of truth. This module
adds one uniform discovery surface over those nodes, records scientific score
semantics, and checks that configs, evaluators, prompts, seeds, and external
prerequisites have not drifted apart.

Usage::

    python -m experiments.catalog list
    python -m experiments.catalog check
    python -m experiments.catalog check --require-ready
    python -m experiments.catalog markdown
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import DictConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    family: str
    title: str
    preset: str
    experiment_key: str
    selector_override: str | None
    phase: str
    enabled_by_default: bool
    candidate_contract: str
    evolution_score: str
    source: str
    scientific_note: str


@dataclass(frozen=True, slots=True)
class AuditFinding:
    severity: str  # error | blocked | warning | info
    task_id: str
    code: str
    message: str


_CO_BENCH_TASKS = (
    ("TSP", "Travelling salesman problem"),
    ("BIN_PACKING_1D", "Bin packing — one-dimensional"),
    ("MULTI_KNAPSACK", "Multidimensional knapsack problem"),
    ("SET_COVERING", "Set covering"),
    ("GRAPH_COLOURING", "Graph colouring"),
    ("JOB_SHOP", "Job shop scheduling"),
)

_OPTIMIZATION_SIMPLE = (
    "synthetic_logreg",
    "mnist_mlp",
    "poly_regression",
    "rosenbrock_net",
    "xor_mlp",
    "sin_regression",
    "matrix_factorization",
    "tiny_autoencoder",
    "two_spirals",
    "linear_denoiser",
    "conv1d_classify",
    "quadratic_bowl",
)
_OPTIMIZATION_GREAT_FILTER = (
    "cifar_convnet",
    "audio_transformer",
    "minigpt_wikitext2",
    "ddpm_cifar10",
    "lora_sft",
)
_OPTIMIZATION_DISABLED = {"audio_transformer", "ddpm_cifar10", "lora_sft"}


def task_specs() -> tuple[TaskSpec, ...]:
    """Return every task shipped by the repository in stable order."""

    specs: list[TaskSpec] = [
        TaskSpec(
            task_id="circle_packing_shinka/unit_square_26",
            family="circle_packing_shinka",
            title="Pack exactly 26 unequal circles in a unit square",
            preset="config_circle_packing_shinka",
            experiment_key="unit_square_26",
            selector_override=None,
            phase="simple",
            enabled_by_default=True,
            candidate_contract="run_packing() -> (centers[26,2], radii[26], reported_sum)",
            evolution_score="sum(radii), maximized; feasibility checked at atol=1e-6",
            source="Shinka circle-packing benchmark",
            scientific_note="Evaluator recomputes the objective; candidate-reported sum is diagnostic only.",
        ),
        TaskSpec(
            task_id="awtf2025_heuristic/group_commands_and_wall_planning",
            family="awtf2025_heuristic",
            title="AtCoder AWTF 2025 group commands and wall planning",
            preset="config_awtf2025_heuristic",
            experiment_key="group_commands_and_wall_planning",
            selector_override=None,
            phase="simple",
            enabled_by_default=True,
            candidate_contract="solve_case(input_text: str) -> output_text: str",
            evolution_score="negative mean absolute contest score over the fixed corpus, maximized",
            source="AtCoder AWTF 2025 heuristic task",
            scientific_note=(
                "This is a fixed-corpus surrogate, not AtCoder's relative leaderboard score; "
                "cross-run comparison requires the same corpus."
            ),
        ),
    ]
    specs.extend(
        TaskSpec(
            task_id=f"co_bench/{identifier}",
            family="co_bench",
            title=title,
            preset="config_co-bench",
            experiment_key="co_bench",
            selector_override=f"experiments.co_bench.CO_BENCH_TASK={identifier}",
            phase="simple",
            enabled_by_default=True,
            candidate_contract="solve(**instance) -> solution: dict",
            evolution_score="CO-Bench normalized dev_score, maximized",
            source="CO-Bench task dataset and evaluator",
            scientific_note=(
                "Selection uses the dev split only; held-out test metrics are hidden unless "
                "expose_test_metrics=true is explicitly requested for final reporting."
            ),
        )
        for identifier, title in _CO_BENCH_TASKS
    )
    for name in (*_OPTIMIZATION_SIMPLE, *_OPTIMIZATION_GREAT_FILTER):
        phase = "simple" if name in _OPTIMIZATION_SIMPLE else "great_filter"
        specs.append(
            TaskSpec(
                task_id=f"optimization_survey/{name}",
                family="optimization_survey",
                title=name.replace("_", " ").title(),
                preset="config_optimization_survey",
                experiment_key=name,
                selector_override=None,
                phase=phase,
                enabled_by_default=name not in _OPTIMIZATION_DISABLED,
                candidate_contract="build_optimizer(model, max_steps) -> optimizer controller",
                evolution_score=(
                    "harmonic mean of baseline/final loss and baseline/steps-to-baseline, maximized"
                ),
                source="Repository optimization survey",
                scientific_note=(
                    "A version-matched baseline profile is mandatory; without it scoring fails "
                    "instead of silently inventing a reference."
                ),
            )
        )
    return tuple(specs)


def _compose_for_spec(
    spec: TaskSpec,
    *,
    conf_dir: Path,
    cache: dict[tuple[str, str | None], DictConfig],
) -> DictConfig:
    cache_key = (spec.preset, spec.selector_override)
    if cache_key not in cache:
        overrides = [spec.selector_override] if spec.selector_override else []
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cache[cache_key] = compose(config_name=spec.preset, overrides=overrides)
    return cache[cache_key]


def _repo_path(value: object, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _append_path_error(
    findings: list[AuditFinding],
    spec: TaskSpec,
    path: Path,
    *,
    code: str,
    label: str,
) -> None:
    if not path.is_file():
        findings.append(
            AuditFinding("error", spec.task_id, code, f"Missing {label}: {path}")
        )


def audit_catalog(repo_root: str | Path = REPO_ROOT) -> list[AuditFinding]:
    """Audit executable contracts separately from optional readiness blockers."""

    root = Path(repo_root).expanduser().resolve()
    conf_dir = root / "conf"
    findings: list[AuditFinding] = []
    specs = task_specs()
    task_ids = [spec.task_id for spec in specs]
    if len(task_ids) != len(set(task_ids)):
        findings.append(
            AuditFinding("error", "catalog", "duplicate_task_id", "Task ids are not unique.")
        )

    cache: dict[tuple[str, str | None], DictConfig] = {}
    for spec in specs:
        try:
            cfg = _compose_for_spec(spec, conf_dir=conf_dir, cache=cache)
        except Exception as exc:  # noqa: BLE001
            findings.append(
                AuditFinding(
                    "error",
                    spec.task_id,
                    "compose_failed",
                    f"Hydra preset did not compose: {type(exc).__name__}: {exc}",
                )
            )
            continue
        if spec.experiment_key not in cfg.experiments:
            findings.append(
                AuditFinding(
                    "error",
                    spec.task_id,
                    "experiment_missing",
                    f"{spec.experiment_key!r} is absent from cfg.experiments.",
                )
            )
            continue
        exp_cfg = cfg.experiments[spec.experiment_key]
        enabled = bool(exp_cfg.get("enabled", False))
        if enabled != spec.enabled_by_default:
            findings.append(
                AuditFinding(
                    "error",
                    spec.task_id,
                    "enabled_drift",
                    f"Catalog enabled={spec.enabled_by_default}, config enabled={enabled}.",
                )
            )
        target = str(exp_cfg.get("_target_", ""))
        try:
            evaluator_cls = get_class(target)
            parameters = tuple(inspect.signature(evaluator_cls.evaluate_organism).parameters)
            if parameters[:3] != ("self", "organism_dir", "cfg"):
                raise TypeError(f"unexpected signature parameters={parameters}")
        except Exception as exc:  # noqa: BLE001
            findings.append(
                AuditFinding(
                    "error",
                    spec.task_id,
                    "evaluator_contract",
                    f"{target!r} lacks evaluate_organism(organism_dir, cfg): {exc}",
                )
            )

        seed_path = _repo_path(cfg.evolver.islands.seed_program_path, root)
        _append_path_error(
            findings,
            spec,
            seed_path,
            code="seed_missing",
            label="seed implementation",
        )
        for prompt_name, prompt_value in cfg.evolver.prompts.items():
            if not isinstance(prompt_value, str) or not prompt_value.endswith(".txt"):
                continue
            _append_path_error(
                findings,
                spec,
                _repo_path(prompt_value, root),
                code="prompt_missing",
                label=f"prompt evolver.prompts.{prompt_name}",
            )

        if spec.family == "circle_packing_shinka":
            continue
        if spec.family == "awtf2025_heuristic":
            corpus_dir = _repo_path(exp_cfg.validation.corpus_dir, root)
            expected_ids = [int(value) for value in exp_cfg.validation.full_case_ids]
            missing = [case_id for case_id in expected_ids if not (corpus_dir / f"{case_id:04d}.txt").is_file()]
            if missing:
                findings.append(
                    AuditFinding(
                        "blocked",
                        spec.task_id,
                        "corpus_missing",
                        f"Missing {len(missing)}/{len(expected_ids)} fixed corpus files under {corpus_dir}.",
                    )
                )
            findings.append(
                AuditFinding(
                    "warning",
                    spec.task_id,
                    "surrogate_score",
                    spec.scientific_note,
                )
            )
            continue
        if spec.family == "co_bench":
            checkout = Path(os.environ.get("COBENCH_ROOT", root / "third_party" / "CO-Bench")).expanduser().resolve()
            data_root = _repo_path(exp_cfg.data.cobench_src_dir, root)
            if not (checkout / "evaluation" / "__init__.py").is_file():
                findings.append(
                    AuditFinding(
                        "blocked",
                        spec.task_id,
                        "cobench_checkout_missing",
                        f"CO-Bench checkout is unavailable at {checkout}.",
                    )
                )
            task_data = data_root / str(exp_cfg.co_bench_task) / "config.py"
            if not task_data.is_file():
                findings.append(
                    AuditFinding(
                        "blocked",
                        spec.task_id,
                        "cobench_data_missing",
                        f"CO-Bench task data is unavailable at {task_data}.",
                    )
                )
            continue
        if spec.family == "optimization_survey":
            baseline_path = _repo_path(exp_cfg.baseline.profile_path, root)
            if spec.enabled_by_default and not baseline_path.is_file():
                findings.append(
                    AuditFinding(
                        "blocked",
                        spec.task_id,
                        "baseline_missing",
                        f"Generate the required baseline profile at {baseline_path}.",
                    )
                )
            elif not spec.enabled_by_default:
                findings.append(
                    AuditFinding(
                        "info",
                        spec.task_id,
                        "disabled_by_default",
                        "Task is shipped but intentionally disabled in the canonical preset.",
                    )
                )

    # Family-level exhaustiveness checks ensure new Hydra tasks cannot appear
    # without an explicit scientific contract in this catalog.
    optimization_cfg = next(
        cfg for (preset, selector), cfg in cache.items()
        if preset == "config_optimization_survey" and selector is None
    )
    catalog_optimization = {
        spec.experiment_key for spec in specs if spec.family == "optimization_survey"
    }
    configured_optimization = set(optimization_cfg.experiments)
    if catalog_optimization != configured_optimization:
        findings.append(
            AuditFinding(
                "error",
                "optimization_survey",
                "catalog_exhaustiveness",
                "Catalog/config mismatch: "
                f"catalog_only={sorted(catalog_optimization - configured_optimization)}, "
                f"config_only={sorted(configured_optimization - catalog_optimization)}.",
            )
        )
    return findings


def render_markdown(specs: Iterable[TaskSpec] | None = None) -> str:
    """Render the compact registry table embedded in the knowledge base."""

    rows = list(specs or task_specs())
    lines = [
        "| Task ID | Default | Phase | Candidate contract | Evolution score |",
        "|---|---:|---|---|---|",
    ]
    for spec in rows:
        enabled = "yes" if spec.enabled_by_default else "no"
        values = (
            spec.task_id,
            enabled,
            spec.phase,
            spec.candidate_contract,
            spec.evolution_score,
        )
        escaped = [value.replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def _print_findings(findings: list[AuditFinding]) -> None:
    for finding in findings:
        print(
            f"{finding.severity.upper():7} {finding.task_id} "
            f"[{finding.code}] {finding.message}"
        )
    counts = {
        severity: sum(item.severity == severity for item in findings)
        for severity in ("error", "blocked", "warning", "info")
    }
    print("SUMMARY " + " ".join(f"{key}={value}" for key, value in counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="print catalog records")
    list_parser.add_argument("--json", action="store_true", help="emit JSON")
    check_parser = subparsers.add_parser("check", help="audit contracts and readiness")
    check_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="treat missing external prerequisites as a failing result",
    )
    subparsers.add_parser("markdown", help="render the knowledge-base registry table")
    args = parser.parse_args()

    if args.command == "list":
        specs = task_specs()
        if args.json:
            print(json.dumps([asdict(spec) for spec in specs], indent=2, ensure_ascii=False))
        else:
            print(render_markdown(specs))
        return
    if args.command == "markdown":
        print(render_markdown())
        return

    findings = audit_catalog()
    _print_findings(findings)
    errors = any(finding.severity == "error" for finding in findings)
    blocked = any(finding.severity == "blocked" for finding in findings)
    raise SystemExit(1 if errors or (args.require_ready and blocked) else 0)


if __name__ == "__main__":
    main()
