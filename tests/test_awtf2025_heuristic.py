"""Config, evaluator, and runtime tests for awtf2025_heuristic."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.awtf2025_heuristic._runtime.validation import (
    compute_absolute_score,
    parse_input,
    parse_output,
)
from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.run import run_evolution
from src.evolve.seed_run import run_seed_population
from src.evolve.storage import read_json

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "experiments" / "awtf2025_heuristic" / "assets" / "corpus" / "in"

VALID_CODE = """from __future__ import annotations

def solve_case(input_text: str) -> str:
    lines = [line.strip() for line in input_text.splitlines() if line.strip()]
    n, k = map(int, lines[0].split())
    vertical = ["0" * (n - 1) for _ in range(n)]
    horizontal = ["0" * n for _ in range(n - 1)]
    groups = [str(i) for i in range(k)]
    return "\\n".join(vertical + horizontal + groups)
"""


def _compose_awtf_cfg(tmp_path: Path, *, max_generations: int = 1):
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_awtf2025_heuristic",
            overrides=[
                f"paths.population_root={tmp_path / 'populations'}",
                f"paths.stats_root={tmp_path / 'stats'}",
                f"paths.data_root={tmp_path / 'data'}",
                f"paths.runs_root={tmp_path / 'runs'}",
                f"paths.api_platform_runtime_root={tmp_path / '.api_platform_runtime'}",
                "evolver.islands.seed_organisms_per_island=1",
                "evolver.islands.max_organisms_per_island=1",
                "evolver.phases.great_filter.top_h_per_island=1",
                f"evolver.max_generations={max_generations}",
                "experiments.group_commands_and_wall_planning.validation.smoke_case_ids=[0,1]",
                "experiments.group_commands_and_wall_planning.validation.full_case_ids=[0,1,2]",
            ],
        )
    cfg.api_platforms = {
        "mock": {
            "_target_": "api_platforms.mock.platform.build_platform",
        }
    }
    cfg.evolver.llm.route_weights = {"mock": 1.0}
    return cfg


def _write_organism_dir(tmp_path: Path, code: str) -> Path:
    organism_dir = tmp_path / "organism"
    organism_dir.mkdir(parents=True, exist_ok=True)
    (organism_dir / "implementation.py").write_text(code, encoding="utf-8")
    return organism_dir


def _mutable_experiment_cfg(cfg, *, mode: str):
    exp_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.experiments.group_commands_and_wall_planning, resolve=False)
    )
    exp_cfg.runtime = {"mode": mode}
    return exp_cfg


def _noop_output_for_input_text(input_text: str) -> str:
    lines = [line.strip() for line in input_text.splitlines() if line.strip()]
    n, k = map(int, lines[0].split())
    vertical = ["0" * (n - 1) for _ in range(n)]
    horizontal = ["0" * n for _ in range(n - 1)]
    groups = [str(i) for i in range(k)]
    return "\n".join(vertical + horizontal + groups)


def _expected_noop_case_score(case_id: int) -> int:
    case_path = CORPUS_DIR / f"{case_id:04d}.txt"
    raw_text = case_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    _, k = map(int, lines[0].split())
    score = 0
    for line in lines[1 : 1 + k]:
        sx, sy, tx, ty = map(int, line.split())
        score += 100 * (abs(sx - tx) + abs(sy - ty))
    return score


def test_awtf2025_config_composes() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_awtf2025_heuristic")

    assert set(cfg.experiments.keys()) == {"group_commands_and_wall_planning"}
    assert "safety" not in cfg
    assert set(cfg.api_platforms.keys()) == {
        "ollama_nemotron_cascade_2_30b",
        "ollama_qwen35_35b",
        "ollama_gemma4_31b",
    }
    assert cfg.experiments.group_commands_and_wall_planning.need_cuda is False
    assert cfg.evolver.phases.simple.experiments == ["group_commands_and_wall_planning"]
    assert cfg.evolver.phases.great_filter.enabled is True
    assert cfg.resources.evaluation.gpu_ranks == []
    assert cfg.resources.evaluation.cpu_parallel_jobs == 25
    assert cfg.evolver.prompts.project_context == "conf/experiments/awtf2025_heuristic/prompts/shared/project_context.txt"


def test_awtf2025_prompt_bundle_loads() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_awtf2025_heuristic")

    bundle = load_prompt_bundle(cfg)

    assert "Group Commands and Wall Planning" in bundle.project_context
    assert "solve_case" in bundle.implementation_system
    assert "SOLVE_CASE_BODY" in bundle.implementation_template


def test_awtf2025_noop_score_matches_closed_form_case0() -> None:
    case_path = CORPUS_DIR / "0000.txt"
    input_text = case_path.read_text(encoding="utf-8")
    contest_input = parse_input(input_text)
    contest_output = parse_output(contest_input, _noop_output_for_input_text(input_text))

    score, final_positions = compute_absolute_score(contest_input, contest_output)

    assert score == _expected_noop_case_score(0)
    assert final_positions == contest_input.starts


def test_awtf2025_evaluator_accepts_valid_candidate(tmp_path: Path) -> None:
    cfg = _compose_awtf_cfg(tmp_path)
    exp_cfg = _mutable_experiment_cfg(cfg, mode="smoke")
    experiment = instantiate(exp_cfg, _recursive_=False)
    organism_dir = _write_organism_dir(tmp_path, VALID_CODE)

    report = experiment.evaluate_organism(str(organism_dir), exp_cfg)

    smoke_case_ids = [int(case_id) for case_id in exp_cfg.validation.smoke_case_ids]
    expected_case_scores = {
        f"{case_id:04d}": _expected_noop_case_score(case_id)
        for case_id in smoke_case_ids
    }
    expected_mean = sum(expected_case_scores.values()) / float(len(expected_case_scores))

    assert report["status"] == "ok"
    assert report["num_cases"] == len(smoke_case_ids)
    assert report["case_scores"] == expected_case_scores
    assert report["mean_absolute_score"] == pytest.approx(expected_mean)
    assert report["score"] == pytest.approx(-expected_mean)
    assert Path(report["extra_json_path"]).exists()


@pytest.mark.parametrize(
    ("code", "error_match"),
    [
        ("def solve_case():\n    return ''\n", "must define solve_case"),
        ("from __future__ import annotations\n\ndef solve_case(input_text: str) -> str:\n    return ''\n", "Unexpected EOF"),
        (
            "from __future__ import annotations\n\n"
            "def solve_case(input_text: str) -> str:\n"
            "    return 123\n",
            "must return str",
        ),
        (
            "from __future__ import annotations\n\n"
            "def solve_case(input_text: str) -> str:\n"
            "    lines = [line.strip() for line in input_text.splitlines() if line.strip()]\n"
            "    int(lines[0])\n"
            "    return ''\n",
            r"solve_case failed on case 0000.*'30 59'",
        ),
    ],
)
def test_awtf2025_evaluator_rejects_invalid_candidates(tmp_path: Path, code: str, error_match: str) -> None:
    cfg = _compose_awtf_cfg(tmp_path)
    exp_cfg = _mutable_experiment_cfg(cfg, mode="smoke")
    experiment = instantiate(exp_cfg, _recursive_=False)
    organism_dir = _write_organism_dir(tmp_path, code)

    with pytest.raises(Exception, match=error_match):
        experiment.evaluate_organism(str(organism_dir), exp_cfg)


def test_run_one_executes_awtf2025_experiment(tmp_path: Path) -> None:
    cfg = _compose_awtf_cfg(tmp_path)
    config_dir = tmp_path / "eval_conf"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    organism_dir = _write_organism_dir(tmp_path, VALID_CODE)
    output_json = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validate.run_one",
            "--experiment",
            "group_commands_and_wall_planning",
            "--organism_dir",
            str(organism_dir),
            "--output_json",
            str(output_json),
            "--seed",
            "123",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--mode",
            "smoke",
            "--config_path",
            str(config_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    payload = read_json(output_json)
    assert payload["status"] == "ok"
    assert payload["num_cases"] == 2


def test_run_one_accepts_explicit_config_name_for_awtf2025_repo_conf(tmp_path: Path) -> None:
    organism_dir = _write_organism_dir(tmp_path, VALID_CODE)
    output_json = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validate.run_one",
            "--experiment",
            "group_commands_and_wall_planning",
            "--organism_dir",
            str(organism_dir),
            "--output_json",
            str(output_json),
            "--seed",
            "123",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--mode",
            "smoke",
            "--config_path",
            str(ROOT / "conf"),
            "--config_name",
            "config_awtf2025_heuristic",
            "--override",
            "experiments.group_commands_and_wall_planning.validation.smoke_case_ids=[0,1]",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    payload = read_json(output_json)
    assert payload["status"] == "ok"
    assert payload["num_cases"] == 2


def test_awtf2025_seed_and_evolve_with_mock_route(tmp_path: Path) -> None:
    cfg = _compose_awtf_cfg(tmp_path, max_generations=1)

    seed_summary = run_seed_population(cfg)
    evolve_summary = run_evolution(cfg)

    assert seed_summary["total_generations"] == 0
    assert seed_summary["active_population_size"] == 2
    assert evolve_summary["total_generations"] == 1
    assert evolve_summary["active_population_size"] == 2

    population_state = read_json(Path(str(cfg.paths.population_root)) / "population_state.json")
    assert population_state["current_generation"] == 1
    assert population_state["inflight_seed"] is None
    assert population_state["inflight_generation"] is None

    for entry in population_state["active_organisms"]:
        implementation = (Path(entry["organism_dir"]) / "implementation.py").read_text(encoding="utf-8")
        assert "def solve_case(input_text: str) -> str:" in implementation
