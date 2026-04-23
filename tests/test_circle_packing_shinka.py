"""Config, evaluator, and runtime tests for circle_packing_shinka."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.run import run_evolution
from src.evolve.seed_run import run_seed_population
from src.evolve.storage import read_json

ROOT = Path(__file__).resolve().parents[1]

VALID_CODE = """import numpy as np

def run_packing():
    centers = np.asarray([
        [0.10, 0.15], [0.24, 0.15], [0.38, 0.15], [0.52, 0.15], [0.66, 0.15], [0.80, 0.15], [0.94, 0.15],
        [0.17, 0.33], [0.31, 0.33], [0.45, 0.33], [0.59, 0.33], [0.73, 0.33], [0.87, 0.33],
        [0.10, 0.51], [0.24, 0.51], [0.38, 0.51], [0.52, 0.51], [0.66, 0.51], [0.80, 0.51], [0.94, 0.51],
        [0.17, 0.69], [0.31, 0.69], [0.45, 0.69], [0.59, 0.69], [0.73, 0.69], [0.87, 0.69],
    ], dtype=float)
    radii = np.full(26, 0.04, dtype=float)
    reported_sum = float(np.sum(radii))
    return centers, radii, reported_sum
"""


def _compose_circle_cfg(tmp_path: Path, *, max_generations: int = 1):
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config_circle_packing_shinka",
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


def test_circle_packing_config_composes() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_circle_packing_shinka")

    assert set(cfg.experiments.keys()) == {"unit_square_26"}
    assert "safety" not in cfg
    assert set(cfg.api_platforms.keys()) == {
        "ollama_qwen35_35b",
        "ollama_gemma4_31b",
    }
    assert cfg.experiments.unit_square_26.need_cuda is False
    assert cfg.evolver.phases.simple.experiments == ["unit_square_26"]
    assert cfg.evolver.phases.great_filter.enabled is False
    assert cfg.resources.evaluation.gpu_ranks == []
    assert cfg.resources.evaluation.cpu_parallel_jobs == 20
    assert cfg.evolver.max_organism_creations is False
    assert cfg.evolver.prompts.project_context == "conf/experiments/circle_packing_shinka/prompts/shared/project_context.txt"


def test_circle_packing_prompt_bundle_loads() -> None:
    conf_dir = ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config_circle_packing_shinka")

    bundle = load_prompt_bundle(cfg)

    assert "hidden evaluator constants" in bundle.project_context
    assert "26 circles in the unit square" not in bundle.project_context
    assert "Single rewrite contract:" in bundle.implementation_system
    assert "return ONLY the final full `implementation.py`" in bundle.implementation_system
    assert "# EVOLVE-BLOCK-START" in bundle.implementation_template
    assert "# SECTION: INIT_GEOMETRY" in bundle.implementation_template
    assert "# SECTION: OPTIONAL_CODE_SKETCH" in bundle.implementation_template
    assert "RUN_PACKING_BODY" not in bundle.implementation_template


def test_circle_packing_evaluator_accepts_valid_candidate(tmp_path: Path) -> None:
    cfg = _compose_circle_cfg(tmp_path)
    experiment = instantiate(cfg.experiments.unit_square_26, _recursive_=False)
    organism_dir = _write_organism_dir(tmp_path, VALID_CODE)

    report = experiment.evaluate_organism(str(organism_dir), cfg.experiments.unit_square_26)

    assert report["status"] == "ok"
    assert report["score"] == pytest.approx(1.04)
    assert report["num_circles"] == 26
    assert Path(report["extra_npz_path"]).exists()


@pytest.mark.parametrize(
    ("code", "error_match"),
    [
        ("def run_packing():\n    return [[0.5, 0.5]], [0.1], 0.1\n", "Centers shape incorrect"),
        (
            "import numpy as np\n"
            "def run_packing():\n"
            "    centers = np.zeros((26, 2), dtype=float)\n"
            "    radii = np.full(26, -0.1, dtype=float)\n"
            "    return centers, radii, float(np.sum(radii))\n",
            "Negative radii",
        ),
        (
            "import numpy as np\n"
            "def run_packing():\n"
            "    centers = np.asarray([[0.01, 0.01]] * 26, dtype=float)\n"
            "    radii = np.full(26, 0.05, dtype=float)\n"
            "    return centers, radii, float(np.sum(radii))\n",
            "outside the square|overlap",
        ),
        (
            "import numpy as np\n"
            "def run_packing():\n"
            "    centers = np.asarray([[0.2 + 0.02 * i, 0.2] for i in range(26)], dtype=float)\n"
            "    radii = np.full(26, 0.04, dtype=float)\n"
            "    return centers, radii, float(np.sum(radii))\n",
            "overlap",
        ),
        (
            "import numpy as np\n"
            "def run_packing():\n"
            "    centers = np.asarray([[0.10 + 0.02 * (i % 5), 0.10 + 0.02 * (i // 5)] for i in range(26)], dtype=float)\n"
            "    radii = np.full(26, 0.0, dtype=float)\n"
            "    return centers, radii, 1.0\n",
            "does not match reported_sum",
        ),
    ],
)
def test_circle_packing_evaluator_rejects_invalid_candidates(tmp_path: Path, code: str, error_match: str) -> None:
    cfg = _compose_circle_cfg(tmp_path)
    experiment = instantiate(cfg.experiments.unit_square_26, _recursive_=False)
    organism_dir = _write_organism_dir(tmp_path, code)

    with pytest.raises(Exception, match=error_match):
        experiment.evaluate_organism(str(organism_dir), cfg.experiments.unit_square_26)


def test_run_one_executes_circle_packing_experiment(tmp_path: Path) -> None:
    cfg = _compose_circle_cfg(tmp_path)
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
            "unit_square_26",
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
    assert payload["score"] == pytest.approx(1.04)


def test_run_one_requires_explicit_config_name_for_repo_conf(tmp_path: Path) -> None:
    organism_dir = _write_organism_dir(tmp_path, VALID_CODE)
    output_json = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validate.run_one",
            "--experiment",
            "unit_square_26",
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
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 1
    payload = read_json(output_json)
    assert payload["status"] == "failed"
    assert "pass --config_name <preset> explicitly" in payload["error_msg"].lower()


def test_run_one_accepts_explicit_config_name_for_repo_conf(tmp_path: Path) -> None:
    organism_dir = _write_organism_dir(tmp_path, VALID_CODE)
    output_json = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validate.run_one",
            "--experiment",
            "unit_square_26",
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
            "config_circle_packing_shinka",
            "--override",
            f"paths.data_root={tmp_path / 'data'}",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert completed.returncode == 0, completed.stderr
    payload = read_json(output_json)
    assert payload["status"] == "ok"
    assert payload["score"] == pytest.approx(1.04)


def test_circle_packing_seed_and_evolve_with_mock_route(tmp_path: Path) -> None:
    cfg = _compose_circle_cfg(tmp_path, max_generations=1)

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
        assert "def run_packing():" in implementation
