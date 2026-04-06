"""Single-experiment evaluator used by evolve subprocess workers."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.evolve.storage import write_json

LOGGER = logging.getLogger(__name__)


def _detect_root_config_name(conf_dir: Path) -> str:
    """Pick the Hydra config name for a given config directory."""

    if (conf_dir / "config.yaml").exists():
        return "config"
    if (conf_dir / "config_optimization_survey.yaml").exists():
        return "config_optimization_survey"
    raise FileNotFoundError(
        f"Could not find a supported root config in {conf_dir}. "
        "Expected config.yaml or config_optimization_survey.yaml."
    )


def _prepare_experiment_cfg(
    cfg: DictConfig,
    experiment_name: str,
    mode: str,
    device: str,
    precision: str,
    seed: int,
    organism_dir: str,
) -> DictConfig:
    exp_cfg = OmegaConf.create(OmegaConf.to_container(cfg.experiments[experiment_name], resolve=False))

    exp_cfg.seed = int(seed)
    exp_cfg.deterministic = bool(cfg.deterministic)
    exp_cfg.runtime = {
        "mode": mode,
        "smoke": mode == "smoke",
        "organism_dir": organism_dir,
    }

    exp_cfg.compute.device = str(device)
    exp_cfg.compute.precision = str(precision)
    exp_cfg.compute.num_workers = int(cfg.num_workers)

    if mode == "smoke":
        exp_cfg.compute.max_steps = int(exp_cfg.compute.smoke_steps)

    exp_cfg.paths = OmegaConf.create(OmegaConf.to_container(cfg.paths, resolve=True))
    if "data" not in exp_cfg:
        exp_cfg.data = {}
    if not exp_cfg.data.get("root"):
        exp_cfg.data.root = str(Path(cfg.paths.data_root) / experiment_name)

    if "safety" not in exp_cfg:
        exp_cfg.safety = {}
    for key, value in cfg.safety.items():
        if key not in exp_cfg.safety or exp_cfg.safety[key] is None:
            exp_cfg.safety[key] = value

    return exp_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--organism_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--config_path", default="conf")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    out_path = Path(args.output_json).expanduser().resolve()

    try:
        conf_dir = Path(args.config_path).expanduser().resolve()
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cfg = compose(config_name=_detect_root_config_name(conf_dir), overrides=list(args.override))

        experiment_name = str(args.experiment)
        if experiment_name not in cfg.experiments:
            raise KeyError(f"Experiment '{experiment_name}' not found in cfg.experiments")

        exp_cfg = _prepare_experiment_cfg(
            cfg=cfg,
            experiment_name=experiment_name,
            mode=str(args.mode),
            device=str(args.device),
            precision=str(args.precision),
            seed=int(args.seed),
            organism_dir=str(args.organism_dir),
        )

        experiment = instantiate(exp_cfg, _recursive_=False)
        if not hasattr(experiment, "evaluate_organism"):
            raise AttributeError(
                f"Experiment '{experiment_name}' must define evaluate_organism(organism_dir, cfg)."
            )

        payload = experiment.evaluate_organism(str(args.organism_dir), exp_cfg)
        if not isinstance(payload, dict):
            raise TypeError(f"Experiment '{experiment_name}' returned non-dict payload: {type(payload).__name__}")
        if "score" not in payload:
            raise ValueError(f"Experiment '{experiment_name}' report is missing required field 'score'.")

        write_json(out_path, payload)
        sys.exit(0 if str(payload.get("status", "ok")) == "ok" else 1)

    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "score": None,
            "error_msg": str(exc),
        }
        write_json(out_path, payload)
        LOGGER.exception("run_one failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
