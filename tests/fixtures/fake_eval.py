"""Fast fake evaluator module for integration tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--optimizer_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--config_path", default="conf")
    args = parser.parse_args()

    # Deterministic synthetic metric per experiment.
    final_score = 0.8 if args.experiment.endswith("a") else 0.6

    payload = {
        "status": "ok",
        "final_score": final_score,
        "objective_name": "train_loss",
        "objective_direction": "min",
        "objective_last": final_score,
        "objective_best": final_score,
        "objective_best_step": 5,
        "first_step_at_or_below_baseline": 5,
        "time_sec": 0.01,
        "steps": 5,
        "converged": True,
        "error_msg": None,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    # Ensure there is some runtime for subprocess accounting.
    time.sleep(0.02)


if __name__ == "__main__":
    main()
