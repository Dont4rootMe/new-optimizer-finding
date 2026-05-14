"""Build ShinkaEvolve model URL list from the host project's api_platforms config.

ShinkaEvolve accepts OpenAI-compatible local routes in the form
`local/<model>@http://host:port/v1[?api_key_env=ENV]`. Our `api_platforms.*`
Hydra blocks describe Ollama instances at `http://host:port/api`. This module
bridges them: for every route referenced by `evolver.llm.route_weights` it
derives one shinka URL string per logical Ollama route (the first instance from
`derive_ollama_instance_configs`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from api_platforms._core.config import derive_ollama_instance_configs
from api_platforms._core.discovery import load_route_configs


def _ollama_api_to_openai(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = "/v1"
    return urlunsplit((parsed.scheme or "http", parsed.netloc, path, "", ""))


def build_shinka_model_urls(cfg: DictConfig) -> list[str]:
    """Return shinka-format `local/<model>@<openai-base>` URLs for enabled routes.

    The set of "enabled" routes is the keys of `evolver.llm.route_weights`.
    Non-Ollama routes are skipped with a warning to stderr — ShinkaEvolve's
    `local/` prefix targets OpenAI-compatible endpoints, and our cloud routes
    (Anthropic / OpenAI) need different shinka prefixes that this baseline
    template doesn't support yet.
    """

    weights = (
        cfg.get("evolver", {}).get("llm", {}).get("route_weights")
        if cfg.get("evolver") is not None
        else None
    )
    if weights is None:
        raise RuntimeError(
            "Cannot derive shinka model list: cfg.evolver.llm.route_weights is missing. "
            "Either set baseline shinka_evolve.llm_models explicitly or inherit a config "
            "that defines evolver.llm.route_weights."
        )

    requested_route_ids = [route_id for route_id, weight in weights.items() if float(weight) > 0]
    if not requested_route_ids:
        raise RuntimeError("evolver.llm.route_weights is empty or all-zero; nothing to pass to shinka.")

    routes = load_route_configs(cfg)
    urls: list[str] = []
    for route_id in requested_route_ids:
        route_cfg = routes.get(route_id)
        if route_cfg is None:
            print(
                f"[shinka_baseline] WARNING: route_weights references '{route_id}' "
                f"but it is not present in api_platforms; skipping.",
                file=sys.stderr,
            )
            continue
        if route_cfg.backend != "ollama":
            print(
                f"[shinka_baseline] WARNING: route '{route_id}' has backend "
                f"'{route_cfg.backend}', which this baseline does not yet support; skipping. "
                f"Only Ollama routes are bridged into ShinkaEvolve via local/ URLs.",
                file=sys.stderr,
            )
            continue
        instances = derive_ollama_instance_configs(route_cfg)
        if not instances:
            continue
        # Emit one shinka URL per concrete ollama instance — multi-GPU
        # routes spawn one process per `gpu_ranks` group, each on its own
        # port, and Shinka's UCB1 bandit treats them as independent arms.
        # Previously only ``instances[0]`` was emitted, so all extra
        # instances launched by `ensure_ollama_runtime` sat idle.
        for instance in instances:
            openai_base = _ollama_api_to_openai(str(instance.base_url))
            urls.append(f"local/{instance.provider_model_id}@{openai_base}")

    if not urls:
        raise RuntimeError(
            "Resolved an empty model URL list. Check that evolver.llm.route_weights references "
            "at least one Ollama route declared under api_platforms."
        )
    return urls


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Print the shinka-format model URL list resolved from a Hydra config."
    )
    parser.add_argument("--config-name", required=True, help="Hydra config name (e.g. baselines/awtf2025_heuristic).")
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Project root that contains conf/ (defaults to cwd).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides (e.g. evolver.llm.route_weights.ollama_gemma4_31b=2).",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    with initialize_config_dir(config_dir=str(project_root / "conf"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=list(args.overrides))

    urls = build_shinka_model_urls(cfg)
    for url in urls:
        print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
