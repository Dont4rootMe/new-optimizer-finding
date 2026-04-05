"""Module entrypoint used to spawn one route broker subprocess."""

from __future__ import annotations

import argparse
import json
import signal
import sys

from api_platforms._core.broker import RouteBroker
from api_platforms._core.types import ApiRouteConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()

    route_cfg = ApiRouteConfig.from_dict(json.loads(args.config_json))
    broker = RouteBroker(route_cfg, args.runtime_dir)

    def _shutdown(*_args: object) -> None:
        broker.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    broker.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
