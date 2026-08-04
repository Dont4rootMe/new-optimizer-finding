"""Download one exact Hugging Face model snapshot and record its local path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, help="JSON manifest path")
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    snapshot = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        max_workers=max(1, int(args.max_workers)),
    )
    payload = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot_path": str(Path(snapshot).resolve()),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
