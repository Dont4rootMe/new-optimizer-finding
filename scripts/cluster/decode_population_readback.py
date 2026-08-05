"""Decode ``EVOLUTIONLOOP_READBACK`` blob events from scheduler-log text."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.cluster.population_readback import EVENT_PREFIX


def parse_events(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        marker = line.find(EVENT_PREFIX)
        if marker < 0:
            continue
        try:
            payload = json.loads(line[marker + len(EVENT_PREFIX) :].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def decode_blobs(events: Iterable[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if event.get("event") != "blob_chunk":
            continue
        name = str(event.get("name", ""))
        if not name or Path(name).name != name:
            raise ValueError(f"unsafe or empty blob name: {name!r}")
        index = int(event["index"])
        previous = grouped[name].get(index)
        if previous is not None and previous.get("data") != event.get("data"):
            raise ValueError(f"conflicting duplicate chunk {index} for {name}")
        grouped[name][index] = event

    output_dir.mkdir(parents=True, exist_ok=True)
    decoded: list[dict[str, Any]] = []
    for name, indexed in sorted(grouped.items()):
        exemplar = next(iter(indexed.values()))
        total = int(exemplar["total"])
        if sorted(indexed) != list(range(total)):
            raise ValueError(f"incomplete chunks for {name}: have {sorted(indexed)}, expected 0..{total - 1}")
        metadata_fields = ("encoding", "sha256", "uncompressed_bytes", "compressed_bytes", "total")
        for event in indexed.values():
            if any(event.get(field) != exemplar.get(field) for field in metadata_fields):
                raise ValueError(f"inconsistent chunk metadata for {name}")
        if exemplar.get("encoding") != "gzip+base64":
            raise ValueError(f"unsupported blob encoding for {name}: {exemplar.get('encoding')!r}")
        encoded = "".join(str(indexed[index]["data"]) for index in range(total))
        compressed = base64.b64decode(encoded, validate=True)
        if len(compressed) != int(exemplar["compressed_bytes"]):
            raise ValueError(f"compressed size mismatch for {name}")
        payload = gzip.decompress(compressed)
        if len(payload) != int(exemplar["uncompressed_bytes"]):
            raise ValueError(f"uncompressed size mismatch for {name}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != exemplar["sha256"]:
            raise ValueError(f"sha256 mismatch for {name}")
        destination = output_dir / name
        destination.write_bytes(payload)
        decoded.append(
            {
                "name": name,
                "path": str(destination),
                "bytes": len(payload),
                "sha256": digest,
                "media_type": exemplar.get("media_type"),
            }
        )
    return decoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", help="scheduler log file; stdin when omitted")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.log:
        with Path(args.log).open(encoding="utf-8", errors="replace") as handle:
            events = parse_events(handle)
    else:
        events = parse_events(sys.stdin)
    decoded = decode_blobs(events, Path(args.output_dir).expanduser().resolve())
    non_blob_events = [event for event in events if event.get("event") != "blob_chunk"]
    manifest = {"events": non_blob_events, "decoded": decoded}
    manifest_path = Path(args.output_dir).expanduser().resolve() / "readback_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "decoded": decoded}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
