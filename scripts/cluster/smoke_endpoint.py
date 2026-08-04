"""Concurrent OpenAI-compatible smoke probe with persisted usage payloads."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request as urllib_request


def _one_request(url: str, model: str, index: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a health-check endpoint."},
            {"role": "user", "content": f"Reply with READY {index}."},
        ],
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1024,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"},
    }
    started = time.perf_counter()
    http_request = urllib_request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(http_request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"smoke request {index} returned no choices")
    message = choices[0].get("message", {})
    if not message.get("content") and not message.get("reasoning_content"):
        raise RuntimeError(f"smoke request {index} returned no content")
    return {
        "index": index,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "response": response_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    concurrency = max(1, int(args.concurrency))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(
                lambda index: _one_request(args.base_url, args.model, index, args.timeout),
                range(concurrency),
            )
        )
    payload = {
        "status": "ok",
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": concurrency,
        "results": results,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "concurrency": concurrency}, sort_keys=True))


if __name__ == "__main__":
    main()
