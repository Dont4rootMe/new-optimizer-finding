"""Very small JSON-lines IPC helpers for Unix-domain brokers."""

from __future__ import annotations

import json
import socket
from typing import Any, BinaryIO


def write_json_line(stream: BinaryIO, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"
    stream.write(data)
    stream.flush()


def read_json_line(stream: BinaryIO) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise EOFError("Broker IPC stream closed before a JSON payload was received.")
    decoded = json.loads(line.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Broker IPC expected a JSON object payload.")
    return decoded


def send_ipc_message(socket_path: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_sec)
        sock.connect(socket_path)
        with sock.makefile("rwb") as stream:
            write_json_line(stream, payload)
            return read_json_line(stream)
