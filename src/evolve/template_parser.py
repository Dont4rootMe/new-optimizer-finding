"""Generic structured-response parsing helpers for evolve."""

from __future__ import annotations

import re


def parse_llm_response(text: str) -> dict[str, str]:
    """Parse a structured markdown response into `## SECTION` blocks."""

    result: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        header_match = re.match(r"^##\s+(\w+)\s*$", line.strip())
        if header_match:
            if current_key is not None:
                result[current_key] = "\n".join(current_lines).strip()
            current_key = header_match.group(1)
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        result[current_key] = "\n".join(current_lines).strip()

    return result
