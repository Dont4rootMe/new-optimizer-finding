"""Parser for optimizer template: extract editable sections, render, validate."""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import Any

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "optimizer_template.txt"

_EDITABLE_RE = re.compile(
    r"#\s*===\s*EDITABLE:\s*(\w+)\s*===\s*\n(.*?)#\s*===\s*END EDITABLE\s*===",
    re.DOTALL,
)

SECTION_NAMES = ("IMPORTS", "INIT_BODY", "STEP_BODY", "ZERO_GRAD_BODY")


def _load_template() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def extract_editable_sections(code: str) -> dict[str, str]:
    """Extract editable section contents from rendered optimizer code."""
    sections: dict[str, str] = {}
    for match in _EDITABLE_RE.finditer(code):
        name = match.group(1)
        content = match.group(2)
        sections[name] = content.rstrip("\n")
    return sections


def render_template(
    sections: dict[str, str],
    optimizer_name: str,
    class_name: str,
) -> str:
    """Render optimizer.py from template and editable sections.

    Section values should be raw code lines (properly indented for their context).
    """
    template = _load_template()

    # Default empty sections
    imports = sections.get("IMPORTS", "import math")
    init_body = sections.get("INIT_BODY", "        pass")
    step_body = sections.get("STEP_BODY", "        pass")
    zero_grad_body = sections.get("ZERO_GRAD_BODY", "        pass")

    rendered = template.format(
        imports=imports,
        init_body=init_body,
        step_body=step_body,
        zero_grad_body=zero_grad_body,
        optimizer_name=optimizer_name,
        class_name=class_name,
    )
    return rendered


def validate_rendered_code(code: str) -> tuple[bool, str | None]:
    """Validate rendered optimizer code: syntax + required structure."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    has_builder = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_optimizer"
        for node in ast.walk(tree)
    )
    if not has_builder:
        return False, "Missing required function: build_optimizer"

    has_controller_class = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        method_names = {
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if {"step", "zero_grad", "__init__"}.issubset(method_names):
            has_controller_class = True
            break

    if not has_controller_class:
        return False, "Missing controller class with __init__/step/zero_grad"

    return True, None


def parse_llm_response(text: str) -> dict[str, str]:
    """Parse structured LLM response into sections.

    Expected format:
    ## IDEA_DNA
    ...
    ## CHANGE_DESCRIPTION
    ...
    ## IMPORTS
    ...
    ## INIT_BODY
    ...
    ## STEP_BODY
    ...
    ## ZERO_GRAD_BODY
    ...
    """
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
