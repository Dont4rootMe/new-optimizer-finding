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


def _positional_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return positional argument names, including positional-only args."""

    args = list(node.args.posonlyargs) + list(node.args.args)
    return [arg.arg for arg in args]


def _match_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    expected: list[str],
) -> bool:
    return _positional_arg_names(node) == expected


def _find_function(
    body: list[ast.stmt],
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for child in body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
            return child
    return None


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

    build_optimizer = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "build_optimizer"
        ),
        None,
    )
    if build_optimizer is None:
        return False, "Missing required function: build_optimizer(model, max_steps)"
    if not _match_signature(build_optimizer, ["model", "max_steps"]):
        return (
            False,
            "build_optimizer must accept exactly (model, max_steps); "
            "legacy build_optimizer(cfg) is not supported",
        )

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        init_method = _find_function(node.body, "__init__")
        step_method = _find_function(node.body, "step")
        zero_grad_method = _find_function(node.body, "zero_grad")
        if init_method is None or step_method is None or zero_grad_method is None:
            continue

        if not _match_signature(init_method, ["self", "model", "max_steps"]):
            return False, (
                f"Controller class '{node.name}' must define "
                "__init__(self, model, max_steps)"
            )
        if not _match_signature(step_method, ["self", "weights", "grads", "activations", "step_fn"]):
            return False, (
                f"Controller class '{node.name}' must define "
                "step(self, weights, grads, activations, step_fn)"
            )
        if not _match_signature(zero_grad_method, ["self", "set_to_none"]):
            return False, (
                f"Controller class '{node.name}' must define "
                "zero_grad(self, set_to_none=True)"
            )
        return True, None

    return False, "Missing controller class with __init__/step/zero_grad"


def parse_llm_response(text: str) -> dict[str, str]:
    """Parse structured LLM response into sections.

    The parser is generic on purpose so explicit legacy callers can still parse
    old section names, while the canonical organism validator decides which
    sections are required for the organism-first path.
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
