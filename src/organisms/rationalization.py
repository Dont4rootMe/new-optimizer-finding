"""Parser and formatter for the Step-1 design-rationalization artifact.

The evolution loop's design stage is a two-step pipeline:

  Step 1 — *rationalization*:
      free-text but section-anchored "design plan" that the LLM produces
      before committing to any structured genetic code. Six `## ` top-level
      sections (see ``REQUIRED_SECTIONS``) carry the diagnosis, regime
      analysis, and the concrete WHAT-to-REMOVE / WHAT-to-ADD directive
      Step 2 will follow.

  Step 2 — *formalization*:
      the existing sectioned `## CORE_GENES ... ## CHANGE_DESCRIPTION`
      artifact. Section names deliberately do NOT collide with Step 1's
      section names so the strict CORE_GENES parser can never mis-anchor.

Step 1's output is consumed by another small local model (gemma/qwen) and
so must be permissive: missing sections degrade to ``"(not produced)"``
rather than blocking. The only hard failure is a fully empty / wholly
unparseable response — in which case the caller decides whether to retry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


REQUIRED_SECTIONS: tuple[str, ...] = (
    "SCORE_BEARING_CORE",
    "LINEAGE_REGIME_DIAGNOSIS",
    "WEAKNESS_HYPOTHESIS",
    "WHAT_TO_REMOVE",
    "WHAT_TO_ADD_OR_INVENT",
    "CHILD_DIRECTION",
)

# Step 1 uses ~~~python ... ~~~ (tilde) fences for pseudocode hints inside
# WHAT_TO_ADD_OR_INVENT. They deliberately differ from Step 2's backtick
# fences so a malformed Step 1 cannot contaminate Step 2's strict CORE_GENES
# parser when the formatted rationale is interpolated into the user prompt.
_HEADER_RE = re.compile(r"^##\s+([A-Z][A-Z0-9_]+)\s*$", re.MULTILINE)
_MISSING_SECTION_BODY = "(not produced)"


@dataclass(frozen=True)
class RationalizationOutput:
    """Parsed Step-1 rationale.

    ``sections`` is keyed by every entry in ``REQUIRED_SECTIONS`` — missing
    sections receive ``_MISSING_SECTION_BODY`` so consumers never have to
    branch on ``None``. ``raw_text`` is the original LLM response, preserved
    for forensic dumps.
    """

    raw_text: str
    sections: dict[str, str] = field(default_factory=dict)

    def section(self, name: str) -> str:
        return self.sections.get(name, _MISSING_SECTION_BODY)

    @property
    def has_actionable_directive(self) -> bool:
        """At least one of REMOVE / ADD_OR_INVENT / CHILD_DIRECTION must be non-empty.

        A rationale that produces only PARENT_DIGEST + LINEAGE_REGIME has done
        diagnosis but no planning — Step 2 cannot act on it. Caller may treat
        an output without an actionable directive as a retry-eligible failure.
        """

        for key in ("WHAT_TO_REMOVE", "WHAT_TO_ADD_OR_INVENT", "CHILD_DIRECTION"):
            body = self.sections.get(key, _MISSING_SECTION_BODY).strip()
            if body and body != _MISSING_SECTION_BODY:
                return True
        return False


def parse_rationalization_response(text: str) -> RationalizationOutput:
    """Parse the LLM's Step-1 response into a section dictionary.

    Tolerant: missing or out-of-order sections produce ``"(not produced)"``
    bodies rather than raising. The only failure mode is an entirely empty
    or whitespace-only input — callers handle that explicitly.

    Sections outside ``REQUIRED_SECTIONS`` are recorded but not promoted to
    the canonical dict (so Step 2 sees a stable shape regardless of LLM
    drift).
    """

    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if not stripped:
        return RationalizationOutput(
            raw_text=raw,
            sections={name: _MISSING_SECTION_BODY for name in REQUIRED_SECTIONS},
        )

    chunks: dict[str, str] = {}
    matches = list(_HEADER_RE.finditer(stripped))
    if not matches:
        # The LLM emitted prose without our header convention. Keep the raw
        # text but mark all required sections as missing; callers may decide
        # to feed the raw text through verbatim or to retry.
        return RationalizationOutput(
            raw_text=raw,
            sections={name: _MISSING_SECTION_BODY for name in REQUIRED_SECTIONS},
        )

    for index, match in enumerate(matches):
        name = match.group(1)
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        body = stripped[body_start:body_end].strip()
        # If the model repeats a section header (rare but observed), keep the
        # last occurrence — it's usually the corrected one after the LLM
        # realized it drifted.
        chunks[name] = body

    sections = {
        name: chunks.get(name, _MISSING_SECTION_BODY) or _MISSING_SECTION_BODY
        for name in REQUIRED_SECTIONS
    }
    # Preserve any unexpected extra sections in case forensic review needs them.
    for name, body in chunks.items():
        if name not in REQUIRED_SECTIONS:
            sections[f"_extra_{name}"] = body
    return RationalizationOutput(raw_text=raw, sections=sections)


def format_rationalization_for_step2(output: RationalizationOutput) -> str:
    """Render the parsed rationale back into the canonical form for Step 2.

    Even when the LLM emits sloppy markdown (extra blank lines, mixed
    capitalization, out-of-order sections), Step 2's user prompt receives a
    normalized rendering: the six required headers in the canonical order,
    each followed by its body. This is what gets substituted into the
    ``{rationalization}`` placeholder of the Step 2 user prompt.
    """

    if not output.sections:
        return _MISSING_SECTION_BODY
    blocks: list[str] = []
    for name in REQUIRED_SECTIONS:
        body = output.sections.get(name, _MISSING_SECTION_BODY).strip() or _MISSING_SECTION_BODY
        blocks.append(f"## {name}\n{body}")
    return "\n\n".join(blocks)


def rationalization_summary(output: RationalizationOutput) -> dict[str, Any]:
    """Compact JSON-friendly summary for ``llm_request.json`` / dumps."""

    summary: dict[str, Any] = {
        "has_actionable_directive": output.has_actionable_directive,
        "sections_present": [
            name
            for name in REQUIRED_SECTIONS
            if output.sections.get(name, _MISSING_SECTION_BODY).strip() not in {"", _MISSING_SECTION_BODY}
        ],
    }
    summary["section_bodies"] = {
        name: output.sections.get(name, _MISSING_SECTION_BODY)
        for name in REQUIRED_SECTIONS
    }
    return summary
