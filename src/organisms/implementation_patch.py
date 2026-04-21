"""Section-aligned implementation patch compilation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.organisms.genetic_code_format import ParsedCoreGeneSection, parse_genetic_code_text

_FULL_MODE = "FULL"
_PATCH_MODE = "PATCH"
_START_MARKER_RE = re.compile(r"^[ \t]*# === REGION: ([A-Z][A-Z0-9_]*) ===[ \t]*$")
_END_MARKER_RE = re.compile(r"^[ \t]*# === END_REGION: ([A-Z][A-Z0-9_]*) ===[ \t]*$")
_SECTION_HINT_RE = re.compile(r"^[ \t]*# SECTION: ([A-Z][A-Z0-9_]*)[ \t]*$", re.MULTILINE)
_PATCH_NAMED_START_RE = re.compile(r"^REGION(?::[ \t]*|[ \t]+)([A-Z][A-Z0-9_]*)$")
_PATCH_NAMED_END_RE = re.compile(r"^END_REGION(?::[ \t]*|[ \t]+)([A-Z][A-Z0-9_]*)$")
_PATCH_SCAFFOLD_END_RE = re.compile(r"^[ \t]*# === END_REGION: ([A-Z][A-Z0-9_]*)(?: ===)?[ \t]*$")


@dataclass(frozen=True)
class ImplementationScaffoldRegion:
    name: str
    start_marker: str
    end_marker: str
    body: str


@dataclass(frozen=True)
class ParsedImplementationPatch:
    compilation_mode: Literal["FULL", "PATCH"]
    region_bodies: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ImplementationCompilationPlan:
    strategy: str
    compilation_mode: Literal["FULL", "PATCH"] | None
    changed_sections: tuple[str, ...]
    maternal_base_required: bool


@dataclass(frozen=True)
class _RegionSpan:
    name: str
    start_marker: str
    end_marker: str
    body: str
    body_start: int
    body_end: int
    marker_indent: str


def parse_implementation_scaffold(
    text: str,
    *,
    expected_region_names: tuple[str, ...],
) -> tuple[ImplementationScaffoldRegion, ...]:
    """Parse and validate the canonical section-aligned implementation scaffold."""

    spans = _parse_region_spans(text)
    _validate_region_order(spans, expected_region_names)
    return tuple(
        ImplementationScaffoldRegion(
            name=span.name,
            start_marker=span.start_marker,
            end_marker=span.end_marker,
            body=span.body,
        )
        for span in spans
    )


def resolve_implementation_region_order(
    text: str,
    *,
    expected_section_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return scaffold region order while validating it matches the expected section set."""

    spans = _parse_region_spans(text)
    if spans:
        actual_names = tuple(span.name for span in spans)
        _validate_region_name_set(actual_names, expected_section_names)
        return actual_names

    hinted_names = _parse_section_hints(text)
    if hinted_names:
        _validate_region_name_set(hinted_names, expected_section_names)
        return hinted_names

    raise ValueError("Implementation template does not declare canonical section order.")


def order_changed_sections_by_region_order(
    changed_sections: tuple[str, ...],
    *,
    region_order: tuple[str, ...],
) -> tuple[str, ...]:
    """Reorder changed section names to match scaffold region order."""

    changed_set = set(changed_sections)
    unknown = sorted(changed_set.difference(region_order))
    if unknown:
        raise ValueError(
            "Changed sections contain names not present in the implementation scaffold: "
            + ", ".join(unknown)
        )
    return tuple(name for name in region_order if name in changed_set)


def extract_region_bodies_from_source(
    source_text: str,
    *,
    expected_region_names: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Extract raw region bodies from a scaffold-compatible implementation source."""

    spans = _parse_region_spans(source_text)
    _validate_region_order(spans, expected_region_names)
    return tuple((span.name, span.body) for span in spans)


def compute_changed_genome_sections(
    maternal_genetic_code_text: str,
    child_genetic_code_text: str,
    *,
    expected_section_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return structurally changed section names in canonical order."""

    maternal = parse_genetic_code_text(
        maternal_genetic_code_text,
        expected_section_names=expected_section_names,
    )
    child = parse_genetic_code_text(
        child_genetic_code_text,
        expected_section_names=expected_section_names,
    )
    if maternal.format_kind != "sectioned" or child.format_kind != "sectioned":
        raise ValueError("Changed-section computation requires sectioned genetic-code artifacts.")

    maternal_sections = _normalized_section_map(maternal.core_gene_sections or ())
    child_sections = _normalized_section_map(child.core_gene_sections or ())
    changed = [
        name
        for name in expected_section_names
        if maternal_sections.get(name) != child_sections.get(name)
    ]
    return tuple(changed)


def parse_implementation_patch_response(
    text: str,
    *,
    expected_mode: str,
    expected_region_names: tuple[str, ...],
) -> ParsedImplementationPatch:
    """Parse a strict implementation region-patch artifact."""

    normalized_mode = str(expected_mode).strip()
    if normalized_mode not in {_FULL_MODE, _PATCH_MODE}:
        raise ValueError("expected_mode must be FULL or PATCH.")

    mode, region_bodies = _parse_patch_sections(text)
    if mode != normalized_mode:
        raise ValueError(f"Implementation patch mode must be {normalized_mode}, got {mode}.")

    actual_region_names = tuple(name for name, _body in region_bodies)
    if actual_region_names != expected_region_names:
        expected = ", ".join(expected_region_names) or "none"
        actual = ", ".join(actual_region_names) or "none"
        raise ValueError(
            "Implementation patch regions must match the expected changed regions exactly "
            f"in name, count, and order; expected [{expected}], got [{actual}]."
        )
    return ParsedImplementationPatch(
        compilation_mode=mode,  # type: ignore[arg-type]
        region_bodies=tuple(region_bodies),
    )


def assemble_implementation_from_patch(
    *,
    scaffold_text: str,
    patch: ParsedImplementationPatch,
    expected_region_names: tuple[str, ...],
    base_source_text: str | None = None,
) -> str:
    """Assemble final implementation source from scaffold or maternal base plus patch bodies."""

    parse_implementation_scaffold(
        scaffold_text,
        expected_region_names=expected_region_names,
    )
    patch_bodies = dict(patch.region_bodies)

    if patch.compilation_mode == _FULL_MODE:
        if tuple(patch_bodies.keys()) != expected_region_names:
            raise ValueError("FULL implementation patches must include all canonical regions.")
        return _replace_region_bodies(scaffold_text, patch_bodies, expected_region_names=expected_region_names)

    if patch.compilation_mode == _PATCH_MODE:
        if base_source_text is None:
            raise ValueError("PATCH implementation assembly requires base_source_text.")
        extract_region_bodies_from_source(
            base_source_text,
            expected_region_names=expected_region_names,
        )
        unexpected = [name for name in patch_bodies if name not in expected_region_names]
        if unexpected:
            raise ValueError(f"PATCH implementation contains unexpected region(s): {', '.join(unexpected)}.")
        return _replace_region_bodies(
            base_source_text,
            patch_bodies,
            expected_region_names=expected_region_names,
        )

    raise ValueError(f"Unsupported compilation mode {patch.compilation_mode!r}.")


def _normalized_section_map(
    sections: tuple[ParsedCoreGeneSection, ...],
) -> dict[str, tuple[str, ...]]:
    return {
        section.name: tuple(_normalize_entry(entry.text) for entry in section.entries)
        for section in sections
    }


def _normalize_entry(text: str) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def _parse_region_spans(text: str) -> tuple[_RegionSpan, ...]:
    source = str(text)
    spans: list[_RegionSpan] = []
    active_name: str | None = None
    active_start_marker: str | None = None
    active_marker_indent = ""
    active_body_start: int | None = None
    offset = 0

    for line_number, line in enumerate(source.splitlines(keepends=True), start=1):
        line_without_newline = line[:-1] if line.endswith("\n") else line
        if line_without_newline.endswith("\r"):
            line_without_newline = line_without_newline[:-1]
        stripped_line = line_without_newline.strip()
        start_match = _START_MARKER_RE.match(line_without_newline)
        end_match = _END_MARKER_RE.match(line_without_newline)

        if start_match is not None:
            if active_name is not None:
                raise ValueError(f"Nested region marker at line {line_number}: {stripped_line!r}")
            active_name = start_match.group(1)
            active_start_marker = stripped_line
            active_marker_indent = line_without_newline[
                : len(line_without_newline) - len(line_without_newline.lstrip(" \t"))
            ]
            active_body_start = offset + len(line)
        elif end_match is not None:
            end_name = end_match.group(1)
            if active_name is None or active_body_start is None or active_start_marker is None:
                raise ValueError(f"End region marker without matching start at line {line_number}: {stripped_line!r}")
            if end_name != active_name:
                raise ValueError(
                    f"Mismatched region end marker at line {line_number}: expected {active_name}, got {end_name}."
                )
            spans.append(
                _RegionSpan(
                    name=active_name,
                    start_marker=active_start_marker,
                    end_marker=stripped_line,
                    body=source[active_body_start:offset],
                    body_start=active_body_start,
                    body_end=offset,
                    marker_indent=active_marker_indent,
                )
            )
            active_name = None
            active_start_marker = None
            active_marker_indent = ""
            active_body_start = None
        elif "=== REGION:" in line or "=== END_REGION:" in line:
            raise ValueError(f"Malformed region marker at line {line_number}: {stripped_line!r}")
        offset += len(line)

    if active_name is not None:
        raise ValueError(f"Region {active_name} is missing its end marker.")
    return tuple(spans)


def _parse_section_hints(text: str) -> tuple[str, ...]:
    hinted_names = [
        match.group(1)
        for match in _SECTION_HINT_RE.finditer(str(text))
    ]
    return tuple(hinted_names)


def _validate_region_order(spans: tuple[_RegionSpan, ...], expected_region_names: tuple[str, ...]) -> None:
    actual_names = tuple(span.name for span in spans)
    _validate_region_name_set(actual_names, expected_region_names)
    if actual_names != expected_region_names:
        expected = ", ".join(expected_region_names)
        actual = ", ".join(actual_names)
        raise ValueError(
            "Implementation scaffold regions must match the expected regions exactly "
            f"in name, count, and order; expected [{expected}], got [{actual}]."
        )


def _validate_region_name_set(actual_names: tuple[str, ...], expected_section_names: tuple[str, ...]) -> None:
    if len(set(actual_names)) != len(actual_names):
        duplicates = sorted({name for name in actual_names if actual_names.count(name) > 1})
        raise ValueError(f"Duplicate implementation region(s): {', '.join(duplicates)}.")

    actual_set = set(actual_names)
    expected_set = set(expected_section_names)
    if actual_set != expected_set or len(actual_names) != len(expected_section_names):
        missing = sorted(expected_set.difference(actual_set))
        unexpected = sorted(actual_set.difference(expected_set))
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        detail_text = "; ".join(details) if details else "region name mismatch"
        raise ValueError(
            "Implementation scaffold regions must match the expected canonical section names exactly; "
            + detail_text
        )


def _parse_patch_sections(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    source = str(text)
    lines = source.splitlines(keepends=True)
    offset = 0
    mode: str | None = None
    mode_seen = False
    active_region: str | None = None
    active_body_start: int | None = None
    regions: list[tuple[str, str]] = []

    for line_number, line in enumerate(lines, start=1):
        line_without_newline = line[:-1] if line.endswith("\n") else line
        if line_without_newline.endswith("\r"):
            line_without_newline = line_without_newline[:-1]

        if line_without_newline.startswith("## "):
            header = line_without_newline[3:].strip()
            mode_inline = _parse_inline_compilation_mode(header)
            if header == "COMPILATION_MODE" or mode_inline is not None:
                if mode_seen:
                    raise ValueError("Implementation patch contains duplicate COMPILATION_MODE section.")
                if active_region is not None:
                    raise ValueError("COMPILATION_MODE cannot appear inside a region body.")
                mode_seen = True
                if mode_inline is None:
                    mode_start = offset + len(line)
                    mode_end = _find_next_header_offset(source, lines, line_number, mode_start)
                    mode = source[mode_start:mode_end].strip()
                else:
                    mode = mode_inline
                if mode not in {_FULL_MODE, _PATCH_MODE}:
                    raise ValueError("Implementation patch COMPILATION_MODE must be FULL or PATCH.")
            elif (region_name := _parse_patch_region_name(header)) is not None:
                if not mode_seen:
                    raise ValueError("Implementation patch must start with ## COMPILATION_MODE before regions.")
                if active_region is not None:
                    raise ValueError(f"Region {active_region} is missing ## END_REGION before line {line_number}.")
                active_region = region_name
                active_body_start = offset + len(line)
            else:
                end_region_name = _parse_patch_end_region_name(header, line_without_newline)
                if end_region_name is None:
                    raise ValueError(f"Unexpected implementation patch section heading at line {line_number}: {header!r}")
                if active_region is None or active_body_start is None:
                    raise ValueError(f"Unexpected ## END_REGION at line {line_number}.")
                if end_region_name and end_region_name != active_region:
                    raise ValueError(
                        f"Mismatched ## END_REGION at line {line_number}: expected {active_region}, got {end_region_name}."
                    )
                regions.append((active_region, source[active_body_start:offset]))
                active_region = None
                active_body_start = None
        elif line_without_newline.startswith("##"):
            raise ValueError(f"Malformed implementation patch heading at line {line_number}: {line_without_newline!r}")
        elif active_region is not None:
            end_region_name = _parse_patch_end_region_name("", line_without_newline)
            if end_region_name is not None:
                if active_body_start is None:
                    raise ValueError(f"Unexpected implementation patch end marker at line {line_number}.")
                if end_region_name and end_region_name != active_region:
                    raise ValueError(
                        f"Mismatched implementation patch end marker at line {line_number}: "
                        f"expected {active_region}, got {end_region_name}."
                    )
                regions.append((active_region, source[active_body_start:offset]))
                active_region = None
                active_body_start = None
        elif not mode_seen and line_without_newline.strip():
            raise ValueError("Implementation patch contains text before ## COMPILATION_MODE.")

        offset += len(line)

    if not mode_seen or mode is None:
        raise ValueError("Implementation patch is missing ## COMPILATION_MODE.")
    if active_region is not None:
        raise ValueError(f"Region {active_region} is missing ## END_REGION.")
    actual_names = tuple(name for name, _body in regions)
    if len(set(actual_names)) != len(actual_names):
        duplicates = sorted({name for name in actual_names if actual_names.count(name) > 1})
        raise ValueError(f"Implementation patch contains duplicate region(s): {', '.join(duplicates)}.")
    return mode, tuple(regions)


def _parse_inline_compilation_mode(header: str) -> str | None:
    for prefix in ("COMPILATION_MODE ", "COMPILATION_MODE:"):
        if header.startswith(prefix):
            mode = header[len(prefix) :].strip()
            return mode or None
    return None


def _parse_patch_region_name(header: str) -> str | None:
    match = _PATCH_NAMED_START_RE.match(header)
    if match is None:
        return None
    return match.group(1)


def _parse_patch_end_region_name(header: str, line_without_newline: str) -> str | None:
    if header == "END_REGION":
        return ""
    named_end_match = _PATCH_NAMED_END_RE.match(header)
    if named_end_match is not None:
        return named_end_match.group(1)
    scaffold_end_match = _PATCH_SCAFFOLD_END_RE.match(line_without_newline)
    if scaffold_end_match is not None:
        return scaffold_end_match.group(1)
    return None


def _find_next_header_offset(source: str, lines: list[str], current_line_number: int, fallback: int) -> int:
    offset = 0
    for index, line in enumerate(lines, start=1):
        if index <= current_line_number:
            offset += len(line)
            continue
        line_without_newline = line[:-1] if line.endswith("\n") else line
        if line_without_newline.endswith("\r"):
            line_without_newline = line_without_newline[:-1]
        if line_without_newline.startswith("## "):
            return offset
        offset += len(line)
    return len(source) if lines else fallback


def _replace_region_bodies(
    source_text: str,
    region_bodies: dict[str, str],
    *,
    expected_region_names: tuple[str, ...],
) -> str:
    spans = _parse_region_spans(source_text)
    _validate_region_order(spans, expected_region_names)
    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(source_text[cursor:span.body_start])
        if span.name in region_bodies:
            parts.append(_align_patch_body_indentation(region_bodies[span.name], span.marker_indent))
        else:
            parts.append(span.body)
        cursor = span.body_end
    parts.append(source_text[cursor:])
    return "".join(parts)


def _align_patch_body_indentation(body: str, marker_indent: str) -> str:
    if not marker_indent or not str(body).strip():
        return body
    lines = str(body).splitlines(keepends=True)
    needs_indent = False
    for line in lines:
        line_without_newline = line[:-1] if line.endswith("\n") else line
        if line_without_newline.endswith("\r"):
            line_without_newline = line_without_newline[:-1]
        if not line_without_newline.strip():
            continue
        if not line_without_newline.startswith((" ", "\t")):
            needs_indent = True
            break
    if not needs_indent:
        return body
    return "".join(marker_indent + line if line.strip() else line for line in lines)
