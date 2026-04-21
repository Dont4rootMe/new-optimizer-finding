"""Structured parsing for canonical organism genetic-code artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TOP_LEVEL_SECTION_NAMES = (
    "CORE_GENES",
    "INTERACTION_NOTES",
    "COMPUTE_NOTES",
    "CHANGE_DESCRIPTION",
)
_SCHEMA_HEADER_RE = re.compile(r"^# ([A-Z][A-Z0-9_]*)$")
_TOP_LEVEL_HEADER_RE = re.compile(r"^## ([A-Z_]+)$")
_SUBSECTION_HEADER_RE = re.compile(r"^### ([A-Z][A-Z0-9_]*)$")
_OPTIONAL_CODE_SKETCH_SECTION = "OPTIONAL_CODE_SKETCH"


@dataclass(frozen=True)
class GenomeSchemaSection:
    name: str
    description: str


@dataclass(frozen=True)
class ParsedGeneEntry:
    text: str


@dataclass(frozen=True)
class ParsedCoreGeneSection:
    name: str
    entries: tuple[ParsedGeneEntry, ...]


@dataclass(frozen=True)
class ParsedGeneticCode:
    format_kind: str
    core_gene_sections: tuple[ParsedCoreGeneSection, ...] | None
    legacy_core_genes: tuple[str, ...] | None
    interaction_notes: str
    compute_notes: str
    change_description: str


def parse_genome_schema_text(text: str) -> tuple[GenomeSchemaSection, ...]:
    """Parse the plain-text genome schema artifact into ordered sections."""

    source = str(text)
    if not source.strip():
        raise ValueError("Genome schema is empty.")

    sections: list[GenomeSchemaSection] = []
    seen_names: set[str] = set()
    current_name: str | None = None
    current_body: list[str] = []

    def flush_current() -> None:
        if current_name is None:
            return
        sections.append(
            GenomeSchemaSection(
                name=current_name,
                description="\n".join(current_body).strip(),
            )
        )

    for line_number, line in enumerate(source.splitlines(), start=1):
        if line.startswith("#"):
            match = _SCHEMA_HEADER_RE.match(line)
            if match is None:
                raise ValueError(f"Malformed genome schema header at line {line_number}: {line!r}")
            flush_current()
            name = match.group(1)
            if not name:
                raise ValueError(f"Genome schema header at line {line_number} has an empty section name.")
            if name in seen_names:
                raise ValueError(f"Genome schema contains duplicate section name {name!r}.")
            seen_names.add(name)
            current_name = name
            current_body = []
            continue

        if current_name is None:
            if line.strip():
                raise ValueError("Genome schema contains body text before the first section header.")
            continue
        current_body.append(line)

    flush_current()
    if not sections:
        raise ValueError("Genome schema is empty.")
    return tuple(sections)


def load_genome_schema(path: str) -> tuple[GenomeSchemaSection, ...]:
    """Load and parse a genome schema artifact from disk."""

    return parse_genome_schema_text(Path(path).read_text(encoding="utf-8"))


def parse_section_issue_list(
    text: str,
    *,
    expected_section_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Parse a strict comma-separated section issue list or NONE."""

    raw = str(text).strip()
    if raw == "NONE":
        return ()
    if not raw:
        raise ValueError("SECTIONS_AT_ISSUE must be NONE or a comma-separated section list.")
    if "\n" in raw or raw.startswith(("- ", "* ", "1. ")):
        raise ValueError("SECTIONS_AT_ISSUE must not use bullets, numbering, or prose blocks.")

    names = tuple(part.strip() for part in raw.split(","))
    if any(not name for name in names):
        raise ValueError("SECTIONS_AT_ISSUE contains an empty section name.")

    seen: set[str] = set()
    if expected_section_names is not None:
        expected_index = {name: index for index, name in enumerate(expected_section_names)}
        for name in names:
            if name not in expected_index:
                raise ValueError(f"SECTIONS_AT_ISSUE contains unknown section name {name!r}.")
            if name in seen:
                raise ValueError(f"SECTIONS_AT_ISSUE contains duplicate section name {name!r}.")
            seen.add(name)
        return tuple(sorted(names, key=lambda name: expected_index[name]))

    for name in names:
        if _SCHEMA_HEADER_RE.match(f"# {name}") is None:
            raise ValueError(f"SECTIONS_AT_ISSUE contains unknown section name {name!r}.")
        if name in seen:
            raise ValueError(f"SECTIONS_AT_ISSUE contains duplicate section name {name!r}.")
        seen.add(name)
    return names


def detect_genetic_code_format(text: str) -> str:
    """Detect whether `CORE_GENES` is legacy-flat or sectioned."""

    source = str(text)
    core_body = _extract_core_genes_body_for_detection(source)
    if core_body is not None and any(line.startswith("### ") for line in core_body.splitlines()):
        return "sectioned"
    return "legacy_flat"


def parse_genetic_code_text(
    text: str,
    *,
    expected_section_names: tuple[str, ...] | None = None,
) -> ParsedGeneticCode:
    """Parse legacy-flat or sectioned genetic-code markdown."""

    source = str(text)
    format_kind = detect_genetic_code_format(source)
    top_sections = _parse_top_level_sections(source)

    required = TOP_LEVEL_SECTION_NAMES
    missing = [name for name in required if name not in top_sections]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Canonical genetic code text is malformed; missing required sections: {joined}")

    if format_kind == "sectioned":
        core_gene_sections = _parse_sectioned_core_genes(
            top_sections["CORE_GENES"],
            expected_section_names=expected_section_names,
        )
        return ParsedGeneticCode(
            format_kind="sectioned",
            core_gene_sections=core_gene_sections,
            legacy_core_genes=None,
            interaction_notes=top_sections["INTERACTION_NOTES"],
            compute_notes=top_sections["COMPUTE_NOTES"],
            change_description=top_sections["CHANGE_DESCRIPTION"],
        )

    # Legacy flat CORE_GENES support is transitional until prompt migration is
    # complete; section assignment from legacy genes is intentionally not attempted.
    return ParsedGeneticCode(
        format_kind="legacy_flat",
        core_gene_sections=None,
        legacy_core_genes=_parse_legacy_core_genes(top_sections["CORE_GENES"]),
        interaction_notes=top_sections["INTERACTION_NOTES"],
        compute_notes=top_sections["COMPUTE_NOTES"],
        change_description=top_sections["CHANGE_DESCRIPTION"],
    )


def _extract_core_genes_body_for_detection(text: str) -> str | None:
    lines = text.splitlines()
    core_start: int | None = None
    for idx, line in enumerate(lines):
        if line == "## CORE_GENES":
            core_start = idx + 1
            break
    if core_start is None:
        return None

    core_lines: list[str] = []
    for line in lines[core_start:]:
        if _TOP_LEVEL_HEADER_RE.match(line):
            break
        core_lines.append(line)
    return "\n".join(core_lines)


def _parse_top_level_sections(text: str) -> dict[str, str]:
    allowed = set(TOP_LEVEL_SECTION_NAMES)
    sections: dict[str, str] = {}
    current_name: str | None = None
    current_body: list[str] = []

    def flush_current() -> None:
        if current_name is None:
            return
        sections[current_name] = "\n".join(current_body).strip()

    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _TOP_LEVEL_HEADER_RE.match(line)
        if match is not None:
            name = match.group(1)
            if name not in allowed:
                raise ValueError(f"Unexpected top-level genetic-code section {name!r} at line {line_number}.")
            if name in sections or name == current_name:
                raise ValueError(f"Duplicate top-level genetic-code section {name!r}.")
            flush_current()
            current_name = name
            current_body = []
            continue

        if line.startswith("##") and not line.startswith("###"):
            raise ValueError(f"Malformed top-level genetic-code heading at line {line_number}: {line!r}")

        if current_name is None:
            if line.strip():
                raise ValueError("Genetic code contains body text before the first top-level section.")
            continue
        current_body.append(line)

    flush_current()
    return sections


def _parse_sectioned_core_genes(
    core_text: str,
    *,
    expected_section_names: tuple[str, ...] | None,
) -> tuple[ParsedCoreGeneSection, ...]:
    sections: list[ParsedCoreGeneSection] = []
    seen_names: set[str] = set()
    current_name: str | None = None
    current_entries: list[ParsedGeneEntry] = []
    current_entry_lines: list[str] | None = None
    inside_optional_code_fence = False

    def flush_entry() -> None:
        nonlocal current_entry_lines
        if current_entry_lines is None:
            return
        text = "\n".join(current_entry_lines).strip()
        if not text:
            raise ValueError(f"CORE_GENES subsection {current_name} contains an empty gene entry.")
        current_entries.append(ParsedGeneEntry(text=text))
        current_entry_lines = None

    def flush_section() -> None:
        if current_name is None:
            return
        flush_entry()
        sections.append(
            ParsedCoreGeneSection(
                name=current_name,
                entries=tuple(current_entries),
            )
        )

    for line_number, line in enumerate(core_text.splitlines(), start=1):
        if inside_optional_code_fence:
            if current_entry_lines is None:
                raise ValueError("OPTIONAL_CODE_SKETCH fenced block lost its current entry state.")
            current_entry_lines.append(line.rstrip())
            if line.strip().startswith("```"):
                inside_optional_code_fence = False
                flush_entry()
            continue

        if line.startswith("### "):
            match = _SUBSECTION_HEADER_RE.match(line)
            if match is None:
                raise ValueError(f"Malformed CORE_GENES subsection heading at line {line_number}: {line!r}")
            flush_section()
            name = match.group(1)
            if name in seen_names:
                raise ValueError(f"Duplicate CORE_GENES subsection {name!r}.")
            seen_names.add(name)
            current_name = name
            current_entries = []
            current_entry_lines = None
            continue

        if line.startswith("#"):
            raise ValueError(f"Nested or malformed CORE_GENES heading at line {line_number}: {line!r}")

        if current_name is None:
            if line.strip():
                raise ValueError("Sectioned CORE_GENES content appears before the first subsection heading.")
            continue

        if not line.strip():
            continue
        if current_name == _OPTIONAL_CODE_SKETCH_SECTION and line.strip().startswith("```"):
            flush_entry()
            current_entry_lines = [line.rstrip()]
            if line.strip().count("```") >= 2:
                flush_entry()
            else:
                inside_optional_code_fence = True
            continue
        if line.startswith("- "):
            flush_entry()
            current_entry_lines = [line[2:].strip()]
            continue
        if line.startswith("  "):
            if current_entry_lines is None:
                raise ValueError(
                    f"CORE_GENES subsection {current_name} has a continuation line without a preceding bullet."
                )
            current_entry_lines.append(line[2:].rstrip())
            continue

        raise ValueError(
            f"CORE_GENES subsection {current_name} contains non-bullet text at line {line_number}: {line!r}"
        )

    if inside_optional_code_fence:
        raise ValueError(f"CORE_GENES subsection {_OPTIONAL_CODE_SKETCH_SECTION} has an unterminated fenced code block.")

    flush_section()
    if not sections:
        raise ValueError("Sectioned CORE_GENES must contain at least one subsection.")

    if expected_section_names is not None:
        _validate_expected_sectioned_core_genes(tuple(sections), expected_section_names)
    return tuple(sections)


def _validate_expected_sectioned_core_genes(
    sections: tuple[ParsedCoreGeneSection, ...],
    expected_section_names: tuple[str, ...],
) -> None:
    actual_names = tuple(section.name for section in sections)
    if actual_names != expected_section_names:
        expected = ", ".join(expected_section_names)
        actual = ", ".join(actual_names)
        raise ValueError(
            "Sectioned CORE_GENES subsections must match the genome schema exactly "
            f"in name, count, and order; expected [{expected}], got [{actual}]."
        )

    optional_section_name = expected_section_names[-1]
    for section in sections[:-1]:
        if not section.entries:
            raise ValueError(f"Required CORE_GENES subsection {section.name} must contain at least one gene entry.")
        if any(entry.text == "None." for entry in section.entries):
            raise ValueError(f"- None. is only valid inside {optional_section_name}.")

    optional = sections[-1]
    if not optional.entries:
        raise ValueError(f"{optional_section_name} must contain at least one gene entry or - None.")
    optional_none_entries = [entry for entry in optional.entries if entry.text == "None."]
    if optional_none_entries and len(optional.entries) != 1:
        raise ValueError(f"- None. must be the only entry when used inside {optional_section_name}.")


def _parse_legacy_core_genes(core_text: str) -> tuple[str, ...]:
    genes: list[str] = []
    current_entry_lines: list[str] | None = None

    def flush_entry() -> None:
        nonlocal current_entry_lines
        if current_entry_lines is None:
            return
        text = "\n".join(current_entry_lines).strip()
        if text:
            genes.append(text)
        current_entry_lines = None

    for line_number, line in enumerate(core_text.splitlines(), start=1):
        if not line.strip():
            continue
        if line.startswith("#"):
            raise ValueError(f"Legacy CORE_GENES contains a malformed heading at line {line_number}: {line!r}")
        if line.startswith("- "):
            flush_entry()
            current_entry_lines = [line[2:].strip()]
            continue
        if line.startswith("  ") and current_entry_lines is not None:
            current_entry_lines.append(line[2:].rstrip())
            continue
        raise ValueError(f"Legacy CORE_GENES contains non-bullet text at line {line_number}: {line!r}")

    flush_entry()
    if not genes:
        raise ValueError("Canonical genetic code text must contain at least one CORE_GENES bullet.")
    return tuple(genes)
