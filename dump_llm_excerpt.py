#!/usr/bin/env python3
"""Create a compact "excerpt" report for LLM traces across generations.

This is a condensed companion to dump_llm.py:
- focuses on key metadata and short snippets
- can scan an entire population folder (gen_*/island_*/org_*)

Usage:
  python dump_llm_excerpt.py <population_dir> <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable


STAGE_LIST_KEYS = (
    "design_attempts",
    "compatibility_checks",
    "novelty_checks",
    "repair_attempts",
)
STAGE_SINGLE_KEYS = ("design", "implementation", "repair")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            v = json.loads(path.read_text())
            return v if isinstance(v, dict) else {"_": v}
    except Exception as e:
        return {"_error": str(e)}
    return {}


def _one_line(s: Any, limit: int) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[: limit - 1] + "…"
    return s


def _pick_text(resp_stage: dict[str, Any]) -> str:
    return (
        resp_stage.get("text")
        or resp_stage.get("content_text")
        or resp_stage.get("parse_text")
        or ""
    )


def _iter_pairs(req: dict[str, Any], resp: dict[str, Any], key: str) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
    ra = req.get(key)
    rb = resp.get(key)
    if isinstance(ra, list) or isinstance(rb, list):
        ra_list = ra if isinstance(ra, list) else []
        rb_list = rb if isinstance(rb, list) else []
        n = max(len(ra_list), len(rb_list))
        for i in range(n):
            yield f"{key}[{i}]", (ra_list[i] if i < len(ra_list) else {}), (rb_list[i] if i < len(rb_list) else {})
    else:
        if key in req or key in resp:
            yield key, (ra if isinstance(ra, dict) else {}), (rb if isinstance(rb, dict) else {})


def dump_excerpt(org_dir: Path, out_dir: Path, gen_name: str) -> dict[str, Any]:
    island = org_dir.parent.name
    org = org_dir.name
    rel = f"{gen_name}/{island}/{org}"

    organism = _read_json(org_dir / "organism.json")
    req = _read_json(org_dir / "llm_request.json")
    resp = _read_json(org_dir / "llm_response.json")
    summary = _read_json(org_dir / "summary.json")
    rationalization = _read_json(org_dir / "llm_rationalization.json")

    out_file = out_dir / f"{gen_name}__{island}__{org}.md"

    pipeline_state = organism.get("pipeline_state")
    operator = organism.get("operator")
    organism_id = organism.get("organism_id")
    score = summary.get("score")

    route_id = req.get("route_id")
    provider = req.get("provider")
    model = req.get("provider_model_id")

    error_msg = organism.get("error_msg") or resp.get("error_msg") or req.get("error_msg")

    lines: list[str] = []
    lines.append(f"# {rel}\n")
    lines.append("## Meta\n")
    if organism_id:
        lines.append(f"- **organism_id**: `{organism_id}`")
    if pipeline_state:
        lines.append(f"- **pipeline_state**: `{pipeline_state}`")
    if operator:
        lines.append(f"- **operator**: `{operator}`")
    if score is not None:
        lines.append(f"- **score**: `{score}`")
    if provider or model or route_id:
        lines.append(
            "- **llm**: "
            + " ".join(
                [
                    f"provider=`{provider}`" if provider else "",
                    f"model=`{model}`" if model else "",
                    f"route=`{route_id}`" if route_id else "",
                ]
            ).strip()
        )
    if error_msg:
        lines.append(f"- **error**: {_one_line(error_msg, 400)}")

    lines.append("\n## Stages (excerpt)\n")
    stage_meta_keys = ("status", "verdict", "rejection_reason", "sections_at_issue", "changed_sections", "error_kind")

    # Step 1 of the two-step design pipeline. Lives in its own JSON file so
    # the design stage's overwrite of llm_request.json doesn't lose it.
    if rationalization:
        lines.append("### rationalization (Step 1)")
        rat_meta_parts: list[str] = []
        rat_status = rationalization.get("status")
        if rat_status:
            rat_meta_parts.append(f"status={rat_status}")
        rat_parsed = rationalization.get("parsed") or {}
        if "has_actionable_directive" in rat_parsed:
            rat_meta_parts.append(f"actionable={rat_parsed['has_actionable_directive']}")
        if rat_parsed.get("sections_present"):
            rat_meta_parts.append(f"sections={','.join(rat_parsed['sections_present'])}")
        if rat_meta_parts:
            lines.append(f"- **meta**: {_one_line('; '.join(rat_meta_parts), 600)}")
        rat_text = rationalization.get("text")
        if rat_text:
            lines.append(f"- **text**: {_one_line(rat_text, 700)}")
        lines.append("")

    for key in (*STAGE_LIST_KEYS, *STAGE_SINGLE_KEYS):
        for label, req_stage, resp_stage in _iter_pairs(req, resp, key):
            req_stage = req_stage or {}
            resp_stage = resp_stage or {}
            # Skip completely empty stages
            if not req_stage and not resp_stage:
                continue

            meta_parts: list[str] = []
            for mk in stage_meta_keys:
                v = resp_stage.get(mk, req_stage.get(mk))
                if v in (None, "", [], {}):
                    continue
                meta_parts.append(f"{mk}={_one_line(v, 180)}")

            text = _pick_text(resp_stage)
            thinking = resp_stage.get("thinking_text")
            err = resp_stage.get("error_msg") or req_stage.get("error_msg")

            lines.append(f"### {label}")
            if meta_parts:
                lines.append(f"- **meta**: {_one_line('; '.join(meta_parts), 600)}")
            if text:
                lines.append(f"- **text**: {_one_line(text, 700)}")
            if thinking:
                lines.append(f"- **thinking**: {_one_line(thinking, 350)}")
            if err:
                lines.append(f"- **stage_error**: {_one_line(err, 500)}")
            lines.append("")

    genetic_code_path = org_dir / "genetic_code.md"
    if genetic_code_path.exists():
        try:
            gc_text = genetic_code_path.read_text()
        except Exception as e:
            gc_text = f"(failed to read genetic_code.md: {e})"
        lines.append("## genetic_code.md\n")
        lines.append(gc_text.rstrip())
        lines.append("")

    out_file.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "rel": rel,
        "out": str(out_file),
        "pipeline_state": pipeline_state,
        "score": score,
        "route_id": route_id,
        "provider_model_id": model,
        "error_msg": error_msg,
    }


def main() -> None:
    population_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []

    gen_dirs = sorted(p for p in population_dir.iterdir() if p.is_dir() and p.name.startswith("gen_"))
    for gen_dir in gen_dirs:
        gen_name = gen_dir.name
        for island_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir() and p.name.startswith("island_")):
            for org_dir in sorted(p for p in island_dir.iterdir() if p.is_dir() and p.name.startswith("org_")):
                summaries.append(dump_excerpt(org_dir, out_dir, gen_name))

    idx = out_dir / "INDEX.md"
    lines: list[str] = [f"# LLM excerpt index for {population_dir}\n", f"Total organisms: {len(summaries)}\n"]

    cur_gen: str | None = None
    for s in summaries:
        gen = s["rel"].split("/", 1)[0]
        if gen != cur_gen:
            cur_gen = gen
            lines.append(f"\n## {gen}\n")
        score = s.get("score")
        score_str = f" score=`{score}`" if score is not None else ""
        err = s.get("error_msg")
        lines.append(
            f"- [{s['rel']}]({Path(s['out']).name}) — state=`{s.get('pipeline_state')}`"
            f"{score_str} route=`{s.get('route_id')}` model=`{s.get('provider_model_id')}`"
            + (f" error=`{_one_line(err, 220)}`" if err else "")
        )
    idx.write_text("\n".join(lines))
    print(f"Wrote {len(summaries)} excerpt dumps + INDEX.md to {out_dir}")


if __name__ == "__main__":
    main()

