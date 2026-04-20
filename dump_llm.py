#!/usr/bin/env python3
"""Dump all LLM requests/responses/errors for organisms in a generation folder.

Usage: python dump_llm.py <gen_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

STAGE_LIST_KEYS = (
    "design_attempts",
    "compatibility_checks",
    "novelty_checks",
    "repair_attempts",
)
STAGE_SINGLE_KEYS = ("design", "implementation", "repair")


def _fmt_messages(req: dict[str, Any]) -> str:
    msgs = req.get("messages") or []
    out: list[str] = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        out.append(f"----- role: {role} -----\n{content}")
    return "\n\n".join(out) if out else "(no messages)"


def _fmt_response_body(resp: dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return str(resp)
    msg = resp.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if content:
        return content
    return json.dumps(resp, indent=2, ensure_ascii=False)[:4000]


def _dump_stage(label: str, req_stage: Any, resp_stage: Any) -> str:
    lines: list[str] = [f"\n\n## STAGE: {label}\n"]
    if req_stage is None and resp_stage is None:
        lines.append("_absent_")
        return "\n".join(lines)

    req_stage = req_stage or {}
    resp_stage = resp_stage or {}

    meta_keys = [
        "status",
        "attempt",
        "design_attempt",
        "operator",
        "provider",
        "provider_model_id",
        "route_id",
        "verdict",
        "rejection_reason",
        "sections_at_issue",
        "compilation_mode",
        "changed_sections",
        "error_msg",
        "error_kind",
        "started_at",
        "finished_at",
        "parse_source",
        "usage",
        "compatibility_rejection_feedback",
        "novelty_rejection_feedback",
    ]
    lines.append("### Meta")
    for k in meta_keys:
        if k in req_stage or k in resp_stage:
            v = resp_stage.get(k, req_stage.get(k))
            if v in (None, "", [], {}):
                continue
            lines.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}")

    sys_prompt = req_stage.get("system_prompt")
    usr_prompt = req_stage.get("user_prompt")
    req_body = req_stage.get("request") or {}

    lines.append("\n### Request — system_prompt\n```\n" + (sys_prompt or "(none)") + "\n```")
    lines.append("\n### Request — user_prompt\n```\n" + (usr_prompt or "(none)") + "\n```")
    if req_body:
        lines.append("\n### Request — raw messages\n```\n" + _fmt_messages(req_body) + "\n```")

    text = resp_stage.get("text") or resp_stage.get("content_text") or resp_stage.get("parse_text")
    thinking = resp_stage.get("thinking_text")
    raw_resp = resp_stage.get("response")

    if thinking:
        lines.append("\n### Response — thinking_text\n```\n" + thinking + "\n```")
    lines.append("\n### Response — text\n```\n" + (text or "(none)") + "\n```")
    if raw_resp:
        lines.append("\n### Response — provider body\n```\n" + _fmt_response_body(raw_resp) + "\n```")

    err = resp_stage.get("error_msg") or req_stage.get("error_msg")
    if err:
        lines.append(f"\n### Error\n```\n{err}\n```")

    return "\n".join(lines)


def dump_organism(org_dir: Path, out_dir: Path) -> dict[str, Any]:
    rel = f"{org_dir.parent.name}/{org_dir.name}"
    out_file = out_dir / f"{org_dir.parent.name}__{org_dir.name}.md"
    req_path = org_dir / "llm_request.json"
    resp_path = org_dir / "llm_response.json"
    req = json.loads(req_path.read_text()) if req_path.exists() else {}
    resp = json.loads(resp_path.read_text()) if resp_path.exists() else {}

    try:
        organism = json.loads((org_dir / "organism.json").read_text())
    except Exception:
        organism = {}

    lines: list[str] = []
    lines.append(f"# {rel}\n")
    lines.append(f"- organism_id: `{organism.get('organism_id','?')}`")
    lines.append(f"- island: `{organism.get('island_id','?')}`")
    lines.append(f"- operator: `{organism.get('operator','?')}`")
    lines.append(f"- provider: `{req.get('provider','?')}` model: `{req.get('provider_model_id','?')}` route: `{req.get('route_id','?')}`")
    lines.append(f"- pipeline_state: `{organism.get('pipeline_state','?')}`")
    if organism.get("error_msg"):
        lines.append(f"- error_msg: `{organism['error_msg']}`")

    for i, (ra, rb) in enumerate(
        zip(req.get("design_attempts", []), resp.get("design_attempts", []))
    ):
        lines.append(_dump_stage(f"design_attempts[{i}]", ra, rb))

    for i, (ra, rb) in enumerate(
        zip(req.get("compatibility_checks", []), resp.get("compatibility_checks", []))
    ):
        lines.append(_dump_stage(f"compatibility_checks[{i}]", ra, rb))

    for i, (ra, rb) in enumerate(
        zip(req.get("novelty_checks", []), resp.get("novelty_checks", []))
    ):
        lines.append(_dump_stage(f"novelty_checks[{i}]", ra, rb))

    for key in STAGE_SINGLE_KEYS:
        if key in req or key in resp:
            lines.append(_dump_stage(key, req.get(key), resp.get(key)))

    for i, (ra, rb) in enumerate(
        zip(req.get("repair_attempts", []), resp.get("repair_attempts", []))
    ):
        lines.append(_dump_stage(f"repair_attempts[{i}]", ra, rb))

    creation_err = org_dir / "logs" / "creation.err"
    if creation_err.exists():
        lines.append("\n\n## LOG: logs/creation.err\n```\n" + creation_err.read_text() + "\n```")

    for extra in sorted((org_dir / "logs").glob("*.err")):
        if extra.name == "creation.err":
            continue
        lines.append(f"\n\n## LOG: logs/{extra.name}\n```\n" + extra.read_text() + "\n```")

    out_file.write_text("\n".join(lines))
    return {
        "rel": rel,
        "out": str(out_file),
        "pipeline_state": organism.get("pipeline_state"),
        "error_msg": organism.get("error_msg"),
        "route_id": req.get("route_id"),
        "provider_model_id": req.get("provider_model_id"),
    }


def main() -> None:
    gen_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for island_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir() and p.name.startswith("island_")):
        for org_dir in sorted(p for p in island_dir.iterdir() if p.is_dir() and p.name.startswith("org_")):
            summaries.append(dump_organism(org_dir, out_dir))

    idx = out_dir / "INDEX.md"
    lines = [f"# LLM dump index for {gen_dir}\n", f"Total organisms: {len(summaries)}\n"]
    for s in summaries:
        lines.append(
            f"- [{s['rel']}]({Path(s['out']).name}) — state=`{s['pipeline_state']}` "
            f"route=`{s['route_id']}` model=`{s['provider_model_id']}`"
            + (f" error=`{s['error_msg']}`" if s["error_msg"] else "")
        )
    idx.write_text("\n".join(lines))
    print(f"Wrote {len(summaries)} organism dumps + INDEX.md to {out_dir}")


if __name__ == "__main__":
    main()
