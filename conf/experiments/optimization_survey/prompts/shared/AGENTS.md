# Shared Prompt Context Guide

This file applies to `conf/experiments/optimization_survey/prompts/shared/`.

## Purpose

- `project_context.txt` holds the stable global invariants for the canonical organism-first evolution program.
- It is prepended to task-specific system prompts via `compose_system_prompt(...)`.

## Rules

- Keep this file focused on invariants, contracts, and research-wide constraints.
- Avoid introducing task-local placeholders here.
- If you tighten or relax runtime contract language here, verify that the downstream task prompts and parser expectations still match.
