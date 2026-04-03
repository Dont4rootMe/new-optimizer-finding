# Legacy Candidate Prompt Guide

This file applies to `conf/prompts/legacy_candidate/`.

## Purpose

- This folder serves the explicit legacy candidate-first path only.
- `system.txt` and `user.txt` instruct the model to emit a complete `optimizer.py` file directly.

## Rules

- Keep the output contract raw Python source code, not structured organism sections.
- Keep the optimizer contract aligned with `build_optimizer(model, max_steps)` and controller methods `step(...)` / `zero_grad(...)`.
- Do not route canonical organism-first generation through this folder.

## Change Discipline

- If you change these prompts, re-check `src/evolve/legacy_generator.py` and legacy-only tests.
- Avoid adding new implicit bridges from canonical code into this legacy surface.
