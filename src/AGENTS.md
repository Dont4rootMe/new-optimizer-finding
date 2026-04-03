# Application Entry Guide

This file applies to `src/` unless a deeper `AGENTS.md` overrides it.

## Purpose

- `main.py` is the top-level dispatch entrypoint.
- `evolve/` contains the canonical organism-first evolution engine.
- `organisms/` contains canonical organism parsing, persistence, and prompt-driven mutation/crossover logic.
- `validate/` contains the subprocess worker used by evolution-time evaluation.

## Rules

- Keep `src/main.py` thin. It should route by mode, not grow its own business logic.
- Put canonical evolution logic in `src/evolve/`.
- Put organism response handling and organism artifact rules in `src/organisms/`.
- Put subprocess evaluation concerns in `src/validate/`.
