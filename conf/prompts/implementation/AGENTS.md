# Implementation Prompt Guide

This file applies to `conf/prompts/implementation/`.

## Purpose

- `system.txt`: defines the shared implementation-stage contract for generating the final `optimizer.py`.
- `user.txt`: injects the organism genetic code, novelty summary, and fixed template scaffold.
- `template.txt`: canonical optimizer scaffold that the runtime expects generated code to preserve.

## Placeholder Contract

`user.txt` is formatted with:

- `organism_genetic_code`
- `change_description`
- `implementation_template`

Keep these names synchronized with `src/organisms/organism.py`.

## Editing Guidance

- The implementation-stage response must be raw Python only.
- Preserve the scaffold markers in `template.txt`; later generations depend on them to recover editable sections from parent organisms.
