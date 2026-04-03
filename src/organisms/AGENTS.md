# Organism Contract Guide

This file applies to `src/organisms/`.

## Purpose

- `organism.py`: canonical organism validation, persistence, lineage handling, and genetic-code formatting
- `mutation.py`: mutation prompt construction and organism creation
- `crossbreeding.py`: crossover prompt construction and organism creation

## Canonical Organism Rules

- Canonical organism responses are structured sections, not raw Python files.
- Required response sections are enforced in `organism.py` and must remain aligned with the prompt contract:
  - `CORE_GENES`
  - `INTERACTION_NOTES`
  - `COMPUTE_NOTES`
  - `CHANGE_DESCRIPTION`
  - `IMPORTS`
  - `INIT_BODY`
  - `STEP_BODY`
  - `ZERO_GRAD_BODY`
- `genetic_code.md` and `lineage.json` are first-class artifacts, not optional summaries.
- Lineage entries should describe actual changes and carry phase-specific backfilled scores through `update_latest_lineage_entry(...)`.

## Editing Rules

- Use `build_organism_from_response(...)` to enforce canonical validation and artifact creation.
- Keep gene-diff summaries grounded in actual parent-vs-child changes.
- Keep mutation and crossover prompt placeholders synchronized with the prompt files under `conf/prompts/`.
- Do not hand-roll alternative artifact formats when the storage helpers already define the canonical ones.

## Verification

- `pytest -q tests/test_organism_contract.py`
- `pytest -q tests/test_mutation_dna.py`
- `pytest -q tests/test_crossbreeding.py`
