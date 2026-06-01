# Crossover Prompt Guide

This file applies to `conf/experiments/optimization_survey/prompts/crossover/`.

## Purpose

- `system.txt`: defines the canonical design-stage response contract for crossbreeding.
- `user.txt`: injects maternal and paternal genes and lineage summaries around the child draft.

## Placeholder Contract

`user.txt` is formatted with:

- `inherited_gene_pool`
- `novelty_rejection_feedback`
- `mother_genetic_code`
- `mother_lineage_summary`
- `father_genetic_code`
- `father_lineage_summary`

Keep these names synchronized with `src/organisms/crossbreeding.py`.

## Editing Guidance

- Preserve the explicit maternal bias language unless the selection/recombination logic is being changed in code too.
- Preserve the expectation of a structured design response, not raw Python.
