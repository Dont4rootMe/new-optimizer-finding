# Crossover Prompt Guide

This file applies to `conf/prompts/crossover/`.

## Purpose

- `system.txt`: defines the canonical design-stage response contract for crossbreeding.
- `user.txt`: injects maternal and paternal genes, lineage, and editable code sections.

## Placeholder Contract

`user.txt` is formatted with:

- `inherited_gene_pool`
- `mother_genetic_code`
- `mother_lineage_summary`
- `mother_imports`
- `mother_init_body`
- `mother_step_body`
- `mother_zero_grad_body`
- `father_genetic_code`
- `father_lineage_summary`
- `father_imports`
- `father_init_body`
- `father_step_body`
- `father_zero_grad_body`

Keep these names synchronized with `src/organisms/crossbreeding.py`.

## Editing Guidance

- Preserve the explicit maternal bias language unless the selection/recombination logic is being changed in code too.
- Preserve the expectation of a structured design response, not raw Python.
