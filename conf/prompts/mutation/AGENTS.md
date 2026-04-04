# Mutation Prompt Guide

This file applies to `conf/prompts/mutation/`.

## Purpose

- `system.txt`: defines the canonical design-stage response contract for mutation.
- `user.txt`: injects the parent organism's genes, lineage, and editable code sections.

## Placeholder Contract

`user.txt` is formatted with:

- `inherited_gene_pool`
- `removed_gene_pool`
- `parent_genetic_code`
- `parent_lineage_summary`
- `parent_imports`
- `parent_init_body`
- `parent_step_body`
- `parent_zero_grad_body`

Keep these names synchronized with `src/organisms/mutation.py`.

## Editing Guidance

- Preserve the distinction between inherited genes and removed genes.
- Preserve the expectation that the model returns only the design artifact, not raw Python.
