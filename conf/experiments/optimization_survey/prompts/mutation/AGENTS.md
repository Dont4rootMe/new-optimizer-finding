# Mutation Prompt Guide

This file applies to `conf/experiments/optimization_survey/prompts/mutation/`.

## Purpose

- `system.txt`: defines the canonical design-stage response contract for mutation.
- `user.txt`: injects the parent organism's genes, lineage, and raw implementation code.

## Placeholder Contract

`user.txt` is formatted with:

- `inherited_gene_pool`
- `removed_gene_pool`
- `novelty_rejection_feedback`
- `parent_genetic_code`
- `parent_lineage_summary`
- `parent_implementation_code`

Keep these names synchronized with `src/organisms/mutation.py`.

## Editing Guidance

- Preserve the distinction between inherited genes and removed genes.
- Preserve the expectation that the model returns only the design artifact, not raw Python.
