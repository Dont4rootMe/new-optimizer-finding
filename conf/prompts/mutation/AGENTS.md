# Mutation Prompt Guide

This file applies to `conf/prompts/mutation/`.

## Purpose

- `system.txt`: defines the canonical structured response contract for mutation.
- `user.txt`: injects the parent organism's genes, rewards, lineage, and editable code sections.

## Placeholder Contract

`user.txt` is formatted with:

- `inherited_gene_pool`
- `removed_gene_pool`
- `parent_genetic_code`
- `parent_selection_reward`
- `parent_simple_reward`
- `parent_hard_reward`
- `parent_lineage_summary`
- `parent_imports`
- `parent_init_body`
- `parent_step_body`
- `parent_zero_grad_body`

Keep these names synchronized with `src/organisms/mutation.py`.

## Editing Guidance

- Preserve the distinction between inherited genes and removed genes.
- Preserve the expectation that the model returns a structured organism, not a free-form explanation.
