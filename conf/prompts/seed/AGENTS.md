# Seed Prompt Guide

This file applies to `conf/prompts/seed/`.

## Purpose

- `system.txt`: defines the canonical structured response contract for seed organisms.
- `user.txt`: injects island-specific context for creating a brand-new organism.

## Placeholder Contract

`user.txt` is formatted with:

- `island_id`
- `island_name`
- `island_description`

Keep these names synchronized with `src/evolve/operators.py`.

## Response Contract

The seed prompt expects a structured organism response with the canonical sections enforced by `src/organisms/organism.py`.
Do not turn this task into raw Python output.
