# Island Prompt Guide

This file applies to `conf/prompts/islands/`.

## Purpose

- Each `.txt` file defines one research island for canonical organism-first evolution.
- The filename stem becomes `island_id`.
- The file body becomes `description_text`.

## Rules

- Keep one island per file.
- Keep descriptions plain-text and concise enough to be prompt-friendly.
- Describe a school of thought, not a concrete implementation template.
- Stay within the runtime constraints of the project: lightweight optimizer logic, safe defaults, and honest compute tradeoffs.
- Renaming a file changes `island_id`, which affects population layout and resume semantics.
