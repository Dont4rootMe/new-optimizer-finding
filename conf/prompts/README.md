Prompt assets are centralized here.

Shared context:
- `shared/project_context.txt`: shared system-level invariants for the whole evolution program.

Canonical organism tasks:
- `seed/system.txt`
- `seed/user.txt`
- `mutation/system.txt`
- `mutation/user.txt`
- `crossover/system.txt`
- `crossover/user.txt`
- `implementation/system.txt`
- `implementation/user.txt`
- `implementation/template.txt`

`seed/`, `mutation/`, and `crossover/` define the design-stage prompts.
`implementation/` defines the shared second-stage code-generation prompt and fixed template scaffold.

Island prompts:
- `islands/*.txt`: research directions for the island-based search. These are also prompt assets and are loaded by the canonical evolution loop.
