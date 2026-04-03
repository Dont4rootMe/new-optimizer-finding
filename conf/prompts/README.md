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

Each task folder contains the paired system/user prompts for one LLM action.

Island prompts:
- `islands/*.txt`: research directions for the island-based search. These are also prompt assets and are loaded by the canonical evolution loop.
