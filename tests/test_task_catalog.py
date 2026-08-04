"""Machine-readable task catalog contracts."""

from __future__ import annotations

from experiments.catalog import audit_catalog, render_markdown, task_specs


def test_task_catalog_is_exhaustive_and_has_stable_family_counts() -> None:
    specs = task_specs()
    assert len(specs) == 25
    assert len({spec.task_id for spec in specs}) == len(specs)
    counts = {
        family: sum(spec.family == family for spec in specs)
        for family in {spec.family for spec in specs}
    }
    assert counts == {
        "circle_packing_shinka": 1,
        "awtf2025_heuristic": 1,
        "co_bench": 6,
        "optimization_survey": 17,
    }
    assert sum(spec.enabled_by_default for spec in specs) == 22


def test_task_catalog_contract_audit_has_no_repository_errors() -> None:
    errors = [finding for finding in audit_catalog() if finding.severity == "error"]
    assert errors == []


def test_task_catalog_markdown_contains_every_task() -> None:
    markdown = render_markdown()
    for spec in task_specs():
        assert spec.task_id in markdown


def test_optimization_seed_satisfies_controller_contract_and_updates_weights() -> None:
    import torch

    from experiments.optimization_survey._runtime.candidate_loader import (
        load_candidate_builder,
    )
    from experiments.catalog import REPO_ROOT

    seed_path = REPO_ROOT / "experiments/optimization_survey/_baselines/initial_program.py"
    builder, _, name = load_candidate_builder(str(seed_path))
    model = torch.nn.Linear(2, 1, bias=False)
    before = model.weight.detach().clone()
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    controller = builder(model, 10)
    controller.step({}, {}, {}, lambda: (0.0, {}, {}))
    controller.zero_grad()

    assert name == "SeedAdam"
    assert not torch.equal(before, model.weight.detach())
