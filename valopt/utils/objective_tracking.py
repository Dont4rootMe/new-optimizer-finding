"""Helpers for tracking the train objective across optimization steps."""

from __future__ import annotations

import math
from typing import Any


def safe_objective_float(value: Any) -> float | None:
    """Convert objective-like values to a finite float."""

    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if output != output or math.isinf(output):
        return None
    return output


class TrainObjectiveTracker:
    """Track the current/best train loss and baseline hit step."""

    def __init__(self, baseline_threshold: float | None = None) -> None:
        self.baseline_threshold = safe_objective_float(baseline_threshold)
        self.last_train_loss: float | None = None
        self.best_train_loss: float | None = None
        self.best_train_loss_step = 0
        self.first_step_at_or_below_baseline: int | None = None

    def update(self, loss_value: Any, step: int) -> float | None:
        """Record one observed train loss at the provided step."""

        loss_float = safe_objective_float(loss_value)
        if loss_float is None:
            return None

        self.last_train_loss = loss_float
        if self.best_train_loss is None or loss_float < self.best_train_loss:
            self.best_train_loss = loss_float
            self.best_train_loss_step = int(step)

        if (
            self.baseline_threshold is not None
            and self.first_step_at_or_below_baseline is None
            and loss_float <= self.baseline_threshold
        ):
            self.first_step_at_or_below_baseline = int(step)

        return loss_float

    def attach_train_loss(self, metrics: dict[str, float]) -> dict[str, float]:
        """Return metrics enriched with the latest train loss."""

        output = dict(metrics)
        output["train_loss"] = (
            float(self.last_train_loss)
            if self.last_train_loss is not None
            else float("nan")
        )
        return output

