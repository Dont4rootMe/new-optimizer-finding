"""Training loop for synthetic logistic regression experiment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from valopt.utils.optimizer_runtime import (
    ActivationRecorder,
    StepBudget,
    collect_named_trainable_parameters,
    collect_step_tensors,
)
from valopt.utils.safety import check_params_finite, grad_norm, loss_is_finite
from valopt.utils.seed import set_seed
from valopt.utils.timer import WallTimer

from .metrics import compute_primary, is_target_reached


def _device_from_cfg(cfg: DictConfig) -> torch.device:
    requested = str(cfg.compute.device)
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(cfg: DictConfig, model: nn.Module, datamodule: dict[str, Any]) -> dict[str, float]:
    """Evaluate model on validation split."""
    device = _device_from_cfg(cfg)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loader = datamodule["val"]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in val_loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(labels.size(0))

    if total_samples == 0:
        return {"val_loss": float("nan"), "val_acc": float("nan")}

    return {
        "val_loss": total_loss / total_samples,
        "val_acc": total_correct / total_samples,
    }


def train(
    cfg: DictConfig,
    model: nn.Module,
    datamodule: dict[str, Any],
    optimizer_factory,
) -> dict[str, Any]:
    """Run synthetic logreg training and return structured summary."""
    set_seed(int(cfg.seed), bool(cfg.deterministic))
    device = _device_from_cfg(cfg)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    max_steps = int(cfg.compute.max_steps)
    eval_every = max(1, int(cfg.train.eval_every_steps))

    optimizer_controller = optimizer_factory(model, max_steps)

    named_params = collect_named_trainable_parameters(model)
    params = [param for _, param in named_params]
    step_budget = StepBudget(max_steps)

    train_loader = datamodule["train"]
    train_iter = iter(train_loader)

    activation_recorder = ActivationRecorder(
        model,
        max_modules=int(cfg.train.get("max_activation_hooks", 8)),
    )

    grad_norm_log: list[float] = []
    best_metrics: dict[str, float] = {}
    final_metrics: dict[str, float] = {}
    best_primary = -float("inf")
    best_step = 0
    steps_to_target: int | None = None
    converged = False
    patience = cfg.target.get("patience")
    patience_value = int(patience) if patience is not None else None
    hold_count = 0
    samples_seen = 0

    timer = WallTimer()
    timer.start()

    optimizer_controller.zero_grad(set_to_none=True)
    model.train()

    step = 0
    try:
        while step < max_steps:
            step += 1
            try:
                features, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                features, labels = next(train_iter)

            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            samples_seen += int(labels.size(0))

            logits = model(features)
            loss = criterion(logits, labels)

            if bool(cfg.safety.detect_nan) and not loss_is_finite(loss):
                msg = f"Non-finite loss detected at step {step}."
                if bool(cfg.safety.abort_on_nan):
                    raise RuntimeError(msg)
                continue

            loss.backward()

            current_grad_norm = grad_norm(params)
            if bool(cfg.safety.log_grad_norm):
                grad_norm_log.append(current_grad_norm)

            weights, grads = collect_step_tensors(named_params)
            activations = activation_recorder.snapshot()

            def _step_fn(
                _train_iter=train_iter,
                _model=model,
                _criterion=criterion,
                _device=device,
                _named_params=named_params,
                _activation_recorder=activation_recorder,
                _step_budget=step_budget,
            ):
                nonlocal train_iter, samples_seen
                try:
                    feats, lbls = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    feats, lbls = next(train_iter)
                feats = feats.to(_device, non_blocking=True)
                lbls = lbls.to(_device, non_blocking=True)
                samples_seen += int(lbls.size(0))
                out = _model(feats)
                l = _criterion(out, lbls)
                l.backward()
                w, g = collect_step_tensors(_named_params)
                a = _activation_recorder.snapshot()
                _step_budget.increment()
                return float(l.item()), g, a

            optimizer_controller.step(weights, grads, activations, _step_fn)
            optimizer_controller.zero_grad(set_to_none=True)
            activation_recorder.clear()
            extra_steps = step_budget.consumed_since_last
            if extra_steps > 0:
                step += extra_steps
                if step >= max_steps:
                    break

            if bool(cfg.safety.detect_nan) and not check_params_finite(model):
                msg = f"Non-finite model parameters detected at step {step}."
                if bool(cfg.safety.abort_on_nan):
                    raise RuntimeError(msg)

            if step % eval_every == 0 or step == max_steps:
                metrics = evaluate(cfg, model, datamodule)
                final_metrics = metrics
                primary_value = compute_primary(metrics, cfg.primary_metric)
                if primary_value > best_primary:
                    best_primary = primary_value
                    best_metrics = dict(metrics)
                    best_step = step

                reached = is_target_reached(metrics, cfg.target)
                if reached and steps_to_target is None:
                    steps_to_target = step

                if reached:
                    hold_count += 1
                    if patience_value is None or hold_count >= patience_value:
                        converged = True
                        break
                else:
                    hold_count = 0

                model.train()
    finally:
        activation_recorder.close()

    if not final_metrics:
        final_metrics = evaluate(cfg, model, datamodule)
    if not best_metrics:
        best_metrics = dict(final_metrics)

    wall_time = timer.elapsed()
    throughput = samples_seen / wall_time if wall_time > 0 else 0.0

    return {
        "status": "ok",
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "steps": int(step),
        "wall_time_sec": wall_time,
        "samples_or_tokens_seen": samples_seen,
        "converged": converged,
        "steps_to_target": steps_to_target,
        "extra_metrics": {
            "best_step": best_step,
            "throughput_samples_per_sec": throughput,
            "grad_norm_mean": float(sum(grad_norm_log) / len(grad_norm_log)) if grad_norm_log else 0.0,
            "grad_norm_max": max(grad_norm_log) if grad_norm_log else 0.0,
        },
    }
