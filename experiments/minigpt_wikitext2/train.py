"""Training loop for MiniGPT WikiText-2 experiment."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
from omegaconf import DictConfig

from valopt.utils.optimizer_runtime import (
    ActivationRecorder,
    StepBudget,
    collect_named_trainable_parameters,
    collect_step_tensors,
)
from valopt.utils.safety import check_params_finite, grad_norm, loss_is_finite
from valopt.utils.objective_tracking import TrainObjectiveTracker
from valopt.utils.seed import set_seed
from valopt.utils.timer import WallTimer

from .metrics import compute_primary, is_target_reached, perplexity_from_loss

LOGGER = logging.getLogger(__name__)


def _device_from_cfg(cfg: DictConfig) -> torch.device:
    if str(cfg.compute.device) == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _autocast_context(cfg: DictConfig, device: torch.device):
    precision = str(cfg.compute.precision)
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    if not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


@torch.no_grad()
def evaluate(cfg: DictConfig, model: torch.nn.Module, datamodule: dict[str, Any]) -> dict[str, float]:
    """Evaluate causal LM loss and perplexity on validation split."""

    device = _device_from_cfg(cfg)
    model.eval()

    val_loader = datamodule["val"]
    eval_max_batches = int(cfg.train.get("eval_max_batches", 100))

    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= eval_max_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with _autocast_context(cfg, device):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        total_loss += float(loss.item())
        total_batches += 1

    if total_batches == 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}

    mean_loss = total_loss / total_batches
    return {
        "val_loss": mean_loss,
        "val_ppl": perplexity_from_loss(mean_loss),
    }


def train(
    cfg: DictConfig,
    model: torch.nn.Module,
    datamodule: dict[str, Any],
    optimizer_factory,
) -> dict[str, Any]:
    """Run MiniGPT training and return structured summary."""

    set_seed(int(cfg.seed), bool(cfg.deterministic))
    device = _device_from_cfg(cfg)
    model = model.to(device)

    max_steps = int(cfg.compute.max_steps)
    grad_accum_steps = max(1, int(cfg.compute.grad_accum_steps))
    eval_every = max(1, int(cfg.train.eval_every_steps))
    log_every = max(1, int(cfg.train.get("log_every_steps", eval_every)))

    optimizer_controller = optimizer_factory(model, max_steps)

    named_params = collect_named_trainable_parameters(model)
    params = [param for _, param in named_params]
    step_budget = StepBudget(max_steps)

    grad_clip_norm = cfg.safety.get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    train_loader = datamodule["train"]
    train_iter = iter(train_loader)

    activation_recorder = ActivationRecorder(
        model,
        max_modules=int(cfg.train.get("max_activation_hooks", 32)),
    )

    best_metrics: dict[str, float] = {}
    final_metrics: dict[str, float] = {}
    best_primary = float("inf")
    best_step = 0
    steps_to_target: int | None = None
    converged = False
    patience = cfg.target.get("patience")
    patience_value = int(patience) if patience is not None else None
    hold_count = 0
    grad_norm_log: list[float] = []
    tokens_seen = 0

    timer = WallTimer()
    timer.start()

    objective_tracker = TrainObjectiveTracker(
        baseline_threshold=cfg.runtime.get("baseline_last_train_loss"),
    )
    status = "ok"
    error_msg: str | None = None

    optimizer_controller.zero_grad(set_to_none=True)
    model.train()

    step = 0
    try:
        while step < max_steps:
            step += 1
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            tokens_seen += int(input_ids.numel())

            with _autocast_context(cfg, device):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                scaled_loss = loss / grad_accum_steps

            if bool(cfg.safety.detect_nan) and not loss_is_finite(loss):
                msg = f"Non-finite loss detected at step {step}."
                if bool(cfg.safety.abort_on_nan):
                    raise RuntimeError(msg)
                continue

            loss_item = float(loss.item())
            objective_tracker.update(loss_item, step)
            if step % log_every == 0:
                LOGGER.info("step=%d train_loss=%.6f", step, loss_item)

            scaled_loss.backward()

            if step % grad_accum_steps == 0:
                current_grad_norm = grad_norm(params)
                if bool(cfg.safety.log_grad_norm):
                    grad_norm_log.append(current_grad_norm)

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)

                weights, grads = collect_step_tensors(named_params)
                activations = activation_recorder.snapshot()

                def _step_fn(
                    _train_iter=train_iter,
                    _model=model,
                    _device=device,
                    _cfg=cfg,
                    _grad_accum_steps=grad_accum_steps,
                    _named_params=named_params,
                    _activation_recorder=activation_recorder,
                    _step_budget=step_budget,
                ):
                    nonlocal train_iter, tokens_seen
                    try:
                        b = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        b = next(train_iter)
                    ids = b["input_ids"].to(_device, non_blocking=True)
                    mask = b["attention_mask"].to(_device, non_blocking=True)
                    lab = b["labels"].to(_device, non_blocking=True)
                    tokens_seen += int(ids.numel())
                    with _autocast_context(_cfg, _device):
                        out = _model(input_ids=ids, attention_mask=mask, labels=lab)
                        l = out.loss
                        (l / _grad_accum_steps).backward()
                    w, g = collect_step_tensors(_named_params)
                    a = _activation_recorder.snapshot()
                    _step_budget.increment()
                    _objective_step = step + _step_budget.current_cycle_consumed
                    objective_tracker.update(float(l.item()), _objective_step)
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
                metrics = objective_tracker.attach_train_loss(evaluate(cfg, model, datamodule))
                final_metrics = metrics
                primary_value = compute_primary(metrics, cfg.primary_metric)
                if primary_value < best_primary:
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
    except KeyboardInterrupt:
        status = "interrupted"
        error_msg = "interrupted by user"
    finally:
        activation_recorder.close()

    if not final_metrics:
        if status == "ok":
            final_metrics = objective_tracker.attach_train_loss(evaluate(cfg, model, datamodule))
        else:
            final_metrics = objective_tracker.attach_train_loss({})
    else:
        final_metrics = objective_tracker.attach_train_loss(final_metrics)
    if not best_metrics:
        best_metrics = dict(final_metrics)

    wall_time = timer.elapsed()
    throughput = tokens_seen / wall_time if wall_time > 0 else 0.0

    return {
        "status": status,
        "error_msg": error_msg,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "steps": int(step),
        "wall_time_sec": wall_time,
        "samples_or_tokens_seen": tokens_seen,
        "converged": converged,
        "steps_to_target": steps_to_target,
        "objective_name": "train_loss",
        "objective_direction": "min",
        "objective_last": float(objective_tracker.last_train_loss if objective_tracker.last_train_loss is not None else float("nan")),
        "objective_best": float(objective_tracker.best_train_loss if objective_tracker.best_train_loss is not None else float("nan")),
        "objective_best_step": int(objective_tracker.best_train_loss_step),
        "first_step_at_or_below_baseline": objective_tracker.first_step_at_or_below_baseline,
        "extra_metrics": {
            "best_step": best_step,
            "throughput_tokens_per_sec": throughput,
            "grad_norm_mean": float(sum(grad_norm_log) / len(grad_norm_log)) if grad_norm_log else 0.0,
            "grad_norm_max": max(grad_norm_log) if grad_norm_log else 0.0,
        },
    }
