"""Train loop for DDPM CIFAR-10 experiment."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from experiments.optimization_survey._runtime.compute import autocast_context, resolve_torch_device
from experiments.optimization_survey._runtime.optimizer_runtime import (
    ActivationRecorder,
    StepBudget,
    collect_named_trainable_parameters,
    collect_step_tensors,
)
from experiments.optimization_survey._runtime.safety import check_params_finite, grad_norm, loss_is_finite
from experiments.optimization_survey._runtime.objective_tracking import TrainObjectiveTracker
from experiments.optimization_survey._runtime.seed import set_seed
from experiments.optimization_survey._runtime.timer import WallTimer

from .metrics import compute_primary, is_target_reached

LOGGER = logging.getLogger(__name__)


def _device_from_cfg(cfg: DictConfig) -> torch.device:
    return resolve_torch_device(str(cfg.compute.device))


def _autocast_context(cfg: DictConfig, device: torch.device):
    return autocast_context(device=device, precision=str(cfg.compute.precision))


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def _build_betas(cfg: DictConfig, device: torch.device) -> torch.Tensor:
    timesteps = int(cfg.diffusion.timesteps)
    schedule = str(cfg.diffusion.schedule_type)
    if schedule == "cosine":
        betas = _cosine_beta_schedule(timesteps)
    else:
        betas = torch.linspace(1e-4, 2e-2, timesteps)
    return betas.to(device)


@torch.no_grad()
def evaluate(cfg: DictConfig, model: torch.nn.Module, datamodule: dict[str, Any]) -> dict[str, float]:
    """Return quick proxy metric for DDPM from one validation batch."""

    device = _device_from_cfg(cfg)
    model.eval()
    val_loader = datamodule["val"]

    try:
        images, _ = next(iter(val_loader))
    except StopIteration:
        return {"train_loss": float("nan"), "ema_loss": float("nan")}

    images = images.to(device)
    timesteps = int(cfg.diffusion.timesteps)
    t = torch.randint(0, timesteps, (images.size(0),), device=device)

    betas = _build_betas(cfg, device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    noise = torch.randn_like(images)
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(a_bar) * images + torch.sqrt(1.0 - a_bar) * noise
    pred = model(x_t, t)

    loss = F.mse_loss(pred, noise)
    return {"train_loss": float(loss.item()), "ema_loss": float(loss.item())}


def train(
    cfg: DictConfig,
    model: torch.nn.Module,
    datamodule: dict[str, Any],
    optimizer_factory,
) -> dict[str, Any]:
    """Run DDPM optimization and return metrics summary."""

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

    betas = _build_betas(cfg, device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    timesteps = int(cfg.diffusion.timesteps)

    ema_decay = float(cfg.diffusion.get("ema_decay", 0.9999))
    ema_model_params = [param.detach().clone() for param in model.parameters()]

    def update_ema() -> None:
        with torch.no_grad():
            for ema_param, param in zip(ema_model_params, model.parameters()):
                ema_param.mul_(ema_decay).add_(param.detach(), alpha=1.0 - ema_decay)

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

    samples_seen = 0
    ema_loss: float | None = None
    ema_loss_start: float | None = None
    last_loss = 0.0

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
                images, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, _ = next(train_iter)

            images = images.to(device, non_blocking=True)
            samples_seen += int(images.size(0))

            t = torch.randint(0, timesteps, (images.size(0),), device=device)
            noise = torch.randn_like(images)
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
            x_t = torch.sqrt(alpha_bar_t) * images + torch.sqrt(1.0 - alpha_bar_t) * noise

            with _autocast_context(cfg, device):
                pred = model(x_t, t)
                loss = F.mse_loss(pred, noise)
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
                    _alpha_bars=alpha_bars,
                    _timesteps=timesteps,
                ):
                    nonlocal train_iter, samples_seen
                    try:
                        imgs, _ = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        imgs, _ = next(train_iter)
                    imgs = imgs.to(_device, non_blocking=True)
                    samples_seen += int(imgs.size(0))
                    t = torch.randint(0, _timesteps, (imgs.size(0),), device=_device)
                    noise = torch.randn_like(imgs)
                    a_bar = _alpha_bars[t].view(-1, 1, 1, 1)
                    x_t = torch.sqrt(a_bar) * imgs + torch.sqrt(1.0 - a_bar) * noise
                    with _autocast_context(_cfg, _device):
                        pred = _model(x_t, t)
                        l = F.mse_loss(pred, noise)
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
                update_ema()

                if bool(cfg.safety.detect_nan) and not check_params_finite(model):
                    msg = f"Non-finite model parameters detected at step {step}."
                    if bool(cfg.safety.abort_on_nan):
                        raise RuntimeError(msg)

            last_loss = loss_item
            ema_loss = loss_item if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_item)

            if step % eval_every == 0 or step == max_steps:
                if ema_loss_start is None:
                    ema_loss_start = float(ema_loss)

                metrics = {
                    "train_loss": loss_item,
                    "ema_loss": float(ema_loss),
                    "ema_loss_at_start": float(ema_loss_start),
                }
                final_metrics = objective_tracker.attach_train_loss(metrics)

                primary_value = compute_primary(metrics, cfg.primary_metric)
                if primary_value < best_primary:
                    best_primary = primary_value
                    best_metrics = dict(final_metrics)
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
    except KeyboardInterrupt:
        status = "interrupted"
        error_msg = "interrupted by user"
    finally:
        activation_recorder.close()

    if not final_metrics:
        if status == "ok":
            if ema_loss is None:
                ema_loss = float("nan")
            final_metrics = {
                "train_loss": last_loss,
                "ema_loss": float(ema_loss),
                "ema_loss_at_start": float(ema_loss if ema_loss_start is None else ema_loss_start),
            }
            final_metrics = objective_tracker.attach_train_loss(final_metrics)
        else:
            final_metrics = objective_tracker.attach_train_loss({})
    else:
        final_metrics = objective_tracker.attach_train_loss(final_metrics)
    if not best_metrics:
        best_metrics = dict(final_metrics)

    wall_time = timer.elapsed()
    throughput = samples_seen / wall_time if wall_time > 0 else 0.0

    return {
        "status": status,
        "error_msg": error_msg,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "steps": int(step),
        "wall_time_sec": wall_time,
        "samples_or_tokens_seen": samples_seen,
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
            "throughput_samples_per_sec": throughput,
            "grad_norm_mean": float(sum(grad_norm_log) / len(grad_norm_log)) if grad_norm_log else 0.0,
            "grad_norm_max": max(grad_norm_log) if grad_norm_log else 0.0,
        },
    }
