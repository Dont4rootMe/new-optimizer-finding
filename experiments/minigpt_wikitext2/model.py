"""Model definition for MiniGPT WikiText-2 experiment."""

from __future__ import annotations

from omegaconf import DictConfig

from optbench.schemas import OptionalDependencyError


def build_model(cfg: DictConfig):
    """Build compact GPT2LMHeadModel for single-GPU training."""

    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise OptionalDependencyError("hf", "Install extra dependencies: pip install -e .[hf]") from exc

    model_cfg = GPT2Config(
        vocab_size=int(cfg.model.vocab_size),
        n_positions=int(cfg.data.seq_len),
        n_ctx=int(cfg.data.seq_len),
        n_layer=int(cfg.model.n_layer),
        n_head=int(cfg.model.n_head),
        n_embd=int(cfg.model.n_embd),
        n_inner=int(cfg.model.n_inner),
        resid_pdrop=float(cfg.model.dropout),
        embd_pdrop=float(cfg.model.dropout),
        attn_pdrop=float(cfg.model.dropout),
    )
    model = GPT2LMHeadModel(model_cfg)

    if bool(cfg.compute.gradient_checkpointing):
        model.gradient_checkpointing_enable()

    return model
