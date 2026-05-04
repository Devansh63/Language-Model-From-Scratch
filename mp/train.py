from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .model import TransformerConfig


def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy from logits using logsumexp + gather.

    TODO:
    Compute next-token cross-entropy directly from logits.

    Required behavior:
    1) Treat the last dimension of `logits` as the vocabulary dimension.
    2) For every position, compute:
       log(normalizer over all classes) - logit of the target class
    3) Average the per-position losses into one scalar tensor.

    Important details:
    - this must work for logits with any number of leading batch dimensions
    - `targets` has the same leading shape as `logits[..., 0]`
    - the result should numerically match the standard cross-entropy definition
    """
    # log(sum(exp(logits))) - logit of the correct class, averaged over all positions
    log_normalizer = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    loss = log_normalizer - target_logits
    return loss.mean()


def get_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random next-token prediction batches.

    TODO:
    Sample training examples from a 1D token stream.

    Required behavior:
    1) Randomly choose `batch_size` starting positions in `token_ids`.
    2) For each start position, create:
       - an input sequence of length `seq_len`
       - a target sequence of length `seq_len` shifted by one token to the right
    3) Stack all sampled examples into tensors of shape [B, T].
    4) Move the returned tensors to `device`.

    Important details:
    - `token_ids` is a single long 1D sequence, not a batch
    - every target token should be the next token after the corresponding input token
    - sampled windows must stay within bounds
    """
    # need seq_len + 1 tokens per window so targets don't go out of bounds
    max_start = len(token_ids) - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))

    x_list = []
    y_list = []
    for start in starts:
        start = int(start)
        x_list.append(token_ids[start : start + seq_len])
        y_list.append(token_ids[start + 1 : start + seq_len + 1])

    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)

    return x, y


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
    Execute one full training iteration.

    Required behavior:
    1) Put the model in training mode.
    2) Clear old gradients before the new forward/backward pass.
    3) Run the model on `x` and compute the cross-entropy loss against `y`.
    4) Backpropagate through the loss.
    5) Update parameters with the optimizer.
    6) Return the loss as a Python float.

    Important details:
    - do not skip the backward pass or optimizer step
    - the returned value should be easy to log or append to a list
    - the function should work on CPU and GPU
    """
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss = cross_entropy_from_logits(logits, y)

    loss.backward()
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
) -> float:
    """Evaluate average validation loss over eval_iters mini-batches."""
    model.eval()
    device = str(next(model.parameters()).device)

    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_tokens, batch_size=batch_size, seq_len=seq_len, device=device)
        logits = model(x)
        losses.append(float(cross_entropy_from_logits(logits, y).item()))
    return sum(losses) / len(losses)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TransformerConfig,
) -> None:
    """Save training state."""
    payload: dict[str, Any] = {
        "step": step,
        "config": asdict(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load training state into model (+ optimizer if provided)."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return {"step": payload.get("step", 0), "config": payload.get("config", {})}
