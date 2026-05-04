from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import TransformerConfig, TransformerLM
from .tokenizer import BPETokenizer


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sample_starts(max_start: int, max_eval_windows: int) -> list[int]:
    starts = list(range(max_start + 1))
    if len(starts) <= max_eval_windows:
        return starts
    if max_eval_windows == 1:
        return [max_start]
    return [round(i * max_start / (max_eval_windows - 1)) for i in range(max_eval_windows)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MP2 local public-tail perplexity sanity check."
    )
    parser.add_argument("--checkpoint", default="mp/artifacts/checkpoint.pt")
    parser.add_argument("--tokenizer", default="mp/artifacts/tokenizer.json")
    parser.add_argument("--eval-text", default="data/tinystories_subset.txt")
    parser.add_argument("--target-perplexity", type=float, default=120.0)
    parser.add_argument("--margin-perplexity", type=float, default=100.0)
    parser.add_argument("--eval-char-budget", type=int, default=50_000)
    parser.add_argument("--max-eval-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()
    if args.max_eval_windows < 1:
        parser.error("--max-eval-windows must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.eval_char_budget < 1:
        parser.error("--eval-char-budget must be at least 1.")

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    eval_text_path = Path(args.eval_text)

    for required_path in [checkpoint_path, tokenizer_path, eval_text_path]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing {required_path}. Run training/download cells first."
            )

    device = args.device or _default_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TransformerConfig(**checkpoint["config"])
    model = TransformerLM(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer.load(str(tokenizer_path))
    eval_text = eval_text_path.read_text(encoding="utf-8")[-args.eval_char_budget :]

    encode_start = time.time()
    all_ids = tokenizer.encode(eval_text)
    print(
        f"Encoded {len(eval_text):,} characters into {len(all_ids):,} tokens "
        f"in {time.time() - encode_start:.1f}s."
    )
    if len(all_ids) < 3:
        raise ValueError("Tokenizer produced too few tokens for a perplexity check.")

    tokens = torch.tensor(all_ids, dtype=torch.long)
    seq_len = min(config.max_seq_len, 128, tokens.numel() - 1)
    if seq_len < 2:
        raise ValueError("Evaluation split is too short for a perplexity check.")

    max_start = tokens.numel() - seq_len - 1
    starts = _sample_starts(max_start=max_start, max_eval_windows=args.max_eval_windows)

    total_nll = 0.0
    total_tokens = 0
    eval_start = time.time()
    with torch.no_grad():
        for i in range(0, len(starts), args.batch_size):
            batch_starts = starts[i : i + args.batch_size]
            x = torch.stack([tokens[s : s + seq_len] for s in batch_starts]).to(device)
            y = torch.stack(
                [tokens[s + 1 : s + seq_len + 1] for s in batch_starts]
            ).to(device)
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss_sum.item())
            total_tokens += y.numel()

    print(f"Evaluated {len(starts)} windows in {time.time() - eval_start:.1f}s.")

    mean_nll = total_nll / total_tokens
    perplexity = math.exp(mean_nll)

    print(f"Public-tail perplexity: {perplexity:.2f}")
    print(f"Minimum local sanity target: below {args.target_perplexity:.0f}")
    print(f"Stronger safety margin: below {args.margin_perplexity:.0f}")
    print(f"Mean next-token NLL: {mean_nll:.4f}")

    if perplexity < args.margin_perplexity:
        print("Local check passed with a stronger margin.")
    elif perplexity < args.target_perplexity:
        print("Local check passed. Below 100 would give a stronger margin.")
    else:
        raise SystemExit(
            "Local check is above target. Consider rerunning training before submitting."
        )

    print("The hidden grader still uses a different TinyStories subset.")


if __name__ == "__main__":
    main()
