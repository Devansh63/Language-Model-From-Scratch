from __future__ import annotations

import argparse

import torch

from .model import TransformerConfig, TransformerLM
from .tokenizer import BPETokenizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cpu",
) -> tuple[TransformerLM, BPETokenizer]:
    """Load model checkpoint and tokenizer."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = TransformerConfig(**ckpt["config"])
    model = TransformerLM(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer.load(tokenizer_path)
    return model, tokenizer


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(output_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
