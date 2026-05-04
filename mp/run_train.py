from __future__ import annotations

import itertools
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import TransformerConfig, TransformerLM
from .train import evaluate, get_batch, save_checkpoint, train_step

OUT_DIR = None
VOCAB_SIZE = 1024
BATCH_SIZE = 32
SEQ_LEN = 128
MAX_STEPS = 1200
EVAL_INTERVAL = 200
EVAL_ITERS = 20
LEARNING_RATE = 3e-4
N_LAYERS = 4
N_HEADS = 8
D_MODEL = 384
D_FF = 1536
DROPOUT = 0.1
SEED = 0


@dataclass
class _TokenizerState:
    merges: list[tuple[int, int]]


class _FastBPETokenizer:
    """Provided efficient byte-level BPE tokenizer used only by this training script."""

    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self.merges = merges
        self.base_vocab_size = 256
        self.vocab_size = self.base_vocab_size + len(merges)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        min_pair_freq: int = 2,
        progress_callback=None,
    ) -> "_FastBPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 for byte-level BPE.")
        if min_pair_freq < 1:
            raise ValueError("min_pair_freq must be >= 1.")

        token_ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        next_id = 256
        total_possible_merges = max(0, vocab_size - 256)

        while next_id < vocab_size and len(token_ids) >= 2:
            pair_counts: dict[tuple[int, int], int] = {}
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair, best_count = min(
                pair_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if best_count < min_pair_freq:
                break

            merges.append(best_pair)
            token_ids = cls._replace_pair(token_ids, best_pair, next_id)
            if progress_callback is not None:
                progress_callback(len(merges), total_possible_merges, best_pair, best_count)
            next_id += 1

        return cls(merges)

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        if not token_ids:
            return []

        for merge_rank, pair in enumerate(self.merges):
            merged_id = self.base_vocab_size + merge_rank
            token_ids = self._replace_pair(token_ids, pair, merged_id)
        return token_ids

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state = _TokenizerState(merges=self.merges)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"merges": state.merges}, f)

    @staticmethod
    def _replace_pair(
        token_ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        result: list[int] = []
        i = 0
        while i < len(token_ids):
            if (
                i + 1 < len(token_ids)
                and token_ids[i] == pair[0]
                and token_ids[i + 1] == pair[1]
            ):
                result.append(new_token_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        return result


def _default_corpus_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "tinystories_subset.txt"


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _render_tokenizer_progress(
    merges_done: int,
    total_merges: int,
    best_pair: tuple[int, int],
    pair_count: int,
    start_time: float,
) -> str:
    width = 28
    filled = 0 if total_merges == 0 else int(width * merges_done / total_merges)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    eta_seconds = None
    if merges_done > 0 and total_merges > 0:
        eta_seconds = (elapsed / merges_done) * max(0, total_merges - merges_done)
    eta_text = _format_duration(eta_seconds) if eta_seconds is not None else "--:--"
    return (
        f"[{bar}] merges {merges_done:4d}/{total_merges} "
        f"best_pair={best_pair} freq={pair_count} eta={eta_text}"
    )


def _render_progress(step: int, total_steps: int, loss: float, eta_seconds: float | None) -> str:
    width = 28
    filled = int(width * step / total_steps)
    bar = "#" * filled + "-" * (width - filled)
    eta_text = _format_duration(eta_seconds) if eta_seconds is not None else "--:--"
    return f"[{bar}] step {step:4d}/{total_steps} train_loss={loss:.4f} eta={eta_text}"


def _encode_with_indicator(tokenizer: _FastBPETokenizer, text: str) -> list[int]:
    if not sys.stdout.isatty():
        print("Encoding corpus... this can take a while on large corpora.")
        start_time = time.time()
        token_ids = tokenizer.encode(text)
        print(f"Finished encoding corpus in {_format_duration(time.time() - start_time)}.")
        return token_ids

    done = threading.Event()
    start_time = time.time()

    def render() -> None:
        frames = itertools.cycle(["[#---------------------------]", "[####------------------------]", "[########--------------------]", "[############----------------]", "[################------------]", "[####################--------]", "[########################----]", "[############################]"])
        while not done.wait(0.2):
            elapsed = _format_duration(time.time() - start_time)
            print(
                f"{next(frames)} encoding corpus elapsed={elapsed}",
                end="\r",
                flush=True,
            )

    thread = threading.Thread(target=render, daemon=True)
    thread.start()
    try:
        return tokenizer.encode(text)
    finally:
        done.set()
        thread.join()
        print(" " * 80, end="\r", flush=True)
        print(f"Finished encoding corpus in {_format_duration(time.time() - start_time)}.")


def main() -> None:
    torch.manual_seed(SEED)
    device = _default_device()
    corpus_path = _default_corpus_path()
    use_live_bar = sys.stdout.isatty()

    print(f"Loading corpus from: {corpus_path}", flush=True)
    corpus_text = corpus_path.read_text(encoding="utf-8")
    print("Training tokenizer...", flush=True)
    tokenizer_start = time.time()

    def tokenizer_progress(
        merges_done: int,
        total_merges: int,
        best_pair: tuple[int, int],
        pair_count: int,
    ) -> None:
        line = _render_tokenizer_progress(
            merges_done,
            total_merges,
            best_pair,
            pair_count,
            tokenizer_start,
        )
        if use_live_bar:
            print(line, end="\r", flush=True)
        elif merges_done == 1 or merges_done % 50 == 0 or merges_done == total_merges:
            print(line, flush=True)

    tokenizer = _FastBPETokenizer.train(
        corpus_text,
        vocab_size=VOCAB_SIZE,
        min_pair_freq=2,
        progress_callback=tokenizer_progress,
    )
    if use_live_bar:
        print()
    print("Encoding corpus with learned tokenizer...", flush=True)
    token_ids = _encode_with_indicator(tokenizer, corpus_text)
    all_tokens = torch.tensor(token_ids, dtype=torch.long)

    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=SEQ_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=D_FF,
        dropout=DROPOUT,
    )

    model = TransformerLM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on device: {device}", flush=True)
    train_start = time.time()
    for step in range(1, MAX_STEPS + 1):
        x, y = get_batch(train_tokens, BATCH_SIZE, SEQ_LEN, device)
        loss = train_step(model, optimizer, x, y)
        elapsed = time.time() - train_start
        eta_seconds = (elapsed / step) * (MAX_STEPS - step)
        if use_live_bar:
            print(_render_progress(step, MAX_STEPS, loss, eta_seconds), end="\r", flush=True)
        elif step == 1 or step % 50 == 0 or step == MAX_STEPS:
            print(_render_progress(step, MAX_STEPS, loss, eta_seconds), flush=True)

        if step % EVAL_INTERVAL == 0 or step == 1 or step == MAX_STEPS:
            val_loss = evaluate(
                model,
                val_tokens,
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
                eval_iters=EVAL_ITERS,
            )
            if use_live_bar:
                print()
            print(f"step={step:4d} train_loss={loss:.4f} val_loss={val_loss:.4f}", flush=True)

    if use_live_bar:
        print()

    out_dir = Path(OUT_DIR) if OUT_DIR else Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = out_dir / "tokenizer.json"
    checkpoint_path = out_dir / "checkpoint.pt"

    tokenizer.save(str(tokenizer_path))
    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        step=MAX_STEPS,
        config=config,
    )

    print(f"Saved tokenizer to: {tokenizer_path}", flush=True)
    print(f"Saved checkpoint to: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
