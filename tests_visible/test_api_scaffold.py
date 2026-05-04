import inspect

import torch

from mp.model import (
    CausalSelfAttention,
    TransformerBlock,
    TransformerConfig,
    TransformerLM,
    sinusoidal_position_encoding,
)
from mp.tokenizer import BPETokenizer
from mp.train import (
    cross_entropy_from_logits,
    evaluate,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    train_step,
)


def test_api_exports_and_core_entrypoints():
    assert inspect.isclass(BPETokenizer)
    assert inspect.isclass(TransformerConfig)
    assert inspect.isclass(CausalSelfAttention)
    assert inspect.isclass(TransformerBlock)
    assert inspect.isclass(TransformerLM)
    assert callable(sinusoidal_position_encoding)
    assert callable(cross_entropy_from_logits)
    assert callable(get_batch)
    assert callable(train_step)
    assert callable(evaluate)
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)

    tok = BPETokenizer.train("abba", vocab_size=260, min_pair_freq=1)
    ids = tok.encode("abba")

    cfg = TransformerConfig(
        vocab_size=max(tok.vocab_size, 260),
        max_seq_len=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    model = TransformerLM(cfg)
    x = torch.tensor([ids[:4]], dtype=torch.long)
    logits = model(x)
    y = x.clone()
    loss = cross_entropy_from_logits(logits, y)
    assert torch.isfinite(loss)
