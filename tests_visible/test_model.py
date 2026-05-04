import pytest
import torch
from torch import nn

from mp.model import TransformerConfig, TransformerLM, sinusoidal_position_encoding


def _small_config(dropout: float = 0.0) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=100,
        max_seq_len=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=dropout,
    )


def test_model_forward_returns_expected_shape_and_dtype():
    model = TransformerLM(_small_config())
    x = torch.randint(0, 100, (3, 10), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (3, 10, 100)
    assert logits.dtype == torch.float32


def test_sinusoidal_position_encoding_matches_reference_values():
    enc = sinusoidal_position_encoding(seq_len=3, d_model=6)

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.84147096, 0.54030234, 0.04639922, 0.99892294, 0.00215443, 0.99999768],
                [0.90929741, -0.41614684, 0.09269850, 0.99569422, 0.00430886, 0.99999070],
            ]
        ],
        dtype=torch.float32,
    )

    assert enc.shape == (1, 3, 6)
    assert enc.dtype == torch.float32
    torch.testing.assert_close(enc, expected, atol=1e-5, rtol=1e-5)


def test_model_forward_uses_sinusoidal_positions():
    cfg = TransformerConfig(
        vocab_size=8,
        max_seq_len=8,
        n_layers=0,
        n_heads=2,
        d_model=6,
        d_ff=12,
        dropout=0.0,
    )
    model = TransformerLM(cfg).eval()
    model.drop = nn.Identity()
    model.ln_f = nn.Identity()
    model.lm_head = nn.Identity()

    with torch.no_grad():
        model.token_emb.weight.zero_()

    x = torch.tensor([[3, 3, 3]], dtype=torch.long)
    logits = model(x)
    expected = sinusoidal_position_encoding(seq_len=3, d_model=6).to(x.device)
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)


def test_model_forward_rejects_overlong_sequence():
    model = TransformerLM(_small_config())
    x = torch.randint(0, 100, (2, 17), dtype=torch.long)
    with pytest.raises(ValueError):
        model(x)


def test_model_attention_causal_mask_blocks_future_information():
    model = TransformerLM(_small_config(dropout=0.0)).eval()

    prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    suffix_a = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    suffix_b = torch.tensor([[9, 9, 9, 9]], dtype=torch.long)

    a = torch.cat([prefix, suffix_a], dim=1)
    b = torch.cat([prefix, suffix_b], dim=1)

    logits_a = model(a)
    logits_b = model(b)

    torch.testing.assert_close(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-5, rtol=1e-5)


def test_model_generate_greedy_matches_topk1_decoding():
    torch.manual_seed(0)
    model = TransformerLM(_small_config(dropout=0.0)).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out1 = model.generate(prompt, max_new_tokens=4, temperature=0.0, top_k=None)
    out2 = model.generate(prompt, max_new_tokens=4, temperature=1.0, top_k=1)
    assert torch.equal(out1, out2)


def test_model_generate_respects_max_seq_len_cropping(monkeypatch):
    cfg = TransformerConfig(
        vocab_size=100,
        max_seq_len=4,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )
    model = TransformerLM(cfg).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    seen_context_lengths: list[int] = []
    original_forward = model.forward

    def wrapped_forward(input_ids: torch.Tensor) -> torch.Tensor:
        seen_context_lengths.append(int(input_ids.shape[1]))
        return original_forward(input_ids)

    monkeypatch.setattr(model, "forward", wrapped_forward)

    out = model.generate(prompt, max_new_tokens=3, temperature=0.0, top_k=None)

    assert out.shape == (1, 9)
    assert seen_context_lengths == [4, 4, 4]


def test_model_generate_supports_batch_size_gt1():
    torch.manual_seed(0)
    model = TransformerLM(_small_config(dropout=0.0)).eval()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=2, temperature=0.0, top_k=None)
    assert out.shape == (2, 5)
