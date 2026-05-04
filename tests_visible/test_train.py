import torch
import torch.nn.functional as F

from mp.model import TransformerConfig, TransformerLM
from mp.train import cross_entropy_from_logits, get_batch, train_step


def _build_tiny_model() -> TransformerLM:
    cfg = TransformerConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    return TransformerLM(cfg)


def test_train_cross_entropy_matches_torch_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3))

    manual = cross_entropy_from_logits(logits, targets)
    reference = F.cross_entropy(logits.view(-1, 5), targets.view(-1))
    torch.testing.assert_close(manual, reference, atol=1e-5, rtol=1e-5)


def test_train_cross_entropy_handles_batched_shapes():
    torch.manual_seed(2)
    logits = torch.randn(3, 4, 2, 7)
    targets = torch.randint(0, 7, (3, 4, 2))

    manual = cross_entropy_from_logits(logits, targets)
    reference = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    torch.testing.assert_close(manual, reference, atol=1e-6, rtol=1e-6)


def test_train_cross_entropy_is_stable_across_random_seeds():
    losses: list[float] = []
    for seed in [0, 1, 7, 11]:
        torch.manual_seed(seed)
        logits = torch.randn(2, 5, 9)
        targets = torch.randint(0, 9, (2, 5))
        loss = cross_entropy_from_logits(logits, targets)
        assert torch.isfinite(loss)
        losses.append(float(loss.item()))

    assert len(set(round(v, 6) for v in losses)) > 1


def test_train_get_batch_produces_shifted_targets():
    torch.manual_seed(0)
    tokens = torch.arange(0, 40, dtype=torch.long)
    x, y = get_batch(tokens, batch_size=6, seq_len=5, device="cpu")
    assert x.shape == (6, 5)
    assert y.shape == (6, 5)
    torch.testing.assert_close(y[:, :-1], x[:, 1:])


def test_train_step_reduces_loss_on_toy_problem():
    torch.manual_seed(0)
    model = _build_tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    # Repeating pattern should be learnable quickly.
    tokens = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=torch.long)

    losses = []
    for _ in range(30):
        x, y = get_batch(tokens, batch_size=4, seq_len=4, device="cpu")
        losses.append(train_step(model, optimizer, x, y))

    assert losses[-1] < losses[0]
