import pytest

from mp.tokenizer import BPETokenizer


def test_tokenizer_roundtrip_empty_string():
    tok = BPETokenizer.train("", vocab_size=260, min_pair_freq=2)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_tokenizer_training_uses_deterministic_merge_tiebreak():
    # Pair counts: ('a','b') and ('b','a') both appear 3 times.
    # Tie-break should choose lexicographically smaller pair ('a','b').
    text = "abababa"
    tok = BPETokenizer.train(text, vocab_size=257, min_pair_freq=1)
    assert tok.merges[0] == (ord("a"), ord("b"))


def test_tokenizer_roundtrip_unicode_text():
    text = "Hello, 世界! cafe"
    tok = BPETokenizer.train(text, vocab_size=280, min_pair_freq=2)
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_tokenizer_encode_uses_merge_priority_and_non_overlapping_merges():
    priority_tok = BPETokenizer([(ord("a"), ord("b")), (ord("b"), ord("c"))])
    assert priority_tok.encode("abc") == [256, ord("c")]

    non_overlap_tok = BPETokenizer([(ord("a"), ord("a"))])
    assert non_overlap_tok.encode("aaa") == [256, ord("a")]


def test_tokenizer_train_rejects_vocab_size_below_256():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", vocab_size=255, min_pair_freq=1)
