"""Pointwise LLM reranker: prompt, encoding and scoring, with no model."""

import math

import pytest

from src.llm_point import (
    fine_tune_point,
    iterate_batches,
    point_record,
    point_targets,
    SUFFIX,
    collate_left,
    encode_pair,
    order_by_scores,
    p_yes,
    query_prefix,
    truncation_share,
    yes_no_ids,
)


class WordTokenizer:
    """One id per whitespace-separated word; enough to test the encoding."""

    pad_token_id = 0
    unk_token_id = 1

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "yes": 2, "no": 3}

    def _id(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word]

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            pass

        encoded = Encoded()
        encoded.input_ids = [self._id(w) for w in text.split()]
        return encoded

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


def test_every_candidate_of_a_query_shares_the_prefix():
    tok = WordTokenizer()
    a, _ = encode_pair(tok, "red running shoes", "Nike Pegasus red", max_length=200)
    b, _ = encode_pair(tok, "red running shoes", "Adidas blue", max_length=200)
    prefix = tok(query_prefix("red running shoes")).input_ids
    assert a[: len(prefix)] == prefix
    assert b[: len(prefix)] == prefix


def test_the_query_comes_before_the_document():
    text = query_prefix("q1") + "doc" + SUFFIX
    assert text.index("q1") < text.index("doc")


def test_only_the_document_is_truncated():
    tok = WordTokenizer()
    long_doc = " ".join(f"w{i}" for i in range(500))
    prefix = tok(query_prefix("shoes")).input_ids
    suffix = tok(SUFFIX).input_ids
    ids, truncated = encode_pair(tok, "shoes", long_doc, max_length=len(prefix) + len(suffix) + 10)
    assert truncated
    assert len(ids) == len(prefix) + len(suffix) + 10
    assert ids[: len(prefix)] == prefix
    assert ids[-len(suffix):] == suffix


def test_a_short_document_is_not_truncated():
    tok = WordTokenizer()
    _, truncated = encode_pair(tok, "shoes", "a short title", max_length=200)
    assert not truncated


def test_a_template_longer_than_the_budget_is_refused():
    tok = WordTokenizer()
    with pytest.raises(ValueError, match="max_length"):
        encode_pair(tok, "shoes", "doc", max_length=5)


def test_left_padding_puts_a_real_token_last_in_every_row():
    ids, mask = collate_left([[5, 6, 7], [8]], pad_id=0)
    assert ids == [[5, 6, 7], [0, 0, 8]]
    assert mask == [[1, 1, 1], [0, 0, 1]]
    assert all(row[-1] == 1 for row in mask)


def test_yes_and_no_must_be_single_known_tokens():
    tok = WordTokenizer()
    assert yes_no_ids(tok) == (2, 3)
    tok.vocab.pop("yes")
    with pytest.raises(ValueError, match="yes"):
        yes_no_ids(tok)


def test_p_yes_is_a_sigmoid_of_the_logit_difference():
    assert p_yes(0.0, 0.0) == pytest.approx(0.5)
    assert p_yes(0.0, 2.0) == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert p_yes(5.0, 1.0) < p_yes(1.0, 5.0)


def test_p_yes_does_not_overflow_on_extreme_logits():
    assert p_yes(-1000.0, 1000.0) == pytest.approx(1.0)
    assert p_yes(1000.0, -1000.0) == pytest.approx(0.0)


def test_order_by_scores_breaks_ties_by_document_id():
    assert order_by_scores(["b", "a", "c"], [0.5, 0.5, 0.9]) == ["c", "a", "b"]


def test_order_by_scores_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="scores"):
        order_by_scores(["a", "b"], [0.1])


@pytest.mark.slow
def test_the_real_reranker_prefers_the_matching_product():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from src.llm_point import DEFAULT_BACKBONE, load_point_model, score_pairs

    model, tok = load_point_model(DEFAULT_BACKBONE)
    match, other = score_pairs(model, tok, [
        ("red running shoes for women", "Nike Women's Pegasus 40 Running Shoe, Red"),
        ("red running shoes for women", "Stainless Steel Kitchen Sink Strainer"),
    ])
    assert match > 0.5 > other


def test_truncation_share_counts_pairs_whose_document_was_cut():
    tok = WordTokenizer()
    long_doc = " ".join(f"w{i}" for i in range(500))
    pairs = [("shoes", "short"), ("shoes", long_doc), ("shoes", "also short")]
    assert truncation_share(tok, pairs, max_length=200) == pytest.approx(1 / 3)


# --- Task 4: the pointwise fine-tune -------------------------------------------

import pandas as pd


def test_point_targets_are_the_gains():
    assert point_targets([1.0, 0.1, 0.01, 0.0]) == [1.0, 0.1, 0.01, 0.0]


def test_point_targets_refuse_a_gain_outside_zero_one():
    with pytest.raises(ValueError, match="gain"):
        point_targets([1.0, 100.0])


def test_point_fine_tune_refuses_fold_zero():
    matrix = pd.DataFrame({"query_id": ["1"], "product_id": ["a"], "gain": [1.0], "fold": [0]})
    with pytest.raises(ValueError):
        fine_tune_point(matrix, {"1": "q"}, {"a": "d"}, out_dir="unused")


def test_batches_cover_every_row_once_per_epoch():
    batches = iterate_batches(10, 3, seed=0)
    assert sorted(i for b in batches for i in b) == list(range(10))
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_batches_are_reproducible_for_a_seed():
    assert iterate_batches(50, 8, seed=1) == iterate_batches(50, 8, seed=1)
    assert iterate_batches(50, 8, seed=1) != iterate_batches(50, 8, seed=2)


def test_point_record_reports_pairs_per_update_and_the_projection():
    record = point_record(
        backbone="b", lora_rank=None, batch_size=8, grad_accum=2, epochs=1,
        learning_rate=1e-5, max_length=256, n_queries=10, n_pairs=3200,
        updates=10, minutes=1.0, peak_reserved_gb=12.0, seed=0,
    )
    assert record["pairs_per_update"] == 16
    assert record["updates_per_second"] == pytest.approx(10 / 60)
    # 3,200 pairs / 16 per update = 200 updates an epoch at 1/6 per second.
    assert record["projected_epoch_hours"] == pytest.approx(200 / (10 / 60) / 3600)
