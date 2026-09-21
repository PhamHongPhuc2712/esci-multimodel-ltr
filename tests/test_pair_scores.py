import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.pair_scores import (
    PAIR_SCORE_COLUMNS,
    bm25_pair_scores,
    bm25_token_lists,
    similarity_scores,
)


class FakeStore:
    """An EmbeddingStore with the two methods pair scoring uses."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = 2):
        self.dim = dim
        self._vectors = vectors

    def lookup(self, wanted):
        matrix = np.full((len(wanted), self.dim), np.nan, dtype=np.float32)
        present = np.zeros(len(wanted), dtype=bool)
        for i, key in enumerate(wanted):
            if key in self._vectors:
                matrix[i] = self._vectors[key]
                present[i] = True
        return matrix, present


# --- similarity_scores ------------------------------------------------------

def test_similarity_is_the_dot_product_of_the_two_vectors():
    store = FakeStore({"p1": [1.0, 0.0], "p2": [0.0, 1.0]})
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = similarity_scores(q, [0, 0], ["p1", "p2"], store)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_each_row_uses_its_own_query_vector():
    store = FakeStore({"p1": [1.0, 0.0]})
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = similarity_scores(q, [0, 1], ["p1", "p1"], store)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_a_key_the_store_does_not_have_scores_nan():
    # Review Focus 5. 20.6% of judged pairs have no image vector. A 0.0 cosine
    # is a score meaning "orthogonal to the query"; NaN is an absence.
    store = FakeStore({"p1": [1.0, 0.0]})
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = similarity_scores(q, [0, 0], ["p1", "missing"], store)
    assert scores[0] == pytest.approx(1.0)
    assert math.isnan(scores[1])


def test_an_empty_key_scores_nan_rather_than_raising():
    # Products with no image URL carry "" rather than a key.
    store = FakeStore({"p1": [1.0, 0.0]})
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    assert math.isnan(similarity_scores(q, [0], [""], store)[0])


def test_scoring_nothing_returns_an_empty_array():
    store = FakeStore({})
    q = np.zeros((0, 2), dtype=np.float32)
    assert similarity_scores(q, [], [], store).shape == (0,)


def test_a_query_vector_of_the_wrong_width_raises():
    store = FakeStore({"p1": [1.0, 0.0]}, dim=2)
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="width"):
        similarity_scores(q, [0], ["p1"], store)


# --- bm25_pair_scores -------------------------------------------------------

def _pairs():
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "query": ["red shoes", "red shoes", "blue hat", "blue hat"],
            "row": [0, 1, 0, 2],
        }
    )


def test_bm25_reads_each_pairs_own_row_from_the_score_vector():
    scores = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out, empty = bm25_pair_scores(
        _pairs(),
        tokenize=lambda qs: [q.split() for q in qs],
        score_query=lambda toks: scores,
    )
    assert out.tolist() == [10.0, 20.0, 10.0, 30.0]
    assert empty == 0


def test_bm25_scores_each_query_once():
    calls = []

    def score_query(toks):
        calls.append(list(toks))
        return np.zeros(3, dtype=np.float32)

    bm25_pair_scores(
        _pairs(), tokenize=lambda qs: [q.split() for q in qs], score_query=score_query
    )
    # Two distinct queries, four rows. Scoring per row would be 20x the work
    # at ~20 judgements per query.
    assert calls == [["red", "shoes"], ["blue", "hat"]]


def test_a_query_that_tokenises_to_nothing_yields_nan_and_is_counted():
    # Review Focus 3. bm25s.get_scores([]) raises IndexError from
    # query_tokens_single[0]; one of fold 0's 4,130 queries is such a query
    # after stopword removal. A 15.7-minute pass must not die on it.
    def score_query(toks):
        assert toks, "an empty token list must never reach get_scores"
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    pairs = _pairs()
    out, empty = bm25_pair_scores(
        pairs,
        tokenize=lambda qs: [[], ["blue", "hat"]],
        score_query=score_query,
    )
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(1.0)
    assert empty == 1


def test_every_query_tokenising_to_nothing_gives_all_nan():
    out, empty = bm25_pair_scores(
        _pairs(),
        tokenize=lambda qs: [[], []],
        score_query=lambda toks: pytest.fail("must not be called"),
    )
    assert np.isnan(out).all()
    assert empty == 2


def test_bm25_result_is_aligned_to_the_input_rows():
    # The frame is deliberately not grouped by query_id: the returned array
    # must line up with the rows as given, not with some internal ordering.
    pairs = pd.DataFrame(
        {
            "query_id": [1, 2, 1],
            "query": ["a", "b", "a"],
            "row": [0, 1, 2],
        }
    )
    out, _ = bm25_pair_scores(
        pairs,
        tokenize=lambda qs: [q.split() for q in qs],
        score_query=lambda toks: np.array([7.0, 8.0, 9.0], dtype=np.float32),
    )
    assert out.tolist() == [7.0, 8.0, 9.0]


def test_progress_is_reported_per_query():
    seen = []
    bm25_pair_scores(
        _pairs(),
        tokenize=lambda qs: [q.split() for q in qs],
        score_query=lambda toks: np.zeros(3, dtype=np.float32),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_a_missing_row_column_raises_rather_than_scoring_zero():
    with pytest.raises(KeyError, match="row"):
        bm25_pair_scores(
            pd.DataFrame({"query_id": [1], "query": ["a"]}),
            tokenize=lambda qs: [["a"]],
            score_query=lambda toks: np.zeros(1, dtype=np.float32),
        )


# --- the Tokenized -> strings conversion ------------------------------------

def test_token_lists_map_ids_back_through_the_vocabulary():
    # bm25s returns per-batch integer ids and a vocab dict; get_scores wants
    # strings. Passing the ids through is accepted and silently wrong - they
    # index the batch's vocabulary, not the index's.
    channel = SimpleNamespace(stemmer=None)
    tokenized = SimpleNamespace(ids=[[1, 0], [0]], vocab={"shoes": 0, "red": 1})
    lists = bm25_token_lists(
        ["red shoes", "shoes"], channel, tokenize=lambda qs, stemmer: tokenized
    )
    assert lists == [["red", "shoes"], ["shoes"]]


def test_token_lists_handle_a_query_that_lost_every_token():
    channel = SimpleNamespace(stemmer=None)
    tokenized = SimpleNamespace(ids=[[], [0]], vocab={"hat": 0})
    assert bm25_token_lists(
        ["the", "hat"], channel, tokenize=lambda qs, stemmer: tokenized
    ) == [[], ["hat"]]


def test_the_score_columns_are_the_three_retrieval_signals():
    assert PAIR_SCORE_COLUMNS == ("bm25_score", "dense_sim", "clip_image_sim")


@pytest.mark.data
def test_the_real_pair_scores_are_shaped_as_the_plan_measured():
    frame = pd.read_parquet("data/features/pair-scores-test.parquet")
    assert len(frame) == 181_701
    assert list(frame.columns) == ["query_id", "product_id", *PAIR_SCORE_COLUMNS]

    # Measured 2026-09-21: dense reaches every judged product, the image store
    # reaches 77.50% of them and ~79% of pairs, BM25 all but a rounding error.
    assert np.isfinite(frame["dense_sim"]).mean() == pytest.approx(1.0)
    assert 0.74 < np.isfinite(frame["clip_image_sim"]).mean() < 0.84
    assert np.isfinite(frame["bm25_score"]).mean() > 0.999

    # Cosines of L2-normalised vectors. Outside [-1, 1] means a store of
    # unnormalised vectors, which silently ranks partly by magnitude.
    for column in ("dense_sim", "clip_image_sim"):
        values = frame[column].to_numpy()
        finite = values[np.isfinite(values)]
        assert finite.min() >= -1.001 and finite.max() <= 1.001
