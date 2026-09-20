import pandas as pd
import pytest

from src.recall import (
    DEFAULT_KS,
    RELEVANT_SETS,
    recall_at_k,
    recall_table,
    relevant_sets,
)


def _judgements(rows):
    return pd.DataFrame(rows)


# --- the ground truth -------------------------------------------------------

def test_relevant_sets_default_to_exact_only():
    frame = _judgements([
        {"query_id": 1, "product_id": "a", "esci_label": "E"},
        {"query_id": 1, "product_id": "b", "esci_label": "S"},
        {"query_id": 1, "product_id": "c", "esci_label": "I"},
    ])
    assert relevant_sets(frame) == {1: {"a"}}


def test_relevant_sets_can_include_substitutes():
    frame = _judgements([
        {"query_id": 1, "product_id": "a", "esci_label": "E"},
        {"query_id": 1, "product_id": "b", "esci_label": "S"},
        {"query_id": 1, "product_id": "c", "esci_label": "C"},
    ])
    assert relevant_sets(frame, relevance="E+S") == {1: {"a", "b"}}


def test_relevant_sets_rejects_an_unknown_relevance():
    frame = _judgements([{"query_id": 1, "product_id": "a", "esci_label": "E"}])
    with pytest.raises(ValueError, match="relevance"):
        relevant_sets(frame, relevance="everything")


def test_the_relevance_definitions_are_the_spec_s():
    assert RELEVANT_SETS == {"E": ("E",), "E+S": ("E", "S")}


def test_a_query_with_no_relevant_product_is_absent_from_the_sets():
    # query_id 45928 in the real test split: 15 judgements, all S, no E.
    frame = _judgements([
        {"query_id": 45928, "product_id": "a", "esci_label": "S"},
        {"query_id": 45928, "product_id": "b", "esci_label": "S"},
    ])
    assert relevant_sets(frame) == {}
    assert relevant_sets(frame, relevance="E+S") == {45928: {"a", "b"}}


# --- the metric -------------------------------------------------------------

def test_recall_counts_the_share_of_relevant_products_found():
    result = recall_at_k({1: ["x", "a", "y", "b"]}, {1: {"a", "b", "c"}}, k=4)
    assert result.mean == pytest.approx(2 / 3)


def test_recall_respects_k():
    # "b" sits at rank 4 and must not count at k=2.
    result = recall_at_k({1: ["x", "a", "y", "b"]}, {1: {"a", "b"}}, k=2)
    assert result.mean == pytest.approx(0.5)


def test_recall_of_a_perfect_retrieval_is_one():
    result = recall_at_k({1: ["a", "b"]}, {1: {"a", "b"}}, k=10)
    assert result.mean == pytest.approx(1.0)


def test_recall_averages_over_queries_not_over_products():
    # A query with 1 relevant found and a query with 0 of 10 found average to
    # 0.5, not to 1/11. Macro-averaging is what the spec's Recall@k means.
    result = recall_at_k(
        {1: ["a"], 2: []},
        {1: {"a"}, 2: {f"p{i}" for i in range(10)}},
        k=10,
    )
    assert result.mean == pytest.approx(0.5)
    assert result.n_queries == 2


def test_a_duplicate_in_the_retrieved_list_is_not_counted_twice():
    result = recall_at_k({1: ["a", "a"]}, {1: {"a", "b"}}, k=10)
    assert result.mean == pytest.approx(0.5)


# --- Review Focus 1: a query with no relevant product -----------------------

def test_a_query_with_no_relevant_product_is_skipped_not_scored_zero():
    # 0/0 is undefined. Scoring it 0.0 punishes a retriever for a query where
    # no retrieval could have scored, and drags the reported mean down.
    result = recall_at_k({1: ["a"], 2: ["x"]}, {1: {"a"}}, k=10)
    assert result.mean == pytest.approx(1.0)
    assert result.n_queries == 1
    assert result.n_skipped == 1


def test_the_skipped_count_is_reported_so_denominators_are_comparable():
    result = recall_at_k({1: ["a"], 2: ["x"]}, {1: {"a"}}, k=10)
    payload = result.to_dict()
    assert payload["n_queries"] == 1
    assert payload["n_skipped"] == 1
    assert payload["k"] == 10


def test_a_query_whose_relevant_set_is_empty_is_also_skipped():
    result = recall_at_k({1: ["a"]}, {1: set()}, k=10)
    assert result.n_queries == 0
    assert result.n_skipped == 1


def test_every_query_being_skipped_gives_a_nan_mean_not_a_zero():
    # A silent 0.0 here would read as "the retriever found nothing" when the
    # truth is "nothing was measurable".
    import math

    result = recall_at_k({1: ["a"]}, {}, k=10)
    assert result.n_queries == 0
    assert math.isnan(result.mean)


# --- the table --------------------------------------------------------------

def test_recall_table_covers_every_k():
    table = recall_table({1: ["a", "b", "c"]}, {1: {"a", "c"}})
    assert set(table) == set(DEFAULT_KS)
    assert table[10].mean == pytest.approx(1.0)


def test_recall_is_monotonic_in_k():
    retrieved = {1: [f"p{i}" for i in range(200)]}
    relevant = {1: {"p5", "p150"}}
    table = recall_table(retrieved, relevant, ks=(10, 100, 1000))
    assert table[10].mean <= table[100].mean <= table[1000].mean
    assert table[10].mean == pytest.approx(0.5)
    assert table[1000].mean == pytest.approx(1.0)


def test_per_query_scores_are_keyed_for_the_bootstrap():
    # src.bootstrap.paired_delta_ci pairs on the mapping's keys, so the keys
    # must be the query ids and must survive as strings or ints consistently.
    result = recall_at_k({1: ["a"], 2: ["b"]}, {1: {"a"}, 2: {"b"}}, k=10)
    assert set(result.per_query) == {1, 2}
    assert result.per_query[1] == pytest.approx(1.0)
