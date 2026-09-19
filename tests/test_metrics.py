import math

import pytest

from src.metrics import mean_ndcg, ndcg_per_query


def test_perfect_ranking_scores_one():
    qrels = {"q": {"a": 100, "b": 10, "c": 0}}
    run = {"q": {"a": 0.9, "b": 0.8, "c": 0.7}}
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(1.0)


def test_reversed_ranking_scores_the_hand_computed_value():
    qrels = {"q": {"a": 100, "b": 10, "c": 0}}
    run = {"q": {"c": 0.9, "b": 0.8, "a": 0.7}}
    dcg = 0 / math.log2(2) + 10 / math.log2(3) + 100 / math.log2(4)
    idcg = 100 / math.log2(2) + 10 / math.log2(3) + 0 / math.log2(4)
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(dcg / idcg)


def test_gain_is_linear_not_exponential():
    # Gains 1 and 0 map to 1 and 0 under 2**rel - 1 as well, so they cannot
    # tell the two forms apart. Gains 2 and 1 become 3 and 1, which can.
    # trec_eval -m ndcg uses the linear form.
    qrels = {"q": {"a": 2, "b": 1}}
    run = {"q": {"b": 1.0, "a": 0.5}}
    linear = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    exponential = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert linear != pytest.approx(exponential)
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(linear)


def test_ndcg_is_invariant_to_gain_scale():
    # Justifies using integer qrels (100/10/1/0) for pytrec_eval while the
    # spec states float gains (1.0/0.1/0.01/0.0).
    run = {"q": {"a": 3.0, "b": 2.0, "c": 1.0}}
    as_ints = {"q": {"a": 10, "b": 100, "c": 1}}
    as_floats = {"q": {"a": 0.1, "b": 1.0, "c": 0.01}}
    assert ndcg_per_query(run, as_ints)["q"] == pytest.approx(
        ndcg_per_query(run, as_floats)["q"]
    )


# --- Review Focus 1: partial runs -------------------------------------------

def test_judged_documents_the_run_omitted_still_count_against_it():
    qrels = {"q": {"a": 100, "b": 100}}
    full = {"q": {"a": 2.0, "b": 1.0}}
    partial = {"q": {"a": 2.0}}  # b was never retrieved
    assert ndcg_per_query(full, qrels)["q"] == pytest.approx(1.0)
    idcg = 100 / math.log2(2) + 100 / math.log2(3)
    assert ndcg_per_query(partial, qrels)["q"] == pytest.approx(100 / idcg)


def test_query_absent_from_the_run_raises():
    qrels = {"q1": {"a": 100}, "q2": {"b": 100}}
    run = {"q1": {"a": 1.0}}
    with pytest.raises(KeyError, match="q2"):
        ndcg_per_query(run, qrels)


# --- Review Focus 2: unjudged documents -------------------------------------

def test_unjudged_documents_count_as_zero_gain_and_consume_a_rank():
    qrels = {"q": {"a": 100}}
    only_judged = {"q": {"a": 2.0}}
    with_intruder = {"q": {"x": 3.0, "a": 2.0}}  # x has no judgement
    assert ndcg_per_query(only_judged, qrels)["q"] == pytest.approx(1.0)
    assert ndcg_per_query(with_intruder, qrels)["q"] == pytest.approx(1 / math.log2(3))


# --- Review Focus 3: no positive gain ---------------------------------------

def test_query_with_no_relevant_document_scores_zero_not_nan():
    qrels = {"q": {"a": 0, "b": 0}}
    run = {"q": {"a": 2.0, "b": 1.0}}
    score = ndcg_per_query(run, qrels)["q"]
    assert score == 0.0
    assert not math.isnan(score)


def test_an_all_irrelevant_query_does_not_poison_the_mean():
    qrels = {"good": {"a": 100}, "hopeless": {"b": 0}}
    run = {"good": {"a": 1.0}, "hopeless": {"b": 1.0}}
    assert mean_ndcg(run, qrels) == pytest.approx(0.5)


# --- Review Focus 4: ties ---------------------------------------------------

def test_ties_break_by_ascending_document_id():
    qrels = {"q": {"a": 0, "b": 100}}
    run = {"q": {"a": 1.0, "b": 1.0}}
    # "a" sorts first, so the relevant document lands at rank 2.
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(1 / math.log2(3))


def test_tie_order_does_not_depend_on_dict_insertion_order():
    qrels = {"q": {"a": 0, "b": 100}}
    one = {"q": {"a": 1.0, "b": 1.0}}
    other = {"q": {"b": 1.0, "a": 1.0}}
    assert ndcg_per_query(one, qrels)["q"] == ndcg_per_query(other, qrels)["q"]


def test_a_fully_tied_run_gets_no_credit_for_a_lucky_tie_break():
    # The relevant document is "c", so ascending-id tie-breaking puts it last
    # and the run scores 0.5. Naming the relevant document "a" instead would
    # make this pass at 1.0 by luck rather than by the ranker being right.
    qrels = {"q": {"a": 0, "b": 0, "c": 100}}
    run = {"q": {"a": 1.0, "b": 1.0, "c": 1.0}}
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(1 / math.log2(4))


# --- aggregation ------------------------------------------------------------

def test_mean_is_over_queries_not_over_documents():
    qrels = {"q1": {"a": 100}, "q2": {"b": 100, "c": 100, "d": 100}}
    run = {"q1": {"a": 1.0}, "q2": {"b": 3.0, "c": 2.0, "d": 1.0}}
    # Both queries are perfectly ranked; q2 having 3 documents must not
    # weight it more heavily than q1.
    assert mean_ndcg(run, qrels) == pytest.approx(1.0)


def test_mean_of_empty_qrels_raises():
    with pytest.raises(ValueError, match="no queries"):
        mean_ndcg({}, {})
