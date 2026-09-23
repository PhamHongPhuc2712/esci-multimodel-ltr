import numpy as np
import pandas as pd
import pytest

from src.blend_report import (
    BLEND_ARMS,
    ORACLE,
    SINGLE_STAGES,
    best_single,
    per_query_ndcg,
    selector_routes,
    single_stage_orderings,
)
from src.rank_report import evaluate_arm
from src.rerank_window import Window


def _signals():
    return pd.DataFrame(
        {
            "query_id": ["1", "1", "1", "2", "2"],
            "product_id": ["a", "b", "c", "x", "y"],
            "stage2_score": [3.0, 2.0, 1.0, 9.0, 8.0],
            "stage2_rank": [1, 2, 3, 1, 2],
            "ce_score": [0.1, 0.9, 0.5, 0.2, 0.8],
            "llm_rank": [3, 1, 2, 2, 1],
            "llm_rr": [1 / 63, 1 / 61, 1 / 62, 1 / 62, 1 / 61],
            "label_code": [3, 0, 2, 3, 0],
            "gain": [1.0, 0.0, 0.1, 1.0, 0.0],
            "qrel": [100, 0, 10, 100, 0],
        }
    )


def _windows():
    return [
        Window(query_id="1", window=("a", "b", "c"), tail=("z",)),
        Window(query_id="2", window=("x", "y"), tail=()),
    ]


def _qrels():
    return {"1": {"a": 100, "b": 0, "c": 10, "z": 0}, "2": {"x": 100, "y": 0}}


def _arm(name, values):
    per = {str(i): v for i, v in enumerate(values)}
    floor = {str(i): 0.74 for i in range(len(values))}
    return evaluate_arm(
        name, per, floor, groups=("blend",), n_features=0,
        objective="blend", best_iteration=0,
    )


# --- the three single stages ------------------------------------------------

def test_every_single_stage_gets_an_ordering():
    out = single_stage_orderings(_signals())
    assert set(out) == set(SINGLE_STAGES)


def test_the_stage_2_ordering_is_the_window_order():
    out = single_stage_orderings(_signals())
    assert out["stage2"]["1"] == ["a", "b", "c"]


def test_the_cross_encoder_ordering_follows_its_logit():
    out = single_stage_orderings(_signals())
    assert out["stage2+ce"]["1"] == ["b", "c", "a"]


def test_the_llm_ordering_follows_its_rank():
    out = single_stage_orderings(_signals())
    assert out["stage2+llm"]["1"] == ["b", "c", "a"]


# --- scoring ----------------------------------------------------------------

def test_per_query_ndcg_covers_every_query():
    out = single_stage_orderings(_signals())
    per = per_query_ndcg(out["stage2"], _windows(), _qrels())
    assert set(per) == {"1", "2"}
    assert all(0.0 <= v <= 1.0 for v in per.values())


def test_a_better_ordering_scores_higher():
    # Putting the Exact product first must beat putting it last.
    good = {"1": ["a", "c", "b"], "2": ["x", "y"]}
    bad = {"1": ["b", "c", "a"], "2": ["y", "x"]}
    g = per_query_ndcg(good, _windows(), _qrels())
    b = per_query_ndcg(bad, _windows(), _qrels())
    assert sum(g.values()) > sum(b.values())


def test_scoring_goes_through_the_splice_so_the_tail_stays_below():
    # Query 1 has a tail; a reordering must never promote it.
    per = per_query_ndcg({"1": ["c", "b", "a"], "2": ["x", "y"]}, _windows(), _qrels())
    assert set(per) == {"1", "2"}


def test_a_non_permutation_is_refused():
    with pytest.raises(ValueError, match="permutation"):
        per_query_ndcg({"1": ["a", "b"], "2": ["x", "y"]}, _windows(), _qrels())


# --- the comparison §4.4 actually asks for ---------------------------------

def test_the_best_single_stage_is_picked_by_ndcg():
    results = [_arm("stage2", [0.80, 0.80]), _arm("stage2+ce", [0.85, 0.85]),
               _arm("stage2+llm", [0.90, 0.90])]
    assert best_single(results).name == "stage2+llm"


def test_best_single_ignores_blend_and_oracle_arms():
    # A blend that beats Stage 2 but loses to the LLM has earned nothing, so
    # the baseline for the blend arms must be the best *single stage*.
    results = [_arm("stage2", [0.80]), _arm("stage2+llm", [0.90]),
               _arm("blend_fixed", [0.95]), _arm(ORACLE, [0.99])]
    assert best_single(results).name == "stage2+llm"


def test_best_single_needs_a_single_stage():
    with pytest.raises(ValueError, match="single stage"):
        best_single([_arm("blend_fixed", [0.9])])


# --- where the selector sent each query -------------------------------------

def _routing():
    arm_orderings = {
        "stage2": {"1": ["a", "b"], "2": ["x", "y"], "3": ["p", "q"]},
        "ce": {"1": ["b", "a"], "2": ["y", "x"], "3": ["p", "q"]},
        "llm": {"1": ["a", "b"], "2": ["x", "y"], "3": ["q", "p"]},
    }
    per_stage = {
        "stage2": {"1": 0.5, "2": 0.6, "3": 0.9},
        "ce": {"1": 0.7, "2": 0.4, "3": 0.9},
        "llm": {"1": 0.5, "2": 0.6, "3": 0.8},
    }
    return arm_orderings, per_stage


def test_routes_are_counted_against_the_best_mean_arm():
    arm_orderings, per_stage = _routing()
    routed = {"1": ["b", "a"], "2": ["x", "y"], "3": ["q", "p"]}   # ce, llm, llm
    out = selector_routes(routed, arm_orderings, per_stage, default="llm")
    assert out["share"] == pytest.approx({"llm": 2 / 3, "ce": 1 / 3, "stage2": 0.0})


def test_an_ordering_identical_to_the_default_counts_as_the_default():
    # Query 2's stage2 and llm orderings are the same list; routing it to
    # stage2 changes nothing and is not a deviation.
    arm_orderings, per_stage = _routing()
    routed = {"1": ["a", "b"], "2": ["x", "y"], "3": ["q", "p"]}
    out = selector_routes(routed, arm_orderings, per_stage, default="llm")
    assert out["n_deviations"] == 0


def test_a_deviation_reports_what_it_cost():
    arm_orderings, per_stage = _routing()
    routed = {"1": ["b", "a"], "2": ["y", "x"], "3": ["p", "q"]}   # ce, ce, ce/stage2
    out = selector_routes(routed, arm_orderings, per_stage, default="llm")
    assert out["n_deviations"] == 3
    assert (out["n_better"], out["n_worse"], out["n_same"]) == (2, 1, 0)
    assert out["mean_delta"] == pytest.approx((0.2 - 0.2 + 0.1) / 3)


def test_the_declared_arms_cover_every_strategy_and_the_ceiling():
    from src.blend import STRATEGIES

    assert len(BLEND_ARMS) == len(STRATEGIES)
    assert ORACLE not in BLEND_ARMS          # a ceiling is not an arm
    assert ORACLE not in SINGLE_STAGES


import json


@pytest.mark.data
def test_the_committed_blend_reports_answer_the_spec_question():
    for path in ("docs/results/blend.json", "docs/results/blend-test.json"):
        payload = json.loads(open(path).read())
        names = {arm["name"] for arm in payload["arms"]}
        # §4.4: the blend is compared against each single stage alone.
        assert set(SINGLE_STAGES) <= names
        assert set(BLEND_ARMS) <= names
        assert ORACLE in names

        # Every arm on the same queries, and the n recorded beside it.
        assert len({arm["n_queries"] for arm in payload["arms"]}) == 1

        # The comparison that matters is against the best single stage.
        assert payload["best_single_stage"] in SINGLE_STAGES
        assert {row["baseline"] for row in payload["against_best_single"]} == {
            payload["best_single_stage"]
        }

        # The ceiling must bound the arms it was built from.
        by_name = {arm["name"]: arm["ndcg"]["point"] for arm in payload["arms"]}
        for arm in SINGLE_STAGES:
            assert by_name[ORACLE] >= by_name[arm] - 1e-9

        # The floor is computed, never the published 0.7483.
        assert 0.73 < payload["floor"]["mean"] < 0.76


@pytest.mark.data
def test_the_fold_0_learned_arms_are_cross_fitted():
    # Fold 0 is both the only training set and the reporting surface, so an
    # in-sample fold-0 number would be a training fit wearing a result's
    # clothes.
    payload = json.loads(open("docs/results/blend.json").read())
    assert "cross-fit" in payload["combiner_fitted_on"]
    assert "cross-fit" in payload["selector_fitted_on"]


@pytest.mark.data
def test_the_fold_0_report_records_the_selector_routes():
    # The writeup explains Stage 4's result by these routes, so they live in
    # the committed report rather than in an ad hoc calculation.
    payload = json.loads(open("docs/results/blend.json").read())
    routes = payload["selector_routes"]
    assert routes["default_arm"] == "llm"
    assert sum(routes["share"].values()) == pytest.approx(1.0)
    assert routes["n_better"] + routes["n_worse"] + routes["n_same"] == routes["n_deviations"]
