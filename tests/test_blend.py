import numpy as np
import pandas as pd
import pytest

from src.blend import (
    DEFAULT_WEIGHTS,
    STRATEGIES,
    cross_fit_predict,
    cross_fit_select,
    fixed_weight_ordering,
    oracle_ordering,
    orderings_from_scores,
    predict_combiner,
    selector_ordering,
    selector_targets,
    train_combiner,
    train_selector,
)
from src.stage_signals import BLEND_FEATURES


def _frame(n_queries=60, seed=0):
    """A signal frame where the label follows ce_score, with noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        for position in range(1, 5):
            code = int(rng.integers(0, 4))
            rows.append(
                {
                    "query_id": str(q),
                    "product_id": f"p{q}_{position}",
                    "stage2_score": rng.normal(),
                    "stage2_rank": position,
                    "ce_score": code + rng.normal(scale=0.3),
                    "llm_rank": int(rng.integers(1, 5)),
                    "llm_rr": 0.0,
                    "label_code": code,
                    "gain": [0.0, 0.01, 0.1, 1.0][code],
                    "qrel": [0, 1, 10, 100][code],
                }
            )
    frame = pd.DataFrame(rows)
    frame["llm_rr"] = 1.0 / (60 + frame["llm_rank"])
    return frame


# --- orderings --------------------------------------------------------------

def test_an_ordering_is_one_permutation_per_query():
    frame = _frame(n_queries=5)
    out = orderings_from_scores(frame, "ce_score")
    assert set(out) == set(frame["query_id"])
    for q, docs in out.items():
        assert sorted(docs) == sorted(frame.loc[frame["query_id"] == q, "product_id"])


def test_higher_scores_come_first():
    frame = _frame(n_queries=1)
    out = orderings_from_scores(frame, "ce_score")["0"]
    scores = frame.set_index("product_id")["ce_score"]
    assert list(out) == list(scores.sort_values(ascending=False).index)


def test_rank_columns_sort_ascending():
    # A rank of 1 is best, so the ordering must not put 4 first.
    frame = _frame(n_queries=1)
    out = orderings_from_scores(frame, "stage2_rank", ascending=True)["0"]
    assert out[0] == frame.sort_values("stage2_rank")["product_id"].iat[0]


def test_ties_break_on_document_id():
    frame = pd.DataFrame({
        "query_id": ["1", "1"], "product_id": ["b", "a"], "ce_score": [1.0, 1.0],
    })
    assert orderings_from_scores(frame, "ce_score")["1"] == ["a", "b"]


# --- the fixed-weight arm ---------------------------------------------------

def test_the_fixed_weight_arm_returns_permutations():
    frame = _frame(n_queries=10)
    out = fixed_weight_ordering(frame)
    for q, docs in out.items():
        assert sorted(docs) == sorted(frame.loc[frame["query_id"] == q, "product_id"])


def test_weighting_one_stage_to_the_exclusion_of_the_rest_reproduces_it():
    # A sanity anchor: all weight on the LLM must give the LLM's own ordering.
    frame = _frame(n_queries=8)
    out = fixed_weight_ordering(frame, {"stage2": 0.0, "ce": 0.0, "llm": 1.0})
    expected = orderings_from_scores(frame, "llm_rank", ascending=True)
    assert out == expected


def test_the_default_weights_favour_the_llm():
    # Measured: the best swept fusion was 1:1:4, and it still lost to the LLM
    # alone. The default records the sweep, not a hope.
    assert DEFAULT_WEIGHTS["llm"] > DEFAULT_WEIGHTS["ce"]
    assert DEFAULT_WEIGHTS["llm"] > DEFAULT_WEIGHTS["stage2"]


# --- the learned combiner ---------------------------------------------------

def test_the_combiner_learns_the_informative_feature():
    frame = _frame(n_queries=120)
    booster = train_combiner(frame)
    frame = frame.assign(blend=predict_combiner(booster, frame))
    exact = frame.loc[frame["label_code"] == 3, "blend"].mean()
    irrelevant = frame.loc[frame["label_code"] == 0, "blend"].mean()
    assert exact > irrelevant


def test_the_combiner_uses_the_declared_features_only():
    frame = _frame(n_queries=40)
    booster = train_combiner(frame)
    assert list(booster.feature_name()) == list(BLEND_FEATURES)


def test_the_combiner_refuses_a_label_gain_override():
    # src.ranker derives label_gain from src.labels and refuses an override;
    # the blend must not be the place that reintroduces LightGBM's default.
    frame = _frame(n_queries=20)
    with pytest.raises(ValueError, match="label_gain"):
        train_combiner(frame, params={"label_gain": [0, 1, 3, 7]})


# --- Review Focus 1: no query scored by a model that trained on it ---------

def test_cross_fit_scores_every_row_exactly_once():
    frame = _frame(n_queries=40)
    scores = cross_fit_predict(frame)
    assert len(scores) == len(frame)
    assert np.isfinite(scores).all()


def test_cross_fit_does_not_let_a_query_score_itself():
    # The label here is a pure function of a feature the model can memorise
    # through query_id-shaped noise. A leaking implementation reproduces the
    # training fit; a correct one is measurably worse than it.
    frame = _frame(n_queries=60, seed=3)
    booster = train_combiner(frame)
    in_sample = predict_combiner(booster, frame)
    out_of_sample = cross_fit_predict(frame, seed=3)
    assert not np.allclose(in_sample, out_of_sample)


def test_cross_fit_partitions_by_query_not_by_row():
    # Splitting by row would put four documents of one query on both sides,
    # which is the same leak wearing a different hat.
    frame = _frame(n_queries=20)
    parts = cross_fit_predict(frame, n_folds=2, seed=0, _return_parts=True)
    for part in parts:
        assert len(set(part)) == len(part)
    assert set().union(*[set(p) for p in parts]) == set(frame["query_id"])
    assert not set(parts[0]) & set(parts[1])


def test_cross_fit_needs_more_queries_than_folds():
    with pytest.raises(ValueError, match="folds"):
        cross_fit_predict(_frame(n_queries=1), n_folds=2)


# --- the selector -----------------------------------------------------------

def _per_arm():
    return {
        "stage2": {"0": 0.9, "1": 0.2, "2": 0.5},
        "ce": {"0": 0.4, "1": 0.8, "2": 0.5},
        "llm": {"0": 0.1, "1": 0.3, "2": 0.9},
    }


def _arm_orderings():
    return {
        "stage2": {"0": ["a", "b"], "1": ["a", "b"], "2": ["a", "b"]},
        "ce": {"0": ["b", "a"], "1": ["b", "a"], "2": ["b", "a"]},
        "llm": {"0": ["a", "b"], "1": ["b", "a"], "2": ["a", "b"]},
    }


def test_the_oracle_takes_the_best_arm_per_query():
    out = oracle_ordering(_per_arm(), _arm_orderings())
    assert out["0"] == ["a", "b"]   # stage2 wins query 0
    assert out["1"] == ["b", "a"]   # ce wins query 1
    assert out["2"] == ["a", "b"]   # llm wins query 2


def test_the_oracle_is_a_ceiling_not_an_arm():
    # It reads the labels, so it can only ever bound a method, never be one.
    per_arm = _per_arm()
    out = oracle_ordering(per_arm, _arm_orderings())
    assert set(out) == set(per_arm["stage2"])


def test_the_selector_returns_a_permutation_from_one_of_the_arms():
    frame = _frame(n_queries=30)
    per_arm = {
        arm: {q: float(np.random.default_rng(i).random()) for q in frame["query_id"].unique()}
        for i, arm in enumerate(("stage2", "ce", "llm"))
    }
    arm_orderings = {
        arm: orderings_from_scores(frame, column, ascending=ascending)
        for arm, column, ascending in [
            ("stage2", "stage2_rank", True),
            ("ce", "ce_score", False),
            ("llm", "llm_rank", True),
        ]
    }
    selector = train_selector(frame, per_arm)
    out = selector_ordering(selector, frame, arm_orderings)
    for q, docs in out.items():
        assert docs in [arm_orderings[arm][q] for arm in ("stage2", "ce", "llm")]


def test_a_strict_winner_is_the_target():
    per_arm = {
        "stage2": {"0": 0.9, "1": 0.2, "2": 0.5},
        "ce": {"0": 0.4, "1": 0.8, "2": 0.5},
        "llm": {"0": 0.1, "1": 0.3, "2": 0.9},
    }
    assert list(selector_targets(per_arm, ["0", "1", "2"])) == [0, 1, 2]


def test_a_tie_goes_to_the_arm_with_the_best_mean_not_the_first_listed():
    # Measured on fold 0: argmax hands every tie to stage2, the first-listed
    # and weakest arm, labelling 31.7% of queries "stage2" when it is strictly
    # best on 17.7%. A tied query scores the same whichever tied arm routes
    # it, so it should teach the default, not the list order.
    per_arm = {
        "stage2": {"tie": 0.8, "a": 0.5, "b": 0.5},
        "ce": {"tie": 0.8, "a": 0.5, "b": 0.5},
        "llm": {"tie": 0.8, "a": 0.9, "b": 0.9},
    }
    assert list(selector_targets(per_arm, ["tie", "a", "b"])) == [2, 2, 2]


def test_a_query_missing_from_the_arm_scores_is_an_error_not_a_zero():
    per_arm = {"stage2": {"0": 0.5}, "ce": {"0": 0.5}, "llm": {}}
    with pytest.raises(KeyError, match="0"):
        selector_targets(per_arm, ["0"])


def _selector_inputs(n_queries=40):
    frame = _frame(n_queries=n_queries)
    per_arm = {
        arm: {q: float(np.random.default_rng(i).random()) for q in frame["query_id"].unique()}
        for i, arm in enumerate(("stage2", "ce", "llm"))
    }
    arm_orderings = {
        arm: orderings_from_scores(frame, column, ascending=ascending)
        for arm, column, ascending in [
            ("stage2", "stage2_rank", True),
            ("ce", "ce_score", False),
            ("llm", "llm_rank", True),
        ]
    }
    return frame, per_arm, arm_orderings


def test_cross_fit_select_routes_every_query_to_one_arms_ordering():
    frame, per_arm, arm_orderings = _selector_inputs()
    out = cross_fit_select(frame, per_arm, arm_orderings)
    assert set(out) == set(frame["query_id"])
    for q, docs in out.items():
        assert docs in [arm_orderings[arm][q] for arm in ("stage2", "ce", "llm")]


def test_cross_fit_select_uses_the_combiner_s_query_partition():
    # Review Focus 1 for the selector. Its target is per-query NDCG - the
    # label in another form - so a selector trained on all of fold 0 and
    # scored on fold 0 reports its own training routes. Sharing the combiner's
    # partition also keeps the two learned arms' fold-0 numbers comparable.
    frame, per_arm, arm_orderings = _selector_inputs()
    assert cross_fit_select(
        frame, per_arm, arm_orderings, seed=4, _return_parts=True
    ) == cross_fit_predict(frame, seed=4, _return_parts=True)


def test_the_declared_strategies_are_the_three_the_plan_measures():
    assert STRATEGIES == ("fixed", "combiner", "selector")
