import numpy as np
import pandas as pd
import pytest

from src.feature_matrix import LABEL_CODES
from src.labels import ESCI_GAINS
from src.metrics import ndcg_per_query
from src.ranker import (
    EARLY_STOP_FOLD,
    LABEL_GAIN,
    LABEL_ORDER,
    OBJECTIVES,
    REPORT_FOLD,
    TRAIN_FOLDS,
    TrainedRanker,
    folds,
    predict,
    predict_run,
    train_ranker,
)


def _frame(n_queries=40, seed=0, fold=0):
    """A small ranking frame with one informative feature."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        for p in range(8):
            code = int(rng.choice([0, 1, 2, 3], p=[0.17, 0.05, 0.35, 0.43]))
            rows.append(
                {
                    "query_id": q,
                    "product_id": f"p{q}_{p}",
                    "label_code": code,
                    "gain": ESCI_GAINS[LABEL_ORDER[code]],
                    "qrel": round(ESCI_GAINS[LABEL_ORDER[code]] * 100),
                    "fold": fold,
                    "signal": code + rng.normal(scale=0.3),
                    "noise": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


FEATURES = ["signal", "noise"]


# --- Review Focus 1: the gain mapping --------------------------------------

def test_label_gain_is_derived_from_the_projects_own_mapping():
    assert LABEL_ORDER == ("I", "C", "S", "E")
    assert LABEL_GAIN == [ESCI_GAINS[label] for label in LABEL_ORDER]
    assert LABEL_GAIN == [0.0, 0.01, 0.1, 1.0]


def test_the_gain_mapping_is_not_lightgbms_default():
    # LightGBM defaults to 2**rel - 1 = [0, 1, 3, 7]. On a matrix of the real
    # shape the same booster reports ndcg@10 0.7931 under ESCI's gains and
    # 0.8575 under that default - a 6.4-point gap, larger than any effect in
    # the ablation table.
    assert LABEL_GAIN != [0, 1, 3, 7]


def test_label_gain_agrees_with_the_matrix_label_codes():
    for label, code in LABEL_CODES.items():
        assert LABEL_GAIN[code] == ESCI_GAINS[label]


def test_overriding_the_gain_mapping_raises():
    train, valid = _frame(), _frame(seed=1)
    with pytest.raises(ValueError, match="label_gain"):
        train_ranker(train, valid, FEATURES, params={"label_gain": [0, 1, 3, 7]})


def test_the_trained_ranker_reports_no_ndcg_of_its_own():
    # The only NDCG in this project comes from src.metrics. A number lifted
    # from a training log is a different metric wearing the same name.
    ranker = train_ranker(_frame(), _frame(seed=1), FEATURES, num_boost_round=5)
    assert not hasattr(ranker, "ndcg")
    assert not hasattr(ranker, "score")


def test_truncation_covers_the_longest_real_query():
    # The largest train query has 188 judged candidates; LightGBM's default
    # truncation of 30 would ignore two thirds of it.
    from src.ranker import LAMBDARANK_PARAMS

    assert LAMBDARANK_PARAMS["lambdarank_truncation_level"] >= 188


# --- training ---------------------------------------------------------------

def test_lambdarank_learns_the_informative_feature():
    train, valid = _frame(n_queries=120), _frame(n_queries=60, seed=1)
    ranker = train_ranker(train, valid, FEATURES, num_boost_round=60)
    importance = dict(
        zip(ranker.booster.feature_name(), ranker.booster.feature_importance("gain"))
    )
    assert importance["signal"] > importance["noise"]


def test_every_objective_trains_and_predicts_one_score_per_row():
    train, valid = _frame(n_queries=60), _frame(n_queries=30, seed=1)
    for objective in OBJECTIVES:
        ranker = train_ranker(
            train, valid, FEATURES, objective=objective, num_boost_round=10
        )
        scores = predict(ranker, valid)
        assert scores.shape == (len(valid),), objective
        assert np.isfinite(scores).all(), objective


def test_the_multiclass_arm_is_scored_by_expected_gain():
    # A pointwise classifier emits four probabilities. Collapsing them with
    # argmax would throw away the graded structure the whole metric is about;
    # the expected gain keeps it.
    train, valid = _frame(n_queries=60), _frame(n_queries=30, seed=1)
    ranker = train_ranker(
        train, valid, FEATURES, objective="pointwise_class", num_boost_round=10
    )
    scores = predict(ranker, valid)
    assert scores.min() >= -1e-9
    assert scores.max() <= 1.0 + 1e-9


def test_an_unknown_objective_raises():
    with pytest.raises(ValueError, match="objective"):
        train_ranker(_frame(), _frame(seed=1), FEATURES, objective="pairwise")


def test_training_uses_only_the_named_features():
    train, valid = _frame(), _frame(seed=1)
    ranker = train_ranker(train, valid, ["signal"], num_boost_round=5)
    assert ranker.features == ("signal",)
    assert ranker.booster.feature_name() == ["signal"]


def test_training_is_deterministic_for_a_seed():
    train, valid = _frame(n_queries=60), _frame(n_queries=30, seed=1)
    a = predict(train_ranker(train, valid, FEATURES, num_boost_round=20, seed=7), valid)
    b = predict(train_ranker(train, valid, FEATURES, num_boost_round=20, seed=7), valid)
    np.testing.assert_allclose(a, b)


def test_training_sorts_the_frame_before_grouping():
    # Review Focus 2, one layer up: a caller handing over an unsorted frame
    # must get a correct model, not a silently mis-grouped one.
    train = _frame(n_queries=60).sample(frac=1.0, random_state=0)
    valid = _frame(n_queries=30, seed=1).sample(frac=1.0, random_state=0)
    ranker = train_ranker(train, valid, FEATURES, num_boost_round=10)
    assert predict(ranker, valid).shape == (len(valid),)


# --- predict_run ------------------------------------------------------------

def test_predict_run_is_scoreable_by_src_metrics():
    train, valid = _frame(n_queries=120), _frame(n_queries=60, seed=1)
    ranker = train_ranker(train, valid, FEATURES, num_boost_round=60)
    run = predict_run(ranker, valid)

    qrels: dict[str, dict[str, int]] = {}
    for q, p, r in zip(valid["query_id"], valid["product_id"], valid["qrel"]):
        qrels.setdefault(str(q), {})[str(p)] = int(r)

    per_query = ndcg_per_query(run, qrels)
    assert len(per_query) == valid["query_id"].nunique()
    # A model that learned the informative feature must beat a random ordering,
    # whose full-list NDCG on this label mix sits near 0.75.
    assert sum(per_query.values()) / len(per_query) > 0.80


def test_predict_run_keys_are_strings():
    # pytrec_eval and src.metrics both key on strings; query_id is int64.
    train, valid = _frame(), _frame(seed=1)
    run = predict_run(train_ranker(train, valid, FEATURES, num_boost_round=5), valid)
    assert all(isinstance(q, str) for q in run)
    assert all(isinstance(p, str) for docs in run.values() for p in docs)


def test_predict_run_scores_every_judged_pair():
    train, valid = _frame(), _frame(seed=1)
    run = predict_run(train_ranker(train, valid, FEATURES, num_boost_round=5), valid)
    assert sum(len(docs) for docs in run.values()) == len(valid)


# --- the fold protocol ------------------------------------------------------

def test_the_three_fold_roles_are_disjoint():
    # Selecting, early-stopping and reporting on one fold is how a plan ends
    # up reporting the number it optimised.
    assert REPORT_FOLD not in TRAIN_FOLDS
    assert EARLY_STOP_FOLD not in TRAIN_FOLDS
    assert REPORT_FOLD != EARLY_STOP_FOLD


def test_folds_selects_the_requested_folds():
    frame = pd.concat([_frame(fold=f, seed=f) for f in range(5)], ignore_index=True)
    assert set(folds(frame, TRAIN_FOLDS)["fold"]) == set(TRAIN_FOLDS)
    assert set(folds(frame, [REPORT_FOLD])["fold"]) == {REPORT_FOLD}


def test_folds_raises_when_a_requested_fold_is_absent():
    with pytest.raises(ValueError, match="fold"):
        folds(_frame(fold=0), [3])
