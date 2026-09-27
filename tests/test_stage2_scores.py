import numpy as np
import pandas as pd
import pytest

from src.ranker import EARLY_STOP_FOLD, REPORT_FOLD, TRAIN_FOLDS
from src.stage2_scores import (
    DEFAULT_DIR,
    OOF_SPLIT,
    STAGE2_COLUMNS,
    load_stage2,
    out_of_fold_scores,
    require_out_of_sample,
    score_split,
    stage2_ndcg,
    window_shift,
)


def _frame(n_queries=40, folds=(0, 1, 2, 3, 4), seed=0):
    """A small feature matrix shaped like data/features/train.parquet."""
    rng = np.random.default_rng(seed)
    rows = []
    q = 0
    for fold in folds:
        for _ in range(n_queries):
            for p in range(6):
                code = int(rng.integers(0, 4))
                rows.append(
                    {
                        "query_id": q,
                        "product_id": f"p{q}_{p}",
                        "label_code": code,
                        "gain": [0.0, 0.01, 0.1, 1.0][code],
                        "qrel": [0, 1, 10, 100][code],
                        "fold": fold,
                        "signal": code + rng.normal(scale=0.3),
                        "noise": rng.normal(),
                    }
                )
            q += 1
    return pd.DataFrame(rows)


FEATURES = ["signal", "noise"]


def test_score_split_returns_the_declared_columns():
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES)
    assert list(out.columns) == list(STAGE2_COLUMNS)


def test_every_target_row_gets_exactly_one_score():
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES)
    assert len(out) == len(frame)
    assert not out.duplicated(subset=["query_id", "product_id"]).any()


def test_training_folds_are_flagged_in_sample():
    # The model trained on 2/3/4, so their scores are optimistic. A flagged row
    # is recoverable; a missing one is indistinguishable from a bug.
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES).merge(
        frame[["query_id", "product_id", "fold"]], on=["query_id", "product_id"]
    )
    assert out.loc[out["fold"].isin(TRAIN_FOLDS), "in_sample"].all()
    assert not out.loc[out["fold"] == REPORT_FOLD, "in_sample"].any()
    assert not out.loc[out["fold"] == EARLY_STOP_FOLD, "in_sample"].any()


def test_a_target_with_no_folds_column_is_all_out_of_sample():
    # The test split carries fold = -1 and was never trained on.
    train = _frame()
    target = _frame(n_queries=10, folds=(-1,), seed=7)
    out = score_split(train, target, features=FEATURES)
    assert not out["in_sample"].any()


def test_scores_order_the_informative_feature():
    frame = _frame(n_queries=200)
    out = score_split(frame, frame, features=FEATURES).merge(
        frame[["query_id", "product_id", "label_code"]], on=["query_id", "product_id"]
    )
    # A model that learned `signal` must score Exact above Irrelevant on average.
    assert (
        out.loc[out["label_code"] == 3, "stage2_score"].mean()
        > out.loc[out["label_code"] == 0, "stage2_score"].mean()
    )


def test_scoring_is_deterministic_for_a_seed():
    frame = _frame()
    a = score_split(frame, frame, features=FEATURES, seed=3)["stage2_score"].to_numpy()
    b = score_split(frame, frame, features=FEATURES, seed=3)["stage2_score"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_require_out_of_sample_drops_the_training_folds():
    frame = pd.DataFrame(
        {
            "query_id": [1, 2],
            "product_id": ["a", "b"],
            "stage2_score": [1.0, 2.0],
            "in_sample": [True, False],
        }
    )
    kept = require_out_of_sample(frame)
    assert kept["query_id"].tolist() == [2]


def test_require_out_of_sample_raises_when_nothing_survives():
    frame = pd.DataFrame(
        {
            "query_id": [1],
            "product_id": ["a"],
            "stage2_score": [1.0],
            "in_sample": [True],
        }
    )
    with pytest.raises(ValueError, match="in-sample"):
        require_out_of_sample(frame)


def test_load_stage2_rejects_an_unknown_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="stage2_scores"):
        load_stage2("train", directory=tmp_path)


def test_the_default_directory_is_the_feature_directory():
    assert DEFAULT_DIR.as_posix().endswith("data/features")


# --- the fit set, which differs by split ------------------------------------
# Plan 5 trained fold 0's model on folds 2/3/4 but its *test* model on every
# train fold (src/rank_report.py: `fit = train` when --split test). Reproducing
# both of its headlines means mirroring both fits, and `in_sample` must follow
# whichever one ran rather than a knob a caller can set to disagree with it.


def test_fitting_every_fold_marks_a_train_target_entirely_in_sample():
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES, fit_folds=None)
    assert out["in_sample"].all()


def test_in_sample_follows_the_fit_set():
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES, fit_folds=(3, 4)).merge(
        frame[["query_id", "product_id", "fold"]], on=["query_id", "product_id"]
    )
    assert out.loc[out["fold"].isin([3, 4]), "in_sample"].all()
    assert not out.loc[out["fold"].isin([0, 1, 2]), "in_sample"].any()


def test_a_disjoint_split_is_out_of_sample_even_when_every_fold_is_fit():
    # The test split is disjoint from every train fold, so training on all of
    # them leaves it out-of-sample. This is what Plan 5's test headline did.
    train = _frame()
    target = _frame(n_queries=10, folds=(-1,), seed=7)
    out = score_split(train, target, features=FEATURES, fit_folds=None)
    assert not out["in_sample"].any()


# --- out-of-fold scores for the training folds (Plan 8) ---------------------
# A student trained on Stage 2's window must see the window it meets at test
# time. The persisted ordering of folds 2/3/4 is in-sample - 0.9035 against
# fold 0's 0.8519 - so the training windows come from these instead.


def test_out_of_fold_scores_cover_exactly_the_training_folds():
    frame = _frame()
    out = out_of_fold_scores(frame, features=FEATURES)
    expected = frame.loc[frame["fold"].isin(TRAIN_FOLDS)]
    assert list(out.columns) == list(STAGE2_COLUMNS)
    assert len(out) == len(expected)
    assert set(zip(out["query_id"], out["product_id"])) == set(
        zip(expected["query_id"], expected["product_id"])
    )


def test_no_fold_is_scored_by_a_model_that_trained_on_it(monkeypatch):
    # Review Focus 1. `in_sample` is derived from the fit, so check the fits
    # themselves: record the folds each model trained on and the folds it
    # then scored.
    import src.stage2_scores as module

    real_train, real_predict = module.train_ranker, module.predict
    fitted: dict[int, set] = {}
    calls: list[tuple[set, set]] = []

    def spy_train(fit, early_stop, columns, **kwargs):
        ranker = real_train(fit, early_stop, columns, **kwargs)
        fitted[id(ranker)] = set(fit["fold"])
        return ranker

    def spy_predict(ranker, target):
        calls.append((fitted[id(ranker)], set(target["fold"])))
        return real_predict(ranker, target)

    monkeypatch.setattr(module, "train_ranker", spy_train)
    monkeypatch.setattr(module, "predict", spy_predict)
    out_of_fold_scores(_frame(), features=FEATURES)

    assert len(calls) == len(TRAIN_FOLDS)
    for fit_folds, scored_folds in calls:
        assert not fit_folds & scored_folds
        assert REPORT_FOLD not in fit_folds
        assert EARLY_STOP_FOLD not in fit_folds
    assert set().union(*(scored for _, scored in calls)) == set(TRAIN_FOLDS)


def test_every_out_of_fold_row_is_out_of_sample():
    out = out_of_fold_scores(_frame(), features=FEATURES)
    assert not out["in_sample"].any()
    assert len(require_out_of_sample(out)) == len(out)


def test_out_of_fold_scores_are_not_the_in_sample_scores():
    frame = _frame()
    oof = out_of_fold_scores(frame, features=FEATURES)
    in_sample = score_split(
        frame, frame.loc[frame["fold"].isin(TRAIN_FOLDS)], features=FEATURES
    )
    merged = oof.merge(in_sample, on=["query_id", "product_id"], suffixes=("_oof", "_in"))
    assert len(merged) == len(oof)
    assert not np.allclose(merged["stage2_score_oof"], merged["stage2_score_in"])


def test_the_reporting_fold_is_refused():
    with pytest.raises(ValueError, match="reporting surface"):
        out_of_fold_scores(_frame(), fit_folds=(REPORT_FOLD, 2, 3), features=FEATURES)


def test_the_early_stop_fold_is_refused():
    with pytest.raises(ValueError, match=rf"\[{EARLY_STOP_FOLD}\]"):
        out_of_fold_scores(_frame(), fit_folds=(EARLY_STOP_FOLD, 2), features=FEATURES)


def test_one_fold_cannot_be_scored_out_of_fold():
    with pytest.raises(ValueError, match="at least two"):
        out_of_fold_scores(_frame(), fit_folds=(2,), features=FEATURES)


def test_a_missing_out_of_fold_file_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--out-of-fold"):
        load_stage2(OOF_SPLIT, directory=tmp_path)


def test_stage2_ndcg_is_one_for_an_ideal_ordering():
    joined = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": ["a", "b", "c"],
            "qrel": [100, 10, 0],
            "stage2_score": [3.0, 2.0, 1.0],
        }
    )
    assert stage2_ndcg(joined) == pytest.approx(1.0)


def _scores(frame, sign=1.0):
    return pd.DataFrame(
        {
            "query_id": frame["query_id"],
            "product_id": frame["product_id"],
            "stage2_score": sign * frame["signal"],
        }
    )


def test_identical_orderings_show_no_shift():
    frame = _frame(n_queries=5, folds=(2,))
    shift = window_shift(frame, _scores(frame), _scores(frame), k=3)
    assert shift["same_window_share"] == 1.0
    assert shift["ndcg_in_sample"] == shift["ndcg_out_of_fold"]
    assert shift["exact_on_top_in_sample"] == shift["exact_on_top_out_of_fold"]
    assert shift["n_queries"] == 5


def test_opposite_orderings_share_no_window():
    # Six products a query and windows of three: reversing the order leaves
    # the two windows disjoint.
    frame = _frame(n_queries=5, folds=(2,))
    shift = window_shift(frame, _scores(frame), _scores(frame, sign=-1.0), k=3)
    assert shift["same_window_share"] == 0.0
    assert shift["ndcg_out_of_fold"] > shift["ndcg_in_sample"]


def test_exact_on_top_reads_the_first_window_document():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "product_id": ["a", "b", "c", "d"],
            "qrel": [100, 0, 0, 100],
            "signal": [2.0, 1.0, 2.0, 1.0],
        }
    )
    shift = window_shift(frame, _scores(frame), _scores(frame, sign=-1.0), k=1)
    assert shift["exact_on_top_out_of_fold"] == 0.5     # query 1 only
    assert shift["exact_on_top_in_sample"] == 0.5       # query 2 only


def test_a_score_frame_that_misses_rows_is_refused():
    frame = _frame(n_queries=5, folds=(2,))
    with pytest.raises(ValueError, match="cover"):
        window_shift(frame, _scores(frame).iloc[1:], _scores(frame), k=3)


@pytest.mark.data
def test_the_persisted_ordering_reproduces_plan_5():
    from src.metrics import ndcg_per_query

    scores = load_stage2("train")
    matrix = pd.read_parquet("data/features/train.parquet")
    assert len(scores) == 419_653
    assert int(scores["in_sample"].sum()) == 250_485

    joined = matrix.merge(scores, on=["query_id", "product_id"])
    fold0 = joined.loc[joined["fold"] == REPORT_FOLD]
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q, p, r, s in zip(
        fold0["query_id"], fold0["product_id"], fold0["qrel"], fold0["stage2_score"]
    ):
        qrels.setdefault(str(q), {})[str(p)] = int(r)
        run.setdefault(str(q), {})[str(p)] = float(s)
    per_query = ndcg_per_query(run, qrels)
    assert sum(per_query.values()) / len(per_query) == pytest.approx(0.8519, abs=0.0005)


@pytest.mark.data
def test_the_test_ordering_is_entirely_out_of_sample():
    scores = load_stage2("test")
    assert len(scores) == 181_701
    assert not scores["in_sample"].any()
    require_out_of_sample(scores)  # must not raise


@pytest.mark.data
def test_the_persisted_test_ordering_reproduces_plan_5():
    # Plan 5's 0.8579 came from a model fitted on every train fold. Fitting
    # only 2/3/4 lands at 0.8559, so this pins the fit set as much as the
    # score.
    from src.metrics import ndcg_per_query

    scores = load_stage2("test")
    matrix = pd.read_parquet("data/features/test.parquet")
    joined = matrix.merge(scores, on=["query_id", "product_id"])
    assert len(joined) == len(matrix)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q, p, r, s in zip(
        joined["query_id"], joined["product_id"], joined["qrel"], joined["stage2_score"]
    ):
        qrels.setdefault(str(q), {})[str(p)] = int(r)
        run.setdefault(str(q), {})[str(p)] = float(s)
    per_query = ndcg_per_query(run, qrels)
    assert sum(per_query.values()) / len(per_query) == pytest.approx(0.8579, abs=0.0005)


@pytest.mark.data
def test_the_out_of_fold_ordering_looks_like_fold_0_not_its_own_training_folds():
    # Measured 2026-09-27: 0.8534 out-of-fold against fold 0's 0.8519, where
    # the in-sample ordering of the same three folds scores 0.9035.
    scores = load_stage2(OOF_SPLIT)
    assert len(scores) == 250_485
    assert not scores["in_sample"].any()
    matrix = pd.read_parquet(
        "data/features/train.parquet",
        columns=["query_id", "product_id", "qrel", "fold"],
    )
    joined = matrix.merge(scores, on=["query_id", "product_id"])
    assert len(joined) == 250_485
    assert set(joined["fold"]) == set(TRAIN_FOLDS)
    assert joined["query_id"].nunique() == 12_519
    assert stage2_ndcg(joined) == pytest.approx(0.8534, abs=0.0005)


def test_the_committed_window_shift_says_the_in_sample_windows_are_easier():
    # docs/results/stage2-oof.json, written by --out-of-fold. The out-of-fold
    # windows look like fold 0's; the in-sample ones are far easier and mostly
    # different documents.
    import json
    from pathlib import Path

    shift = json.loads(Path("docs/results/stage2-oof.json").read_text(encoding="utf-8"))
    assert shift["n_queries"] == 12_519
    assert abs(shift["ndcg_out_of_fold"] - shift["fold0_ndcg"]) < 0.01
    assert shift["ndcg_in_sample"] > shift["ndcg_out_of_fold"] + 0.03
    assert shift["exact_on_top_in_sample"] > shift["exact_on_top_out_of_fold"]
    assert shift["same_window_share"] < 0.5
