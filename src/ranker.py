"""LightGBM for the coarse rank stage, on ESCI's gain mapping rather than its own.

PROJECT_SPEC.md §4.2 asks for `objective: lambdarank` - a real LTR loss, not a
binary classification - and §6's Ablation 5 asks for the pointwise comparison
that shows whether the listwise loss earned its keep.

**LightGBM will report an NDCG, and it is not this project's NDCG.** Two
independent reasons compound:

  1. `lambdarank` defaults to a 2**rel - 1 gain. ESCI's mapping is
     1.0/0.1/0.01/0.0 - not exponential and not proportional to one.
  2. `eval_at` is a cutoff; this project reports full-list NDCG, which is why
     its random floor sits near 0.75 rather than near 0.

Measured on a matrix of the real shape (385,829 rows, 20,888 groups), the same
booster reports ndcg@10 0.7931 under ESCI's gains and 0.8575 under LightGBM's
default - a 6.4-point gap, larger than every effect in the ablation table. So
LABEL_GAIN is derived from src.labels rather than written out, `train_ranker`
refuses to let a caller override it, and TrainedRanker carries no score of any
kind. The only NDCG in this project comes from src.metrics.

LightGBM's internal metric still earns its keep as an early-stopping proxy,
and early stopping happens on fold 1, which no reported number comes from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.feature_matrix import (
    CATEGORICAL_FEATURES,
    group_sizes,
    sort_for_ranking,
)
from src.labels import ESCI_GAINS

# Integer label codes, ordered by gain. LightGBM indexes label_gain by the
# label and requires the list to be non-decreasing.
LABEL_ORDER: tuple[str, ...] = ("I", "C", "S", "E")
LABEL_GAIN: list[float] = [ESCI_GAINS[label] for label in LABEL_ORDER]

OBJECTIVES: tuple[str, ...] = (
    "lambdarank",
    "pointwise_regression",
    "pointwise_class",
)

# Three roles, three folds. Selecting, early-stopping and reporting on one
# fold is how a plan ends up reporting the number it optimised.
TRAIN_FOLDS: tuple[int, ...] = (2, 3, 4)
EARLY_STOP_FOLD: int = 1
REPORT_FOLD: int = 0

_SHARED = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "num_threads": 8,
    "deterministic": True,
}

LAMBDARANK_PARAMS: dict[str, Any] = _SHARED | {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "label_gain": LABEL_GAIN,
    # The largest train query has 188 judged candidates and the reported
    # metric has no cutoff; LightGBM's default of 30 would ignore two thirds
    # of the longest lists.
    "lambdarank_truncation_level": 200,
}

POINTWISE_REGRESSION_PARAMS: dict[str, Any] = _SHARED | {
    "objective": "regression",
    "metric": "l2",
}

POINTWISE_CLASS_PARAMS: dict[str, Any] = _SHARED | {
    "objective": "multiclass",
    "num_class": len(LABEL_ORDER),
    "metric": "multi_logloss",
}

_PARAMS: dict[str, dict[str, Any]] = {
    "lambdarank": LAMBDARANK_PARAMS,
    "pointwise_regression": POINTWISE_REGRESSION_PARAMS,
    "pointwise_class": POINTWISE_CLASS_PARAMS,
}


@dataclass(frozen=True)
class TrainedRanker:
    """A trained booster and the columns it was trained on. Carries no score."""

    booster: Any
    objective: str
    features: tuple[str, ...]
    best_iteration: int


def folds(frame: pd.DataFrame, which: Sequence[int]) -> pd.DataFrame:
    """The rows of `frame` in the requested folds."""
    wanted = list(which)
    available = set(frame["fold"])
    missing = set(wanted) - available
    if missing:
        raise ValueError(
            f"fold(s) {sorted(missing)} are not in this frame, which has "
            f"{sorted(available)}; test rows carry fold = -1 and have none"
        )
    return frame.loc[frame["fold"].isin(wanted)].reset_index(drop=True)


def _dataset(
    frame: pd.DataFrame, features: Sequence[str], objective: str, reference=None
):
    import lightgbm as lgb

    ordered = sort_for_ranking(frame)
    matrix = ordered[list(features)]
    label = (
        ordered["label_code"].to_numpy()
        if objective != "pointwise_regression"
        else ordered["gain"].to_numpy()
    )
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    dataset = lgb.Dataset(
        matrix,
        label=label,
        categorical_feature=categorical or "auto",
        free_raw_data=False,
        reference=reference,
    )
    if objective == "lambdarank":
        dataset.set_group(group_sizes(ordered["query_id"]))
    return dataset


def train_ranker(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    *,
    objective: str = "lambdarank",
    params: Mapping[str, Any] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    seed: int = 0,
) -> TrainedRanker:
    """Train one arm. `valid` is for early stopping and nothing else."""
    import lightgbm as lgb

    if objective not in _PARAMS:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of {OBJECTIVES}"
        )

    overrides = dict(params or {})
    if "label_gain" in overrides:
        raise ValueError(
            "label_gain is derived from src.labels.ESCI_GAINS and must not be "
            "overridden. LightGBM's default 2**rel - 1 reports an NDCG 6.4 "
            "points away from this project's on the same booster."
        )

    settings = _PARAMS[objective] | overrides | {"seed": seed, "data_random_seed": seed}
    train_set = _dataset(train, features, objective)
    valid_set = _dataset(valid, features, objective, reference=train_set)

    booster = lgb.train(
        settings,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["early_stop"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return TrainedRanker(
        booster=booster,
        objective=objective,
        features=tuple(features),
        best_iteration=int(booster.best_iteration or num_boost_round),
    )


def predict(ranker: TrainedRanker, frame: pd.DataFrame) -> np.ndarray:
    """One score per row of `frame`, higher is better.

    The multiclass arm emits four probabilities; they are collapsed by
    *expected gain* rather than by argmax. Argmax would throw away the graded
    structure the whole metric is about - an E and an S both become "the top
    class" and the ordering between two S products is lost.
    """
    raw = ranker.booster.predict(
        frame[list(ranker.features)], num_iteration=ranker.best_iteration
    )
    raw = np.asarray(raw)
    if ranker.objective == "pointwise_class":
        return (raw @ np.asarray(LABEL_GAIN)).astype(np.float64)
    return raw.astype(np.float64)


def predict_run(
    ranker: TrainedRanker, frame: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """A nested run dict for src.metrics.ndcg_per_query.

    Ids are stringified here because query_id is int64 in the parquet and both
    src.metrics and pytrec_eval key on strings.
    """
    scores = predict(ranker, frame)
    run: dict[str, dict[str, float]] = {}
    for query_id, product_id, score in zip(
        frame["query_id"], frame["product_id"], scores
    ):
        run.setdefault(str(query_id), {})[str(product_id)] = float(score)
    return run
