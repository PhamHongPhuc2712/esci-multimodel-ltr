"""Stage 4: one ordering from three stages' opinions.

PROJECT_SPEC.md §4.4 - "weighted combination, or a small learned combiner over
stage scores" - and it asks for the comparison against each single stage alone.
This module is the strategies; src/blend_report.py measures them.

Three arms, and a ceiling:

  * **fixed**    reciprocal-rank fusion at hand-set weights. RRF rather than a
                 score average because a LambdaMART score, a cross-encoder
                 logit and a rank are three incomparable scales, and averaging
                 them is exactly the mismatch src/rerank_window.py exists to
                 prevent.
  * **combiner** LightGBM lambdarank over the five signal columns.
  * **selector** a per-query choice of which single arm to trust.
  * **oracle**   the best arm per query, chosen with the labels. Not a method -
                 it bounds one.

**Measured before this module was written, on fold 0**: the oracle reaches
0.9076 against the best single arm's 0.8814, so there is +0.0262 to aim at; but
every fixed weighting loses (best 0.8800 at 1:1:4) and a cross-fitted combiner
loses too (0.8793). The selector exists because the oracle's advantage is
*choosing*, not mixing: the LLM is strictly best on 41.6% of queries, the
cross-encoder on 21.3%, Stage 2 on 17.7%, and 19.4% are tied at the top.

Nothing here reads a file or calls an API, so every path is testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.stage_signals import BLEND_FEATURES, RRF_K

STRATEGIES: tuple[str, ...] = ("fixed", "combiner", "selector")

ARM_COLUMNS: dict[str, tuple[str, bool]] = {
    # arm -> (column, ascending). A rank sorts ascending, a score descending.
    "stage2": ("stage2_rank", True),
    "ce": ("ce_score", False),
    "llm": ("llm_rank", True),
}

# The best of the sweep measured on fold 0 - which still lost to the LLM
# alone. Recorded so the arm reproduces what was measured, not a guess.
DEFAULT_WEIGHTS: dict[str, float] = {"stage2": 1.0, "ce": 1.0, "llm": 4.0}

_SELECTOR_ARMS: tuple[str, ...] = ("stage2", "ce", "llm")


def orderings_from_scores(
    frame: pd.DataFrame, column: str, *, ascending: bool = False
) -> dict[str, list[str]]:
    """One ordering per query, ties broken on product_id."""
    sign = 1.0 if ascending else -1.0
    out: dict[str, list[str]] = {}
    for query_id, group in frame.groupby("query_id", sort=True):
        pairs = sorted(
            zip(group[column].to_numpy(), group["product_id"].astype(str)),
            key=lambda sd: (sign * float(sd[0]), sd[1]),
        )
        out[str(query_id)] = [doc for _, doc in pairs]
    return out


def fixed_weight_ordering(
    frame: pd.DataFrame,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    *,
    k: int = RRF_K,
) -> dict[str, list[str]]:
    """Reciprocal-rank fusion of the three arms at fixed weights."""
    per_arm = {
        arm: orderings_from_scores(frame, column, ascending=ascending)
        for arm, (column, ascending) in ARM_COLUMNS.items()
    }
    out: dict[str, list[str]] = {}
    for query_id in per_arm["stage2"]:
        fused: dict[str, float] = {}
        for arm, orderings in per_arm.items():
            weight = float(weights.get(arm, 0.0))
            for rank, doc in enumerate(orderings[query_id], start=1):
                fused[doc] = fused.get(doc, 0.0) + weight / (k + rank)
        out[query_id] = sorted(fused, key=lambda d: (-fused[d], d))
    return out


def _group_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Run lengths over consecutive query_id, as LightGBM's `group` wants."""
    ids = frame["query_id"].to_numpy()
    if len(ids) == 0:
        return np.array([], dtype=np.int64)
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    sizes = np.diff(np.concatenate([[0], boundaries, [len(ids)]]))
    if len(sizes) != len(pd.unique(ids)):
        raise ValueError(
            "query_id is not contiguous; sort by query_id before training or "
            "LightGBM will learn comparisons that straddle queries"
        )
    return sizes.astype(np.int64)


def train_combiner(
    frame: pd.DataFrame,
    *,
    features: Sequence[str] = BLEND_FEATURES,
    params: Mapping[str, Any] | None = None,
    num_boost_round: int = 150,
    seed: int = 0,
) -> Any:
    """LightGBM lambdarank over the stage signals.

    `label_gain` comes from src.ranker, which derives it from src.labels. The
    same booster reports an NDCG 6.4 points apart under LightGBM's default
    2**rel - 1 gain, so it is never a caller's to set.
    """
    import lightgbm as lgb

    from src.ranker import LABEL_GAIN

    overrides = dict(params or {})
    if "label_gain" in overrides:
        raise ValueError(
            "label_gain is derived from src.labels.ESCI_GAINS and must not be "
            "overridden; LightGBM's default 2**rel - 1 measures a different "
            "metric from the one this project reports"
        )

    ordered = frame.sort_values("query_id", kind="stable")
    settings = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "label_gain": LABEL_GAIN,
        "lambdarank_truncation_level": 20,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "deterministic": True,
        "seed": seed,
        "data_random_seed": seed,
    } | overrides
    dataset = lgb.Dataset(
        ordered[list(features)],
        label=ordered["label_code"].to_numpy(),
        group=_group_sizes(ordered),
        free_raw_data=False,
    )
    return lgb.train(settings, dataset, num_boost_round=num_boost_round)


def predict_combiner(
    booster, frame: pd.DataFrame, *, features: Sequence[str] = BLEND_FEATURES
) -> np.ndarray:
    """One blend score per row, higher is better."""
    return np.asarray(booster.predict(frame[list(features)]), dtype=np.float64)


def _query_parts(frame: pd.DataFrame, n_folds: int, seed: int) -> list[list[str]]:
    """A seeded partition of the frame's query ids into `n_folds` parts.

    One function for both learned arms, so the combiner and the selector are
    cross-fitted over the same split and their fold-0 numbers compare.
    """
    queries = np.array(sorted(pd.unique(frame["query_id"].astype(str))))
    if len(queries) < n_folds:
        raise ValueError(
            f"{len(queries)} queries cannot be split into {n_folds} folds"
        )
    shuffled = queries[np.random.default_rng(seed).permutation(len(queries))]
    return [list(part) for part in np.array_split(shuffled, n_folds)]


def cross_fit_predict(
    frame: pd.DataFrame,
    *,
    n_folds: int = 2,
    seed: int = 0,
    features: Sequence[str] = BLEND_FEATURES,
    num_boost_round: int = 150,
    _return_parts: bool = False,
):
    """Out-of-fold blend scores: no query is scored by a model that saw it.

    Fold 0 is the only query set with both an LLM ordering and a label, so it
    is the only training set the combiner can have - and it is also the surface
    Plans 5 and 6 reported on. Five features over 4,130 queries memorise
    easily, so a fold-0 number from a model fitted on all of fold 0 would be a
    training fit wearing a result's clothes.

    Partitioning is by `query_id`, never by row: four documents of one query on
    both sides of the split is the same leak in a different shape.
    """
    parts = _query_parts(frame, n_folds, seed)
    if _return_parts:
        return parts

    ids = frame["query_id"].astype(str).to_numpy()
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    for part in parts:
        held = np.isin(ids, list(part))
        booster = train_combiner(
            frame.loc[~held],
            features=features,
            num_boost_round=num_boost_round,
            seed=seed,
        )
        scores[held] = predict_combiner(
            booster, frame.loc[held], features=features
        )
    if np.isnan(scores).any():
        raise ValueError("some rows were never scored; the partition is not a cover")
    return scores


def _selector_features(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per query: how much the arms disagree, and how confident each is.

    Deliberately query-level and label-free. A selector fed the labels would be
    the oracle.
    """
    rows = []
    for query_id, group in frame.groupby("query_id", sort=True):
        stage2 = group["stage2_rank"].to_numpy(dtype=float)
        ce = group["ce_score"].to_numpy(dtype=float)
        llm = group["llm_rank"].to_numpy(dtype=float)
        n = len(group)
        rows.append(
            {
                "query_id": str(query_id),
                "n_candidates": n,
                "ce_spread": float(ce.max() - ce.min()) if n > 1 else 0.0,
                "ce_top_margin": (
                    float(np.sort(ce)[-1] - np.sort(ce)[-2]) if n > 1 else 0.0
                ),
                "stage2_score_spread": float(
                    group["stage2_score"].max() - group["stage2_score"].min()
                ) if n > 1 else 0.0,
                # Rank agreement: 1.0 when two arms order identically.
                "agree_stage2_llm": float(
                    np.corrcoef(stage2, llm)[0, 1]
                ) if n > 1 and llm.std() > 0 and stage2.std() > 0 else 0.0,
                "agree_ce_llm": float(
                    np.corrcoef(-ce, llm)[0, 1]
                ) if n > 1 and llm.std() > 0 and ce.std() > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


SELECTOR_FEATURES: tuple[str, ...] = (
    "n_candidates",
    "ce_spread",
    "ce_top_margin",
    "stage2_score_spread",
    "agree_stage2_llm",
    "agree_ce_llm",
)


def selector_targets(
    per_arm: Mapping[str, Mapping[str, float]], query_ids: Sequence[str]
) -> np.ndarray:
    """Which arm the selector should learn to pick for each query.

    The arm with the highest NDCG - and on a tie, the tied arm with the best
    mean over these queries, never the first one listed. Measured on fold 0,
    19.4% of queries are tied at the top; np.argmax hands every such tie to
    stage2, the weakest arm, labelling 31.7% of queries "stage2" when it is
    strictly best on 17.7%. A tied query scores the same whichever tied arm
    routes it, so it should teach the default, not the list order.
    """
    missing = [
        q for q in query_ids
        if any(str(q) not in per_arm[arm] for arm in _SELECTOR_ARMS)
    ]
    if missing:
        raise KeyError(
            f"{len(missing):,} queries have no NDCG for some arm (first: "
            f"{missing[:3]}); scoring them 0.0 would teach the selector a "
            "loss that never happened"
        )
    scores = np.array(
        [[float(per_arm[arm][str(q)]) for arm in _SELECTOR_ARMS] for q in query_ids]
    )
    if scores.size == 0:
        return np.zeros(0, dtype=np.int64)
    preference = np.argsort(-scores.mean(axis=0), kind="stable")
    winners = np.isclose(scores, scores.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12)
    return np.array(
        [next(int(i) for i in preference if row[i]) for row in winners],
        dtype=np.int64,
    )


def train_selector(
    frame: pd.DataFrame,
    per_arm: Mapping[str, Mapping[str, float]],
    *,
    seed: int = 0,
    num_boost_round: int = 120,
) -> Any:
    """A multiclass model over which arm wins each query.

    The oracle's advantage is choosing, not mixing: on fold 0 the LLM is
    strictly best on 41.6% of queries, the cross-encoder on 21.3%, Stage 2 on
    17.7%, and 19.4% are tied at the top. This is the only Stage 4 arm whose
    shape matches that.
    """
    import lightgbm as lgb

    features = _selector_features(frame)
    target = selector_targets(per_arm, list(features["query_id"]))
    dataset = lgb.Dataset(
        features[list(SELECTOR_FEATURES)],
        label=target,
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "multiclass",
            "num_class": len(_SELECTOR_ARMS),
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 30,
            "verbose": -1,
            "deterministic": True,
            "seed": seed,
            "data_random_seed": seed,
        },
        dataset,
        num_boost_round=num_boost_round,
    )


def selector_ordering(
    selector,
    frame: pd.DataFrame,
    arm_orderings: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    """Each query takes the ordering of the arm the selector picked."""
    features = _selector_features(frame)
    probabilities = np.asarray(selector.predict(features[list(SELECTOR_FEATURES)]))
    out: dict[str, list[str]] = {}
    for query_id, row in zip(features["query_id"], probabilities):
        arm = _SELECTOR_ARMS[int(np.argmax(row))]
        out[str(query_id)] = list(arm_orderings[arm][str(query_id)])
    return out


def cross_fit_select(
    frame: pd.DataFrame,
    per_arm: Mapping[str, Mapping[str, float]],
    arm_orderings: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    n_folds: int = 2,
    seed: int = 0,
    _return_parts: bool = False,
):
    """Out-of-fold selector routes: no query is routed by a selector that saw it.

    The selector's target is which arm scored best on each query - the label
    in another form - so it leaks on fold 0 exactly as the combiner would.
    Same partition as cross_fit_predict, via _query_parts.
    """
    parts = _query_parts(frame, n_folds, seed)
    if _return_parts:
        return parts
    ids = frame["query_id"].astype(str)
    out: dict[str, list[str]] = {}
    for part in parts:
        held = ids.isin(set(part)).to_numpy()
        selector = train_selector(frame.loc[~held], per_arm, seed=seed)
        out.update(selector_ordering(selector, frame.loc[held], arm_orderings))
    return out


def oracle_ordering(
    per_arm_per_query: Mapping[str, Mapping[str, float]],
    arm_orderings: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    """The best arm per query, chosen with the labels.

    A ceiling, never a method: it needs the NDCG it is trying to produce. It
    is reported beside the three arms so a tie can be read as "the methods
    failed to reach available headroom" rather than "there was none".
    """
    arms = list(per_arm_per_query)
    out: dict[str, list[str]] = {}
    for query_id in per_arm_per_query[arms[0]]:
        best = max(arms, key=lambda a: float(per_arm_per_query[a][query_id]))
        out[str(query_id)] = list(arm_orderings[best][str(query_id)])
    return out
