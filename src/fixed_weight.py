"""Ablation 7's control: one hand-tuned global text/image weight.

PROJECT_SPEC.md §5 records the published best zero-shot text+image combination
as "a single hand-tuned global weight", and §6's Ablation 7 asks whether a
learned combination beats it. Al Ghossein et al. listed learned combination as
out of scope; this module builds the thing they did, so the comparison is
against their method rather than against a strawman.

Measured on fold 0, sweeping the weight on fold 0 itself: 0.8094 at pure
image, 0.8285 at pure text, peaking at **0.8347** with w_text = 0.7. That is an
*oracle* value - the best any fixed weight could do on the surface being
reported. `sweep` is therefore run on fold 1 and the chosen weight applied to
fold 0, which lands at or just below the oracle.

Three decisions make the control stronger, not weaker. A quietly handicapped
control turns the ablation into a formality:

  * **Standardise before blending.** Both columns are cosines but not on
    comparable scales, so one weight over raw values weights two different
    units. Mean and sd are fitted on the training folds, never on the fold
    being reported.
  * **Fall back to text where the image is absent**, rather than substituting
    the mean. 20.6% of judged pairs have no image vector, and pulling them
    toward the average would penalise a product for a 2022 scrape failure -
    the thing CLAUDE.md forbids the learned arm from doing.
  * **Sweep finely.** 21 weights at 0.05, not 3 at 0.25.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import Qrels, ndcg_per_query

DEFAULT_WEIGHTS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))


@dataclass(frozen=True)
class Normaliser:
    mean: float
    std: float


def fit_normaliser(values: np.ndarray) -> Normaliser:
    """Mean and standard deviation over the finite values only.

    NaN is an absence, not a zero: counting 20.6% of image similarities as
    zeros would drag the mean and mis-scale every real value.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot fit a normaliser: the column has no finite values")
    std = float(finite.std())
    return Normaliser(mean=float(finite.mean()), std=std if std > 1e-12 else 1.0)


def apply_normaliser(normaliser: Normaliser, values: np.ndarray) -> np.ndarray:
    """Centre and scale, leaving NaN as NaN."""
    return (np.asarray(values, dtype=np.float64) - normaliser.mean) / normaliser.std


def blend(text: np.ndarray, image: np.ndarray, weight: float) -> np.ndarray:
    """`w * text + (1 - w) * image`, falling back to text where image is NaN."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight}")
    text = np.asarray(text, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    mixed = weight * text + (1.0 - weight) * image
    return np.where(np.isfinite(image), mixed, text)


def _run(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for query_id, product_id, score in zip(
        frame["query_id"], frame["product_id"], scores
    ):
        # A pair with no text score at all still has to occupy a rank; -inf
        # sorts it last without inventing a similarity for it.
        run.setdefault(str(query_id), {})[str(product_id)] = (
            float(score) if np.isfinite(score) else float("-inf")
        )
    return run


def sweep(
    frame: pd.DataFrame,
    qrels: Qrels,
    *,
    text_column: str,
    image_column: str,
    normalisers: Mapping[str, Normaliser],
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> list[tuple[float, float]]:
    """Mean NDCG at each weight, using normalisers fitted elsewhere."""
    text = apply_normaliser(normalisers[text_column], frame[text_column].to_numpy())
    image = apply_normaliser(normalisers[image_column], frame[image_column].to_numpy())

    out: list[tuple[float, float]] = []
    for weight in weights:
        per_query = ndcg_per_query(_run(frame, blend(text, image, weight)), qrels)
        out.append((float(weight), sum(per_query.values()) / len(per_query)))
    return out


def best_weight(result: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The best (weight, ndcg). Ties break toward the lower weight, for determinism."""
    if not result:
        raise ValueError("cannot pick a weight from an empty sweep")
    return max(result, key=lambda item: (item[1], -item[0]))


def per_query_scores(
    frame: pd.DataFrame,
    qrels: Qrels,
    *,
    text_column: str,
    image_column: str,
    normalisers: Mapping[str, Normaliser],
    weight: float,
) -> dict[str, float]:
    """Per-query NDCG at one weight, for a paired comparison against a model."""
    text = apply_normaliser(normalisers[text_column], frame[text_column].to_numpy())
    image = apply_normaliser(normalisers[image_column], frame[image_column].to_numpy())
    return ndcg_per_query(_run(frame, blend(text, image, weight)), qrels)
