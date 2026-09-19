"""Bootstrap confidence intervals over queries.

With ~9K test queries and a random floor near 0.748, the gap between a real
improvement and noise can be a couple of NDCG points, so every reported number
carries an interval.

Method comparisons use the paired form: one resample of query ids per
replicate, applied to both methods. Resampling independently would add back
the between-query variance that pairing cancels - the variance that dominates
here, since a hard query is hard for every method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


def _percentiles(replicates: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def bootstrap_ci(
    per_query: Mapping[str, float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for the mean over queries."""
    if not per_query:
        raise ValueError("cannot bootstrap an empty set of per-query scores")
    values = np.asarray([per_query[q] for q in sorted(per_query)], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    replicates = values[draws].mean(axis=1)
    low, high = _percentiles(replicates, alpha)
    return Interval(point=float(values.mean()), low=low, high=high)


def paired_delta_ci(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for mean(a) - mean(b), paired by query."""
    if set(a) != set(b):
        raise ValueError(
            "paired comparison needs the same queries on both sides; "
            f"{len(set(a) - set(b))} only in a, {len(set(b) - set(a))} only in b"
        )
    if not a:
        raise ValueError("cannot bootstrap an empty set of per-query scores")
    queries = sorted(a)
    deltas = np.asarray([a[q] - b[q] for q in queries], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(n_resamples, len(deltas)))
    replicates = deltas[draws].mean(axis=1)
    low, high = _percentiles(replicates, alpha)
    return Interval(point=float(deltas.mean()), low=low, high=high)
