"""The random-ordering NDCG floor, computed rather than quoted.

Roughly 44% of ESCI judgements are Exact, so shuffling the judged candidate
list already scores near 0.75 on full-list NDCG. A headline of 0.86 means
nothing without that floor attached.

The number is computed on every report because it moves with the gain mapping:
CLAUDE.md records 0.7467 with the standard discount, the published SQID figure
is 0.7483, and under the S/C swap it is 0.7141. That 3.3-point spread is larger
than most method gains, so a hard-coded constant would be a way to be wrong by
more than the effect being measured.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from src.metrics import Qrels, ndcg_per_query


@dataclass(frozen=True)
class FloorResult:
    mean: float
    low: float
    high: float
    per_trial: tuple[float, ...]
    n_trials: int
    seed: int


def random_run(qrels: Qrels, rng: random.Random) -> dict[str, dict[str, float]]:
    """Score every judged document with a distinct random value.

    Distinct scores keep the floor a measurement of random *ordering* rather
    than of the tie-breaking policy.
    """
    run: dict[str, dict[str, float]] = {}
    for qid, judged in qrels.items():
        order = list(judged)
        rng.shuffle(order)
        run[qid] = {doc_id: float(rank) for rank, doc_id in enumerate(order)}
    return run


def random_floor(
    qrels: Qrels,
    *,
    n_trials: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> FloorResult:
    """Mean NDCG of a random ordering, with a percentile interval over trials."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")

    rng = random.Random(seed)
    per_trial: list[float] = []
    for _ in range(n_trials):
        scores = ndcg_per_query(random_run(qrels, rng), qrels)
        per_trial.append(sum(scores.values()) / len(scores))

    ordered = sorted(per_trial)
    low_index = int((alpha / 2) * (len(ordered) - 1))
    high_index = int((1 - alpha / 2) * (len(ordered) - 1))
    return FloorResult(
        mean=statistics.fmean(per_trial),
        low=ordered[low_index],
        high=ordered[high_index],
        per_trial=tuple(per_trial),
        n_trials=n_trials,
        seed=seed,
    )
