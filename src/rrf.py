"""Reciprocal Rank Fusion over ranked lists.

score(d) = sum over the channels that returned d of  w_c / (k + rank_c(d))

The "channels that returned d" clause is the whole design. The image channel
has no vector for 22.5% of products, so those products are simply *absent*
from its list. Absence must contribute nothing: a worst-possible rank is still
a vote, and it would push products down for a 2022 scrape failure rather than
for anything about the product. `CLAUDE.md`'s evaluation discipline is explicit
that this kind of missingness is an artefact and must not be treated as signal.

k = 60 is the constant from Cormack et al.'s original RRF paper. It is a
tunable; src/recall_report.py tunes it on the frozen validation folds and never
on test.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_K = 60


def fuse(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Fuse one query's ranked lists into a single ranking, best first."""
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            f"{len(weights)} weights for {len(ranked_lists)} channels; "
            "pass one weight per channel or none at all"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for channel, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else float(weights[channel])
        seen: set[str] = set()
        for position, document in enumerate(ranked, start=1):
            # A channel that returns a document twice votes for it once.
            if document in seen:
                continue
            seen.add(document)
            scores[document] = scores.get(document, 0.0) + weight / (k + position)
            if document not in first_seen:
                first_seen[document] = order
                order += 1

    # Ties break on first appearance, so fusion is deterministic across runs
    # rather than dependent on dict iteration order.
    ranking = sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
    return ranking[:limit] if limit is not None else ranking


def fuse_batches(
    channel_results: Sequence[Sequence[Sequence[str]]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[list[str]]:
    """Fuse many queries at once. Indexing is channel_results[channel][query]."""
    if not channel_results:
        return []

    n_queries = len(channel_results[0])
    for channel, results in enumerate(channel_results):
        if len(results) != n_queries:
            raise ValueError(
                f"channel {channel} covers {len(results)} queries but channel "
                f"0 covers {n_queries}; every channel must answer every query"
            )

    return [
        fuse(
            [results[query] for results in channel_results],
            k=k,
            weights=weights,
            limit=limit,
        )
        for query in range(n_queries)
    ]
