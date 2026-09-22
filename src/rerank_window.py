"""The top-K window both Stage 3 arms re-rank, and the splice that puts it back.

PROJECT_SPEC.md §4.3 compares the two fine-rank variants "on the top-K from
Stage 2". This module owns what that means, because it is the one piece of
logic the two arms share and the one place a scale mismatch can corrupt a run.

**Scores are never mixed across stages.** A cross-encoder logit can be -8.4
while a lambdarank score is +3.1. Writing Stage 3 scores onto the window and
leaving Stage 2 scores on the tail interleaves two orderings that were never on
the same scale, so a product the reranker actively demoted can still land above
one it never looked at - silently, with NDCG moving for a reason that has
nothing to do with the reranker.

So the splice emits **rank-derived scores**: descending integers from n - 1
down to 0 over window ++ tail. The window is strictly above the tail by
construction, whatever either stage's native units were, and the tail keeps its
Stage 2 order exactly.

K = 10 by default. Measured on fold 0: an oracle reorder of the top-10 reaches
0.9533 against Stage 2's 0.8519 - two thirds of the +0.1481 total headroom - at
roughly a quarter of the LLM cost of ranking the whole list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

DEFAULT_K = 10


@dataclass(frozen=True)
class Window:
    """One query's top-K candidates and the tail no reranker will touch."""

    query_id: str
    window: tuple[str, ...]
    tail: tuple[str, ...]

    @property
    def documents(self) -> tuple[str, ...]:
        return self.window + self.tail


def windows(stage2: pd.DataFrame, k: int = DEFAULT_K) -> list[Window]:
    """Carve every query's Stage 2 ordering into a top-k window and a tail.

    Ties break on product_id so two runs of the same code carve the same
    window; otherwise the two arms would be re-ranking different candidate
    sets and the difference would be reported as a reranker effect.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    ordered = stage2.sort_values(
        ["query_id", "stage2_score", "product_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    out: list[Window] = []
    for query_id, group in ordered.groupby("query_id", sort=True):
        documents = [str(p) for p in group["product_id"]]
        out.append(
            Window(
                query_id=str(query_id),
                window=tuple(documents[:k]),
                tail=tuple(documents[k:]),
            )
        )
    return out


def splice_scores(
    window_order: Sequence[str], tail: Sequence[str]
) -> dict[str, float]:
    """Rank-derived scores placing `window_order` above `tail`, both in order.

    Never a blend of two stages' scales - see the module docstring.
    """
    overlap = set(window_order) & set(tail)
    if overlap:
        raise ValueError(
            f"{sorted(overlap)[:3]} appear in both the window and the tail; a "
            "document can only hold one rank"
        )
    documents = list(window_order) + list(tail)
    total = len(documents)
    return {doc: float(total - position) for position, doc in enumerate(documents)}


def stage2_run(windows_: Sequence[Window]) -> dict[str, dict[str, float]]:
    """The Stage 2 ordering itself, as a run. The baseline arm."""
    return {w.query_id: splice_scores(w.window, w.tail) for w in windows_}


def spliced_run(
    windows_: Sequence[Window],
    orderings: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Apply per-query re-rankings of the window, leaving every tail alone.

    A query absent from `orderings` keeps its Stage 2 order, which is what a
    reranker that declined to answer - a malformed LLM response, a timeout -
    must fall back to.
    """
    orderings = dict(orderings or {})
    known = {w.query_id for w in windows_}
    unknown = set(orderings) - known
    if unknown:
        raise KeyError(
            f"re-rankings given for queries that have no window: "
            f"{sorted(unknown)[:3]}"
        )

    run: dict[str, dict[str, float]] = {}
    for w in windows_:
        order = orderings.get(w.query_id)
        if order is None:
            order = list(w.window)
        elif sorted(order) != sorted(w.window):
            raise ValueError(
                f"query {w.query_id}: the re-ranking is not a permutation of "
                f"its window ({len(order)} documents against {len(w.window)}); "
                "a dropped candidate silently leaves the ranking and a "
                "duplicated one silently occupies two ranks"
            )
        run[w.query_id] = splice_scores(order, w.tail)
    return run
