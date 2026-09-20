"""Recall@k over the full corpus, and the relevant sets it is measured against.

PROJECT_SPEC.md §2 is explicit that this is a re-ranking task and that recall
is evaluated *separately*: Recall@k over the ~1.2M corpus, NDCG over the judged
candidate list. This module owns the first and knows nothing about ranking.

Two decisions are made here rather than left to callers, because both are
silent when got wrong:

  * **A query with no relevant product is skipped, and the skip is counted.**
    Its recall is 0/0. Scored as 0.0 it punishes a retriever for a query no
    retriever could have scored; dropped without a count, two runs with
    different denominators look comparable. `query_id` 45928 of the real test
    split - 15 judgements, every one Substitute - is why this is not
    hypothetical.
  * **The mean is macro-averaged over queries**, not micro-averaged over
    products, so a query with 40 relevant products does not outvote one with
    three.

`recall_at_k` takes already-retrieved lists rather than a Channel, so the
metric is tested against hand-written lists in milliseconds and no index has
to exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

DEFAULT_JUDGEMENTS = Path("data/combined/judgements.parquet")

# The spec's two reasonable definitions of "relevant" for retrieval. E alone is
# the strict reading; E+S matches what a shopper would accept and is the only
# definition under which every test query has a non-empty set.
RELEVANT_SETS: dict[str, tuple[str, ...]] = {"E": ("E",), "E+S": ("E", "S")}

DEFAULT_KS: tuple[int, ...] = (10, 50, 100, 500, 1000)


@runtime_checkable
class Channel(Protocol):
    """A retrieval channel: queries in, ranked product_ids out."""

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        """One ranked list of product_ids per query, best first, at most k."""


@dataclass(frozen=True)
class RecallResult:
    per_query: dict[int, float] = field(default_factory=dict)
    n_queries: int = 0
    n_skipped: int = 0
    k: int = 0

    @property
    def mean(self) -> float:
        """Macro-average over the queries that had a relevant product.

        NaN rather than 0.0 when nothing was measurable: a silent zero reads
        as "the retriever found nothing" when the truth is "nothing was
        measurable", and the two call for opposite reactions.
        """
        if not self.per_query:
            return math.nan
        return sum(self.per_query.values()) / len(self.per_query)

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "recall": self.mean,
            "n_queries": self.n_queries,
            "n_skipped": self.n_skipped,
        }


def relevant_sets(
    judgements: pd.DataFrame, relevance: str = "E"
) -> dict[int, set[str]]:
    """product_ids counted as relevant, per query_id.

    Queries whose set would be empty are omitted entirely rather than mapped
    to an empty set, so callers cannot accidentally divide by zero.
    """
    if relevance not in RELEVANT_SETS:
        raise ValueError(
            f"unknown relevance {relevance!r}; expected one of "
            f"{tuple(RELEVANT_SETS)}"
        )
    labels = RELEVANT_SETS[relevance]
    matching = judgements.loc[judgements["esci_label"].isin(labels)]
    grouped = matching.groupby("query_id")["product_id"].apply(set)
    return {int(qid): products for qid, products in grouped.items() if products}


def recall_at_k(
    retrieved: Mapping[int, Sequence[str]],
    relevant: Mapping[int, set[str]],
    k: int,
) -> RecallResult:
    """Share of each query's relevant products found in its top k."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    per_query: dict[int, float] = {}
    skipped = 0
    for query_id, ranked in retrieved.items():
        wanted = relevant.get(query_id)
        if not wanted:
            skipped += 1
            continue
        # set() de-duplicates: a channel that returns the same product twice
        # must not be credited twice.
        found = set(ranked[:k]) & wanted
        per_query[query_id] = len(found) / len(wanted)

    return RecallResult(
        per_query=per_query,
        n_queries=len(per_query),
        n_skipped=skipped,
        k=k,
    )


def recall_table(
    retrieved: Mapping[int, Sequence[str]],
    relevant: Mapping[int, set[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[int, RecallResult]:
    """Recall at each k, from one set of retrieved lists."""
    return {k: recall_at_k(retrieved, relevant, k) for k in ks}


def load_queries(
    split: str,
    judgements_path: Path = DEFAULT_JUDGEMENTS,
    folds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """One row per distinct query in `split`, optionally limited to folds.

    `folds` is how model selection stays off the test split: pass the frozen
    validation folds from Plan 1 and tune against those.
    """
    frame = pd.read_parquet(
        judgements_path, columns=["query_id", "query", "split", "fold"]
    )
    frame = frame.loc[frame["split"] == split]
    if folds is not None:
        frame = frame.loc[frame["fold"].isin(list(folds))]
    return (
        frame.drop_duplicates("query_id")[["query_id", "query"]]
        .reset_index(drop=True)
    )
