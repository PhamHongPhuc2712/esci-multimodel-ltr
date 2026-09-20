"""The lexical retrieval channel: BM25 over the full product corpus.

Built once by the CLI and memory-mapped by every consumer afterwards. That
split is not tidiness - measured on the real 1,215,854-product corpus, the
naive build-and-query-in-one-process path peaks at **18.3 GB** on a 23 GB
machine, while the same index saved and reopened with `mmap=True` costs
**1.55 GB** and answers faster (28 ms/query amortised over a batch, against
37). Building still peaks at ~18 GB even with the corpus strings freed - the
caller's DataFrame is live throughout - so the build is a separate CLI run and
never happens inside a consumer. The index is 1.1 GB on disk.

`bm25s.retrieve` returns *positions into the indexed matrix*, not product ids,
so the id array is written by the same function from the same frame and its
length is asserted on open. A mismatched id array returns real-looking
products for every query and raises nothing: recall simply comes out low,
which reads as a weak retriever rather than a broken one.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baseline_sbert import product_text

DEFAULT_INDEX_DIR = Path("data/bm25")
DEFAULT_PRODUCTS = Path("data/combined/products.parquet")
IDS_NAME = "product_ids.npy"

# The coalesced `description` rather than either source column: Plan 2 measured
# it at 89.2% coverage against ESCI's own 52.2%, and its label confound largely
# cancels (-0.0065, not significant).
DEFAULT_FIELDS = ("product_title", "description", "product_bullet_point")


def _stemmer() -> Any:
    import Stemmer

    return Stemmer.Stemmer("english")


def tokenize_texts(texts: Sequence[str], stemmer: Any | None = None) -> Any:
    """Tokenise with English stopwords and stemming, as the index expects.

    Queries and documents must go through the same function - a query stemmed
    differently from the corpus matches nothing, and nothing errors.
    """
    import bm25s

    return bm25s.tokenize(
        list(texts),
        stopwords="en",
        stemmer=stemmer if stemmer is not None else _stemmer(),
        show_progress=False,
    )


def build_index(
    products: pd.DataFrame,
    fields: Sequence[str] = DEFAULT_FIELDS,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> int:
    """Build and save the index plus its id array. Returns the row count.

    The id array is written here, from this frame, in this order. Nothing else
    may write it.
    """
    import bm25s

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    product_ids = products["product_id"].to_numpy(dtype=object)
    texts = product_text(products, list(fields)).tolist()

    tokens = tokenize_texts(texts)
    del texts  # ~2 GB of strings, freed before indexing. Measured end to end
    # the build still peaks near 18 GB: the caller's DataFrame stays live.

    index = bm25s.BM25()
    index.index(tokens, show_progress=False)
    index.save(str(index_dir), corpus=None)
    np.save(index_dir / IDS_NAME, product_ids.astype(str), allow_pickle=False)
    return len(product_ids)


@dataclass(frozen=True)
class Bm25Channel:
    """A memory-mapped BM25 index. Open it via open_channel."""

    index: Any
    product_ids: np.ndarray
    stemmer: Any

    @property
    def n_products(self) -> int:
        return len(self.product_ids)

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        """Ranked product_ids per query, best first, at most k."""
        queries = list(queries)
        if not queries:
            return []

        k = min(k, self.n_products)
        tokens = tokenize_texts(queries, stemmer=self.stemmer)
        rows, scores = self.index.retrieve(
            tokens, k=k, show_progress=False, n_threads=8
        )

        results: list[list[str]] = []
        for row, score in zip(rows, scores):
            # A zero score is "this document shares no term with the query".
            # Returning those would hand RRF a full-length ranked list of
            # arbitrary documents to vote for.
            hits = [
                str(self.product_ids[position])
                for position, value in zip(row, score)
                if value > 0
            ]
            results.append(hits)
        return results


def open_channel(index_dir: Path = DEFAULT_INDEX_DIR) -> Bm25Channel:
    """Memory-map a built index, checking it agrees with its id array."""
    import bm25s

    index_dir = Path(index_dir)
    ids_path = index_dir / IDS_NAME
    if not ids_path.exists():
        raise FileNotFoundError(
            f"no bm25 index at {index_dir}; run python -m src.bm25_index"
        )

    index = bm25s.BM25.load(str(index_dir), mmap=True, load_corpus=False)
    product_ids = np.load(ids_path, allow_pickle=False)

    n_indexed = int(index.scores["num_docs"])
    if len(product_ids) != n_indexed:
        raise ValueError(
            f"the index holds {n_indexed:,} documents but its ids file has "
            f"{len(product_ids):,} entries; retrieve() returns row positions, "
            "so a mismatch silently returns the wrong products for every query"
        )
    return Bm25Channel(index=index, product_ids=product_ids, stemmer=_stemmer())


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    args = parser.parse_args()

    fields = tuple(args.fields.split(","))
    frame = pd.read_parquet(args.products, columns=["product_id", *fields])
    print(f"indexing {len(frame):,} products on {fields}")
    print("this peaks near 18 GB of RAM and takes about five minutes")

    started = time.time()
    n = build_index(frame, fields=fields, index_dir=args.index_dir)
    print(f"indexed {n:,} products in {time.time() - started:.0f}s -> {args.index_dir}")

    channel = open_channel(args.index_dir)
    hits = channel.search(["stainless steel water bottle"], k=5)[0]
    print(f"smoke test: {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
