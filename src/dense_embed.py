"""The dense-text retrieval channel: SBERT over the full product corpus.

Mirrors src/embed_images.py in shape - a chunked, resumable pass into an
append-only EmbeddingStore - but keyed by `product_id` rather than by image
URL, and 384-dimensional rather than 512. Measured on the RTX 3080: 821
docs/s, so all 1,215,854 products take about 25 minutes and 0.93 GB at float16.

Field order is load-bearing. `all-MiniLM-L12-v2` has `max_seq_length = 128`
tokens and the concatenated product text averages 1,748 characters - roughly
440 tokens - so about 70% of it never reaches the model. Whatever leads is
what gets embedded. The title leads deliberately; Plan 3 learned the same
lesson the expensive way when a 77-*character* title slice cost its semantic
gate 6 points.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baseline_sbert import DEFAULT_MODEL, product_text
from src.embedding_store import EmbeddingStore, open_store
from src.recall import Channel  # noqa: F401  (DenseChannel implements it)

DEFAULT_STORE = Path("data/embeddings/dense")
DEFAULT_PRODUCTS = Path("data/combined/products.parquet")

DENSE_DIM = 384
# all-MiniLM-L12-v2's context window. Recorded here, beside the field order it
# justifies, so a model swap that changes it cannot pass silently.
MAX_SEQ_TOKENS = 128
DEFAULT_FIELDS = ("product_title", "description", "product_bullet_point")

DEFAULT_CHUNK = 8192


def embed_products(
    products: pd.DataFrame,
    store: EmbeddingStore,
    encode_texts: Callable[..., Any],
    *,
    fields: Sequence[str] = DEFAULT_FIELDS,
    chunk: int = DEFAULT_CHUNK,
    batch_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every product not already in the store. Returns how many were added."""
    known = store.known_keys()
    todo = products.loc[~products["product_id"].isin(known)]
    if todo.empty:
        return 0

    texts = product_text(todo, list(fields))
    ids = todo["product_id"].astype(str).tolist()
    added = 0
    for start in range(0, len(ids), chunk):
        keys = ids[start : start + chunk]
        batch = texts.iloc[start : start + chunk].tolist()
        vectors = np.asarray(
            encode_texts(batch, batch_size=batch_size), dtype=np.float32
        )
        store.append(keys, vectors)
        added += len(keys)
        if progress is not None:
            progress(added, len(ids))
    return added


@dataclass(frozen=True)
class DenseChannel:
    """Query text -> SBERT vector -> nearest products. Open it via open_channel."""

    store: EmbeddingStore
    model: Any
    device: str

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        from src.vector_search import top_k

        queries = list(queries)
        if not queries:
            return []
        vectors = self.model.encode(
            queries,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        keys, _ = top_k(vectors, self.store, k, device=self.device)
        return keys


def open_channel(
    store_dir: Path = DEFAULT_STORE,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
) -> DenseChannel:
    """Open the dense store and load the query encoder onto the chosen device."""
    from sentence_transformers import SentenceTransformer

    from src.clip_encoder import resolve_device

    store = open_store(store_dir, dim=DENSE_DIM)
    if len(store) == 0:
        raise FileNotFoundError(
            f"the dense store at {store_dir} is empty; run python -m src.dense_embed"
        )
    resolved = resolve_device(device)
    return DenseChannel(
        store=store,
        model=SentenceTransformer(model_name, device=resolved),
        device=resolved,
    )


def _main() -> int:
    from sentence_transformers import SentenceTransformer

    from src.clip_encoder import resolve_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    fields = tuple(args.fields.split(","))
    frame = pd.read_parquet(args.products, columns=["product_id", *fields])
    if args.limit is not None:
        frame = frame.head(args.limit)

    store = open_store(args.store, dim=DENSE_DIM)
    resolved = resolve_device(args.device)
    model = SentenceTransformer(args.model, device=resolved)
    print(
        f"{len(frame):,} products, {len(store):,} already embedded; "
        f"encoding on {resolved}, {model.max_seq_length} tokens per document"
    )

    started = time.time()

    def report(done: int, total: int) -> None:
        rate = done / max(time.time() - started, 1e-9)
        print(f"\r  {done:,}/{total:,}  {rate:,.0f} docs/s", end="", flush=True)

    added = embed_products(
        frame,
        store,
        lambda texts, batch_size: model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        fields=fields,
        chunk=args.chunk,
        batch_size=args.batch_size,
        progress=report,
    )
    print()
    print(f"added {added:,} vectors in {time.time() - started:.0f}s")
    print(f"store now holds {len(store):,} vectors at {args.store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
