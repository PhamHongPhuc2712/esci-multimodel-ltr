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
import json
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

# Which encoder wrote a store, beside its vectors. Two encoders can share a
# width - all-MiniLM-L12-v2 and bge-small-en-v1.5 are both 384-d - and a store
# queried by the wrong one passes every dimension check and returns
# real-looking products from another vector space.
MODEL_RECORD = "model.json"


def write_model_record(store_dir: Path, model_name: str, *, max_seq: int) -> None:
    """Record the encoder and token budget a store was written with."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / MODEL_RECORD).write_text(
        json.dumps({"model": str(model_name), "max_seq": int(max_seq)}) + "\n",
        encoding="utf-8",
    )


def check_model_record(
    store_dir: Path, model_name: str, *, max_seq: int | None = None
) -> None:
    """Raise unless `model_name` (and `max_seq`, when given) wrote the store.

    A store with vectors and no record predates the record: it is Plan 4's
    data/embeddings/dense, written by DEFAULT_MODEL, and only that model may
    read it. An empty store has no vectors to disagree with.

    The token budget matters as much as the model for a *writer*: a store
    resumed at a different `--max-seq` would hold vectors from two budgets
    under a record claiming one. Until Plan 12 only the model was compared,
    and the record was rewritten on every run.
    """
    from src.embedding_store import VECTORS_NAME

    store_dir = Path(store_dir)
    record_path = store_dir / MODEL_RECORD
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        recorded = record["model"]
        if recorded != str(model_name):
            raise ValueError(
                f"the store at {store_dir} was written by {recorded!r}, not "
                f"{model_name!r}; queries from another encoder land in a "
                "different vector space and nothing else would notice"
            )
        if max_seq is not None and "max_seq" in record and int(record["max_seq"]) != int(max_seq):
            raise ValueError(
                f"the store at {store_dir} was written at max_seq "
                f"{int(record['max_seq'])}, not {int(max_seq)}; resuming it would "
                "mix vectors from two token budgets under one record"
            )
        return
    vectors = store_dir / VECTORS_NAME
    if vectors.exists() and vectors.stat().st_size > 0 and str(model_name) != DEFAULT_MODEL:
        raise ValueError(
            f"the store at {store_dir} has no record of its encoder, so it is "
            f"Plan 4's, written by {DEFAULT_MODEL!r}; refusing to read it with "
            f"{model_name!r}"
        )


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

    check_model_record(store_dir, model_name)
    resolved = resolve_device(device)
    model = SentenceTransformer(model_name, device=resolved)
    store = open_store(store_dir, dim=_embedding_dim(model))
    if len(store) == 0:
        raise FileNotFoundError(
            f"the dense store at {store_dir} is empty; run python -m src.dense_embed"
        )
    return DenseChannel(store=store, model=model, device=resolved)


def _embedding_dim(model: Any) -> int:
    """The encoder's output width, across sentence-transformers versions."""
    getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension"
    )
    return int(getter())


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
    parser.add_argument(
        "--max-seq", type=int, default=None,
        help="token budget per document; default is the model's own")
    parser.add_argument(
        "--half", action="store_true",
        help="encode in float16: 2.5x faster on the 3080, cosine to float32 "
             ">= 0.99975 - the store keeps float16 anyway")
    args = parser.parse_args()

    fields = tuple(args.fields.split(","))
    frame = pd.read_parquet(args.products, columns=["product_id", *fields])
    if args.limit is not None:
        frame = frame.head(args.limit)

    check_model_record(args.store, args.model)
    resolved = resolve_device(args.device)
    model = SentenceTransformer(args.model, device=resolved)
    if args.max_seq is not None:
        model.max_seq_length = args.max_seq
    if args.half:
        model.half()
    # The budget is known only once the model is loaded; check it before any
    # vector is written, and write the record only for a fresh store. A resume
    # never rewrites it.
    check_model_record(args.store, args.model, max_seq=model.max_seq_length)
    store = open_store(args.store, dim=_embedding_dim(model))
    if len(store) == 0:
        write_model_record(args.store, args.model, max_seq=model.max_seq_length)
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
