"""Stream image URLs into CLIP vectors, never writing an image to disk.

One pass: URL -> bytes -> decoded image -> 512-d vector -> store, a chunk at a
time so memory stays bounded and progress is written as it goes. The run takes
between 1.8 and 7.3 hours for the re-ranking scope - three measured samples
varied sixfold in throughput - and will be interrupted, so every chunk is
appended before the next is fetched.

The network is the bottleneck by roughly 5x: 3,378-5,516 images/minute fetched
against 18,140/minute embedded on the RTX 3080. Embedding in flight is
therefore free, which is what makes never persisting the images practical.

`embed_urls` takes `fetch` and `encode_images` as callables, so dedup,
chunking, skip-existing and failure accounting are all tested with no network,
no GPU and no model.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.clip_encoder import DEFAULT_BATCH_SIZE, EMBEDDING_DIM
from src.combine import DEFAULT_DEST as COMBINED_DEST
from src.embedding_store import EmbeddingStore, open_store
from src.image_fetch import DEFAULT_CONCURRENCY, fetch_many

DEFAULT_STORE_ROOT = Path("data/embeddings")
DEFAULT_PRODUCTS = COMBINED_DEST / "products.parquet"

# 512 URLs in flight is ~5 MB of image bytes at the measured 10 KB mean, and
# eight full CLIP batches - enough to keep the GPU fed without holding a
# meaningful fraction of the run in memory.
DEFAULT_CHUNK = 512

SCOPES = ("rerank", "catalogue")

# The one image store. The rerank scope's URLs were a strict subset of the
# catalogue's (0 of 361,875 missing, re-checked 2026-09-30), so its separate
# store was deleted; a scope now picks the products, never the store. Opening
# data/embeddings/rerank/ would return an empty store rather than raise.
STORE_NAME = "catalogue"


@dataclass
class RunStats:
    requested: int = 0
    skipped: int = 0
    fetched: int = 0
    failed: int = 0
    embedded: int = 0
    seconds: float = 0.0
    status_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        rate = 60 * self.embedded / self.seconds if self.seconds else 0.0
        return {
            "requested": self.requested,
            "skipped": self.skipped,
            "fetched": self.fetched,
            "failed": self.failed,
            "embedded": self.embedded,
            "seconds": round(self.seconds, 1),
            "images_per_minute": round(rate, 1),
            "status_counts": self.status_counts,
        }


def product_image_urls(
    scope: str, products_path: Path = DEFAULT_PRODUCTS
) -> pd.DataFrame:
    """product_id and url for every product that has an image URL."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")

    frame = pd.read_parquet(products_path, columns=["product_id", "s_image_url"])
    urls = frame["s_image_url"]
    has_url = urls.notna() & (urls.astype("str").str.len() > 0)
    frame = frame.loc[has_url].rename(columns={"s_image_url": "url"})

    if scope == "rerank":
        from src.coverage import rerank_product_ids

        frame = frame.loc[frame["product_id"].isin(rerank_product_ids())]
    return frame[["product_id", "url"]].reset_index(drop=True)


def pending_urls(frame: pd.DataFrame, store: EmbeddingStore) -> list[str]:
    """Distinct URLs not already in the store, in first-seen order.

    Distinct because 932,320 catalogue products share 887,041 URLs: keying the
    fetch on the product would repeat 5% of a multi-hour run and write the
    same vector several times.
    """
    known = store.known_keys()
    seen: dict[str, None] = {}
    for url in frame["url"]:
        if url not in known:
            seen.setdefault(url, None)
    return list(seen)


def embed_urls(
    urls: Sequence[str],
    store: EmbeddingStore,
    encode_images: Callable[..., Any],
    *,
    fetch: Callable[..., Iterable[Any]] = fetch_many,
    chunk: int = DEFAULT_CHUNK,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[RunStats], None] | None = None,
) -> RunStats:
    """Fetch, decode, embed and store, one chunk at a time."""
    stats = RunStats(requested=len(urls))

    known = store.known_keys()
    ordered: dict[str, None] = {}
    for url in urls:
        if url in known:
            stats.skipped += 1
        else:
            ordered.setdefault(url, None)
    todo = list(ordered)

    started = time.time()
    for start in range(0, len(todo), chunk):
        batch_urls = todo[start : start + chunk]
        images: list[Any] = []
        keys: list[str] = []
        for result in fetch(batch_urls, concurrency=concurrency):
            status = str(result.status)
            stats.status_counts[status] = stats.status_counts.get(status, 0) + 1
            if result.ok:
                stats.fetched += 1
                images.append(result.image)
                keys.append(result.url)
            else:
                stats.failed += 1

        if images:
            vectors = encode_images(images, batch_size=batch_size)
            store.append(keys, vectors)
            stats.embedded += len(keys)
        stats.seconds = time.time() - started
        if progress is not None:
            progress(stats)

    stats.seconds = time.time() - started
    return stats


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="catalogue", choices=list(SCOPES))
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many URLs"
    )
    args = parser.parse_args()

    frame = product_image_urls(args.scope, products_path=args.products)
    store = open_store(args.store_root / STORE_NAME, dim=EMBEDDING_DIM)
    todo = pending_urls(frame, store)
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"scope {args.scope}: {len(frame):,} products with a URL, "
        f"{frame['url'].nunique():,} distinct, {len(store):,} already stored, "
        f"{len(todo):,} to fetch"
    )

    from src.clip_encoder import load_encoder

    encoder = load_encoder(device=args.device)
    print(f"encoding on {encoder.device}, dim {encoder.dim}")

    def report(stats: RunStats) -> None:
        done = stats.embedded + stats.failed
        rate = 60 * done / stats.seconds if stats.seconds else 0.0
        print(
            f"\r  {done:,}/{len(todo):,}  embedded {stats.embedded:,}  "
            f"failed {stats.failed:,}  {rate:,.0f}/min",
            end="",
            flush=True,
        )

    stats = embed_urls(
        todo,
        store,
        encoder.encode_images,
        chunk=args.chunk,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        progress=report,
    )
    print()
    print(json.dumps(stats.to_dict(), indent=2))
    print(f"store now holds {len(store):,} vectors at {args.store_root / STORE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
