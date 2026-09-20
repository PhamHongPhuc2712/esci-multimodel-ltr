"""Image coverage and the semantic gate.

Coverage divides by the product set, never by the products that happen to have
a URL - dividing by the latter is exactly how ESCI-S's 91.5% headline gets
mistaken for image coverage when the end-to-end figure is ~77.5%.

The gate is a retrieval check: encode N product images and their own titles,
and ask how often an image's nearest title is its own. Measured on 600 real
pairs: top-1 0.753, top-5 0.927, median rank 1, against a 0.00167 chance rate.
The threshold sits at 0.60 because it is there to catch catastrophic breakage
- ids misaligned with vectors, vectors unnormalised, the wrong pooling output
- not to police the third decimal.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MIN_TOP1 = 0.60
GATE_POOL = 600


class SemanticGateError(ValueError):
    """Raised when image and text embeddings no longer line up."""


@dataclass(frozen=True)
class CoverageRecord:
    scope: str
    products: int
    with_url: int
    embedded: int
    url_coverage: float
    embedding_coverage: float

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "products": self.products,
            "with_url": self.with_url,
            "embedded": self.embedded,
            "url_coverage": self.url_coverage,
            "embedding_coverage": self.embedding_coverage,
        }


def image_coverage(frame: pd.DataFrame, store, n_products: int, scope: str = "rerank"):
    """Coverage of `n_products`, by URL and by stored embedding."""
    if n_products <= 0:
        raise ValueError("n_products must be positive")
    known = store.known_keys()
    with_url = int(frame["product_id"].nunique())
    embedded = int(frame.loc[frame["url"].isin(known), "product_id"].nunique())
    return CoverageRecord(
        scope=scope,
        products=n_products,
        with_url=with_url,
        embedded=embedded,
        url_coverage=with_url / n_products,
        embedding_coverage=embedded / n_products,
    )


def top1_accuracy(image_vectors: np.ndarray, text_vectors: np.ndarray) -> float:
    """Share of images whose nearest text is their own, by cosine.

    Ties count as misses: two identical candidate texts must not be scored as
    a hit because argmax happened to land on the right index.
    """
    similarity = np.asarray(image_vectors, dtype=np.float32) @ np.asarray(
        text_vectors, dtype=np.float32
    ).T
    truth = np.diag(similarity)[:, None]
    better_or_equal = (similarity >= truth).sum(axis=1)
    return float((better_or_equal == 1).mean())


def _main() -> int:
    from src.clip_encoder import load_encoder
    from src.coverage import rerank_product_ids
    from src.embed_images import DEFAULT_PRODUCTS, DEFAULT_STORE_ROOT, product_image_urls
    from src.embedding_store import open_store
    from src.image_fetch import fetch_many

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="rerank", choices=["rerank", "catalogue"])
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--pool", type=int, default=GATE_POOL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("docs/results/image-embeddings.json"))
    args = parser.parse_args()

    frame = product_image_urls(args.scope, products_path=args.products)
    store = open_store(args.store_root / args.scope)
    n_products = (
        len(rerank_product_ids())
        if args.scope == "rerank"
        else len(pd.read_parquet(args.products, columns=["product_id"]))
    )
    record = image_coverage(frame, store, n_products, scope=args.scope)
    print(
        f"{record.scope}: {record.with_url:,}/{record.products:,} have a URL "
        f"({record.url_coverage:.2%}); {record.embedded:,} have an embedding "
        f"({record.embedding_coverage:.2%})"
    )

    # --- the semantic gate --------------------------------------------------
    titles = pd.read_parquet(
        args.products, columns=["product_id", "product_title", "s_image_url"]
    ).dropna(subset=["product_title", "s_image_url"])
    titles = titles.loc[titles["s_image_url"].isin(store.known_keys())]
    titles = titles.drop_duplicates("s_image_url").sample(
        n=min(args.pool, len(titles)), random_state=args.seed
    )

    encoder = load_encoder(device=args.device)
    results = {r.url: r for r in fetch_many(list(titles["s_image_url"]))}
    # The full title, not str(t)[:77]: CLIP's 77 is a *token* limit that the
    # processor already applies, while slicing 77 *characters* cuts 64.2% of
    # ESCI titles (median 96 chars) mid-word. Measured on this pool: the slice
    # costs 6 points of top-1, 70.0% against 76.0%.
    usable = [
        (results[u].image, str(t))
        for u, t in zip(titles["s_image_url"], titles["product_title"])
        if u in results and results[u].ok
    ]
    images = [i for i, _ in usable]
    texts = [t for _, t in usable]
    top1 = top1_accuracy(encoder.encode_images(images), encoder.encode_texts(texts))
    print(f"semantic gate: top-1 {top1:.3%} over a {len(usable)}-title pool "
          f"(chance {1 / max(len(usable), 1):.3%}, threshold {MIN_TOP1:.0%})")

    payload = record.to_dict() | {
        "semantic_gate": {
            "pool": len(usable),
            "top1": top1,
            "threshold": MIN_TOP1,
            "seed": args.seed,
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"written to {args.out}")

    if top1 < MIN_TOP1:
        raise SemanticGateError(
            f"top-1 {top1:.3%} is below {MIN_TOP1:.0%}; image and text "
            "embeddings are not lining up. Check that the store's keys match "
            "its rows, that vectors are L2-normalised, and that the encoder "
            "reads .pooler_output rather than .last_hidden_state."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
