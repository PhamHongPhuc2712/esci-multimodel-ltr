"""Zero-shot SBERT re-ranking: the baseline that proves the harness.

PROJECT_SPEC.md §5 puts SBERT_text zero-shot at 0.8292 on this split with
all-MiniLM-L12-v2. Reproducing it is the acceptance test for the loader, the
metric and the floor together, which is why it is in Plan 1 rather than in the
retrieval plan.

Scoring takes an `encode` callable rather than a model so the logic is unit
tested in milliseconds; the encode run itself lives in __main__ and goes to
the GPU when one is present.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import load_split
from src.runs import run_from_scores, write_run

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
Encoder = Callable[[list[str]], np.ndarray]


def product_text(products: pd.DataFrame, fields: Sequence[str]) -> pd.Series:
    """Join the requested product fields into one string per product.

    Null fields are dropped rather than stringified: str(None) would embed the
    literal "None" as if it were product text, and ~half of product_color is
    null.
    """
    missing = [f for f in fields if f not in products.columns]
    if missing:
        raise KeyError(f"no such product field(s): {missing}")
    parts = [products[f].fillna("").astype(str).str.strip() for f in fields]
    joined = parts[0]
    for part in parts[1:]:
        joined = (joined + " " + part).str.strip()
    return joined.str.replace(r"\s+", " ", regex=True)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def score_pairs(
    judgements: pd.DataFrame,
    products: pd.DataFrame,
    encode: Encoder,
    fields: Sequence[str],
) -> pd.DataFrame:
    """Cosine similarity for every judged (query, product) pair.

    Encodes each distinct query once and each distinct product once: there are
    ~20 judgements per query, so encoding per judgement would be 20x the work.
    """
    unknown = set(judgements["product_id"]) - set(products["product_id"])
    if unknown:
        raise ValueError(
            f"{len(unknown)} judged products have no metadata row "
            f"(for example {sorted(unknown)[:3]}); scoring them as zero would "
            "silently depress the result"
        )

    queries = judgements[["query_id", "query"]].drop_duplicates("query_id")
    query_vectors = _normalise(encode(list(queries["query"])))
    query_row = {qid: i for i, qid in enumerate(queries["query_id"])}

    catalogue = products.drop_duplicates("product_id").reset_index(drop=True)
    product_vectors = _normalise(encode(list(product_text(catalogue, fields))))
    product_row = {pid: i for i, pid in enumerate(catalogue["product_id"])}

    left = query_vectors[[query_row[q] for q in judgements["query_id"]]]
    right = product_vectors[[product_row[p] for p in judgements["product_id"]]]
    return pd.DataFrame(
        {
            "query_id": judgements["query_id"].to_numpy(),
            "product_id": judgements["product_id"].to_numpy(),
            "score": np.einsum("ij,ij->i", left, right),
        }
    )


def resolve_device(requested: str) -> str:
    """Pick the encode device. "auto" means the GPU when there is one.

    Explicit rather than relying on the library default: a silent fall back to
    CPU turns a one-minute encode into the better part of an hour, and looks
    identical in the logs.
    """
    import torch

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(
                f"--device {requested} requested but torch.cuda.is_available() "
                "is False; pass --device cpu to run on CPU deliberately"
            )
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fields", default="product_title")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="runs/sbert-title.trec")
    parser.add_argument("--tag", default="sbert-title")
    parser.add_argument(
        "--device", default="auto", help="auto (GPU if present), cuda, cuda:N or cpu"
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    device = resolve_device(args.device)
    print(f"encoding on {device}")
    model = SentenceTransformer(args.model, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=device,
        )

    loaded = load_split(args.split)
    scores = score_pairs(
        loaded.judgements, loaded.products, encode, args.fields.split(",")
    )
    write_run(run_from_scores(scores), Path(args.out), run_tag=args.tag)
    print(f"wrote {len(scores):,} scores to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
