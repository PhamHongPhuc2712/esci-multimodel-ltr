"""The three retrieval scores, for judged (query, product) pairs.

PROJECT_SPEC.md §4.2 lists BM25 score, dense text similarity and CLIP image
similarity as the coarse ranker's "Retrieval scores". They are the only
features that need an artefact on disk, which is why they live apart from
src/features.py.

Measured on validation fold 0 (4,130 queries, 82,751 judgements), each one
alone re-ranks the judged candidate list well above the 0.7440 floor:

    bm25_score      0.8230    pair coverage 0.9999
    dense_sim       0.8285    pair coverage 1.0000
    clip_image_sim  0.7905    pair coverage 0.7939

Retrieval, not re-ranking, is what Plan 4's channels do: they return the top k
out of 1,215,854 products. Here the product is given and only its score is
wanted, which is a different call - `get_scores` rather than `retrieve` for
BM25, and a keyed lookup rather than a top-k for the two vector stores.

Two failure modes are handled rather than assumed away:

  * **A query can tokenise to nothing.** After English stopword removal and
    the index's vocabulary filter, one of fold 0's 4,130 queries has no tokens
    left, and `bm25s.BM25.get_scores([])` raises IndexError from
    `query_tokens_single[0]`. Those rows get NaN and the count is reported;
    the 15.7-minute pass finishes.
  * **A product can have no vector.** 20.6% of judged pairs have no image
    embedding. `EmbeddingStore.lookup` returns NaN plus a presence mask, and
    that NaN is propagated: a 0.0 cosine is a score meaning "orthogonal to the
    query", which would rank a never-scraped product below a genuinely bad
    match.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUT = Path("data/features")
DEFAULT_PRODUCTS = Path("data/combined/products.parquet")
DEFAULT_JUDGEMENTS = Path("data/combined/judgements.parquet")

PAIR_SCORE_COLUMNS: tuple[str, ...] = ("bm25_score", "dense_sim", "clip_image_sim")

DENSE_DIM = 384
CLIP_DIM = 512


def bm25_token_lists(
    queries: Sequence[str],
    channel: Any,
    *,
    tokenize: Callable[[Sequence[str], Any], Any] | None = None,
) -> list[list[str]]:
    """Token *strings* per query, as `get_scores` wants them.

    `bm25_index.tokenize_texts` returns a Tokenized carrying `.ids` (per
    query, integer ids) and `.vocab` (token -> id). Those ids index the
    *batch's* vocabulary, not the index's, so handing them to `get_scores` -
    which the signature permits - scores the wrong terms and raises nothing.
    """
    if tokenize is None:
        from src.bm25_index import tokenize_texts

        tokenize = tokenize_texts

    tokenized = tokenize(list(queries), channel.stemmer)
    ids = getattr(tokenized, "ids", tokenized)
    vocab = getattr(tokenized, "vocab", None)
    if vocab is None:
        return [list(row) for row in ids]
    inverse = [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return [[inverse[i] for i in row] for row in ids]


def bm25_pair_scores(
    pairs: pd.DataFrame,
    *,
    tokenize: Callable[[Sequence[str]], Sequence[Sequence[str]]],
    score_query: Callable[[Sequence[str]], np.ndarray],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, int]:
    """BM25 score for every row of `pairs`, aligned to `pairs`.

    `pairs` needs `query_id`, `query` and `row` - the latter being the
    product's position in the indexed matrix, which is what
    `bm25s` scores are indexed by. Returns the scores and the number of
    queries that tokenised to nothing; those queries' rows are NaN.

    Scoring is per distinct query, not per row: there are ~20 judgements per
    query, so per-row scoring would be 20x the work for the same answer.
    """
    for column in ("query_id", "query", "row"):
        if column not in pairs.columns:
            raise KeyError(
                f"pairs has no {column!r} column; bm25 scores are read by "
                "matrix position, and guessing one would score the wrong product"
            )

    out = np.full(len(pairs), np.nan, dtype=np.float32)
    rows = pairs["row"].to_numpy()
    groups = pairs.groupby("query_id", sort=False)
    query_ids = list(groups.groups)
    texts = [str(pairs["query"].iloc[groups.indices[q][0]]) for q in query_ids]

    token_lists = list(tokenize(texts))
    n_empty = 0
    for done, (query_id, query_tokens) in enumerate(zip(query_ids, token_lists), start=1):
        if len(query_tokens) == 0:
            # Every term was a stopword or out of vocabulary. get_scores would
            # raise IndexError here; NaN is the honest score.
            n_empty += 1
        else:
            index = groups.indices[query_id]
            scores = np.asarray(score_query(query_tokens))
            out[index] = scores[rows[index]]
        if progress is not None:
            progress(done, len(query_ids))
    return out, n_empty


def similarity_scores(
    query_vectors: np.ndarray,
    query_index: Sequence[int],
    keys: Sequence[str],
    store: Any,
) -> np.ndarray:
    """Cosine of each row's query vector against its key's stored vector.

    One function for both vector channels: the dense store is keyed by
    product_id and the CLIP store by image URL, and the arithmetic is
    identical. Stored vectors are L2-normalised by Plan 3's contract, so the
    cosine is a plain dot product.

    A key the store does not hold scores NaN. `lookup` already returns a NaN
    row and a presence mask for exactly this reason; the mask is re-applied
    here so that a future lookup that returned zeros could not slip through.
    """
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    if len(keys) == 0:
        return np.zeros(0, dtype=np.float32)
    if query_vectors.ndim != 2:
        raise ValueError(f"expected a 2-D query matrix, got {query_vectors.ndim}-D")
    if query_vectors.shape[1] != store.dim:
        raise ValueError(
            f"query width {query_vectors.shape[1]} does not match the store's "
            f"dim {store.dim}"
        )

    matrix, present = store.lookup(list(keys))
    left = query_vectors[np.asarray(query_index, dtype=np.int64)]
    scores = np.einsum("ij,ij->i", left, matrix).astype(np.float32)
    scores[~present] = np.nan
    return scores


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--judgements", type=Path, default=DEFAULT_JUDGEMENTS)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image-store", type=Path,
                        default=Path("data/embeddings/catalogue"))
    parser.add_argument("--dense-store", type=Path, default=Path("data/embeddings/dense"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from src.baseline_sbert import DEFAULT_MODEL
    from src.bm25_index import open_channel as open_bm25
    from src.clip_encoder import load_encoder, resolve_device
    from src.embedding_store import open_store

    judgements = pd.read_parquet(
        args.judgements, columns=["query_id", "query", "product_id", "split"]
    )
    pairs = judgements.loc[judgements["split"] == args.split].reset_index(drop=True)
    queries = pairs.drop_duplicates("query_id")
    print(f"{len(pairs):,} judged pairs, {len(queries):,} queries, split {args.split}")

    # --- BM25: 31.5 ms/query, ~16 minutes for both splits -------------------
    channel = open_bm25()
    row_of = {pid: i for i, pid in enumerate(channel.product_ids)}
    unknown = set(pairs["product_id"]) - row_of.keys()
    if unknown:
        raise ValueError(
            f"{len(unknown)} judged products are not in the bm25 index "
            f"(for example {sorted(unknown)[:3]}); measured 0 on the real "
            "corpus, so this means the index and the product table disagree"
        )
    pairs["row"] = pairs["product_id"].map(row_of).astype(np.int64)

    started = time.time()
    bm25, n_empty = bm25_pair_scores(
        pairs,
        tokenize=lambda texts: bm25_token_lists(texts, channel),
        score_query=channel.index.get_scores,
        progress=lambda done, total: print(
            f"\r  bm25 {done:,}/{total:,}  {done / max(time.time() - started, 1e-9):.0f}/s",
            end="",
            flush=True,
        ),
    )
    print()
    print(f"  bm25 done in {time.time() - started:.0f}s; "
          f"{n_empty} queries tokenised to nothing")
    del channel, row_of

    query_row = {qid: i for i, qid in enumerate(queries["query_id"])}
    query_index = pairs["query_id"].map(query_row).to_numpy()
    device = resolve_device(args.device)
    print(f"encoding queries on {device}")

    # --- dense text ---------------------------------------------------------
    sbert = SentenceTransformer(DEFAULT_MODEL, device=device)
    dense_queries = sbert.encode(
        queries["query"].astype(str).tolist(),
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dense_store = open_store(args.dense_store, dim=DENSE_DIM)
    dense = similarity_scores(
        dense_queries, query_index, pairs["product_id"].astype(str).tolist(), dense_store
    )
    print(f"  dense_sim covers {np.isfinite(dense).mean():.4f} of pairs")
    del sbert, dense_store

    # --- CLIP image ---------------------------------------------------------
    urls = pd.read_parquet(args.products, columns=["product_id", "s_image_url"])
    url_of = dict(zip(urls["product_id"], urls["s_image_url"]))
    del urls
    encoder = load_encoder(device=args.device)
    clip_queries = encoder.encode_texts(queries["query"].astype(str).tolist())
    image_store = open_store(args.image_store, dim=CLIP_DIM)
    image = similarity_scores(
        clip_queries,
        query_index,
        [str(url_of.get(p) or "") for p in pairs["product_id"]],
        image_store,
    )
    print(f"  clip_image_sim covers {np.isfinite(image).mean():.4f} of pairs")

    frame = pd.DataFrame(
        {
            "query_id": pairs["query_id"].to_numpy(),
            "product_id": pairs["product_id"].to_numpy(),
            "bm25_score": bm25,
            "dense_sim": dense,
            "clip_image_sim": image,
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"pair-scores-{args.split}.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {len(frame):,} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
