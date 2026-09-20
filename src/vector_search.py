"""Top-k cosine search over an EmbeddingStore, without materialising the scores.

The dense-text and image channels are the same operation against different
stores: encode a query, cosine it against a float16 matrix, take the best k.
Writing that twice is how two channels drift apart, so it is written once here.

Nothing is materialised in full. 8,956 test queries against 1,215,854 products
is 1.09e10 scores - 43 GB at float32 - so the store is walked in row blocks and
the queries in their own blocks, keeping a running top-k. Peak memory is
`query_chunk x store_chunk x 4` bytes, 205 MB at the defaults, whatever the
corpus size.

Stored vectors are already L2-normalised (Plan 3 guarantees it, and
`dense_embed` does the same), so cosine is a plain dot product.
"""

from __future__ import annotations

import numpy as np

from src.embedding_store import EmbeddingStore

DEFAULT_QUERY_CHUNK = 256
DEFAULT_STORE_CHUNK = 200_000


def _matmul(queries: np.ndarray, block: np.ndarray, device: str) -> np.ndarray:
    """queries @ block.T, on the GPU when there is one."""
    if device == "cpu":
        return queries @ block.T

    import torch

    with torch.inference_mode():
        q = torch.from_numpy(queries).to(device)
        b = torch.from_numpy(block).to(device)
        return (q @ b.T).float().cpu().numpy()


def top_k(
    query_vectors: np.ndarray,
    store: EmbeddingStore,
    k: int,
    *,
    device: str = "auto",
    query_chunk: int = DEFAULT_QUERY_CHUNK,
    store_chunk: int = DEFAULT_STORE_CHUNK,
) -> tuple[list[list[str]], np.ndarray]:
    """The k nearest stored keys for each query vector, best first."""
    from src.clip_encoder import resolve_device

    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    if query_vectors.ndim != 2:
        raise ValueError(f"expected a 2-D query matrix, got {query_vectors.ndim}-D")
    if len(query_vectors) == 0:
        return [], np.zeros((0, 0), dtype=np.float32)
    if query_vectors.shape[1] != store.dim:
        raise ValueError(
            f"query width {query_vectors.shape[1]} does not match the store's "
            f"dim {store.dim}"
        )

    n_stored = len(store)
    if n_stored == 0:
        return [[] for _ in query_vectors], np.zeros(
            (len(query_vectors), 0), dtype=np.float32
        )

    resolved = resolve_device(device)
    keys = store.keys()
    stored = store.vectors()
    k = min(k, n_stored)

    all_keys: list[list[str]] = []
    all_scores: list[np.ndarray] = []

    for q_start in range(0, len(query_vectors), query_chunk):
        q_block = query_vectors[q_start : q_start + query_chunk]
        # Running top-k for this query block, carried across store chunks as
        # (scores, absolute row indices) so a key can never be admitted twice.
        best_scores = np.full((len(q_block), 0), -np.inf, dtype=np.float32)
        best_rows = np.zeros((len(q_block), 0), dtype=np.int64)

        for s_start in range(0, n_stored, store_chunk):
            block = np.asarray(
                stored[s_start : s_start + store_chunk], dtype=np.float32
            )
            scores = _matmul(q_block, block, resolved)
            rows = np.arange(s_start, s_start + len(block), dtype=np.int64)
            rows = np.broadcast_to(rows, scores.shape)

            merged_scores = np.concatenate([best_scores, scores], axis=1)
            merged_rows = np.concatenate([best_rows, rows], axis=1)

            take = min(k, merged_scores.shape[1])
            top = np.argpartition(-merged_scores, take - 1, axis=1)[:, :take]
            best_scores = np.take_along_axis(merged_scores, top, axis=1)
            best_rows = np.take_along_axis(merged_rows, top, axis=1)

        order = np.argsort(-best_scores, axis=1, kind="stable")
        best_scores = np.take_along_axis(best_scores, order, axis=1)
        best_rows = np.take_along_axis(best_rows, order, axis=1)

        for row_ids, row_scores in zip(best_rows, best_scores):
            all_keys.append([keys[int(r)] for r in row_ids])
            all_scores.append(row_scores)

    return all_keys, np.vstack(all_scores)
