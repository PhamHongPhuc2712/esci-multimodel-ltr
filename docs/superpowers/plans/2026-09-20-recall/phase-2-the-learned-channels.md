# Phase 2 — The Learned Channels

**Plan 4 of 7 · Phase 2 of 3 · Tasks 3–4.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 1 gate](phase-1-the-metric-and-the-lexical-channel.md#phase-1-gate) passes.

**Delivers:** `src/vector_search.py`, `src/dense_embed.py` and
`src/image_channel.py` — top-k cosine search over any `EmbeddingStore`, the
SBERT corpus embedding run, and the CLIP image channel built on the store
Plan 3 already filled.

**Needs on disk:** `data/combined/products.parquet`, and Plan 3's
`data/embeddings/rerank/`. Task 3 writes 0.93 GB in 25 minutes on the GPU;
Task 4 extends the image store by 525,166 URLs in about 3.6 hours.

**Owns Review Focus item 3** (truncation deciding the result).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **SBERT reads 128 tokens** of a ~440-token string, so field order decides what is embedded at all.
- **The image store is keyed by image URL, not `product_id`.** 11,771 re-ranking products share another product's image; a URL maps to one *or more* products.
- **The image channel covers 77.50% of products**, and that gap is a 2022 scrape artefact, not a product property. Absence must stay representable.
- **Never materialise the full score matrix.** 8,956 queries × 1,215,854 products is 1.09e10 scores — 43 GB at float32.
- **Run GPU-capable work on the GPU**, and fail loudly if `cuda` is asked for and missing.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 3: Vector search and the dense-text channel

**Files:**
- Create: `src/vector_search.py`
- Create: `src/dense_embed.py`
- Create: `tests/test_vector_search.py`
- Create: `tests/test_dense_embed.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `src.embedding_store.open_store`, `src.embedding_store.EmbeddingStore`, `src.clip_encoder.resolve_device`, `src.clip_encoder.l2_normalise`, `src.baseline_sbert.product_text`, `src.baseline_sbert.DEFAULT_MODEL`, `src.recall.Channel`.
- Produces:
  - `src.vector_search.DEFAULT_QUERY_CHUNK: int` (256), `DEFAULT_STORE_CHUNK: int` (200_000)
  - `src.vector_search.top_k(query_vectors, store, k, *, device="auto", query_chunk=..., store_chunk=...) -> tuple[list[list[str]], np.ndarray]`
  - `src.dense_embed.DEFAULT_STORE: Path` (`data/embeddings/dense`), `DENSE_DIM: int` (384), `DEFAULT_FIELDS: tuple[str, ...]`, `MAX_SEQ_TOKENS: int` (128)
  - `src.dense_embed.embed_products(products, store, encode_texts, *, fields=..., chunk=8192) -> int`
  - `src.dense_embed.DenseChannel` with `.search(queries, k) -> list[list[str]]`
  - `src.dense_embed.open_channel(store_dir=DEFAULT_STORE, model=..., device="auto") -> DenseChannel`
  - `python -m src.dense_embed`

`top_k` never materialises the full score matrix. It walks the store in
`store_chunk` row blocks and the queries in `query_chunk` blocks, keeping a
running top-k per query, so peak memory is `query_chunk × store_chunk × 4`
bytes — 205 MB at the defaults — regardless of corpus size.

The dense store is keyed by **`product_id`**, unlike Plan 3's image store
which is keyed by URL. `EmbeddingStore` takes any string key, which is why one
store class serves both.

- [x] **Step 1: Write the failing test for vector search**

Create `tests/test_vector_search.py`:

```python
import numpy as np
import pytest

from src.embedding_store import open_store
from src.vector_search import DEFAULT_STORE_CHUNK, top_k


def _store(tmp_path, keys, vectors):
    store = open_store(tmp_path / "s", dim=vectors.shape[1])
    store.append(keys, vectors)
    return store


def _unit(rows):
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


# --- the happy path ---------------------------------------------------------

def test_the_nearest_vector_comes_first(tmp_path):
    store = _store(tmp_path, ["a", "b", "c"], _unit([[1, 0], [0, 1], [-1, 0]]))
    keys, scores = top_k(_unit([[1, 0]]), store, k=3)
    assert keys[0][0] == "a"
    assert scores[0][0] == pytest.approx(1.0, abs=1e-3)


def test_results_are_ordered_by_descending_similarity(tmp_path):
    store = _store(tmp_path, ["a", "b", "c"], _unit([[1, 0], [0.9, 0.1], [0, 1]]))
    keys, scores = top_k(_unit([[1, 0]]), store, k=3)
    assert keys[0] == ["a", "b", "c"]
    assert list(scores[0]) == sorted(scores[0], reverse=True)


def test_one_result_list_per_query(tmp_path):
    store = _store(tmp_path, ["a", "b"], _unit([[1, 0], [0, 1]]))
    keys, _ = top_k(_unit([[1, 0], [0, 1]]), store, k=1)
    assert keys == [["a"], ["b"]]


def test_k_larger_than_the_store_returns_everything_once(tmp_path):
    store = _store(tmp_path, ["a", "b"], _unit([[1, 0], [0, 1]]))
    keys, _ = top_k(_unit([[1, 0]]), store, k=50)
    assert sorted(keys[0]) == ["a", "b"]


def test_searching_an_empty_store_returns_empty_lists(tmp_path):
    store = open_store(tmp_path / "s", dim=2)
    keys, scores = top_k(_unit([[1, 0]]), store, k=5)
    assert keys == [[]]
    assert scores.shape == (1, 0)


def test_searching_with_no_queries_returns_nothing(tmp_path):
    store = _store(tmp_path, ["a"], _unit([[1, 0]]))
    keys, _ = top_k(np.zeros((0, 2), dtype=np.float32), store, k=5)
    assert keys == []


# --- chunking must not change the answer ------------------------------------

def test_chunking_the_store_gives_the_same_answer_as_not(tmp_path):
    # 8,956 queries x 1,215,854 products is 1.09e10 scores - 43 GB at
    # float32 - so the full matrix is never materialised. A running top-k
    # across chunks must still be a global top-k.
    rng = np.random.default_rng(0)
    vectors = _unit(rng.normal(size=(500, 8)))
    keys = [f"p{i}" for i in range(500)]
    store = _store(tmp_path, keys, vectors)
    queries = _unit(rng.normal(size=(7, 8)))

    whole, _ = top_k(queries, store, k=10, store_chunk=10_000)
    chunked, _ = top_k(queries, store, k=10, store_chunk=37)
    assert whole == chunked


def test_chunking_the_queries_gives_the_same_answer_as_not(tmp_path):
    rng = np.random.default_rng(1)
    vectors = _unit(rng.normal(size=(200, 8)))
    store = _store(tmp_path, [f"p{i}" for i in range(200)], vectors)
    queries = _unit(rng.normal(size=(9, 8)))

    whole, _ = top_k(queries, store, k=5, query_chunk=100)
    chunked, _ = top_k(queries, store, k=5, query_chunk=2)
    assert whole == chunked


def test_a_key_is_never_returned_twice(tmp_path):
    # A running top-k that merges chunks carelessly can re-admit a key it
    # already holds, silently inflating recall.
    rng = np.random.default_rng(2)
    vectors = _unit(rng.normal(size=(300, 8)))
    store = _store(tmp_path, [f"p{i}" for i in range(300)], vectors)
    keys, _ = top_k(_unit(rng.normal(size=(3, 8))), store, k=20, store_chunk=11)
    for row in keys:
        assert len(row) == len(set(row)) == 20


def test_the_default_store_chunk_bounds_memory():
    # query_chunk x store_chunk x 4 bytes = 205 MB at the defaults.
    assert DEFAULT_STORE_CHUNK == 200_000


def test_a_query_of_the_wrong_width_raises(tmp_path):
    store = _store(tmp_path, ["a"], _unit([[1, 0]]))
    with pytest.raises(ValueError, match="width"):
        top_k(np.ones((1, 5), dtype=np.float32), store, k=1)
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_vector_search.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.vector_search'`.

- [x] **Step 3: Write `src/vector_search.py`**

```python
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

from typing import Any

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
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_vector_search.py -v
```

Expected: PASS, 11 tests.

- [x] **Step 5: Write the failing test for the dense channel**

Create `tests/test_dense_embed.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.dense_embed import (
    DEFAULT_FIELDS,
    DENSE_DIM,
    MAX_SEQ_TOKENS,
    embed_products,
)
from src.embedding_store import open_store


def _products(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(n)],
        "product_title": [f"title {i}" for i in range(n)],
        "description": [f"description {i}" for i in range(n)],
        "product_bullet_point": [f"bullet {i}" for i in range(n)],
    })


def _fake_encode(dim: int = DENSE_DIM):
    seen: list[str] = []

    def encode_texts(texts, batch_size=None):
        seen.extend(texts)
        out = np.ones((len(texts), dim), dtype=np.float32)
        return out / np.sqrt(dim)

    return encode_texts, seen


def _store(tmp_path):
    return open_store(tmp_path / "dense", dim=DENSE_DIM)


# --- the happy path ---------------------------------------------------------

def test_every_product_lands_in_the_store_keyed_by_product_id(tmp_path):
    store = _store(tmp_path)
    encode_texts, _ = _fake_encode()
    assert embed_products(_products(3), store, encode_texts) == 3
    assert store.known_keys() == {"P0", "P1", "P2"}


def test_resuming_skips_what_is_already_embedded(tmp_path):
    store = _store(tmp_path)
    encode_texts, seen = _fake_encode()
    embed_products(_products(2), store, encode_texts)
    seen.clear()
    assert embed_products(_products(3), store, encode_texts) == 1
    assert len(seen) == 1
    assert len(store) == 3


def test_the_run_is_chunked_so_progress_is_written_as_it_goes(tmp_path):
    store = _store(tmp_path)
    encode_texts, _ = _fake_encode()
    embed_products(_products(10), store, encode_texts, chunk=3)
    assert len(store) == 10


def test_embedding_nothing_is_not_an_error(tmp_path):
    store = _store(tmp_path)
    encode_texts, seen = _fake_encode()
    assert embed_products(_products(0), store, encode_texts) == 0
    assert seen == []


def test_the_dense_dimension_is_the_models(tmp_path):
    # all-MiniLM-L12-v2 is 384-d, so 1,215,854 products cost 0.93 GB at fp16.
    assert DENSE_DIM == 384


# --- Review Focus 3: truncation deciding the result -------------------------

def test_the_title_is_embedded_first(tmp_path):
    # all-MiniLM-L12-v2 reads 128 tokens; the concatenated product text
    # averages 1,748 characters, roughly 440 tokens. About 70% never reaches
    # the model, so whichever field leads is what gets embedded at all.
    assert DEFAULT_FIELDS[0] == "product_title"


def test_the_token_limit_is_recorded_next_to_the_field_order():
    # The number that makes the field order matter, pinned so a model swap
    # that changes it cannot pass silently.
    assert MAX_SEQ_TOKENS == 128


def test_the_text_handed_to_the_encoder_leads_with_the_title(tmp_path):
    store = _store(tmp_path)
    encode_texts, seen = _fake_encode()
    embed_products(_products(1), store, encode_texts)
    assert seen[0].startswith("title 0")


def test_a_null_field_is_dropped_rather_than_stringified(tmp_path):
    # src.baseline_sbert.product_text exists because str(None) would embed the
    # literal "None" as product text.
    store = _store(tmp_path)
    encode_texts, seen = _fake_encode()
    frame = _products(1)
    frame.loc[0, "description"] = None
    embed_products(frame, store, encode_texts)
    assert "None" not in seen[0]


def test_a_missing_field_raises(tmp_path):
    store = _store(tmp_path)
    encode_texts, _ = _fake_encode()
    with pytest.raises(KeyError, match="no such product field"):
        embed_products(_products(1), store, encode_texts, fields=("nope",))


# --- the real store ---------------------------------------------------------

@pytest.mark.data
def test_the_real_dense_store_covers_the_corpus():
    from pathlib import Path

    from src.dense_embed import DEFAULT_STORE

    if not (DEFAULT_STORE / "keys.txt").exists():
        pytest.skip(f"no store at {DEFAULT_STORE}; run python -m src.dense_embed")
    store = open_store(DEFAULT_STORE, dim=DENSE_DIM)
    assert len(store) == 1_215_854
    sample = np.asarray(store.vectors()[:256], dtype=np.float32)
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)
```

- [x] **Step 6: Run the test to verify it fails**

```bash
python -m pytest tests/test_dense_embed.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.dense_embed'`.

- [x] **Step 7: Write `src/dense_embed.py`**

```python
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
```

- [x] **Step 8: Run the fast tests, then the real embed**

```bash
python -m pytest tests/test_dense_embed.py -v
python -m src.dense_embed --limit 2000     # smoke test, under a minute
python -m src.dense_embed                  # the real run, ~25 min on the GPU
python -m pytest tests/test_dense_embed.py -m data -v
```

Expected: 10 fast tests pass; the smoke run reports roughly 821 docs/s on
`cuda`; the full run ends with `store now holds 1,215,854 vectors` and
`data/embeddings/dense/` is about 0.93 GB; the marked test passes.

**If it reports `cpu`**, stop — a CPU run is hours, not minutes. `--device cuda`
will fail loudly rather than fall back.

- [x] **Step 9: Record the command in `CLAUDE.md`**

Add under the retrieval-channels comment added in Phase 1:

````markdown
python -m src.dense_embed                     # -> data/embeddings/dense/, ~25 min GPU
````

- [x] **Step 10: Commit**

```bash
git add src/vector_search.py src/dense_embed.py tests/test_vector_search.py \
        tests/test_dense_embed.py CLAUDE.md
git commit -m "Add chunked vector search and the dense-text retrieval channel"
```

---

## Task 4: The image channel

**Files:**
- Create: `src/image_channel.py`
- Create: `tests/test_image_channel.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `src.vector_search.top_k`, `src.embedding_store.open_store`, `src.clip_encoder.load_encoder`, `src.clip_encoder.EMBEDDING_DIM`, `src.embed_images.DEFAULT_STORE_ROOT`, `src.recall.Channel`.
- Produces:
  - `src.image_channel.DEFAULT_SCOPE: str` (`catalogue`), `OVERSAMPLE: int` (2)
  - `src.image_channel.url_to_products(products_path=...) -> dict[str, list[str]]`
  - `src.image_channel.ImageChannel` with `.search(queries, k) -> list[list[str]]`, `.n_urls -> int`, `.coverage(n_products) -> float`
  - `src.image_channel.open_channel(scope=DEFAULT_SCOPE, store_root=..., products_path=..., device="auto") -> ImageChannel`

**The store is keyed by URL, not by product.** Plan 3 keyed the fetch on the
image URL because 11,771 re-ranking products (and 45,279 catalogue ones) share
another product's image. So a hit is a URL, and expanding it yields one *or
more* `product_id`s. The channel therefore retrieves `OVERSAMPLE × k` URLs and
truncates the expanded product list to `k` — without the oversample, a query
whose top hits are all shared images returns fewer than `k` products and its
recall is understated for a reason that has nothing to do with retrieval.

- [x] **Step 1: Write the failing test**

Create `tests/test_image_channel.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.clip_encoder import EMBEDDING_DIM
from src.embedding_store import open_store
from src.image_channel import ImageChannel, OVERSAMPLE, url_to_products
from src.recall import Channel


def _unit(rows):
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def _wide(rows):
    """Pad 2-d toy vectors out to CLIP's width."""
    array = np.zeros((len(rows), EMBEDDING_DIM), dtype=np.float32)
    array[:, :2] = rows
    return _unit(array)


class _Encoder:
    """Encodes a query string to a fixed toy vector."""

    device = "cpu"

    def __init__(self, mapping):
        self.mapping = mapping

    def encode_texts(self, texts, batch_size=None):
        return _wide([self.mapping[t] for t in texts])


def _channel(tmp_path, url_vectors, url_products, mapping):
    store = open_store(tmp_path / "img", dim=EMBEDDING_DIM)
    store.append(list(url_vectors), _wide(list(url_vectors.values())))
    return ImageChannel(
        store=store,
        encoder=_Encoder(mapping),
        url_products=url_products,
        device="cpu",
    )


# --- the URL -> product expansion -------------------------------------------

def test_url_to_products_groups_products_that_share_an_image(tmp_path):
    path = tmp_path / "products.parquet"
    pd.DataFrame({
        "product_id": ["A", "B", "C"],
        "s_image_url": ["u1", "u1", "u2"],
    }).to_parquet(path, index=False)
    mapping = url_to_products(path)
    assert sorted(mapping["u1"]) == ["A", "B"]
    assert mapping["u2"] == ["C"]


def test_url_to_products_ignores_products_with_no_url(tmp_path):
    path = tmp_path / "products.parquet"
    pd.DataFrame({
        "product_id": ["A", "B"],
        "s_image_url": ["u1", None],
    }).to_parquet(path, index=False)
    assert set(url_to_products(path)) == {"u1"}


# --- Review Focus: a URL is not a product -----------------------------------

def test_a_shared_image_yields_every_product_that_uses_it(tmp_path):
    # 11,771 re-ranking products share another product's image. Returning the
    # URL's first product only would silently drop the rest from recall.
    channel = _channel(
        tmp_path,
        {"u1": [1, 0]},
        {"u1": ["A", "B"]},
        {"shoes": [1, 0]},
    )
    assert sorted(channel.search(["shoes"], k=10)[0]) == ["A", "B"]


def test_search_returns_product_ids_not_urls(tmp_path):
    channel = _channel(
        tmp_path, {"u1": [1, 0]}, {"u1": ["A"]}, {"shoes": [1, 0]}
    )
    assert channel.search(["shoes"], k=5)[0] == ["A"]


def test_the_result_is_truncated_to_k_after_expansion(tmp_path):
    channel = _channel(
        tmp_path,
        {"u1": [1, 0]},
        {"u1": ["A", "B", "C"]},
        {"shoes": [1, 0]},
    )
    assert len(channel.search(["shoes"], k=2)[0]) == 2


def test_oversampling_keeps_k_products_reachable(tmp_path):
    # Without the oversample a query whose best URLs are shared returns fewer
    # than k products, and its recall is understated for a reason that has
    # nothing to do with retrieval.
    assert OVERSAMPLE >= 2


def test_a_url_in_the_store_but_not_in_the_product_table_is_dropped(tmp_path):
    # The store outlives the table it was built from; a stale URL must not
    # become a KeyError mid-run.
    channel = _channel(
        tmp_path,
        {"u1": [1, 0], "u2": [0, 1]},
        {"u1": ["A"]},
        {"shoes": [1, 0]},
    )
    assert channel.search(["shoes"], k=10)[0] == ["A"]


def test_a_product_is_never_returned_twice(tmp_path):
    # Two URLs can both expand to the same product.
    channel = _channel(
        tmp_path,
        {"u1": [1, 0], "u2": [0.9, 0.1]},
        {"u1": ["A"], "u2": ["A", "B"]},
        {"shoes": [1, 0]},
    )
    hits = channel.search(["shoes"], k=10)[0]
    assert hits == list(dict.fromkeys(hits))


# --- the protocol and coverage ---------------------------------------------

def test_the_channel_satisfies_the_protocol(tmp_path):
    channel = _channel(tmp_path, {"u1": [1, 0]}, {"u1": ["A"]}, {"q": [1, 0]})
    assert isinstance(channel, Channel)


def test_coverage_is_reported_against_the_product_set(tmp_path):
    # The image channel reaches 77.50% of products, and the gap is a 2022
    # scrape artefact rather than a property of the products.
    channel = _channel(
        tmp_path, {"u1": [1, 0]}, {"u1": ["A", "B"]}, {"q": [1, 0]}
    )
    assert channel.coverage(n_products=4) == pytest.approx(0.5)


def test_searching_with_no_queries_returns_nothing(tmp_path):
    channel = _channel(tmp_path, {"u1": [1, 0]}, {"u1": ["A"]}, {"q": [1, 0]})
    assert channel.search([], k=5) == []


# --- the real store ---------------------------------------------------------

@pytest.mark.data
def test_the_real_image_channel_reaches_the_measured_coverage():
    from pathlib import Path

    from src.image_channel import open_channel

    if not Path("data/embeddings/catalogue/keys.txt").exists():
        pytest.skip("no catalogue image store; run python -m src.embed_images "
                    "--scope catalogue")
    channel = open_channel(scope="catalogue")
    assert channel.coverage(n_products=1_215_854) > 0.75
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_image_channel.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.image_channel'`.

- [x] **Step 3: Write `src/image_channel.py`**

```python
"""The image retrieval channel: a text query against CLIP image embeddings.

CLIP puts images and text in one 512-d space, so the query side is
`clip_encoder.encode_texts` and the corpus side is the store Plan 3 filled -
no image is fetched at query time.

The store is keyed by **image URL**, not by product: Plan 3 keyed the fetch on
the URL because 45,279 catalogue products (11,771 in the re-ranking scope)
reuse another product's image. So a hit expands to one *or more* product ids,
and the channel oversamples URLs before truncating the expanded product list
to k. Without the oversample a query whose best URLs are shared returns fewer
than k products and its recall is understated for a reason unrelated to
retrieval.

Coverage is 77.50% of products, and that gap is a 2022 scrape artefact rather
than a property of the products. `CLAUDE.md`'s evaluation discipline requires
it to be reported separately, never quietly imputed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.clip_encoder import EMBEDDING_DIM
from src.embed_images import DEFAULT_PRODUCTS, DEFAULT_STORE_ROOT
from src.embedding_store import EmbeddingStore, open_store
from src.recall import Channel  # noqa: F401  (ImageChannel implements it)

DEFAULT_SCOPE = "catalogue"

# Retrieve OVERSAMPLE x k URLs before expanding to products, so shared images
# cannot starve a query of candidates.
OVERSAMPLE = 2


def url_to_products(products_path: Path = DEFAULT_PRODUCTS) -> dict[str, list[str]]:
    """Every product id that uses each image URL."""
    frame = pd.read_parquet(products_path, columns=["product_id", "s_image_url"])
    urls = frame["s_image_url"]
    frame = frame.loc[urls.notna() & (urls.astype("str").str.len() > 0)]
    grouped = frame.groupby("s_image_url")["product_id"].apply(list)
    return {str(url): [str(p) for p in ids] for url, ids in grouped.items()}


@dataclass(frozen=True)
class ImageChannel:
    store: EmbeddingStore
    encoder: Any
    url_products: dict[str, list[str]]
    device: str

    @property
    def n_urls(self) -> int:
        return len(self.store)

    def coverage(self, n_products: int) -> float:
        """Share of `n_products` reachable through a stored image."""
        if n_products <= 0:
            raise ValueError("n_products must be positive")
        reachable = {
            product
            for url in self.store.known_keys()
            for product in self.url_products.get(url, ())
        }
        return len(reachable) / n_products

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        from src.vector_search import top_k

        queries = list(queries)
        if not queries:
            return []

        vectors = self.encoder.encode_texts(queries)
        url_hits, _ = top_k(
            vectors, self.store, k * OVERSAMPLE, device=self.device
        )

        results: list[list[str]] = []
        for urls in url_hits:
            products: dict[str, None] = {}
            for url in urls:
                # A URL in the store but absent from the product table is
                # stale, not fatal: the store outlives the table it was built
                # from.
                for product in self.url_products.get(url, ()):
                    products.setdefault(product, None)
                    if len(products) >= k:
                        break
                if len(products) >= k:
                    break
            results.append(list(products))
        return results


def open_channel(
    scope: str = DEFAULT_SCOPE,
    store_root: Path = DEFAULT_STORE_ROOT,
    products_path: Path = DEFAULT_PRODUCTS,
    device: str = "auto",
) -> ImageChannel:
    """Open the CLIP image store for `scope` and load the text encoder."""
    from src.clip_encoder import load_encoder

    store = open_store(Path(store_root) / scope, dim=EMBEDDING_DIM)
    if len(store) == 0:
        raise FileNotFoundError(
            f"the image store at {Path(store_root) / scope} is empty; run "
            f"python -m src.embed_images --scope {scope}"
        )
    encoder = load_encoder(device=device)
    return ImageChannel(
        store=store,
        encoder=encoder,
        url_products=url_to_products(products_path),
        device=encoder.device,
    )
```

- [x] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_image_channel.py -v
```

Expected: PASS, 11 tests; the `data`-marked one is deselected.

- [ ] **Step 5: Extend the image store to the full catalogue**

Recall is measured over all 1,215,854 products, so the channel needs the
catalogue scope. Plan 3's store is append-only and keyed by URL, so this
re-fetches only the difference — **525,166 URLs, about 3.6 hours** at the
measured 2,463 img/min. It is resumable; restart it if throughput collapses.

```bash
python -m src.embed_images --scope catalogue
python -m src.image_report --scope catalogue
python -m pytest tests/test_image_channel.py -m data -v
```

Expected: the store reaches about 886,900 of 887,041 URLs (Plan 3 measured a
0.036% failure rate); `image_report` prints a catalogue embedding coverage
near **0.7668** and a passing semantic gate; the marked test passes.

**Do not open the store while that run is writing.** `open_store` repairs a
half-finished write by truncating, so a concurrent open can corrupt an
in-progress run. Wait for it to exit.

- [ ] **Step 6: Record the command in `CLAUDE.md`**

Add under the retrieval-channels comment:

````markdown
python -m src.embed_images --scope catalogue  # recall needs the full corpus, ~3.6 h
````

- [ ] **Step 7: Commit**

```bash
git add src/image_channel.py tests/test_image_channel.py CLAUDE.md
git commit -m "Add the CLIP image retrieval channel with URL-to-product expansion"
```

---

## Phase 2 Gate

Phase 3 does not start until all of these hold:

- [ ] `python -m pytest tests/test_vector_search.py tests/test_dense_embed.py tests/test_image_channel.py -v` passes — 11 + 10 + 11 tests.
- [ ] `python -m pytest` still passes end to end, with Plans 1–3 and Phase 1 untouched.
- [ ] `data/embeddings/dense/` holds 1,215,854 vectors, about 0.93 GB.
- [ ] `data/embeddings/catalogue/` holds roughly 886,900 vectors, about 0.91 GB, and `python -m src.image_report --scope catalogue` passes its semantic gate.
- [ ] Each channel answers a 100-query batch at k=1000 without exceeding 2 GB of RSS.
- [ ] A dense Recall@100 measured on the validation folds is reported beside BM25's. Either may win; a channel that ties is a reportable result.

Next: [Phase 3 — Fusion and the Ablations](phase-3-fusion-and-the-ablations.md).
