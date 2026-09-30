from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.clip_encoder import EMBEDDING_DIM
from src.embed_images import (
    DEFAULT_CHUNK,
    STORE_NAME,
    embed_urls,
    pending_urls,
    product_image_urls,
)
from src.embedding_store import open_store
from src.image_fetch import FetchResult


def open_store_in_memory():
    """A store in a throwaway directory, for tests that do not assert on disk."""
    import tempfile

    return open_store(tempfile.mkdtemp(), dim=EMBEDDING_DIM)


def _image() -> Image.Image:
    return Image.new("RGB", (300, 300), (255, 0, 0))


def _fake_fetch(failing: set[str] | None = None):
    failing = failing or set()
    seen: list[str] = []

    def fetch(urls, **kwargs):
        for url in urls:
            seen.append(url)
            if url in failing:
                yield FetchResult(url, None, 404, 1)
            else:
                yield FetchResult(url, _image(), 200, 1)

    return fetch, seen


def _fake_encode(dim: int = EMBEDDING_DIM):
    batches: list[int] = []

    def encode_images(images, batch_size=None):
        batches.append(len(images))
        return np.ones((len(images), dim), dtype=np.float32) / np.sqrt(dim)

    return encode_images, batches


def _products(tmp_path, rows) -> Path:
    path = tmp_path / "products.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# --- Review Focus 4: duplicate URLs -----------------------------------------

def test_a_url_shared_by_several_products_is_fetched_once():
    # 932,320 catalogue products share 887,041 URLs. Keying on the product
    # would waste 5% of a multi-hour fetch and store the same vector several
    # times under different ids.
    store = open_store_in_memory()
    fetch, seen = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls(["u1", "u1", "u2"], store, encode_images, fetch=fetch)
    assert sorted(seen) == ["u1", "u2"]
    assert len(store) == 2


def test_pending_urls_deduplicates_and_skips_what_is_stored():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls(["u1"], store, encode_images, fetch=fetch)
    frame = pd.DataFrame(
        {"product_id": ["a", "b", "c"], "url": ["u1", "u2", "u2"]}
    )
    assert pending_urls(frame, store) == ["u2"]


# --- the pipeline -----------------------------------------------------------

def test_every_successful_url_lands_in_the_store():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch()
    encode_images, _ = _fake_encode()
    stats = embed_urls(["u1", "u2", "u3"], store, encode_images, fetch=fetch)
    assert stats.embedded == 3
    assert store.known_keys() == {"u1", "u2", "u3"}


def test_failures_are_counted_and_not_stored():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch(failing={"u2"})
    encode_images, _ = _fake_encode()
    stats = embed_urls(["u1", "u2", "u3"], store, encode_images, fetch=fetch)
    assert stats.embedded == 2
    assert stats.failed == 1
    assert "u2" not in store.known_keys()


def test_a_chunk_that_fails_entirely_does_not_break_the_run():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch(failing={"u1", "u2"})
    encode_images, batches = _fake_encode()
    stats = embed_urls(["u1", "u2"], store, encode_images, fetch=fetch, chunk=2)
    assert stats.embedded == 0
    assert batches == []  # nothing to encode, and no empty batch sent
    assert len(store) == 0


def test_the_run_is_chunked_so_progress_is_written_as_it_goes():
    # The run takes hours and will be killed. Appending once at the end would
    # lose everything.
    store = open_store_in_memory()
    fetch, _ = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls([f"u{i}" for i in range(10)], store, encode_images, fetch=fetch, chunk=4)
    assert len(store) == 10


def test_resuming_skips_what_is_already_stored():
    store = open_store_in_memory()
    fetch, seen = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls(["u1", "u2"], store, encode_images, fetch=fetch)
    seen.clear()
    stats = embed_urls(["u1", "u2", "u3"], store, encode_images, fetch=fetch)
    assert seen == ["u3"]
    assert stats.skipped == 2
    assert len(store) == 3


def test_default_chunk_keeps_memory_bounded():
    assert DEFAULT_CHUNK == 512


def test_stats_serialise_for_the_committed_record():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch(failing={"u2"})
    encode_images, _ = _fake_encode()
    stats = embed_urls(["u1", "u2"], store, encode_images, fetch=fetch)
    payload = stats.to_dict()
    assert payload["embedded"] == 1
    assert payload["failed"] == 1
    assert payload["requested"] == 2


# --- reading the product table ---------------------------------------------

def test_product_image_urls_drops_products_with_no_url(tmp_path):
    path = _products(
        tmp_path,
        [
            {"product_id": "a", "s_image_url": "u1"},
            {"product_id": "b", "s_image_url": None},
            {"product_id": "c", "s_image_url": ""},
        ],
    )
    frame = product_image_urls("catalogue", products_path=path)
    assert list(frame["product_id"]) == ["a"]
    assert list(frame.columns) == ["product_id", "url"]


def test_product_image_urls_rejects_an_unknown_scope(tmp_path):
    path = _products(tmp_path, [{"product_id": "a", "s_image_url": "u1"}])
    with pytest.raises(ValueError, match="scope"):
        product_image_urls("everything", products_path=path)


def test_every_scope_reads_and_writes_the_catalogue_store():
    # The rerank store was a strict subset of the catalogue one (0 of its
    # 361,875 URLs missing) and was deleted on 2026-09-30. A scope that still
    # opened data/embeddings/rerank/ would get an empty store back from
    # open_store, and image_report would print 0% image coverage without error.
    assert STORE_NAME == "catalogue"
