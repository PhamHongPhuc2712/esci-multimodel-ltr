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
    from src.dense_embed import DEFAULT_STORE

    if not (DEFAULT_STORE / "keys.txt").exists():
        pytest.skip(f"no store at {DEFAULT_STORE}; run python -m src.dense_embed")
    store = open_store(DEFAULT_STORE, dim=DENSE_DIM)
    assert len(store) == 1_215_854
    sample = np.asarray(store.vectors()[:256], dtype=np.float32)
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)
