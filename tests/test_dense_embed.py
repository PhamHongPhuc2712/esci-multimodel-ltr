import numpy as np
import pandas as pd
import pytest

from esci_rerank.features.dense_embed import (
    DEFAULT_FIELDS,
    DENSE_DIM,
    MAX_SEQ_TOKENS,
    embed_products,
)
from esci_rerank.features.embedding_store import open_store


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
    # esci_rerank.evaluation.baseline_sbert.product_text exists because str(None) would embed the
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
    from esci_rerank.features.dense_embed import DEFAULT_STORE

    if not (DEFAULT_STORE / "keys.txt").exists():
        pytest.skip(f"no store at {DEFAULT_STORE}; run python -m esci_rerank.features.dense_embed")
    store = open_store(DEFAULT_STORE, dim=DENSE_DIM)
    assert len(store) == 1_215_854
    sample = np.asarray(store.vectors()[:256], dtype=np.float32)
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)


# --- which encoder wrote a store ---------------------------------------------

def test_a_store_remembers_the_encoder_that_wrote_it(tmp_path):
    from esci_rerank.features.dense_embed import check_model_record, write_model_record

    write_model_record(tmp_path, "BAAI/bge-small-en-v1.5", max_seq=256)
    check_model_record(tmp_path, "BAAI/bge-small-en-v1.5")  # no raise


def test_querying_a_store_with_another_encoder_raises(tmp_path):
    # bge-small and all-MiniLM-L12-v2 are both 384-d, so the dimension check
    # passes and every query returns real-looking products from the wrong
    # space. Nothing else would catch it.
    from esci_rerank.features.dense_embed import check_model_record, write_model_record

    write_model_record(tmp_path, "BAAI/bge-small-en-v1.5", max_seq=256)
    with pytest.raises(ValueError, match="bge-small"):
        check_model_record(tmp_path, "sentence-transformers/all-MiniLM-L12-v2")


def test_the_landed_store_has_no_record_and_belongs_to_the_default_model(tmp_path):
    # data/embeddings/dense was written before stores recorded their encoder.
    from esci_rerank.features.dense_embed import DEFAULT_MODEL, check_model_record

    store = _store(tmp_path)
    encode_texts, _ = _fake_encode()
    embed_products(_products(2), store, encode_texts)
    check_model_record(store.directory, DEFAULT_MODEL)  # no raise
    with pytest.raises(ValueError, match="no record"):
        check_model_record(store.directory, "models/retriever/bge-base")


def test_an_empty_store_accepts_any_encoder(tmp_path):
    from esci_rerank.features.dense_embed import check_model_record

    check_model_record(tmp_path / "fresh", "models/retriever/bge-base")  # no raise


def test_the_record_carries_the_token_budget(tmp_path):
    import json

    from esci_rerank.features.dense_embed import MODEL_RECORD, write_model_record

    write_model_record(tmp_path, "m", max_seq=256)
    assert json.loads((tmp_path / MODEL_RECORD).read_text()) == {"model": "m", "max_seq": 256}


# --- Plan 12: the record checks the token budget too --------------------------

def test_a_different_token_budget_is_refused_naming_both(tmp_path):
    from esci_rerank.features.dense_embed import check_model_record, write_model_record

    write_model_record(tmp_path, "BAAI/bge-base-en-v1.5", max_seq=256)
    with pytest.raises(ValueError, match="max_seq") as excinfo:
        check_model_record(tmp_path, "BAAI/bge-base-en-v1.5", max_seq=128)
    assert "256" in str(excinfo.value) and "128" in str(excinfo.value)


def test_the_same_model_and_budget_pass(tmp_path):
    from esci_rerank.features.dense_embed import check_model_record, write_model_record

    write_model_record(tmp_path, "BAAI/bge-base-en-v1.5", max_seq=256)
    check_model_record(tmp_path, "BAAI/bge-base-en-v1.5", max_seq=256)  # no raise


def test_a_reader_that_does_not_say_its_budget_is_checked_on_the_model_only(tmp_path):
    # Readers (open_channel) compare the model; only a writer has a budget to
    # disagree about.
    from esci_rerank.features.dense_embed import check_model_record, write_model_record

    write_model_record(tmp_path, "BAAI/bge-base-en-v1.5", max_seq=256)
    check_model_record(tmp_path, "BAAI/bge-base-en-v1.5")  # no raise


def test_a_refused_resume_leaves_the_record_as_it_was(tmp_path):
    from esci_rerank.features.dense_embed import MODEL_RECORD, check_model_record, write_model_record

    write_model_record(tmp_path, "m", max_seq=256)
    before = (tmp_path / MODEL_RECORD).read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        check_model_record(tmp_path, "m", max_seq=128)
    assert (tmp_path / MODEL_RECORD).read_text(encoding="utf-8") == before
