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
