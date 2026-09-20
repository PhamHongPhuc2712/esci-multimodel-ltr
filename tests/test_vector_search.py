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
