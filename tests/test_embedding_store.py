import json
from pathlib import Path

import numpy as np
import pytest

from src.clip_encoder import EMBEDDING_DIM
from src.embedding_store import (
    KEYS_NAME,
    META_NAME,
    STORE_DTYPE,
    VECTORS_NAME,
    open_store,
)


def _vectors(n: int, dim: int = 4, start: float = 0.0) -> np.ndarray:
    return (np.arange(n * dim, dtype=np.float32).reshape(n, dim) + start) / 100.0


# --- the happy path ---------------------------------------------------------

def test_a_new_store_is_empty(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    assert len(store) == 0
    assert store.known_keys() == set()


def test_append_then_read_back(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a", "b"], _vectors(2))
    assert len(store) == 2
    assert store.keys() == ["a", "b"]
    assert np.allclose(store.vectors(), _vectors(2), atol=1e-3)


def test_vectors_are_stored_as_float16(tmp_path):
    # float32 would put the catalogue store at 1.8 GB, past the spec's <1 GB
    # budget. Measured: float16 leaves top-1 retrieval identical at 75.333%
    # with a max cosine drift of 1.22e-04.
    assert STORE_DTYPE == np.float16
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    assert store.vectors().dtype == np.float16


def test_appending_twice_accumulates(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    store.append(["b", "c"], _vectors(2, start=100.0))
    assert len(store) == 3
    assert store.keys() == ["a", "b", "c"]


def test_reopening_sees_everything_written_before(tmp_path):
    # This is resume: the run is killed, and the next one must not refetch.
    directory = tmp_path / "s"
    open_store(directory, dim=4).append(["a", "b"], _vectors(2))
    reopened = open_store(directory, dim=4)
    assert len(reopened) == 2
    assert reopened.known_keys() == {"a", "b"}


# --- Review Focus 3: an interrupted run must not corrupt the store ----------

def test_recovery_truncates_vectors_written_without_their_key(tmp_path):
    # Killed after the vector write, before the key write.
    directory = tmp_path / "s"
    store = open_store(directory, dim=4)
    store.append(["a", "b"], _vectors(2))
    with (directory / "vectors.f16").open("ab") as handle:
        handle.write(np.zeros(4, dtype=np.float16).tobytes())

    recovered = open_store(directory, dim=4)
    assert len(recovered) == 2
    assert recovered.keys() == ["a", "b"]
    assert (directory / "vectors.f16").stat().st_size == 2 * 4 * 2


def test_recovery_drops_keys_written_without_their_vector(tmp_path):
    # Killed after the key write, before the vector write.
    directory = tmp_path / "s"
    store = open_store(directory, dim=4)
    store.append(["a", "b"], _vectors(2))
    with (directory / "keys.txt").open("a", encoding="utf-8") as handle:
        handle.write("c\n")

    recovered = open_store(directory, dim=4)
    assert len(recovered) == 2
    assert recovered.known_keys() == {"a", "b"}


def test_recovery_survives_a_torn_final_vector(tmp_path):
    # Killed mid-write: the file ends part way through a row.
    directory = tmp_path / "s"
    store = open_store(directory, dim=4)
    store.append(["a", "b"], _vectors(2))
    with (directory / "vectors.f16").open("ab") as handle:
        handle.write(b"\x00\x00\x00")  # 3 bytes: not a whole 4-wide f16 row

    recovered = open_store(directory, dim=4)
    assert len(recovered) == 2
    assert np.allclose(recovered.vectors(), _vectors(2), atol=1e-3)


def test_appending_a_key_already_stored_raises(tmp_path):
    # Silent double-writes would grow the store without bound on every resume.
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    with pytest.raises(ValueError, match="already"):
        store.append(["a"], _vectors(1))


def test_appending_a_duplicate_within_one_batch_raises(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="duplicate"):
        store.append(["a", "a"], _vectors(2))


def test_appending_the_wrong_number_of_vectors_raises(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="rows"):
        store.append(["a", "b"], _vectors(3))


def test_appending_the_wrong_width_raises(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="width"):
        store.append(["a"], np.zeros((1, 9), dtype=np.float32))


def test_reopening_with_a_different_dim_raises(tmp_path):
    directory = tmp_path / "s"
    open_store(directory, dim=4).append(["a"], _vectors(1))
    with pytest.raises(ValueError, match="dim"):
        open_store(directory, dim=8)


def test_a_key_containing_a_newline_raises(tmp_path):
    # Line i is row i. A key with a newline would silently shift every row
    # after it by one on the next open.
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="newline"):
        store.append(["a\nb"], _vectors(1))


# --- Review Focus 2: absence is not a zero vector ---------------------------

def test_lookup_returns_nan_and_false_for_an_unknown_key(tmp_path):
    # 22.47% of re-ranking products have no image. A zero row would score
    # cosine 0 against every query - a score, not an absence - and would rank
    # those products as mildly irrelevant rather than unknown.
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    matrix, present = store.lookup(["a", "missing"])
    assert matrix.shape == (2, 4)
    assert present.tolist() == [True, False]
    assert not np.isnan(matrix[0]).any()
    assert np.isnan(matrix[1]).all()


def test_lookup_returns_float32_for_arithmetic(tmp_path):
    # Stored at float16; handed out at float32 so downstream dot products do
    # not silently accumulate half-precision error.
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    matrix, _ = store.lookup(["a"])
    assert matrix.dtype == np.float32


def test_lookup_of_nothing_returns_an_empty_matrix(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    matrix, present = store.lookup([])
    assert matrix.shape == (0, 4)
    assert present.shape == (0,)


def test_lookup_preserves_the_requested_order(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a", "b"], _vectors(2))
    matrix, _ = store.lookup(["b", "a"])
    assert np.allclose(matrix[0], store.vectors()[1].astype(np.float32), atol=1e-3)


def test_the_default_dim_is_clips(tmp_path):
    assert open_store(tmp_path / "s").dim == EMBEDDING_DIM


# --- the real run -----------------------------------------------------------

REAL_STORE = Path("data/embeddings/rerank")


@pytest.mark.data
def test_the_real_store_agrees_with_itself_and_round_trips():
    """The plan gate: one float16 vector per fetched image, index in step.

    Checked on the raw files first, deliberately. `open_store` *repairs* a
    half-finished write by truncating to min(rows, keys), so opening the store
    is what hides the very disagreement this asserts - and because the repair
    writes, a store must have exactly one writer: do not run this while
    `python -m src.embed_images` is going.
    """
    if not (REAL_STORE / VECTORS_NAME).exists():
        pytest.skip(f"no store at {REAL_STORE}; run python -m src.embed_images")

    dim = json.loads((REAL_STORE / META_NAME).read_text(encoding="utf-8"))["dim"]
    row_bytes = dim * np.dtype(STORE_DTYPE).itemsize
    size = (REAL_STORE / VECTORS_NAME).stat().st_size
    n_keys = len(
        (REAL_STORE / KEYS_NAME).read_text(encoding="utf-8").splitlines()
    )
    assert size % row_bytes == 0, "the vector file ends mid-row"
    assert size // row_bytes == n_keys, "vector rows and keys disagree"

    store = open_store(REAL_STORE, dim=dim)
    assert len(store) == n_keys == len(store.keys())
    assert len(store.known_keys()) == n_keys, "duplicate keys in the index"

    # Vectors survived the float16 round trip as unit vectors. 1e-3 is
    # float16's resolution near 1.0, not a fudge factor.
    sample = np.asarray(store.vectors()[:256], dtype=np.float32)
    assert np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-3)

    # A stored key returns its own row; an absent one is NaN, never zeros.
    key = store.keys()[0]
    matrix, present = store.lookup([key, "https://example.invalid/none.jpg"])
    assert present.tolist() == [True, False]
    assert np.allclose(matrix[0], sample[0], atol=1e-3)
    assert np.isnan(matrix[1]).all()
