import numpy as np
import pandas as pd
import pytest

from src.image_report import (
    GATE_POOL,
    MIN_TOP1,
    image_coverage,
    top1_accuracy,
)


class _Store:
    def __init__(self, keys):
        self._keys = set(keys)

    def known_keys(self):
        return set(self._keys)


def test_perfectly_aligned_vectors_score_one():
    vectors = np.eye(5, dtype=np.float32)
    assert top1_accuracy(vectors, vectors) == pytest.approx(1.0)


def test_shuffled_vectors_score_near_chance():
    # This is the failure the gate exists for: embeddings and ids drifting out
    # of alignment. It does not raise anywhere; it just quietly stops meaning
    # anything.
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    shuffled = vectors[rng.permutation(200)]
    assert top1_accuracy(vectors, shuffled) < 0.10


def test_top1_counts_a_tie_as_a_miss():
    # Two identical candidate texts must not be scored as a hit by argmax
    # happening to land on the right index.
    images = np.array([[1.0, 0.0]], dtype=np.float32)
    texts = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert top1_accuracy(images, texts) == 0.0


def test_the_gate_reflects_the_measured_baseline():
    # Measured 0.753 top-1 over a 600-title pool, chance 0.00167.
    assert MIN_TOP1 == 0.60
    assert GATE_POOL == 600


def test_coverage_divides_by_the_product_set_not_the_matches():
    # PROJECT_SPEC.md: report ~75% end to end, not ESCI-S's 91.5% headline.
    # Dividing by products that have a URL rather than by all products is
    # exactly how the headline overstates it.
    frame = pd.DataFrame(
        {"product_id": ["a", "b"], "url": ["u1", "u2"]}
    )
    record = image_coverage(frame, _Store({"u1"}), n_products=4)
    assert record.with_url == 2
    assert record.url_coverage == pytest.approx(0.5)
    assert record.embedded == 1
    assert record.embedding_coverage == pytest.approx(0.25)


def test_coverage_counts_a_shared_url_once_per_product():
    # Two products sharing one URL are both covered by one stored vector.
    frame = pd.DataFrame({"product_id": ["a", "b"], "url": ["u1", "u1"]})
    record = image_coverage(frame, _Store({"u1"}), n_products=2)
    assert record.embedded == 2
    assert record.embedding_coverage == pytest.approx(1.0)


def test_coverage_serialises(tmp_path):
    record = image_coverage(
        pd.DataFrame({"product_id": ["a"], "url": ["u1"]}),
        _Store({"u1"}),
        n_products=1,
    )
    assert record.to_dict()["embedding_coverage"] == pytest.approx(1.0)
