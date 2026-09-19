import json

import numpy as np
import pandas as pd
import pytest

from src.coverage import (
    MAX_DENSE_GAIN_SHIFT,
    MIN_JOIN_COVERAGE,
    MissingnessError,
    check_missingness,
    join_coverage,
    missingness_bias,
)


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asin": ["B1", "B2", "B3", "B4"],
            "image_url": ["http://i/1.jpg", None, "http://i/3.jpg", "http://i/4.jpg"],
        }
    )


def test_coverage_is_the_matched_share_of_the_requested_products():
    result = join_coverage({"B1", "B2", "B9"}, _corpus(), scope="rerank")
    assert result.n_products == 3
    assert result.n_matched == 2
    assert result.coverage == pytest.approx(2 / 3)


def test_corpus_rows_outside_the_requested_products_do_not_inflate_coverage():
    # Review Focus 5: the corpus holds every us product, far more than the
    # judged set. Dividing by the corpus size instead of the product set would
    # report a number that is not about this project at all.
    result = join_coverage({"B1"}, _corpus(), scope="rerank")
    assert result.coverage == pytest.approx(1.0)


def test_image_coverage_counts_only_matched_products_with_a_url():
    result = join_coverage({"B1", "B2", "B9"}, _corpus(), scope="rerank")
    assert result.n_with_image == 1
    assert result.image_coverage == pytest.approx(1 / 3)


def test_image_coverage_is_reported_against_the_product_set_not_the_matches():
    # PROJECT_SPEC.md warns that the headline 91.5% is not image coverage:
    # end to end it is ~75%. Dividing by matches rather than by the product
    # set would reproduce exactly that overstatement.
    result = join_coverage({"B1", "B2"}, _corpus(), scope="rerank")
    assert result.image_coverage == pytest.approx(0.5)


def test_scope_is_carried_into_the_result():
    assert join_coverage({"B1"}, _corpus(), scope="catalogue").scope == "catalogue"


def test_empty_product_set_raises():
    with pytest.raises(ValueError, match="no products"):
        join_coverage(set(), _corpus(), scope="rerank")


def test_result_serialises_for_the_committed_record():
    payload = join_coverage({"B1", "B2"}, _corpus(), scope="rerank").to_dict()
    assert json.loads(json.dumps(payload))["scope"] == "rerank"


def test_the_gate_threshold_is_ninety_percent():
    assert MIN_JOIN_COVERAGE == 0.90


@pytest.mark.data
def test_real_join_coverage_clears_the_gate():
    from pathlib import Path

    payload = json.loads(Path("docs/results/esci-s-coverage.json").read_text())
    rerank = next(r for r in payload["scopes"] if r["scope"] == "rerank")
    assert rerank["n_products"] == 482_105
    assert rerank["coverage"] >= MIN_JOIN_COVERAGE


def _judgements(n_queries: int = 60) -> pd.DataFrame:
    rows = []
    for q in range(n_queries):
        for j in range(4):
            rows.append(
                {"query_id": q, "product_id": f"P{q:03d}{j}", "gain": [1.0, 0.1, 0.01, 0.0][j]}
            )
    return pd.DataFrame(rows)


def _label_independent_mask(n_queries: int = 60) -> np.ndarray:
    """A missingness pattern genuinely independent of the gain.

    `np.arange(len(judgements)) % 2 == 0` looks independent and is not:
    judgements come four to a query in gain order, so index parity is position
    parity, which keeps gains {1.0, 0.01} and drops {0.1, 0.0} - a delta of
    0.455. Alternating the offset per query puts every gain level on both
    sides in equal numbers, so the true delta is exactly zero.
    """
    return np.array(
        [(q + j) % 2 == 0 for q in range(n_queries) for j in range(4)]
    )


def _enrichment(judgements: pd.DataFrame, present: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asin": judgements["product_id"],
            "stars": np.where(present, 4.5, np.nan),
            "ratings": np.where(present, 10, np.nan),
            "category": [["Books"] if p else [] for p in present],
            "template": np.where(present, "book", None),
            "price": np.where(present, 9.99, np.nan),
            "bsr_rank": np.where(present, 12, np.nan),
            "attrs_json": np.where(present, '{"Brand": "X"}', None),
            "info_json": np.where(present, "{}", None),
            "image_url": np.where(present, "http://i/1.jpg", None),
        }
    )


def test_missingness_uncorrelated_with_the_label_reports_near_zero():
    judgements = _judgements()
    present = _label_independent_mask()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    stars = next(b for b in biases if b.field == "stars")
    assert abs(stars.delta.point) < MAX_DENSE_GAIN_SHIFT
    check_missingness(biases)


def test_missingness_aligned_with_the_label_is_caught():
    # Every Exact judgement enriched, every Irrelevant one not: the confound
    # the gate exists to catch.
    judgements = _judgements()
    present = (judgements["gain"] > 0.5).to_numpy()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    with pytest.raises(MissingnessError, match="stars"):
        check_missingness(biases)


def test_dense_and_sparse_fields_are_labelled():
    judgements = _judgements()
    present = _label_independent_mask()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    by_field = {b.field: b.dense for b in biases}
    assert by_field["stars"] is True
    assert by_field["price"] is False


def test_a_sparse_field_confounded_with_the_label_is_reported_not_raised():
    # PROJECT_SPEC.md expects sparse fields to correlate; they are carried
    # with missingness indicators rather than rejected.
    judgements = _judgements()
    present = (judgements["gain"] > 0.5).to_numpy()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    price = next(b for b in biases if b.field == "price")
    assert abs(price.delta.point) > MAX_DENSE_GAIN_SHIFT
    assert price.dense is False


def test_a_judged_product_absent_from_the_corpus_counts_as_missing():
    judgements = _judgements()
    present = np.ones(len(judgements), dtype=bool)
    enrichment = _enrichment(judgements, present).iloc[:100]
    biases = missingness_bias(judgements, enrichment, n_resamples=100)
    stars = next(b for b in biases if b.field == "stars")
    assert stars.present_share == pytest.approx(100 / len(judgements))


@pytest.mark.data
def test_real_dense_fields_are_not_confounded_with_the_label():
    from pathlib import Path

    payload = json.loads(Path("docs/results/esci-s-missingness.json").read_text())
    dense = [f for f in payload["fields"] if f["dense"]]
    # `category` is deliberately absent: the real report measured it at
    # -0.0210 and Task 5 Step 8 reclassified it as sparse.
    assert {f["field"] for f in dense} == {"stars", "ratings", "template"}
    for f in dense:
        assert abs(f["delta"]["point"]) < MAX_DENSE_GAIN_SHIFT, f
