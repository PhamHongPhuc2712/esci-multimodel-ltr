import json

import pandas as pd
import pytest

from src.coverage import MIN_JOIN_COVERAGE, join_coverage


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
