import math

import numpy as np
import pandas as pd
import pytest

from src.fixed_weight import (
    DEFAULT_WEIGHTS,
    apply_normaliser,
    best_weight,
    blend,
    fit_normaliser,
    sweep,
)


# --- normalisation ----------------------------------------------------------

def test_fit_normaliser_ignores_nan():
    # 20.6% of image similarities are NaN. Counting them as zeros would drag
    # the mean toward zero and mis-scale every real value.
    norm = fit_normaliser(np.array([1.0, 3.0, np.nan]))
    assert norm.mean == pytest.approx(2.0)
    assert norm.std == pytest.approx(1.0)


def test_apply_normaliser_centres_and_scales():
    norm = fit_normaliser(np.array([1.0, 3.0]))
    out = apply_normaliser(norm, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])


def test_apply_normaliser_keeps_nan_as_nan():
    norm = fit_normaliser(np.array([1.0, 3.0]))
    assert math.isnan(apply_normaliser(norm, np.array([np.nan]))[0])


def test_a_constant_column_does_not_divide_by_zero():
    norm = fit_normaliser(np.array([5.0, 5.0, 5.0]))
    assert np.isfinite(apply_normaliser(norm, np.array([5.0]))).all()


def test_fitting_on_nothing_raises():
    with pytest.raises(ValueError, match="no finite values"):
        fit_normaliser(np.array([np.nan, np.nan]))


# --- blending ---------------------------------------------------------------

def test_blend_is_the_weighted_sum():
    out = blend(np.array([1.0]), np.array([3.0]), 0.5)
    assert out[0] == pytest.approx(2.0)


def test_a_weight_of_one_is_text_only():
    np.testing.assert_allclose(blend(np.array([1.0, 2.0]), np.array([9.0, 9.0]), 1.0), [1.0, 2.0])


def test_a_weight_of_zero_is_image_only():
    np.testing.assert_allclose(blend(np.array([9.0, 9.0]), np.array([1.0, 2.0]), 0.0), [1.0, 2.0])


def test_a_missing_image_falls_back_to_text_alone():
    # Not to the mean: substituting the average would penalise a product for a
    # 2022 scrape failure, which is what CLAUDE.md forbids the learned arm from
    # doing. The control gets the same courtesy.
    out = blend(np.array([2.0, 2.0]), np.array([np.nan, 0.0]), 0.5)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(1.0)


def test_blend_never_returns_nan_when_the_text_score_is_present():
    out = blend(np.array([1.0, 2.0, 3.0]), np.full(3, np.nan), 0.3)
    assert np.isfinite(out).all()


def test_a_weight_outside_zero_to_one_raises():
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend(np.array([1.0]), np.array([1.0]), 1.5)


# --- the sweep --------------------------------------------------------------

def _frame_and_qrels():
    # Two queries of three products. The image signal alone gets query 1 right
    # and query 2 wrong; the text signal is the other way round. A blend beats
    # either.
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 2],
            "product_id": ["a", "b", "c", "d", "e", "f"],
            "dense_sim": [0.1, 0.2, 0.9, 0.9, 0.2, 0.1],
            "clip_image_sim": [0.9, 0.2, 0.1, 0.1, 0.2, 0.9],
            "qrel": [100, 0, 0, 100, 0, 0],
        }
    )
    qrels = {
        "1": {"a": 100, "b": 0, "c": 0},
        "2": {"d": 100, "e": 0, "f": 0},
    }
    return frame, qrels


def test_sweep_returns_one_ndcg_per_weight():
    frame, qrels = _frame_and_qrels()
    norms = {
        "dense_sim": fit_normaliser(frame["dense_sim"].to_numpy()),
        "clip_image_sim": fit_normaliser(frame["clip_image_sim"].to_numpy()),
    }
    result = sweep(frame, qrels, text_column="dense_sim",
                   image_column="clip_image_sim", normalisers=norms)
    assert len(result) == len(DEFAULT_WEIGHTS)
    assert [w for w, _ in result] == list(DEFAULT_WEIGHTS)
    assert all(0.0 <= score <= 1.0 for _, score in result)


def test_best_weight_picks_the_highest_ndcg():
    assert best_weight([(0.0, 0.70), (0.5, 0.90), (1.0, 0.80)]) == (0.5, 0.90)


def test_best_weight_breaks_a_tie_toward_the_lower_weight():
    # Deterministic, so two runs of the same sweep report the same control.
    assert best_weight([(0.3, 0.9), (0.7, 0.9)]) == (0.3, 0.9)


def test_best_weight_of_an_empty_sweep_raises():
    with pytest.raises(ValueError, match="empty sweep"):
        best_weight([])


def test_the_default_sweep_covers_both_extremes():
    assert DEFAULT_WEIGHTS[0] == 0.0
    assert DEFAULT_WEIGHTS[-1] == 1.0
    assert len(set(DEFAULT_WEIGHTS)) == len(DEFAULT_WEIGHTS)
