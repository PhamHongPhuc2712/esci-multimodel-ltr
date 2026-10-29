"""Calibration data for quantisation comes from the training folds only."""

import pandas as pd
import pytest

from src.quantize_llm import calibration_rows, recipe


def _matrix():
    return pd.DataFrame({"query_id": [str(i) for i in range(50)],
                         "product_id": [f"p{i}" for i in range(50)],
                         "gain": [1.0] * 50, "fold": [i % 5 for i in range(50)]})


def test_calibration_rows_come_from_training_folds_only():
    rows = calibration_rows(_matrix(), n=20, seed=0)
    assert len(rows) == 20
    assert set(rows["fold"]) <= {2, 3, 4}


def test_calibration_refuses_more_rows_than_the_training_folds_hold():
    with pytest.raises(ValueError, match="calibration"):
        calibration_rows(_matrix(), n=31, seed=0)


def test_an_unknown_scheme_is_refused_before_importing_anything():
    with pytest.raises(ValueError, match="scheme"):
        recipe("fp8")
