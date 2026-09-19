import os
import subprocess
import sys

import pandas as pd
import pytest

from src.splits import (
    FOLDS_PATH,
    N_FOLDS,
    assign_folds,
    fold_for_query_id,
    freeze_folds,
    load_folds,
    split_train_val,
)


def test_fold_is_in_range():
    assert all(0 <= fold_for_query_id(q) < N_FOLDS for q in range(1000))


def test_fold_assignment_is_stable_across_processes():
    # Python salts str.__hash__ per process, so a fold built on hash() would
    # silently reshuffle between runs and quietly leak validation queries into
    # training. This is the test that pins it.
    code = "from src.splits import fold_for_query_id; print(fold_for_query_id(12345))"
    outputs = []
    for hash_seed in ("0", "1", "random"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
        )
        outputs.append(result.stdout.strip())
    assert len(set(outputs)) == 1, outputs


def test_fold_assignment_does_not_depend_on_input_order():
    forwards = assign_folds(range(500))
    backwards = assign_folds(reversed(range(500)))
    merged = forwards.merge(backwards, on="query_id", suffixes=("_f", "_b"))
    assert (merged["fold_f"] == merged["fold_b"]).all()


def test_every_query_is_in_exactly_one_fold():
    folds = assign_folds(range(10_000))
    assert len(folds) == 10_000
    assert folds["query_id"].is_unique


def test_folds_are_roughly_balanced():
    folds = assign_folds(range(20_000))
    sizes = folds["fold"].value_counts()
    assert len(sizes) == N_FOLDS
    assert sizes.max() - sizes.min() < 0.05 * sizes.mean()


def test_split_train_val_never_puts_a_query_on_both_sides():
    judgements = pd.DataFrame(
        {
            "query_id": [q for q in range(200) for _ in range(3)],
            "product_id": [f"B{i}" for i in range(600)],
        }
    )
    folds = assign_folds(range(200))
    train, val = split_train_val(judgements, folds, val_fold=0)
    assert set(train["query_id"]).isdisjoint(set(val["query_id"]))
    assert len(train) + len(val) == len(judgements)
    assert len(val) > 0


def test_every_fold_can_serve_as_validation():
    judgements = pd.DataFrame({"query_id": list(range(500)), "product_id": ["B"] * 500})
    folds = assign_folds(range(500))
    for val_fold in range(N_FOLDS):
        train, val = split_train_val(judgements, folds, val_fold)
        assert len(val) > 0
        assert len(train) > 0


def test_unknown_validation_fold_raises():
    folds = assign_folds(range(100))
    judgements = pd.DataFrame({"query_id": list(range(100)), "product_id": ["B"] * 100})
    with pytest.raises(ValueError, match="val_fold"):
        split_train_val(judgements, folds, val_fold=N_FOLDS)


def test_judgement_for_an_unassigned_query_raises():
    # A query in the judgements but missing from the frozen folds would
    # otherwise be dropped from both sides without a word.
    folds = assign_folds(range(10))
    judgements = pd.DataFrame({"query_id": [3, 999], "product_id": ["B1", "B2"]})
    with pytest.raises(ValueError, match="not in the frozen folds"):
        split_train_val(judgements, folds, val_fold=0)


def test_freeze_and_load_round_trip(tmp_path):
    folds = assign_folds(range(100))
    path = tmp_path / "val_folds.csv"
    digest = freeze_folds(folds, path)
    assert (tmp_path / "val_folds.sha256").read_text().split()[0] == digest
    loaded = load_folds(path)
    assert loaded.equals(folds.sort_values("query_id").reset_index(drop=True))


def test_freezing_twice_produces_the_same_bytes(tmp_path):
    folds = assign_folds(range(100))
    first = freeze_folds(folds, tmp_path / "a.csv")
    second = freeze_folds(folds.sample(frac=1, random_state=0), tmp_path / "b.csv")
    assert first == second


def test_load_detects_a_tampered_file(tmp_path):
    folds = assign_folds(range(100))
    path = tmp_path / "val_folds.csv"
    freeze_folds(folds, path)
    path.write_text(path.read_text() + "999999,0\n")
    with pytest.raises(ValueError, match="sha256"):
        load_folds(path)


@pytest.mark.data
def test_committed_folds_still_match_the_real_train_split():
    from src.dataset import load_split

    committed = load_folds(FOLDS_PATH)
    recomputed = assign_folds(load_split("train").judgements["query_id"].unique())
    assert committed.equals(recomputed.sort_values("query_id").reset_index(drop=True))
