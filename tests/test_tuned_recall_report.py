import json
import sys

import pytest

from src.tuned_recall_report import (
    CHANNELS,
    PUBLISHED_FUSION,
    SWEEP_GRID,
    best_config,
    check_published,
    load_choice,
)


def _row(r100, r1000, k=60, weights=(1.0, 1.0, 1.0, 1.0)):
    return {"k": k, "weights": list(weights), "recall_at_100": r100, "recall_at_1000": r1000}


# --- choosing the fusion on fold 1 --------------------------------------------

def test_the_best_config_has_the_highest_recall_at_100():
    rows = [_row(0.61, 0.82, k=60), _row(0.62, 0.81, k=100), _row(0.60, 0.83, k=200)]
    assert best_config(rows)["k"] == 100


def test_a_tie_at_100_goes_to_the_deeper_recall():
    rows = [_row(0.62, 0.81, k=60), _row(0.62, 0.83, k=100)]
    assert best_config(rows)["k"] == 100


def test_an_exact_tie_breaks_deterministically_toward_the_first_listed():
    rows = [_row(0.62, 0.83, k=60), _row(0.62, 0.83, k=100)]
    assert best_config(rows)["k"] == 60


def test_no_rows_raises():
    with pytest.raises(ValueError, match="empty"):
        best_config([])


def test_the_grid_weighs_every_channel_and_never_silences_the_tuned_one():
    for k, weights in SWEEP_GRID:
        assert k > 0
        assert len(weights) == len(CHANNELS)
        assert weights[CHANNELS.index("dense_ft")] > 0


def test_the_published_fusion_is_plan_4_s_three_channels_at_equal_weight():
    k, weights = PUBLISHED_FUSION
    assert k == 60
    named = dict(zip(CHANNELS, weights))
    assert named == {"dense_ft": 0.0, "dense_base": 1.0, "bm25": 1.0, "image": 1.0}


# --- reading the choice back --------------------------------------------------

def test_the_choice_is_read_from_the_committed_sweep(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({"fold": 1, "chosen": {"k": 100, "weights": [1.5, 0.5, 1.0, 0.25]}}))
    assert load_choice(path) == (100, (1.5, 0.5, 1.0, 0.25))


def test_a_choice_made_anywhere_but_fold_1_is_refused(tmp_path):
    # Fold 0 is reported and test is measured once; a fusion tuned on either
    # is reporting the number it optimised.
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({"fold": 0, "chosen": {"k": 100, "weights": [1, 1, 1, 1]}}))
    with pytest.raises(ValueError, match="fold 1"):
        load_choice(path)


def test_a_missing_sweep_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--sweep"):
        load_choice(tmp_path / "absent.json")


# --- the published number must reproduce ---------------------------------------

def test_the_published_fusion_reproducing_passes():
    check_published({100: 0.55512, 1000: 0.79101}, {100: 0.55510, 1000: 0.79100})


def test_the_published_fusion_drifting_raises():
    # If the BM25 index, the landed dense store or the image store changed,
    # every delta here would be against a baseline nobody published.
    with pytest.raises(ValueError, match="drifted"):
        check_published({100: 0.5600, 1000: 0.7910}, {100: 0.5551, 1000: 0.7910})


# --- the CLI's guards ----------------------------------------------------------

def test_the_test_split_needs_final(monkeypatch):
    from src.tuned_recall_report import _main

    monkeypatch.setattr(sys, "argv", ["tuned_recall_report", "--split", "test"])
    with pytest.raises(SystemExit, match="--final"):
        _main()


def test_the_sweep_never_touches_test(monkeypatch):
    from src.tuned_recall_report import _main

    monkeypatch.setattr(sys, "argv", ["tuned_recall_report", "--sweep", "--split", "test", "--final"])
    with pytest.raises(SystemExit, match="fold 1"):
        _main()


# --- the committed record -------------------------------------------------------

def _committed(name):
    from pathlib import Path

    return json.loads(Path("docs/results", name).read_text(encoding="utf-8"))


def _recall(report, arm, k):
    return next(a for a in report["arms"] if a["name"] == arm)["recall"][str(k)]["recall"]


def test_the_committed_sweep_chose_its_best_row_on_fold_1():
    sweep = _committed("recall-fusion-sweep.json")
    assert sweep["fold"] == 1
    assert len(sweep["grid"]) == len(SWEEP_GRID)
    assert sweep["chosen"] == best_config(sweep["grid"])
    assert list(sweep["published"]["weights"]) == list(PUBLISHED_FUSION[1])


def test_the_fold_0_report_uses_the_fold_1_choice():
    sweep, report = _committed("recall-fusion-sweep.json"), _committed("recall-tuned.json")
    assert report["scope"] == "fold 0" and report["n_queries"] == 4130
    assert report["fusion"]["k"] == sweep["chosen"]["k"]
    assert report["fusion"]["weights"] == sweep["chosen"]["weights"]


def test_the_fold_0_report_reproduces_plan_4_s_published_fusion():
    report, plan4 = _committed("recall-tuned.json"), _committed("recall.json")
    for k in (100, 1000):
        assert _recall(report, "published fusion", k) == pytest.approx(
            _recall(plan4, "dense+bm25+image", k), abs=1e-4)
    assert report["retriever"]["training"]["folds"] == [2, 3, 4]


def test_the_test_report_scores_every_fold_0_arm_once_with_the_same_fusion():
    fold0, test = _committed("recall-tuned.json"), _committed("recall-tuned-test.json")
    assert test["split"] == "test" and test["n_queries"] == 8956
    assert [a["name"] for a in test["arms"]] == [a["name"] for a in fold0["arms"]]
    assert test["fusion"] == fold0["fusion"]
