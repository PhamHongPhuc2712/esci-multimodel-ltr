import json

import pytest

from src.report import evaluate_run, format_report, write_report


def _qrels(n_queries: int = 60) -> dict[str, dict[str, int]]:
    return {
        f"q{i}": {"a": 100, "b": 10, "c": 1, "d": 0} for i in range(n_queries)
    }


def _perfect_run(qrels) -> dict[str, dict[str, float]]:
    return {q: {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0} for q in qrels}


def _worst_run(qrels) -> dict[str, dict[str, float]]:
    return {q: {"d": 4.0, "c": 3.0, "b": 2.0, "a": 1.0} for q in qrels}


def test_perfect_run_scores_one_and_beats_the_floor():
    qrels = _qrels()
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="perfect", split="test", n_floor_trials=10
    )
    assert report.ndcg.point == pytest.approx(1.0)
    assert report.lift_over_floor.point > 0
    assert report.lift_over_floor.low > 0


def test_worst_run_loses_to_the_floor():
    qrels = _qrels()
    report = evaluate_run(
        _worst_run(qrels), qrels, run_tag="worst", split="test", n_floor_trials=10
    )
    assert report.lift_over_floor.point < 0
    assert report.lift_over_floor.high < 0


def test_report_records_the_query_count_and_the_floor():
    qrels = _qrels(n_queries=37)
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="t", split="test", n_floor_trials=10
    )
    assert report.n_queries == 37
    assert 0.0 < report.floor_mean < 1.0


def test_report_is_reproducible_for_a_given_seed():
    qrels = _qrels()
    run = _perfect_run(qrels)
    first = evaluate_run(run, qrels, run_tag="t", split="test", n_floor_trials=10, seed=3)
    second = evaluate_run(run, qrels, run_tag="t", split="test", n_floor_trials=10, seed=3)
    assert first.to_dict() == second.to_dict()


def test_formatted_report_shows_the_floor_next_to_the_headline():
    # PROJECT_SPEC.md: "A headline number of 0.86 means nothing without that
    # floor attached." The formatter is where that is enforced.
    qrels = _qrels()
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="t", split="test", n_floor_trials=10
    )
    text = format_report(report)
    assert "NDCG" in text
    assert "random floor" in text
    assert "95% CI" in text


def test_written_report_is_valid_json_with_the_floor_included(tmp_path):
    qrels = _qrels()
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="t", split="test", n_floor_trials=10
    )
    path = tmp_path / "t.json"
    write_report(report, path)
    payload = json.loads(path.read_text())
    assert payload["run_tag"] == "t"
    assert payload["floor_mean"] == pytest.approx(report.floor_mean)
    assert payload["ndcg"]["low"] <= payload["ndcg"]["point"] <= payload["ndcg"]["high"]


def test_evaluating_a_run_that_skips_a_query_raises():
    qrels = _qrels()
    run = _perfect_run(qrels)
    del run["q0"]
    with pytest.raises(KeyError, match="q0"):
        evaluate_run(run, qrels, run_tag="t", split="test", n_floor_trials=5)
