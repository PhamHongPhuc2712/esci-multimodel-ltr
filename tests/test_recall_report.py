import math

import pytest

from src.recall_report import (
    SPEC_BUDGET_GB,
    STORAGE_GB,
    compare,
    evaluate_arm,
    format_table,
)


def _arm(name, retrieved, relevant):
    return evaluate_arm(name, retrieved, relevant, ks=(1, 10))


def test_an_arm_reports_recall_at_every_k():
    arm = _arm("bm25", {1: ["a", "b"]}, {1: {"a", "b"}})
    assert arm.table[1].mean == pytest.approx(0.5)
    assert arm.table[10].mean == pytest.approx(1.0)


def test_an_arm_serialises_with_its_denominator():
    # Two arms with different query coverage are not comparable, so the
    # denominator travels with the number.
    arm = _arm("bm25", {1: ["a"], 2: ["x"]}, {1: {"a"}})
    payload = arm.to_dict()
    assert payload["name"] == "bm25"
    assert payload["recall"]["10"]["n_queries"] == 1
    assert payload["recall"]["10"]["n_skipped"] == 1


def test_compare_reports_a_paired_delta_with_an_interval():
    better = _arm("fused", {1: ["a"], 2: ["b"]}, {1: {"a"}, 2: {"b"}})
    worse = _arm("bm25", {1: ["x"], 2: ["y"]}, {1: {"a"}, 2: {"b"}})
    result = compare(better, worse, k=10)
    assert result["delta"]["point"] == pytest.approx(1.0)
    assert result["baseline"] == "bm25"


def test_compare_pairs_on_queries_not_on_position():
    # paired_delta_ci pairs by key. An arm that answered a different set of
    # queries must not be silently compared position by position.
    a = _arm("a", {1: ["a"], 2: ["b"]}, {1: {"a"}, 2: {"b"}})
    b = _arm("b", {2: ["b"], 1: ["a"]}, {1: {"a"}, 2: {"b"}})
    assert compare(a, b, k=10)["delta"]["point"] == pytest.approx(0.0)


def test_compare_on_a_k_the_arm_does_not_have_raises():
    arm = _arm("a", {1: ["a"]}, {1: {"a"}})
    with pytest.raises(KeyError, match="100"):
        compare(arm, arm, k=100)


def test_the_table_names_every_arm():
    arms = [
        _arm("bm25", {1: ["a"]}, {1: {"a"}}),
        _arm("dense", {1: ["a"]}, {1: {"a"}}),
    ]
    rendered = format_table(arms)
    assert "bm25" in rendered and "dense" in rendered


def test_the_table_renders_a_nan_recall_without_crashing():
    # Every query skipped gives a NaN mean; the report must say so rather
    # than raise or print a misleading 0.000.
    arm = _arm("empty", {1: ["a"]}, {})
    assert math.isnan(arm.table[10].mean)
    assert "nan" in format_table([arm]).lower()


def test_the_storage_overrun_is_recorded_not_hidden():
    # PROJECT_SPEC.md §9 budgets "<1 GB" for embeddings. The three channels
    # total 2.94 GB. The honest number is worth more than a flattering one.
    assert SPEC_BUDGET_GB == 1.0
    assert sum(STORAGE_GB.values()) > SPEC_BUDGET_GB
    assert set(STORAGE_GB) == {
        "bm25",
        "dense",
        "image_catalogue",
        # A separate store from image_catalogue, not an extension of it: the
        # scopes write to different directories, so running both costs both.
        "image_rerank",
    }
