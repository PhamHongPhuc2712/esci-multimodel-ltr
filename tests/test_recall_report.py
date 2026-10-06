import math

import pytest

from pathlib import Path

from src.recall_report import (
    SPEC_BUDGET_GB,
    STORAGE_ROOTS,
    compare,
    evaluate_arm,
    format_table,
    storage_gb,
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


def test_storage_is_measured_from_disk(tmp_path):
    # PROJECT_SPEC.md §9 budgets "<1 GB" for embeddings. The figure was a
    # quoted dict until Plan 12, and it still counted a store deleted on
    # 2026-09-30; measuring it means it cannot go stale again.
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "vectors.f16").write_bytes(b"x" * 1_500_000)
    (tmp_path / "a" / "keys.txt").write_bytes(b"y" * 500_000)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "index.npy").write_bytes(b"z" * 10_000_000)
    assert storage_gb({"a": tmp_path / "a", "b": tmp_path / "b"}) == {
        "a": pytest.approx(0.002, abs=1e-6),
        "b": pytest.approx(0.01, abs=1e-6),
    }


def test_a_missing_store_is_named_not_counted_as_zero(tmp_path):
    with pytest.raises(FileNotFoundError, match="dense_ft"):
        storage_gb({"dense_ft": tmp_path / "absent"})


def test_the_live_stores_are_named_and_the_deleted_one_is_not():
    assert SPEC_BUDGET_GB == 1.0
    assert set(STORAGE_ROOTS) == {"bm25", "dense", "dense_ft", "image_catalogue"}
    assert all(isinstance(root, Path) for root in STORAGE_ROOTS.values())


def test_no_recall_number_is_quoted_in_the_module():
    # 0.5018 was BM25's Recall@100 on a scratch sample, written into every
    # results file as "bm25_baseline_r100". The arm's own measurement is in
    # `arms`; nothing else should be typed.
    assert "0.5018" not in Path("src/recall_report.py").read_text(encoding="utf-8")


def test_the_test_split_needs_final(monkeypatch):
    import sys

    from src.recall_report import _main

    monkeypatch.setattr(sys, "argv", ["recall_report", "--split", "test"])
    with pytest.raises(SystemExit, match="--final"):
        _main()
