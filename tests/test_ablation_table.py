import json

import pytest

from src.ablation_table import (
    ABLATIONS,
    Row,
    by_arm,
    build_rows,
    format_markdown,
    load_results,
)


def _row(number=1, **kwargs):
    base = dict(
        number=number, label=str(number), name="a thing", metric="NDCG",
        scope="test", n_queries=8956,
        delta={"point": 0.01, "low": 0.005, "high": 0.015},
        significant=True, source="docs/results/x.json",
    )
    base.update(kwargs)
    return Row(**base)


# --- the seven ablations ----------------------------------------------------

def test_all_seven_ablations_are_declared():
    assert sorted({a["number"] for a in ABLATIONS}) == [1, 2, 3, 4, 5, 6, 7]
    assert len({a["label"] for a in ABLATIONS}) == len(ABLATIONS)


def test_every_declared_ablation_names_its_source_and_metric():
    for ablation in ABLATIONS:
        assert ablation["source"].endswith(".json")
        assert ablation["metric"] in {"NDCG", "Recall@100"}


def test_the_recall_ablations_are_the_first_two():
    # §6: Ablations 1 and 2 are Recall@k, the rest NDCG.
    for ablation in ABLATIONS:
        expected = "Recall@100" if ablation["number"] in (1, 2) else "NDCG"
        assert ablation["metric"] == expected


def test_ablation_2_is_reported_rung_by_rung():
    # Fusion-vs-BM25 (+0.0661) credits the image channel with the whole fusion
    # gain; its own step is +0.0048. Plan 4 stored the ladder for this reason.
    import src.ablation_table as module

    rungs = sorted(a["label"] for a in ABLATIONS if a["number"] == 2)
    assert rungs == ["2a", "2b"]
    assert "ablation_2_ladder" in __import__("inspect").getsource(module)


# --- Review Focus 3: scope is not optional ---------------------------------

def test_a_row_without_a_scope_is_refused():
    with pytest.raises(ValueError, match="scope"):
        build_rows([_row(scope="")])


def test_a_row_without_an_n_is_refused():
    with pytest.raises(ValueError, match="n_queries"):
        build_rows([_row(n_queries=0)])


def test_the_table_prints_scope_and_n_as_columns():
    table = format_markdown([_row(), _row(number=2, scope="fold 0", n_queries=4130)])
    header = table.splitlines()[0]
    assert "scope" in header.lower()
    assert "n" in header.lower()


def test_rows_of_different_scopes_are_visibly_different():
    table = format_markdown([
        _row(number=1, metric="Recall@100", scope="fold 0", n_queries=4130),
        _row(number=6, metric="NDCG", scope="test sample", n_queries=2000),
    ])
    assert "4,130" in table
    assert "2,000" in table
    assert "Recall@100" in table
    assert "NDCG" in table


def test_the_table_carries_a_scope_warning():
    # The one sentence that stops the table being read as seven comparable
    # numbers.
    table = format_markdown([_row()])
    assert "not comparable" in table.lower() or "different" in table.lower()


# --- assembling from the committed files ------------------------------------

def test_a_missing_results_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="recall.json"):
        load_results(tmp_path)


def test_results_load_by_stem(tmp_path):
    for name in ("recall", "coarse-rank-test", "fine-rank-test-full", "blend-test-full"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"stem": name}))
    loaded = load_results(tmp_path)
    assert loaded["recall"]["stem"] == "recall"
    assert set(loaded) >= {"recall", "coarse-rank-test", "fine-rank-test-full", "blend-test-full"}


def test_ablation_1_is_read_from_the_paired_bootstrap_not_re_derived():
    # recall.json stores the rewrite comparison as its own paired bootstrap.
    # Subtracting two vs-BM25 intervals is the interval arithmetic that
    # over-corrected Plan 5's Ablation 4 by 0.0143.
    import src.ablation_table as module
    source = __import__("inspect").getsource(module)
    assert "ablation_1_rewrite" in source


def test_a_row_records_where_it_came_from():
    rows = build_rows([_row(source="docs/results/recall.json")])
    assert rows[0].source == "docs/results/recall.json"


def test_an_ordered_table_is_numbered_one_to_seven():
    rows = build_rows([_row(number=n) for n in (3, 1, 2)])
    assert [row.number for row in rows] == [1, 2, 3]


def test_ablations_5_and_6_report_each_measured_comparison():
    # Ablation 5 measured two pointwise baselines and Ablation 6 two rerankers;
    # showing one of each hides the other, the same way one fusion-vs-BM25 row
    # hid Ablation 2's ladder.
    for number, expected in [(5, ["5a", "5b"]), (6, ["6a", "6b"])]:
        assert sorted(a["label"] for a in ABLATIONS if a["number"] == number) == expected


def test_every_row_name_states_its_direction():
    # A negative Ablation 5 means lambdarank wins; unlabelled, it reads as
    # "the listwise objective hurts".
    for ablation in ABLATIONS:
        assert " − " in ablation["name"], ablation["label"]


def test_a_comparison_is_found_by_arm_name():
    rows = [{"arm": "full/pointwise_regression", "delta": 1},
            {"arm": "full/pointwise_class", "delta": 2}]
    assert by_arm(rows, "full/pointwise_class")["delta"] == 2


def test_a_missing_comparison_is_named_not_guessed():
    # Picking by position is how "pointwise classifier" silently became the
    # regression arm: the plan read 5_objective[0].
    with pytest.raises(KeyError, match="pointwise_class"):
        by_arm([{"arm": "full/pointwise_regression"}], "full/pointwise_class")


def test_rows_sharing_a_number_order_by_label():
    rows = build_rows([_row(number=2, label="2b"), _row(number=2, label="2a")])
    assert [row.label for row in rows] == ["2a", "2b"]
