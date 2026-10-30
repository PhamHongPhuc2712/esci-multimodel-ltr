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


# --- the re-ranking ablations, 3 to 7 ----------------------------------------

def test_the_reranking_ablations_are_declared():
    # 1 and 2 were Recall@k and left with the recall stage.
    assert sorted({a["number"] for a in ABLATIONS}) == [3, 4, 5, 6, 7]
    assert len({a["label"] for a in ABLATIONS}) == len(ABLATIONS)


def test_every_declared_ablation_names_its_source_and_metric():
    for ablation in ABLATIONS:
        assert ablation["source"].endswith(".json")
        assert ablation["metric"] == "NDCG"


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
        _row(number=3, scope="fold 0", n_queries=4130),
        _row(number=6, scope="test sample", n_queries=2000),
    ])
    assert "4,130" in table and "fold 0" in table
    assert "2,000" in table and "test sample" in table


def test_the_table_says_what_each_delta_is():
    note = format_markdown([_row()]).splitlines()[-1]
    assert "bootstrap" in note and "scope" in note


# --- assembling from the committed files ------------------------------------

def test_a_missing_results_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="coarse-rank-test.json"):
        load_results(tmp_path)


def test_results_load_by_stem(tmp_path):
    from src.ablation_table import REQUIRED_FILES

    for name in REQUIRED_FILES:
        (tmp_path / f"{name}.json").write_text(json.dumps({"stem": name}))
    loaded = load_results(tmp_path)
    assert loaded["coarse-rank-test"]["stem"] == "coarse-rank-test"
    assert set(loaded) >= set(REQUIRED_FILES)


def test_a_row_records_where_it_came_from():
    rows = build_rows([_row(source="docs/results/coarse-rank-test.json")])
    assert rows[0].source == "docs/results/coarse-rank-test.json"


def test_an_ordered_table_is_numbered_one_to_seven():
    rows = build_rows([_row(number=n) for n in (3, 1, 2)])
    assert [row.number for row in rows] == [1, 2, 3]


def test_ablations_5_and_6_report_each_measured_comparison():
    # Ablation 5 measured two pointwise baselines and Ablation 6 two rerankers;
    # showing one of each hides the other.
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


# --- Plan 12: the findings after the spec ---------------------------------------

def test_every_finding_names_its_file_block_and_both_arms():
    from src.ablation_table import FINDINGS, REQUIRED_FILES

    for finding in FINDINGS:
        assert finding["file"] in REQUIRED_FILES, finding["label"]
        assert finding["block"] in {"comparisons", "against_best_single"}
        assert finding["arm"] and finding["baseline"] and "−" in finding["name"]
        assert finding["scope"] and finding["metric"] and finding["owner"]


def test_the_findings_cover_stage_4_and_the_reranking_plans():
    from src.ablation_table import FINDINGS

    assert [f["label"] for f in FINDINGS] == ["S4", "8a", "9a", "9b", "9c", "11", "13a", "13b"]


def test_a_comparison_is_found_by_both_names():
    from src.ablation_table import find_comparison

    rows = [
        {"arm": "a", "baseline": "stage2", "delta": 1},
        {"arm": "a", "baseline": "stage2+ce", "delta": 2},
    ]
    assert find_comparison(rows, "a", "stage2+ce")["delta"] == 2


def test_a_missing_pair_is_named_not_guessed():
    from src.ablation_table import find_comparison

    with pytest.raises(KeyError, match="'a' against 'c'"):
        find_comparison([{"arm": "a", "baseline": "b"}], "a", "c")


def test_build_findings_reads_each_file_by_name():
    from src.ablation_table import FINDINGS, build_findings

    results = {}
    for finding in FINDINGS:
        payload = results.setdefault(finding["file"], {"n_queries": 10})
        payload.setdefault(finding["block"], []).append({
            "arm": finding["arm"], "baseline": finding["baseline"], "k": finding.get("k"),
            "delta": {"point": 0.5, "low": 0.4, "high": 0.6}, "significant": True,
        })
    rows = build_findings(results)
    assert [row.label for row in rows] == ["S4", "8a", "9a", "9b", "9c", "11", "13a", "13b"]
    assert all(row.n_queries == 10 and row.delta["point"] == 0.5 for row in rows)


def test_every_readme_finding_with_an_interval_is_in_the_committed_table():
    # Review Focus 4. The README's findings table used to be typed by hand
    # for Plans 8-11; every bracketed delta in it must now be a row here.
    import re
    from pathlib import Path

    table = json.loads(Path("docs/results/ablation-table.json").read_text(encoding="utf-8"))
    rows = table["ablations"] + table["findings"]
    deltas = {(round(r["delta"]["point"], 4), round(r["delta"]["low"], 4), round(r["delta"]["high"], 4))
              for r in rows if r["delta"]}
    readme = Path("README.md").read_text(encoding="utf-8")
    section = readme.split("## What the ablations found")[1].split("\n## ")[0]
    pattern = re.compile(r"([+−-]\d\.\d{4}) \[([+−-]\d\.\d{4}), ([+−-]\d\.\d{4})\]")
    found = [tuple(round(float(v.replace("−", "-")), 4) for v in m) for m in pattern.findall(section)]
    assert found, "no bracketed delta in the README's findings table"
    for triple in found:
        assert triple in deltas, triple
