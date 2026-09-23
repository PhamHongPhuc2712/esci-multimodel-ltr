import numpy as np
import pandas as pd
import pytest

from src.error_analysis import (
    MIN_CATEGORY_QUERIES,
    OTHER,
    category_breakdown,
    category_buckets,
    coverage_strata,
    delta_by_stratum,
    failure_profile,
    query_categories,
    top_level_category,
)


# --- the category path ------------------------------------------------------

def test_the_top_level_is_the_first_element():
    assert top_level_category(["Electronics", "Headphones", "Over-Ear"]) == "Electronics"


def test_a_numpy_array_path_works_too():
    # s_category arrives from parquet as an ndarray, not a list.
    assert top_level_category(np.array(["Home & Kitchen", "Bedding"])) == "Home & Kitchen"


def test_an_empty_or_missing_path_has_no_category():
    assert top_level_category([]) is None
    assert top_level_category(None) is None
    assert top_level_category(float("nan")) is None


def test_a_query_takes_the_modal_category_of_its_judged_products():
    judgements = pd.DataFrame({
        "query_id": ["1", "1", "1", "2"],
        "product_id": ["a", "b", "c", "x"],
    })
    products = pd.DataFrame({
        "product_id": ["a", "b", "c", "x"],
        "s_category": [["Electronics"], ["Electronics"], ["Toys & Games"], ["Beauty"]],
    })
    assert query_categories(judgements, products) == {"1": "Electronics", "2": "Beauty"}


def test_a_query_whose_products_have_no_category_is_absent():
    judgements = pd.DataFrame({"query_id": ["1"], "product_id": ["a"]})
    products = pd.DataFrame({"product_id": ["a"], "s_category": [None]})
    assert query_categories(judgements, products) == {}


# --- Review Focus 4: small categories are noise ----------------------------

def _per_arm(n_big=120, n_small=4):
    rng = np.random.default_rng(0)
    per = {}
    categories = {}
    for i in range(n_big):
        q = f"big{i}"
        per[q] = 0.80 + rng.normal(scale=0.02)
        categories[q] = "Electronics"
    for i in range(n_small):
        q = f"small{i}"
        per[q] = 0.50 + rng.normal(scale=0.02)
        categories[q] = "Musical Instruments"
    return {"stage2": per, "stage2+llm": {k: v + 0.03 for k, v in per.items()}}, categories


def test_a_category_below_the_threshold_is_collapsed():
    per_arm, categories = _per_arm()
    rows = category_breakdown(per_arm, categories)
    names = {row["category"] for row in rows}
    assert "Electronics" in names
    assert "Musical Instruments" not in names
    assert OTHER in names


def test_every_row_carries_its_n():
    per_arm, categories = _per_arm()
    rows = category_breakdown(per_arm, categories)
    assert all(row["n_queries"] > 0 for row in rows)
    assert sum(row["n_queries"] for row in rows) == len(categories)


def test_every_row_carries_every_arm():
    per_arm, categories = _per_arm()
    rows = category_breakdown(per_arm, categories)
    for row in rows:
        assert set(row["ndcg"]) == set(per_arm)


def test_rows_are_ordered_by_size_with_other_last():
    per_arm, categories = _per_arm()
    rows = category_breakdown(per_arm, categories)
    assert rows[-1]["category"] == OTHER
    sizes = [row["n_queries"] for row in rows[:-1]]
    assert sizes == sorted(sizes, reverse=True)


def test_the_threshold_is_the_measured_one():
    # 13 of 49 top-level categories clear 100 queries, covering 86.3%.
    assert MIN_CATEGORY_QUERIES == 100


def test_category_buckets_collapse_exactly_as_the_breakdown_does():
    # The per-category intervals and the per-category table must be over the
    # same queries, or a row's n and its interval describe two populations.
    per_arm, categories = _per_arm()
    buckets = category_buckets(categories)
    rows = category_breakdown(per_arm, categories)
    assert {row["category"]: row["n_queries"] for row in rows} == {
        name: len(queries) for name, queries in buckets.items()
    }


def test_a_breakdown_with_no_large_category_is_all_other():
    per_arm = {"a": {"q1": 0.8, "q2": 0.9}}
    rows = category_breakdown(per_arm, {"q1": "X", "q2": "Y"})
    assert [row["category"] for row in rows] == [OTHER]


# --- image strata -----------------------------------------------------------

def test_strata_partition_every_query():
    coverage = {"a": 0.0, "b": 0.4, "c": 0.7, "d": 1.0}
    strata = coverage_strata(coverage)
    assigned = [q for queries in strata.values() for q in queries]
    assert sorted(assigned) == ["a", "b", "c", "d"]


def test_full_coverage_lands_in_the_top_stratum():
    strata = coverage_strata({"d": 1.0})
    top = list(strata)[-1]
    assert strata[top] == ["d"]


def test_an_all_imaged_query_has_its_own_stratum():
    # Presence can only rank within a query when some candidates lack an
    # image. Where every candidate has one, has_image_vector is constant and
    # the delta is the image signal alone - so that population stands apart.
    strata = coverage_strata({"most": 0.95, "all": 1.0})
    assert strata["= 1.00"] == ["all"]
    assert "most" not in strata["= 1.00"]


def test_stratum_names_do_not_hide_the_closed_top_edge():
    # "[0.90, 1.00)" read as excluding 1.0 while the stratum contained it.
    names = list(coverage_strata({}))
    assert names[-2] == "[0.90, 1.00)"
    assert names[-1] == "= 1.00"


def test_a_stratum_delta_carries_an_interval_and_an_n():
    delta = {f"q{i}": 0.01 * (i % 5) for i in range(40)}
    strata = {"low": [f"q{i}" for i in range(20)], "high": [f"q{i}" for i in range(20, 40)]}
    rows = delta_by_stratum(delta, strata)
    assert {row["stratum"] for row in rows} == {"low", "high"}
    for row in rows:
        assert row["n_queries"] == 20
        assert set(row["delta"]) == {"point", "low", "high"}


def test_an_empty_stratum_is_reported_not_dropped():
    rows = delta_by_stratum({"q": 0.1}, {"empty": [], "full": ["q"]})
    empty = [row for row in rows if row["stratum"] == "empty"][0]
    assert empty["n_queries"] == 0
    assert empty["delta"] is None


# --- the LLM failure population --------------------------------------------

def test_the_failure_profile_counts_the_three_populations():
    arm = {"a": 0.9, "b": 0.5, "c": 0.7}
    baseline = {"a": 0.5, "b": 0.9, "c": 0.7}
    profile = failure_profile(arm, baseline, {})
    assert profile["n_better"] == 1
    assert profile["n_worse"] == 1
    assert profile["n_same"] == 1


def test_the_failure_profile_separates_mean_gain_from_mean_loss():
    # A single "mean delta" hides that the arm both helps a lot and hurts a
    # lot; §6 asks for the failure cases, not the average.
    arm = {"a": 0.9, "b": 0.5}
    baseline = {"a": 0.5, "b": 0.9}
    profile = failure_profile(arm, baseline, {})
    assert profile["mean_gain_when_better"] == pytest.approx(0.4)
    assert profile["mean_loss_when_worse"] == pytest.approx(-0.4)


def test_the_failure_profile_contrasts_attributes_across_the_split():
    # The point is to characterise the damaged population, not just count it.
    arm = {"a": 0.9, "b": 0.5}
    baseline = {"a": 0.5, "b": 0.9}
    profile = failure_profile(arm, baseline, {"n_candidates": {"a": 3.0, "b": 9.0}})
    assert profile["attributes"]["n_candidates"]["better"] == pytest.approx(3.0)
    assert profile["attributes"]["n_candidates"]["worse"] == pytest.approx(9.0)


def test_the_failure_profile_needs_the_same_queries():
    with pytest.raises(ValueError, match="same queries"):
        failure_profile({"a": 0.1}, {"b": 0.1}, {})
