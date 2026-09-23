# Phase 3 — The Analysis and the Writeup

**Plan 7 of 7 · Phase 3 of 3 · Tasks 4–6.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 2 gate](phase-2-the-blend-report.md#phase-2-gate) passes.

**Delivers:** `src/error_analysis.py`, `src/ablation_table.py` and
`docs/RESULTS.md` — the three things `PROJECT_SPEC.md` §6 and §8.8 ask for
beyond the numbers themselves.

**Needs on disk:** everything Phases 1–2 built, plus the committed
`docs/results/*.json` of Plans 4–6.

**Owns Review Focus items 3, 4 and 5** (a table that mixes scopes silently;
per-category NDCG over categories too small to mean anything; a writeup that
quotes a number no file contains).

### Constraints that bite hardest here

- **Scope travels with every number.** Recall@k on 4,130 fold-0 queries, NDCG on 8,956 test queries and NDCG on a 2,000-query sample are three measurements, not three rows of one.
- **The floor is computed, never quoted.** 0.7467 measured; the published 0.7483 is wrong for this discount.
- **Field presence is itself a ranker.** The images analysis must not let "has an image vector" stand in for "images help" — `CLAUDE.md` measured presence flags alone at +0.0084 NDCG.
- **Every figure in the writeup comes from a committed JSON**, except the published baselines, which are cited from `PROJECT_SPEC.md` §5.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## What was measured, and what it decides

- **The LLM arm damages 26.4% of fold-0 queries** (better on 59.0%, identical
  on 14.6%), mean loss **−0.0607** where it hurts against a mean gain of
  **+0.0772** where it helps. About 1,090 queries — a population to
  characterise, not a handful of anecdotes to quote.
- **`s_category` is a path**, mean depth 3.7 and max 9, present on **86.7%** of
  fold-0 judgements with **49 distinct** top-level values. Only **13** have
  ≥100 queries; those cover **86.3%**. So the breakdown is 13 rows and an
  "other" bucket.
- **Per-query image coverage** averages **0.78**, with **1.7%** of queries
  having no imaged candidate and **25.4%** having all of them — enough spread
  to stratify by.

---

## Task 4: The error analysis

**Files:**
- Create: `src/error_analysis.py`
- Create: `tests/test_error_analysis.py`

**Interfaces:**
- Consumes: `src.stage_signals.load_signals`, `src.blend_report.{SINGLE_STAGES, single_stage_orderings, per_query_ndcg, rebuild_windows}`, `src.rank_report.qrels_from_frame`, `src.bootstrap.paired_delta_ci`.
- Produces:
  - `src.error_analysis.MIN_CATEGORY_QUERIES: int` = `100`
  - `src.error_analysis.OTHER: str` = `"(other)"`
  - `src.error_analysis.top_level_category(value) -> str | None`
  - `src.error_analysis.query_categories(judgements, products) -> dict[str, str]`
  - `src.error_analysis.category_breakdown(per_query_by_arm, categories, *, min_queries=MIN_CATEGORY_QUERIES) -> list[dict]`
  - `src.error_analysis.coverage_strata(coverage, *, edges=(0.0, 0.5, 0.9, 1.0001)) -> dict[str, list[str]]`
  - `src.error_analysis.delta_by_stratum(delta, strata, *, seed=0) -> list[dict]`
  - `src.error_analysis.failure_profile(arm, baseline, attributes, *, seed=0) -> dict`
  - CLI: `python -m src.error_analysis --out docs/results/error-analysis.json`

**Review Focus 4 lives in `category_breakdown`.** There are 49 top-level
categories and only 13 with ≥100 queries. A category with four queries will
swing ±0.15 from noise, and "images hurt in Musical Instruments" is exactly the
sentence a reader lifts out of a results file. Everything below the threshold
collapses into one `(other)` row, and every row carries its `n` so no reader
has to trust the threshold blindly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_error_analysis.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.error_analysis import (
    MIN_CATEGORY_QUERIES,
    OTHER,
    category_breakdown,
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_error_analysis.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.error_analysis'`.

- [ ] **Step 3: Write the implementation**

Create `src/error_analysis.py`:

```python
"""PROJECT_SPEC.md §6's last line: per-category error analysis, where images
help vs. hurt, and failure cases for the LLM reranker.

Three breakdowns, each shaped by something measured first:

  * **Per category.** s_category is a path (mean depth 3.7, max 9) present on
    86.7% of judgements with 49 distinct top-level values - but only 13 clear
    100 queries, and those cover 86.3%. A category with four queries swings
    +-0.15 on noise alone, so everything below the threshold collapses into one
    bucket and every row carries its n.

  * **By image coverage.** Per query, the share of judged candidates carrying
    an image vector averages 0.78; 1.7% of queries have none and 25.4% have
    all. That spread is the axis for "where images help vs. hurt" - and the
    question is asked of Ablation 3's own two arms, because *presence* of an
    image is itself a weak ranker (CLAUDE.md measured presence flags at
    +0.0084 NDCG) and must not be allowed to stand in for the image signal.

  * **The LLM's damage.** Measured on fold 0: the best arm in the project is
    better than Stage 2 on 59.0% of queries, *worse on 26.4%*, and identical on
    14.6%, with a mean loss of -0.0607 where it hurts. A single mean delta
    hides both halves, so the profile reports them separately and contrasts
    query attributes across the split.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.bootstrap import paired_delta_ci

DEFAULT_OUT = Path("docs/results/error-analysis.json")

# 13 of 49 top-level categories clear this, covering 86.3% of queries.
MIN_CATEGORY_QUERIES = 100
OTHER = "(other)"


def top_level_category(value) -> str | None:
    """The first element of an s_category path, or None."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0]) if len(value) else None
    return None


def query_categories(
    judgements: pd.DataFrame, products: pd.DataFrame
) -> dict[str, str]:
    """Each query's modal top-level category over its judged products."""
    categories = products.assign(
        _cat=products["s_category"].map(top_level_category)
    )[["product_id", "_cat"]]
    merged = judgements[["query_id", "product_id"]].merge(
        categories, on="product_id", how="left"
    ).dropna(subset=["_cat"])
    if merged.empty:
        return {}
    modal = merged.groupby("query_id")["_cat"].agg(lambda s: s.mode().iat[0])
    return {str(k): str(v) for k, v in modal.items()}


def category_breakdown(
    per_query_by_arm: Mapping[str, Mapping[str, float]],
    categories: Mapping[str, str],
    *,
    min_queries: int = MIN_CATEGORY_QUERIES,
) -> list[dict]:
    """Mean NDCG per arm per category, small categories collapsed.

    Review Focus 4: 49 categories, 13 of them meaningful. A four-query row
    reads like a finding and is noise.
    """
    counts: dict[str, int] = {}
    for query_id in categories:
        counts[categories[query_id]] = counts.get(categories[query_id], 0) + 1
    large = {name for name, n in counts.items() if n >= min_queries}

    buckets: dict[str, list[str]] = {}
    for query_id, category in categories.items():
        name = category if category in large else OTHER
        buckets.setdefault(name, []).append(query_id)

    rows = []
    for name, queries in buckets.items():
        rows.append(
            {
                "category": name,
                "n_queries": len(queries),
                "ndcg": {
                    arm: float(np.mean([per[q] for q in queries if q in per]))
                    for arm, per in per_query_by_arm.items()
                },
            }
        )
    named = sorted(
        [row for row in rows if row["category"] != OTHER],
        key=lambda row: -row["n_queries"],
    )
    other = [row for row in rows if row["category"] == OTHER]
    return named + other


def coverage_strata(
    coverage: Mapping[str, float],
    *,
    edges: Sequence[float] = (0.0, 0.5, 0.9, 1.0001),
) -> dict[str, list[str]]:
    """Queries bucketed by the share of their candidates carrying an image."""
    names = [f"[{edges[i]:.2f}, {edges[i + 1]:.2f})" for i in range(len(edges) - 1)]
    strata: dict[str, list[str]] = {name: [] for name in names}
    for query_id, value in coverage.items():
        for i in range(len(edges) - 1):
            if edges[i] <= float(value) < edges[i + 1]:
                strata[names[i]].append(str(query_id))
                break
    return strata


def delta_by_stratum(
    delta: Mapping[str, float],
    strata: Mapping[str, Sequence[str]],
    *,
    seed: int = 0,
) -> list[dict]:
    """A paired interval on the per-query delta within each stratum."""
    rows = []
    for name, queries in strata.items():
        present = [q for q in queries if q in delta]
        if not present:
            rows.append({"stratum": name, "n_queries": 0, "delta": None})
            continue
        arm = {q: float(delta[q]) for q in present}
        zero = {q: 0.0 for q in present}
        interval = paired_delta_ci(arm, zero, seed=seed)
        rows.append(
            {
                "stratum": name,
                "n_queries": len(present),
                "delta": {
                    "point": interval.point,
                    "low": interval.low,
                    "high": interval.high,
                },
                "significant": not (interval.low <= 0.0 <= interval.high),
            }
        )
    return rows


def failure_profile(
    arm: Mapping[str, float],
    baseline: Mapping[str, float],
    attributes: Mapping[str, Mapping[str, float]],
    *,
    seed: int = 0,
    tolerance: float = 1e-9,
) -> dict:
    """Where an arm helps, where it hurts, and how those queries differ.

    A single mean delta hides that the LLM both gains +0.0772 where it helps
    and loses -0.0607 where it hurts, on 26.4% of queries. §6 asks for the
    failure cases; this is the population, not a sample of it.
    """
    if set(arm) != set(baseline):
        raise ValueError(
            "arm and baseline must cover the same queries, or the populations "
            "are not comparable"
        )
    queries = sorted(arm)
    delta = {q: float(arm[q]) - float(baseline[q]) for q in queries}
    better = [q for q in queries if delta[q] > tolerance]
    worse = [q for q in queries if delta[q] < -tolerance]
    same = [q for q in queries if abs(delta[q]) <= tolerance]

    def mean_attribute(name: str, group: Sequence[str]) -> float | None:
        values = [
            float(attributes[name][q]) for q in group if q in attributes[name]
        ]
        return float(np.mean(values)) if values else None

    return {
        "n_queries": len(queries),
        "n_better": len(better),
        "n_worse": len(worse),
        "n_same": len(same),
        "share_worse": len(worse) / len(queries) if queries else 0.0,
        "mean_gain_when_better": (
            float(np.mean([delta[q] for q in better])) if better else 0.0
        ),
        "mean_loss_when_worse": (
            float(np.mean([delta[q] for q in worse])) if worse else 0.0
        ),
        "attributes": {
            name: {
                "better": mean_attribute(name, better),
                "worse": mean_attribute(name, worse),
                "same": mean_attribute(name, same),
            }
            for name in attributes
        },
    }


def _main() -> int:
    from src.blend_report import (
        SINGLE_STAGES,
        per_query_ndcg,
        rebuild_windows,
        single_stage_orderings,
    )
    from src.feature_matrix import select_columns
    from src.metrics import ndcg_per_query
    from src.rank_report import qrels_from_frame
    from src.ranker import EARLY_STOP_FOLD, TRAIN_FOLDS, folds, predict_run, train_ranker
    from src.stage_signals import load_signals

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="fold0")
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--products", type=Path, default=Path("data/combined/products.parquet"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signals = load_signals(args.scope, args.features_dir)
    ws, matrix = rebuild_windows(args.scope, features_dir=args.features_dir, seed=args.seed)
    qrels = qrels_from_frame(matrix)
    orderings = single_stage_orderings(signals)
    per_arm = {arm: per_query_ndcg(orderings[arm], ws, qrels) for arm in SINGLE_STAGES}

    # --- per category --------------------------------------------------------
    products = pd.read_parquet(args.products, columns=["product_id", "s_category"])
    judged = matrix[["query_id", "product_id"]].astype(str)
    categories = query_categories(judged, products)
    rows = category_breakdown(per_arm, categories)
    print(f"{len(rows) - 1} categories with >= {MIN_CATEGORY_QUERIES} queries, "
          f"plus {OTHER}")
    for row in rows:
        cells = "  ".join(f"{arm} {row['ndcg'][arm]:.4f}" for arm in SINGLE_STAGES)
        print(f"  {row['category'][:32]:34s} n={row['n_queries']:5,}  {cells}")

    # --- where images help vs. hurt -----------------------------------------
    # Asked of Ablation 3's own arms, not of image *presence*, which is itself
    # a weak ranker (CLAUDE.md: presence flags alone are worth +0.0084).
    train = pd.read_parquet(args.features_dir / "train.parquet")
    fit, early = folds(train, TRAIN_FOLDS), folds(train, [EARLY_STOP_FOLD])
    text_only = select_columns(["text", "esci_indicators", "retrieval"])
    with_image = select_columns(["text", "esci_indicators", "retrieval", "image"])
    per_text = ndcg_per_query(
        predict_run(train_ranker(fit, early, text_only, seed=args.seed), matrix), qrels
    )
    per_image = ndcg_per_query(
        predict_run(train_ranker(fit, early, with_image, seed=args.seed), matrix), qrels
    )
    image_delta = {q: per_image[q] - per_text[q] for q in per_text}

    coverage = (
        matrix.assign(_has=matrix["has_image_vector"].fillna(0).astype(float))
        .groupby("query_id")["_has"].mean()
    )
    coverage = {str(k): float(v) for k, v in coverage.items()}
    image_rows = delta_by_stratum(image_delta, coverage_strata(coverage), seed=args.seed)
    print("\n  -- Ablation 3's delta by image coverage --")
    for row in image_rows:
        if row["delta"] is None:
            print(f"  {row['stratum']:16s} n=0")
            continue
        d = row["delta"]
        print(f"  {row['stratum']:16s} n={row['n_queries']:5,}  {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]")

    # --- the LLM's failure population ---------------------------------------
    window_size = signals.groupby("query_id").size()
    attributes = {
        "n_candidates": {str(k): float(v) for k, v in
                         matrix.groupby("query_id").size().items()},
        "window_size": {str(k): float(v) for k, v in window_size.items()},
        "image_coverage": coverage,
        "stage2_ndcg": per_arm["stage2"],
    }
    profile = failure_profile(
        per_arm["stage2+llm"], per_arm["stage2"], attributes, seed=args.seed
    )
    print(f"\n  -- the LLM arm against Stage 2, per query --")
    print(f"  better {profile['n_better']:,} ({1 - profile['share_worse']:.1%} not worse), "
          f"worse {profile['n_worse']:,} ({profile['share_worse']:.1%}), "
          f"same {profile['n_same']:,}")
    print(f"  mean gain when better {profile['mean_gain_when_better']:+.4f}, "
          f"mean loss when worse {profile['mean_loss_when_worse']:+.4f}")
    for name, values in profile["attributes"].items():
        print(f"  {name:16s} better {values['better']}, worse {values['worse']}")

    payload = {
        "scope": args.scope,
        "n_queries": len(qrels),
        "min_category_queries": MIN_CATEGORY_QUERIES,
        "categories": rows,
        "image_coverage_strata": image_rows,
        "llm_failure_profile": profile,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_error_analysis.py -q`
Expected: PASS, 20 tests.

- [ ] **Step 5: Produce the analysis**

Run: `python -m src.error_analysis`

Expected: 13 named categories plus `(other)`; an Ablation 3 delta that is
larger in the high-image-coverage strata than the low ones (if it is not, say
so — "images help exactly where there are images" is a hypothesis, not a
finding); and an LLM failure profile near the measured 59.0 / 26.4 / 14.6 split
with a mean loss near −0.0607.

- [ ] **Step 6: Commit**

```bash
git add src/error_analysis.py tests/test_error_analysis.py docs/results/error-analysis.json
git commit -m "Break results down by category, image coverage and LLM failure"
```

---

## Task 5: The ablation table

**Files:**
- Create: `src/ablation_table.py`
- Create: `tests/test_ablation_table.py`

**Interfaces:**
- Consumes: the committed `docs/results/*.json` of Plans 4–7.
- Produces:
  - `src.ablation_table.ABLATIONS: tuple[dict, ...]` — the seven rows, each with its owner file, metric and scope
  - `src.ablation_table.Row` (frozen dataclass: `number`, `name`, `metric`, `scope`, `n_queries`, `delta`, `significant`, `source`)
  - `src.ablation_table.load_results(directory) -> dict[str, dict]`
  - `src.ablation_table.build_rows(results) -> list[Row]`
  - `src.ablation_table.format_markdown(rows) -> str`
  - CLI: `python -m src.ablation_table --out docs/results/ablation-table.json`

**Review Focus 3 lives in `Row.scope` and `format_markdown`.** The seven
ablations were measured on three different query populations with two different
metrics: 1–2 are Recall@100 over 4,130 fold-0 queries, 3–5 and 7 are NDCG over
the 8,956-query test split, 6 and Stage 4 are NDCG over a 2,000-query test
sample. Printed as seven rows with a single "delta" column and no scope, they
read as comparable, and every summary sentence written from that table inherits
the error. So `scope` and `n_queries` are required fields on every row, the
Markdown renderer prints them as columns, and `build_rows` raises if a row is
missing either.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ablation_table.py`:

```python
import json

import pytest

from src.ablation_table import (
    ABLATIONS,
    Row,
    build_rows,
    format_markdown,
    load_results,
)


def _row(number=1, **kwargs):
    base = dict(
        number=number, name="a thing", metric="NDCG", scope="test",
        n_queries=8956, delta={"point": 0.01, "low": 0.005, "high": 0.015},
        significant=True, source="docs/results/x.json",
    )
    base.update(kwargs)
    return Row(**base)


# --- the seven rows ---------------------------------------------------------

def test_all_seven_ablations_are_declared():
    assert [a["number"] for a in ABLATIONS] == [1, 2, 3, 4, 5, 6, 7]


def test_every_declared_ablation_names_its_source_and_metric():
    for ablation in ABLATIONS:
        assert ablation["source"].endswith(".json")
        assert ablation["metric"] in {"NDCG", "Recall@100"}


def test_the_recall_ablations_are_the_first_two():
    # §6: Ablations 1 and 2 are Recall@k, the rest NDCG.
    by_number = {a["number"]: a for a in ABLATIONS}
    assert by_number[1]["metric"] == "Recall@100"
    assert by_number[2]["metric"] == "Recall@100"
    assert by_number[3]["metric"] == "NDCG"


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
    for name in ("recall", "coarse-rank-test", "fine-rank-test", "blend-test"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"stem": name}))
    loaded = load_results(tmp_path)
    assert loaded["recall"]["stem"] == "recall"
    assert set(loaded) >= {"recall", "coarse-rank-test", "fine-rank-test", "blend-test"}


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ablation_table.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.ablation_table'`.

- [ ] **Step 3: Write the implementation**

Create `src/ablation_table.py`:

```python
"""PROJECT_SPEC.md §6 as one table, assembled from the committed results.

Seven ablations, owned by four plans, measured on **three different query
populations with two different metrics**:

    1, 2   Recall@100   4,130 fold-0 queries, full 1.2M-product corpus
    3,4,5,7  NDCG       8,956 test queries
    6        NDCG       2,000-query frozen test sample

Printed as seven rows of one "delta" column with no scope, those read as
comparable, and the summary sentence anyone writes from the table inherits the
error. So `scope` and `n_queries` are required on every row, the renderer
prints them, and the table carries the sentence that says they are not
comparable.

Nothing here computes a number. Every value is read from a committed JSON, so
the table cannot drift from the files it claims to summarise.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path("docs/results")
DEFAULT_OUT = DEFAULT_DIR / "ablation-table.json"

REQUIRED_FILES: tuple[str, ...] = (
    "recall",
    "coarse-rank-test",
    "fine-rank-test",
    "blend-test",
)

ABLATIONS: tuple[dict, ...] = (
    {"number": 1, "name": "Raw vs. LLM-rewritten query", "metric": "Recall@100",
     "scope": "fold 0, full corpus", "source": "docs/results/recall.json", "owner": "Plan 4"},
    {"number": 2, "name": "Dense-only vs. +BM25 vs. +image (RRF)", "metric": "Recall@100",
     "scope": "fold 0, full corpus", "source": "docs/results/recall.json", "owner": "Plan 4"},
    {"number": 3, "name": "Text-only vs. text+image features", "metric": "NDCG",
     "scope": "test", "source": "docs/results/coarse-rank-test.json", "owner": "Plan 5"},
    {"number": 4, "name": "With vs. without behavioural features", "metric": "NDCG",
     "scope": "test", "source": "docs/results/coarse-rank-test.json", "owner": "Plan 5"},
    {"number": 5, "name": "Pointwise vs. lambdarank", "metric": "NDCG",
     "scope": "test", "source": "docs/results/coarse-rank-test.json", "owner": "Plan 5"},
    {"number": 6, "name": "Coarse-only vs. +cross-encoder vs. +LLM listwise", "metric": "NDCG",
     "scope": "test sample", "source": "docs/results/fine-rank-test.json", "owner": "Plan 6"},
    {"number": 7, "name": "Learned fusion vs. fixed global weight", "metric": "NDCG",
     "scope": "test", "source": "docs/results/coarse-rank-test.json", "owner": "Plan 5"},
)

_SCOPE_WARNING = (
    "> These rows are **not comparable to each other**: they were measured on "
    "three different query populations with two different metrics. The scope "
    "and n columns say which."
)


@dataclass(frozen=True)
class Row:
    number: int
    name: str
    metric: str
    scope: str
    n_queries: int
    delta: Mapping[str, float] | None
    significant: bool | None
    source: str

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "name": self.name,
            "metric": self.metric,
            "scope": self.scope,
            "n_queries": self.n_queries,
            "delta": dict(self.delta) if self.delta else None,
            "significant": self.significant,
            "source": self.source,
        }


def load_results(directory: Path = DEFAULT_DIR) -> dict[str, dict]:
    """Every committed results file, keyed by stem."""
    directory = Path(directory)
    loaded: dict[str, dict] = {}
    for stem in REQUIRED_FILES:
        path = directory / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; the ablation table reads the committed "
                "results rather than recomputing them, so every plan's file "
                "must be present"
            )
        loaded[stem] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(directory.glob("*.json")):
        loaded.setdefault(path.stem, json.loads(path.read_text(encoding="utf-8")))
    return loaded


def build_rows(rows: Sequence[Row]) -> list[Row]:
    """Validate and order the rows. Scope and n are not optional."""
    for row in rows:
        if not row.scope:
            raise ValueError(
                f"ablation {row.number} has no scope; a table without one "
                "invites reading seven incomparable numbers as comparable"
            )
        if not row.n_queries:
            raise ValueError(
                f"ablation {row.number} has no n_queries; an interval without "
                "its sample size cannot be judged"
            )
    return sorted(rows, key=lambda row: row.number)


def format_markdown(rows: Sequence[Row]) -> str:
    """The §6 table, with the scope columns that keep it honest."""
    lines = [
        "| # | ablation | metric | scope | n | delta | |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.delta is None:
            cell, verdict = "—", ""
        else:
            cell = (f"{row.delta['point']:+.4f} "
                    f"[{row.delta['low']:+.4f}, {row.delta['high']:+.4f}]")
            verdict = "significant" if row.significant else "ties"
        lines.append(
            f"| {row.number} | {row.name} | {row.metric} | {row.scope} | "
            f"{row.n_queries:,} | {cell} | {verdict} |"
        )
    return "\n".join(lines) + "\n\n" + _SCOPE_WARNING


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    recall, coarse = results["recall"], results["coarse-rank-test"]
    fine, blend = results["fine-rank-test"], results["blend-test"]

    by_number = {a["number"]: dict(a) for a in ABLATIONS}

    # 1 - the rewritten query against the same fused channel. Plan 4 already
    # computed this as a *paired* bootstrap and stored it under
    # `ablation_1_rewrite`; subtracting two vs-BM25 intervals instead would be
    # the same interval arithmetic that over-corrected Plan 5's Ablation 4 by
    # 0.0143. Read the file, never re-derive.
    by_number[1]["delta"] = recall["ablation_1_rewrite"]["delta"]
    by_number[1]["significant"] = recall["ablation_1_rewrite"]["significant"]
    by_number[1]["n_queries"] = recall["n_queries"]

    # 2 - the full fusion against the BM25 baseline.
    fused = [c for c in recall["comparisons"] if c["arm"] == "dense+bm25+image"][0]
    by_number[2]["delta"] = fused["delta"]
    by_number[2]["significant"] = fused["significant"]
    by_number[2]["n_queries"] = recall["n_queries"]

    for number, key in [(3, "3_text_vs_image"), (7, "7_learned_fusion")]:
        entry = coarse["ablations"][key]
        by_number[number]["delta"] = entry["delta"]
        by_number[number]["significant"] = entry["significant"]
        by_number[number]["n_queries"] = coarse["n_queries"]

    behavioural = coarse["ablations"]["4_behavioural"]
    by_number[4]["delta"] = behavioural["values_over_indicators"]
    by_number[4]["significant"] = behavioural["values_over_indicators_significant"]
    by_number[4]["n_queries"] = coarse["n_queries"]

    objective = coarse["ablations"]["5_objective"][0]
    by_number[5]["delta"] = objective["delta"]
    by_number[5]["significant"] = objective["significant"]
    by_number[5]["n_queries"] = coarse["n_queries"]

    llm = [row for row in fine["ablation_6"] if row["arm"] == "stage2+llm"][0]
    by_number[6]["delta"] = llm["delta"]
    by_number[6]["significant"] = llm["significant"]
    by_number[6]["n_queries"] = fine["n_queries"]

    rows = build_rows([
        Row(number=a["number"], name=a["name"], metric=a["metric"],
            scope=a["scope"], n_queries=a["n_queries"], delta=a["delta"],
            significant=a["significant"], source=a["source"])
        for a in by_number.values()
    ])
    table = format_markdown(rows)
    print(table)

    payload = {
        "ablations": [row.to_dict() for row in rows],
        "markdown": table,
        "stage4": {
            "best_single_stage": blend["best_single_stage"],
            "any_blend_beats_best_single": blend["any_blend_beats_best_single"],
            "scope": blend["scope"],
            "n_queries": blend["n_queries"],
            "arms": {arm["name"]: arm["ndcg"] for arm in blend["arms"]},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ablation_table.py -q`
Expected: PASS, 13 tests.

- [ ] **Step 5: Produce the table**

Run: `python -m src.ablation_table`

Expected: seven rows with three distinct scopes visible, and
`docs/results/ablation-table.json` written. Check each delta against its source
file by eye once — this module's whole value is that it does not recompute, so
a wrong key selection would silently publish the wrong number under the right
name.

- [ ] **Step 6: Commit**

```bash
git add src/ablation_table.py tests/test_ablation_table.py docs/results/ablation-table.json
git commit -m "Assemble the seven ablations into one table with their scopes"
```

---

## Task 6: The writeup

**Files:**
- Create: `docs/RESULTS.md`
- Create: `tests/test_results_doc.py`
- Modify: `CLAUDE.md` (Commands section)

**Interfaces:**
- Consumes: every committed `docs/results/*.json`.
- Produces: no Python API; `docs/RESULTS.md` is the deliverable.

**Review Focus 5 lives in `tests/test_results_doc.py`.** The writeup is the one
artifact a reader will trust without opening a JSON, and this project has spent
six plans discovering that plausible numbers are often wrong: the published
floor (0.7483) is wrong for this discount, the published product count
(1,215,851) is off by three, the widely-copied label distribution has S and C
swapped, and Plan 6's first latency was a cached replay off by 214x. So the
test extracts **every four-decimal figure** from `RESULTS.md` and asserts each
one is either present in a committed results JSON or on a short allowlist of
figures cited from `PROJECT_SPEC.md` §5. A number that is neither is a number
someone typed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_results_doc.py`:

```python
import json
import re
from pathlib import Path

import pytest

DOC = Path("docs/RESULTS.md")
RESULTS = Path("docs/results")

# Cited from PROJECT_SPEC.md §5, not measured here. Everything else in the
# writeup must come from a committed results file.
PUBLISHED = {
    0.7483,   # the published random floor - wrong for this discount, quoted only to say so
    0.8107,   # CLIP_text zero-shot
    0.8225,   # CLIP_image zero-shot
    0.8292,   # SBERT_text zero-shot
    0.8562,   # ESCI_baseline, the target
    0.9043,   # KDD Cup 2022 winner
}

NUMBER = re.compile(r"\b0\.\d{4}\b")


def _all_values(node, out):
    if isinstance(node, dict):
        for value in node.values():
            _all_values(value, out)
    elif isinstance(node, list):
        for value in node:
            _all_values(value, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.add(round(float(node), 4))
        out.add(round(abs(float(node)), 4))


def _committed_values():
    values = set()
    for path in sorted(RESULTS.glob("*.json")):
        _all_values(json.loads(path.read_text(encoding="utf-8")), values)
    return values


def test_the_writeup_exists():
    assert DOC.exists(), "PROJECT_SPEC.md §8.8 asks for the writeup"


def test_the_writeup_names_the_target_and_the_measured_floor():
    text = DOC.read_text(encoding="utf-8")
    assert "0.8562" in text          # the target, §5
    assert "0.7467" in text or "0.7454" in text or "0.7468" in text


def test_the_writeup_does_not_quote_the_published_floor_as_this_project_s():
    # CLAUDE.md: compute the floor, never quote it. If 0.7483 appears it must
    # be in a sentence that says it is the published one.
    text = DOC.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "0.7483" in line:
            assert "publish" in line.lower() or "sqid" in line.lower()


@pytest.mark.data
def test_every_figure_in_the_writeup_comes_from_a_committed_result():
    # The writeup is the one artifact read without checking, and this project
    # has met four plausible-but-wrong published numbers already.
    text = DOC.read_text(encoding="utf-8")
    committed = _committed_values() | PUBLISHED
    unsourced = sorted(
        {float(m) for m in NUMBER.findall(text)} - committed
    )
    assert not unsourced, (
        f"figures in docs/RESULTS.md that no committed results file contains: "
        f"{unsourced}. Either quote the file's number or add the measurement."
    )


@pytest.mark.data
def test_the_writeup_states_the_headline_with_its_scope():
    # A 2,000-query sample number presented as a full-test number is the one
    # scope error a reader cannot catch.
    text = DOC.read_text(encoding="utf-8").lower()
    assert "2,000" in text or "2000" in text
    assert "sample" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_results_doc.py -q`
Expected: FAIL on `test_the_writeup_exists`.

- [ ] **Step 3: Write `docs/RESULTS.md`**

Write it from the committed JSON only — open each file and quote it; do not
retype a number from memory or from this plan, whose figures were measured
before the final runs. The structure:

1. **What this is.** One paragraph: a five-stage funnel on ESCI, the task is
   re-ranking judged candidates, recall is measured separately.
2. **Headline.** The project's best number, its scope, its CI, the measured
   floor beside it, and the `ESCI_baseline` target of 0.8562. Say plainly
   whether it beats, matches or loses to that target, and note that Plan 5's
   full-test coarse ranker matched it at 0.8579 while the LLM arm reached its
   number on a 2,000-query sample — and that the sample's Stage 2 value
   (0.8576) sits within 0.0003 of the full-test one (0.8579), which is the
   evidence that the sample is representative.
3. **The funnel, stage by stage.** One short section each, with the number that
   stage contributed and the file it came from.
4. **The ablation table.** Paste `docs/results/ablation-table.json`'s
   `markdown` field verbatim, including its scope warning.
5. **Stage 4.** What the blend did, against the best single stage, with the
   oracle beside it. If nothing beat its own best input, say so in the first
   sentence of the section, not the last.
6. **Error analysis.** The category table, the image-coverage strata, and the
   LLM failure population.
7. **What did not work.** A real section: Ablation 1's tie, Ablation 7's loss,
   the zero-shot cross-encoders losing to the coarse ranker, the redundant
   cascade, the listwise loss not paying at Stage 3, and whatever Stage 4
   measured. This project's plans recorded losses as results throughout; the
   writeup should too.
8. **What would come next**, and what it would cost.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_results_doc.py -q && python -m pytest tests/test_results_doc.py -m data -q`
Expected: PASS. If the sourcing test fails it has done its job: either the
figure is wrong or it needs to come from a file.

- [ ] **Step 5: Update `CLAUDE.md`'s Commands section**

Add to the fenced command block, after the fine-rank section:

```bash
# Blend and report (Plan 7). No API calls - the LLM orderings are cached.
python -m src.stage_signals --scope fold0         # -> data/features/stage-signals-fold0.parquet
python -m src.stage_signals --scope test-sample
python -m src.blend_report --scope fold0          # Stage 4 vs every stage alone
python -m src.blend_report --scope test-sample --final --out docs/results/blend-test.json
python -m src.error_analysis                      # per-category, image strata, LLM failures
python -m src.ablation_table                      # the seven-row §6 table
```

- [ ] **Step 6: Run the whole suite, both markers**

Run: `python -m pytest -q && python -m pytest -m data -q`
Expected: PASS, no new failures and no new skips.

- [ ] **Step 7: Commit**

```bash
git add docs/RESULTS.md tests/test_results_doc.py CLAUDE.md
git commit -m "Write the results document from the committed measurements"
```

- [ ] **Step 8: Mark the plan gate**

Tick the [Plan Gate](README.md#plan-gate) boxes in `README.md`, and update the
series entry for Plan 7 in `docs/superpowers/plans/README.md` with the landed
numbers in the style of Plans 1–6 — including, explicitly, whether Stage 4 beat
its best input.

```bash
git add docs/superpowers/plans/
git commit -m "Mark the blend and report plan gate as passed"
```

---

## Phase 3 Gate — and the Plan Gate

This phase's gate *is* the [Plan Gate](README.md#plan-gate), and the Plan Gate
is the project's. Check it there.

The three that are easiest to skip:

- [ ] Every row of the §6 table carries its scope and `n`, and the table carries the sentence saying the rows are not comparable.
- [ ] The per-category breakdown reports only categories with ≥100 queries and collapses the rest.
- [ ] Every four-decimal figure in `docs/RESULTS.md` is traceable to a committed results file or to the published baselines in `PROJECT_SPEC.md` §5.
