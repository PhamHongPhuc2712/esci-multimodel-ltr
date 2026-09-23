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


def category_buckets(
    categories: Mapping[str, str], *, min_queries: int = MIN_CATEGORY_QUERIES
) -> dict[str, list[str]]:
    """Query ids per category, small categories collapsed into OTHER.

    Ordered largest first with OTHER last. The table and the per-category
    intervals both come from here, so a row's n and its interval always
    describe the same queries.
    """
    counts: dict[str, int] = {}
    for category in categories.values():
        counts[category] = counts.get(category, 0) + 1
    large = {name for name, n in counts.items() if n >= min_queries}

    buckets: dict[str, list[str]] = {}
    for query_id, category in categories.items():
        name = category if category in large else OTHER
        buckets.setdefault(name, []).append(str(query_id))
    named = sorted((n for n in buckets if n != OTHER), key=lambda n: -len(buckets[n]))
    return {name: buckets[name] for name in [*named, *([OTHER] if OTHER in buckets else [])]}


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
    return [
        {
            "category": name,
            "n_queries": len(queries),
            "ndcg": {
                arm: float(np.mean([per[q] for q in queries if q in per]))
                for arm, per in per_query_by_arm.items()
            },
        }
        for name, queries in category_buckets(categories, min_queries=min_queries).items()
    ]


def _stratum_name(low: float, high: float) -> str:
    """"[0.50, 0.90)", or for the closed top edge "[0.90, 1.00]" / "= 1.00"."""
    if high > 1.0:
        return "= 1.00" if low >= 1.0 else f"[{low:.2f}, 1.00]"
    return f"[{low:.2f}, {high:.2f})"


def coverage_strata(
    coverage: Mapping[str, float],
    *,
    edges: Sequence[float] = (0.0, 0.5, 0.9, 1.0, 1.0001),
) -> dict[str, list[str]]:
    """Queries bucketed by the share of their candidates carrying an image.

    The last default stratum is exactly 1.0 - every candidate imaged - and it
    stands apart on purpose. has_image_vector is a weak ranker in its own
    right (CLAUDE.md: presence flags alone are worth +0.0084), and it can only
    rank *within* a query whose candidates differ in it. Where all of them
    carry an image it is constant, so Ablation 3's delta there is the image
    signal alone.
    """
    names = [_stratum_name(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
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
        single_stage_orderings,
    )
    from src.feature_matrix import select_columns
    from src.metrics import ndcg_per_query
    from src.rank_report import qrels_from_frame
    from src.ranker import EARLY_STOP_FOLD, TRAIN_FOLDS, folds, predict_run, train_ranker
    from src.stage_signals import SCOPES, load_signals, scope_windows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="fold0", choices=list(SCOPES))
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--products", type=Path, default=Path("data/combined/products.parquet"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    signals = load_signals(args.scope, args.features_dir)
    ws, matrix = scope_windows(args.scope, features_dir=args.features_dir, seed=args.seed)
    qrels = qrels_from_frame(matrix)
    orderings = single_stage_orderings(signals)
    per_arm = {arm: per_query_ndcg(orderings[arm], ws, qrels) for arm in SINGLE_STAGES}

    # --- per category --------------------------------------------------------
    products = pd.read_parquet(args.products, columns=["product_id", "s_category"])
    judged = matrix[["query_id", "product_id"]].astype(str)
    categories = query_categories(judged, products)
    rows = category_breakdown(per_arm, categories)
    # A query none of whose judged products carries s_category has no row.
    # Counted, not dropped silently.
    n_uncategorised = len(qrels) - len(categories)
    # Whether the LLM's gain is uniform across categories needs an interval
    # per row, over exactly the queries the row's n counts.
    llm_delta = {q: per_arm["stage2+llm"][q] - per_arm["stage2"][q] for q in per_arm["stage2"]}
    llm_by_category = {
        row["stratum"]: row
        for row in delta_by_stratum(llm_delta, category_buckets(categories), seed=args.seed)
    }
    for row in rows:
        row["llm_minus_stage2"] = llm_by_category[row["category"]]["delta"]
        row["llm_minus_stage2_significant"] = llm_by_category[row["category"]]["significant"]
    print(f"{len(rows) - 1} categories with >= {MIN_CATEGORY_QUERIES} queries, "
          f"plus {OTHER}; {n_uncategorised} queries have no category")
    for row in rows:
        cells = "  ".join(f"{arm} {row['ndcg'][arm]:.4f}" for arm in SINGLE_STAGES)
        d = row["llm_minus_stage2"]
        print(f"  {row['category'][:32]:34s} n={row['n_queries']:5,}  {cells}  "
              f"llm-s2 {d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]")

    # --- where images help vs. hurt -----------------------------------------
    # Asked of Ablation 3's own arms, not of image *presence*, which is itself
    # a weak ranker (CLAUDE.md: presence flags alone are worth +0.0084).
    train = pd.read_parquet(args.features_dir / "train.parquet")
    # Plan 5's two fits: folds 2/3/4 when fold 0 is reported, every train fold
    # when test is. Fitting the narrow set for test lands 0.0020 low - more
    # than Ablation 4's whole honest effect (CLAUDE.md).
    fit = train if args.scope == "test-sample" else folds(train, TRAIN_FOLDS)
    early = folds(train, [EARLY_STOP_FOLD])
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
    overall = sum(image_delta.values()) / len(image_delta)
    print(f"  {'all':16s} n={len(image_delta):5,}  {overall:+.4f}  "
          "(Ablation 3 on this surface; must match its committed number)")

    # --- the LLM's failure population ---------------------------------------
    window_size = signals.groupby("query_id").size()
    attributes = {
        "n_candidates": {str(k): float(v) for k, v in
                         matrix.groupby("query_id").size().items()},
        "window_size": {str(k): float(v) for k, v in window_size.items()},
        "image_coverage": coverage,
        "stage2_ndcg": per_arm["stage2"],
        # Short queries leave the LLM little to reason from.
        "query_words": {str(k): float(v) for k, v in
                        matrix.groupby("query_id")["query_words"].first().items()},
        # A window of all-Exact (or all-Irrelevant) products cannot be
        # reordered for better or worse; the share says how much room there is.
        "window_exact_share": {
            str(k): float(v) for k, v in
            signals.assign(_e=(signals["label_code"] == 3).astype(float))
            .groupby("query_id")["_e"].mean().items()
        },
    }
    profile = failure_profile(
        per_arm["stage2+llm"], per_arm["stage2"], attributes, seed=args.seed
    )
    print(f"\n  -- the LLM arm against Stage 2, per query --")
    n = profile["n_queries"]
    print(f"  better {profile['n_better']:,} ({profile['n_better'] / n:.1%}), "
          f"worse {profile['n_worse']:,} ({profile['n_worse'] / n:.1%}), "
          f"same {profile['n_same']:,} ({profile['n_same'] / n:.1%})")
    print(f"  mean gain when better {profile['mean_gain_when_better']:+.4f}, "
          f"mean loss when worse {profile['mean_loss_when_worse']:+.4f}")
    print(f"  {'attribute (mean)':20s} {'better':>8s} {'worse':>8s} {'same':>8s}")
    for name, values in profile["attributes"].items():
        print(f"  {name:20s} " + " ".join(f"{values[k]:8.3f}" for k in ("better", "worse", "same")))

    payload = {
        "scope": args.scope,
        "n_queries": len(qrels),
        "min_category_queries": MIN_CATEGORY_QUERIES,
        "n_queries_without_category": n_uncategorised,
        "categories": rows,
        "image_coverage_strata": image_rows,
        "image_delta_overall": overall,
        "llm_failure_profile": profile,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
