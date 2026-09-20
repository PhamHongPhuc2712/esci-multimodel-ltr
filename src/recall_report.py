"""Ablations 1 and 2: the recall table this plan exists to produce.

Ablation 1 - raw vs. LLM-rewritten query, on the fused run.
Ablation 2 - dense-only vs. +BM25 vs. +image, all fused with RRF.

Every arm carries its own denominator, because a query with no relevant
product is skipped and two arms that skipped different queries are not
comparable. Every comparison is a paired bootstrap over queries, via
src.bootstrap, which pairs on the query id rather than on position.

Selection happens on the frozen validation folds from Plan 1. `--split test`
exists, but running it before the folds have settled the configuration is
exactly the thing CLAUDE.md's evaluation discipline forbids.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.bootstrap import bootstrap_ci, paired_delta_ci
from src.recall import DEFAULT_KS, RecallResult, recall_table

DEFAULT_OUT = Path("docs/results/recall.json")

# Measured on 2026-09-20. PROJECT_SPEC.md §9 budgets "<1 GB" for embeddings;
# the three channels together do not fit, and that is a reportable fact rather
# than a reason to shrink a channel.
STORAGE_GB = {"bm25": 1.10, "dense": 0.93, "image": 0.91}
SPEC_BUDGET_GB = 1.0


@dataclass(frozen=True)
class ArmResult:
    name: str
    table: dict[int, RecallResult]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "recall": {str(k): r.to_dict() for k, r in sorted(self.table.items())},
        }


def evaluate_arm(
    name: str,
    retrieved: Mapping[int, Sequence[str]],
    relevant: Mapping[int, set[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> ArmResult:
    """Recall@k for one configuration."""
    return ArmResult(name=name, table=recall_table(retrieved, relevant, ks=ks))


def _interval(interval) -> dict:
    """An src.bootstrap.Interval as plain JSON."""
    return {"point": interval.point, "low": interval.low, "high": interval.high}


def compare(arm: ArmResult, baseline: ArmResult, k: int = 100) -> dict:
    """Paired bootstrap delta of `arm` against `baseline` at k."""
    if k not in arm.table:
        raise KeyError(f"arm {arm.name!r} has no recall at k={k}")
    if k not in baseline.table:
        raise KeyError(f"arm {baseline.name!r} has no recall at k={k}")

    delta = paired_delta_ci(arm.table[k].per_query, baseline.table[k].per_query)
    return {
        "arm": arm.name,
        "baseline": baseline.name,
        "k": k,
        "delta": _interval(delta),
        "significant": not (delta.low <= 0.0 <= delta.high),
    }


def format_table(arms: Sequence[ArmResult]) -> str:
    """A markdown table of every arm at every k."""
    if not arms:
        return "(no arms)"
    ks = sorted(arms[0].table)
    header = "| arm | " + " | ".join(f"R@{k}" for k in ks) + " | queries |"
    rule = "|---" * (len(ks) + 2) + "|"
    lines = [header, rule]
    for arm in arms:
        cells = []
        for k in ks:
            mean = arm.table[k].mean
            cells.append("nan" if math.isnan(mean) else f"{mean:.4f}")
        n = arm.table[ks[-1]].n_queries
        lines.append(f"| {arm.name} | " + " | ".join(cells) + f" | {n:,} |")
    return "\n".join(lines)


def _main() -> int:
    import pandas as pd

    from src.bm25_index import open_channel as open_bm25
    from src.dense_embed import open_channel as open_dense
    from src.image_channel import open_channel as open_image
    from src.query_rewrite import RewriteCache
    from src.recall import DEFAULT_JUDGEMENTS, load_queries, relevant_sets
    from src.rrf import DEFAULT_K, fuse_batches

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--folds", default="0", help="comma-separated, or 'all'")
    parser.add_argument("--relevance", default="E", choices=["E", "E+S"])
    parser.add_argument("--k", type=int, default=1000, help="depth to retrieve")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_K)
    parser.add_argument("--image-scope", default="catalogue")
    parser.add_argument("--rewrites", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    folds = None if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    queries = load_queries(args.split, folds=folds)
    if args.sample is not None and args.sample < len(queries):
        queries = queries.sample(n=args.sample, random_state=args.seed)
    judgements = pd.read_parquet(DEFAULT_JUDGEMENTS)
    judgements = judgements.loc[
        judgements["query_id"].isin(set(queries["query_id"]))
    ]
    relevant = relevant_sets(judgements, relevance=args.relevance)
    query_ids = queries["query_id"].tolist()
    raw = queries["query"].astype(str).tolist()
    print(f"{len(raw):,} queries from {args.split} folds {args.folds}; "
          f"{len(relevant):,} have a relevant product under {args.relevance}")

    channels = {
        "bm25": open_bm25(),
        "dense": open_dense(),
        "image": open_image(scope=args.image_scope),
    }
    hits = {}
    for name, channel in channels.items():
        hits[name] = channel.search(raw, args.k)
        print(f"  {name}: searched")

    def as_map(ranked: Sequence[Sequence[str]]) -> dict[int, Sequence[str]]:
        return dict(zip(query_ids, ranked))

    arms = [
        evaluate_arm("bm25", as_map(hits["bm25"]), relevant),
        evaluate_arm("dense", as_map(hits["dense"]), relevant),
        evaluate_arm("image", as_map(hits["image"]), relevant),
    ]

    # --- Ablation 2: dense-only vs. +BM25 vs. +image -----------------------
    combos = [
        ("dense+bm25", ["dense", "bm25"]),
        ("dense+bm25+image", ["dense", "bm25", "image"]),
    ]
    for name, members in combos:
        fused = fuse_batches(
            [hits[m] for m in members], k=args.rrf_k, limit=args.k
        )
        arms.append(evaluate_arm(name, as_map(fused), relevant))

    # --- Ablation 1: raw vs. rewritten query -------------------------------
    ablation1 = None
    if args.rewrites is not None:
        cache = RewriteCache(args.rewrites)
        rewritten = [cache.get(q) or q for q in raw]
        changed = sum(1 for a, b in zip(raw, rewritten) if a != b)
        print(f"  rewrites: {changed:,}/{len(raw):,} queries changed")
        rewritten_hits = {
            name: channel.search(rewritten, args.k)
            for name, channel in channels.items()
        }
        fused = fuse_batches(
            [rewritten_hits[m] for m in ["dense", "bm25", "image"]],
            k=args.rrf_k,
            limit=args.k,
        )
        arm = evaluate_arm("rewritten (dense+bm25+image)", as_map(fused), relevant)
        raw_arm = next(a for a in arms if a.name == "dense+bm25+image")
        arms.append(arm)
        ablation1 = compare(arm, raw_arm, k=100)

    print()
    print(format_table(arms))

    baseline = next(a for a in arms if a.name == "bm25")
    comparisons = [
        compare(arm, baseline, k=100) for arm in arms if arm is not baseline
    ]
    intervals = {
        arm.name: _interval(bootstrap_ci(arm.table[100].per_query))
        for arm in arms
        if 100 in arm.table and arm.table[100].per_query
    }
    print()
    for row in comparisons:
        flag = "significant" if row["significant"] else "ties"
        d = row["delta"]
        print(
            f"  {row['arm']} vs {row['baseline']} @100: "
            f"{d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  {flag}"
        )

    payload = {
        "split": args.split,
        "folds": args.folds,
        "relevance": args.relevance,
        "n_queries": len(raw),
        "retrieval_depth": args.k,
        "rrf_k": args.rrf_k,
        "bm25_baseline_r100": 0.5018,
        "arms": [arm.to_dict() for arm in arms],
        "comparisons": comparisons,
        "recall_at_100_intervals": intervals,
        "ablation_1_rewrite": ablation1,
        "storage_gb": STORAGE_GB
        | {
            "total": round(sum(STORAGE_GB.values()), 2),
            "spec_budget": SPEC_BUDGET_GB,
            "over_budget": round(sum(STORAGE_GB.values()) - SPEC_BUDGET_GB, 2),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
