"""Ablation 6: coarse-only vs. +cross-encoder vs. +LLM listwise.

PROJECT_SPEC.md §6 asks for NDCG, latency and cost across the three arms. This
module produces all three, plus two context arms: the zero-shot cross-encoder
(what the fine-tune bought) and the cascade (whether the two rerankers are
complementary, which Plan 7's blend stage will need).

Three things the report refuses to do:

  * **Quote one latency.** The cross-encoder scores 256 pairs a forward pass
    and the LLM cannot batch at all, so a single amortised figure flatters the
    cross-encoder by 3-5x on top of the real 20-600x gap. Both are recorded.
  * **Invent a price.** Tokens and seconds are measurable from here; dollars
    are not. `cost_usd` stays None unless the operator passes both rates.
  * **Compare arms over different query sets.** The LLM arm on test runs over a
    frozen 2,000-query sample; every other arm is restricted to that sample
    before anything is compared.

Selection happens on fold 0. `--split test` requires `--final`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.cross_encoder import Latency
from src.llm_rerank import Usage
from src.rank_report import compare, evaluate_arm, format_table, qrels_from_frame

DEFAULT_OUT = Path("docs/results/fine-rank.json")

# PROJECT_SPEC.md §5. Plan 5 already matched it at 0.8579 [0.8551, 0.8611].
ESCI_BASELINE = 0.8562

# All 8,956 test queries through the LLM is ~3.3 h and real money, so the
# three-way test comparison runs on a frozen sample. Every arm uses the same
# one, and `n` travels with every number.
TEST_SAMPLE = 2000

ARMS: tuple[str, ...] = (
    "stage2",
    "stage2+ce",
    "stage2+ce_bce",
    "stage2+ce_zeroshot",
    "stage2+llm",
    "stage2+ce+llm",
)


@dataclass(frozen=True)
class ArmCost:
    latency: Latency | None = None
    usage: Usage | None = None
    cost: float | None = None

    def to_dict(self) -> dict:
        return {
            "latency": self.latency.to_dict() if self.latency else None,
            "usage": self.usage.to_dict() if self.usage else None,
            "cost_usd": self.cost,
        }


def cost_usd(
    usage: Usage, *, price_in: float | None = None, price_out: float | None = None
) -> float | None:
    """Dollars, or None when either rate is unknown.

    Rates are per million tokens. Prices change and are not measurable from
    here; a fabricated number in a results file outlives the day it was right.
    """
    if price_in is None or price_out is None:
        return None
    return (
        usage.prompt_tokens / 1e6 * price_in
        + usage.completion_tokens / 1e6 * price_out
    )


def llm_latency(usage: Usage, n_windows: int) -> Latency | None:
    """Latency for an LLM arm, or None when the pass was served from cache.

    The first fold-0 report derived this from `usage.seconds`, which is wall
    clock for the whole pass. On a warm cache that is replay time: it published
    the LLM arm at **5.5 ms/query** when the uncached pass had measured
    **1,180 ms/query**, a 214x understatement in the one column Ablation 6
    exists to report. Latency is only a latency if calls were actually made.

    `single_ms` is the mean time inside one call - what a user waits.
    `batched_ms` is wall clock over the windows, which at concurrency 4 is the
    throughput figure. Both are None-or-both, never invented.
    """
    if usage.n_calls == 0 or n_windows <= 0:
        return None
    if usage.n_calls < 0.9 * n_windows:
        # A partly cached pass times neither thing: the wall clock covers
        # windows that cost nothing.
        return None
    return Latency(
        single_ms=usage.call_seconds * 1000 / usage.n_calls,
        batched_ms=usage.seconds * 1000 / n_windows,
        n_queries=n_windows,
        n_pairs=usage.n_calls,
    )


def check_same_queries(arms: Sequence) -> None:
    """Refuse a table whose arms cover different query sets."""
    if len(arms) < 2:
        return
    reference = set(arms[0].per_query)
    for arm in arms[1:]:
        if set(arm.per_query) != reference:
            raise ValueError(
                f"{arm.name!r} covers {len(arm.per_query):,} queries but "
                f"{arms[0].name!r} covers {len(reference):,}; arms must be "
                "scored on the same queries or the difference is not a "
                "reranker effect"
            )


def _main() -> int:
    import pandas as pd

    from src.cross_encoder import (
        DEFAULT_BACKBONE,
        DEFAULT_MODEL_DIR,
        text_maps,
        load_reranker,
        measure_latency,
        rerank,
    )
    from src.floor import random_floor
    from src.llm_rerank import (
        DEFAULT_CACHE,
        DEFAULT_CONCURRENCY,
        RerankCache,
        openai_call,
        rerank_windows,
    )
    from src.metrics import ndcg_per_query
    from src.ranker import REPORT_FOLD
    from src.rerank_window import DEFAULT_K, Window, spliced_run, stage2_run, windows
    from src.stage2_scores import load_stage2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--price-per-mtok-in", type=float, default=None)
    parser.add_argument("--price-per-mtok-out", type=float, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--final", action="store_true",
                        help="required with --split test")
    parser.add_argument(
        "--llm-latency-probe", type=int, default=0,
        help="measure LLM latency on N uncached windows (costs N API calls). "
             "A warm cache makes the arm's own pass unmeasurable, so this is "
             "the only honest source of the number once fold 0 is cached.")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        raise SystemExit(
            "refusing to touch the test split without --final. K, the loss, "
            "the backbone and the prompt are all chosen on folds 1-4."
        )

    matrix = pd.read_parquet(args.features_dir / f"{args.split}.parquet")
    if args.split == "train":
        matrix = matrix.loc[matrix["fold"] == REPORT_FOLD]
        label = f"fold {REPORT_FOLD}"
    else:
        keep = matrix["query_id"].drop_duplicates().sample(
            n=TEST_SAMPLE, random_state=args.seed
        )
        matrix = matrix.loc[matrix["query_id"].isin(set(keep))]
        label = f"test sample of {TEST_SAMPLE:,}"
    matrix = matrix.reset_index(drop=True)

    scores = load_stage2(args.split)
    joined = matrix[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    ws = windows(joined, k=args.k)
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    print(f"{label}: {len(matrix):,} judgements over {len(qrels):,} queries; "
          f"floor {floor.mean:.4f}")

    query_text, doc_text = text_maps(
        matrix,
        Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}

    results: list = []
    costs: dict[str, ArmCost] = {}

    def record(name: str, run: dict, cost: ArmCost) -> None:
        per_query = ndcg_per_query(run, qrels)
        arm = evaluate_arm(
            name, per_query, floor.per_query, groups=("retrieval",),
            n_features=0, objective="rerank", best_iteration=0, seed=args.seed,
        )
        results.append(arm)
        costs[name] = cost
        print(f"  {name:20s} {arm.ndcg.point:.4f}")

    record("stage2", stage2_run(ws), ArmCost())

    ce_orderings: dict[str, list[str]] = {}
    for name, path in [
        ("stage2+ce", args.model_dir / "lambda"),
        ("stage2+ce_bce", args.model_dir / "bce"),
        ("stage2+ce_zeroshot", DEFAULT_BACKBONE),
    ]:
        model = load_reranker(path)
        orderings = rerank(model, ws, query_text, doc_text)
        if name == "stage2+ce":
            ce_orderings = orderings
        record(
            name,
            spliced_run(ws, orderings),
            ArmCost(latency=measure_latency(model, ws, query_text, doc_text, n_queries=50)),
        )
        del model

    cache = RerankCache(args.cache)
    llm_orderings, usage = rerank_windows(
        ws, query_text, doc_text, call=openai_call(), cache=cache
    )
    record(
        "stage2+llm",
        spliced_run(ws, llm_orderings),
        ArmCost(
            latency=llm_latency(usage, len(ws)),
            usage=usage,
            cost=cost_usd(usage, price_in=args.price_per_mtok_in,
                          price_out=args.price_per_mtok_out),
        ),
    )
    print(f"    {usage.n_fallback:,} windows fell back to the Stage 2 order")

    # The cascade: the LLM re-ranks the cross-encoder's window order.
    cascade_windows = [
        Window(query_id=w.query_id,
               window=tuple(ce_orderings.get(w.query_id, w.window)),
               tail=w.tail)
        for w in ws
    ]
    cascade_orderings, cascade_usage = rerank_windows(
        cascade_windows, query_text, doc_text, call=openai_call(), cache=cache
    )
    record(
        "stage2+ce+llm",
        spliced_run(cascade_windows, cascade_orderings),
        ArmCost(latency=llm_latency(cascade_usage, len(ws)),
                usage=cascade_usage,
                cost=cost_usd(cascade_usage, price_in=args.price_per_mtok_in,
                              price_out=args.price_per_mtok_out)),
    )

    # A warm cache leaves the LLM arms with no measurable latency, so the
    # number comes from an explicit uncached probe or not at all.
    probe = None
    if args.llm_latency_probe > 0:
        probe_windows = [
            Window(query_id=w.query_id, window=w.window[::-1], tail=w.tail)
            for w in ws[: args.llm_latency_probe]
        ]
        _, probe_usage = rerank_windows(
            probe_windows, query_text, doc_text, call=openai_call(), cache=None
        )
        probe_latency = llm_latency(probe_usage, len(probe_windows))
        probe = {
            "n_windows": len(probe_windows),
            "concurrency": DEFAULT_CONCURRENCY,
            "latency": probe_latency.to_dict() if probe_latency else None,
            "usage": probe_usage.to_dict(),
        }
        if probe_latency:
            print(f"\n  -- LLM latency probe ({len(probe_windows)} uncached windows) --")
            print(f"  single {probe_latency.single_ms:.0f} ms/call, "
                  f"batched {probe_latency.batched_ms:.0f} ms/query "
                  f"at concurrency {DEFAULT_CONCURRENCY}")

    check_same_queries(results)
    print()
    print(format_table(results))

    baseline = results[0]
    comparisons = [compare(arm, baseline, seed=args.seed) for arm in results[1:]]
    print("\n  -- Ablation 6: each arm against coarse-only --")
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:20s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]  "
              f"{'significant' if row['significant'] else 'ties'}")

    print("\n  -- latency and cost --")
    for arm in results:
        c = costs[arm.name]
        if c.latency:
            print(f"  {arm.name:20s} single {c.latency.single_ms:8.1f} ms  "
                  f"batched {c.latency.batched_ms:7.1f} ms")
        if c.usage and not c.latency:
            print(f"  {arm.name:20s} latency null: {c.usage.n_cached:,} of "
                  f"{len(ws):,} windows came from cache, so the wall clock is "
                  "replay time, not latency")
        if c.usage:
            per = c.usage.per_query(len(ws))
            print(f"  {'':20s} {per['prompt_tokens']:.0f} prompt + "
                  f"{per['completion_tokens']:.0f} completion tok/query"
                  + (f", ${c.cost:.2f} total" if c.cost is not None else
                     ", cost_usd null (no rates given)"))

    payload = {
        "split": args.split,
        "scope": label,
        "k": args.k,
        "n_queries": len(qrels),
        "n_judgements": len(matrix),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "esci_baseline_target": ESCI_BASELINE,
        "arms": [arm.to_dict() | {"cost": costs[arm.name].to_dict()} for arm in results],
        "ablation_6": comparisons,
        "llm_latency_probe": probe,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
