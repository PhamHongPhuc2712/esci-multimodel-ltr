"""Stage 4 against every stage alone - PROJECT_SPEC.md §4.4.

The spec asks for "a weighted combination, or a small learned combiner over
stage scores" and for the comparison "against each single stage alone". That
last clause is the whole test: a blend that beats Stage 2 while losing to the
LLM it contains has earned nothing, so every blend arm is compared against the
**best single stage**, not just against the coarse ranker.

The oracle - best arm per query, chosen with the labels - is reported beside
them. It is not a method and cannot be one. It is here so that a losing blend
reads as "the strategies missed 0.0262 of available headroom" rather than
"there was nothing to find".

Measured before this module existed, on fold 0: fixed-weight fusion 0.8800 at
its best swept weights and a cross-fitted combiner 0.8793, against the LLM's
own 0.8814 and an oracle of 0.9076. The expected result is a loss. It is
written down as one.

Fold-0 numbers for the learned arms are **cross-fitted**, because fold 0 is the
only query set with both an LLM ordering and a label, so it is both the only
training set and the reporting surface. Every such number is labelled.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.blend import ARM_COLUMNS, orderings_from_scores
from src.metrics import ndcg_per_query
from src.rank_report import compare, evaluate_arm, format_table, qrels_from_frame
from src.rerank_window import spliced_run

DEFAULT_OUT = Path("docs/results/blend.json")

# PROJECT_SPEC.md §5. Plan 5 matched it at 0.8579; Plan 6's LLM arm reached
# 0.8855 on the 2,000-query sample.
ESCI_BASELINE = 0.8562

SINGLE_STAGES: tuple[str, ...] = ("stage2", "stage2+ce", "stage2+llm")
BLEND_ARMS: tuple[str, ...] = ("blend_fixed", "blend_combiner", "blend_selector")
ORACLE = "oracle_best_of_three"

_STAGE_OF_ARM = {"stage2": "stage2", "stage2+ce": "ce", "stage2+llm": "llm"}


def single_stage_orderings(frame: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Each single stage's own ordering of every window."""
    out: dict[str, dict[str, list[str]]] = {}
    for arm, stage in _STAGE_OF_ARM.items():
        column, ascending = ARM_COLUMNS[stage]
        out[arm] = orderings_from_scores(frame, column, ascending=ascending)
    return out


def per_query_ndcg(
    orderings: Mapping[str, Sequence[str]],
    windows_: Sequence,
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, float]:
    """NDCG per query for one ordering of the windows.

    Goes through `spliced_run`, which refuses anything that is not a
    permutation of the window and keeps the untouched tail strictly below it.
    """
    return ndcg_per_query(spliced_run(windows_, dict(orderings)), qrels)


def best_single(results: Sequence) -> object:
    """The strongest single-stage arm - the baseline §4.4 actually asks for."""
    singles = [arm for arm in results if arm.name in SINGLE_STAGES]
    if not singles:
        raise ValueError(
            "no single stage in the results; the blend arms need one to be "
            "compared against, or a blend that merely beats Stage 2 looks like "
            "a win"
        )
    return max(singles, key=lambda arm: arm.ndcg.point)


def selector_routes(
    routed: Mapping[str, Sequence[str]],
    arm_orderings: Mapping[str, Mapping[str, Sequence[str]]],
    per_query_by_stage: Mapping[str, Mapping[str, float]],
    *,
    default: str,
) -> dict:
    """Where the selector sent each query, and what leaving `default` cost.

    A query counts as routed to `default` whenever its ordering equals the
    default arm's, whichever arm the selector named: identical lists score
    identically, so that is not a decision. The rest are deviations, and
    their deltas against `default` say whether the selector knew better.
    """
    order = [default, *[arm for arm in arm_orderings if arm != default]]
    counts = {arm: 0 for arm in arm_orderings}
    deltas: list[float] = []
    for query_id, ordering in routed.items():
        chosen = next(
            arm for arm in order if list(arm_orderings[arm][query_id]) == list(ordering)
        )
        counts[chosen] += 1
        if chosen != default:
            deltas.append(
                float(per_query_by_stage[chosen][query_id])
                - float(per_query_by_stage[default][query_id])
            )
    n = max(len(routed), 1)
    return {
        "default_arm": default,
        "share": {arm: counts[arm] / n for arm in counts},
        "n_deviations": len(deltas),
        "n_better": sum(d > 0 for d in deltas),
        "n_worse": sum(d < 0 for d in deltas),
        "n_same": sum(d == 0 for d in deltas),
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
    }


def _check_windows_match_signals(windows_, signals: pd.DataFrame) -> None:
    """The signal frame holds window documents only; NDCG needs the tail too.

    So the windows are rebuilt through src.stage_signals.scope_windows and
    checked against the frame - if Stage 2's parquet were ever re-dumped under
    a different fit, the two would disagree about which ten documents Stage 3
    ranked and every number here would be measured over a candidate set no
    arm ever saw.
    """
    from_windows = {(w.query_id, d) for w in windows_ for d in w.window}
    from_signals = set(
        zip(signals["query_id"].astype(str), signals["product_id"].astype(str))
    )
    if from_windows != from_signals:
        raise ValueError(
            f"the rebuilt windows and the signal frame disagree about "
            f"{len(from_windows ^ from_signals):,} (query, document) pairs; "
            "re-run python -m src.stage_signals for this scope"
        )


def _main() -> int:
    from src.blend import (
        DEFAULT_WEIGHTS,
        cross_fit_predict,
        cross_fit_select,
        fixed_weight_ordering,
        oracle_ordering,
        predict_combiner,
        selector_ordering,
        train_combiner,
        train_selector,
    )
    from src.fine_rank_report import check_same_queries
    from src.floor import random_floor
    from src.stage_signals import SCOPES, load_signals, scope_windows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="fold0", choices=list(SCOPES))
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--final", action="store_true",
                        help="required with either test scope")
    args = parser.parse_args()

    if args.scope != "fold0" and not args.final:
        raise SystemExit(
            "refusing to touch the test split without --final. The weights, "
            "the features and the round count are all chosen on fold 0."
        )

    signals = load_signals(args.scope, args.features_dir)
    ws, matrix = scope_windows(
        args.scope, features_dir=args.features_dir, seed=args.seed
    )
    _check_windows_match_signals(ws, signals)

    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    print(f"{args.scope}: {len(matrix):,} judgements over {len(qrels):,} queries; "
          f"floor {floor.mean:.4f}")

    orderings: dict[str, dict[str, list[str]]] = single_stage_orderings(signals)
    per_query: dict[str, dict[str, float]] = {}

    # --- the three single stages --------------------------------------------
    for arm in SINGLE_STAGES:
        per_query[arm] = per_query_ndcg(orderings[arm], ws, qrels)

    # --- Stage 4, arm 1: fixed weights --------------------------------------
    orderings["blend_fixed"] = fixed_weight_ordering(signals, DEFAULT_WEIGHTS)

    # --- Stage 4, arm 2: the learned combiner -------------------------------
    # Fold 0 is cross-fitted because it is both the only training set and the
    # reporting surface; the test sample is predicted by a model fitted on all
    # of fold 0, which it has never seen.
    if args.scope == "fold0":
        blend_scores = cross_fit_predict(signals, seed=args.seed)
        fitted_on = "fold 0, cross-fitted"
    else:
        booster = train_combiner(load_signals("fold0", args.features_dir), seed=args.seed)
        blend_scores = predict_combiner(booster, signals)
        fitted_on = "fold 0"
    orderings["blend_combiner"] = orderings_from_scores(
        signals.assign(_blend=blend_scores), "_blend"
    )

    # --- Stage 4, arm 3: the per-query selector -----------------------------
    # Its target is which arm scored best per query - the label in another
    # form - so fold 0 is cross-fitted over the combiner's own partition. The
    # test sample is routed by a selector fitted on all of fold 0.
    arm_orderings = {_STAGE_OF_ARM[arm]: orderings[arm] for arm in SINGLE_STAGES}
    if args.scope == "fold0":
        orderings["blend_selector"] = cross_fit_select(
            signals,
            {_STAGE_OF_ARM[arm]: per_query[arm] for arm in SINGLE_STAGES},
            arm_orderings,
            seed=args.seed,
        )
        selector_fitted_on = "fold 0, cross-fitted"
    else:
        train_signals = load_signals("fold0", args.features_dir)
        train_ws, train_matrix = scope_windows(
            "fold0", features_dir=args.features_dir, seed=args.seed
        )
        train_qrels = qrels_from_frame(train_matrix)
        train_orderings = single_stage_orderings(train_signals)
        train_per_arm = {
            _STAGE_OF_ARM[arm]: per_query_ndcg(train_orderings[arm], train_ws, train_qrels)
            for arm in SINGLE_STAGES
        }
        selector = train_selector(train_signals, train_per_arm, seed=args.seed)
        orderings["blend_selector"] = selector_ordering(selector, signals, arm_orderings)
        selector_fitted_on = "fold 0"

    # --- the ceiling ---------------------------------------------------------
    orderings[ORACLE] = oracle_ordering(
        {_STAGE_OF_ARM[arm]: per_query[arm] for arm in SINGLE_STAGES},
        {_STAGE_OF_ARM[arm]: orderings[arm] for arm in SINGLE_STAGES},
    )

    for arm in (*BLEND_ARMS, ORACLE):
        per_query[arm] = per_query_ndcg(orderings[arm], ws, qrels)

    # Why the selector ends where it does: how often it leaves the strongest
    # single stage, and whether those departures pay.
    routes = selector_routes(
        orderings["blend_selector"],
        arm_orderings,
        {_STAGE_OF_ARM[arm]: per_query[arm] for arm in SINGLE_STAGES},
        default=_STAGE_OF_ARM[max(SINGLE_STAGES, key=lambda a: np.mean(list(per_query[a].values())))],
    )

    results = [
        evaluate_arm(
            arm, per_query[arm], floor.per_query, groups=("blend",),
            n_features=0, objective="blend", best_iteration=0, seed=args.seed,
        )
        for arm in (*SINGLE_STAGES, *BLEND_ARMS, ORACLE)
    ]
    check_same_queries(results)
    for arm in results:
        print(f"  {arm.name:24s} {arm.ndcg.point:.4f}")
    print()
    print(format_table(results))

    # §4.4's actual question: does Stage 4 beat the best stage it contains?
    baseline = best_single(results)
    by_name = {arm.name: arm for arm in results}
    against_best = [
        compare(by_name[arm], baseline, seed=args.seed)
        for arm in (*BLEND_ARMS, ORACLE)
    ]
    print(f"\n  -- Stage 4 against the best single stage ({baseline.name}) --")
    for row in against_best:
        d = row["delta"]
        verdict = "significant" if row["significant"] else "ties"
        print(f"  {row['arm']:24s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]  {verdict}")

    against_stage2 = [
        compare(by_name[arm], by_name["stage2"], seed=args.seed)
        for arm in (*SINGLE_STAGES[1:], *BLEND_ARMS, ORACLE)
    ]

    shares = ", ".join(f"{arm} {share:.1%}" for arm, share in routes["share"].items())
    print(f"\n  -- the selector's routes: {shares} --")
    print(f"  left {routes['default_arm']} on {routes['n_deviations']:,} queries: better "
          f"{routes['n_better']:,}, worse {routes['n_worse']:,}, same {routes['n_same']:,}, "
          f"mean {routes['mean_delta']:+.4f}")

    wins = [r for r in against_best if r["arm"] in BLEND_ARMS
            and r["significant"] and r["delta"]["point"] > 0]
    if not wins:
        print(
            f"\n  No Stage 4 strategy beats {baseline.name}. The oracle reaches "
            f"{by_name[ORACLE].ndcg.point:.4f}, so the headroom is real and "
            "these strategies do not reach it. Reported as measured."
        )

    payload = {
        "scope": args.scope,
        "n_queries": len(qrels),
        "n_judgements": len(matrix),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "esci_baseline_target": ESCI_BASELINE,
        "combiner_fitted_on": fitted_on,
        "selector_fitted_on": selector_fitted_on,
        "selector_routes": routes,
        "fixed_weights": dict(DEFAULT_WEIGHTS),
        # 1:1:4 is the best of a sweep run on fold 0 itself, so on fold 0 this
        # arm is tuned in-sample - an optimistic number that still lost.
        "fixed_weights_chosen_on": (
            "fold 0 sweep, in-sample on this surface"
            if args.scope == "fold0" else "fold 0 sweep"
        ),
        "best_single_stage": baseline.name,
        "arms": [arm.to_dict() for arm in results],
        "against_best_single": against_best,
        "against_stage2": against_stage2,
        "any_blend_beats_best_single": bool(wins),
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
