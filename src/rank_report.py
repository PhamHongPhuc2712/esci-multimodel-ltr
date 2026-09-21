"""Ablations 3, 4, 5 and 7: the table Plan 5 exists to produce.

  3 - text-only vs. text+image features          (the multimodal contribution)
  4 - with vs. without behavioural features       (three arms, not two)
  5 - pointwise vs. lambdarank                    (the LTR objective's value)
  7 - learned fusion vs. one fixed global weight  (the learned-combination gap)

Every arm carries the computed random floor beside it, because
PROJECT_SPEC.md §2 is explicit that a headline NDCG means nothing without one.
Every comparison is a paired bootstrap over queries via src.bootstrap, which
pairs on the query id rather than on position.

Every number is scored by src.metrics. None comes from LightGBM: its ndcg@k
differs from this project's full-list NDCG by up to 6.4 points on the same
booster, for two independent reasons - see src/ranker.py.

**Ablation 4's arithmetic.** CLAUDE.md states the rule twice and the two are
not the same sum. "the honest contribution is arm 2 - arm 3" reads literally
as one NDCG minus another, which is roughly +0.09: arm 2 is a real ranker and
arm 3 is barely above the floor. The neighbouring sentence - "the ~+0.0084
must be subtracted from any behavioural-feature result" - describes the
coherent version, which is what `honest_behavioural_delta` computes:

    naive     = arm2 - arm1            text+behavioural vs text
    artefact  = arm3 - floor           the ESCI-S presence pattern alone
    honest    = naive - artefact

All three are reported. The subtraction assumes the effects are additive, and
CLAUDE.md measured them as super-additive (+0.0166 together against +0.0052 +
+0.0084 apart), so the honest figure is a point estimate under an assumption
rather than a bound. The caveat travels with the number in the JSON.

Selection happens on folds 1-4. `--split test` exists and requires `--final`,
because running it before the configuration has settled is exactly the thing
CLAUDE.md's evaluation discipline forbids.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.bootstrap import Interval, bootstrap_ci, paired_delta_ci
from src.feature_matrix import FEATURE_GROUPS

DEFAULT_OUT = Path("docs/results/coarse-rank.json")

# PROJECT_SPEC.md §5: the cross-encoder fine-tuned on SQD train. The project's
# stated realistic goal.
ESCI_BASELINE = 0.8562

ARMS: dict[str, tuple[str, ...]] = {
    # Ablations 3 and 4's arm 1. ESCI's own text and presence, plus the
    # retrieval scores. No ESCI-S values, no ESCI-S indicators, no image.
    "text": ("text", "esci_indicators", "retrieval"),
    # Ablation 3's treatment arm. The indicator travels with the score.
    "text+image": ("text", "esci_indicators", "retrieval", "image"),
    # Ablation 4's arm 2: values AND their missingness indicators.
    "text+behavioural": (
        "text",
        "esci_indicators",
        "retrieval",
        "behavioural",
        "categorical",
        "attrs",
        "s_indicators",
    ),
    # Ablation 4's arm 3. The artefact on its own.
    "indicators_only": ("s_indicators",),
    # Ablation 4's clean control, added during execution. Same base as arm 2
    # but WITHOUT the ESCI-S values, so (arm2 - this) isolates the values with
    # no additivity assumption at all - see honest_behavioural_delta.
    "text+indicators": ("text", "esci_indicators", "retrieval", "s_indicators"),
    # Not an ablation arm - a cross-check against the presence-flag ceilings
    # CLAUDE.md records at +0.0052, +0.0084 and +0.0166 over the floor.
    "esci_presence_only": ("esci_indicators",),
    "both_presence_only": ("esci_indicators", "s_indicators"),
    # The headline, and Ablation 5's feature set.
    "full": tuple(FEATURE_GROUPS),
}

# Ablation 7's learned arm fuses exactly what the fixed weight fuses. Handing
# LambdaMART all 47 features and calling the difference "learned fusion" would
# measure the other 45.
FUSION_FEATURES: tuple[str, ...] = ("dense_sim", "clip_image_sim", "has_image_vector")

ADDITIVITY_CAVEAT = (
    "Subtracting the indicators-only arm assumes the presence effects are "
    "additive. CLAUDE.md measured them as super-additive (+0.0166 for both "
    "presence sets together against +0.0052 + +0.0084 apart), so this is a "
    "point estimate under that assumption, not a bound."
)


def _interval(interval: Interval) -> dict:
    return {"point": interval.point, "low": interval.low, "high": interval.high}


@dataclass(frozen=True)
class ArmResult:
    name: str
    groups: tuple[str, ...]
    n_features: int
    objective: str
    best_iteration: int
    n_queries: int
    ndcg: Interval
    lift_over_floor: Interval
    per_query: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "groups": list(self.groups),
            "n_features": self.n_features,
            "objective": self.objective,
            "best_iteration": self.best_iteration,
            "n_queries": self.n_queries,
            "ndcg": _interval(self.ndcg),
            "lift_over_floor": _interval(self.lift_over_floor),
        }


def qrels_from_frame(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Nested qrels from a feature matrix's own `qrel` column."""
    duplicated = frame.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = frame.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}); building a "
            "dict would silently keep only the last one"
        )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, product_id, qrel in zip(
        frame["query_id"], frame["product_id"], frame["qrel"]
    ):
        qrels.setdefault(str(query_id), {})[str(product_id)] = int(qrel)
    return qrels


def evaluate_arm(
    name: str,
    per_query: Mapping[str, float],
    floor_per_query: Mapping[str, float],
    *,
    groups: Sequence[str],
    n_features: int,
    objective: str,
    best_iteration: int,
    seed: int = 0,
) -> ArmResult:
    """Wrap one arm's per-query NDCG with its interval and its lift."""
    return ArmResult(
        name=name,
        groups=tuple(groups),
        n_features=n_features,
        objective=objective,
        best_iteration=best_iteration,
        n_queries=len(per_query),
        ndcg=bootstrap_ci(per_query, seed=seed),
        lift_over_floor=paired_delta_ci(per_query, floor_per_query, seed=seed),
        per_query=dict(per_query),
    )


def compare(arm: ArmResult, baseline: ArmResult, seed: int = 0) -> dict:
    """Paired bootstrap delta of `arm` against `baseline`."""
    if set(arm.per_query) != set(baseline.per_query):
        raise ValueError(
            f"{arm.name} and {baseline.name} must cover the same queries; "
            f"{len(set(arm.per_query) - set(baseline.per_query))} differ"
        )
    delta = paired_delta_ci(arm.per_query, baseline.per_query, seed=seed)
    return {
        "arm": arm.name,
        "baseline": baseline.name,
        "delta": _interval(delta),
        "significant": not (delta.low <= 0.0 <= delta.high),
    }


def check_ablation_4(arm_names: Sequence[str]) -> None:
    """Refuse to report Ablation 4 without its third arm.

    The failure mode is not computing the subtraction wrong, it is quietly not
    computing it and crediting the behavioural features with an artefact worth
    roughly twice Plan 4's entire image contribution.
    """
    required = {"text", "text+behavioural", "indicators_only"}
    missing = required - set(arm_names)
    if missing:
        raise ValueError(
            f"Ablation 4 needs three arms and {sorted(missing)} are missing. "
            "ESCI-S presence alone ranks at +0.0084 over the floor and is not "
            "available at serving time, so it must be measured and subtracted, "
            "not assumed away."
        )


def honest_behavioural_delta(
    values: Mapping[str, float],
    indicators: Mapping[str, float],
    text: Mapping[str, float],
    floor: Mapping[str, float],
    *,
    text_indicators: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict:
    """Ablation 4's three numbers, each with a paired interval.

    honest = (arm2 - arm1) - (arm3 - floor), computed per query so the
    interval is a real paired bootstrap rather than two intervals subtracted.
    """
    naive = paired_delta_ci(values, text, seed=seed)
    artefact = paired_delta_ci(indicators, floor, seed=seed)
    # mean(a2 + floor) - mean(a1 + a3) == (a2 - a1) - (a3 - floor)
    left = {q: values[q] + floor[q] for q in values}
    right = {q: text[q] + indicators[q] for q in values}
    honest = paired_delta_ci(left, right, seed=seed)
    out = {
        "naive_delta": _interval(naive),
        "artefact_lift": _interval(artefact),
        "honest_delta": _interval(honest),
        "significant": not (honest.low <= 0.0 <= honest.high),
        "caveat": ADDITIVITY_CAVEAT,
    }
    if text_indicators is not None:
        # The assumption-free reading: both arms carry the same text base AND
        # the same ESCI-S presence flags, so the difference is the ESCI-S
        # *values* alone. Nothing is subtracted, so nothing has to be additive.
        clean = paired_delta_ci(values, text_indicators, seed=seed)
        out["values_over_indicators"] = _interval(clean)
        out["values_over_indicators_significant"] = not (
            clean.low <= 0.0 <= clean.high
        )
    return out


def format_table(arms: Sequence[ArmResult]) -> str:
    """A markdown table that cannot be printed without the floor."""
    if not arms:
        return "(no arms)"
    lines = [
        "| arm | objective | features | NDCG | 95% CI | lift over floor |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        lift = arm.lift_over_floor
        lines.append(
            f"| {arm.name} | {arm.objective} | {arm.n_features} | "
            f"{arm.ndcg.point:.4f} | [{arm.ndcg.low:.4f}, {arm.ndcg.high:.4f}] | "
            f"{lift.point:+.4f} [{lift.low:+.4f}, {lift.high:+.4f}] |"
        )
    return "\n".join(lines)


def _main() -> int:
    from src.feature_matrix import select_columns
    from src.fixed_weight import (
        best_weight,
        fit_normaliser,
        per_query_scores,
        sweep,
    )
    from src.floor import random_floor
    from src.metrics import ndcg_per_query
    from src.ranker import (
        EARLY_STOP_FOLD,
        OBJECTIVES,
        REPORT_FOLD,
        TRAIN_FOLDS,
        folds,
        predict_run,
        train_ranker,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--final",
        action="store_true",
        help="required with --split test; the test split is predicted ONCE, "
        "after the configuration is frozen",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        raise SystemExit(
            "refusing to touch the test split without --final. Every weight, "
            "round count and feature set is chosen on folds 1-4; test is "
            "measured once, at the end."
        )

    train = pd.read_parquet(args.features_dir / "train.parquet")
    fit = folds(train, TRAIN_FOLDS)
    early = folds(train, [EARLY_STOP_FOLD])
    if args.split == "test":
        # The one test run: every train fold for gradient, the same early-stop
        # fold so the round count is comparable, test for reporting.
        fit = train
        report = pd.read_parquet(args.features_dir / "test.parquet")
    else:
        report = folds(train, [REPORT_FOLD])

    qrels = qrels_from_frame(report)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    print(f"{len(report):,} judgements over {len(qrels):,} queries; "
          f"random floor {floor.mean:.4f} [{floor.low:.4f}, {floor.high:.4f}]")

    results: dict[str, ArmResult] = {}

    def run_arm(name: str, columns: Sequence[str], objective: str,
                groups: Sequence[str]) -> ArmResult:
        ranker = train_ranker(fit, early, list(columns), objective=objective,
                              seed=args.seed)
        per_query = ndcg_per_query(predict_run(ranker, report), qrels)
        arm = evaluate_arm(
            name, per_query, floor.per_query, groups=groups,
            n_features=len(columns), objective=objective,
            best_iteration=ranker.best_iteration, seed=args.seed,
        )
        print(f"  {name:22s} {objective:22s} {arm.ndcg.point:.4f} "
              f"({ranker.best_iteration} rounds, {len(columns)} features)")
        return arm

    print("\ntraining arms:")
    for name, groups in ARMS.items():
        results[name] = run_arm(name, select_columns(groups), "lambdarank", groups)

    # --- Ablation 5: the objective -----------------------------------------
    for objective in OBJECTIVES:
        if objective == "lambdarank":
            continue
        results[f"full/{objective}"] = run_arm(
            f"full/{objective}", select_columns(ARMS["full"]), objective, ARMS["full"]
        )

    # --- Ablation 7: learned fusion vs. one global weight -------------------
    results["learned_fusion"] = run_arm(
        "learned_fusion", FUSION_FEATURES, "lambdarank", ("retrieval", "image")
    )
    normalisers = {
        column: fit_normaliser(fit[column].to_numpy())
        for column in ("dense_sim", "clip_image_sim")
    }
    # Fold 1 in both cases: the weight is a tuned parameter, so it is never
    # chosen on the surface being reported.
    tuning = early
    swept = sweep(
        tuning, qrels_from_frame(tuning), text_column="dense_sim",
        image_column="clip_image_sim", normalisers=normalisers,
    )
    chosen_weight, tuning_ndcg = best_weight(swept)
    fixed_per_query = per_query_scores(
        report, qrels, text_column="dense_sim", image_column="clip_image_sim",
        normalisers=normalisers, weight=chosen_weight,
    )
    results["fixed_weight"] = evaluate_arm(
        "fixed_weight", fixed_per_query, floor.per_query,
        groups=("retrieval", "image"), n_features=2, objective="fixed_weight",
        best_iteration=0, seed=args.seed,
    )
    print(f"  fixed_weight swept on fold {EARLY_STOP_FOLD}: w_text="
          f"{chosen_weight:.2f} ({tuning_ndcg:.4f} there), "
          f"{results['fixed_weight'].ndcg.point:.4f} here")

    ordered = list(results.values())
    print()
    print(format_table(ordered))

    check_ablation_4(list(results))
    ablations = {
        "3_text_vs_image": compare(results["text+image"], results["text"]),
        "4_behavioural": honest_behavioural_delta(
            results["text+behavioural"].per_query,
            results["indicators_only"].per_query,
            results["text"].per_query,
            floor.per_query,
            text_indicators=results["text+indicators"].per_query,
            seed=args.seed,
        )
        | {
            "additivity_check": {
                "esci_presence_lift": results["esci_presence_only"].lift_over_floor.point,
                "s_presence_lift": results["indicators_only"].lift_over_floor.point,
                "both_presence_lift": results["both_presence_only"].lift_over_floor.point,
            }
        },
        "5_objective": [
            compare(results[f"full/{o}"], results["full"])
            for o in OBJECTIVES
            if o != "lambdarank"
        ],
        "7_learned_fusion": compare(results["learned_fusion"], results["fixed_weight"])
        | {"fixed_weight_w_text": chosen_weight, "sweep": swept},
    }

    print("\n  -- Ablation 3: text+image vs text --")
    d = ablations["3_text_vs_image"]["delta"]
    print(f"  {d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  "
          f"{'significant' if ablations['3_text_vs_image']['significant'] else 'ties'}")

    print("  -- Ablation 4: behavioural, three arms --")
    a4 = ablations["4_behavioural"]
    for key in ("naive_delta", "artefact_lift", "honest_delta",
                "values_over_indicators"):
        if key not in a4:
            continue
        v = a4[key]
        print(f"  {key:24s} {v['point']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}]")

    print("  -- Ablation 5: objective, against lambdarank --")
    for row in ablations["5_objective"]:
        d = row["delta"]
        print(f"  {row['arm']:26s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]  "
              f"{'significant' if row['significant'] else 'ties'}")

    print("  -- Ablation 7: learned fusion vs one global weight --")
    d = ablations["7_learned_fusion"]["delta"]
    print(f"  {d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  "
          f"{'significant' if ablations['7_learned_fusion']['significant'] else 'ties'}")

    # The presence-flag cross-check against CLAUDE.md's recorded ceilings.
    print("\n  -- presence-only cross-check (CLAUDE.md records "
          "+0.0052 / +0.0084 / +0.0166 on test) --")
    for name in ("esci_presence_only", "indicators_only", "both_presence_only"):
        print(f"  {name:22s} {results[name].lift_over_floor.point:+.4f}")

    headline = results["full"]
    print(f"\nheadline: {headline.ndcg.point:.4f} "
          f"[{headline.ndcg.low:.4f}, {headline.ndcg.high:.4f}] against a floor "
          f"of {floor.mean:.4f} and the ESCI_baseline target of {ESCI_BASELINE}")

    payload = {
        "split": args.split,
        "report_fold": None if args.split == "test" else REPORT_FOLD,
        "train_folds": list(TRAIN_FOLDS) if args.split == "train" else "all",
        "early_stop_fold": EARLY_STOP_FOLD,
        "n_queries": len(qrels),
        "n_judgements": len(report),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "esci_baseline_target": ESCI_BASELINE,
        "arms": [arm.to_dict() for arm in ordered],
        "ablations": ablations,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
