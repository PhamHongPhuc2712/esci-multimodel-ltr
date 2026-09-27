"""Plan 8's report: can the LLM teach a cross-encoder, and what is free?

Three modes, one module, because all three score the same windows the same way:

  * ``--pilot``: fold 0, cross-fitted in halves. The same student is trained
    with the labels, the teacher's ordering and the hybrid target on the same
    queries, from two starting points, and the pre-registered gate on the paid
    teacher pass is decided (docs/results/distill-pilot.json).
  * default: every trained arm on fold 0, beside the three stages it sits
    between - Stage 2, the landed cross-encoder and the LLM
    (docs/results/distill.json).
  * ``--split test --final``: the same arms on all 8,956 test queries, once
    (docs/results/distill-test.json).

The reference arms are read from Plan 7's signal frame and checked against
the NDCG Plan 6 published for them, so every arm here is scored on exactly the
windows the published numbers came from.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

REFERENCE_ARMS: tuple[str, ...] = ("stage2", "stage2+ce", "stage2+llm")

MODEL_ARMS: dict[str, Path] = {
    "stage2+ce_windows": Path("models/cross-encoder/windows"),
    "stage2+ce_bge": Path("models/cross-encoder/bge"),
    "stage2+student": Path("models/cross-encoder/student"),
}
FREE_ARMS: tuple[str, ...] = ("stage2+ce_windows", "stage2+ce_bge")

TRAIN_COMMANDS: dict[str, str] = {
    "stage2+ce_windows": (
        "python -m src.distill --target labels --out models/cross-encoder/windows"
    ),
    "stage2+ce_bge": (
        "python -m src.distill --target labels --init BAAI/bge-reranker-base "
        "--batch-size 8 --out models/cross-encoder/bge"
    ),
    "stage2+student": (
        "python -m src.distill --from-gate docs/results/distill-pilot.json "
        "--out models/cross-encoder/student"
    ),
}

# The landed cross-encoder's own directory, loaded only to time it beside the
# new arms on the same machine state.
LANDED_MODEL = Path("models/cross-encoder/lambda")

# The pilot trains on ~2,065 queries a half. From the ms-marco checkpoint that
# is two epochs; from the landed arm, which has already seen folds 2/3/4, one.
PILOT_EPOCHS: dict[str, int] = {"scratch": 2, "landed": 1}

PUBLISHED_REFERENCE: dict[str, Path] = {
    "train": Path("docs/results/fine-rank.json"),
    "test": Path("docs/results/fine-rank-test-full.json"),
}
DEFAULT_OUT: dict[str, Path] = {
    "pilot": Path("docs/results/distill-pilot.json"),
    "train": Path("docs/results/distill.json"),
    "test": Path("docs/results/distill-test.json"),
}

GATE_RULE = (
    "Pay for the teacher's answers on folds 2/3/4 only if, in the fold-0 "
    "cross-fitted pilot, a teacher target (llm or hybrid) beats the labels "
    "target from the same starting point with a paired 95% interval wholly "
    "above zero. Written into Plan 8 before the committed pilot ran."
)


def pilot_arm(init: str, target: str) -> str:
    """The pilot arm's name: `pilot:<init>:<target>`."""
    return f"pilot:{init}:{target}"


def paid_run_gate(comparisons: Sequence[Mapping]) -> dict:
    """The pre-registered gate on Phase 2's paid pass.

    Only a teacher target against the labels target from the same init
    counts. A teacher target beating Stage 2, or beating a labels arm that
    started somewhere else, says nothing about whether the teacher is a
    better target than the labels. `choice` is the win with the largest
    lower bound, so Phase 3's student is not picked by hand.
    """
    wins = []
    for row in comparisons:
        arm = str(row["arm"]).split(":")
        baseline = str(row["baseline"]).split(":")
        if len(arm) != 3 or len(baseline) != 3:
            continue
        if arm[0] != "pilot" or baseline[0] != "pilot" or arm[1] != baseline[1]:
            continue
        if arm[2] not in ("llm", "hybrid") or baseline[2] != "labels":
            continue
        if row["delta"]["low"] > 0:
            wins.append((row["delta"]["low"], arm[1], arm[2], str(row["arm"])))
    wins.sort(reverse=True)
    return {
        "rule": GATE_RULE,
        "passed": bool(wins),
        "choice": {"init": wins[0][1], "target": wins[0][2]} if wins else None,
        "because": [name for *_, name in wins],
    }


def share_of_teacher_gain(arm: float, stage2: float, teacher: float) -> float:
    """How much of the teacher's gain over Stage 2 an arm keeps: 0 at Stage 2, 1 at the teacher."""
    if teacher <= stage2:
        raise ValueError(
            "the teacher does not beat Stage 2 on these queries, so it has no "
            "gain to share"
        )
    return (arm - stage2) / (teacher - stage2)


def resolve_model_arms(names: Sequence[str], root: Path = Path(".")) -> dict[str, Path]:
    """Model directory per requested arm; a missing one names the command that trains it."""
    out: dict[str, Path] = {}
    for name in names:
        if name not in MODEL_ARMS:
            raise ValueError(f"unknown arm {name!r}; expected one of {sorted(MODEL_ARMS)}")
        path = Path(root) / MODEL_ARMS[name]
        if not path.exists():
            raise FileNotFoundError(
                f"no model for {name} at {path}; train it with: {TRAIN_COMMANDS[name]}"
            )
        out[name] = path
    return out


def training_records(models: Mapping[str, Path]) -> dict[str, dict]:
    """What trained each model arm, from the record `python -m src.distill` writes beside it."""
    from src.distill import TRAINING_RECORD

    out: dict[str, dict] = {}
    for name, path in models.items():
        record = Path(path) / TRAINING_RECORD
        if not record.exists():
            raise FileNotFoundError(
                f"{name} at {path} has no {TRAINING_RECORD}, so nothing says what "
                f"trained it; retrain with: {TRAIN_COMMANDS[name]}"
            )
        out[name] = json.loads(record.read_text(encoding="utf-8"))
    return out


def check_split(split: str, final: bool) -> None:
    """The test split is touched once, behind --final."""
    if split == "test" and not final:
        raise SystemExit(
            "refusing to touch the test split without --final. The target, the "
            "starting point, the backbone and the epochs are all chosen on fold 0."
        )


def check_reference(
    measured: Mapping[str, float], published: Mapping[str, float], tol: float = 5e-4
) -> None:
    """Raise unless each reference arm reproduces the NDCG Plan 6 published.

    If Stage 2's parquet or the signal frame were ever re-dumped with a
    different fit, the arms here would be scored on windows the published
    numbers never saw, and every delta would be against the wrong baseline.
    """
    for arm in REFERENCE_ARMS:
        if arm not in published:
            raise KeyError(f"no published NDCG for {arm!r}")
        if abs(measured[arm] - published[arm]) > tol:
            raise ValueError(
                f"{arm} measures {measured[arm]:.4f} here against a published "
                f"{published[arm]:.4f}; the windows have drifted"
            )


def _published(split: str) -> dict[str, float]:
    payload = json.loads(PUBLISHED_REFERENCE[split].read_text(encoding="utf-8"))
    return {arm["name"]: arm["ndcg"]["point"] for arm in payload["arms"]}


def _scope(split: str, seed: int):
    """Windows, judged matrix, signal frame and text maps for fold 0 or the test split."""
    from src.cross_encoder import text_maps
    from src.stage_signals import load_signals, scope_windows

    scope = "fold0" if split == "train" else "test"
    windows_, matrix = scope_windows(scope, seed=seed)
    frame = load_signals(scope)
    query_text, doc_text = text_maps(
        matrix,
        Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}
    return scope, windows_, matrix, frame, query_text, doc_text


def _evaluate(name, per_query, floor, seed):
    from src.rank_report import evaluate_arm

    return evaluate_arm(
        name, per_query, floor.per_query, groups=("retrieval",),
        n_features=0, objective="rerank", best_iteration=0, seed=seed,
    )


def _pilot(args) -> int:
    from src.blend_report import per_query_ndcg, single_stage_orderings
    from src.cross_encoder import rerank
    from src.distill import (
        INITS, TARGETS, cross_fit_orderings, fine_tune_windows,
        gains_from_frame, split_halves, window_dataset,
    )
    from src.fine_rank_report import check_same_queries
    from src.floor import random_floor
    from src.llm_rerank import RerankCache
    from src.rank_report import compare, format_table, qrels_from_frame
    from src.stage_signals import check_fallback_share, llm_orderings_from_cache

    _, windows_, matrix, frame, query_text, doc_text = _scope("train", args.seed)
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    gains = gains_from_frame(matrix)

    # The teacher's real answers. The malformed one (query 55755) is missing
    # here rather than carried in Stage 2 order, and every target trains on
    # the same answered windows so the target is the only difference.
    teacher, missing = llm_orderings_from_cache(windows_, query_text, RerankCache(args.cache))
    check_fallback_share(missing, len(windows_))
    halves = split_halves([w.query_id for w in windows_], seed=args.seed)
    print(f"fold 0: {len(windows_):,} windows, halves of {len(halves[0]):,} and "
          f"{len(halves[1]):,}; {len(missing)} without a teacher answer {missing}")

    results = []
    reference = single_stage_orderings(frame)
    for name in REFERENCE_ARMS:
        results.append(_evaluate(name, per_query_ndcg(reference[name], windows_, qrels),
                                 floor, args.seed))

    def score(model, ws):
        return rerank(model, ws, query_text, doc_text)

    for init, path in INITS.items():
        for target in TARGETS:
            def fit(ws, init=init, path=path, target=target):
                answered = [w for w in ws if w.query_id in teacher]
                dataset, _ = window_dataset(
                    answered, query_text, doc_text,
                    kind=target, gains=gains, llm_orderings=teacher,
                )
                return fine_tune_windows(
                    dataset, init=path, epochs=PILOT_EPOCHS[init], seed=args.seed
                )

            orderings = cross_fit_orderings(windows_, halves, fit=fit, score=score)
            arm = _evaluate(pilot_arm(init, target),
                            per_query_ndcg(orderings, windows_, qrels), floor, args.seed)
            results.append(arm)
            print(f"  {arm.name:24s} {arm.ndcg.point:.4f}")

    check_same_queries(results)
    by_name = {arm.name: arm for arm in results}
    comparisons = [
        compare(by_name[pilot_arm(init, target)], by_name[pilot_arm(init, "labels")],
                seed=args.seed)
        for init in INITS for target in ("llm", "hybrid")
    ]
    gate = paid_run_gate(comparisons)

    print()
    print(format_table(results))
    print("\n  -- the teacher's target against the labels, same queries, same start --")
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:24s} {d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  "
              f"{'significant' if row['significant'] else 'ties'}")
    print(f"\n  gate: {'PASSED' if gate['passed'] else 'not passed'} - {gate['rule']}")

    payload = {
        "scope": "fold 0, cross-fitted in halves",
        "n_queries": len(qrels),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "halves": {"seed": args.seed, "sizes": [len(halves[0]), len(halves[1])]},
        "teacher_missing": sorted(missing),
        "config": {
            "loss": "RankNetLoss", "inits": INITS, "epochs": PILOT_EPOCHS,
            "targets": list(TARGETS),
        },
        "arms": [arm.to_dict() for arm in results],
        "comparisons": comparisons,
        "gate": gate,
        "seed": args.seed,
    }
    out = args.out or DEFAULT_OUT["pilot"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def _arms(args) -> int:
    from src.blend_report import per_query_ndcg, single_stage_orderings
    from src.cross_encoder import load_reranker, measure_latency, rerank
    from src.distill import gains_from_frame, pairwise_accuracy
    from src.fine_rank_report import check_same_queries
    from src.floor import random_floor
    from src.rank_report import compare, format_table, qrels_from_frame

    check_split(args.split, args.final)
    models = resolve_model_arms(args.arms)
    training = training_records(models)
    scope, windows_, matrix, frame, query_text, doc_text = _scope(args.split, args.seed)
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    gains = gains_from_frame(matrix)
    print(f"{scope}: {len(windows_):,} windows; floor {floor.mean:.4f}")

    results, latency, accuracy = [], {}, {}
    reference = single_stage_orderings(frame)
    for name in REFERENCE_ARMS:
        results.append(_evaluate(name, per_query_ndcg(reference[name], windows_, qrels),
                                 floor, args.seed))
        accuracy[name] = pairwise_accuracy(windows_, reference[name], gains)
    check_reference({arm.name: arm.ndcg.point for arm in results}, _published(args.split))

    landed = load_reranker(LANDED_MODEL)
    latency["stage2+ce"] = measure_latency(
        landed, windows_, query_text, doc_text, n_queries=50).to_dict()
    del landed

    for name, path in models.items():
        model = load_reranker(path)
        orderings = rerank(model, windows_, query_text, doc_text)
        results.append(_evaluate(name, per_query_ndcg(orderings, windows_, qrels),
                                 floor, args.seed))
        accuracy[name] = pairwise_accuracy(windows_, orderings, gains)
        latency[name] = measure_latency(
            model, windows_, query_text, doc_text, n_queries=50).to_dict()
        del model
        print(f"  {name:22s} {results[-1].ndcg.point:.4f}  "
              f"single {latency[name]['single_ms']:.1f} ms")

    check_same_queries(results)
    by_name = {arm.name: arm for arm in results}
    stage2, ce, llm = (by_name[name] for name in REFERENCE_ARMS)
    comparisons = [compare(arm, stage2, seed=args.seed) for arm in results[1:]]
    for name in models:
        comparisons.append(compare(by_name[name], ce, seed=args.seed))
        comparisons.append(compare(by_name[name], llm, seed=args.seed))
    shares = {
        name: share_of_teacher_gain(by_name[name].ndcg.point, stage2.ndcg.point,
                                    llm.ndcg.point)
        for name in ("stage2+ce", *models)
    }

    print()
    print(format_table(results))
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:22s} vs {row['baseline']:12s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]")

    payload = {
        "split": args.split,
        "scope": scope,
        "n_queries": len(qrels),
        "n_judgements": len(matrix),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "arms": [arm.to_dict() for arm in results],
        "comparisons": comparisons,
        "share_of_llm_gain": shares,
        "pairwise_accuracy": accuracy,
        "latency": latency,
        "models": {name: str(path) for name, path in models.items()},
        "training": training,
        "seed": args.seed,
    }
    out = args.out or DEFAULT_OUT[args.split]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def _main() -> int:
    from src.llm_rerank import DEFAULT_CACHE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="the fold-0 cross-fitted pilot and the gate")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--final", action="store_true", help="required with --split test")
    parser.add_argument("--arms", nargs="+", default=list(FREE_ARMS),
                        choices=list(MODEL_ARMS))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.pilot:
        if args.split != "train":
            raise SystemExit("the pilot trains on fold 0; it never touches the test split")
        return _pilot(args)
    return _arms(args)


if __name__ == "__main__":
    raise SystemExit(_main())
