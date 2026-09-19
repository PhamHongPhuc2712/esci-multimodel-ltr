"""Command line entry point: python -m src.cli {evaluate,floor,freeze-splits}."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import load_split
from src.floor import random_floor
from src.report import evaluate_run, format_report, write_report
from src.runs import qrels_from_judgements, read_run
from src.splits import assign_folds, freeze_folds

RESULTS_DIR = Path("docs/results")


def _evaluate(args: argparse.Namespace) -> int:
    qrels = qrels_from_judgements(load_split(args.split).judgements)
    report = evaluate_run(
        read_run(args.run),
        qrels,
        run_tag=args.tag or Path(args.run).stem,
        split=args.split,
        n_floor_trials=args.floor_trials,
        seed=args.seed,
    )
    print(format_report(report))
    out = RESULTS_DIR / f"{report.run_tag}-{args.split}.json"
    write_report(report, out)
    print(f"written to {out}")
    return 0


def _floor(args: argparse.Namespace) -> int:
    qrels = qrels_from_judgements(load_split(args.split).judgements)
    result = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    print(
        f"random floor ({args.split}, {result.n_trials} trials, seed {result.seed}): "
        f"{result.mean:.4f}  [{result.low:.4f}, {result.high:.4f}]"
    )
    return 0


def _freeze_splits(args: argparse.Namespace) -> int:
    folds = assign_folds(load_split("train").judgements["query_id"].unique())
    digest = freeze_folds(folds)
    sizes = folds["fold"].value_counts().sort_index().to_dict()
    print(f"{len(folds):,} queries, fold sizes {sizes}")
    print(f"sha256 {digest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="src.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="score a TREC run file")
    evaluate.add_argument("--run", required=True, help="path to a TREC run file")
    evaluate.add_argument("--split", default="test", choices=["train", "test"])
    evaluate.add_argument("--tag", default=None, help="name for the results file")
    evaluate.add_argument("--floor-trials", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.set_defaults(handler=_evaluate)

    floor = sub.add_parser("floor", help="compute the random-ordering floor")
    floor.add_argument("--split", default="test", choices=["train", "test"])
    floor.add_argument("--floor-trials", type=int, default=100)
    floor.add_argument("--seed", type=int, default=0)
    floor.set_defaults(handler=_floor)

    freeze = sub.add_parser("freeze-splits", help="write splits/val_folds.csv")
    freeze.set_defaults(handler=_freeze_splits)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
