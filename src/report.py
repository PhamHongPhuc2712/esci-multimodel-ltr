"""Assemble an evaluation report: NDCG, its interval, and the floor it beats.

PROJECT_SPEC.md §2: "A headline number of 0.86 means nothing without that floor
attached." This module is the only place a headline NDCG is produced, and it
cannot produce one without the floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.bootstrap import Interval, bootstrap_ci, paired_delta_ci
from src.floor import random_floor
from src.metrics import Qrels, Run, ndcg_per_query


@dataclass(frozen=True)
class EvaluationReport:
    run_tag: str
    split: str
    n_queries: int
    ndcg: Interval
    floor_mean: float
    lift_over_floor: Interval
    n_floor_trials: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "run_tag": self.run_tag,
            "split": self.split,
            "n_queries": self.n_queries,
            "ndcg": {"point": self.ndcg.point, "low": self.ndcg.low, "high": self.ndcg.high},
            "floor_mean": self.floor_mean,
            "lift_over_floor": {
                "point": self.lift_over_floor.point,
                "low": self.lift_over_floor.low,
                "high": self.lift_over_floor.high,
            },
            "n_floor_trials": self.n_floor_trials,
            "seed": self.seed,
        }


def evaluate_run(
    run: Run,
    qrels: Qrels,
    *,
    run_tag: str,
    split: str,
    n_floor_trials: int = 100,
    seed: int = 0,
) -> EvaluationReport:
    """Score a run and pair it against a random ordering of the same queries."""
    per_query = ndcg_per_query(run, qrels)
    floor = random_floor(qrels, n_trials=n_floor_trials, seed=seed)
    return EvaluationReport(
        run_tag=run_tag,
        split=split,
        n_queries=len(per_query),
        ndcg=bootstrap_ci(per_query, seed=seed),
        floor_mean=floor.mean,
        lift_over_floor=paired_delta_ci(per_query, floor.per_query, seed=seed),
        n_floor_trials=n_floor_trials,
        seed=seed,
    )


def format_report(report: EvaluationReport) -> str:
    """A human-readable block that always carries the floor."""
    lift = report.lift_over_floor
    verdict = (
        "beats the floor"
        if lift.low > 0
        else "loses to the floor"
        if lift.high < 0
        else "ties the floor (interval contains zero)"
    )
    return (
        f"run:          {report.run_tag}\n"
        f"split:        {report.split} ({report.n_queries:,} queries)\n"
        f"NDCG:         {report.ndcg.point:.4f}  "
        f"95% CI [{report.ndcg.low:.4f}, {report.ndcg.high:.4f}]\n"
        f"random floor: {report.floor_mean:.4f}  "
        f"({report.n_floor_trials} trials, seed {report.seed})\n"
        f"lift:         {lift.point:+.4f}  "
        f"95% CI [{lift.low:+.4f}, {lift.high:+.4f}]  - {verdict}\n"
    )


def write_report(report: EvaluationReport, path: Path) -> None:
    """Persist a report as JSON so the ablation table can be rebuilt from disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
