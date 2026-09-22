"""Plan 5's ordering, persisted so both Stage 3 arms re-rank the same one.

Plan 5 reported NDCG but never wrote an ordering to disk: src/rank_report.py
trains eleven arms, prints a table and writes a JSON summary, and the per-pair
predictions live only inside that process. Both Stage 3 arms need *the same*
Stage 2 ordering, and re-deriving it in two modules is how two arms end up
re-ranking two different windows while the difference gets reported as a
reranker effect.

The configuration here is Plan 5's, frozen and unchanged: all 47 features,
`lambdarank`, `label_gain` derived from src.labels, early-stop on fold 1.
Nothing is re-tuned - if it were, Ablation 6 would be measuring a different
Stage 2 from the one Plan 5 reported at 0.8519 on fold 0 and 0.8579 on test.

**The fit set differs by split, because Plan 5's did.** src/rank_report.py
trains on folds 2/3/4 when it reports fold 0, and on *every* train fold when it
reports test (`fit = train`, with fold 1 still the early-stop set so the round
count stays comparable). Fitting folds 2/3/4 for both reproduces fold 0 exactly
and lands 0.0020 *below* Plan 5 on test, which would put Ablation 6 on a
weaker Stage 2 than the one Plan 5 published. So --split test fits everything
and --split train fits 2/3/4, matching each headline to the model that produced
it. Test is disjoint from every train fold, so the wider fit leaks nothing.

Folds 2/3/4 are scored too, and flagged `in_sample`. The model trained on them
so their scores are optimistic; nothing in Plan 6 consumes them (the
cross-encoder fine-tunes on raw judged pairs, not on windows, precisely to
avoid needing them). They are written rather than dropped because a flagged row
is recoverable and a missing one is indistinguishable from a bug.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_matrix import ALL_FEATURES
from src.ranker import (
    EARLY_STOP_FOLD,
    TRAIN_FOLDS,
    folds,
    predict,
    train_ranker,
)

DEFAULT_DIR = Path("data/features")

STAGE2_COLUMNS: tuple[str, ...] = (
    "query_id",
    "product_id",
    "stage2_score",
    "in_sample",
)


def score_split(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    fit_folds: Sequence[int] | None = TRAIN_FOLDS,
    seed: int = 0,
) -> pd.DataFrame:
    """Train the frozen Stage 2 configuration and score every row of `target`.

    `train` is the full train matrix. `fit_folds` selects the gradient set -
    TRAIN_FOLDS when fold 0 is the reporting surface, None for every fold when
    the target is the disjoint test split, which is what Plan 5 did. The
    early-stop fold is taken from src.ranker rather than from an argument, so
    it cannot drift from what Plan 5 reported.
    """
    columns = list(features) if features is not None else list(ALL_FEATURES)
    fit = train if fit_folds is None else folds(train, fit_folds)
    ranker = train_ranker(
        fit,
        folds(train, [EARLY_STOP_FOLD]),
        columns,
        seed=seed,
    )
    scores = predict(ranker, target)

    # Derived from the rows actually fitted, never from a separate argument: a
    # knob that can disagree with the gradient set defeats the whole point of
    # the flag.
    fitted_folds = set(fit["fold"]) if "fold" in fit.columns else set()
    if "fold" in target.columns:
        in_sample = target["fold"].isin(fitted_folds).to_numpy()
    else:
        in_sample = np.zeros(len(target), dtype=bool)

    return pd.DataFrame(
        {
            "query_id": target["query_id"].to_numpy(),
            "product_id": target["product_id"].to_numpy(),
            "stage2_score": scores.astype(np.float64),
            "in_sample": in_sample,
        }
    )


def load_stage2(split: str, directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Read a persisted Stage 2 ordering."""
    path = Path(directory) / f"stage2-{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no Stage 2 scores at {path}; run "
            f"python -m src.stage2_scores --split {split}"
        )
    return pd.read_parquet(path)


def require_out_of_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the Stage 2 model trained on, or raise if none are left.

    Folds 2/3/4 are in the model's weights; their Stage 2 ordering is better
    than it would be in production, and a reranker measured on top of it would
    be measured against an inflated baseline.
    """
    kept = frame.loc[~frame["in_sample"].astype(bool)].reset_index(drop=True)
    if kept.empty:
        raise ValueError(
            "every row is in-sample: the Stage 2 model trained on all of them, "
            "so nothing here can be used to measure a reranker. Score fold 0, "
            "fold 1 or the test split instead."
        )
    return kept


def _main() -> int:
    from src.metrics import ndcg_per_query
    from src.ranker import REPORT_FOLD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = pd.read_parquet(args.features_dir / "train.parquet")
    target = (
        train
        if args.split == "train"
        else pd.read_parquet(args.features_dir / "test.parquet")
    )
    print(f"scoring {len(target):,} rows of {args.split} with the frozen Stage 2 model")

    # Plan 5's own per-split fit; see the module docstring.
    scores = score_split(
        train,
        target,
        fit_folds=None if args.split == "test" else TRAIN_FOLDS,
        seed=args.seed,
    )

    # Reproduce Plan 5's headline as a check that nothing drifted. Fold 0 for
    # the train split, the whole thing for test.
    check = target.merge(scores, on=["query_id", "product_id"])
    if args.split == "train":
        check = check.loc[check["fold"] == REPORT_FOLD]
        expected, label = 0.8519, f"fold {REPORT_FOLD}"
    else:
        expected, label = 0.8579, "test"
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q, p, r, s in zip(
        check["query_id"], check["product_id"], check["qrel"], check["stage2_score"]
    ):
        qrels.setdefault(str(q), {})[str(p)] = int(r)
        run.setdefault(str(q), {})[str(p)] = float(s)
    per_query = ndcg_per_query(run, qrels)
    measured = sum(per_query.values()) / len(per_query)
    print(f"  {label} NDCG {measured:.4f} (Plan 5 reported {expected:.4f})")
    if abs(measured - expected) > 0.0005:
        print(
            f"  WARNING: {abs(measured - expected):.4f} away from Plan 5's number; "
            "the Stage 2 configuration has drifted and Ablation 6 would sit on "
            "a different baseline than the one Plan 5 reported"
        )

    path = args.features_dir / f"stage2-{args.split}.parquet"
    scores.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {len(scores):,} rows ({int(scores['in_sample'].sum()):,} in-sample) "
          f"to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
