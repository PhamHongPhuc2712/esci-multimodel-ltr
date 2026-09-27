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
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_matrix import ALL_FEATURES
from src.ranker import (
    EARLY_STOP_FOLD,
    REPORT_FOLD,
    TRAIN_FOLDS,
    folds,
    predict,
    train_ranker,
)

DEFAULT_DIR = Path("data/features")

# Folds 2/3/4, each scored by a Stage 2 that never saw it. Read with
# load_stage2(OOF_SPLIT); written by `--split train --out-of-fold`.
OOF_SPLIT = "train-oof"

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


def out_of_fold_scores(
    train: pd.DataFrame,
    *,
    fit_folds: Sequence[int] = TRAIN_FOLDS,
    features: Sequence[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Each training fold scored by a Stage 2 fitted on the other training folds.

    A student fine-tuned on Stage 2's window has to see the window it will
    meet at test time, and the persisted ordering cannot supply it for folds
    2/3/4: the model trained on them. Measured 2026-09-27, the in-sample
    ordering scores 0.9035 on its own folds against fold 0's 0.8519, with an
    Exact on top of 87.7% of windows against 71.1%. Fitting on the other two
    folds lands at 0.8534 and 71.2% - fold 0's distribution - and only 34.5%
    of windows keep the same ten documents, so a teacher asked about the
    in-sample windows would be answering about the wrong products.

    Every fit early-stops on fold 1, as score_split's always does.
    """
    fit_folds = [int(f) for f in fit_folds]
    reserved = {REPORT_FOLD, EARLY_STOP_FOLD} & set(fit_folds)
    if reserved:
        raise ValueError(
            f"fold(s) {sorted(reserved)} cannot be scored out-of-fold: fold "
            f"{REPORT_FOLD} is the reporting surface and fold {EARLY_STOP_FOLD} "
            "the early-stop set, and neither may enter a fit"
        )
    if len(fit_folds) < 2:
        raise ValueError(
            f"out-of-fold scoring needs at least two training folds, got {fit_folds}"
        )
    parts = [
        score_split(
            train,
            folds(train, [fold]),
            features=features,
            fit_folds=[f for f in fit_folds if f != fold],
            seed=seed,
        )
        for fold in fit_folds
    ]
    scores = pd.concat(parts, ignore_index=True)
    if scores["in_sample"].any():
        raise AssertionError(
            "an out-of-fold row is flagged in-sample: a fold was scored by a "
            "model that trained on it"
        )
    return scores


def stage2_ndcg(joined: pd.DataFrame) -> float:
    """Mean full-list NDCG of the `stage2_score` ordering over `joined`'s queries.

    `joined` needs query_id, product_id, qrel and stage2_score.
    """
    from src.metrics import ndcg_per_query

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q, p, r, s in zip(
        joined["query_id"], joined["product_id"], joined["qrel"], joined["stage2_score"]
    ):
        qrels.setdefault(str(q), {})[str(p)] = int(r)
        run.setdefault(str(q), {})[str(p)] = float(s)
    per_query = ndcg_per_query(run, qrels)
    return sum(per_query.values()) / len(per_query)


def window_shift(
    matrix: pd.DataFrame,
    out_of_fold: pd.DataFrame,
    in_sample: pd.DataFrame,
    *,
    k: int | None = None,
) -> dict:
    """How far the in-sample windows of the training folds sit from out-of-fold ones.

    `matrix` needs query_id, product_id and qrel for exactly the rows both
    score frames cover. Measured 2026-09-27: NDCG 0.9035 in-sample against
    0.8534 out-of-fold, an Exact on top of 87.7% of windows against 71.2%, and
    34.5% of windows holding the same documents in both.
    """
    from src.labels import label_to_qrel
    from src.rerank_window import DEFAULT_K, windows

    k = DEFAULT_K if k is None else k
    exact = label_to_qrel("E")
    out: dict = {}
    carved: dict[str, dict[str, set]] = {}
    for name, scores in (("out_of_fold", out_of_fold), ("in_sample", in_sample)):
        joined = matrix.merge(
            scores[["query_id", "product_id", "stage2_score"]],
            on=["query_id", "product_id"],
        )
        if len(joined) != len(matrix):
            raise ValueError(
                f"the {name} scores cover {len(joined):,} of {len(matrix):,} rows"
            )
        qrel = {
            (str(q), str(p)): int(r)
            for q, p, r in zip(joined["query_id"], joined["product_id"], joined["qrel"])
        }
        ws = windows(joined[["query_id", "product_id", "stage2_score"]], k=k)
        carved[name] = {w.query_id: set(w.window) for w in ws}
        out[f"ndcg_{name}"] = stage2_ndcg(joined)
        out[f"exact_on_top_{name}"] = sum(
            qrel[(w.query_id, w.window[0])] == exact for w in ws
        ) / len(ws)
    same = sum(
        carved["in_sample"][q] == documents
        for q, documents in carved["out_of_fold"].items()
    )
    out["same_window_share"] = same / len(carved["out_of_fold"])
    out["n_queries"] = len(carved["out_of_fold"])
    out["k"] = k
    return out


def load_stage2(split: str, directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Read a persisted Stage 2 ordering."""
    path = Path(directory) / f"stage2-{split}.parquet"
    if not path.exists():
        command = (
            "--split train --out-of-fold" if split == OOF_SPLIT else f"--split {split}"
        )
        raise FileNotFoundError(
            f"no Stage 2 scores at {path}; run "
            f"python -m src.stage2_scores {command}"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-of-fold", action="store_true",
        help="score folds 2/3/4 each with a model fitted on the other two - the "
             "windows Plan 8's student trains on - and write stage2-train-oof.parquet")
    parser.add_argument(
        "--results-out", type=Path, default=Path("docs/results/stage2-oof.json"),
        help="where --out-of-fold commits its comparison with the in-sample ordering")
    args = parser.parse_args()

    train = pd.read_parquet(args.features_dir / "train.parquet")

    if args.out_of_fold:
        if args.split != "train":
            raise SystemExit("--out-of-fold scores the training folds; use --split train")
        print(f"scoring folds {list(TRAIN_FOLDS)} out-of-fold")
        scores = out_of_fold_scores(train, seed=args.seed)
        check = train.merge(scores, on=["query_id", "product_id"])
        measured = stage2_ndcg(check)
        # Measured 2026-09-27: 0.8534 here, fold 0 0.8519, and the in-sample
        # ordering of these same folds 0.9035.
        print(f"  folds {list(TRAIN_FOLDS)} out-of-fold NDCG {measured:.4f} "
              "(fold 0: 0.8519; the in-sample ordering of these folds: 0.9035)")
        if abs(measured - 0.8519) > 0.01:
            print("  WARNING: more than 0.01 from fold 0; these windows would not "
                  "look like the ones a student meets at test time")
        path = args.features_dir / f"stage2-{OOF_SPLIT}.parquet"
        scores.to_parquet(path, index=False, compression="zstd")
        print(f"wrote {len(scores):,} rows over {scores['query_id'].nunique():,} "
              f"queries to {path}")

        # The shift from the persisted, in-sample ordering of the same folds,
        # committed so the writeup can quote it.
        persisted = load_stage2("train", args.features_dir)
        rows = train.loc[train["fold"].isin(TRAIN_FOLDS)]
        shift = window_shift(rows, scores, persisted)
        fold0 = train.loc[train["fold"] == REPORT_FOLD].merge(
            persisted, on=["query_id", "product_id"]
        )
        record = {"fold0_ndcg": stage2_ndcg(fold0), **shift,
                  "train_folds": list(TRAIN_FOLDS), "seed": args.seed}
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"  in-sample {shift['ndcg_in_sample']:.4f}, Exact on top "
              f"{shift['exact_on_top_in_sample']:.1%} in-sample against "
              f"{shift['exact_on_top_out_of_fold']:.1%}; "
              f"{shift['same_window_share']:.1%} of windows unchanged -> {args.results_out}")
        return 0

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
    measured = stage2_ndcg(check)
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
