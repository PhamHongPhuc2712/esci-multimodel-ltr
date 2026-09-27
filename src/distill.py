"""Training a cross-encoder on Stage 2's window, from the labels or from the LLM.

Plan 8 asks whether the LLM listwise arm's gain - +0.0276 NDCG on the full
test split, at 4.7 s a query - can be moved into a cross-encoder that answers
in about 25 ms. This module owns the three things distillation needs:

  * **the training windows**, carved by the out-of-fold Stage 2
    (src.stage2_scores.out_of_fold_scores). The in-sample ordering of folds
    2/3/4 scores 0.9035 against fold 0's 0.8519, and only 34.5% of its windows
    hold the same ten documents as the out-of-fold ones;
  * **the targets**, one per window document, higher meaning better: the
    labels, the teacher's ordering, or the labels with the teacher breaking
    ties inside a grade;
  * **the cross-fit** the fold-0 pilot needs, because fold 0 holds the only
    free teacher answers and is also the reporting surface.

A window the teacher did not answer is dropped from a teacher-target training
set, never filled with Stage 2's order. src.llm_rerank falls back to that order
on a malformed answer, and a student taught it would be learning Stage 2 under
the teacher's name.

Every window-trained arm uses one loss, RankNet - what Sun et al. distilled
RankGPT's permutations with - so the target is the only thing that differs
between them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.cross_encoder import (
    DEFAULT_BACKBONE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_DIR,
    _lookup,
)

TARGETS: tuple[str, ...] = ("labels", "llm", "hybrid")

# Where a fine-tune starts. "landed" is Plan 6's LambdaLoss arm, already
# trained on every judged pair of folds 2/3/4.
INITS: dict[str, str] = {
    "scratch": DEFAULT_BACKBONE,
    "landed": str(DEFAULT_MODEL_DIR / "lambda"),
}

WINDOW_BATCH_SIZE = 16
WINDOW_EPOCHS = 1
WINDOW_LEARNING_RATE = 2e-5

# Written beside every model `python -m src.distill` trains.
TRAINING_RECORD = "training.json"


def gains_from_frame(matrix: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(query_id, product_id) -> gain, with both ids as strings, as windows hold them."""
    return {
        (str(q), str(p)): float(g)
        for q, p, g in zip(matrix["query_id"], matrix["product_id"], matrix["gain"])
    }


def _gain(gains: Mapping, query_id: str, document: str) -> float:
    try:
        return float(gains[(query_id, document)])
    except KeyError:
        raise KeyError(
            f"no label for ({query_id!r}, {document!r}); a window document "
            "without a judgement cannot be trained on or scored"
        ) from None


def window_targets(
    window,
    kind: str,
    *,
    gains: Mapping,
    llm_ordering: Sequence[str] | None = None,
) -> list[float]:
    """One target per window document, in window order. Higher is better.

    `labels` is the gain. `llm` is n for the teacher's first choice down to 1
    for its last: the loss reads a higher label as more relevant, so a target
    built from the position itself would teach the student to invert the
    teacher. `hybrid` orders by grade first and lets the teacher order the
    documents inside a grade - the only place the labels are silent.
    """
    if kind not in TARGETS:
        raise ValueError(f"unknown target {kind!r}; expected one of {TARGETS}")
    documents = list(window.window)
    labels = [_gain(gains, window.query_id, d) for d in documents]
    if kind == "labels":
        return labels
    if llm_ordering is None:
        raise KeyError(
            f"query {window.query_id}: no teacher ordering for a {kind!r} "
            "target. Drop the window; never substitute Stage 2's order."
        )
    if sorted(llm_ordering) != sorted(documents):
        raise ValueError(
            f"query {window.query_id}: the teacher ordering is not a "
            "permutation of its window"
        )
    position = {d: i for i, d in enumerate(llm_ordering)}
    n = len(documents)
    if kind == "llm":
        return [float(n - position[d]) for d in documents]
    grade = dict(zip(documents, labels))
    order = sorted(documents, key=lambda d: (-grade[d], position[d]))
    rank = {d: i for i, d in enumerate(order)}
    return [float(n - rank[d]) for d in documents]


def window_dataset(
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    kind: str,
    gains: Mapping,
    llm_orderings: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, list], list[str]]:
    """(query, docs, labels) per window, and the query ids dropped for want of a teacher.

    A `labels` target drops nothing. A teacher target drops every window the
    teacher did not answer and says which - Review Focus 3.
    """
    if kind not in TARGETS:
        raise ValueError(f"unknown target {kind!r}; expected one of {TARGETS}")
    orderings = dict(llm_orderings or {})
    out: dict[str, list] = {"query": [], "docs": [], "labels": []}
    dropped: list[str] = []
    for w in windows_:
        ordering = orderings.get(w.query_id)
        if kind != "labels" and ordering is None:
            dropped.append(w.query_id)
            continue
        out["query"].append(_lookup(query_text, w.query_id, "query text"))
        out["docs"].append([_lookup(doc_text, d, "document text") for d in w.window])
        out["labels"].append(
            window_targets(w, kind, gains=gains, llm_ordering=ordering)
        )
    return out, dropped


def split_halves(
    query_ids: Iterable, *, seed: int = 0
) -> tuple[frozenset[str], frozenset[str]]:
    """Two disjoint halves of a query set, fixed by the seed alone."""
    ids = sorted({str(q) for q in query_ids})
    if len(ids) < 2:
        raise ValueError("a cross-fit needs at least two queries")
    shuffled = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    middle = len(ids) // 2
    return frozenset(shuffled[:middle]), frozenset(shuffled[middle:])


def cross_fit_orderings(
    windows_: Sequence,
    halves: tuple[Iterable[str], Iterable[str]],
    *,
    fit: Callable[[list], object],
    score: Callable[[object, list], Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    """Every window ordered by a model that never trained on it.

    Train on one half, order the other, swap. Fold 0 is the only train fold
    with free teacher answers and also the reporting surface, so the pilot
    trains there - and this is the guard that keeps it honest.
    """
    first, second = (frozenset(str(q) for q in half) for half in halves)
    shared = first & second
    if shared:
        raise ValueError(
            f"the halves share {len(shared)} queries; a query in both would "
            "be scored by a model that trained on it"
        )
    stray = {w.query_id for w in windows_} - (first | second)
    if stray:
        raise ValueError(
            f"{len(stray)} windows are in neither half, so no model would score them"
        )
    out: dict[str, list[str]] = {}
    for train_ids, score_ids in ((first, second), (second, first)):
        model = fit([w for w in windows_ if w.query_id in train_ids])
        scored = score(model, [w for w in windows_ if w.query_id in score_ids])
        outside = set(scored) - score_ids
        if outside:
            raise AssertionError(
                f"the scorer ordered {len(outside)} queries outside its half"
            )
        out.update({q: list(order) for q, order in scored.items()})
    return out


def pairwise_accuracy(
    windows_: Sequence, orderings: Mapping[str, Sequence[str]], gains: Mapping
) -> float:
    """Share of mixed-grade window pairs an ordering puts the right way round.

    Pairs inside one grade are skipped: NDCG does not care how they fall.
    Measured on fold 0 before this plan was written: the LLM 0.7404, the
    cross-encoder 0.6555, Stage 2 0.6373.
    """
    right = total = 0
    for w in windows_:
        if w.query_id not in orderings:
            raise KeyError(f"query {w.query_id}: no ordering to judge")
        position = {d: i for i, d in enumerate(orderings[w.query_id])}
        documents = list(w.window)
        for i, a in enumerate(documents):
            for b in documents[i + 1:]:
                grade_a = _gain(gains, w.query_id, a)
                grade_b = _gain(gains, w.query_id, b)
                if grade_a == grade_b:
                    continue
                better, worse = (a, b) if grade_a > grade_b else (b, a)
                right += position[better] < position[worse]
                total += 1
    if total == 0:
        raise ValueError("no mixed-grade pairs to judge")
    return right / total


def fine_tune_windows(
    dataset: Mapping[str, list],
    *,
    init: str = DEFAULT_BACKBONE,
    out_dir: Path | None = None,
    epochs: int = WINDOW_EPOCHS,
    batch_size: int = WINDOW_BATCH_SIZE,
    learning_rate: float = WINDOW_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "auto",
    seed: int = 0,
):
    """Fine-tune a cross-encoder with RankNet on (query, docs, labels) windows."""
    import tempfile

    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import RankNetLoss

    from src.clip_encoder import resolve_device

    if not dataset.get("query"):
        raise ValueError("an empty training set; nothing to fine-tune on")
    resolved = resolve_device(device)
    model = CrossEncoder(str(init), num_labels=1, device=resolved, max_length=max_length)
    with tempfile.TemporaryDirectory() as scratch:
        arguments = CrossEncoderTrainingArguments(
            output_dir=scratch,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=resolved.startswith("cuda"),
            save_strategy="no",
            logging_steps=200,
            report_to=[],
            seed=seed,
            disable_tqdm=True,
        )
        CrossEncoderTrainer(
            model=model,
            args=arguments,
            train_dataset=Dataset.from_dict(dict(dataset)),
            loss=RankNetLoss(model),
        ).train()
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
    return model


def training_windows(features_dir: Path = Path("data/features")) -> tuple[list, pd.DataFrame]:
    """The out-of-fold windows of folds 2/3/4, and the judged rows behind them."""
    from src.cross_encoder import check_training_folds
    from src.ranker import TRAIN_FOLDS
    from src.rerank_window import DEFAULT_K, windows
    from src.stage2_scores import OOF_SPLIT, load_stage2

    features_dir = Path(features_dir)
    matrix = pd.read_parquet(
        features_dir / "train.parquet",
        columns=["query_id", "product_id", "gain", "fold"],
    )
    matrix = matrix.loc[matrix["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    check_training_folds(matrix)
    scores = load_stage2(OOF_SPLIT, features_dir)
    joined = matrix[["query_id", "product_id"]].merge(
        scores[["query_id", "product_id", "stage2_score"]],
        on=["query_id", "product_id"],
    )
    if len(joined) != len(matrix):
        raise ValueError(
            f"the out-of-fold ordering covers {len(joined):,} of {len(matrix):,} "
            "training rows; re-run python -m src.stage2_scores --split train --out-of-fold"
        )
    return windows(joined, k=DEFAULT_K), matrix


def _main() -> int:
    import time

    from src.cross_encoder import text_maps
    from src.llm_rerank import DEFAULT_CACHE, RerankCache
    from src.stage_signals import check_fallback_share, llm_orderings_from_cache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    parser.add_argument(
        "--init", default="scratch",
        help=f"one of {sorted(INITS)}, or any cross-encoder name or path "
             "(the capacity arm passes BAAI/bge-reranker-base)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=WINDOW_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=WINDOW_BATCH_SIZE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--products", type=Path, default=Path("data/combined/products.parquet"))
    parser.add_argument("--judgements", type=Path, default=Path("data/combined/judgements.parquet"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    windows_, matrix = training_windows(args.features_dir)
    query_text, doc_text = text_maps(matrix, args.products, args.judgements)
    query_text = {str(k): v for k, v in query_text.items()}
    gains = gains_from_frame(matrix)

    orderings = None
    if args.target != "labels":
        orderings, missing = llm_orderings_from_cache(
            windows_, query_text, RerankCache(args.cache)
        )
        if len(missing) == len(windows_):
            raise SystemExit(
                "the cache holds no teacher answer for any training window; "
                "Plan 8's paid pass (Phase 2) has not run"
            )
        check_fallback_share(missing, len(windows_))

    dataset, dropped = window_dataset(
        windows_, query_text, doc_text,
        kind=args.target, gains=gains, llm_orderings=orderings,
    )
    print(f"{len(dataset['query']):,} training windows from folds 2/3/4, target "
          f"{args.target!r}, init {args.init!r}; {len(dropped):,} dropped for want "
          "of a teacher answer")
    started = time.time()
    fine_tune_windows(
        dataset, init=INITS.get(args.init, args.init), out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
    )
    minutes = (time.time() - started) / 60
    # What trained this model, beside it: the report records it with the
    # arm's numbers, so a results file says which target made which arm.
    (args.out / TRAINING_RECORD).write_text(json.dumps({
        "target": args.target, "init": args.init, "loss": "RankNetLoss",
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "n_windows": len(dataset["query"]), "n_dropped": len(dropped),
        "minutes": minutes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"trained in {minutes:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
