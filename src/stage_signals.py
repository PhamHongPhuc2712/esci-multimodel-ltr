"""Every stage's opinion of every window document, in one frame.

PROJECT_SPEC.md §4.4 asks for "a small learned combiner over stage scores".
This module is the stage scores. Plan 6 produced orderings and NDCG; the
blend, the per-category analysis and the writeup all need the same per-document
signals, and building them three times is how three sections of one report come
to quote three different numbers.

Four signals per (query, window document):

    stage2_score   Plan 5's LambdaMART score, persisted by Plan 6
    stage2_rank    its 1-based position in the window
    ce_score       the fine-tuned cross-encoder's logit
    llm_rank       the LLM's 1-based position for that document

The LLM has no score - it returns a permutation - so rank is the only signal it
offers, and `llm_rr` is the reciprocal-rank form a fusion would want.

**A cache miss is counted, flagged and capped - never absorbed.**
data/llm-rerank.json holds a permutation for 4,129 of the 4,130 fold-0 windows
and 1,998 of the 2,000 test-sample windows. The three misses (fold-0 query
55755, test queries 49855 and 66028 - two TV-series titles and a book) are
windows whose LLM answer was malformed; malformed answers are deliberately not
cached, and fold 0's failed twice. Plan 6 scored all three in Stage 2 order
and counted them in `n_fallback`, so they carry that order here with
`llm_fallback = True`, which is what lets the frame reproduce Plan 6's
0.8814 and 0.8855 exactly.

What must never happen is the *scope* error: over the full 8,956-query test
split the lookup misses 6,956 windows, and quietly keeping the Stage 2 order
would label Stage 2's ordering `llm` and report the dilution as a blend
effect. So misses are returned rather than absorbed, `check_fallback_share`
refuses more than MAX_FALLBACK_SHARE of them, and `build_signals` raises for
any window that has neither an ordering nor a declared fallback.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import pandas as pd

# The same constant src.rrf uses, so a reciprocal rank means the same thing in
# Stage 1 and Stage 4.
from src.rrf import DEFAULT_K as RRF_K

DEFAULT_DIR = Path("data/features")

# The two query sets the LLM actually ran on. Anything else has no llm_rank.
SCOPES: tuple[str, ...] = ("fold0", "test-sample")

SIGNAL_COLUMNS: tuple[str, ...] = (
    "query_id",
    "product_id",
    "stage2_score",
    "stage2_rank",
    "ce_score",
    "llm_rank",
    "llm_rr",
    "llm_fallback",
    "label_code",
    "gain",
    "qrel",
)

# What the learned combiner is allowed to see. Deliberately not `gain`, `qrel`
# or `label_code` - those are the answer - and not `llm_fallback`, which marks
# the three windows the LLM malfunctioned on.
BLEND_FEATURES: tuple[str, ...] = (
    "stage2_score",
    "stage2_rank",
    "ce_score",
    "llm_rank",
    "llm_rr",
)

# Measured 2026-09-23: 1 of 4,130 fold-0 windows (0.02%) and 2 of 2,000
# test-sample windows (0.10%) have no cached permutation. The scope error this
# guards against - the full test split - misses 6,956 of 8,956 (77.7%). The
# ceiling sits 5x above the worst measured rate and >100x below a scope error.
MAX_FALLBACK_SHARE = 0.005


def scope_windows(
    scope: str,
    *,
    k: int | None = None,
    features_dir: Path = DEFAULT_DIR,
    seed: int = 0,
) -> tuple[list, pd.DataFrame]:
    """The windows Plan 6 re-ranked for `scope`, and the judged matrix behind them.

    The one place a scope becomes query ids. The blend report and the error
    analysis rebuild their windows through here rather than re-deriving the
    sample, so every reader of the signal frame sees the same candidate sets.
    The test sample is drawn exactly as src/fine_rank_report.py drew it; the
    data test pins that by reproducing Plan 6's per-arm NDCG from the frame.

    The returned matrix carries `stage2_score` for every judged pair.
    """
    from src.fine_rank_report import TEST_SAMPLE
    from src.ranker import REPORT_FOLD
    from src.rerank_window import DEFAULT_K, windows
    from src.stage2_scores import load_stage2

    if scope not in SCOPES:
        raise ValueError(
            f"unknown scope {scope!r}; expected one of {SCOPES}. The LLM ran on "
            "fold 0 and the 2,000-query test sample, nothing else."
        )
    k = DEFAULT_K if k is None else k
    features_dir = Path(features_dir)

    if scope == "fold0":
        matrix = pd.read_parquet(features_dir / "train.parquet")
        matrix = matrix.loc[matrix["fold"] == REPORT_FOLD]
        split = "train"
    else:
        matrix = pd.read_parquet(features_dir / "test.parquet")
        keep = matrix["query_id"].drop_duplicates().sample(
            n=TEST_SAMPLE, random_state=seed
        )
        matrix = matrix.loc[matrix["query_id"].isin(set(keep))]
        split = "test"
    matrix = matrix.reset_index(drop=True)

    stage2 = load_stage2(split, features_dir)[["query_id", "product_id", "stage2_score"]]
    matrix = matrix.merge(stage2, on=["query_id", "product_id"], how="left")
    if matrix["stage2_score"].isna().any():
        raise ValueError(
            f"{int(matrix['stage2_score'].isna().sum()):,} judged pairs have no "
            f"Stage 2 score; re-run python -m src.stage2_scores --split {split}"
        )
    return windows(matrix[["query_id", "product_id", "stage2_score"]], k=k), matrix


def llm_orderings_from_cache(
    windows_: Sequence, query_text: Mapping, cache
) -> tuple[dict[str, list[str]], list[str]]:
    """The LLM's ordering per window, plus the query ids it has no answer for.

    Misses are returned rather than absorbed: see the module docstring.
    """
    from src.llm_rerank import window_key

    orderings: dict[str, list[str]] = {}
    missing: list[str] = []
    for w in windows_:
        documents = list(w.window)
        permutation = cache.get(window_key(str(query_text[w.query_id]), documents))
        if permutation is None or sorted(permutation) != list(
            range(1, len(documents) + 1)
        ):
            missing.append(w.query_id)
            continue
        orderings[w.query_id] = [documents[i - 1] for i in permutation]
    return orderings, missing


def check_fallback_share(missing: Collection[str], n_windows: int) -> None:
    """Refuse a miss count that means the scope is wrong, not the LLM."""
    if n_windows <= 0:
        raise ValueError("no windows to check")
    share = len(missing) / n_windows
    if share > MAX_FALLBACK_SHARE:
        raise ValueError(
            f"{len(missing):,} of {n_windows:,} windows ({share:.1%}) have no "
            f"cached LLM permutation, past the {MAX_FALLBACK_SHARE:.1%} a "
            "handful of malformed answers explains. This is a scope error: the "
            "LLM ran on fold 0 and the 2,000-query test sample only, and this "
            "plan issues no API calls."
        )


def build_signals(
    windows_: Sequence,
    labels: pd.DataFrame,
    *,
    ce_scores: Mapping[tuple[str, str], float],
    llm_orderings: Mapping[str, Sequence[str]],
    llm_fallbacks: Collection[str] = (),
) -> pd.DataFrame:
    """One row per window document, carrying every stage's signal and the label.

    `labels` needs query_id, product_id, label_code, gain, qrel and
    stage2_score. The tail is excluded: no Stage 3 arm looked at it, so no
    stage has an opinion of it to blend, and including it would let the
    combiner reorder documents the window never contained.

    A window in `llm_fallbacks` keeps its Stage 2 order under `llm` and is
    flagged - how Plan 6 scored a malformed answer. A window in neither
    mapping raises.
    """
    if "stage2_score" not in labels.columns:
        raise KeyError(
            "labels has no stage2_score column; a blend feature that is "
            "silently NaN trains the combiner on nothing"
        )
    fallbacks = {str(q) for q in llm_fallbacks}
    both = fallbacks & set(llm_orderings)
    if both:
        raise ValueError(
            f"{sorted(both)[:3]} have an LLM ordering and a fallback flag; a "
            "window is one or the other"
        )

    by_pair = {
        (str(q), str(p)): (int(c), float(g), int(r), float(s))
        for q, p, c, g, r, s in zip(
            labels["query_id"], labels["product_id"], labels["label_code"],
            labels["gain"], labels["qrel"], labels["stage2_score"],
        )
    }

    rows: list[dict] = []
    for w in windows_:
        if w.query_id in fallbacks:
            order, fell_back = list(w.window), True
        elif w.query_id in llm_orderings:
            order, fell_back = list(llm_orderings[w.query_id]), False
            if sorted(order) != sorted(w.window):
                raise ValueError(
                    f"query {w.query_id}: the LLM ordering is not a permutation "
                    "of its window"
                )
        else:
            raise KeyError(
                f"no LLM ordering for query {w.query_id} and it is not a declared "
                "fallback; keeping the Stage 2 order here would label it `llm`"
            )
        llm_position = {doc: i for i, doc in enumerate(order, start=1)}

        for position, doc in enumerate(w.window, start=1):
            key = (w.query_id, doc)
            if key not in ce_scores:
                raise KeyError(
                    f"no cross-encoder score for {key}; a blend feature that "
                    "is silently NaN trains the combiner on nothing"
                )
            if key not in by_pair:
                raise KeyError(
                    f"no label for {key}; a window document with no judgement "
                    "cannot be trained on or scored"
                )
            code, gain, qrel, stage2_score = by_pair[key]
            rank = llm_position[doc]
            rows.append(
                {
                    "query_id": w.query_id,
                    "product_id": doc,
                    "stage2_score": stage2_score,
                    "stage2_rank": position,
                    "ce_score": float(ce_scores[key]),
                    "llm_rank": rank,
                    "llm_rr": 1.0 / (RRF_K + rank),
                    "llm_fallback": fell_back,
                    "label_code": code,
                    "gain": gain,
                    "qrel": qrel,
                }
            )
    return pd.DataFrame(rows, columns=list(SIGNAL_COLUMNS))


def require_full_coverage(frame: pd.DataFrame) -> None:
    """Refuse a frame where any stage has no opinion of some document."""
    incomplete = {
        column: int(frame[column].isna().sum())
        for column in BLEND_FEATURES
        if frame[column].isna().any()
    }
    if incomplete:
        raise ValueError(
            f"incomplete signals: {incomplete}. Every stage must have scored "
            "every window document, or the blend is comparing arms over "
            "different candidate sets."
        )


def load_signals(scope: str, directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Read a persisted signal frame."""
    path = Path(directory) / f"stage-signals-{scope}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no signals at {path}; run python -m src.stage_signals --scope {scope}"
        )
    return pd.read_parquet(path)


def _main() -> int:
    from src.cross_encoder import (
        DEFAULT_MODEL_DIR,
        load_reranker,
        text_maps,
        window_scores,
    )
    from src.llm_rerank import DEFAULT_CACHE, RerankCache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="fold0", choices=list(SCOPES))
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ws, matrix = scope_windows(
        args.scope, k=args.k, features_dir=args.features_dir, seed=args.seed
    )
    print(f"{args.scope}: {len(ws):,} windows over {matrix['query_id'].nunique():,} "
          f"queries, {sum(len(w.window) for w in ws):,} window documents")

    query_text, doc_text = text_maps(
        matrix,
        Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}

    llm_orderings, missing = llm_orderings_from_cache(
        ws, query_text, RerankCache(args.cache)
    )
    check_fallback_share(missing, len(ws))
    if missing:
        print(f"  {len(missing)} window(s) have no cached permutation and keep the "
              f"Stage 2 order, flagged llm_fallback, as Plan 6 scored them: {missing}")

    model = load_reranker(args.model_dir / "lambda", device=args.device)
    ce_scores = window_scores(model, ws, query_text, doc_text)
    del model

    frame = build_signals(
        ws, matrix, ce_scores=ce_scores, llm_orderings=llm_orderings,
        llm_fallbacks=missing,
    )
    require_full_coverage(frame)

    path = Path(args.features_dir) / f"stage-signals-{args.scope}.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {len(frame):,} rows over {frame['query_id'].nunique():,} queries to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
