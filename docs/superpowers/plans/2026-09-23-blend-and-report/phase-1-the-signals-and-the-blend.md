# Phase 1 — The Signals and the Blend

**Plan 7 of 7 · Phase 1 of 3 · Tasks 1–2.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into.

**Delivers:** `src/stage_signals.py` — every stage's opinion of every window
document, persisted once — and `src/blend.py`, the three Stage 4 strategies
plus the oracle that bounds them.

**Needs on disk:** `data/features/{train,test}.parquet` and
`data/features/stage2-{train,test}.parquet` (Plans 5–6),
`models/cross-encoder/lambda/` (Plan 6), `data/llm-rerank.json` (Plan 6) and
`data/combined/products.parquet` (Plan 2). A GPU for Task 1's cross-encoder
pass.

**Owns Review Focus items 1 and 2** (the combiner trained on the queries it
reports; blending a stage that never ran on those queries).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **No new API calls.** Every LLM ordering is already cached; a miss is an error, never a fallback.
- **No query may be scored by a model that trained on it.**
- **Stage 4 orderings go through `spliced_run`**, which still refuses anything that is not a permutation of the window.
- **`label_gain` comes from `src.ranker.LABEL_GAIN`**, never LightGBM's default.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: Persist every stage's signal

**Files:**
- Create: `src/stage_signals.py`
- Create: `tests/test_stage_signals.py`
- Modify: `src/cross_encoder.py` (add `window_scores` beside `rerank`)

**Interfaces:**
- Consumes: `src.rerank_window.{Window, windows, DEFAULT_K}`, `src.stage2_scores.load_stage2`, `src.llm_rerank.{RerankCache, window_key}`, `src.cross_encoder.{load_reranker, text_maps, DEFAULT_MODEL_DIR}`.
- Produces:
  - `src.cross_encoder.window_scores(model, windows_, query_text, doc_text, *, batch_size=256) -> dict[tuple[str, str], float]`
  - `src.stage_signals.DEFAULT_DIR: Path` = `Path("data/features")`
  - `src.stage_signals.SCOPES: tuple[str, ...]` = `("fold0", "test-sample")`
  - `src.stage_signals.SIGNAL_COLUMNS: tuple[str, ...]`
  - `src.stage_signals.BLEND_FEATURES: tuple[str, ...]`
  - `src.stage_signals.llm_orderings_from_cache(windows_, query_text, cache) -> tuple[dict[str, list[str]], list[str]]`
  - `src.stage_signals.build_signals(windows_, labels, *, ce_scores, llm_orderings) -> pd.DataFrame`
  - `src.stage_signals.require_full_coverage(frame) -> None`
  - `src.stage_signals.load_signals(scope, directory=DEFAULT_DIR) -> pd.DataFrame`
  - CLI: `python -m src.stage_signals --scope {fold0,test-sample}` writing `data/features/stage-signals-<scope>.parquet`

**Review Focus 2 lives in `llm_orderings_from_cache` and
`require_full_coverage`.** `data/llm-rerank.json` holds permutations for the
4,130 fold-0 windows and the 2,000 test-sample windows — and nothing else. The
tempting shape is "look it up, and if it is missing keep the Stage 2 order",
which is exactly what `rerank_windows` does at runtime and exactly wrong here:
a signal frame built over the full 8,956-query test split would give 6,956
queries the Stage 2 order *labelled `llm`*, and Stage 4 would report the
dilution as a blend effect. So the lookup returns the misses and the caller
raises.

**Why the cross-encoder needs a new entry point.** Plan 6's `rerank` returns
orderings on purpose, so a logit can never land in a run beside a Stage 2
score (its Review Focus 1). A blend needs the logits themselves.
`window_scores` exposes them for exactly one consumer, and `rerank` is left
alone — in fact it becomes a thin wrapper over it, which also guarantees the
two can never disagree about a tie-break.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_signals.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.rerank_window import Window
from src.stage_signals import (
    BLEND_FEATURES,
    SCOPES,
    SIGNAL_COLUMNS,
    build_signals,
    llm_orderings_from_cache,
    require_full_coverage,
)


class FakeCache:
    def __init__(self, entries):
        self._entries = dict(entries)

    def get(self, key):
        return self._entries.get(key)


def _windows():
    return [
        Window(query_id="1", window=("a", "b", "c"), tail=("z",)),
        Window(query_id="2", window=("x", "y"), tail=()),
    ]


def _labels():
    return pd.DataFrame(
        {
            "query_id": ["1", "1", "1", "1", "2", "2"],
            "product_id": ["a", "b", "c", "z", "x", "y"],
            "label_code": [3, 0, 2, 1, 3, 0],
            "gain": [1.0, 0.0, 0.1, 0.01, 1.0, 0.0],
            "qrel": [100, 0, 10, 1, 100, 0],
        }
    )


def _ce():
    return {("1", "a"): 0.1, ("1", "b"): 0.9, ("1", "c"): 0.5,
            ("2", "x"): 0.2, ("2", "y"): 0.8}


def _llm():
    return {"1": ["c", "a", "b"], "2": ["y", "x"]}


# --- the frame --------------------------------------------------------------

def test_the_frame_has_one_row_per_window_document():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    assert len(frame) == 5          # 3 + 2 window documents; the tail is not scored
    assert list(frame.columns) == list(SIGNAL_COLUMNS)


def test_the_tail_is_not_in_the_frame():
    # Stage 3 never looked at it, so no stage has an opinion to blend.
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    assert "z" not in set(frame["product_id"])


def test_stage_2_rank_is_the_window_position():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    q1 = frame[frame["query_id"] == "1"].set_index("product_id")
    assert q1.loc["a", "stage2_rank"] == 1
    assert q1.loc["c", "stage2_rank"] == 3


def test_llm_rank_is_the_position_in_the_llm_ordering():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    q1 = frame[frame["query_id"] == "1"].set_index("product_id")
    assert q1.loc["c", "llm_rank"] == 1      # the LLM put c first
    assert q1.loc["b", "llm_rank"] == 3


def test_reciprocal_rank_falls_with_rank():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    q1 = frame[frame["query_id"] == "1"].set_index("product_id")
    assert q1.loc["c", "llm_rr"] > q1.loc["b", "llm_rr"]


def test_labels_travel_with_the_signals():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    q1 = frame[frame["query_id"] == "1"].set_index("product_id")
    assert q1.loc["a", "label_code"] == 3
    assert q1.loc["a", "qrel"] == 100


def test_every_blend_feature_is_a_column():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    assert set(BLEND_FEATURES) <= set(frame.columns)


def test_a_missing_cross_encoder_score_raises():
    with pytest.raises(KeyError, match="cross-encoder"):
        build_signals(_windows(), _labels(), ce_scores={}, llm_orderings=_llm())


def test_a_missing_label_raises():
    # A window document with no judgement would train the blend on a NaN.
    labels = _labels().drop(index=0)
    with pytest.raises(KeyError, match="label"):
        build_signals(_windows(), labels, ce_scores=_ce(), llm_orderings=_llm())


# --- Review Focus 2: the cache covers these windows, or it does not ---------

def test_cached_permutations_become_orderings():
    from src.llm_rerank import window_key

    query_text = {"1": "red shoes", "2": "blue hat"}
    cache = FakeCache({
        window_key("red shoes", ["a", "b", "c"]): [3, 1, 2],
        window_key("blue hat", ["x", "y"]): [2, 1],
    })
    orderings, missing = llm_orderings_from_cache(_windows(), query_text, cache)
    assert orderings["1"] == ["c", "a", "b"]
    assert missing == []


def test_a_cache_miss_is_reported_not_papered_over():
    # The cache covers fold 0 and the 2,000-query test sample, nothing else.
    # Falling back to the Stage 2 order here would label 6,956 test queries
    # `llm` and report the dilution as a blend effect.
    query_text = {"1": "red shoes", "2": "blue hat"}
    orderings, missing = llm_orderings_from_cache(_windows(), query_text, FakeCache({}))
    assert sorted(missing) == ["1", "2"]


def test_a_permutation_of_the_wrong_length_counts_as_a_miss():
    from src.llm_rerank import window_key

    query_text = {"1": "red shoes", "2": "blue hat"}
    cache = FakeCache({window_key("red shoes", ["a", "b", "c"]): [2, 1]})
    _, missing = llm_orderings_from_cache(_windows(), query_text, cache)
    assert "1" in missing


def test_require_full_coverage_passes_a_complete_frame():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    require_full_coverage(frame)      # must not raise


def test_require_full_coverage_rejects_a_null_signal():
    frame = build_signals(_windows(), _labels(), ce_scores=_ce(), llm_orderings=_llm())
    frame.loc[0, "ce_score"] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        require_full_coverage(frame)


def test_the_scopes_are_the_two_the_llm_actually_ran_on():
    assert SCOPES == ("fold0", "test-sample")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_stage_signals.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.stage_signals'`.

- [ ] **Step 3: Expose the cross-encoder's scores**

In `src/cross_encoder.py`, add `window_scores` immediately above `rerank`, and
rewrite `rerank` to call it so the two can never disagree:

```python
def window_scores(
    model,
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    batch_size: int = 256,
) -> dict[tuple[str, str], float]:
    """The raw logit for every window document, keyed (query_id, product_id).

    `rerank` returns orderings on purpose - a cross-encoder logit must never
    enter a run beside a Stage 2 score (see src/rerank_window.py). Stage 4
    blends the scores themselves, so it needs them, and it is the only caller
    that should. Exposing them here rather than duplicating the forward pass
    keeps one tie-break and one batching path.
    """
    windows_ = list(windows_)
    if not windows_:
        return {}
    pairs, keys = _window_pairs(windows_, query_text, doc_text)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return {key: float(score) for key, score in zip(keys, scores)}
```

and replace `rerank`'s body with:

```python
def rerank(
    model,
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    batch_size: int = 256,
) -> dict[str, list[str]]:
    """A new ordering of each window, best first.

    Returns orderings rather than scores on purpose: src.rerank_window builds
    the run from rank positions, so the cross-encoder's raw logits never enter
    a run alongside Stage 2's scores. Ties break on document id so two runs of
    the same model produce the same ordering.
    """
    windows_ = list(windows_)
    if not windows_:
        return {}
    scores = window_scores(
        model, windows_, query_text, doc_text, batch_size=batch_size
    )
    return {
        w.query_id: sorted(w.window, key=lambda d: (-scores[(w.query_id, d)], d))
        for w in windows_
    }
```

Add to `tests/test_cross_encoder.py`:

```python
def test_window_scores_keys_on_query_and_document():
    from src.cross_encoder import window_scores

    q, d = _maps()
    model = FakeModel({"doc b": 0.9})
    scores = window_scores(model, _windows(), q, d)
    assert scores[("1", "b")] == pytest.approx(0.9)
    assert set(scores) == {("1", "a"), ("1", "b"), ("1", "c"), ("2", "x"), ("2", "y")}


def test_rerank_and_window_scores_agree():
    # rerank is a wrapper over window_scores, so an ordering can never
    # disagree with the scores it came from.
    from src.cross_encoder import window_scores

    q, d = _maps()
    model = FakeModel({"doc a": 0.1, "doc b": 0.9, "doc c": 0.5})
    scores = window_scores(model, _windows(), q, d)
    ordering = rerank(FakeModel({"doc a": 0.1, "doc b": 0.9, "doc c": 0.5}), _windows(), q, d)
    assert ordering["1"] == sorted(
        ["a", "b", "c"], key=lambda doc: (-scores[("1", doc)], doc)
    )


def test_window_scores_of_nothing_is_nothing():
    from src.cross_encoder import window_scores

    assert window_scores(FakeModel(), [], *_maps()) == {}
```

- [ ] **Step 4: Write the implementation**

Create `src/stage_signals.py`:

```python
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

**A cache miss is an error, not a fallback.** data/llm-rerank.json covers the
4,130 fold-0 windows and the 2,000-query test sample. Over any other query set
the lookup misses, and the runtime behaviour of src.llm_rerank - keep the
Stage 2 order - would here mean labelling Stage 2's ordering `llm` and
reporting the dilution as a blend effect. So the lookup returns its misses and
the caller refuses to build.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

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
    "label_code",
    "gain",
    "qrel",
)

# What the learned combiner is allowed to see. Deliberately not `gain` or
# `qrel`: those are the answer.
BLEND_FEATURES: tuple[str, ...] = (
    "stage2_score",
    "stage2_rank",
    "ce_score",
    "llm_rank",
    "llm_rr",
)

# The same constant src.rrf uses, so a reciprocal rank means the same thing
# in Stage 1 and Stage 4.
RRF_K = 60


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


def build_signals(
    windows_: Sequence,
    labels: pd.DataFrame,
    *,
    ce_scores: Mapping[tuple[str, str], float],
    llm_orderings: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """One row per window document, carrying every stage's signal and the label.

    The tail is excluded: no Stage 3 arm looked at it, so no stage has an
    opinion of it to blend, and including it would let the combiner reorder
    documents the window never contained.
    """
    label_by = {
        (str(q), str(p)): (int(c), float(g), int(r))
        for q, p, c, g, r in zip(
            labels["query_id"], labels["product_id"],
            labels["label_code"], labels["gain"], labels["qrel"],
        )
    }
    stage2_by = {
        (str(q), str(p)): float(s)
        for q, p, s in zip(
            labels["query_id"], labels["product_id"], labels["stage2_score"]
        )
    } if "stage2_score" in labels.columns else None

    rows: list[dict] = []
    for w in windows_:
        order = list(llm_orderings.get(w.query_id, w.window))
        llm_position = {doc: i for i, doc in enumerate(order, start=1)}
        for position, doc in enumerate(w.window, start=1):
            key = (w.query_id, doc)
            if key not in ce_scores:
                raise KeyError(
                    f"no cross-encoder score for {key}; a blend feature that "
                    "is silently NaN trains the combiner on nothing"
                )
            if key not in label_by:
                raise KeyError(
                    f"no label for {key}; a window document with no judgement "
                    "cannot be trained on or scored"
                )
            code, gain, qrel = label_by[key]
            rank = llm_position[doc]
            rows.append(
                {
                    "query_id": w.query_id,
                    "product_id": doc,
                    "stage2_score": (
                        stage2_by[key] if stage2_by is not None else float("nan")
                    ),
                    "stage2_rank": position,
                    "ce_score": float(ce_scores[key]),
                    "llm_rank": rank,
                    "llm_rr": 1.0 / (RRF_K + rank),
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
    from src.fine_rank_report import TEST_SAMPLE
    from src.llm_rerank import DEFAULT_CACHE, RerankCache
    from src.ranker import REPORT_FOLD
    from src.rerank_window import DEFAULT_K, windows
    from src.stage2_scores import load_stage2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="fold0", choices=list(SCOPES))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--model-dir", type=Path, default=Path("models/cross-encoder"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The two scopes are exactly the two query sets Plan 6 ran the LLM over.
    if args.scope == "fold0":
        matrix = pd.read_parquet(args.features_dir / "train.parquet")
        matrix = matrix.loc[matrix["fold"] == REPORT_FOLD]
        split = "train"
    else:
        matrix = pd.read_parquet(args.features_dir / "test.parquet")
        keep = matrix["query_id"].drop_duplicates().sample(
            n=TEST_SAMPLE, random_state=args.seed
        )
        matrix = matrix.loc[matrix["query_id"].isin(set(keep))]
        split = "test"
    matrix = matrix.reset_index(drop=True)

    scores = load_stage2(split)
    joined = matrix[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    ws = windows(joined, k=args.k)
    print(f"{args.scope}: {len(ws):,} windows over {matrix['query_id'].nunique():,} queries")

    query_text, doc_text = text_maps(
        matrix,
        Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}

    llm_orderings, missing = llm_orderings_from_cache(
        ws, query_text, RerankCache(args.cache)
    )
    if missing:
        raise SystemExit(
            f"{len(missing):,} of {len(ws):,} windows have no cached LLM "
            f"permutation (first: {missing[:3]}). This plan issues no API "
            "calls; run `python -m src.llm_rerank` for this scope first, or "
            "the blend would label Stage 2's ordering `llm`."
        )

    model = load_reranker(args.model_dir / "lambda")
    ce_scores = window_scores(model, ws, query_text, doc_text)
    del model

    labels = matrix[["query_id", "product_id", "label_code", "gain", "qrel"]].copy()
    labels["query_id"] = labels["query_id"].astype(str)
    labels["product_id"] = labels["product_id"].astype(str)
    labels = labels.merge(
        joined[["query_id", "product_id", "stage2_score"]].astype(
            {"query_id": str, "product_id": str}
        ),
        on=["query_id", "product_id"],
    )

    frame = build_signals(
        ws, labels, ce_scores=ce_scores, llm_orderings=llm_orderings
    )
    require_full_coverage(frame)

    path = args.features_dir / f"stage-signals-{args.scope}.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {len(frame):,} rows over {frame['query_id'].nunique():,} queries to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stage_signals.py tests/test_cross_encoder.py -q`
Expected: PASS, 16 + 31 tests.

- [ ] **Step 6: Dump both real signal frames**

Run:

```bash
python -m src.stage_signals --scope fold0
python -m src.stage_signals --scope test-sample
```

Expected: 4,130 windows over 4,130 fold-0 queries and 2,000 over 2,000 test
queries, no cache misses, and roughly 33,000 and 16,000 rows (fold 0's median
candidate count is 16, so most windows are full at `K = 10`).

**If this reports cache misses, stop.** One miss on fold 0 is expected and
documented — Plan 6 measured 4,129 of 4,130 cached, because the LLM's malformed
responses are deliberately not cached so they retry. Either re-run
`python -m src.llm_rerank --split train --folds 0` to fill it (a handful of
calls, since the rest are cached), or record the count. Thousands of misses
means the scope is wrong.

- [ ] **Step 7: Write the data-marked test**

Append to `tests/test_stage_signals.py`:

```python
@pytest.mark.data
def test_the_real_frames_are_complete():
    from src.stage_signals import load_signals

    for scope, n_queries in [("fold0", 4130), ("test-sample", 2000)]:
        frame = load_signals(scope)
        assert frame["query_id"].nunique() == n_queries
        require_full_coverage(frame)
        # Every window is at most K documents and at least one.
        sizes = frame.groupby("query_id").size()
        assert sizes.max() <= 10
        assert sizes.min() >= 1
        # The three stages disagree; if any two are identical the frame is
        # carrying one stage's opinion twice under two names.
        assert not frame["ce_score"].equals(frame["stage2_score"])
        assert (frame["llm_rank"] != frame["stage2_rank"]).any()
```

Run: `python -m pytest tests/test_stage_signals.py -m data -q`
Expected: PASS.

- [ ] **Step 8: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 9: Commit**

```bash
git add src/stage_signals.py tests/test_stage_signals.py src/cross_encoder.py tests/test_cross_encoder.py
git commit -m "Persist every stage's signal for the blend and the error analysis"
```

---

## Task 2: The three blend strategies

**Files:**
- Create: `src/blend.py`
- Create: `tests/test_blend.py`

**Interfaces:**
- Consumes: `src.stage_signals.BLEND_FEATURES`, `src.ranker.LABEL_GAIN`.
- Produces:
  - `src.blend.STRATEGIES: tuple[str, ...]` = `("fixed", "combiner", "selector")`
  - `src.blend.DEFAULT_WEIGHTS: dict[str, float]`
  - `src.blend.orderings_from_scores(frame, column, *, ascending=False) -> dict[str, list[str]]`
  - `src.blend.fixed_weight_ordering(frame, weights=DEFAULT_WEIGHTS, *, k=RRF_K) -> dict[str, list[str]]`
  - `src.blend.train_combiner(frame, *, features=BLEND_FEATURES, num_boost_round=150, seed=0) -> Any`
  - `src.blend.predict_combiner(booster, frame, *, features=BLEND_FEATURES) -> np.ndarray`
  - `src.blend.cross_fit_predict(frame, *, n_folds=2, seed=0, **kwargs) -> np.ndarray`
  - `src.blend.train_selector(frame, per_arm, *, seed=0) -> Any`
  - `src.blend.selector_ordering(selector, frame, arm_orderings) -> dict[str, list[str]]`
  - `src.blend.oracle_ordering(per_arm_per_query, arm_orderings) -> dict[str, list[str]]`

**Review Focus 1 lives in `cross_fit_predict`.** Fold 0 is the only query set
with both an LLM ordering and a label, so it is the only training set the
combiner can have — and it is also the surface Plans 5 and 6 reported on. Five
features over 4,130 queries memorise easily. `cross_fit_predict` partitions by
`query_id`, trains on the complement of each part and predicts only that part,
so no query is ever scored by a model that saw it. The test asserts it on a
frame where the label is a pure function of a feature the model can find: a
leaking implementation scores near-perfectly, a correct one does not.

**The fixed-weight arm is RRF, not a score average.** The three signals are on
three incomparable scales — a LambdaMART score, a cross-encoder logit and a
rank — and averaging them is the scale mismatch Plan 6's Review Focus 1 exists
to prevent. Reciprocal rank puts all three on one scale by construction.
`DEFAULT_WEIGHTS` is `{"stage2": 1.0, "ce": 1.0, "llm": 4.0}`, the best of the
sweep measured in [`README.md`](README.md#where-these-numbers-came-from) — and
that sweep's best still lost to the LLM alone, which is the point.

**The selector is the only arm aimed at the oracle's structure.** The oracle
wins by choosing between arms per query, not by mixing them, and no arm is best
more than 42% of the time. `train_selector` fits a multiclass LightGBM over
query-level features — window size, the three arms' score spreads, their mutual
rank agreement — to predict which arm to trust, and `selector_ordering` applies
it. If it cannot beat the best single arm either, that is the plan's finding.

- [ ] **Step 1: Write the failing test**

Create `tests/test_blend.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.blend import (
    DEFAULT_WEIGHTS,
    STRATEGIES,
    cross_fit_predict,
    fixed_weight_ordering,
    oracle_ordering,
    orderings_from_scores,
    predict_combiner,
    selector_ordering,
    train_combiner,
    train_selector,
)
from src.stage_signals import BLEND_FEATURES


def _frame(n_queries=60, seed=0):
    """A signal frame where the label follows ce_score, with noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        for position in range(1, 5):
            code = int(rng.integers(0, 4))
            rows.append(
                {
                    "query_id": str(q),
                    "product_id": f"p{q}_{position}",
                    "stage2_score": rng.normal(),
                    "stage2_rank": position,
                    "ce_score": code + rng.normal(scale=0.3),
                    "llm_rank": int(rng.integers(1, 5)),
                    "llm_rr": 0.0,
                    "label_code": code,
                    "gain": [0.0, 0.01, 0.1, 1.0][code],
                    "qrel": [0, 1, 10, 100][code],
                }
            )
    frame = pd.DataFrame(rows)
    frame["llm_rr"] = 1.0 / (60 + frame["llm_rank"])
    return frame


# --- orderings --------------------------------------------------------------

def test_an_ordering_is_one_permutation_per_query():
    frame = _frame(n_queries=5)
    out = orderings_from_scores(frame, "ce_score")
    assert set(out) == set(frame["query_id"])
    for q, docs in out.items():
        assert sorted(docs) == sorted(frame.loc[frame["query_id"] == q, "product_id"])


def test_higher_scores_come_first():
    frame = _frame(n_queries=1)
    out = orderings_from_scores(frame, "ce_score")["0"]
    scores = frame.set_index("product_id")["ce_score"]
    assert list(out) == list(scores.sort_values(ascending=False).index)


def test_rank_columns_sort_ascending():
    # A rank of 1 is best, so the ordering must not put 4 first.
    frame = _frame(n_queries=1)
    out = orderings_from_scores(frame, "stage2_rank", ascending=True)["0"]
    assert out[0] == frame.sort_values("stage2_rank")["product_id"].iat[0]


def test_ties_break_on_document_id():
    frame = pd.DataFrame({
        "query_id": ["1", "1"], "product_id": ["b", "a"], "ce_score": [1.0, 1.0],
    })
    assert orderings_from_scores(frame, "ce_score")["1"] == ["a", "b"]


# --- the fixed-weight arm ---------------------------------------------------

def test_the_fixed_weight_arm_returns_permutations():
    frame = _frame(n_queries=10)
    out = fixed_weight_ordering(frame)
    for q, docs in out.items():
        assert sorted(docs) == sorted(frame.loc[frame["query_id"] == q, "product_id"])


def test_weighting_one_stage_to_the_exclusion_of_the_rest_reproduces_it():
    # A sanity anchor: all weight on the LLM must give the LLM's own ordering.
    frame = _frame(n_queries=8)
    out = fixed_weight_ordering(frame, {"stage2": 0.0, "ce": 0.0, "llm": 1.0})
    expected = orderings_from_scores(frame, "llm_rank", ascending=True)
    assert out == expected


def test_the_default_weights_favour_the_llm():
    # Measured: the best swept fusion was 1:1:4, and it still lost to the LLM
    # alone. The default records the sweep, not a hope.
    assert DEFAULT_WEIGHTS["llm"] > DEFAULT_WEIGHTS["ce"]
    assert DEFAULT_WEIGHTS["llm"] > DEFAULT_WEIGHTS["stage2"]


# --- the learned combiner ---------------------------------------------------

def test_the_combiner_learns_the_informative_feature():
    frame = _frame(n_queries=120)
    booster = train_combiner(frame)
    frame = frame.assign(blend=predict_combiner(booster, frame))
    exact = frame.loc[frame["label_code"] == 3, "blend"].mean()
    irrelevant = frame.loc[frame["label_code"] == 0, "blend"].mean()
    assert exact > irrelevant


def test_the_combiner_uses_the_declared_features_only():
    frame = _frame(n_queries=40)
    booster = train_combiner(frame)
    assert list(booster.feature_name()) == list(BLEND_FEATURES)


def test_the_combiner_refuses_a_label_gain_override():
    # src.ranker derives label_gain from src.labels and refuses an override;
    # the blend must not be the place that reintroduces LightGBM's default.
    frame = _frame(n_queries=20)
    with pytest.raises(ValueError, match="label_gain"):
        train_combiner(frame, params={"label_gain": [0, 1, 3, 7]})


# --- Review Focus 1: no query scored by a model that trained on it ---------

def test_cross_fit_scores_every_row_exactly_once():
    frame = _frame(n_queries=40)
    scores = cross_fit_predict(frame)
    assert len(scores) == len(frame)
    assert np.isfinite(scores).all()


def test_cross_fit_does_not_let_a_query_score_itself():
    # The label here is a pure function of a feature the model can memorise
    # through query_id-shaped noise. A leaking implementation reproduces the
    # training fit; a correct one is measurably worse than it.
    frame = _frame(n_queries=60, seed=3)
    booster = train_combiner(frame)
    in_sample = predict_combiner(booster, frame)
    out_of_sample = cross_fit_predict(frame, seed=3)
    assert not np.allclose(in_sample, out_of_sample)


def test_cross_fit_partitions_by_query_not_by_row():
    # Splitting by row would put four documents of one query on both sides,
    # which is the same leak wearing a different hat.
    frame = _frame(n_queries=20)
    parts = cross_fit_predict(frame, n_folds=2, seed=0, _return_parts=True)
    for part in parts:
        assert len(set(part)) == len(part)
    assert set().union(*[set(p) for p in parts]) == set(frame["query_id"])
    assert not set(parts[0]) & set(parts[1])


def test_cross_fit_needs_more_queries_than_folds():
    with pytest.raises(ValueError, match="folds"):
        cross_fit_predict(_frame(n_queries=1), n_folds=2)


# --- the selector -----------------------------------------------------------

def _per_arm():
    return {
        "stage2": {"0": 0.9, "1": 0.2, "2": 0.5},
        "ce": {"0": 0.4, "1": 0.8, "2": 0.5},
        "llm": {"0": 0.1, "1": 0.3, "2": 0.9},
    }


def _arm_orderings():
    return {
        "stage2": {"0": ["a", "b"], "1": ["a", "b"], "2": ["a", "b"]},
        "ce": {"0": ["b", "a"], "1": ["b", "a"], "2": ["b", "a"]},
        "llm": {"0": ["a", "b"], "1": ["b", "a"], "2": ["a", "b"]},
    }


def test_the_oracle_takes_the_best_arm_per_query():
    out = oracle_ordering(_per_arm(), _arm_orderings())
    assert out["0"] == ["a", "b"]   # stage2 wins query 0
    assert out["1"] == ["b", "a"]   # ce wins query 1
    assert out["2"] == ["a", "b"]   # llm wins query 2


def test_the_oracle_is_a_ceiling_not_an_arm():
    # It reads the labels, so it can only ever bound a method, never be one.
    per_arm = _per_arm()
    out = oracle_ordering(per_arm, _arm_orderings())
    assert set(out) == set(per_arm["stage2"])


def test_the_selector_returns_a_permutation_from_one_of_the_arms():
    frame = _frame(n_queries=30)
    per_arm = {
        arm: {q: float(np.random.default_rng(i).random()) for q in frame["query_id"].unique()}
        for i, arm in enumerate(("stage2", "ce", "llm"))
    }
    arm_orderings = {
        arm: orderings_from_scores(frame, column, ascending=ascending)
        for arm, column, ascending in [
            ("stage2", "stage2_rank", True),
            ("ce", "ce_score", False),
            ("llm", "llm_rank", True),
        ]
    }
    selector = train_selector(frame, per_arm)
    out = selector_ordering(selector, frame, arm_orderings)
    for q, docs in out.items():
        assert docs in [arm_orderings[arm][q] for arm in ("stage2", "ce", "llm")]


def test_the_declared_strategies_are_the_three_the_plan_measures():
    assert STRATEGIES == ("fixed", "combiner", "selector")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_blend.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.blend'`.

- [ ] **Step 3: Write the implementation**

Create `src/blend.py`:

```python
"""Stage 4: one ordering from three stages' opinions.

PROJECT_SPEC.md §4.4 - "weighted combination, or a small learned combiner over
stage scores" - and it asks for the comparison against each single stage alone.
This module is the strategies; src/blend_report.py measures them.

Three arms, and a ceiling:

  * **fixed**    reciprocal-rank fusion at hand-set weights. RRF rather than a
                 score average because a LambdaMART score, a cross-encoder
                 logit and a rank are three incomparable scales, and averaging
                 them is exactly the mismatch src/rerank_window.py exists to
                 prevent.
  * **combiner** LightGBM lambdarank over the five signal columns.
  * **selector** a per-query choice of which single arm to trust.
  * **oracle**   the best arm per query, chosen with the labels. Not a method -
                 it bounds one.

**Measured before this module was written, on fold 0**: the oracle reaches
0.9076 against the best single arm's 0.8814, so there is +0.0262 to aim at; but
every fixed weighting loses (best 0.8800 at 1:1:4) and a cross-fitted combiner
loses too (0.8793). The selector exists because the oracle's advantage is
*choosing*, not mixing: Stage 2 is best on 31.7% of queries, the cross-encoder
on 26.7%, the LLM on 41.6%.

Nothing here reads a file or calls an API, so every path is testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.stage_signals import BLEND_FEATURES, RRF_K

STRATEGIES: tuple[str, ...] = ("fixed", "combiner", "selector")

ARM_COLUMNS: dict[str, tuple[str, bool]] = {
    # arm -> (column, ascending). A rank sorts ascending, a score descending.
    "stage2": ("stage2_rank", True),
    "ce": ("ce_score", False),
    "llm": ("llm_rank", True),
}

# The best of the sweep measured on fold 0 - which still lost to the LLM
# alone. Recorded so the arm reproduces what was measured, not a guess.
DEFAULT_WEIGHTS: dict[str, float] = {"stage2": 1.0, "ce": 1.0, "llm": 4.0}

_SELECTOR_ARMS: tuple[str, ...] = ("stage2", "ce", "llm")


def orderings_from_scores(
    frame: pd.DataFrame, column: str, *, ascending: bool = False
) -> dict[str, list[str]]:
    """One ordering per query, ties broken on product_id."""
    sign = 1.0 if ascending else -1.0
    out: dict[str, list[str]] = {}
    for query_id, group in frame.groupby("query_id", sort=True):
        pairs = sorted(
            zip(group[column].to_numpy(), group["product_id"].astype(str)),
            key=lambda sd: (sign * float(sd[0]), sd[1]),
        )
        out[str(query_id)] = [doc for _, doc in pairs]
    return out


def fixed_weight_ordering(
    frame: pd.DataFrame,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    *,
    k: int = RRF_K,
) -> dict[str, list[str]]:
    """Reciprocal-rank fusion of the three arms at fixed weights."""
    per_arm = {
        arm: orderings_from_scores(frame, column, ascending=ascending)
        for arm, (column, ascending) in ARM_COLUMNS.items()
    }
    out: dict[str, list[str]] = {}
    for query_id in per_arm["stage2"]:
        fused: dict[str, float] = {}
        for arm, orderings in per_arm.items():
            weight = float(weights.get(arm, 0.0))
            for rank, doc in enumerate(orderings[query_id], start=1):
                fused[doc] = fused.get(doc, 0.0) + weight / (k + rank)
        out[query_id] = sorted(fused, key=lambda d: (-fused[d], d))
    return out


def _group_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Run lengths over consecutive query_id, as LightGBM's `group` wants."""
    ids = frame["query_id"].to_numpy()
    if len(ids) == 0:
        return np.array([], dtype=np.int64)
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    sizes = np.diff(np.concatenate([[0], boundaries, [len(ids)]]))
    if len(sizes) != len(pd.unique(ids)):
        raise ValueError(
            "query_id is not contiguous; sort by query_id before training or "
            "LightGBM will learn comparisons that straddle queries"
        )
    return sizes.astype(np.int64)


def train_combiner(
    frame: pd.DataFrame,
    *,
    features: Sequence[str] = BLEND_FEATURES,
    params: Mapping[str, Any] | None = None,
    num_boost_round: int = 150,
    seed: int = 0,
) -> Any:
    """LightGBM lambdarank over the stage signals.

    `label_gain` comes from src.ranker, which derives it from src.labels. The
    same booster reports an NDCG 6.4 points apart under LightGBM's default
    2**rel - 1 gain, so it is never a caller's to set.
    """
    import lightgbm as lgb

    from src.ranker import LABEL_GAIN

    overrides = dict(params or {})
    if "label_gain" in overrides:
        raise ValueError(
            "label_gain is derived from src.labels.ESCI_GAINS and must not be "
            "overridden; LightGBM's default 2**rel - 1 measures a different "
            "metric from the one this project reports"
        )

    ordered = frame.sort_values("query_id", kind="stable")
    settings = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "label_gain": LABEL_GAIN,
        "lambdarank_truncation_level": 20,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "deterministic": True,
        "seed": seed,
        "data_random_seed": seed,
    } | overrides
    dataset = lgb.Dataset(
        ordered[list(features)],
        label=ordered["label_code"].to_numpy(),
        group=_group_sizes(ordered),
        free_raw_data=False,
    )
    return lgb.train(settings, dataset, num_boost_round=num_boost_round)


def predict_combiner(
    booster, frame: pd.DataFrame, *, features: Sequence[str] = BLEND_FEATURES
) -> np.ndarray:
    """One blend score per row, higher is better."""
    return np.asarray(booster.predict(frame[list(features)]), dtype=np.float64)


def cross_fit_predict(
    frame: pd.DataFrame,
    *,
    n_folds: int = 2,
    seed: int = 0,
    features: Sequence[str] = BLEND_FEATURES,
    num_boost_round: int = 150,
    _return_parts: bool = False,
):
    """Out-of-fold blend scores: no query is scored by a model that saw it.

    Fold 0 is the only query set with both an LLM ordering and a label, so it
    is the only training set the combiner can have - and it is also the surface
    Plans 5 and 6 reported on. Five features over 4,130 queries memorise
    easily, so a fold-0 number from a model fitted on all of fold 0 would be a
    training fit wearing a result's clothes.

    Partitioning is by `query_id`, never by row: four documents of one query on
    both sides of the split is the same leak in a different shape.
    """
    queries = np.array(sorted(pd.unique(frame["query_id"].astype(str))))
    if len(queries) < n_folds:
        raise ValueError(
            f"{len(queries)} queries cannot be split into {n_folds} folds"
        )
    shuffled = queries[np.random.default_rng(seed).permutation(len(queries))]
    parts = [list(part) for part in np.array_split(shuffled, n_folds)]
    if _return_parts:
        return parts

    ids = frame["query_id"].astype(str).to_numpy()
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    for part in parts:
        held = np.isin(ids, list(part))
        booster = train_combiner(
            frame.loc[~held],
            features=features,
            num_boost_round=num_boost_round,
            seed=seed,
        )
        scores[held] = predict_combiner(
            booster, frame.loc[held], features=features
        )
    if np.isnan(scores).any():
        raise ValueError("some rows were never scored; the partition is not a cover")
    return scores


def _selector_features(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per query: how much the arms disagree, and how confident each is.

    Deliberately query-level and label-free. A selector fed the labels would be
    the oracle.
    """
    rows = []
    for query_id, group in frame.groupby("query_id", sort=True):
        stage2 = group["stage2_rank"].to_numpy(dtype=float)
        ce = group["ce_score"].to_numpy(dtype=float)
        llm = group["llm_rank"].to_numpy(dtype=float)
        n = len(group)
        rows.append(
            {
                "query_id": str(query_id),
                "n_candidates": n,
                "ce_spread": float(ce.max() - ce.min()) if n > 1 else 0.0,
                "ce_top_margin": (
                    float(np.sort(ce)[-1] - np.sort(ce)[-2]) if n > 1 else 0.0
                ),
                "stage2_score_spread": float(
                    group["stage2_score"].max() - group["stage2_score"].min()
                ) if n > 1 else 0.0,
                # Rank agreement: 1.0 when two arms order identically.
                "agree_stage2_llm": float(
                    np.corrcoef(stage2, llm)[0, 1]
                ) if n > 1 and llm.std() > 0 and stage2.std() > 0 else 0.0,
                "agree_ce_llm": float(
                    np.corrcoef(-ce, llm)[0, 1]
                ) if n > 1 and llm.std() > 0 and ce.std() > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


SELECTOR_FEATURES: tuple[str, ...] = (
    "n_candidates",
    "ce_spread",
    "ce_top_margin",
    "stage2_score_spread",
    "agree_stage2_llm",
    "agree_ce_llm",
)


def train_selector(
    frame: pd.DataFrame,
    per_arm: Mapping[str, Mapping[str, float]],
    *,
    seed: int = 0,
    num_boost_round: int = 120,
) -> Any:
    """A multiclass model over which arm wins each query.

    The oracle's advantage is choosing, not mixing: on fold 0 no arm is best
    more than 41.6% of the time. This is the only Stage 4 arm whose shape
    matches that.
    """
    import lightgbm as lgb

    features = _selector_features(frame)
    target = []
    for query_id in features["query_id"]:
        scores = [float(per_arm[arm].get(query_id, 0.0)) for arm in _SELECTOR_ARMS]
        target.append(int(np.argmax(scores)))
    dataset = lgb.Dataset(
        features[list(SELECTOR_FEATURES)],
        label=np.asarray(target),
        free_raw_data=False,
    )
    return lgb.train(
        {
            "objective": "multiclass",
            "num_class": len(_SELECTOR_ARMS),
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 30,
            "verbose": -1,
            "deterministic": True,
            "seed": seed,
            "data_random_seed": seed,
        },
        dataset,
        num_boost_round=num_boost_round,
    )


def selector_ordering(
    selector,
    frame: pd.DataFrame,
    arm_orderings: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    """Each query takes the ordering of the arm the selector picked."""
    features = _selector_features(frame)
    probabilities = np.asarray(selector.predict(features[list(SELECTOR_FEATURES)]))
    out: dict[str, list[str]] = {}
    for query_id, row in zip(features["query_id"], probabilities):
        arm = _SELECTOR_ARMS[int(np.argmax(row))]
        out[str(query_id)] = list(arm_orderings[arm][str(query_id)])
    return out


def oracle_ordering(
    per_arm_per_query: Mapping[str, Mapping[str, float]],
    arm_orderings: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    """The best arm per query, chosen with the labels.

    A ceiling, never a method: it needs the NDCG it is trying to produce. It
    is reported beside the three arms so a tie can be read as "the methods
    failed to reach available headroom" rather than "there was none".
    """
    arms = list(per_arm_per_query)
    out: dict[str, list[str]] = {}
    for query_id in per_arm_per_query[arms[0]]:
        best = max(arms, key=lambda a: float(per_arm_per_query[a][query_id]))
        out[str(query_id)] = list(arm_orderings[best][str(query_id)])
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_blend.py -q`
Expected: PASS, 20 tests.

- [ ] **Step 5: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 6: Commit**

```bash
git add src/blend.py tests/test_blend.py
git commit -m "Add the fixed-weight, learned and selector blend strategies"
```

---

## Phase 1 Gate

Phase 2 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the real signal-frame completeness check.
- [ ] `data/features/stage-signals-{fold0,test-sample}.parquet` exist, cover 4,130 and 2,000 queries, and `require_full_coverage` passes on both.
- [ ] `llm_orderings_from_cache` returns its misses, and `python -m src.stage_signals` refuses to write a frame with any.
- [ ] `cross_fit_predict` partitions by `query_id` and produces scores that differ from an in-sample fit on the same frame.
- [ ] `src/blend.py` reads no file, imports no store and issues no API call.
- [ ] `rerank` still returns orderings, and `window_scores` is the only way to see a logit.

Then: [Phase 2 — The Blend Report](phase-2-the-blend-report.md).
