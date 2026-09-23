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

- **No new API calls.** Every usable LLM ordering is already cached. The three windows whose answer was malformed carry the Stage 2 order, *flagged*, exactly as Plan 6 scored them; any miss beyond that is a scope error and stops the run.
- **No query may be scored — or routed — by a model that trained on it.** The combiner and the selector both.
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
- Consumes: `src.rerank_window.{Window, windows, DEFAULT_K}`, `src.stage2_scores.load_stage2`, `src.fine_rank_report.TEST_SAMPLE`, `src.ranker.REPORT_FOLD`, `src.rrf.DEFAULT_K`, `src.llm_rerank.{RerankCache, window_key, DEFAULT_CACHE}`, `src.cross_encoder.{load_reranker, text_maps, DEFAULT_MODEL_DIR}`.
- Produces:
  - `src.cross_encoder.window_scores(model, windows_, query_text, doc_text, *, batch_size=256) -> dict[tuple[str, str], float]`
  - `src.stage_signals.DEFAULT_DIR: Path` = `Path("data/features")`
  - `src.stage_signals.SCOPES: tuple[str, ...]` = `("fold0", "test-sample")`
  - `src.stage_signals.SIGNAL_COLUMNS: tuple[str, ...]` (includes `llm_fallback`)
  - `src.stage_signals.BLEND_FEATURES: tuple[str, ...]`
  - `src.stage_signals.RRF_K: int` (re-exported from `src.rrf.DEFAULT_K`)
  - `src.stage_signals.MAX_FALLBACK_SHARE: float` = `0.005`
  - `src.stage_signals.scope_windows(scope, *, k=None, features_dir=DEFAULT_DIR, seed=0) -> tuple[list[Window], pd.DataFrame]` — the only place a scope becomes query ids; Phases 2 and 3 call it rather than re-deriving the sample
  - `src.stage_signals.llm_orderings_from_cache(windows_, query_text, cache) -> tuple[dict[str, list[str]], list[str]]`
  - `src.stage_signals.check_fallback_share(missing, n_windows) -> None`
  - `src.stage_signals.build_signals(windows_, labels, *, ce_scores, llm_orderings, llm_fallbacks=()) -> pd.DataFrame`
  - `src.stage_signals.require_full_coverage(frame) -> None`
  - `src.stage_signals.load_signals(scope, directory=DEFAULT_DIR) -> pd.DataFrame`
  - CLI: `python -m src.stage_signals --scope {fold0,test-sample}` writing `data/features/stage-signals-<scope>.parquet`

**Review Focus 2 lives in `llm_orderings_from_cache`, `check_fallback_share`
and `build_signals`.** `data/llm-rerank.json` holds permutations for fold 0 and
the 2,000-query test sample — and nothing else. The tempting shape is "look it
up, and if it is missing keep the Stage 2 order", which is what `rerank_windows`
does at runtime and exactly wrong here: a signal frame built over the full
8,956-query test split would give 6,956 queries the Stage 2 order *labelled
`llm`*, and Stage 4 would report the dilution as a blend effect.

**But the cache is not complete even for the right scopes**, and this plan as
first written said it was. Measured against the cache on 2026-09-23 with no API
calls: **1 of 4,130** fold-0 windows (query 55755, "jeff shalfon the family")
and **2 of 2,000** test-sample windows (49855, "heartland season 13"; 66028,
"marvelous mrs maisel season 2") have no permutation. They are windows whose
LLM answer was malformed; malformed answers are deliberately not cached, and
fold 0's has now failed twice, so re-running would likely fail again — and on
test it would also be a second measurement. Plan 6 scored all three in Stage 2
order and counted them in `n_fallback` (1 and 2 in its committed results).

So the frame carries them the same way, **flagged**: `llm_fallback = True`,
`llm_rank = stage2_rank`. That is the only choice under which the frame's `llm`
column is the arm Plan 6 published (0.8814 fold 0, 0.8855 test), and the data
test proves it by reproducing both. What separates this from the scope error
is the count: `check_fallback_share` refuses more than 0.5% of windows (the
worst measured is 0.10%; the scope error is 77.7%), and `build_signals` raises
on any window that has neither an ordering nor a declared fallback, so the
silent path does not exist even for a caller that skips the CLI.

**Why the cross-encoder needs a new entry point.** Plan 6's `rerank` returns
orderings on purpose, so a logit can never land in a run beside a Stage 2
score (its Review Focus 1). A blend needs the logits themselves.
`window_scores` exposes them for exactly one consumer, and `rerank` is left
alone — in fact it becomes a thin wrapper over it, which also guarantees the
two can never disagree about a tie-break.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_signals.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from src.rerank_window import Window
from src.stage_signals import (
    BLEND_FEATURES,
    MAX_FALLBACK_SHARE,
    RRF_K,
    SCOPES,
    SIGNAL_COLUMNS,
    build_signals,
    check_fallback_share,
    llm_orderings_from_cache,
    require_full_coverage,
    scope_windows,
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
            "stage2_score": [3.0, 2.0, 1.0, 0.5, 9.0, 8.0],
        }
    )


def _ce():
    return {("1", "a"): 0.1, ("1", "b"): 0.9, ("1", "c"): 0.5,
            ("2", "x"): 0.2, ("2", "y"): 0.8}


def _llm():
    return {"1": ["c", "a", "b"], "2": ["y", "x"]}


def _frame(**kwargs):
    arguments = dict(ce_scores=_ce(), llm_orderings=_llm()) | kwargs
    return build_signals(_windows(), _labels(), **arguments)


# --- the frame --------------------------------------------------------------

def test_the_frame_has_one_row_per_window_document():
    frame = _frame()
    assert len(frame) == 5          # 3 + 2 window documents; the tail is not scored
    assert list(frame.columns) == list(SIGNAL_COLUMNS)


def test_the_tail_is_not_in_the_frame():
    # Stage 3 never looked at it, so no stage has an opinion to blend.
    assert "z" not in set(_frame()["product_id"])


def test_stage_2_rank_is_the_window_position():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["a", "stage2_rank"] == 1
    assert q1.loc["c", "stage2_rank"] == 3


def test_stage_2_score_travels_from_the_labels():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["b", "stage2_score"] == pytest.approx(2.0)


def test_llm_rank_is_the_position_in_the_llm_ordering():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["c", "llm_rank"] == 1      # the LLM put c first
    assert q1.loc["b", "llm_rank"] == 3


def test_reciprocal_rank_uses_the_stage_1_constant():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["c", "llm_rr"] == pytest.approx(1.0 / (RRF_K + 1))
    assert q1.loc["c", "llm_rr"] > q1.loc["b", "llm_rr"]


def test_labels_travel_with_the_signals():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["a", "label_code"] == 3
    assert q1.loc["a", "qrel"] == 100


def test_every_blend_feature_is_a_column():
    assert set(BLEND_FEATURES) <= set(_frame().columns)


def test_the_answer_is_not_a_blend_feature():
    # gain, qrel and label_code are the label. llm_fallback is bookkeeping:
    # three windows carry it, and a combiner that learned from it would be
    # learning which queries made the LLM malfunction.
    assert not {"gain", "qrel", "label_code", "llm_fallback"} & set(BLEND_FEATURES)


def test_a_missing_cross_encoder_score_raises():
    with pytest.raises(KeyError, match="cross-encoder"):
        _frame(ce_scores={})


def test_a_missing_label_raises():
    # A window document with no judgement would train the blend on a NaN.
    labels = _labels().drop(index=0)
    with pytest.raises(KeyError, match="label"):
        build_signals(_windows(), labels, ce_scores=_ce(), llm_orderings=_llm())


def test_labels_without_a_stage_2_score_raise():
    # The written plan let this through as NaN, and require_full_coverage then
    # rejected every frame - including the one its own test called complete.
    labels = _labels().drop(columns=["stage2_score"])
    with pytest.raises(KeyError, match="stage2_score"):
        build_signals(_windows(), labels, ce_scores=_ce(), llm_orderings=_llm())


# --- Review Focus 2: the cache covers these windows, or it says so ---------

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
    # Falling back silently here would label 6,956 test queries `llm` and
    # report the dilution as a blend effect.
    query_text = {"1": "red shoes", "2": "blue hat"}
    orderings, missing = llm_orderings_from_cache(_windows(), query_text, FakeCache({}))
    assert sorted(missing) == ["1", "2"]
    assert orderings == {}


def test_a_permutation_of_the_wrong_length_counts_as_a_miss():
    from src.llm_rerank import window_key

    query_text = {"1": "red shoes", "2": "blue hat"}
    cache = FakeCache({window_key("red shoes", ["a", "b", "c"]): [2, 1]})
    _, missing = llm_orderings_from_cache(_windows(), query_text, cache)
    assert "1" in missing


def test_a_window_with_no_ordering_and_no_fallback_flag_raises():
    # build_signals itself refuses the silent fallback; the CLI is not the
    # only guard.
    with pytest.raises(KeyError, match="LLM ordering"):
        _frame(llm_orderings={"1": ["c", "a", "b"]})


def test_a_declared_fallback_keeps_the_stage_2_order_and_is_flagged():
    # Exactly how Plan 6 scored a window whose LLM answer was malformed, so
    # the frame reproduces Plan 6's arm rather than a different one.
    frame = _frame(llm_orderings={"1": ["c", "a", "b"]}, llm_fallbacks=["2"])
    q2 = frame.query("query_id == '2'").set_index("product_id")
    assert list(q2["llm_rank"]) == list(q2["stage2_rank"])
    assert q2["llm_fallback"].all()
    assert not frame.query("query_id == '1'")["llm_fallback"].any()


def test_a_query_cannot_be_both_ranked_and_a_fallback():
    with pytest.raises(ValueError, match="fallback"):
        _frame(llm_fallbacks=["1"])


def test_an_llm_ordering_that_is_not_a_permutation_raises():
    with pytest.raises(ValueError, match="permutation"):
        _frame(llm_orderings={"1": ["a", "a", "b"], "2": ["y", "x"]})


def test_the_measured_handful_of_fallbacks_is_accepted():
    check_fallback_share(["55755"], 4130)                 # fold 0, measured
    check_fallback_share(["49855", "66028"], 2000)        # test sample, measured


def test_a_scope_error_is_refused():
    # The full test split against a cache that covers 2,000 of its queries.
    missing = [str(i) for i in range(6956)]
    with pytest.raises(ValueError, match="scope"):
        check_fallback_share(missing, 8956)


def test_the_fallback_ceiling_sits_between_the_measurement_and_a_scope_error():
    assert 2 / 2000 <= MAX_FALLBACK_SHARE < 6956 / 8956


def test_require_full_coverage_passes_a_complete_frame():
    require_full_coverage(_frame())      # must not raise


def test_require_full_coverage_rejects_a_null_signal():
    frame = _frame()
    frame.loc[0, "ce_score"] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        require_full_coverage(frame)


def test_the_scopes_are_the_two_the_llm_actually_ran_on():
    assert SCOPES == ("fold0", "test-sample")


def test_an_unknown_scope_is_refused_before_anything_is_read(tmp_path):
    with pytest.raises(ValueError, match="scope"):
        scope_windows("test", features_dir=tmp_path)
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
# --- Plan 7: the scores themselves, for the blend ---------------------------

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
    scores = window_scores(FakeModel({"doc a": 0.1, "doc b": 0.9, "doc c": 0.5}), _windows(), q, d)
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stage_signals.py tests/test_cross_encoder.py -q`
Expected: PASS, 26 + 31 tests.

- [ ] **Step 6: Dump both real signal frames**

Run:

```bash
python -m src.stage_signals --scope fold0
python -m src.stage_signals --scope test-sample
```

Expected: 4,130 windows over 4,130 fold-0 queries with **41,255** rows, and
2,000 over 2,000 test queries with **19,974** rows — 99.3% and 99.2% of
windows are full at `K = 10`, since the mean candidate count is 20. (This plan
first predicted "roughly 33,000 and 16,000", which does not follow from its own
median of 16.) Each run names its fallbacks: **1** on fold 0 (55755) and **2**
on the test sample (49855, 66028), the windows Plan 6 counted in `n_fallback`.

**Do not fill them by calling the LLM.** Fold 0's has failed twice, and a
test-sample call would be a second measurement of an arm already reported
once. A different set of fallbacks, or any more than these three, means the
cache or the scope changed — stop and find out which.

- [ ] **Step 7: Write the data-marked test**

Append to `tests/test_stage_signals.py`:

```python
# --- the real frames --------------------------------------------------------

_PLAN_6_RESULTS = {
    "fold0": "docs/results/fine-rank.json",
    "test-sample": "docs/results/fine-rank-test.json",
}


def _ordering(frame, column, ascending):
    out = {}
    for query_id, group in frame.groupby("query_id", sort=False):
        pairs = sorted(
            zip(group[column], group["product_id"]),
            key=lambda sd: (sd[0] if ascending else -sd[0], sd[1]),
        )
        out[str(query_id)] = [doc for _, doc in pairs]
    return out


@pytest.mark.data
def test_the_real_frames_are_complete():
    from src.stage_signals import load_signals

    for scope, n_queries, n_fallback in [("fold0", 4130, 1), ("test-sample", 2000, 2)]:
        frame = load_signals(scope)
        assert frame["query_id"].nunique() == n_queries
        require_full_coverage(frame)
        # Every window is at most K documents and at least one.
        sizes = frame.groupby("query_id").size()
        assert sizes.max() <= 10
        assert sizes.min() >= 1
        # The windows whose LLM answer was malformed, as Plan 6 counted them.
        assert frame.groupby("query_id")["llm_fallback"].first().sum() == n_fallback
        # The three stages disagree; if any two are identical the frame is
        # carrying one stage's opinion twice under two names.
        assert not frame["ce_score"].equals(frame["stage2_score"])
        assert (frame["llm_rank"] != frame["stage2_rank"]).any()


@pytest.mark.data
def test_the_frames_reproduce_plan_6s_three_arms():
    # The frame is only the stage scores if ordering by each column gives back
    # the NDCG Plan 6 published for that stage. This also pins the test sample:
    # a different 2,000 queries could not land on all three numbers.
    from src.metrics import ndcg_per_query
    from src.rank_report import qrels_from_frame
    from src.rerank_window import spliced_run
    from src.stage_signals import load_signals

    for scope, path in _PLAN_6_RESULTS.items():
        published = {
            arm["name"]: arm["ndcg"]["point"]
            for arm in json.loads(open(path, encoding="utf-8").read())["arms"]
        }
        frame = load_signals(scope)
        windows_, matrix = scope_windows(scope)
        qrels = qrels_from_frame(matrix)
        for arm, column, ascending in [
            ("stage2", "stage2_rank", True),
            ("stage2+ce", "ce_score", False),
            ("stage2+llm", "llm_rank", True),
        ]:
            run = spliced_run(windows_, _ordering(frame, column, ascending))
            per_query = ndcg_per_query(run, qrels)
            measured = sum(per_query.values()) / len(per_query)
            assert measured == pytest.approx(published[arm], abs=5e-4), (scope, arm)
```

Run: `python -m pytest tests/test_stage_signals.py -m data -q`
Expected: PASS — both frames complete with 1 and 2 flagged fallbacks, and
`stage2`, `stage2+ce` and `stage2+llm` each reproduce Plan 6's committed NDCG
to within 0.0005 on both scopes. That second test is what makes the frame
trustworthy: it is the stage scores only if ordering by each column gives back
the number Plan 6 published for that stage.

- [ ] **Step 8: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 9: Commit**

```bash
git add src/stage_signals.py tests/test_stage_signals.py src/cross_encoder.py tests/test_cross_encoder.py
git commit -m "Persist every stage's signal for the blend and the error analysis"
```

**Landed 2026-09-23, as written.** `stage-signals-fold0.parquet` holds 41,255
rows over 4,130 queries (66 s end to end on the 3080) and
`stage-signals-test-sample.parquet` 19,974 over 2,000 (33 s); each run flagged
exactly the predicted fallbacks, 55755 and then 49855 and 66028. Ordering by
each column reproduces Plan 6's committed NDCG to float precision — every
difference is under 2e-16:

| scope | `stage2` | `stage2+ce` | `stage2+llm` |
|---|---|---|---|
| fold 0 | 0.851866 | 0.858663 | 0.881422 |
| test sample | 0.857626 | 0.861947 | 0.885471 |

Fast suite 623 passed (594 before), 27 deselected; `-m data` for this module
2 passed. No API call was made.

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
  - `src.blend.selector_targets(per_arm, query_ids) -> np.ndarray` — the arm each query should teach; ties go to the best-mean arm
  - `src.blend.train_selector(frame, per_arm, *, seed=0) -> Any`
  - `src.blend.selector_ordering(selector, frame, arm_orderings) -> dict[str, list[str]]`
  - `src.blend.cross_fit_select(frame, per_arm, arm_orderings, *, n_folds=2, seed=0) -> dict[str, list[str]]`
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
wins by choosing between arms per query, not by mixing them, and no arm is
strictly best more than 42% of the time. `train_selector` fits a multiclass LightGBM over
query-level features — window size, the three arms' score spreads, their mutual
rank agreement — to predict which arm to trust, and `selector_ordering` applies
it. If it cannot beat the best single arm either, that is the plan's finding.

**The selector leaks exactly as the combiner does, and the plan as first
written only guarded the combiner.** Its training target is "which arm had the
highest NDCG on this query" — the label in another form — and Phase 2 trained
it on all of fold 0 and then scored fold 0 with it. A 120-round multiclass
model over 4,130 queries fits its own training routes, so that fold-0 number
would have been in-sample while the JSON beside it said "cross-fit".
`cross_fit_select` routes each query with a selector trained on the other
part, over the **same** query partition `cross_fit_predict` uses (both go
through `_query_parts`), so the two learned arms' fold-0 numbers are
comparable with each other.

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
    cross_fit_select,
    fixed_weight_ordering,
    oracle_ordering,
    orderings_from_scores,
    predict_combiner,
    selector_ordering,
    selector_targets,
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


def test_a_strict_winner_is_the_target():
    per_arm = {
        "stage2": {"0": 0.9, "1": 0.2, "2": 0.5},
        "ce": {"0": 0.4, "1": 0.8, "2": 0.5},
        "llm": {"0": 0.1, "1": 0.3, "2": 0.9},
    }
    assert list(selector_targets(per_arm, ["0", "1", "2"])) == [0, 1, 2]


def test_a_tie_goes_to_the_arm_with_the_best_mean_not_the_first_listed():
    # Measured on fold 0: argmax hands every tie to stage2, the first-listed
    # and weakest arm, labelling 31.7% of queries "stage2" when it is strictly
    # best on 17.7%. A tied query scores the same whichever tied arm routes
    # it, so it should teach the default, not the list order.
    per_arm = {
        "stage2": {"tie": 0.8, "a": 0.5, "b": 0.5},
        "ce": {"tie": 0.8, "a": 0.5, "b": 0.5},
        "llm": {"tie": 0.8, "a": 0.9, "b": 0.9},
    }
    assert list(selector_targets(per_arm, ["tie", "a", "b"])) == [2, 2, 2]


def test_a_query_missing_from_the_arm_scores_is_an_error_not_a_zero():
    per_arm = {"stage2": {"0": 0.5}, "ce": {"0": 0.5}, "llm": {}}
    with pytest.raises(KeyError, match="0"):
        selector_targets(per_arm, ["0"])


def _selector_inputs(n_queries=40):
    frame = _frame(n_queries=n_queries)
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
    return frame, per_arm, arm_orderings


def test_cross_fit_select_routes_every_query_to_one_arms_ordering():
    frame, per_arm, arm_orderings = _selector_inputs()
    out = cross_fit_select(frame, per_arm, arm_orderings)
    assert set(out) == set(frame["query_id"])
    for q, docs in out.items():
        assert docs in [arm_orderings[arm][q] for arm in ("stage2", "ce", "llm")]


def test_cross_fit_select_uses_the_combiner_s_query_partition():
    # Review Focus 1 for the selector. Its target is per-query NDCG - the
    # label in another form - so a selector trained on all of fold 0 and
    # scored on fold 0 reports its own training routes. Sharing the combiner's
    # partition also keeps the two learned arms' fold-0 numbers comparable.
    frame, per_arm, arm_orderings = _selector_inputs()
    assert cross_fit_select(
        frame, per_arm, arm_orderings, seed=4, _return_parts=True
    ) == cross_fit_predict(frame, seed=4, _return_parts=True)


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
*choosing*, not mixing: the LLM is strictly best on 41.6% of queries, the
cross-encoder on 21.3%, Stage 2 on 17.7%, and 19.4% are tied at the top.

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


def _query_parts(frame: pd.DataFrame, n_folds: int, seed: int) -> list[list[str]]:
    """A seeded partition of the frame's query ids into `n_folds` parts.

    One function for both learned arms, so the combiner and the selector are
    cross-fitted over the same split and their fold-0 numbers compare.
    """
    queries = np.array(sorted(pd.unique(frame["query_id"].astype(str))))
    if len(queries) < n_folds:
        raise ValueError(
            f"{len(queries)} queries cannot be split into {n_folds} folds"
        )
    shuffled = queries[np.random.default_rng(seed).permutation(len(queries))]
    return [list(part) for part in np.array_split(shuffled, n_folds)]


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
    parts = _query_parts(frame, n_folds, seed)
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


def selector_targets(
    per_arm: Mapping[str, Mapping[str, float]], query_ids: Sequence[str]
) -> np.ndarray:
    """Which arm the selector should learn to pick for each query.

    The arm with the highest NDCG - and on a tie, the tied arm with the best
    mean over these queries, never the first one listed. Measured on fold 0,
    19.4% of queries are tied at the top; np.argmax hands every such tie to
    stage2, the weakest arm, labelling 31.7% of queries "stage2" when it is
    strictly best on 17.7%. A tied query scores the same whichever tied arm
    routes it, so it should teach the default, not the list order.
    """
    missing = [
        q for q in query_ids
        if any(str(q) not in per_arm[arm] for arm in _SELECTOR_ARMS)
    ]
    if missing:
        raise KeyError(
            f"{len(missing):,} queries have no NDCG for some arm (first: "
            f"{missing[:3]}); scoring them 0.0 would teach the selector a "
            "loss that never happened"
        )
    scores = np.array(
        [[float(per_arm[arm][str(q)]) for arm in _SELECTOR_ARMS] for q in query_ids]
    )
    if scores.size == 0:
        return np.zeros(0, dtype=np.int64)
    preference = np.argsort(-scores.mean(axis=0), kind="stable")
    winners = np.isclose(scores, scores.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12)
    return np.array(
        [next(int(i) for i in preference if row[i]) for row in winners],
        dtype=np.int64,
    )


def train_selector(
    frame: pd.DataFrame,
    per_arm: Mapping[str, Mapping[str, float]],
    *,
    seed: int = 0,
    num_boost_round: int = 120,
) -> Any:
    """A multiclass model over which arm wins each query.

    The oracle's advantage is choosing, not mixing: on fold 0 the LLM is
    strictly best on 41.6% of queries, the cross-encoder on 21.3%, Stage 2 on
    17.7%, and 19.4% are tied at the top. This is the only Stage 4 arm whose
    shape matches that.
    """
    import lightgbm as lgb

    features = _selector_features(frame)
    target = selector_targets(per_arm, list(features["query_id"]))
    dataset = lgb.Dataset(
        features[list(SELECTOR_FEATURES)],
        label=target,
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


def cross_fit_select(
    frame: pd.DataFrame,
    per_arm: Mapping[str, Mapping[str, float]],
    arm_orderings: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    n_folds: int = 2,
    seed: int = 0,
    _return_parts: bool = False,
):
    """Out-of-fold selector routes: no query is routed by a selector that saw it.

    The selector's target is which arm scored best on each query - the label
    in another form - so it leaks on fold 0 exactly as the combiner would.
    Same partition as cross_fit_predict, via _query_parts.
    """
    parts = _query_parts(frame, n_folds, seed)
    if _return_parts:
        return parts
    ids = frame["query_id"].astype(str)
    out: dict[str, list[str]] = {}
    for part in parts:
        held = ids.isin(set(part)).to_numpy()
        selector = train_selector(frame.loc[~held], per_arm, seed=seed)
        out.update(selector_ordering(selector, frame.loc[held], arm_orderings))
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
Expected: PASS, 23 tests.

- [ ] **Step 5: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 6: Commit**

```bash
git add src/blend.py tests/test_blend.py
git commit -m "Add the fixed-weight, learned and selector blend strategies"
```

**Landed 2026-09-23, with one correction made during execution: the
selector's labels.** As written, `train_selector` labelled each query with
`np.argmax` over the three arms' NDCG, and `argmax` breaks ties toward the
first-listed arm — `stage2`, the weakest. Measured on the real fold-0 frame
before any selector was trained: **19.4%** of queries are tied at the top
(11.2% two-way, 8.2% all three), so the labels said `stage2` on **31.7%** of
queries when it is strictly best on **17.7%** — 14.0% of all queries taught
"route to the coarse ranker" by list order alone. A tied query scores the same
whichever tied arm routes it, so it should teach the default instead. The new
`selector_targets` breaks ties toward the tied arm with the best mean over the
training queries, and raises on a query with no NDCG where the old code scored
it 0.0. The labels are now `llm` 59.8%, `ce` 22.5%, `stage2` 17.7%. The same
`argmax` artefact was in this plan's README ("Stage 2 best on 31.7%"), now
corrected there too.

This was decided on training labels, before any Stage 4 arm was scored — it is
a label-construction fix, not tuning toward a win, and whether the selector
now beats the LLM is still Task 3's question. Smoke-tested on the real fold-0
frame without computing NDCG: `fixed_weight_ordering`, `cross_fit_predict` and
`cross_fit_select` each return a valid permutation of every one of the 4,130
windows through `spliced_run` (2 s, 5 s and 8 s). Fast suite 646 passed, 27
deselected.

---

## Phase 1 Gate

Phase 2 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the real signal-frame completeness check and the reproduction of Plan 6's three arms from the frame.
- [ ] `data/features/stage-signals-{fold0,test-sample}.parquet` exist, cover 4,130 and 2,000 queries, and `require_full_coverage` passes on both.
- [ ] `llm_orderings_from_cache` returns its misses; the frame flags exactly the 1 + 2 malformed-answer windows Plan 6 counted, and `check_fallback_share` refuses a scope error.
- [ ] `cross_fit_predict` and `cross_fit_select` partition by `query_id` through the same `_query_parts`, and the combiner's scores differ from an in-sample fit on the same frame.
- [ ] `src/blend.py` reads no file, imports no store and issues no API call.
- [ ] `rerank` still returns orderings, and `window_scores` is the only way to see a logit.

Then: [Phase 2 — The Blend Report](phase-2-the-blend-report.md).
