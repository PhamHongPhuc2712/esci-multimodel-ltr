# Phase 1 — The Window

**Plan 6 of 7 · Phase 1 of 3 · Tasks 1–2.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into.

**Delivers:** `src/stage2_scores.py` and `src/rerank_window.py` — Plan 5's
ordering persisted to disk, and the top-K splice both Stage 3 arms re-rank
through.

**Needs on disk:** `data/features/{train,test}.parquet` from Plan 5.

**Owns Review Focus item 1** (the untouched tail outranking the re-ranked
window).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **The re-ranked window must sit strictly above the untouched tail.** Stage 3 scores are on a different scale from Stage 2's.
- **The cross-encoder trains on folds 2/3/4 only**, so the Stage 2 scores for those folds are in-sample and must be marked as such.
- **Every reported NDCG comes from `src.metrics`.**
- **Never tune on test.** `K` is chosen on fold 1.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: Persist Plan 5's ordering

**Files:**
- Create: `src/stage2_scores.py`
- Create: `tests/test_stage2_scores.py`

**Interfaces:**
- Consumes: `src.ranker.{train_ranker, predict, folds, TRAIN_FOLDS, EARLY_STOP_FOLD, REPORT_FOLD}` (Plan 5), `src.feature_matrix.ALL_FEATURES` (Plan 5).
- Produces:
  - `src.stage2_scores.STAGE2_COLUMNS: tuple[str, ...]` = `("query_id", "product_id", "stage2_score", "in_sample")`
  - `src.stage2_scores.DEFAULT_DIR: Path` = `Path("data/features")`
  - `src.stage2_scores.score_split(train, target, *, features=None, fit_folds=TRAIN_FOLDS, seed=0) -> pd.DataFrame`
  - `src.stage2_scores.load_stage2(split, directory=DEFAULT_DIR) -> pd.DataFrame`
  - `src.stage2_scores.require_out_of_sample(frame) -> pd.DataFrame`
  - CLI: `python -m src.stage2_scores --split {train,test}` writing `data/features/stage2-{split}.parquet`

Plan 5 reported NDCG but never wrote an ordering. `rank_report` trains eleven
arms, prints a table and writes a JSON summary; the per-pair predictions live
only inside that process. Both Stage 3 arms need **the same** Stage 2 ordering,
and re-deriving it independently in two modules is how two arms end up
re-ranking two different windows and the difference gets called a reranker
effect.

**The frozen configuration is Plan 5's, unchanged:** all 47 features,
`lambdarank`, `label_gain` derived from `src.labels`, early-stop on fold 1,
seed 0. This module does not re-tune anything; if it did, Ablation 6 would be
measuring a different Stage 2 from the one Plan 5 reported.

**Corrected during execution: the gradient set differs by split, because Plan
5's did.** This task was written asserting folds 2/3/4 for both splits, and
`src/rank_report.py:309` does not do that — when `--split test` it sets
`fit = train`, fitting *every* train fold while keeping fold 1 as the
early-stop set, which is why every test arm in `docs/results/coarse-rank-test.json`
records `best_iteration: 500` and `"train_folds": "all"`. Measured: fitting
2/3/4 for both reproduces fold 0 at **0.8519** exactly and lands at **0.8559**
on test, 0.0020 under Plan 5's 0.8579 and outside this phase's own 0.0005 gate.
So `score_split` takes `fit_folds`, `_main` passes `None` for test and
`TRAIN_FOLDS` for train, and each headline is reproduced by the model that
produced it. Test is disjoint from every train fold, so the wider fit leaks
nothing; fold 0 must keep the narrow one, since it is the reporting surface.

`in_sample` is derived from the rows actually fitted rather than from a
separate argument. A knob that can disagree with the gradient set defeats the
purpose of the flag — it would have quietly marked only 2/3/4 on a run that
fitted all five.

**Folds 2/3/4 are scored in-sample and marked.** The model trained on them, so
their scores are optimistic. Nothing in Plan 6 uses them — the cross-encoder
fine-tunes on raw judged pairs rather than on windows, precisely to avoid
needing them — but they are written anyway, with `in_sample = True`, because a
missing row is indistinguishable from a bug while a flagged row is not.
`require_out_of_sample` is the guard a consumer calls.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage2_scores.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.ranker import EARLY_STOP_FOLD, REPORT_FOLD, TRAIN_FOLDS
from src.stage2_scores import (
    DEFAULT_DIR,
    STAGE2_COLUMNS,
    load_stage2,
    require_out_of_sample,
    score_split,
)


def _frame(n_queries=40, folds=(0, 1, 2, 3, 4), seed=0):
    """A small feature matrix shaped like data/features/train.parquet."""
    rng = np.random.default_rng(seed)
    rows = []
    q = 0
    for fold in folds:
        for _ in range(n_queries):
            for p in range(6):
                code = int(rng.integers(0, 4))
                rows.append(
                    {
                        "query_id": q,
                        "product_id": f"p{q}_{p}",
                        "label_code": code,
                        "gain": [0.0, 0.01, 0.1, 1.0][code],
                        "qrel": [0, 1, 10, 100][code],
                        "fold": fold,
                        "signal": code + rng.normal(scale=0.3),
                        "noise": rng.normal(),
                    }
                )
            q += 1
    return pd.DataFrame(rows)


FEATURES = ["signal", "noise"]


def test_score_split_returns_the_declared_columns():
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES)
    assert list(out.columns) == list(STAGE2_COLUMNS)


def test_every_target_row_gets_exactly_one_score():
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES)
    assert len(out) == len(frame)
    assert not out.duplicated(subset=["query_id", "product_id"]).any()


def test_training_folds_are_flagged_in_sample():
    # The model trained on 2/3/4, so their scores are optimistic. A flagged row
    # is recoverable; a missing one is indistinguishable from a bug.
    frame = _frame()
    out = score_split(frame, frame, features=FEATURES).merge(
        frame[["query_id", "product_id", "fold"]], on=["query_id", "product_id"]
    )
    assert out.loc[out["fold"].isin(TRAIN_FOLDS), "in_sample"].all()
    assert not out.loc[out["fold"] == REPORT_FOLD, "in_sample"].any()
    assert not out.loc[out["fold"] == EARLY_STOP_FOLD, "in_sample"].any()


def test_a_target_with_no_folds_column_is_all_out_of_sample():
    # The test split carries fold = -1 and was never trained on.
    train = _frame()
    target = _frame(n_queries=10, folds=(-1,), seed=7)
    out = score_split(train, target, features=FEATURES)
    assert not out["in_sample"].any()


def test_scores_order_the_informative_feature():
    frame = _frame(n_queries=200)
    out = score_split(frame, frame, features=FEATURES).merge(
        frame[["query_id", "product_id", "label_code"]], on=["query_id", "product_id"]
    )
    # A model that learned `signal` must score Exact above Irrelevant on average.
    assert (
        out.loc[out["label_code"] == 3, "stage2_score"].mean()
        > out.loc[out["label_code"] == 0, "stage2_score"].mean()
    )


def test_scoring_is_deterministic_for_a_seed():
    frame = _frame()
    a = score_split(frame, frame, features=FEATURES, seed=3)["stage2_score"].to_numpy()
    b = score_split(frame, frame, features=FEATURES, seed=3)["stage2_score"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_require_out_of_sample_drops_the_training_folds():
    frame = pd.DataFrame(
        {
            "query_id": [1, 2],
            "product_id": ["a", "b"],
            "stage2_score": [1.0, 2.0],
            "in_sample": [True, False],
        }
    )
    kept = require_out_of_sample(frame)
    assert kept["query_id"].tolist() == [2]


def test_require_out_of_sample_raises_when_nothing_survives():
    frame = pd.DataFrame(
        {
            "query_id": [1],
            "product_id": ["a"],
            "stage2_score": [1.0],
            "in_sample": [True],
        }
    )
    with pytest.raises(ValueError, match="in-sample"):
        require_out_of_sample(frame)


def test_load_stage2_rejects_an_unknown_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="stage2_scores"):
        load_stage2("train", directory=tmp_path)


def test_the_default_directory_is_the_feature_directory():
    assert DEFAULT_DIR.as_posix().endswith("data/features")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_stage2_scores.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.stage2_scores'`.

- [ ] **Step 3: Write the implementation**

Create `src/stage2_scores.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stage2_scores.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Dump both real orderings**

Run:

```bash
python -m src.stage2_scores --split train
python -m src.stage2_scores --split test
```

Expected: 419,653 rows (250,485 flagged in-sample) and 181,701 rows (0
in-sample); fold-0 NDCG **0.8519** and test NDCG **0.8579**, both within
0.0005 of what Plan 5 reported. A warning here means the frozen configuration
has drifted and must be fixed before anything is built on it.

**Measured on execution: both reproduce exactly**, once the test split is
fitted on every train fold as above. The first run of this step fitted 2/3/4
for both and the warning fired on test at 0.8559 — the drift check earning its
keep on the first thing it was pointed at.

- [ ] **Step 6: Write the data-marked test**

Append to `tests/test_stage2_scores.py`:

```python
@pytest.mark.data
def test_the_persisted_ordering_reproduces_plan_5():
    from src.metrics import ndcg_per_query

    scores = load_stage2("train")
    matrix = pd.read_parquet("data/features/train.parquet")
    assert len(scores) == 419_653
    assert int(scores["in_sample"].sum()) == 250_485

    joined = matrix.merge(scores, on=["query_id", "product_id"])
    fold0 = joined.loc[joined["fold"] == REPORT_FOLD]
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q, p, r, s in zip(
        fold0["query_id"], fold0["product_id"], fold0["qrel"], fold0["stage2_score"]
    ):
        qrels.setdefault(str(q), {})[str(p)] = int(r)
        run.setdefault(str(q), {})[str(p)] = float(s)
    per_query = ndcg_per_query(run, qrels)
    assert sum(per_query.values()) / len(per_query) == pytest.approx(0.8519, abs=0.0005)


@pytest.mark.data
def test_the_test_ordering_is_entirely_out_of_sample():
    scores = load_stage2("test")
    assert len(scores) == 181_701
    assert not scores["in_sample"].any()
    require_out_of_sample(scores)  # must not raise
```

Run: `python -m pytest tests/test_stage2_scores.py -m data -q`
Expected: PASS.

- [ ] **Step 7: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 8: Commit**

```bash
git add src/stage2_scores.py tests/test_stage2_scores.py
git commit -m "Persist the frozen Stage 2 ordering for both Stage 3 arms"
```

---

## Task 2: The top-K window and the splice

**Files:**
- Create: `src/rerank_window.py`
- Create: `tests/test_rerank_window.py`

**Interfaces:**
- Consumes: `src.stage2_scores.load_stage2` (Task 1).
- Produces:
  - `src.rerank_window.DEFAULT_K: int` = `10`
  - `src.rerank_window.Window` (frozen dataclass: `query_id: str`, `window: tuple[str, ...]`, `tail: tuple[str, ...]`)
  - `src.rerank_window.windows(stage2: pd.DataFrame, k: int = DEFAULT_K) -> list[Window]`
  - `src.rerank_window.splice_scores(window_order: Sequence[str], tail: Sequence[str]) -> dict[str, float]`
  - `src.rerank_window.spliced_run(windows, orderings: Mapping[str, Sequence[str]] | None = None) -> dict[str, dict[str, float]]`
  - `src.rerank_window.stage2_run(windows) -> dict[str, dict[str, float]]`

**Review Focus 1 is this module's entire reason to exist.** A cross-encoder
logit can be −8.4 and a Stage 2 `lambdarank` score +3.1. Writing Stage 3 scores
onto the window and leaving Stage 2 scores on the tail interleaves two
orderings that were never on the same scale: a product the reranker actively
demoted can still land above one it never looked at. Nothing raises. NDCG moves
for a reason unrelated to the reranker, and the ablation measures scale
mismatch.

So the splice never mixes scales. It emits **rank-derived scores**: descending
integers from `n - 1` down to `0` over the concatenation of `window_order` and
`tail`. The window is then strictly above the tail by construction, whatever
either stage's native units were, and the tail keeps its Stage 2 order exactly.

**The identity property is the sharpest test available.** Re-ranking a window
with the order it already had must reproduce Stage 2's NDCG to the last
decimal. If it does not, the splice is wrong. `test_the_identity_reranking_is_a_no_op`
pins it on synthetic data and the data-marked test pins it on the real
ordering.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rerank_window.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.metrics import ndcg_per_query
from src.rerank_window import (
    DEFAULT_K,
    Window,
    splice_scores,
    spliced_run,
    stage2_run,
    windows,
)


def _stage2():
    # Query 1: five candidates, descending score. Query 2: three.
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "product_id": ["a", "b", "c", "d", "e", "x", "y", "z"],
            "stage2_score": [5.0, 4.0, 3.0, 2.0, 1.0, 9.0, 8.0, 7.0],
        }
    )


# --- carving the window -----------------------------------------------------

def test_the_window_is_the_top_k_in_stage_2_order():
    w = {x.query_id: x for x in windows(_stage2(), k=3)}
    assert w["1"].window == ("a", "b", "c")
    assert w["1"].tail == ("d", "e")


def test_the_tail_keeps_stage_2_order():
    w = {x.query_id: x for x in windows(_stage2(), k=2)}
    assert w["1"].tail == ("c", "d", "e")


def test_a_short_query_is_all_window_and_no_tail():
    w = {x.query_id: x for x in windows(_stage2(), k=10)}
    assert w["2"].window == ("x", "y", "z")
    assert w["2"].tail == ()


def test_every_query_gets_exactly_one_window():
    assert len(windows(_stage2(), k=3)) == 2


def test_ties_break_deterministically():
    # Two runs of the same code must carve the same window, or two arms are
    # re-ranking different candidate sets.
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": ["b", "a", "c"],
            "stage2_score": [1.0, 1.0, 0.0],
        }
    )
    first = windows(frame, k=2)[0].window
    second = windows(frame.iloc[::-1].reset_index(drop=True), k=2)[0].window
    assert first == second == ("a", "b")


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        windows(_stage2(), k=0)


def test_the_default_window_is_ten():
    # Measured: an oracle reorder of the top-10 holds +0.1014 of the +0.1481
    # total headroom, at a quarter of the LLM cost of the whole list.
    assert DEFAULT_K == 10


# --- Review Focus 1: the splice --------------------------------------------

def test_the_window_sits_strictly_above_the_tail():
    # A cross-encoder logit can be -8 while a lambdarank score is +3. If the
    # two scales were mixed, a demoted window item could outrank a tail item
    # the reranker never saw.
    scores = splice_scores(["c", "a", "b"], ["d", "e"])
    assert min(scores["c"], scores["a"], scores["b"]) > max(scores["d"], scores["e"])


def test_the_splice_preserves_the_given_window_order():
    scores = splice_scores(["c", "a", "b"], [])
    assert sorted(scores, key=lambda d: -scores[d]) == ["c", "a", "b"]


def test_the_splice_preserves_the_tail_order():
    scores = splice_scores(["a"], ["d", "e", "f"])
    ordered = sorted(scores, key=lambda d: -scores[d])
    assert ordered == ["a", "d", "e", "f"]


def test_the_splice_scores_every_document_once():
    scores = splice_scores(["a", "b"], ["c"])
    assert set(scores) == {"a", "b", "c"}
    assert len(set(scores.values())) == 3


def test_the_splice_rejects_a_document_in_both_halves():
    with pytest.raises(ValueError, match="both the window and the tail"):
        splice_scores(["a", "b"], ["b"])


def test_the_splice_of_nothing_is_nothing():
    assert splice_scores([], []) == {}


# --- the run ----------------------------------------------------------------

def _qrels():
    return {
        "1": {"a": 100, "b": 0, "c": 10, "d": 0, "e": 100},
        "2": {"x": 0, "y": 100, "z": 10},
    }


def test_the_identity_reranking_is_a_no_op():
    # The sharpest test of the splice: re-ranking each window with the order it
    # already had must reproduce Stage 2's NDCG exactly.
    ws = windows(_stage2(), k=3)
    identity = {w.query_id: list(w.window) for w in ws}
    before = ndcg_per_query(stage2_run(ws), _qrels())
    after = ndcg_per_query(spliced_run(ws, identity), _qrels())
    assert before == pytest.approx(after)


def test_a_query_with_no_reranking_keeps_its_stage_2_order():
    ws = windows(_stage2(), k=3)
    run = spliced_run(ws, {"2": ["z", "y", "x"]})
    ordered = sorted(run["1"], key=lambda d: -run["1"][d])
    assert ordered == ["a", "b", "c", "d", "e"]


def test_a_reranking_changes_only_the_window():
    ws = windows(_stage2(), k=3)
    run = spliced_run(ws, {"1": ["c", "b", "a"]})
    ordered = sorted(run["1"], key=lambda d: -run["1"][d])
    assert ordered == ["c", "b", "a", "d", "e"]


def test_a_reranking_that_is_not_a_permutation_raises():
    # Dropping or inventing a candidate silently changes what was ranked.
    ws = windows(_stage2(), k=3)
    with pytest.raises(ValueError, match="permutation"):
        spliced_run(ws, {"1": ["a", "b"]})
    with pytest.raises(ValueError, match="permutation"):
        spliced_run(ws, {"1": ["a", "b", "zzz"]})


def test_a_reranking_for_an_unknown_query_raises():
    ws = windows(_stage2(), k=3)
    with pytest.raises(KeyError, match="99"):
        spliced_run(ws, {"99": ["a"]})


def test_the_run_covers_every_document():
    ws = windows(_stage2(), k=2)
    run = spliced_run(ws)
    assert sum(len(docs) for docs in run.values()) == 8


def test_stage2_run_matches_the_original_scores():
    ws = windows(_stage2(), k=3)
    ordered = sorted(stage2_run(ws)["1"], key=lambda d: -stage2_run(ws)["1"][d])
    assert ordered == ["a", "b", "c", "d", "e"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rerank_window.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.rerank_window'`.

- [ ] **Step 3: Write the implementation**

Create `src/rerank_window.py`:

```python
"""The top-K window both Stage 3 arms re-rank, and the splice that puts it back.

PROJECT_SPEC.md §4.3 compares the two fine-rank variants "on the top-K from
Stage 2". This module owns what that means, because it is the one piece of
logic the two arms share and the one place a scale mismatch can corrupt a run.

**Scores are never mixed across stages.** A cross-encoder logit can be -8.4
while a lambdarank score is +3.1. Writing Stage 3 scores onto the window and
leaving Stage 2 scores on the tail interleaves two orderings that were never on
the same scale, so a product the reranker actively demoted can still land above
one it never looked at - silently, with NDCG moving for a reason that has
nothing to do with the reranker.

So the splice emits **rank-derived scores**: descending integers from n - 1
down to 0 over window ++ tail. The window is strictly above the tail by
construction, whatever either stage's native units were, and the tail keeps its
Stage 2 order exactly.

K = 10 by default. Measured on fold 0: an oracle reorder of the top-10 reaches
0.9533 against Stage 2's 0.8519 - two thirds of the +0.1481 total headroom - at
roughly a quarter of the LLM cost of ranking the whole list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

DEFAULT_K = 10


@dataclass(frozen=True)
class Window:
    """One query's top-K candidates and the tail no reranker will touch."""

    query_id: str
    window: tuple[str, ...]
    tail: tuple[str, ...]

    @property
    def documents(self) -> tuple[str, ...]:
        return self.window + self.tail


def windows(stage2: pd.DataFrame, k: int = DEFAULT_K) -> list[Window]:
    """Carve every query's Stage 2 ordering into a top-k window and a tail.

    Ties break on product_id so two runs of the same code carve the same
    window; otherwise the two arms would be re-ranking different candidate
    sets and the difference would be reported as a reranker effect.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    ordered = stage2.sort_values(
        ["query_id", "stage2_score", "product_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    out: list[Window] = []
    for query_id, group in ordered.groupby("query_id", sort=True):
        documents = [str(p) for p in group["product_id"]]
        out.append(
            Window(
                query_id=str(query_id),
                window=tuple(documents[:k]),
                tail=tuple(documents[k:]),
            )
        )
    return out


def splice_scores(
    window_order: Sequence[str], tail: Sequence[str]
) -> dict[str, float]:
    """Rank-derived scores placing `window_order` above `tail`, both in order.

    Never a blend of two stages' scales - see the module docstring.
    """
    overlap = set(window_order) & set(tail)
    if overlap:
        raise ValueError(
            f"{sorted(overlap)[:3]} appear in both the window and the tail; a "
            "document can only hold one rank"
        )
    documents = list(window_order) + list(tail)
    total = len(documents)
    return {doc: float(total - position) for position, doc in enumerate(documents)}


def stage2_run(windows_: Sequence[Window]) -> dict[str, dict[str, float]]:
    """The Stage 2 ordering itself, as a run. The baseline arm."""
    return {w.query_id: splice_scores(w.window, w.tail) for w in windows_}


def spliced_run(
    windows_: Sequence[Window],
    orderings: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Apply per-query re-rankings of the window, leaving every tail alone.

    A query absent from `orderings` keeps its Stage 2 order, which is what a
    reranker that declined to answer - a malformed LLM response, a timeout -
    must fall back to.
    """
    orderings = dict(orderings or {})
    known = {w.query_id for w in windows_}
    unknown = set(orderings) - known
    if unknown:
        raise KeyError(
            f"re-rankings given for queries that have no window: "
            f"{sorted(unknown)[:3]}"
        )

    run: dict[str, dict[str, float]] = {}
    for w in windows_:
        order = orderings.get(w.query_id)
        if order is None:
            order = list(w.window)
        elif sorted(order) != sorted(w.window):
            raise ValueError(
                f"query {w.query_id}: the re-ranking is not a permutation of "
                f"its window ({len(order)} documents against {len(w.window)}); "
                "a dropped candidate silently leaves the ranking and a "
                "duplicated one silently occupies two ranks"
            )
        run[w.query_id] = splice_scores(order, w.tail)
    return run
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_rerank_window.py -q`
Expected: PASS, 20 tests.

- [ ] **Step 5: Write the data-marked identity check**

This is the one that matters: on the real fold-0 ordering, an identity
re-ranking must reproduce Plan 5's 0.8519 exactly.

Append to `tests/test_rerank_window.py`:

```python
@pytest.mark.data
def test_the_identity_splice_reproduces_plan_5_on_the_real_ordering():
    from src.rank_report import qrels_from_frame
    from src.ranker import REPORT_FOLD
    from src.stage2_scores import load_stage2

    matrix = pd.read_parquet("data/features/train.parquet")
    fold0 = matrix.loc[matrix["fold"] == REPORT_FOLD]
    scores = load_stage2("train")
    scores = scores.loc[
        scores["query_id"].isin(set(fold0["query_id"]))
        & scores["product_id"].isin(set(fold0["product_id"]))
    ]
    joined = fold0[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    assert len(joined) == len(fold0)

    ws = windows(joined, k=DEFAULT_K)
    qrels = qrels_from_frame(fold0)
    base = ndcg_per_query(stage2_run(ws), qrels)
    identity = ndcg_per_query(
        spliced_run(ws, {w.query_id: list(w.window) for w in ws}), qrels
    )
    mean_base = sum(base.values()) / len(base)
    assert mean_base == pytest.approx(0.8519, abs=0.0005)
    # Not approx: an identity splice that moves NDCG at all is a broken splice.
    assert base == identity


@pytest.mark.data
def test_the_real_windows_cover_every_judged_pair():
    from src.ranker import REPORT_FOLD
    from src.stage2_scores import load_stage2

    matrix = pd.read_parquet("data/features/train.parquet")
    fold0 = matrix.loc[matrix["fold"] == REPORT_FOLD]
    scores = load_stage2("train")
    joined = fold0[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    ws = windows(joined, k=DEFAULT_K)
    assert len(ws) == fold0["query_id"].nunique()
    assert sum(len(w.documents) for w in ws) == len(fold0)
    # Measured: fold 0 has a median of 16 candidates, so most queries have a
    # tail at k=10 but a meaningful minority do not.
    assert any(w.tail for w in ws)
    assert any(not w.tail for w in ws)
```

Run: `python -m pytest tests/test_rerank_window.py -m data -q`
Expected: PASS.

- [ ] **Step 6: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 7: Commit**

```bash
git add src/rerank_window.py tests/test_rerank_window.py
git commit -m "Carve the Stage 2 top-K window and splice re-rankings above the tail"
```

---

## Phase 1 Gate

**Passed 2026-09-22.** One correction landed on the way: the frozen
configuration's gradient set differs by split, because Plan 5's did. See Task 1.

- [x] `python -m pytest` passes with no failures and no new skips. — 518 passed (485 before this phase), 23 deselected, 0 skipped.
- [x] `python -m pytest -m data` passes, including the identity-splice check. — 22 passed in 6m14s (17 before this phase).
- [x] `data/features/stage2-train.parquet` has 419,653 rows with 250,485 flagged in-sample, and `stage2-test.parquet` has 181,701 with none.
- [x] The persisted ordering reproduces Plan 5's fold-0 NDCG of 0.8519 and test NDCG of 0.8579 to within 0.0005. — **both reproduce exactly** at 4 dp, once test is fitted on every train fold as `src/rank_report.py` does. The first attempt fitted folds 2/3/4 for both and the drift check caught test at 0.8559.
- [x] An identity re-ranking through `spliced_run` reproduces `stage2_run`'s per-query NDCG **exactly**, on the real fold-0 ordering. — asserted with `==` on the per-query dicts, not `approx`, over all 4,130 fold-0 queries.
- [x] `src/rerank_window.py` imports no model, no store and no LightGBM. — it imports `pandas`, `dataclasses` and `collections.abc`, nothing else.

Then: [Phase 2 — The Cross-Encoder](phase-2-the-cross-encoder.md).
