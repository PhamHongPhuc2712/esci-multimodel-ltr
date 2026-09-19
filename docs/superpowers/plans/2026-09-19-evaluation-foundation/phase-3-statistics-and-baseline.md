# Phase 3 — Statistics and the Baseline

**Plan 1 of 7 · Phase 3 of 3 · Tasks 7–10.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 2 gate](phase-2-the-data.md#phase-2-gate) passes.

**Delivers:** bootstrap and paired-delta confidence intervals, the frozen
query-level validation folds, the report and CLI that cannot print a headline
NDCG without the floor attached, and the reproduced zero-shot SBERT baseline
that proves the harness.

**Needs on disk:** the parquet from Phase 2, plus the `baselines` extra
(`sentence-transformers` + torch). Task 10 Step 5 encodes 8,956 queries and
164,900 product titles — roughly 10 minutes on CPU.

**This phase is where the plan is falsifiable.** `PROJECT_SPEC.md` §5 puts
SBERT_text zero-shot at **0.8292** on this split with this metric. If Task 10
lands in `[0.820, 0.840]`, the loader, the metric and the floor are all
trustworthy. If it produces 0.75 or 0.91, one of them is wrong and every later
plan in the series is built on sand — so the band does not move, the code does.

> **Task 9 Step 4 reopens `src/floor.py` from Phase 2** to add a `per_query`
> field. Do not implement `src/report.py` against the Phase 2 shape of
> `FloorResult`; Task 9's Interfaces block names `per_query` as a dependency,
> and the step carries both the dataclass change and its covering test.

**Task 9 Step 7 also edits `CLAUDE.md`**, replacing its "no test suite
configured yet" Commands block with the real invocations.

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Report bootstrap confidence intervals over queries, not bare point estimates.** A method that ties the baseline is a legitimate, reportable result — which is only expressible if the interval can contain zero.
- **There is no official validation split.** Carve one from train by `query_id` and freeze it before tuning anything. Never split within a query group.
- **Never tune on test.**
- **Features must be query×product, never product-alone-from-labels.** 34,756 products appear in both train and test; the official split is query-level, so this is legitimate, but product-level target encoding leaks train labels into test.
- **Compute the random floor, never quote it.** Every headline NDCG this phase prints carries the measured floor beside it.
- **Do not commit datasets.** Frozen split artifacts live at `splits/` in the repo root, deliberately outside `data/`, so they can be committed without fighting the ignore rules. The folds must be committed — that is what "frozen" means here.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line.

---

## Task 7: Bootstrap confidence intervals

**Files:**
- Create: `src/bootstrap.py`
- Create: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: nothing (pure functions over `Mapping[str, float]`).
- Produces:
  - `src.bootstrap.Interval(point: float, low: float, high: float)`
  - `src.bootstrap.bootstrap_ci(per_query: Mapping[str, float], *, n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> Interval`
  - `src.bootstrap.paired_delta_ci(a: Mapping[str, float], b: Mapping[str, float], *, n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> Interval`

`paired_delta_ci` resamples query *ids* once per replicate and applies the same resample to both methods. Resampling the two independently would widen the interval by the between-query variance that the pairing is there to cancel, which is exactly the variance that makes a 2-point NDCG difference hard to call.

- [x] **Step 1: Write the failing test**

Create `tests/test_bootstrap.py`:

```python
import pytest

from src.bootstrap import bootstrap_ci, paired_delta_ci


def test_point_estimate_is_the_plain_mean():
    per_query = {"q1": 0.2, "q2": 0.4, "q3": 0.9}
    assert bootstrap_ci(per_query, n_resamples=200).point == pytest.approx(0.5)


def test_constant_scores_give_a_zero_width_interval():
    per_query = {f"q{i}": 0.8 for i in range(50)}
    interval = bootstrap_ci(per_query, n_resamples=200)
    assert interval.low == pytest.approx(0.8)
    assert interval.high == pytest.approx(0.8)


def test_interval_brackets_the_point_estimate():
    per_query = {f"q{i}": i / 100 for i in range(100)}
    interval = bootstrap_ci(per_query, n_resamples=500)
    assert interval.low <= interval.point <= interval.high


def test_more_queries_narrow_the_interval():
    small = {f"q{i}": (i % 10) / 10 for i in range(20)}
    large = {f"q{i}": (i % 10) / 10 for i in range(2000)}
    narrow = bootstrap_ci(large, n_resamples=500)
    wide = bootstrap_ci(small, n_resamples=500)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_same_seed_reproduces_the_interval():
    per_query = {f"q{i}": (i % 7) / 7 for i in range(100)}
    assert bootstrap_ci(per_query, n_resamples=300, seed=5) == bootstrap_ci(
        per_query, n_resamples=300, seed=5
    )


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci({}, n_resamples=10)


# --- paired deltas ----------------------------------------------------------

def test_identical_methods_have_a_zero_delta_and_a_zero_width_interval():
    per_query = {f"q{i}": (i % 5) / 5 for i in range(100)}
    interval = paired_delta_ci(per_query, per_query, n_resamples=300)
    assert interval.point == pytest.approx(0.0)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(0.0)


def test_a_uniform_improvement_gives_an_interval_strictly_above_zero():
    baseline = {f"q{i}": 0.5 for i in range(200)}
    better = {f"q{i}": 0.6 for i in range(200)}
    interval = paired_delta_ci(better, baseline, n_resamples=500)
    assert interval.point == pytest.approx(0.1)
    assert interval.low > 0


def test_a_noisy_tie_gives_an_interval_that_straddles_zero():
    # The spec calls a method that ties the baseline a legitimate, reportable
    # result, which is only expressible if the interval can contain zero.
    baseline = {f"q{i}": (i % 11) / 11 for i in range(400)}
    noisy = {f"q{i}": (i % 11) / 11 + (0.02 if i % 2 else -0.02) for i in range(400)}
    interval = paired_delta_ci(noisy, baseline, n_resamples=500)
    assert interval.low < 0 < interval.high


def test_pairing_is_tighter_than_treating_the_methods_as_independent():
    # Both methods vary a lot across queries but differ by a constant. Pairing
    # cancels the shared variance; without it the interval would be wide.
    baseline = {f"q{i}": (i % 50) / 50 for i in range(500)}
    better = {f"q{i}": (i % 50) / 50 + 0.05 for i in range(500)}
    interval = paired_delta_ci(better, baseline, n_resamples=500)
    assert interval.high - interval.low < 1e-9


def test_mismatched_query_sets_raise():
    a = {"q1": 0.5, "q2": 0.5}
    b = {"q1": 0.5, "q3": 0.5}
    with pytest.raises(ValueError, match="same queries"):
        paired_delta_ci(a, b, n_resamples=10)
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.bootstrap'`.

- [x] **Step 3: Write `src/bootstrap.py`**

```python
"""Bootstrap confidence intervals over queries.

With ~9K test queries and a random floor near 0.748, the gap between a real
improvement and noise can be a couple of NDCG points, so every reported number
carries an interval.

Method comparisons use the paired form: one resample of query ids per
replicate, applied to both methods. Resampling independently would add back
the between-query variance that pairing cancels — the variance that dominates
here, since a hard query is hard for every method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


def _percentiles(replicates: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def bootstrap_ci(
    per_query: Mapping[str, float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for the mean over queries."""
    if not per_query:
        raise ValueError("cannot bootstrap an empty set of per-query scores")
    values = np.asarray([per_query[q] for q in sorted(per_query)], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    replicates = values[draws].mean(axis=1)
    low, high = _percentiles(replicates, alpha)
    return Interval(point=float(values.mean()), low=low, high=high)


def paired_delta_ci(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Percentile bootstrap interval for mean(a) - mean(b), paired by query."""
    if set(a) != set(b):
        raise ValueError(
            "paired comparison needs the same queries on both sides; "
            f"{len(set(a) - set(b))} only in a, {len(set(b) - set(a))} only in b"
        )
    if not a:
        raise ValueError("cannot bootstrap an empty set of per-query scores")
    queries = sorted(a)
    deltas = np.asarray([a[q] - b[q] for q in queries], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(n_resamples, len(deltas)))
    replicates = deltas[draws].mean(axis=1)
    low, high = _percentiles(replicates, alpha)
    return Interval(point=float(deltas.mean()), low=low, high=high)
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: PASS, 11 tests.

- [x] **Step 5: Commit**

```bash
git add src/bootstrap.py tests/test_bootstrap.py
git commit -m "Add bootstrap and paired-delta confidence intervals over queries"
```
---

## Task 8: Frozen query-level validation folds

**Files:**
- Create: `src/splits.py`
- Create: `tests/test_splits.py`
- Create: `splits/val_folds.csv` (generated in Step 5, committed)
- Create: `splits/val_folds.sha256` (generated in Step 5, committed)

**Interfaces:**
- Consumes: `src.dataset.load_split(split) -> EsciSplit`.
- Produces:
  - `src.splits.N_FOLDS: int` (5), `src.splits.FOLDS_PATH: Path`, `src.splits.SALT: str`
  - `src.splits.fold_for_query_id(query_id: int, n_folds: int = N_FOLDS) -> int`
  - `src.splits.assign_folds(query_ids: Iterable[int], n_folds: int = N_FOLDS) -> pd.DataFrame` with columns `query_id`, `fold`
  - `src.splits.freeze_folds(folds: pd.DataFrame, path: Path = FOLDS_PATH) -> str` (returns the sha256 and writes the sidecar)
  - `src.splits.load_folds(path: Path = FOLDS_PATH) -> pd.DataFrame`
  - `src.splits.split_train_val(judgements: pd.DataFrame, folds: pd.DataFrame, val_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]`

Folds come from a salted SHA-256 of the query id rather than `sklearn.GroupKFold`, for two reasons: the assignment must not change when scikit-learn changes its internal group ordering, and it must not depend on the row order of the frame it was computed from. A frozen split that silently moves is worse than no frozen split.

`splits/` is deliberately outside `data/`, which `.gitignore` blocks. The folds must be committed — that is what "frozen" means here.

- [ ] **Step 1: Write the failing test**

Create `tests/test_splits.py`:

```python
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.splits import (
    FOLDS_PATH,
    N_FOLDS,
    assign_folds,
    fold_for_query_id,
    freeze_folds,
    load_folds,
    split_train_val,
)


def test_fold_is_in_range():
    assert all(0 <= fold_for_query_id(q) < N_FOLDS for q in range(1000))


def test_fold_assignment_is_stable_across_processes():
    # Python salts str.__hash__ per process, so a fold built on hash() would
    # silently reshuffle between runs and quietly leak validation queries into
    # training. This is the test that pins it.
    code = "from src.splits import fold_for_query_id; print(fold_for_query_id(12345))"
    outputs = []
    for hash_seed in ("0", "1", "random"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
        )
        outputs.append(result.stdout.strip())
    assert len(set(outputs)) == 1, outputs


def test_fold_assignment_does_not_depend_on_input_order():
    forwards = assign_folds(range(500))
    backwards = assign_folds(reversed(range(500)))
    merged = forwards.merge(backwards, on="query_id", suffixes=("_f", "_b"))
    assert (merged["fold_f"] == merged["fold_b"]).all()


def test_every_query_is_in_exactly_one_fold():
    folds = assign_folds(range(10_000))
    assert len(folds) == 10_000
    assert folds["query_id"].is_unique


def test_folds_are_roughly_balanced():
    folds = assign_folds(range(20_000))
    sizes = folds["fold"].value_counts()
    assert len(sizes) == N_FOLDS
    assert sizes.max() - sizes.min() < 0.05 * sizes.mean()


def test_split_train_val_never_puts_a_query_on_both_sides():
    judgements = pd.DataFrame(
        {
            "query_id": [q for q in range(200) for _ in range(3)],
            "product_id": [f"B{i}" for i in range(600)],
        }
    )
    folds = assign_folds(range(200))
    train, val = split_train_val(judgements, folds, val_fold=0)
    assert set(train["query_id"]).isdisjoint(set(val["query_id"]))
    assert len(train) + len(val) == len(judgements)
    assert len(val) > 0


def test_every_fold_can_serve_as_validation():
    judgements = pd.DataFrame({"query_id": list(range(500)), "product_id": ["B"] * 500})
    folds = assign_folds(range(500))
    for val_fold in range(N_FOLDS):
        train, val = split_train_val(judgements, folds, val_fold)
        assert len(val) > 0
        assert len(train) > 0


def test_unknown_validation_fold_raises():
    folds = assign_folds(range(100))
    judgements = pd.DataFrame({"query_id": list(range(100)), "product_id": ["B"] * 100})
    with pytest.raises(ValueError, match="val_fold"):
        split_train_val(judgements, folds, val_fold=N_FOLDS)


def test_judgement_for_an_unassigned_query_raises():
    # A query in the judgements but missing from the frozen folds would
    # otherwise be dropped from both sides without a word.
    folds = assign_folds(range(10))
    judgements = pd.DataFrame({"query_id": [3, 999], "product_id": ["B1", "B2"]})
    with pytest.raises(ValueError, match="not in the frozen folds"):
        split_train_val(judgements, folds, val_fold=0)


def test_freeze_and_load_round_trip(tmp_path):
    folds = assign_folds(range(100))
    path = tmp_path / "val_folds.csv"
    digest = freeze_folds(folds, path)
    assert (tmp_path / "val_folds.sha256").read_text().split()[0] == digest
    loaded = load_folds(path)
    assert loaded.equals(folds.sort_values("query_id").reset_index(drop=True))


def test_freezing_twice_produces_the_same_bytes(tmp_path):
    folds = assign_folds(range(100))
    first = freeze_folds(folds, tmp_path / "a.csv")
    second = freeze_folds(folds.sample(frac=1, random_state=0), tmp_path / "b.csv")
    assert first == second


def test_load_detects_a_tampered_file(tmp_path):
    folds = assign_folds(range(100))
    path = tmp_path / "val_folds.csv"
    freeze_folds(folds, path)
    path.write_text(path.read_text() + "999999,0\n")
    with pytest.raises(ValueError, match="sha256"):
        load_folds(path)


@pytest.mark.data
def test_committed_folds_still_match_the_real_train_split():
    from src.dataset import load_split

    committed = load_folds(FOLDS_PATH)
    recomputed = assign_folds(load_split("train").judgements["query_id"].unique())
    assert committed.equals(recomputed.sort_values("query_id").reset_index(drop=True))
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_splits.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.splits'`.

- [x] **Step 3: Write `src/splits.py`**

```python
"""The frozen validation split, carved from train by query_id.

ESCI ships no official validation split, and the official train/test split is
query-level. Splitting within a query group would put some of a query's judged
products in train and the rest in validation, which leaks the answer.

Folds come from a salted SHA-256 of the query id rather than a library's
K-fold, so the assignment survives a scikit-learn upgrade and does not depend
on the row order of the frame it was computed from. The result is committed to
splits/val_folds.csv with a checksum: "frozen" has to mean something.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

N_FOLDS = 5
SALT = "esci-multimodel-ltr/val-folds/v1"
FOLDS_PATH = Path("splits/val_folds.csv")


def fold_for_query_id(query_id: int, n_folds: int = N_FOLDS) -> int:
    """Deterministic fold for one query id.

    SHA-256, not hash(): Python salts str.__hash__ per process, so a fold built
    on hash() reshuffles between runs.
    """
    digest = hashlib.sha256(f"{SALT}:{int(query_id)}".encode()).hexdigest()
    return int(digest[:16], 16) % n_folds


def assign_folds(query_ids: Iterable[int], n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Fold assignment for a set of query ids, sorted by query id."""
    unique = sorted({int(q) for q in query_ids})
    return pd.DataFrame(
        {
            "query_id": unique,
            "fold": [fold_for_query_id(q, n_folds) for q in unique],
        }
    )


def freeze_folds(folds: pd.DataFrame, path: Path = FOLDS_PATH) -> str:
    """Write the folds and a sha256 sidecar. Returns the digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = folds.sort_values("query_id").reset_index(drop=True)
    body = ordered.to_csv(index=False, lineterminator="\n")
    path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def load_folds(path: Path = FOLDS_PATH) -> pd.DataFrame:
    """Load the frozen folds, verifying the checksum."""
    path = Path(path)
    body = path.read_text(encoding="utf-8")
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{path} does not match its recorded sha256 "
                f"({actual} != {expected}); the frozen validation split has "
                "been modified, which invalidates every tuning decision made "
                "against it"
            )
    return pd.read_csv(path)


def split_train_val(
    judgements: pd.DataFrame, folds: pd.DataFrame, val_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split judgements into (train, validation) at a query-group boundary."""
    available = set(folds["fold"])
    if val_fold not in available:
        raise ValueError(
            f"val_fold {val_fold} is not one of {sorted(available)}"
        )
    assignment = dict(zip(folds["query_id"], folds["fold"]))
    missing = set(judgements["query_id"]) - assignment.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} query ids are not in the frozen folds "
            f"(for example {sorted(missing)[:3]}); they would be dropped from "
            "both sides silently"
        )
    is_val = judgements["query_id"].map(assignment) == val_fold
    return (
        judgements.loc[~is_val].reset_index(drop=True),
        judgements.loc[is_val].reset_index(drop=True),
    )
```

- [x] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_splits.py -v
```

Expected: PASS, 12 tests; the real-data test is deselected.

- [x] **Step 5: Freeze the folds against the real train split and commit them**

```bash
python -c "
from src.dataset import load_split
from src.splits import assign_folds, freeze_folds
folds = assign_folds(load_split('train').judgements['query_id'].unique())
print('queries:', len(folds))
print('fold sizes:', folds['fold'].value_counts().sort_index().to_dict())
print('sha256:', freeze_folds(folds))
"
python -m pytest tests/test_splits.py -v -m data
```

Expected: 20,888 queries, five fold sizes near 4,178 each, and the marked test passes.

- [x] **Step 6: Commit**

```bash
git add src/splits.py tests/test_splits.py splits/val_folds.csv splits/val_folds.sha256
git commit -m "Freeze query-level validation folds carved from the train split"
```
---

## Task 9: Evaluation report and CLI

**Files:**
- Create: `src/report.py`
- Create: `src/cli.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: `src.metrics.ndcg_per_query`, `src.floor.random_floor` and `FloorResult` (including its `per_query` field), `src.bootstrap.bootstrap_ci`, `src.bootstrap.paired_delta_ci`, `src.bootstrap.Interval`, `src.runs.read_run`, `src.runs.qrels_from_judgements`, `src.dataset.load_split`, `src.splits.assign_folds`, `src.splits.freeze_folds`.
- Produces:
  - `src.report.EvaluationReport(run_tag, split, n_queries, ndcg: Interval, floor_mean: float, lift_over_floor: Interval)` with `.to_dict() -> dict`
  - `src.report.evaluate_run(run, qrels, *, run_tag, split, n_floor_trials=100, seed=0) -> EvaluationReport`
  - `src.report.format_report(report: EvaluationReport) -> str`
  - `src.report.write_report(report: EvaluationReport, path: Path) -> None`
  - `python -m src.cli evaluate --run PATH --split {train,test} --tag NAME`
  - `python -m src.cli floor --split {train,test}`
  - `python -m src.cli freeze-splits`

`lift_over_floor` is a **paired** delta: the run's per-query NDCG minus the random baseline's per-query mean NDCG, over the same queries. That is the number the spec means by "always report against the random floor" — a bare difference of two means hides whether the gain is uniform or comes from a handful of queries.

- [x] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import json

import pytest

from src.report import evaluate_run, format_report, write_report


def _qrels(n_queries: int = 60) -> dict[str, dict[str, int]]:
    return {
        f"q{i}": {"a": 100, "b": 10, "c": 1, "d": 0} for i in range(n_queries)
    }


def _perfect_run(qrels) -> dict[str, dict[str, float]]:
    return {q: {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0} for q in qrels}


def _worst_run(qrels) -> dict[str, dict[str, float]]:
    return {q: {"d": 4.0, "c": 3.0, "b": 2.0, "a": 1.0} for q in qrels}


def test_perfect_run_scores_one_and_beats_the_floor():
    qrels = _qrels()
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="perfect", split="test", n_floor_trials=10
    )
    assert report.ndcg.point == pytest.approx(1.0)
    assert report.lift_over_floor.point > 0
    assert report.lift_over_floor.low > 0


def test_worst_run_loses_to_the_floor():
    qrels = _qrels()
    report = evaluate_run(
        _worst_run(qrels), qrels, run_tag="worst", split="test", n_floor_trials=10
    )
    assert report.lift_over_floor.point < 0
    assert report.lift_over_floor.high < 0


def test_report_records_the_query_count_and_the_floor():
    qrels = _qrels(n_queries=37)
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="t", split="test", n_floor_trials=10
    )
    assert report.n_queries == 37
    assert 0.0 < report.floor_mean < 1.0


def test_report_is_reproducible_for_a_given_seed():
    qrels = _qrels()
    run = _perfect_run(qrels)
    first = evaluate_run(run, qrels, run_tag="t", split="test", n_floor_trials=10, seed=3)
    second = evaluate_run(run, qrels, run_tag="t", split="test", n_floor_trials=10, seed=3)
    assert first.to_dict() == second.to_dict()


def test_formatted_report_shows_the_floor_next_to_the_headline():
    # PROJECT_SPEC.md: "A headline number of 0.86 means nothing without that
    # floor attached." The formatter is where that is enforced.
    qrels = _qrels()
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="t", split="test", n_floor_trials=10
    )
    text = format_report(report)
    assert "NDCG" in text
    assert "random floor" in text
    assert "95% CI" in text


def test_written_report_is_valid_json_with_the_floor_included(tmp_path):
    qrels = _qrels()
    report = evaluate_run(
        _perfect_run(qrels), qrels, run_tag="t", split="test", n_floor_trials=10
    )
    path = tmp_path / "t.json"
    write_report(report, path)
    payload = json.loads(path.read_text())
    assert payload["run_tag"] == "t"
    assert payload["floor_mean"] == pytest.approx(report.floor_mean)
    assert payload["ndcg"]["low"] <= payload["ndcg"]["point"] <= payload["ndcg"]["high"]


def test_evaluating_a_run_that_skips_a_query_raises():
    qrels = _qrels()
    run = _perfect_run(qrels)
    del run["q0"]
    with pytest.raises(KeyError, match="q0"):
        evaluate_run(run, qrels, run_tag="t", split="test", n_floor_trials=5)
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_report.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.report'`.

- [x] **Step 3: Write `src/report.py`**

```python
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
        f"95% CI [{lift.low:+.4f}, {lift.high:+.4f}]  — {verdict}\n"
    )


def write_report(report: EvaluationReport, path: Path) -> None:
    """Persist a report as JSON so the ablation table can be rebuilt from disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
```

- [x] **Step 4: Add `per_query` to `FloorResult` in `src/floor.py`**

`evaluate_run` needs the floor's per-query scores to pair against. Change the dataclass and the loop in `src/floor.py`:

```python
@dataclass(frozen=True)
class FloorResult:
    mean: float
    low: float
    high: float
    per_query: dict[str, float]
    per_trial: tuple[float, ...]
    n_trials: int
    seed: int
```

and in `random_floor`, accumulate per-query totals alongside the per-trial means:

```python
    rng = random.Random(seed)
    per_trial: list[float] = []
    totals: dict[str, float] = {qid: 0.0 for qid in qrels}
    for _ in range(n_trials):
        scores = ndcg_per_query(random_run(qrels, rng), qrels)
        for qid, score in scores.items():
            totals[qid] += score
        per_trial.append(sum(scores.values()) / len(scores))

    per_query = {qid: total / n_trials for qid, total in totals.items()}
```

and pass `per_query=per_query` when constructing the `FloorResult`.

Add the covering test to `tests/test_floor.py`:

```python
def test_per_query_floor_averages_to_the_headline_floor():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    result = random_floor(qrels, n_trials=10, seed=0)
    assert set(result.per_query) == set(qrels)
    mean_of_per_query = sum(result.per_query.values()) / len(result.per_query)
    assert mean_of_per_query == pytest.approx(result.mean)
```

- [x] **Step 5: Write `src/cli.py`**

```python
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
```

- [x] **Step 6: Run the whole fast suite to verify it passes**

```bash
python -m pytest -v
```

Expected: PASS. No failures, no errors. `tests/test_metrics_vs_pytrec.py` must not skip.

- [x] **Step 7: Record the commands in `CLAUDE.md`**

`CLAUDE.md` says "There is no build step, test suite, or linter configured yet. When adding tests, record the invocation here." Replace that Commands block with:

````markdown
## Commands

```bash
# Install (dev extra brings pytrec_eval, used as the NDCG oracle in tests)
pip install -e ".[dev]"

# Fast test suite - runs in seconds, no data needed
python -m pytest

# Tests that need the real ~1 GB parquet on disk
python -m pytest -m data

# Slow tests (the 100-trial random floor)
python -m pytest -m "data and slow" -s

# Evaluation
python -m src.cli floor --split test
python -m src.cli evaluate --run runs/<name>.trec --split test --tag <name>
python -m src.cli freeze-splits

# Image URL resolution gate - samples live URLs, exits non-zero below 90%
python -m src.esci_images <esci.json.zst>   # a truncated prefix of the file is fine
```
````

- [x] **Step 8: Commit**

```bash
git add src/report.py src/cli.py src/floor.py tests/test_report.py tests/test_floor.py CLAUDE.md
git commit -m "Add evaluation report, CLI and per-query random floor"
```
---

## Task 10: Reproduce the published zero-shot SBERT baseline

**Files:**
- Create: `src/baseline_sbert.py`
- Create: `tests/test_baseline_sbert.py`
- Create: `docs/results/sbert-title-test.json` (generated in Step 5, committed)

**Interfaces:**
- Consumes: `src.dataset.load_split`, `src.runs.run_from_scores`, `src.runs.write_run`.
- Produces:
  - `src.baseline_sbert.DEFAULT_MODEL: str` (`"sentence-transformers/all-MiniLM-L12-v2"`)
  - `src.baseline_sbert.product_text(products: pd.DataFrame, fields: Sequence[str]) -> pd.Series`
  - `src.baseline_sbert.score_pairs(judgements, products, encode, fields) -> pd.DataFrame` with columns `query_id`, `product_id`, `score`
  - `python -m src.baseline_sbert --split test --out runs/sbert-title.trec`

`score_pairs` takes `encode: Callable[[list[str]], np.ndarray]` rather than a model, so the scoring logic is unit-testable in milliseconds with a fake encoder and the 10-minute model run is confined to `__main__`.

This is the task that proves the harness. The spec's §5 table gives SBERT_text zero-shot at **0.8292** on the same split with the same metric. If this pipeline produces that number, the loader, the metric and the floor are all trustworthy. If it produces 0.75 or 0.91, one of them is wrong, and every later plan in the series is built on sand.

- [x] **Step 1: Write the failing test**

Create `tests/test_baseline_sbert.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline_sbert import product_text, score_pairs


def _products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["B1", "B2"],
            "product_title": ["red running shoe", "blue kettle"],
            "product_brand": ["Acme", None],
            "product_color": [None, "blue"],
            "product_description": ["a shoe", None],
            "product_bullet_point": [None, None],
        }
    )


def test_product_text_uses_the_title_by_default():
    assert list(product_text(_products(), ["product_title"])) == [
        "red running shoe",
        "blue kettle",
    ]


def test_product_text_joins_requested_fields_and_skips_nulls():
    text = product_text(_products(), ["product_title", "product_brand", "product_color"])
    assert list(text) == ["red running shoe Acme", "blue kettle blue"]


def test_product_text_never_yields_the_string_none():
    # A naive str() over a null column produces the literal "None" or "nan",
    # which the encoder then embeds as if it were product text.
    text = product_text(_products(), ["product_title", "product_brand"])
    assert not any("None" in t or "nan" in t for t in text)


def test_product_text_rejects_an_unknown_field():
    with pytest.raises(KeyError, match="product_weight"):
        product_text(_products(), ["product_weight"])


def _fake_encode(texts: list[str]) -> np.ndarray:
    """One-hot on the first character; cosine similarity is then 1 or 0."""
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    out = np.zeros((len(texts), len(alphabet)), dtype=np.float32)
    for row, text in enumerate(texts):
        out[row, alphabet.index(text[:1].lower() or " ")] = 1.0
    return out


def test_score_pairs_gives_a_score_to_every_judgement():
    judgements = pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "query": ["red shoe", "red shoe", "blue kettle"],
            "product_id": ["B1", "B2", "B1"],
        }
    )
    scores = score_pairs(judgements, _products(), _fake_encode, ["product_title"])
    assert list(scores.columns) == ["query_id", "product_id", "score"]
    assert len(scores) == 3


def test_score_pairs_rewards_the_matching_product():
    judgements = pd.DataFrame(
        {
            "query_id": [1, 1],
            "query": ["red shoe", "red shoe"],
            "product_id": ["B1", "B2"],
        }
    )
    scores = score_pairs(judgements, _products(), _fake_encode, ["product_title"])
    by_product = dict(zip(scores["product_id"], scores["score"]))
    assert by_product["B1"] > by_product["B2"]


def test_score_pairs_encodes_each_distinct_query_once():
    # 181,701 judgements cover only 8,956 queries. Encoding per judgement
    # would be 20x the work for the same answer.
    seen: list[list[str]] = []

    def counting_encode(texts: list[str]) -> np.ndarray:
        seen.append(texts)
        return _fake_encode(texts)

    judgements = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "query": ["red shoe"] * 3,
            "product_id": ["B1", "B2", "B1"],
        }
    )
    score_pairs(judgements, _products(), counting_encode, ["product_title"])
    queries_encoded, products_encoded = seen
    assert len(queries_encoded) == 1
    assert len(products_encoded) == 2


def test_score_pairs_raises_when_a_judged_product_has_no_metadata():
    judgements = pd.DataFrame(
        {"query_id": [1], "query": ["red shoe"], "product_id": ["B-MISSING"]}
    )
    with pytest.raises(ValueError, match="B-MISSING"):
        score_pairs(judgements, _products(), _fake_encode, ["product_title"])


@pytest.mark.data
@pytest.mark.slow
def test_published_sbert_number_is_reproduced():
    # PROJECT_SPEC.md §5: SBERT_text zero-shot = 0.8292 (all-MiniLM-L12-v2).
    # The cited paper does not state which product fields it encoded, so the
    # band allows a title-only vs. title+brand difference while still failing
    # loudly if the harness itself is wrong.
    payload = json.loads(Path("docs/results/sbert-title-test.json").read_text())
    assert 0.820 <= payload["ndcg"]["point"] <= 0.840
    assert payload["lift_over_floor"]["low"] > 0
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_baseline_sbert.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.baseline_sbert'`.

- [x] **Step 3: Write `src/baseline_sbert.py`**

```python
"""Zero-shot SBERT re-ranking: the baseline that proves the harness.

PROJECT_SPEC.md §5 puts SBERT_text zero-shot at 0.8292 on this split with
all-MiniLM-L12-v2. Reproducing it is the acceptance test for the loader, the
metric and the floor together, which is why it is in Plan 1 rather than in the
retrieval plan.

Scoring takes an `encode` callable rather than a model so the logic is unit
tested in milliseconds; the 10-minute CPU run lives in __main__.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import load_split
from src.runs import run_from_scores, write_run

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
Encoder = Callable[[list[str]], np.ndarray]


def product_text(products: pd.DataFrame, fields: Sequence[str]) -> pd.Series:
    """Join the requested product fields into one string per product.

    Null fields are dropped rather than stringified: str(None) would embed the
    literal "None" as if it were product text, and ~half of product_color is
    null.
    """
    missing = [f for f in fields if f not in products.columns]
    if missing:
        raise KeyError(f"no such product field(s): {missing}")
    parts = [products[f].fillna("").astype(str).str.strip() for f in fields]
    joined = parts[0]
    for part in parts[1:]:
        joined = (joined + " " + part).str.strip()
    return joined.str.replace(r"\s+", " ", regex=True)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def score_pairs(
    judgements: pd.DataFrame,
    products: pd.DataFrame,
    encode: Encoder,
    fields: Sequence[str],
) -> pd.DataFrame:
    """Cosine similarity for every judged (query, product) pair.

    Encodes each distinct query once and each distinct product once: there are
    ~20 judgements per query, so encoding per judgement would be 20x the work.
    """
    unknown = set(judgements["product_id"]) - set(products["product_id"])
    if unknown:
        raise ValueError(
            f"{len(unknown)} judged products have no metadata row "
            f"(for example {sorted(unknown)[:3]}); scoring them as zero would "
            "silently depress the result"
        )

    queries = judgements[["query_id", "query"]].drop_duplicates("query_id")
    query_vectors = _normalise(encode(list(queries["query"])))
    query_row = {qid: i for i, qid in enumerate(queries["query_id"])}

    catalogue = products.drop_duplicates("product_id").reset_index(drop=True)
    product_vectors = _normalise(encode(list(product_text(catalogue, fields))))
    product_row = {pid: i for i, pid in enumerate(catalogue["product_id"])}

    left = query_vectors[[query_row[q] for q in judgements["query_id"]]]
    right = product_vectors[[product_row[p] for p in judgements["product_id"]]]
    return pd.DataFrame(
        {
            "query_id": judgements["query_id"].to_numpy(),
            "product_id": judgements["product_id"].to_numpy(),
            "score": np.einsum("ij,ij->i", left, right),
        }
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fields", default="product_title")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="runs/sbert-title.trec")
    parser.add_argument("--tag", default="sbert-title")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    loaded = load_split(args.split)
    scores = score_pairs(
        loaded.judgements, loaded.products, encode, args.fields.split(",")
    )
    write_run(run_from_scores(scores), Path(args.out), run_tag=args.tag)
    print(f"wrote {len(scores):,} scores to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [x] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_baseline_sbert.py -v
```

Expected: PASS, 8 tests; the reproduction test is deselected.

- [x] **Step 5: Run the real baseline and evaluate it**

```bash
python -m pip install -e ".[baselines]"
python -m src.baseline_sbert --split test --out runs/sbert-title.trec --tag sbert-title
python -m src.cli evaluate --run runs/sbert-title.trec --split test --tag sbert-title
```

Expected: encoding 8,956 queries and 164,900 product titles takes roughly 10 minutes on CPU. The printed NDCG should land near 0.8292, the floor near 0.7467, and the lift interval should sit well above zero.

If the NDCG is outside `[0.820, 0.840]`, **do not adjust the band.** Work through, in order:

1. Is `product_locale == "us"` filtering applied to the products table as well as the examples table? Mixing locales puts Spanish titles under English queries.
2. Are the embeddings normalised before the dot product? Unnormalised dot products rank by title length as much as by similarity.
3. Is the run scoring only judged candidates? This is a re-ranking task; retrieving over the corpus here would produce a different, much lower number.
4. Try `--fields product_title,product_brand`. The cited paper does not state its field recipe, and brand is a plausible difference of a few tenths of a point.

- [x] **Step 6: Run the marked reproduction test**

```bash
python -m pytest tests/test_baseline_sbert.py -v -m "data and slow"
```

Expected: PASS. This reads the committed `docs/results/sbert-title-test.json`, so it is fast once the run above has been done.

- [x] **Step 7: Commit**

```bash
git add src/baseline_sbert.py tests/test_baseline_sbert.py docs/results/sbert-title-test.json
git commit -m "Reproduce the zero-shot SBERT baseline on the test split"
```
---

## Phase 3 Gate

This is the Plan Gate — Plan 2 of the series does not start until all of these
hold. The canonical copy lives in [`README.md`](README.md#plan-gate).

- [x] `python -m pytest` passes with no failures and no skips in `tests/test_metrics_vs_pytrec.py`.
- [x] `python -m pytest -m data` passes for both splits.
- [x] The random floor, measured not quoted, is recorded in `docs/results/` and lands near 0.7467.
- [x] `docs/results/sbert-title-test.json` shows NDCG in `[0.820, 0.840]` with a lift interval strictly above zero.
- [x] `splits/val_folds.csv` and its `.sha256` are committed, covering all 20,888 train queries.
- [x] `CLAUDE.md`'s Commands section lists the test and evaluation invocations.

Then: Plan 2 — Enrichment Corpus. See [`../README.md`](../README.md) for the series.
