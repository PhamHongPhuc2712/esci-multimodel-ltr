# Evaluation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trustworthy NDCG evaluation harness for Amazon ESCI Task 1 — asserted data invariants, a computed random floor, bootstrap confidence intervals, a frozen validation split — and prove it by reproducing a published zero-shot baseline.

**Architecture:** Pure functions over TREC-shaped nested dicts (`qid -> doc_id -> score`), which is `pytrec_eval`'s own interface, so the reference implementation can be used as a test oracle without shaping data twice. The dataset loader asserts documented row counts and the label distribution on every load and fails loudly rather than proceeding, because the most likely failure here is silently loading the wrong source file. Everything numeric is seeded and reproducible.

**Tech Stack:** Python ≥3.11, pandas + pyarrow (Parquet), numpy, `pytrec-eval-terrier` (test oracle only), `sentence-transformers` + torch (baseline only, optional extra), pytest.

**Spec:** `PROJECT_SPEC.md` (§2 The Task, §3.1 Amazon ESCI, §5 Baselines, §8.1 Build Order) and `CLAUDE.md` (Data invariants, Evaluation discipline). The plan argues from both; executors read both.

**Series:** Plan 1 of 7 — see `docs/superpowers/plans/README.md`.

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Use the official parquet, never the `tasksource/esci` HF mirror.** The mirror has no `split` column and encodes the *large-version* split, yielding 185,361 test judgements instead of 181,701 (~2% contamination). Fetch via `https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/...` (the repo is Git LFS; the plain `raw.githubusercontent.com` URL returns an LFS pointer, not a Parquet file).
- **Task 1 English filter is `small_version == 1` and `product_locale == "us"`**, then `split`.
- **Asserted row counts:** test = 181,701 judgements / 8,956 queries / 164,900 products; train = 419,653 judgements / 20,888 queries.
- **Asserted test label distribution:** E 43.87%, S 34.98%, I 16.69%, C 4.46%. Any source reporting Complement ≈ 35% has the known S/C swap bug in the official `prepare_trec_eval_files.py`.
- **Gains are E=1.0, S=0.1, C=0.01, I=0.0.** Integer qrels are the same values ×100 (E=100, S=10, C=1, I=0) so they can be handed to `pytrec_eval`, which requires integers. NDCG is invariant to this scaling; Task 2 tests that it is.
- **NDCG is full-list, no cutoff**, with gain used linearly (not `2^rel - 1`) and discount `1/log2(rank+1)` with rank starting at 1. This matches `trec_eval -m ndcg`, the official ESCI evaluation route.
- **Compute the random floor, never quote it.** `CLAUDE.md` records 0.7467 measured with this discount; the published SQID figure is 0.7483; under the swapped S/C mapping it is 0.7141. That 3.3-point spread is larger than most method gains, so the number must come from the code, not from a constant.
- **There is no official validation split.** Carve one from train by `query_id` and freeze it before tuning anything. Never split within a query group.
- **Never tune on test.**
- **Features must be query×product, never product-alone-from-labels.** 34,756 products appear in both train and test; the official split is query-level, so this is legitimate, but product-level target encoding leaks train labels into test.
- **Report bootstrap confidence intervals over queries, not bare point estimates.** A method that ties the baseline is a legitimate, reportable result.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)
- **Do not commit datasets.** `.gitignore` already blocks `data/`, `runs/`, `*.parquet`, `*.npy`. Frozen split artifacts therefore live at `splits/` in the repo root, deliberately outside `data/`, so they can be committed without fighting the ignore rules.

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **A run that omits judged products for a query.** A ranker that returns only its top 10 of 20 judged candidates must be scored against the ideal over *all* judged candidates. Dividing by the ideal of only what was returned silently inflates every score. — pinned in Task 2.
2. **A run containing products with no judgement for that query.** The spec's funnel retrieves from a ~1.2M corpus, so unjudged products will appear. They must count as gain 0 and still consume a rank slot, per `trec_eval` convention — not raise, and not be dropped as if they were never returned. — pinned in Task 2.
3. **A query where every judgement is Irrelevant.** IDCG is 0, and the naive ratio is a division by zero that produces `nan`, which then silently poisons the mean over 8,956 queries into `nan` or, worse, gets dropped and shifts the denominator. `trec_eval` scores such a query 0.0. — pinned in Task 2.
4. **A scorer that returns identical scores for every candidate.** Tie order must be deterministic, or the same run file scores differently on two machines. A fully-tied run must land near the random floor, never at 1.0. — pinned in Task 2.
5. **Loading the wrong source file.** The HF mirror parses cleanly, has the right column names, and yields 185,361 test rows. Nothing about it fails except the count, so the count must be checked on every load, and the error must name the mirror trap rather than just printing two numbers. — pinned in Task 4.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as `src.<name>` and run as `python -m src.<name>` (`src/esci_images.py`, documented in `CLAUDE.md`). This plan follows that pattern rather than introducing a nested package, so the documented command keeps working.

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, extras, pytest configuration and markers |
| `src/labels.py` | ESCI label → gain and label → integer qrel mappings; nothing else |
| `src/metrics.py` | `ndcg_per_query`, `mean_ndcg`. Pure, no I/O, no pandas |
| `src/dataset.py` | Download, load, filter and *assert invariants on* the official ESCI parquet |
| `src/runs.py` | TREC run-file read/write; DataFrame → qrels/run dict conversion |
| `src/bootstrap.py` | `bootstrap_ci`, `paired_delta_ci` over per-query scores |
| `src/floor.py` | Random-ordering floor, computed from qrels |
| `src/splits.py` | Deterministic query-level fold assignment; freeze and load |
| `src/report.py` | Assemble and format an evaluation report; JSON serialisation |
| `src/cli.py` | `python -m src.cli {evaluate,floor,freeze-splits}` |
| `src/baseline_sbert.py` | Zero-shot SBERT run generation (optional extra) |
| `splits/val_folds.csv` | The frozen validation folds, committed |
| `docs/results/*.json` | Committed evaluation records |
| `tests/*.py` | One test module per source module |

`src/metrics.py` holds no pandas and no I/O on purpose: it is the one file whose correctness everything downstream rests on, and it must be testable against `pytrec_eval` without any data loading in the way.

---

## Task 1: Project scaffolding and label constants

**Files:**
- Create: `pyproject.toml`
- Create: `src/labels.py`
- Create: `tests/test_labels.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `src.labels.ESCI_GAINS: dict[str, float]`, `src.labels.ESCI_QRELS: dict[str, int]`, `src.labels.QREL_SCALE: int`, `src.labels.label_to_gain(label: str) -> float`, `src.labels.label_to_qrel(label: str) -> int`. Both functions raise `ValueError` on an unknown label.

- [ ] **Step 1: Write the failing test**

Create `tests/test_labels.py`:

```python
import pytest

from src.labels import (
    ESCI_GAINS,
    ESCI_QRELS,
    QREL_SCALE,
    label_to_gain,
    label_to_qrel,
)


def test_gains_match_the_spec():
    assert ESCI_GAINS == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def test_substitute_outranks_complement():
    # The official prepare_trec_eval_files.py swaps these two. Under the swap
    # the random floor moves 0.7467 -> 0.7141, which is larger than most
    # method gains, so this ordering is pinned explicitly.
    assert ESCI_GAINS["S"] > ESCI_GAINS["C"]
    assert ESCI_QRELS["S"] > ESCI_QRELS["C"]


def test_integer_qrels_are_the_gains_scaled():
    assert QREL_SCALE == 100
    for label, gain in ESCI_GAINS.items():
        assert ESCI_QRELS[label] == round(gain * QREL_SCALE)


def test_integer_qrels_are_actually_integers():
    # pytrec_eval rejects float relevance values.
    assert all(isinstance(v, int) for v in ESCI_QRELS.values())


def test_lookup_helpers_agree_with_the_tables():
    assert label_to_gain("E") == 1.0
    assert label_to_qrel("E") == 100


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unknown ESCI label"):
        label_to_gain("X")
    with pytest.raises(ValueError, match="unknown ESCI label"):
        label_to_qrel("X")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_labels.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.labels'`.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "esci-multimodel-ltr"
version = "0.1.0"
description = "Multi-stage multimodal product search ranking on Amazon ESCI"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=16",
    "zstandard>=0.22",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytrec-eval-terrier>=0.5.6",
]
baselines = [
    "sentence-transformers>=3.0",
    "torch>=2.4",
]

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not data and not slow'"
markers = [
    "data: needs the real ESCI parquet on disk; opt in with -m data",
    "slow: takes more than a minute; opt in with -m slow",
]
```

The `addopts` line matters: the real parquet is ~1 GB and the SBERT baseline takes ~10 minutes on CPU. `pytest` with no arguments must stay fast enough to run after every edit, so the tests that need either are marked and deselected by default.

- [ ] **Step 4: Write `src/labels.py`**

```python
"""ESCI relevance labels and their gain values.

The gains come from PROJECT_SPEC.md §2. The integer table exists because
pytrec_eval requires integer relevance values; NDCG is invariant to the
scale factor, which tests/test_metrics.py pins.

The official amazon-science/esci-data `prepare_trec_eval_files.py` swaps the
S and C values. Published numbers differ depending on which mapping was used,
so this module is the single place the mapping is defined.
"""

from __future__ import annotations

ESCI_GAINS: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

QREL_SCALE = 100

ESCI_QRELS: dict[str, int] = {
    label: round(gain * QREL_SCALE) for label, gain in ESCI_GAINS.items()
}


def label_to_gain(label: str) -> float:
    """Gain for an ESCI label, as a float on the spec's 1.0/0.1/0.01/0.0 scale."""
    try:
        return ESCI_GAINS[label]
    except KeyError:
        raise ValueError(f"unknown ESCI label {label!r}; expected one of E, S, C, I") from None


def label_to_qrel(label: str) -> int:
    """Gain for an ESCI label as an integer, for pytrec_eval / trec_eval qrels."""
    try:
        return ESCI_QRELS[label]
    except KeyError:
        raise ValueError(f"unknown ESCI label {label!r}; expected one of E, S, C, I") from None
```

- [ ] **Step 5: Add the results directory to git and adjust `.gitignore`**

`.gitignore` already blocks `data/`, `runs/`, `*.parquet` and `*.npy`, which is correct — raw data and run files are outputs. Append a short note so the next person does not "fix" it by un-ignoring `data/`:

```bash
cat >> .gitignore <<'EOF'

# Frozen split artifacts live in splits/ (not data/) precisely so they escape
# the data/ rule above and can be committed. Keep them there.
EOF
```

- [ ] **Step 6: Install and run the tests to verify they pass**

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_labels.py -v
```

Expected: PASS, 6 tests.

If `pytrec-eval-terrier` fails to build, it needs a C toolchain: `sudo apt-get install -y build-essential python3-dev` then retry. Do not skip it — Task 3 uses it as the correctness oracle and there is no substitute.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/labels.py tests/test_labels.py .gitignore
git commit -m "Add project scaffolding and ESCI label gain tables"
```

---

## Task 2: Full-list NDCG

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing (pure functions; no imports from `src.labels`).
- Produces:
  - `src.metrics.Qrels = Mapping[str, Mapping[str, float]]` — `qid -> doc_id -> gain`
  - `src.metrics.Run = Mapping[str, Mapping[str, float]]` — `qid -> doc_id -> score`
  - `src.metrics.ndcg_per_query(run: Run, qrels: Qrels) -> dict[str, float]` — one entry per **qrels** query
  - `src.metrics.mean_ndcg(run: Run, qrels: Qrels) -> float`

Iterating over `qrels` rather than `run` is the design decision that makes Review Focus items 1 and 3 impossible to get wrong by accident: a query the run never scored is a hard error, and a query the run scored partially is still divided by the full ideal.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
import math

import pytest

from src.metrics import mean_ndcg, ndcg_per_query


def test_perfect_ranking_scores_one():
    qrels = {"q": {"a": 100, "b": 10, "c": 0}}
    run = {"q": {"a": 0.9, "b": 0.8, "c": 0.7}}
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(1.0)


def test_reversed_ranking_scores_the_hand_computed_value():
    qrels = {"q": {"a": 100, "b": 10, "c": 0}}
    run = {"q": {"c": 0.9, "b": 0.8, "a": 0.7}}
    dcg = 0 / math.log2(2) + 10 / math.log2(3) + 100 / math.log2(4)
    idcg = 100 / math.log2(2) + 10 / math.log2(3) + 0 / math.log2(4)
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(dcg / idcg)


def test_gain_is_linear_not_exponential():
    # Gains 1 and 0 map to 1 and 0 under 2**rel - 1 as well, so they cannot
    # tell the two forms apart. Gains 2 and 1 become 3 and 1, which can.
    # trec_eval -m ndcg uses the linear form.
    qrels = {"q": {"a": 2, "b": 1}}
    run = {"q": {"b": 1.0, "a": 0.5}}
    linear = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    exponential = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert linear != pytest.approx(exponential)
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(linear)


def test_ndcg_is_invariant_to_gain_scale():
    # Justifies using integer qrels (100/10/1/0) for pytrec_eval while the
    # spec states float gains (1.0/0.1/0.01/0.0).
    run = {"q": {"a": 3.0, "b": 2.0, "c": 1.0}}
    as_ints = {"q": {"a": 10, "b": 100, "c": 1}}
    as_floats = {"q": {"a": 0.1, "b": 1.0, "c": 0.01}}
    assert ndcg_per_query(run, as_ints)["q"] == pytest.approx(
        ndcg_per_query(run, as_floats)["q"]
    )


# --- Review Focus 1: partial runs -------------------------------------------

def test_judged_documents_the_run_omitted_still_count_against_it():
    qrels = {"q": {"a": 100, "b": 100}}
    full = {"q": {"a": 2.0, "b": 1.0}}
    partial = {"q": {"a": 2.0}}  # b was never retrieved
    assert ndcg_per_query(full, qrels)["q"] == pytest.approx(1.0)
    idcg = 100 / math.log2(2) + 100 / math.log2(3)
    assert ndcg_per_query(partial, qrels)["q"] == pytest.approx(100 / idcg)


def test_query_absent_from_the_run_raises():
    qrels = {"q1": {"a": 100}, "q2": {"b": 100}}
    run = {"q1": {"a": 1.0}}
    with pytest.raises(KeyError, match="q2"):
        ndcg_per_query(run, qrels)


# --- Review Focus 2: unjudged documents -------------------------------------

def test_unjudged_documents_count_as_zero_gain_and_consume_a_rank():
    qrels = {"q": {"a": 100}}
    only_judged = {"q": {"a": 2.0}}
    with_intruder = {"q": {"x": 3.0, "a": 2.0}}  # x has no judgement
    assert ndcg_per_query(only_judged, qrels)["q"] == pytest.approx(1.0)
    assert ndcg_per_query(with_intruder, qrels)["q"] == pytest.approx(1 / math.log2(3))


# --- Review Focus 3: no positive gain ---------------------------------------

def test_query_with_no_relevant_document_scores_zero_not_nan():
    qrels = {"q": {"a": 0, "b": 0}}
    run = {"q": {"a": 2.0, "b": 1.0}}
    score = ndcg_per_query(run, qrels)["q"]
    assert score == 0.0
    assert not math.isnan(score)


def test_an_all_irrelevant_query_does_not_poison_the_mean():
    qrels = {"good": {"a": 100}, "hopeless": {"b": 0}}
    run = {"good": {"a": 1.0}, "hopeless": {"b": 1.0}}
    assert mean_ndcg(run, qrels) == pytest.approx(0.5)


# --- Review Focus 4: ties ---------------------------------------------------

def test_ties_break_by_ascending_document_id():
    qrels = {"q": {"a": 0, "b": 100}}
    run = {"q": {"a": 1.0, "b": 1.0}}
    # "a" sorts first, so the relevant document lands at rank 2.
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(1 / math.log2(3))


def test_tie_order_does_not_depend_on_dict_insertion_order():
    qrels = {"q": {"a": 0, "b": 100}}
    one = {"q": {"a": 1.0, "b": 1.0}}
    other = {"q": {"b": 1.0, "a": 1.0}}
    assert ndcg_per_query(one, qrels)["q"] == ndcg_per_query(other, qrels)["q"]


def test_a_fully_tied_run_gets_no_credit_for_a_lucky_tie_break():
    # The relevant document is "c", so ascending-id tie-breaking puts it last
    # and the run scores 0.5. Naming the relevant document "a" instead would
    # make this pass at 1.0 by luck rather than by the ranker being right.
    qrels = {"q": {"a": 0, "b": 0, "c": 100}}
    run = {"q": {"a": 1.0, "b": 1.0, "c": 1.0}}
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(1 / math.log2(4))


# --- aggregation ------------------------------------------------------------

def test_mean_is_over_queries_not_over_documents():
    qrels = {"q1": {"a": 100}, "q2": {"b": 100, "c": 100, "d": 100}}
    run = {"q1": {"a": 1.0}, "q2": {"b": 3.0, "c": 2.0, "d": 1.0}}
    # Both queries are perfectly ranked; q2 having 3 documents must not
    # weight it more heavily than q1.
    assert mean_ndcg(run, qrels) == pytest.approx(1.0)


def test_mean_of_empty_qrels_raises():
    with pytest.raises(ValueError, match="no queries"):
        mean_ndcg({}, {})
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`.

- [ ] **Step 3: Write `src/metrics.py`**

```python
"""Full-list NDCG for ESCI Task 1.

Matches `trec_eval -m ndcg`: gain enters linearly (not 2**rel - 1), the
discount is 1/log2(rank + 1) with rank starting at 1, and there is no cutoff.
The whole judged candidate list is ranked, which is why the random floor for
this task sits near 0.75 rather than near 0.

Data is shaped as nested dicts, qid -> doc_id -> value, which is also
pytrec_eval's interface; tests/test_metrics_vs_pytrec.py uses it as an oracle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

Qrels = Mapping[str, Mapping[str, float]]
Run = Mapping[str, Mapping[str, float]]


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain of an already-ordered list of gains."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_per_query(run: Run, qrels: Qrels) -> dict[str, float]:
    """NDCG for every query in `qrels`.

    Iterates `qrels`, not `run`, so that:
      - documents the run failed to return still count towards the ideal,
      - a query the run skipped entirely is an error rather than a silent
        shrinking of the denominator.

    Documents in the run with no judgement count as gain 0 and still occupy a
    rank, matching trec_eval. Ties are broken by ascending document id so the
    same run file scores identically on any machine. A query whose ideal DCG
    is 0 (no positive gain anywhere) scores 0.0, also matching trec_eval.
    """
    scores: dict[str, float] = {}
    for qid, judged in qrels.items():
        if qid not in run:
            raise KeyError(
                f"query {qid!r} is in qrels but not in the run; "
                "a run must score every judged query"
            )
        ranked = sorted(run[qid].items(), key=lambda kv: (-kv[1], kv[0]))
        dcg = _dcg([float(judged.get(doc_id, 0.0)) for doc_id, _ in ranked])
        idcg = _dcg(sorted((float(g) for g in judged.values()), reverse=True))
        scores[qid] = 0.0 if idcg == 0.0 else dcg / idcg
    return scores


def mean_ndcg(run: Run, qrels: Qrels) -> float:
    """NDCG averaged over queries, weighting every query equally."""
    per_query = ndcg_per_query(run, qrels)
    if not per_query:
        raise ValueError("no queries to score: qrels is empty")
    return sum(per_query.values()) / len(per_query)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: PASS, 14 tests.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "Add full-list NDCG with trec_eval-compatible tie and zero-ideal handling"
```

---

## Task 3: Cross-check NDCG against pytrec_eval

**Files:**
- Create: `tests/test_metrics_vs_pytrec.py`

**Interfaces:**
- Consumes: `src.metrics.ndcg_per_query(run, qrels) -> dict[str, float]`.
- Produces: no source module. This task's deliverable is the oracle test that licenses every number the rest of the project reports.

This is a separate task from Task 2 because a reviewer could reasonably accept the implementation and reject the oracle, or the reverse. It uses randomly generated qrels and runs rather than hand-written cases, so it covers shapes nobody thought to write a case for.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics_vs_pytrec.py`:

```python
import random

import pytest

from src.labels import ESCI_QRELS
from src.metrics import ndcg_per_query

pytrec_eval = pytest.importorskip(
    "pytrec_eval",
    reason="install the dev extra: pip install -e '.[dev]'",
)

LABELS = list(ESCI_QRELS)


def _random_case(seed: int, n_queries: int = 40, max_docs: int = 25):
    """Build an ESCI-shaped qrels/run pair with distinct scores.

    Scores are a shuffled range so no two documents tie: trec_eval breaks ties
    by *descending* document id while this project breaks them by ascending id,
    and that difference is deliberate and tested separately. Feeding ties to
    the oracle would compare tie policies rather than NDCG.
    """
    rng = random.Random(seed)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q in range(n_queries):
        qid = f"q{q}"
        n_docs = rng.randint(1, max_docs)
        doc_ids = [f"B{q:03d}{d:04d}" for d in range(n_docs)]
        qrels[qid] = {d: ESCI_QRELS[rng.choice(LABELS)] for d in doc_ids}
        scores = list(range(n_docs))
        rng.shuffle(scores)
        run[qid] = {d: float(s) for d, s in zip(doc_ids, scores)}
    return qrels, run


@pytest.mark.parametrize("seed", range(10))
def test_matches_pytrec_eval_on_random_esci_shaped_data(seed):
    qrels, run = _random_case(seed)
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    ours = ndcg_per_query(run, qrels)
    assert set(ours) == set(oracle)
    for qid, expected in oracle.items():
        assert ours[qid] == pytest.approx(expected["ndcg"], abs=1e-9), qid


def test_matches_pytrec_eval_when_the_run_has_unjudged_documents():
    qrels, run = _random_case(seed=99)
    for qid, docs in run.items():
        docs[f"UNJUDGED-{qid}"] = 1e6  # ranked first, no judgement
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    ours = ndcg_per_query(run, qrels)
    for qid, expected in oracle.items():
        assert ours[qid] == pytest.approx(expected["ndcg"], abs=1e-9), qid


def test_matches_pytrec_eval_when_the_run_omits_judged_documents():
    qrels, run = _random_case(seed=7)
    for qid, docs in run.items():
        if len(docs) > 1:
            docs.pop(next(iter(docs)))  # drop one retrieved document
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    ours = ndcg_per_query(run, qrels)
    for qid, expected in oracle.items():
        assert ours[qid] == pytest.approx(expected["ndcg"], abs=1e-9), qid


def test_matches_pytrec_eval_when_a_query_has_no_relevant_document():
    qrels = {"q": {"a": 0, "b": 0}}
    run = {"q": {"a": 2.0, "b": 1.0}}
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(oracle["q"]["ndcg"], abs=1e-9)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_metrics_vs_pytrec.py -v
```

Expected: FAIL. If `src/metrics.py` from Task 2 is correct these pass immediately; if they do not, the disagreement is the finding — read the failing qid's qrels and run before changing anything, because `pytrec_eval` is the reference and `src/metrics.py` is what moves.

If instead the whole module skips with "install the dev extra", `pytrec-eval-terrier` is not installed. Install it (`pip install -e ".[dev]"`, and `sudo apt-get install -y build-essential python3-dev` first if the build fails). A skip here is a failed task, not a pass.

- [ ] **Step 3: Reconcile any disagreement in `src/metrics.py`**

If a case fails, the likely causes, in order of likelihood:

1. Discount off by one — `1/log2(rank+1)` with `rank` starting at **1**, so rank 1 is divided by `log2(2) == 1`. Starting `enumerate` at 0 silently divides rank 1 by `log2(1) == 0`.
2. Exponential gain — `2**rel - 1` instead of linear `rel`.
3. Ideal computed over the run's documents rather than over all judged documents.

Make the minimal change to `src/metrics.py` and re-run. Do not weaken the tolerance.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_metrics.py tests/test_metrics_vs_pytrec.py -v
```

Expected: PASS, 14 + 13 tests, no skips.

- [ ] **Step 5: Commit**

```bash
git add tests/test_metrics_vs_pytrec.py
git commit -m "Cross-check NDCG against pytrec_eval on randomised ESCI-shaped data"
```

---

## Task 4: ESCI dataset loader with asserted invariants

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `src.labels.label_to_gain(label) -> float`, `src.labels.label_to_qrel(label) -> int`, `src.labels.ESCI_GAINS`.
- Produces:
  - `src.dataset.DataInvariantError` (subclass of `ValueError`)
  - `src.dataset.SplitStats(judgements: int, queries: int, products: int | None)`
  - `src.dataset.EXPECTED_STATS: dict[str, SplitStats]`
  - `src.dataset.EXPECTED_LABEL_SHARE: dict[str, dict[str, float]]`
  - `src.dataset.EsciSplit(name: str, judgements: pd.DataFrame, products: pd.DataFrame)`
  - `src.dataset.data_dir() -> Path`
  - `src.dataset.ensure_downloaded(name: str, data_dir_override: Path | None = None) -> Path` for `name` in `{"examples", "products"}`
  - `src.dataset.filter_task1(examples: pd.DataFrame, split: str) -> pd.DataFrame`
  - `src.dataset.check_invariants(split: str, judgements: pd.DataFrame, products: pd.DataFrame | None = None) -> None`
  - `src.dataset.load_split(split: str, data_dir_override: Path | None = None) -> EsciSplit`

`EsciSplit.judgements` columns: `example_id` (int64), `query_id` (int64), `query` (str), `product_id` (str), `esci_label` (str), `gain` (float64), `qrel` (int64).
`EsciSplit.products` columns: `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color` — restricted to the `product_id` values appearing in `judgements`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset.py`:

```python
import pandas as pd
import pytest

from src.dataset import (
    EXPECTED_LABEL_SHARE,
    EXPECTED_STATS,
    DataInvariantError,
    check_invariants,
    filter_task1,
    load_split,
)


def _judgements(labels: list[str], n_queries: int) -> pd.DataFrame:
    """A judgements frame with a given label multiset spread over n_queries."""
    n = len(labels)
    label_column = pd.Series(labels, dtype="object")
    return pd.DataFrame(
        {
            "example_id": range(n),
            "query_id": [i % n_queries for i in range(n)],
            "query": "a query",
            "product_id": [f"B{i:09d}" for i in range(n)],
            "esci_label": label_column,
            "gain": label_column.map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}),
            "qrel": label_column.map({"E": 100, "S": 10, "C": 1, "I": 0}),
        }
    )


def _test_shaped_judgements() -> pd.DataFrame:
    """A frame with exactly the documented test-split counts and label shares."""
    stats = EXPECTED_STATS["test"]
    share = EXPECTED_LABEL_SHARE["test"]
    counts = {label: round(stats.judgements * s) for label, s in share.items()}
    counts["E"] += stats.judgements - sum(counts.values())  # absorb rounding
    labels = [label for label, n in counts.items() for _ in range(n)]
    return _judgements(labels, stats.queries)


# --- filtering --------------------------------------------------------------

def test_filter_task1_keeps_only_small_version_us_rows_of_the_split():
    examples = pd.DataFrame(
        {
            "example_id": [0, 1, 2, 3],
            "query_id": [0, 0, 1, 1],
            "query": ["a", "a", "b", "b"],
            "product_id": ["B0", "B1", "B2", "B3"],
            "product_locale": ["us", "es", "us", "us"],
            "esci_label": ["E", "E", "E", "E"],
            "small_version": [1, 1, 0, 1],
            "large_version": [1, 1, 1, 1],
            "split": ["test", "test", "test", "train"],
        }
    )
    kept = filter_task1(examples, "test")
    assert list(kept["example_id"]) == [0]


def test_filter_task1_rejects_a_frame_with_no_split_column():
    # The tasksource/esci HF mirror has no `split` column. Loading it yields
    # 185,361 test judgements instead of 181,701 (~2% contamination).
    examples = pd.DataFrame(
        {
            "example_id": [0],
            "query_id": [0],
            "query": ["a"],
            "product_id": ["B0"],
            "product_locale": ["us"],
            "esci_label": ["E"],
            "small_version": [1],
            "large_version": [1],
        }
    )
    with pytest.raises(DataInvariantError, match="no 'split' column"):
        filter_task1(examples, "test")


# --- Review Focus 5: loading the wrong source file --------------------------

def test_check_invariants_accepts_the_documented_test_split():
    check_invariants("test", _test_shaped_judgements())


def test_wrong_judgement_count_names_the_hf_mirror_trap():
    frame = _test_shaped_judgements().iloc[:-1]
    with pytest.raises(DataInvariantError) as exc:
        check_invariants("test", frame)
    message = str(exc.value)
    assert "181701" in message.replace(",", "")
    assert "tasksource" in message


def test_the_hf_mirror_row_count_is_rejected():
    # The exact count the mirror produces, which is the realistic failure.
    stats = EXPECTED_STATS["test"]
    extra = 185_361 - stats.judgements
    frame = pd.concat(
        [_test_shaped_judgements(), _judgements(["E"] * extra, stats.queries)],
        ignore_index=True,
    )
    with pytest.raises(DataInvariantError, match="tasksource"):
        check_invariants("test", frame)


def test_swapped_substitute_and_complement_shares_are_rejected():
    # Complement 35% / Substitute 4% is the signature of the S/C swap bug in
    # the official prepare_trec_eval_files.py.
    stats = EXPECTED_STATS["test"]
    share = dict(EXPECTED_LABEL_SHARE["test"])
    share["S"], share["C"] = share["C"], share["S"]
    counts = {label: round(stats.judgements * s) for label, s in share.items()}
    counts["E"] += stats.judgements - sum(counts.values())
    labels = [label for label, n in counts.items() for _ in range(n)]
    with pytest.raises(DataInvariantError, match="label distribution"):
        check_invariants("test", _judgements(labels, stats.queries))


def test_wrong_query_count_is_rejected():
    frame = _test_shaped_judgements()
    frame["query_id"] = 0
    with pytest.raises(DataInvariantError, match="quer"):
        check_invariants("test", frame)


def test_train_split_has_no_label_share_expectation():
    # CLAUDE.md documents the label distribution for test only. Asserting the
    # test shares against train would be inventing a number.
    assert "train" not in EXPECTED_LABEL_SHARE
    assert EXPECTED_STATS["train"].products is None


# --- the real files ---------------------------------------------------------

@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "test"])
def test_real_split_satisfies_every_invariant(split):
    loaded = load_split(split)
    check_invariants(split, loaded.judgements, loaded.products)
    assert loaded.judgements["gain"].between(0.0, 1.0).all()
    assert set(loaded.judgements["product_id"]) <= set(loaded.products["product_id"])


@pytest.mark.data
def test_products_overlap_between_splits_is_the_documented_size():
    # 34,756 products appear in both splits. The split is query-level so this
    # is legitimate, but it is exactly why no feature may be computed from a
    # product alone using labels.
    train = set(load_split("train").judgements["product_id"])
    test = set(load_split("test").judgements["product_id"])
    assert len(train & test) == 34_756
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.dataset'`.

- [ ] **Step 3: Write `src/dataset.py`**

```python
"""Load the official Amazon ESCI Task 1 English data, and refuse the wrong file.

The failure this module exists to prevent is silent: the tasksource/esci HF
mirror parses cleanly, has the right column names, and returns 185,361 test
judgements instead of 181,701 because it encodes the large-version split.
Nothing about it looks wrong except the count, so the count is checked on
every load.

The official files live behind Git LFS. raw.githubusercontent.com serves the
LFS *pointer* (a 130-byte text file), not the Parquet, so downloads go through
media.githubusercontent.com/media/.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.labels import ESCI_GAINS, label_to_gain, label_to_qrel

BASE_URL = (
    "https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
    "shopping_queries_dataset/"
)

FILES = {
    "examples": "shopping_queries_dataset_examples.parquet",
    "products": "shopping_queries_dataset_products.parquet",
}

LABEL_SHARE_TOLERANCE = 0.001


class DataInvariantError(ValueError):
    """Raised when loaded data does not match the documented ESCI invariants."""


@dataclass(frozen=True)
class SplitStats:
    judgements: int
    queries: int
    products: int | None = None


EXPECTED_STATS: dict[str, SplitStats] = {
    "train": SplitStats(judgements=419_653, queries=20_888),
    "test": SplitStats(judgements=181_701, queries=8_956, products=164_900),
}

# Documented for the test split only. There is no published train figure, and
# asserting the test shares against train would be inventing a number.
EXPECTED_LABEL_SHARE: dict[str, dict[str, float]] = {
    "test": {"E": 0.4387, "S": 0.3498, "I": 0.1669, "C": 0.0446},
}


@dataclass(frozen=True)
class EsciSplit:
    name: str
    judgements: pd.DataFrame
    products: pd.DataFrame


def data_dir() -> Path:
    """Where the raw parquet lives. Override with ESCI_DATA_DIR."""
    return Path(os.environ.get("ESCI_DATA_DIR", "data/esci"))


def ensure_downloaded(name: str, data_dir_override: Path | None = None) -> Path:
    """Download one official parquet if it is not already on disk.

    products is ~1.03 GB and examples ~48.9 MB, so this prints progress and
    downloads to a .part file that is renamed only on success — an interrupted
    download must not leave a truncated parquet that later parses as garbage.
    """
    if name not in FILES:
        raise ValueError(f"unknown file {name!r}; expected one of {sorted(FILES)}")
    directory = data_dir_override or data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / FILES[name]
    if target.exists():
        return target

    partial = target.with_suffix(target.suffix + ".part")
    url = BASE_URL + FILES[name]
    print(f"downloading {url} -> {target}")

    def _progress(block_count: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(block_count * block_size, total)
            print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, partial, reporthook=_progress)
    print()
    partial.rename(target)
    return target


def filter_task1(examples: pd.DataFrame, split: str) -> pd.DataFrame:
    """Restrict the examples table to Task 1, English, one split."""
    if "split" not in examples.columns:
        raise DataInvariantError(
            "this examples table has no 'split' column, so it cannot be the "
            "official amazon-science/esci-data parquet. The tasksource/esci HF "
            "mirror has this shape and encodes the large-version split, which "
            "yields 185,361 test judgements instead of 181,701. Download from "
            f"{BASE_URL} instead."
        )
    mask = (
        (examples["small_version"] == 1)
        & (examples["product_locale"] == "us")
        & (examples["split"] == split)
    )
    return examples.loc[mask].reset_index(drop=True)


def check_invariants(
    split: str,
    judgements: pd.DataFrame,
    products: pd.DataFrame | None = None,
) -> None:
    """Assert the documented counts and label distribution, or raise.

    These were verified empirically against the real data and each one silently
    corrupts results if violated.
    """
    expected = EXPECTED_STATS[split]

    n_judgements = len(judgements)
    if n_judgements != expected.judgements:
        raise DataInvariantError(
            f"{split} split has {n_judgements:,} judgements, expected "
            f"{expected.judgements:,}. The usual cause is loading the "
            "tasksource/esci HF mirror, which has no 'split' column and encodes "
            "the large-version split. Use the official parquet from "
            f"{BASE_URL}"
        )

    n_queries = judgements["query_id"].nunique()
    if n_queries != expected.queries:
        raise DataInvariantError(
            f"{split} split has {n_queries:,} unique queries, expected "
            f"{expected.queries:,}"
        )

    if expected.products is not None and products is not None:
        n_products = products["product_id"].nunique()
        if n_products != expected.products:
            raise DataInvariantError(
                f"{split} split has {n_products:,} unique products, expected "
                f"{expected.products:,}"
            )

    expected_share = EXPECTED_LABEL_SHARE.get(split)
    if expected_share is not None:
        observed = judgements["esci_label"].value_counts(normalize=True)
        for label, share in expected_share.items():
            seen = float(observed.get(label, 0.0))
            if abs(seen - share) > LABEL_SHARE_TOLERANCE:
                raise DataInvariantError(
                    f"{split} label distribution is wrong: {label} is "
                    f"{seen:.4f}, expected {share:.4f}. Complement near 35% and "
                    "Substitute near 4% is the signature of the S/C swap bug in "
                    "the official prepare_trec_eval_files.py."
                )

    unknown = set(judgements["esci_label"]) - set(ESCI_GAINS)
    if unknown:
        raise DataInvariantError(f"unknown ESCI labels present: {sorted(unknown)}")


def load_split(split: str, data_dir_override: Path | None = None) -> EsciSplit:
    """Load one Task 1 English split, with every invariant asserted."""
    if split not in EXPECTED_STATS:
        raise ValueError(f"unknown split {split!r}; expected train or test")

    examples = pd.read_parquet(ensure_downloaded("examples", data_dir_override))
    judgements = filter_task1(examples, split)
    judgements = judgements[
        ["example_id", "query_id", "query", "product_id", "esci_label"]
    ].copy()
    judgements["gain"] = judgements["esci_label"].map(label_to_gain)
    judgements["qrel"] = judgements["esci_label"].map(label_to_qrel)

    all_products = pd.read_parquet(ensure_downloaded("products", data_dir_override))
    products = all_products.loc[
        (all_products["product_locale"] == "us")
        & all_products["product_id"].isin(set(judgements["product_id"]))
    ]
    products = products[
        [
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
        ]
    ].drop_duplicates("product_id").reset_index(drop=True)

    check_invariants(split, judgements, products)
    return EsciSplit(name=split, judgements=judgements, products=products)
```

- [ ] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: PASS, 8 tests. The two `@pytest.mark.data` tests are deselected.

- [ ] **Step 5: Download the real data and run the marked tests**

```bash
python -c "from src.dataset import ensure_downloaded; ensure_downloaded('examples'); ensure_downloaded('products')"
python -m pytest tests/test_dataset.py -v -m data
```

Expected: PASS, 3 tests (train, test, and the overlap check). ~1.08 GB downloaded, several minutes.

If `test_real_split_satisfies_every_invariant[test]` fails on the judgement count, read the error message before touching the expected numbers — the numbers in `EXPECTED_STATS` were verified empirically and are not the thing that is wrong.

- [ ] **Step 6: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "Load official ESCI Task 1 English splits with asserted invariants"
```

---

## Task 5: TREC run and qrels I/O

**Files:**
- Create: `src/runs.py`
- Create: `tests/test_runs.py`

**Interfaces:**
- Consumes: `src.dataset.EsciSplit`, `src.metrics.Qrels`, `src.metrics.Run`.
- Produces:
  - `src.runs.qrels_from_judgements(judgements: pd.DataFrame) -> dict[str, dict[str, int]]`
  - `src.runs.run_from_scores(scores: pd.DataFrame, score_column: str = "score") -> dict[str, dict[str, float]]`
  - `src.runs.write_run(run: Run, path: Path, run_tag: str) -> None`
  - `src.runs.read_run(path: Path) -> dict[str, dict[str, float]]`
  - `src.runs.write_qrels(qrels: Qrels, path: Path) -> None`

Ids are strings in every dict, because TREC files are text and `query_id` is an `int64` in the parquet. Converting at the boundary once means nothing downstream has to remember to.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runs.py`:

```python
import pandas as pd
import pytest

from src.runs import (
    qrels_from_judgements,
    read_run,
    run_from_scores,
    write_qrels,
    write_run,
)


def _judgements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "product_id": ["B1", "B2", "B3"],
            "esci_label": ["E", "I", "S"],
            "qrel": [100, 0, 10],
        }
    )


def test_qrels_keys_are_strings_not_numpy_integers():
    qrels = qrels_from_judgements(_judgements())
    assert set(qrels) == {"1", "2"}
    assert all(isinstance(k, str) for k in qrels)
    assert qrels["1"] == {"B1": 100, "B2": 0}


def test_qrel_values_are_plain_ints():
    # pytrec_eval rejects numpy.int64.
    qrels = qrels_from_judgements(_judgements())
    assert all(type(v) is int for docs in qrels.values() for v in docs.values())


def test_duplicate_query_product_pair_raises_instead_of_overwriting():
    duplicated = pd.concat([_judgements(), _judgements().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        qrels_from_judgements(duplicated)


def test_run_from_scores_shapes_a_nested_dict():
    scores = pd.DataFrame(
        {"query_id": [1, 1], "product_id": ["B1", "B2"], "score": [0.5, 0.25]}
    )
    assert run_from_scores(scores) == {"1": {"B1": 0.5, "B2": 0.25}}


def test_run_file_round_trip_preserves_scores_exactly(tmp_path):
    # Formatting scores as %.6f would collapse near-identical scores into ties
    # and change the ranking, so the round trip must be exact.
    run = {"1": {"B1": 0.1234567890123456, "B2": 0.1234567890123457}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    assert read_run(path) == run


def test_run_file_has_the_six_trec_columns_in_rank_order(tmp_path):
    run = {"1": {"B2": 0.5, "B1": 0.9}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    rows = [line.split() for line in path.read_text().splitlines()]
    assert [r[0] for r in rows] == ["1", "1"]
    assert [r[1] for r in rows] == ["Q0", "Q0"]
    assert [r[2] for r in rows] == ["B1", "B2"]  # sorted by descending score
    assert [r[3] for r in rows] == ["1", "2"]  # rank is 1-based
    assert [r[5] for r in rows] == ["unit", "unit"]


def test_run_file_ties_are_written_in_ascending_document_id(tmp_path):
    # Must match the tie policy in src/metrics.py, or a run scores differently
    # after a round trip through disk.
    run = {"1": {"B2": 0.5, "B1": 0.5}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    assert [line.split()[2] for line in path.read_text().splitlines()] == ["B1", "B2"]


def test_qrels_file_has_the_four_trec_columns(tmp_path):
    path = tmp_path / "qrels.txt"
    write_qrels({"1": {"B1": 100, "B2": 0}}, path)
    rows = [line.split() for line in path.read_text().splitlines()]
    assert rows == [["1", "0", "B1", "100"], ["1", "0", "B2", "0"]]


def test_reading_a_malformed_run_line_raises_with_the_line_number(tmp_path):
    path = tmp_path / "run.trec"
    path.write_text("1 Q0 B1 1 0.9 tag\nnot a run line\n")
    with pytest.raises(ValueError, match="line 2"):
        read_run(path)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_runs.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.runs'`.

- [ ] **Step 3: Write `src/runs.py`**

```python
"""TREC-format run and qrels I/O, and DataFrame -> nested-dict conversion.

Ids are strings everywhere: TREC files are text, query_id is int64 in the
parquet, and pytrec_eval rejects numpy scalars. Converting once at this
boundary means nothing downstream has to remember to.

Scores are written at full float precision rather than a fixed number of
decimals. Rounding to %.6f would turn distinct scores into ties and silently
change the ranking between an in-memory run and the same run read back.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.metrics import Qrels, Run


def _ranked(docs: Mapping[str, float]) -> list[tuple[str, float]]:
    """Documents ordered the way src.metrics.ndcg_per_query orders them."""
    return sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))


def qrels_from_judgements(judgements: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Nested qrels dict from a judgements frame carrying a `qrel` column."""
    duplicated = judgements.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = judgements.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}) in judgements; "
            "building a dict would silently keep only the last one"
        )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, product_id, qrel in zip(
        judgements["query_id"], judgements["product_id"], judgements["qrel"]
    ):
        qrels.setdefault(str(query_id), {})[str(product_id)] = int(qrel)
    return qrels


def run_from_scores(
    scores: pd.DataFrame, score_column: str = "score"
) -> dict[str, dict[str, float]]:
    """Nested run dict from a frame of query_id / product_id / score."""
    duplicated = scores.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = scores.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}) in scores"
        )
    run: dict[str, dict[str, float]] = {}
    for query_id, product_id, score in zip(
        scores["query_id"], scores["product_id"], scores[score_column]
    ):
        run.setdefault(str(query_id), {})[str(product_id)] = float(score)
    return run


def write_run(run: Run, path: Path, run_tag: str) -> None:
    """Write a TREC run file: qid Q0 docid rank score tag."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(run):
            for rank, (doc_id, score) in enumerate(_ranked(run[qid]), start=1):
                # float() first: numpy 2.x reprs a float64 as "np.float64(0.5)",
                # which read_run could not parse back.
                fh.write(f"{qid} Q0 {doc_id} {rank} {float(score)!r} {run_tag}\n")


def read_run(path: Path) -> dict[str, dict[str, float]]:
    """Read a TREC run file back into a nested dict."""
    run: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}: line {line_number} has {len(fields)} fields, "
                    f"expected 6 (qid Q0 docid rank score tag): {line.strip()!r}"
                )
            qid, _, doc_id, _, score, _ = fields
            try:
                run.setdefault(qid, {})[doc_id] = float(score)
            except ValueError:
                raise ValueError(
                    f"{path}: line {line_number} has a non-numeric score {score!r}"
                ) from None
    return run


def write_qrels(qrels: Qrels, path: Path) -> None:
    """Write a TREC qrels file: qid 0 docid relevance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                fh.write(f"{qid} 0 {doc_id} {int(qrels[qid][doc_id])}\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_runs.py -v
```

Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/runs.py tests/test_runs.py
git commit -m "Add TREC run and qrels I/O with exact score round-tripping"
```

---

## Task 6: The computed random floor

**Files:**
- Create: `src/floor.py`
- Create: `tests/test_floor.py`

**Interfaces:**
- Consumes: `src.metrics.ndcg_per_query(run, qrels)`, `src.metrics.Qrels`.
- Produces:
  - `src.floor.FloorResult(mean: float, low: float, high: float, per_trial: tuple[float, ...], n_trials: int, seed: int)`
  - `src.floor.random_run(qrels: Qrels, rng: random.Random) -> dict[str, dict[str, float]]`
  - `src.floor.random_floor(qrels: Qrels, *, n_trials: int = 100, seed: int = 0, alpha: float = 0.05) -> FloorResult`

`random_run` assigns a distinct random score to every judged document, so the floor measures random *ordering* and never touches the tie-breaking path.

- [ ] **Step 1: Write the failing test**

Create `tests/test_floor.py`:

```python
import pytest

from src.floor import random_floor, random_run


def test_uniform_gains_make_every_ordering_ideal():
    # If every judged document has the same gain, DCG == IDCG for any order,
    # so the floor is exactly 1.0. Anything else means the ideal is being
    # computed from the run rather than from the judgements.
    qrels = {f"q{i}": {f"d{j}": 100 for j in range(5)} for i in range(20)}
    assert random_floor(qrels, n_trials=5, seed=0).mean == pytest.approx(1.0)


def test_floor_of_an_all_irrelevant_query_is_zero():
    qrels = {"q": {"a": 0, "b": 0, "c": 0}}
    assert random_floor(qrels, n_trials=5, seed=0).mean == 0.0


def test_floor_sits_between_zero_and_one_on_mixed_labels():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 1, "d": 0} for i in range(50)}
    result = random_floor(qrels, n_trials=20, seed=0)
    assert 0.0 < result.mean < 1.0
    assert result.low <= result.mean <= result.high


def test_same_seed_gives_the_same_floor():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    assert random_floor(qrels, n_trials=10, seed=7).per_trial == (
        random_floor(qrels, n_trials=10, seed=7).per_trial
    )


def test_different_seeds_give_different_trials():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    assert random_floor(qrels, n_trials=10, seed=0).per_trial != (
        random_floor(qrels, n_trials=10, seed=1).per_trial
    )


def test_result_records_its_own_settings():
    qrels = {"q": {"a": 100, "b": 0}}
    result = random_floor(qrels, n_trials=4, seed=3)
    assert result.n_trials == 4
    assert result.seed == 3
    assert len(result.per_trial) == 4


def test_random_run_scores_every_judged_document_distinctly():
    import random

    qrels = {"q": {f"d{i}": 0 for i in range(100)}}
    run = random_run(qrels, random.Random(0))
    assert set(run["q"]) == set(qrels["q"])
    assert len(set(run["q"].values())) == 100


def test_zero_trials_raises():
    with pytest.raises(ValueError, match="n_trials"):
        random_floor({"q": {"a": 100}}, n_trials=0)


@pytest.mark.data
@pytest.mark.slow
def test_real_test_split_floor_is_near_the_documented_measurement():
    from src.dataset import load_split
    from src.runs import qrels_from_judgements

    qrels = qrels_from_judgements(load_split("test").judgements)
    result = random_floor(qrels, n_trials=100, seed=0)
    print(f"\nrandom floor (test) = {result.mean:.4f} [{result.low:.4f}, {result.high:.4f}]")
    # CLAUDE.md records 0.7467 measured with the 1/log2(rank+1) discount; the
    # published SQID figure is 0.7483; under the swapped S/C mapping it is
    # 0.7141.
    #
    # The band is wider than that spread on purpose. Simulating the documented
    # marginal label distribution (E 43.87 / S 34.98 / I 16.69 / C 4.46) over
    # ~20 candidates per query lands at 0.755-0.776 across i.i.d.,
    # Dirichlet-correlated and count-skewed variants -- consistently above
    # 0.7467. So the real per-query label structure is not reconstructible from
    # the marginal alone, and a band tight around 0.7467 would be a band around
    # a number this project cannot currently derive from first principles.
    #
    # What this test guards is the large moves: the S/C swap (0.7141), an
    # exponential gain, a wrong discount, or a degenerate uniform gain (1.0).
    # It is not a precision check on the fourth decimal.
    assert 0.730 < result.mean < 0.780
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_floor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.floor'`.

- [ ] **Step 3: Write `src/floor.py`**

```python
"""The random-ordering NDCG floor, computed rather than quoted.

Roughly 44% of ESCI judgements are Exact, so shuffling the judged candidate
list already scores near 0.75 on full-list NDCG. A headline of 0.86 means
nothing without that floor attached.

The number is computed on every report because it moves with the gain mapping:
CLAUDE.md records 0.7467 with the standard discount, the published SQID figure
is 0.7483, and under the S/C swap it is 0.7141. That 3.3-point spread is larger
than most method gains, so a hard-coded constant would be a way to be wrong by
more than the effect being measured.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from src.metrics import Qrels, ndcg_per_query


@dataclass(frozen=True)
class FloorResult:
    mean: float
    low: float
    high: float
    per_trial: tuple[float, ...]
    n_trials: int
    seed: int


def random_run(qrels: Qrels, rng: random.Random) -> dict[str, dict[str, float]]:
    """Score every judged document with a distinct random value.

    Distinct scores keep the floor a measurement of random *ordering* rather
    than of the tie-breaking policy.
    """
    run: dict[str, dict[str, float]] = {}
    for qid, judged in qrels.items():
        order = list(judged)
        rng.shuffle(order)
        run[qid] = {doc_id: float(rank) for rank, doc_id in enumerate(order)}
    return run


def random_floor(
    qrels: Qrels,
    *,
    n_trials: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> FloorResult:
    """Mean NDCG of a random ordering, with a percentile interval over trials."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")

    rng = random.Random(seed)
    per_trial: list[float] = []
    for _ in range(n_trials):
        scores = ndcg_per_query(random_run(qrels, rng), qrels)
        per_trial.append(sum(scores.values()) / len(scores))

    ordered = sorted(per_trial)
    low_index = int((alpha / 2) * (len(ordered) - 1))
    high_index = int((1 - alpha / 2) * (len(ordered) - 1))
    return FloorResult(
        mean=statistics.fmean(per_trial),
        low=ordered[low_index],
        high=ordered[high_index],
        per_trial=tuple(per_trial),
        n_trials=n_trials,
        seed=seed,
    )
```

- [ ] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_floor.py -v
```

Expected: PASS, 8 tests; the real-data test is deselected.

- [ ] **Step 5: Measure the floor on the real test split and record it**

```bash
python -m pytest tests/test_floor.py -v -m "data and slow" -s
```

Expected: PASS, and the printed floor lands near 0.7467. Takes 1–2 minutes: 100 trials × 181,701 scored documents.

Record the measured value in the commit message so it is recoverable from history without re-running.

**If the measured floor differs from 0.7467 by more than 0.005, that is a finding, not a nuisance.** Record it, and check in this order: the gain mapping in `src/labels.py` (a swap lands near 0.7141), the discount in `src/metrics.py`, and whether `load_split` filtered to `small_version == 1` and `product_locale == "us"`. If all three are right and the number still differs, report the measured value as this project's floor and note the discrepancy with CLAUDE.md in the commit message. Every later NDCG in the series is quoted against this number, so it must be the one this code actually produces.

- [ ] **Step 6: Commit**

```bash
git add src/floor.py tests/test_floor.py
git commit -m "Compute the random-ordering NDCG floor from qrels"
```

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.bootstrap'`.

- [ ] **Step 3: Write `src/bootstrap.py`**

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

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

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

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_splits.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.splits'`.

- [ ] **Step 3: Write `src/splits.py`**

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

- [ ] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_splits.py -v
```

Expected: PASS, 12 tests; the real-data test is deselected.

- [ ] **Step 5: Freeze the folds against the real train split and commit them**

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

- [ ] **Step 6: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_report.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.report'`.

- [ ] **Step 3: Write `src/report.py`**

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

- [ ] **Step 4: Add `per_query` to `FloorResult` in `src/floor.py`**

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

- [ ] **Step 5: Write `src/cli.py`**

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

- [ ] **Step 6: Run the whole fast suite to verify it passes**

```bash
python -m pytest -v
```

Expected: PASS. No failures, no errors. `tests/test_metrics_vs_pytrec.py` must not skip.

- [ ] **Step 7: Record the commands in `CLAUDE.md`**

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

- [ ] **Step 8: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_baseline_sbert.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.baseline_sbert'`.

- [ ] **Step 3: Write `src/baseline_sbert.py`**

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

- [ ] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_baseline_sbert.py -v
```

Expected: PASS, 8 tests; the reproduction test is deselected.

- [ ] **Step 5: Run the real baseline and evaluate it**

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

- [ ] **Step 6: Run the marked reproduction test**

```bash
python -m pytest tests/test_baseline_sbert.py -v -m "data and slow"
```

Expected: PASS. This reads the committed `docs/results/sbert-title-test.json`, so it is fast once the run above has been done.

- [ ] **Step 7: Commit**

```bash
git add src/baseline_sbert.py tests/test_baseline_sbert.py docs/results/sbert-title-test.json
git commit -m "Reproduce the zero-shot SBERT baseline on the test split"
```

---

## Plan Gate

Plan 2 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no skips in `tests/test_metrics_vs_pytrec.py`.
- [ ] `python -m pytest -m data` passes for both splits.
- [ ] The random floor, measured not quoted, is recorded in `docs/results/` and lands near 0.7467.
- [ ] `docs/results/sbert-title-test.json` shows NDCG in `[0.820, 0.840]` with a lift interval strictly above zero.
- [ ] `splits/val_folds.csv` and its `.sha256` are committed, covering all 20,888 train queries.
- [ ] `CLAUDE.md`'s Commands section lists the test and evaluation invocations.

---

## Self-Review

Run against `PROJECT_SPEC.md` and `CLAUDE.md` after writing the plan.

**1. Spec coverage.** §2 (task, labels, metric, S/C bug) → Tasks 1, 2, 3. §3.1 (dataset, splits, counts) → Task 4. §5 (baselines, the 0.8562 target) → Task 10 establishes the measuring apparatus; beating 0.8562 is Plan 7's gate. §8.1 (the build-order gate) → Task 10. `CLAUDE.md` evaluation discipline (train/test product overlap, GroupKFold by query_id, bootstrap CIs) → Tasks 4, 7, 8.

Deliberately out of scope for this plan, and owned elsewhere: §3.2 and §3.3 (ESCI-S, images) → Plans 2 and 3; §4 (the funnel stages) → Plans 4–7; §6 (the ablation table) → Plans 4–7; §7 (methods) → Plans 4–6; §9 (compute) → Plans 3 and 6, where GPU work first appears.

**2. Placeholder scan.** No "TBD", no "add appropriate error handling", no "similar to Task N". Every code step carries the code. The one repeated idea — the ascending-document-id tie policy — is written out in full in both `src/metrics.py` and `src/runs.py` and cross-tested, rather than one referring to the other.

**3. Type consistency.** `Qrels` and `Run` are both `Mapping[str, Mapping[str, float]]`, defined once in `src/metrics.py` and imported by `src/runs.py` and `src/report.py`. Ids are `str` everywhere past `src/runs.py`. `Interval` is defined once in `src/bootstrap.py` and used by `src/report.py`. `FloorResult.per_query` is added in Task 9 Step 4 with a covering test, and Task 9's Interfaces block names it as a dependency so nobody implements `report.py` against the Task 6 shape.

**4. Review Focus.** All five items have tests in the task that owns the code: partial runs, unjudged documents and zero-ideal queries in `tests/test_metrics.py` (Task 2) and again against `pytrec_eval` (Task 3); ties in `tests/test_metrics.py` and `tests/test_runs.py` (Tasks 2 and 5); the wrong source file in `tests/test_dataset.py` (Task 4), including the mirror's exact 185,361 row count and the S/C-swapped label distribution.
