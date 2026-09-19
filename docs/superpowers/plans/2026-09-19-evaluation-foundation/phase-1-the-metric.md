# Phase 1 — The Metric

**Plan 1 of 7 · Phase 1 of 3 · Tasks 1–3.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into.

**Delivers:** the ESCI label→gain tables, full-list NDCG matching
`trec_eval -m ndcg`, and a randomised cross-check against `pytrec_eval` that
licenses every number the rest of the series reports.

**Needs on disk:** nothing. No dataset download in this phase — that is Phase 2.
`src/metrics.py` deliberately holds no pandas and no I/O so it can be tested
against the oracle with no data loading in the way.

**Owns Review Focus items 1–4** (partial runs, unjudged documents, zero-ideal
queries, ties), all pinned in Task 2 and re-checked against the oracle in Task 3.

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Gains are E=1.0, S=0.1, C=0.01, I=0.0.** Integer qrels are the same values ×100 (E=100, S=10, C=1, I=0) so they can be handed to `pytrec_eval`, which requires integers. NDCG is invariant to this scaling; Task 2 tests that it is.
- **NDCG is full-list, no cutoff**, with gain used linearly (not `2^rel - 1`) and discount `1/log2(rank+1)` with rank starting at 1. This matches `trec_eval -m ndcg`, the official ESCI evaluation route.
- **Any source reporting Complement ≈ 35% has the known S/C swap bug** in the official `prepare_trec_eval_files.py`. Under the swapped mapping the random floor moves 0.7467 → 0.7141, a larger move than most method gains, so the ordering `S > C` is pinned explicitly in Task 1.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line.

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_labels.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.labels'`.

- [x] **Step 3: Write `pyproject.toml`**

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

- [x] **Step 4: Write `src/labels.py`**

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

- [x] **Step 5: Add the results directory to git and adjust `.gitignore`**

`.gitignore` already blocks `data/`, `runs/`, `*.parquet` and `*.npy`, which is correct — raw data and run files are outputs. Append a short note so the next person does not "fix" it by un-ignoring `data/`:

```bash
cat >> .gitignore <<'EOF'

# Frozen split artifacts live in splits/ (not data/) precisely so they escape
# the data/ rule above and can be committed. Keep them there.
EOF
```

- [x] **Step 6: Install and run the tests to verify they pass**

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_labels.py -v
```

Expected: PASS, 6 tests.

If `pytrec-eval-terrier` fails to build, it needs a C toolchain: `sudo apt-get install -y build-essential python3-dev` then retry. Do not skip it — Task 3 uses it as the correctness oracle and there is no substitute.

- [x] **Step 7: Commit**

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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`.

- [x] **Step 3: Write `src/metrics.py`**

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

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_metrics.py -v
```

Expected: PASS, 14 tests.

- [x] **Step 5: Commit**

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

## Phase 1 Gate

Phase 2 does not start until all of these hold:

- [ ] `python -m pytest tests/test_labels.py tests/test_metrics.py tests/test_metrics_vs_pytrec.py -v` passes — 6 + 14 + 13 tests.
- [ ] `tests/test_metrics_vs_pytrec.py` **does not skip.** A skip means `pytrec-eval-terrier` is not installed, and that is a failed task, not a pass: there is no substitute oracle.
- [ ] `python -m pytest` with no arguments is still fast enough to run after every edit (the `data` and `slow` markers are deselected by `addopts`).

Next: [Phase 2 — The Data](phase-2-the-data.md).
