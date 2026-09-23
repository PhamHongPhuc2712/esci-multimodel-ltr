# Phase 2 — The Blend Report

**Plan 7 of 7 · Phase 2 of 3 · Task 3.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 1 gate](phase-1-the-signals-and-the-blend.md#phase-1-gate) passes.

**Delivers:** `src/blend_report.py` — Stage 4's three strategies and the oracle
ceiling, measured against **each single stage alone**, on fold 0 (cross-fitted)
and on the frozen test sample.

**Needs on disk:** `data/features/stage-signals-{fold0,test-sample}.parquet`
from Phase 1, plus the matrices and Stage 2 orderings they were built from.

**Owns no Review Focus item** — Phase 1 owns the two leaks this phase could
suffer from, and Phase 3 owns the three reporting ones. This phase's job is to
not invent a new way to be wrong, which is why it reuses
`src.fine_rank_report.check_same_queries` rather than writing its own.

### Constraints that bite hardest here

- **§4.4 asks for the comparison against each single stage alone**, not just against Stage 2. A blend that beats Stage 2 but loses to the LLM has not earned anything.
- **The oracle is reported beside the arms**, so a tie reads as "the methods missed available headroom" rather than "there was none".
- **Never tune on test.** Weights, features and round counts are fixed on fold 0; the test sample is predicted once per arm.
- **A method that ties, or loses, is the result.** Four pre-measurements say this one loses. Do not tune until an arm wins.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## What the pre-measurement says this will find

From [`README.md`](README.md#where-these-numbers-came-from), all on fold 0:

| | NDCG |
|---|---|
| `stage2` | 0.8519 |
| `stage2+ce` | 0.8587 |
| **`stage2+llm` (best single)** | **0.8814** |
| fixed weight, best swept 1:1:4 | 0.8800 |
| learned combiner, cross-fitted | 0.8793 |
| **oracle best-of-three** | **0.9076** |

So the expected finding is: **both measured strategies lose to their own best
input, by roughly 0.0015 to 0.0020, while +0.0262 of headroom sits unclaimed.**
The selector is the untested arm. Write the report so that outcome is a clean,
publishable row rather than something to explain away — and if the selector
also loses, say so in the same breath.

---

## Task 3: Measure Stage 4 against every stage alone

**Files:**
- Create: `src/blend_report.py`
- Create: `tests/test_blend_report.py`

**Interfaces:**
- Consumes: `src.blend.*` (Task 2), `src.stage_signals.{load_signals, SCOPES}` (Task 1), `src.rerank_window.{windows, spliced_run, stage2_run, DEFAULT_K}`, `src.rank_report.{qrels_from_frame, evaluate_arm, compare, format_table}`, `src.fine_rank_report.check_same_queries`, `src.floor.random_floor`, `src.metrics.ndcg_per_query`.
- Produces:
  - `src.blend_report.SINGLE_STAGES: tuple[str, ...]` = `("stage2", "stage2+ce", "stage2+llm")`
  - `src.blend_report.BLEND_ARMS: tuple[str, ...]` = `("blend_fixed", "blend_combiner", "blend_selector")`
  - `src.blend_report.ORACLE: str` = `"oracle_best_of_three"`
  - `src.blend_report.single_stage_orderings(frame) -> dict[str, dict[str, list[str]]]`
  - `src.blend_report.per_query_ndcg(orderings, windows_, qrels) -> dict[str, float]`
  - `src.blend_report.best_single(results) -> Any`
  - `src.blend_report.rebuild_windows(scope, *, k=None, features_dir=Path("data/features"), seed=0) -> tuple[list, pd.DataFrame]` (`k=None` means `DEFAULT_K`)
  - CLI: `python -m src.blend_report --scope fold0` and `--scope test-sample --final`

**Windows are rebuilt, then checked against the signal frame.** The signal
frame holds only window documents, not tails, and NDCG needs the whole judged
list. `rebuild_windows` runs the same `windows(joined, k)` call Phase 1 ran, and
then asserts the window documents it produced are exactly the ones in the
signal frame. If Stage 2's parquet were ever re-dumped with a different fit, the
two would silently disagree about which ten documents Stage 3 ranked, and every
Stage 4 number would be measured over a candidate set the arms never saw.

- [ ] **Step 1: Write the failing test**

Create `tests/test_blend_report.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.blend_report import (
    BLEND_ARMS,
    ORACLE,
    SINGLE_STAGES,
    best_single,
    per_query_ndcg,
    single_stage_orderings,
)
from src.rank_report import evaluate_arm
from src.rerank_window import Window


def _signals():
    return pd.DataFrame(
        {
            "query_id": ["1", "1", "1", "2", "2"],
            "product_id": ["a", "b", "c", "x", "y"],
            "stage2_score": [3.0, 2.0, 1.0, 9.0, 8.0],
            "stage2_rank": [1, 2, 3, 1, 2],
            "ce_score": [0.1, 0.9, 0.5, 0.2, 0.8],
            "llm_rank": [3, 1, 2, 2, 1],
            "llm_rr": [1 / 63, 1 / 61, 1 / 62, 1 / 62, 1 / 61],
            "label_code": [3, 0, 2, 3, 0],
            "gain": [1.0, 0.0, 0.1, 1.0, 0.0],
            "qrel": [100, 0, 10, 100, 0],
        }
    )


def _windows():
    return [
        Window(query_id="1", window=("a", "b", "c"), tail=("z",)),
        Window(query_id="2", window=("x", "y"), tail=()),
    ]


def _qrels():
    return {"1": {"a": 100, "b": 0, "c": 10, "z": 0}, "2": {"x": 100, "y": 0}}


def _arm(name, values):
    per = {str(i): v for i, v in enumerate(values)}
    floor = {str(i): 0.74 for i in range(len(values))}
    return evaluate_arm(
        name, per, floor, groups=("blend",), n_features=0,
        objective="blend", best_iteration=0,
    )


# --- the three single stages ------------------------------------------------

def test_every_single_stage_gets_an_ordering():
    out = single_stage_orderings(_signals())
    assert set(out) == set(SINGLE_STAGES)


def test_the_stage_2_ordering_is_the_window_order():
    out = single_stage_orderings(_signals())
    assert out["stage2"]["1"] == ["a", "b", "c"]


def test_the_cross_encoder_ordering_follows_its_logit():
    out = single_stage_orderings(_signals())
    assert out["stage2+ce"]["1"] == ["b", "c", "a"]


def test_the_llm_ordering_follows_its_rank():
    out = single_stage_orderings(_signals())
    assert out["stage2+llm"]["1"] == ["b", "c", "a"]


# --- scoring ----------------------------------------------------------------

def test_per_query_ndcg_covers_every_query():
    out = single_stage_orderings(_signals())
    per = per_query_ndcg(out["stage2"], _windows(), _qrels())
    assert set(per) == {"1", "2"}
    assert all(0.0 <= v <= 1.0 for v in per.values())


def test_a_better_ordering_scores_higher():
    # Putting the Exact product first must beat putting it last.
    good = {"1": ["a", "c", "b"], "2": ["x", "y"]}
    bad = {"1": ["b", "c", "a"], "2": ["y", "x"]}
    g = per_query_ndcg(good, _windows(), _qrels())
    b = per_query_ndcg(bad, _windows(), _qrels())
    assert sum(g.values()) > sum(b.values())


def test_scoring_goes_through_the_splice_so_the_tail_stays_below():
    # Query 1 has a tail; a reordering must never promote it.
    per = per_query_ndcg({"1": ["c", "b", "a"], "2": ["x", "y"]}, _windows(), _qrels())
    assert set(per) == {"1", "2"}


def test_a_non_permutation_is_refused():
    with pytest.raises(ValueError, match="permutation"):
        per_query_ndcg({"1": ["a", "b"], "2": ["x", "y"]}, _windows(), _qrels())


# --- the comparison §4.4 actually asks for ---------------------------------

def test_the_best_single_stage_is_picked_by_ndcg():
    results = [_arm("stage2", [0.80, 0.80]), _arm("stage2+ce", [0.85, 0.85]),
               _arm("stage2+llm", [0.90, 0.90])]
    assert best_single(results).name == "stage2+llm"


def test_best_single_ignores_blend_and_oracle_arms():
    # A blend that beats Stage 2 but loses to the LLM has earned nothing, so
    # the baseline for the blend arms must be the best *single stage*.
    results = [_arm("stage2", [0.80]), _arm("stage2+llm", [0.90]),
               _arm("blend_fixed", [0.95]), _arm(ORACLE, [0.99])]
    assert best_single(results).name == "stage2+llm"


def test_best_single_needs_a_single_stage():
    with pytest.raises(ValueError, match="single stage"):
        best_single([_arm("blend_fixed", [0.9])])


def test_the_declared_arms_cover_every_strategy_and_the_ceiling():
    from src.blend import STRATEGIES

    assert len(BLEND_ARMS) == len(STRATEGIES)
    assert ORACLE not in BLEND_ARMS          # a ceiling is not an arm
    assert ORACLE not in SINGLE_STAGES
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_blend_report.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.blend_report'`.

- [ ] **Step 3: Write the implementation**

Create `src/blend_report.py`:

```python
"""Stage 4 against every stage alone - PROJECT_SPEC.md §4.4.

The spec asks for "a weighted combination, or a small learned combiner over
stage scores" and for the comparison "against each single stage alone". That
last clause is the whole test: a blend that beats Stage 2 while losing to the
LLM it contains has earned nothing, so every blend arm is compared against the
**best single stage**, not just against the coarse ranker.

The oracle - best arm per query, chosen with the labels - is reported beside
them. It is not a method and cannot be one. It is here so that a losing blend
reads as "the strategies missed 0.0262 of available headroom" rather than
"there was nothing to find".

Measured before this module existed, on fold 0: fixed-weight fusion 0.8800 at
its best swept weights and a cross-fitted combiner 0.8793, against the LLM's
own 0.8814 and an oracle of 0.9076. The expected result is a loss. It is
written down as one.

Fold-0 numbers for the learned arms are **cross-fitted**, because fold 0 is the
only query set with both an LLM ordering and a label, so it is both the only
training set and the reporting surface. Every such number is labelled.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.blend import ARM_COLUMNS, orderings_from_scores
from src.metrics import ndcg_per_query
from src.rank_report import compare, evaluate_arm, format_table, qrels_from_frame
from src.rerank_window import spliced_run

DEFAULT_OUT = Path("docs/results/blend.json")

# PROJECT_SPEC.md §5. Plan 5 matched it at 0.8579; Plan 6's LLM arm reached
# 0.8855 on the 2,000-query sample.
ESCI_BASELINE = 0.8562

SINGLE_STAGES: tuple[str, ...] = ("stage2", "stage2+ce", "stage2+llm")
BLEND_ARMS: tuple[str, ...] = ("blend_fixed", "blend_combiner", "blend_selector")
ORACLE = "oracle_best_of_three"

_STAGE_OF_ARM = {"stage2": "stage2", "stage2+ce": "ce", "stage2+llm": "llm"}


def single_stage_orderings(frame: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Each single stage's own ordering of every window."""
    out: dict[str, dict[str, list[str]]] = {}
    for arm, stage in _STAGE_OF_ARM.items():
        column, ascending = ARM_COLUMNS[stage]
        out[arm] = orderings_from_scores(frame, column, ascending=ascending)
    return out


def per_query_ndcg(
    orderings: Mapping[str, Sequence[str]],
    windows_: Sequence,
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, float]:
    """NDCG per query for one ordering of the windows.

    Goes through `spliced_run`, which refuses anything that is not a
    permutation of the window and keeps the untouched tail strictly below it.
    """
    return ndcg_per_query(spliced_run(windows_, dict(orderings)), qrels)


def best_single(results: Sequence) -> object:
    """The strongest single-stage arm - the baseline §4.4 actually asks for."""
    singles = [arm for arm in results if arm.name in SINGLE_STAGES]
    if not singles:
        raise ValueError(
            "no single stage in the results; the blend arms need one to be "
            "compared against, or a blend that merely beats Stage 2 looks like "
            "a win"
        )
    return max(singles, key=lambda arm: arm.ndcg.point)


def rebuild_windows(
    scope: str,
    *,
    k: int | None = None,
    features_dir: Path = Path("data/features"),
    seed: int = 0,
):
    """The same windows Phase 1 built, plus the judged matrix behind them.

    The signal frame holds window documents only; NDCG needs the tail too. So
    the windows are rebuilt through the same call and then checked against the
    frame - if Stage 2's parquet were ever re-dumped under a different fit, the
    two would disagree about which ten documents Stage 3 ranked and every
    number here would be measured over a candidate set no arm ever saw.
    """
    from src.fine_rank_report import TEST_SAMPLE
    from src.ranker import REPORT_FOLD
    from src.rerank_window import DEFAULT_K, windows
    from src.stage2_scores import load_stage2

    k = DEFAULT_K if k is None else k
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
    joined = matrix[["query_id", "product_id"]].merge(
        load_stage2(split), on=["query_id", "product_id"]
    )
    return windows(joined, k=k), matrix


def _check_windows_match_signals(windows_, signals: pd.DataFrame) -> None:
    from_windows = {(w.query_id, d) for w in windows_ for d in w.window}
    from_signals = set(
        zip(signals["query_id"].astype(str), signals["product_id"].astype(str))
    )
    if from_windows != from_signals:
        raise ValueError(
            f"the rebuilt windows and the signal frame disagree about "
            f"{len(from_windows ^ from_signals):,} (query, document) pairs; "
            "re-run python -m src.stage_signals for this scope"
        )


def _main() -> int:
    from src.blend import (
        DEFAULT_WEIGHTS,
        cross_fit_predict,
        fixed_weight_ordering,
        oracle_ordering,
        predict_combiner,
        selector_ordering,
        train_combiner,
        train_selector,
    )
    from src.fine_rank_report import check_same_queries
    from src.floor import random_floor
    from src.stage_signals import SCOPES, load_signals

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="fold0", choices=list(SCOPES))
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--final", action="store_true",
                        help="required with --scope test-sample")
    args = parser.parse_args()

    if args.scope == "test-sample" and not args.final:
        raise SystemExit(
            "refusing to touch the test sample without --final. The weights, "
            "the features and the round count are all chosen on fold 0."
        )

    signals = load_signals(args.scope, args.features_dir)
    ws, matrix = rebuild_windows(
        args.scope, features_dir=args.features_dir, seed=args.seed
    )
    _check_windows_match_signals(ws, signals)

    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    print(f"{args.scope}: {len(matrix):,} judgements over {len(qrels):,} queries; "
          f"floor {floor.mean:.4f}")

    orderings: dict[str, dict[str, list[str]]] = single_stage_orderings(signals)
    per_query: dict[str, dict[str, float]] = {}

    # --- the three single stages --------------------------------------------
    for arm in SINGLE_STAGES:
        per_query[arm] = per_query_ndcg(orderings[arm], ws, qrels)

    # --- Stage 4, arm 1: fixed weights --------------------------------------
    orderings["blend_fixed"] = fixed_weight_ordering(signals, DEFAULT_WEIGHTS)

    # --- Stage 4, arm 2: the learned combiner -------------------------------
    # Fold 0 is cross-fitted because it is both the only training set and the
    # reporting surface; the test sample is predicted by a model fitted on all
    # of fold 0, which it has never seen.
    if args.scope == "fold0":
        blend_scores = cross_fit_predict(signals, seed=args.seed)
        fitted_on = "fold 0, cross-fitted"
    else:
        booster = train_combiner(load_signals("fold0", args.features_dir), seed=args.seed)
        blend_scores = predict_combiner(booster, signals)
        fitted_on = "fold 0"
    orderings["blend_combiner"] = orderings_from_scores(
        signals.assign(_blend=blend_scores), "_blend"
    )

    # --- Stage 4, arm 3: the per-query selector -----------------------------
    train_signals = load_signals("fold0", args.features_dir)
    train_ws, train_matrix = rebuild_windows(
        "fold0", features_dir=args.features_dir, seed=args.seed
    )
    train_qrels = qrels_from_frame(train_matrix)
    train_orderings = single_stage_orderings(train_signals)
    train_per_arm = {
        _STAGE_OF_ARM[arm]: per_query_ndcg(train_orderings[arm], train_ws, train_qrels)
        for arm in SINGLE_STAGES
    }
    selector = train_selector(train_signals, train_per_arm, seed=args.seed)
    orderings["blend_selector"] = selector_ordering(
        selector,
        signals,
        {_STAGE_OF_ARM[arm]: orderings[arm] for arm in SINGLE_STAGES},
    )

    # --- the ceiling ---------------------------------------------------------
    orderings[ORACLE] = oracle_ordering(
        {_STAGE_OF_ARM[arm]: per_query[arm] for arm in SINGLE_STAGES},
        {_STAGE_OF_ARM[arm]: orderings[arm] for arm in SINGLE_STAGES},
    )

    for arm in (*BLEND_ARMS, ORACLE):
        per_query[arm] = per_query_ndcg(orderings[arm], ws, qrels)

    results = [
        evaluate_arm(
            arm, per_query[arm], floor.per_query, groups=("blend",),
            n_features=0, objective="blend", best_iteration=0, seed=args.seed,
        )
        for arm in (*SINGLE_STAGES, *BLEND_ARMS, ORACLE)
    ]
    check_same_queries(results)
    for arm in results:
        print(f"  {arm.name:24s} {arm.ndcg.point:.4f}")
    print()
    print(format_table(results))

    # §4.4's actual question: does Stage 4 beat the best stage it contains?
    baseline = best_single(results)
    by_name = {arm.name: arm for arm in results}
    against_best = [
        compare(by_name[arm], baseline, seed=args.seed)
        for arm in (*BLEND_ARMS, ORACLE)
    ]
    print(f"\n  -- Stage 4 against the best single stage ({baseline.name}) --")
    for row in against_best:
        d = row["delta"]
        verdict = "significant" if row["significant"] else "ties"
        print(f"  {row['arm']:24s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]  {verdict}")

    against_stage2 = [
        compare(by_name[arm], by_name["stage2"], seed=args.seed)
        for arm in (*SINGLE_STAGES[1:], *BLEND_ARMS, ORACLE)
    ]

    wins = [r for r in against_best if r["arm"] in BLEND_ARMS
            and r["significant"] and r["delta"]["point"] > 0]
    if not wins:
        print(
            f"\n  No Stage 4 strategy beats {baseline.name}. The oracle reaches "
            f"{by_name[ORACLE].ndcg.point:.4f}, so the headroom is real and "
            "these strategies do not reach it. Reported as measured."
        )

    payload = {
        "scope": args.scope,
        "n_queries": len(qrels),
        "n_judgements": len(matrix),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "esci_baseline_target": ESCI_BASELINE,
        "combiner_fitted_on": fitted_on,
        "fixed_weights": dict(DEFAULT_WEIGHTS),
        "best_single_stage": baseline.name,
        "arms": [arm.to_dict() for arm in results],
        "against_best_single": against_best,
        "against_stage2": against_stage2,
        "any_blend_beats_best_single": bool(wins),
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_blend_report.py -q`
Expected: PASS, 12 tests.

- [ ] **Step 5: Produce the fold-0 table**

Run: `python -m src.blend_report --scope fold0`

Expected: `stage2` 0.8519, `stage2+ce` 0.8587, `stage2+llm` 0.8814,
`blend_fixed` ≈ 0.8800, `blend_combiner` ≈ 0.8793, the oracle 0.9076, and the
selector unknown. The "No Stage 4 strategy beats stage2+llm" line is the
expected outcome for at least the first two.

Sanity checks: if `blend_fixed` equals `stage2+llm` exactly, the weights
collapsed onto one arm. If the oracle is not the highest row, `oracle_ordering`
is being given the wrong per-query scores. If `blend_combiner` on fold 0 beats
the oracle, the cross-fit is leaking and Task 2's guard failed.

- [ ] **Step 6: Run test exactly once**

Nothing about the configuration changes after Step 5.

Run: `python -m src.blend_report --scope test-sample --final --out docs/results/blend-test.json`

Expected: the same six arms plus the oracle over the frozen 2,000 queries,
`stage2+llm` near 0.8855, and every arm's `n_queries` equal to 2,000.

- [ ] **Step 7: Write the data-marked test**

Append to `tests/test_blend_report.py`:

```python
import json


@pytest.mark.data
def test_the_committed_blend_reports_answer_the_spec_question():
    for path in ("docs/results/blend.json", "docs/results/blend-test.json"):
        payload = json.loads(open(path).read())
        names = {arm["name"] for arm in payload["arms"]}
        # §4.4: the blend is compared against each single stage alone.
        assert set(SINGLE_STAGES) <= names
        assert set(BLEND_ARMS) <= names
        assert ORACLE in names

        # Every arm on the same queries, and the n recorded beside it.
        assert len({arm["n_queries"] for arm in payload["arms"]}) == 1

        # The comparison that matters is against the best single stage.
        assert payload["best_single_stage"] in SINGLE_STAGES
        assert {row["baseline"] for row in payload["against_best_single"]} == {
            payload["best_single_stage"]
        }

        # The ceiling must bound the arms it was built from.
        by_name = {arm["name"]: arm["ndcg"]["point"] for arm in payload["arms"]}
        for arm in SINGLE_STAGES:
            assert by_name[ORACLE] >= by_name[arm] - 1e-9

        # The floor is computed, never the published 0.7483.
        assert 0.73 < payload["floor"]["mean"] < 0.76


@pytest.mark.data
def test_the_fold_0_learned_arms_are_cross_fitted():
    # Fold 0 is both the only training set and the reporting surface, so an
    # in-sample fold-0 number would be a training fit wearing a result's
    # clothes.
    payload = json.loads(open("docs/results/blend.json").read())
    assert "cross-fit" in payload["combiner_fitted_on"]
```

Run: `python -m pytest tests/test_blend_report.py -m data -q`
Expected: PASS.

- [ ] **Step 8: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 9: Commit**

```bash
git add src/blend_report.py tests/test_blend_report.py docs/results/blend.json docs/results/blend-test.json
git commit -m "Report Stage 4 against every single stage with an oracle ceiling"
```

---

## Phase 2 Gate

Phase 3 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including both committed blend reports.
- [ ] `docs/results/{blend,blend-test}.json` exist and carry all six arms plus the oracle, every one on the same query set with `n` recorded.
- [ ] Every blend arm is compared against the **best single stage**, not only against `stage2`.
- [ ] The fold-0 learned arms are cross-fitted, and the JSON says so.
- [ ] The oracle bounds every single-stage arm in both files.
- [ ] Exactly one test-sample run exists, behind `--final`.
- [ ] If no blend arm beats the best single stage, `any_blend_beats_best_single` is `false` and the console said so in words.

Then: [Phase 3 — The Analysis and the Writeup](phase-3-the-analysis-and-the-writeup.md).
