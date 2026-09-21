# Phase 3 — The Ablations

**Plan 5 of 7 · Phase 3 of 3 · Tasks 5–6.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 2 gate](phase-2-the-matrix-and-the-ranker.md#phase-2-gate) passes.

**Delivers:** `src/fixed_weight.py` and `src/rank_report.py` — Ablation 7's
control arm, and the four ablations this plan owns reported with bootstrap CIs
against the measured random floor.

**Needs on disk:** `data/features/{train,test}.parquet` from Phase 2.

**Owns Review Focus item 4** (Ablation 4 reported as two arms).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Ablation 4 needs three arms.** ESCI-S presence alone ranks at +0.0084; reporting arm 2 − arm 1 credits the behavioural features with an artefact.
- **Every reported NDCG comes from `src.metrics`, paired with the computed floor.**
- **Never tune on test.** Every weight, round count and feature set is settled on folds 1–4; test is predicted once, behind an explicit flag.
- **A method that ties is a reportable result.** Do not tune until an arm wins.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 5: The fixed-weight fusion control

**Files:**
- Create: `src/fixed_weight.py`
- Create: `tests/test_fixed_weight.py`

**Interfaces:**
- Consumes: `src.metrics.ndcg_per_query` (Plan 1).
- Produces:
  - `src.fixed_weight.DEFAULT_WEIGHTS: tuple[float, ...]` (0.0 to 1.0 in steps of 0.05)
  - `src.fixed_weight.Normaliser` (frozen dataclass: `mean`, `std`)
  - `src.fixed_weight.fit_normaliser(values: np.ndarray) -> Normaliser`
  - `src.fixed_weight.apply_normaliser(norm, values) -> np.ndarray`
  - `src.fixed_weight.blend(text, image, w) -> np.ndarray`
  - `src.fixed_weight.sweep(frame, qrels, *, text_column, image_column, normalisers, weights=DEFAULT_WEIGHTS) -> list[tuple[float, float]]`
  - `src.fixed_weight.best_weight(sweep_result) -> tuple[float, float]`
  - `src.fixed_weight.per_query_scores(frame, qrels, *, text_column, image_column, normalisers, weight) -> dict[str, float]`

This is the arm `PROJECT_SPEC.md` §6 Ablation 7 measures against: *"a single
fixed global text/image weight"*, which is what Al Ghossein et al. used and
explicitly listed learned combination as future work beyond. It is not a
strawman — measured on fold 0 it reaches **0.8347**, within 0.022 of the
`ESCI_baseline` target the whole project is aiming at.

**Three design decisions, each of which makes the control stronger rather than
weaker.** A control that is quietly handicapped turns Ablation 7 into a
formality.

1. **The two similarities are standardised before blending.** `dense_sim` and
   `clip_image_sim` are both cosines but not on comparable scales, and a single
   weight over raw values is really weighting two different units. Mean and
   standard deviation are fitted on the **training folds** and applied to the
   reporting fold — fitting them on the fold being reported would let the
   control peek at its own evaluation set.
2. **Where the image is absent, the score falls back to text alone**, rather
   than substituting the mean. 20.6% of judged pairs have no image vector;
   pulling them toward the average would penalise a product for a 2022 scrape
   failure, which is exactly what `CLAUDE.md` forbids the *learned* arm from
   doing. The control gets the same courtesy.
3. **The weight is swept on fold 1 and reported on fold 0.** The 0.8347 in the
   README came from sweeping on fold 0 itself, which is an *oracle* value — the
   best any fixed weight could do on the surface being reported. The honest
   control tunes on fold 1 and will therefore land at or just below it. Report
   both, and treat the oracle 0.8347 as the conservative bar: an Ablation 7
   result that beats the oracle is unambiguous.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fixed_weight.py`:

```python
import math

import numpy as np
import pandas as pd
import pytest

from src.fixed_weight import (
    DEFAULT_WEIGHTS,
    apply_normaliser,
    best_weight,
    blend,
    fit_normaliser,
    sweep,
)


# --- normalisation ----------------------------------------------------------

def test_fit_normaliser_ignores_nan():
    # 20.6% of image similarities are NaN. Counting them as zeros would drag
    # the mean toward zero and mis-scale every real value.
    norm = fit_normaliser(np.array([1.0, 3.0, np.nan]))
    assert norm.mean == pytest.approx(2.0)
    assert norm.std == pytest.approx(1.0)


def test_apply_normaliser_centres_and_scales():
    norm = fit_normaliser(np.array([1.0, 3.0]))
    out = apply_normaliser(norm, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])


def test_apply_normaliser_keeps_nan_as_nan():
    norm = fit_normaliser(np.array([1.0, 3.0]))
    assert math.isnan(apply_normaliser(norm, np.array([np.nan]))[0])


def test_a_constant_column_does_not_divide_by_zero():
    norm = fit_normaliser(np.array([5.0, 5.0, 5.0]))
    assert np.isfinite(apply_normaliser(norm, np.array([5.0]))).all()


def test_fitting_on_nothing_raises():
    with pytest.raises(ValueError, match="no finite values"):
        fit_normaliser(np.array([np.nan, np.nan]))


# --- blending ---------------------------------------------------------------

def test_blend_is_the_weighted_sum():
    out = blend(np.array([1.0]), np.array([3.0]), 0.5)
    assert out[0] == pytest.approx(2.0)


def test_a_weight_of_one_is_text_only():
    np.testing.assert_allclose(blend(np.array([1.0, 2.0]), np.array([9.0, 9.0]), 1.0), [1.0, 2.0])


def test_a_weight_of_zero_is_image_only():
    np.testing.assert_allclose(blend(np.array([9.0, 9.0]), np.array([1.0, 2.0]), 0.0), [1.0, 2.0])


def test_a_missing_image_falls_back_to_text_alone():
    # Not to the mean: substituting the average would penalise a product for a
    # 2022 scrape failure, which is what CLAUDE.md forbids the learned arm from
    # doing. The control gets the same courtesy.
    out = blend(np.array([2.0, 2.0]), np.array([np.nan, 0.0]), 0.5)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(1.0)


def test_blend_never_returns_nan_when_the_text_score_is_present():
    out = blend(np.array([1.0, 2.0, 3.0]), np.full(3, np.nan), 0.3)
    assert np.isfinite(out).all()


def test_a_weight_outside_zero_to_one_raises():
    with pytest.raises(ValueError, match="between 0 and 1"):
        blend(np.array([1.0]), np.array([1.0]), 1.5)


# --- the sweep --------------------------------------------------------------

def _frame_and_qrels():
    # Two queries of three products. The image signal alone gets query 1 right
    # and query 2 wrong; the text signal is the other way round. A blend beats
    # either.
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 2],
            "product_id": ["a", "b", "c", "d", "e", "f"],
            "dense_sim": [0.1, 0.2, 0.9, 0.9, 0.2, 0.1],
            "clip_image_sim": [0.9, 0.2, 0.1, 0.1, 0.2, 0.9],
            "qrel": [100, 0, 0, 100, 0, 0],
        }
    )
    qrels = {
        "1": {"a": 100, "b": 0, "c": 0},
        "2": {"d": 100, "e": 0, "f": 0},
    }
    return frame, qrels


def test_sweep_returns_one_ndcg_per_weight():
    frame, qrels = _frame_and_qrels()
    norms = {
        "dense_sim": fit_normaliser(frame["dense_sim"].to_numpy()),
        "clip_image_sim": fit_normaliser(frame["clip_image_sim"].to_numpy()),
    }
    result = sweep(frame, qrels, text_column="dense_sim",
                   image_column="clip_image_sim", normalisers=norms)
    assert len(result) == len(DEFAULT_WEIGHTS)
    assert [w for w, _ in result] == list(DEFAULT_WEIGHTS)
    assert all(0.0 <= score <= 1.0 for _, score in result)


def test_best_weight_picks_the_highest_ndcg():
    assert best_weight([(0.0, 0.70), (0.5, 0.90), (1.0, 0.80)]) == (0.5, 0.90)


def test_best_weight_breaks_a_tie_toward_the_lower_weight():
    # Deterministic, so two runs of the same sweep report the same control.
    assert best_weight([(0.3, 0.9), (0.7, 0.9)]) == (0.3, 0.9)


def test_best_weight_of_an_empty_sweep_raises():
    with pytest.raises(ValueError, match="empty sweep"):
        best_weight([])


def test_the_default_sweep_covers_both_extremes():
    assert DEFAULT_WEIGHTS[0] == 0.0
    assert DEFAULT_WEIGHTS[-1] == 1.0
    assert len(set(DEFAULT_WEIGHTS)) == len(DEFAULT_WEIGHTS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_fixed_weight.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.fixed_weight'`.

- [ ] **Step 3: Write the implementation**

Create `src/fixed_weight.py`:

```python
"""Ablation 7's control: one hand-tuned global text/image weight.

PROJECT_SPEC.md §5 records the published best zero-shot text+image combination
as "a single hand-tuned global weight", and §6's Ablation 7 asks whether a
learned combination beats it. Al Ghossein et al. listed learned combination as
out of scope; this module builds the thing they did, so the comparison is
against their method rather than against a strawman.

Measured on fold 0, sweeping the weight on fold 0 itself: 0.8094 at pure
image, 0.8285 at pure text, peaking at **0.8347** with w_text = 0.7. That is an
*oracle* value - the best any fixed weight could do on the surface being
reported. `sweep` is therefore run on fold 1 and the chosen weight applied to
fold 0, which lands at or just below the oracle.

Three decisions make the control stronger, not weaker. A quietly handicapped
control turns the ablation into a formality:

  * **Standardise before blending.** Both columns are cosines but not on
    comparable scales, so one weight over raw values weights two different
    units. Mean and sd are fitted on the training folds, never on the fold
    being reported.
  * **Fall back to text where the image is absent**, rather than substituting
    the mean. 20.6% of judged pairs have no image vector, and pulling them
    toward the average would penalise a product for a 2022 scrape failure -
    the thing CLAUDE.md forbids the learned arm from doing.
  * **Sweep finely.** 21 weights at 0.05, not 3 at 0.25.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import Qrels, ndcg_per_query

DEFAULT_WEIGHTS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))


@dataclass(frozen=True)
class Normaliser:
    mean: float
    std: float


def fit_normaliser(values: np.ndarray) -> Normaliser:
    """Mean and standard deviation over the finite values only.

    NaN is an absence, not a zero: counting 20.6% of image similarities as
    zeros would drag the mean and mis-scale every real value.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot fit a normaliser: the column has no finite values")
    std = float(finite.std())
    return Normaliser(mean=float(finite.mean()), std=std if std > 1e-12 else 1.0)


def apply_normaliser(normaliser: Normaliser, values: np.ndarray) -> np.ndarray:
    """Centre and scale, leaving NaN as NaN."""
    return (np.asarray(values, dtype=np.float64) - normaliser.mean) / normaliser.std


def blend(text: np.ndarray, image: np.ndarray, weight: float) -> np.ndarray:
    """`w * text + (1 - w) * image`, falling back to text where image is NaN."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight}")
    text = np.asarray(text, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    mixed = weight * text + (1.0 - weight) * image
    return np.where(np.isfinite(image), mixed, text)


def _run(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for query_id, product_id, score in zip(
        frame["query_id"], frame["product_id"], scores
    ):
        # A pair with no text score at all still has to occupy a rank; -inf
        # sorts it last without inventing a similarity for it.
        run.setdefault(str(query_id), {})[str(product_id)] = (
            float(score) if np.isfinite(score) else float("-inf")
        )
    return run


def sweep(
    frame: pd.DataFrame,
    qrels: Qrels,
    *,
    text_column: str,
    image_column: str,
    normalisers: Mapping[str, Normaliser],
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> list[tuple[float, float]]:
    """Mean NDCG at each weight, using normalisers fitted elsewhere."""
    text = apply_normaliser(normalisers[text_column], frame[text_column].to_numpy())
    image = apply_normaliser(normalisers[image_column], frame[image_column].to_numpy())

    out: list[tuple[float, float]] = []
    for weight in weights:
        per_query = ndcg_per_query(_run(frame, blend(text, image, weight)), qrels)
        out.append((float(weight), sum(per_query.values()) / len(per_query)))
    return out


def best_weight(result: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The best (weight, ndcg). Ties break toward the lower weight, for determinism."""
    if not result:
        raise ValueError("cannot pick a weight from an empty sweep")
    return max(result, key=lambda item: (item[1], -item[0]))


def per_query_scores(
    frame: pd.DataFrame,
    qrels: Qrels,
    *,
    text_column: str,
    image_column: str,
    normalisers: Mapping[str, Normaliser],
    weight: float,
) -> dict[str, float]:
    """Per-query NDCG at one weight, for a paired comparison against a model."""
    text = apply_normaliser(normalisers[text_column], frame[text_column].to_numpy())
    image = apply_normaliser(normalisers[image_column], frame[image_column].to_numpy())
    return ndcg_per_query(_run(frame, blend(text, image, weight)), qrels)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_fixed_weight.py -q`
Expected: PASS, 17 tests.

- [ ] **Step 5: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 6: Commit**

```bash
git add src/fixed_weight.py tests/test_fixed_weight.py
git commit -m "Add the fixed global text-image weight control for Ablation 7"
```

---

## Task 6: The ablation report

**Files:**
- Create: `src/rank_report.py`
- Create: `tests/test_rank_report.py`
- Modify: `CLAUDE.md` (Commands section)
- Create: `docs/results/coarse-rank.json` (written by the run)

**Interfaces:**
- Consumes: everything above, plus `src.bootstrap.{bootstrap_ci, paired_delta_ci}`, `src.floor.random_floor`, `src.metrics.ndcg_per_query`.
- Produces:
  - `src.rank_report.ARMS: dict[str, tuple[str, ...]]`
  - `src.rank_report.FUSION_FEATURES: tuple[str, ...]` = `("dense_sim", "clip_image_sim", "has_image_vector")`
  - `src.rank_report.ArmResult` (frozen dataclass)
  - `src.rank_report.qrels_from_frame(frame) -> dict[str, dict[str, int]]`
  - `src.rank_report.evaluate_arm(name, per_query, floor_per_query, *, groups, n_features, objective, best_iteration, seed=0) -> ArmResult`
  - `src.rank_report.compare(arm, baseline) -> dict`
  - `src.rank_report.honest_behavioural_delta(values, indicators, text, floor) -> dict`
  - `src.rank_report.check_ablation_4(arm_names) -> None`
  - `src.rank_report.format_table(arms) -> str`
  - CLI: `python -m src.rank_report` (fold 0) and `python -m src.rank_report --split test --final`

### The arms

| arm | groups | used by |
|---|---|---|
| `text` | `text`, `esci_indicators`, `retrieval` | Ablations 3 and 4 |
| `text+image` | ` + image` | Ablation 3 |
| `text+behavioural` | ` + behavioural`, `categorical`, `attrs`, `s_indicators` | Ablation 4 |
| `indicators_only` | `s_indicators` | Ablation 4 arm 3 |
| `esci_presence_only` | `esci_indicators` | cross-check against `CLAUDE.md` |
| `both_presence_only` | `esci_indicators`, `s_indicators` | cross-check against `CLAUDE.md` |
| `full` | all eight | Ablation 5, and the headline |

### Review Focus 4: Ablation 4 arithmetic

`CLAUDE.md` states the rule twice and the two statements are not quite the same
arithmetic. *"the honest contribution is arm 2 − arm 3"* reads literally as one
NDCG minus another, which would be roughly +0.09 — the behavioural arm is a
real ranker and the indicators arm is barely above the floor. But the
neighbouring sentence, *"the ~+0.0084 must be subtracted from any
behavioural-feature result"*, describes the coherent version:

```
naive behavioural gain = arm2 − arm1                     (text+behavioural vs text)
artefact contribution  = arm3 − floor                    (indicators alone, over random)
honest gain            = (arm2 − arm1) − (arm3 − floor)
```

The code implements the coherent version and reports all three numbers, so a
reader can see the shorthand's arithmetic as well. The interval is a proper
paired bootstrap: per query, `d_q = (a2_q − a1_q) − (a3_q − floor_q)`, which is
`paired_delta_ci` over `{q: a2_q + floor_q}` against `{q: a1_q + a3_q}`.

**State the assumption in the writeup.** `CLAUDE.md` also measured that the
presence effects are *super-additive* (+0.0166 for both presence sets together,
against +0.0052 + +0.0084 apart), so subtracting the artefact assumes an
additivity that does not strictly hold. The subtraction is a point estimate
under that assumption, not a bound. Say so; do not launder it into a clean
number.

`check_ablation_4` refuses to emit the ablation at all when the
`indicators_only` arm is missing from the results — the failure mode is not
computing it wrong, it is quietly not computing it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rank_report.py`:

```python
import json

import numpy as np
import pandas as pd
import pytest

from src.feature_matrix import FEATURE_GROUPS
from src.rank_report import (
    ARMS,
    FUSION_FEATURES,
    ArmResult,
    check_ablation_4,
    compare,
    evaluate_arm,
    format_table,
    honest_behavioural_delta,
    qrels_from_frame,
)


def _per_query(values):
    return {str(i): v for i, v in enumerate(values)}


def _arm(name, values, floor=None):
    floor = floor or [0.74] * len(values)
    return evaluate_arm(
        name,
        _per_query(values),
        _per_query(floor),
        groups=("text",),
        n_features=3,
        objective="lambdarank",
        best_iteration=100,
    )


# --- the arms ---------------------------------------------------------------

def test_every_arm_names_real_feature_groups():
    for name, groups in ARMS.items():
        for group in groups:
            assert group in FEATURE_GROUPS, f"{name} -> {group}"


def test_the_text_arm_carries_no_esci_s_group():
    # "Text only" must not quietly include the enrichment.
    forbidden = {"behavioural", "categorical", "attrs", "s_indicators", "image"}
    assert set(ARMS["text"]).isdisjoint(forbidden)


def test_the_image_arm_is_the_text_arm_plus_image():
    assert set(ARMS["text+image"]) == set(ARMS["text"]) | {"image"}


def test_the_full_arm_uses_every_group():
    assert set(ARMS["full"]) == set(FEATURE_GROUPS)


def test_the_fusion_arm_uses_only_the_two_fused_signals():
    # Handing LambdaMART 47 features and calling the difference "learned
    # fusion" would measure the other 45.
    assert FUSION_FEATURES == ("dense_sim", "clip_image_sim", "has_image_vector")


# --- qrels ------------------------------------------------------------------

def test_qrels_come_from_the_frames_own_qrel_column():
    frame = pd.DataFrame(
        {"query_id": [1, 1], "product_id": ["a", "b"], "qrel": [100, 0]}
    )
    assert qrels_from_frame(frame) == {"1": {"a": 100, "b": 0}}


def test_duplicate_pairs_in_the_qrels_raise():
    frame = pd.DataFrame(
        {"query_id": [1, 1], "product_id": ["a", "a"], "qrel": [100, 0]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        qrels_from_frame(frame)


# --- arms and comparisons ---------------------------------------------------

def test_an_arm_carries_its_interval_and_its_lift_over_the_floor():
    arm = _arm("text", [0.80, 0.82, 0.84])
    assert arm.ndcg.point == pytest.approx(0.82)
    assert arm.lift_over_floor.point == pytest.approx(0.08)
    assert arm.ndcg.low <= arm.ndcg.point <= arm.ndcg.high


def test_comparison_is_paired_by_query():
    better = _arm("b", [0.90, 0.70])
    worse = _arm("a", [0.80, 0.60])
    row = compare(better, worse)
    assert row["delta"]["point"] == pytest.approx(0.10)
    assert row["significant"] is True


def test_a_tie_is_reported_as_a_tie_not_a_win():
    # A method that ties the baseline is a legitimate, reportable result.
    a = _arm("a", [0.80, 0.70, 0.90, 0.60])
    b = _arm("b", [0.81, 0.69, 0.89, 0.61])
    assert compare(b, a)["significant"] is False


def test_comparing_arms_with_different_queries_raises():
    a = evaluate_arm("a", {"1": 0.8}, {"1": 0.74}, groups=("text",), n_features=1,
                     objective="lambdarank", best_iteration=1)
    b = evaluate_arm("b", {"2": 0.8}, {"2": 0.74}, groups=("text",), n_features=1,
                     objective="lambdarank", best_iteration=1)
    with pytest.raises(ValueError, match="same queries"):
        compare(a, b)


# --- Review Focus 4 ---------------------------------------------------------

def test_ablation_4_refuses_to_report_without_the_indicators_arm():
    # The failure mode is not computing it wrong, it is quietly not computing
    # it - and then crediting the behavioural features with a scrape artefact
    # worth roughly twice Plan 4's entire image contribution.
    with pytest.raises(ValueError, match="indicators_only"):
        check_ablation_4(["text", "text+behavioural"])


def test_ablation_4_accepts_all_three_arms():
    check_ablation_4(["text", "text+behavioural", "indicators_only"])  # must not raise


def test_the_honest_delta_subtracts_the_artefact_lift():
    #   naive  = arm2 - arm1           = 0.86 - 0.84 = 0.02
    #   artefact = arm3 - floor        = 0.75 - 0.74 = 0.01
    #   honest = naive - artefact      = 0.01
    values = _per_query([0.86, 0.86])
    text = _per_query([0.84, 0.84])
    indicators = _per_query([0.75, 0.75])
    floor = _per_query([0.74, 0.74])

    out = honest_behavioural_delta(values, indicators, text, floor)
    assert out["naive_delta"]["point"] == pytest.approx(0.02)
    assert out["artefact_lift"]["point"] == pytest.approx(0.01)
    assert out["honest_delta"]["point"] == pytest.approx(0.01)


def test_the_honest_delta_records_the_additivity_caveat():
    # CLAUDE.md measured the presence effects as super-additive, so the
    # subtraction is a point estimate under an assumption, not a bound.
    out = honest_behavioural_delta(
        _per_query([0.86]), _per_query([0.75]), _per_query([0.84]), _per_query([0.74])
    )
    assert "super-additive" in out["caveat"]


def test_the_honest_delta_can_be_negative():
    # If the behavioural features buy less than the artefact alone, the honest
    # contribution is negative and must be reported as such.
    out = honest_behavioural_delta(
        _per_query([0.845, 0.845]),
        _per_query([0.78, 0.78]),
        _per_query([0.84, 0.84]),
        _per_query([0.74, 0.74]),
    )
    assert out["honest_delta"]["point"] < 0


# --- formatting and serialisation ------------------------------------------

def test_the_table_carries_the_floor_on_every_row():
    # PROJECT_SPEC.md §2: a headline NDCG means nothing without the floor.
    table = format_table([_arm("text", [0.82]), _arm("full", [0.86])])
    assert "floor" in table.lower()
    assert "0.82" in table and "0.86" in table


def test_an_arm_serialises_to_json():
    payload = _arm("text", [0.82]).to_dict()
    assert json.loads(json.dumps(payload))["name"] == "text"
    assert payload["groups"] == ["text"]
    assert payload["objective"] == "lambdarank"
    assert payload["best_iteration"] == 100
    assert set(payload["ndcg"]) == {"point", "low", "high"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rank_report.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.rank_report'`.

- [ ] **Step 3: Write the implementation**

Create `src/rank_report.py`:

```python
"""Ablations 3, 4, 5 and 7: the table Plan 5 exists to produce.

  3 - text-only vs. text+image features          (the multimodal contribution)
  4 - with vs. without behavioural features       (three arms, not two)
  5 - pointwise vs. lambdarank                    (the LTR objective's value)
  7 - learned fusion vs. one fixed global weight  (the learned-combination gap)

Every arm carries the computed random floor beside it, because
PROJECT_SPEC.md §2 is explicit that a headline NDCG means nothing without one.
Every comparison is a paired bootstrap over queries via src.bootstrap, which
pairs on the query id rather than on position.

Every number is scored by src.metrics. None comes from LightGBM: its ndcg@k
differs from this project's full-list NDCG by up to 6.4 points on the same
booster, for two independent reasons - see src/ranker.py.

**Ablation 4's arithmetic.** CLAUDE.md states the rule twice and the two are
not the same sum. "the honest contribution is arm 2 - arm 3" reads literally
as one NDCG minus another, which is roughly +0.09: arm 2 is a real ranker and
arm 3 is barely above the floor. The neighbouring sentence - "the ~+0.0084
must be subtracted from any behavioural-feature result" - describes the
coherent version, which is what `honest_behavioural_delta` computes:

    naive     = arm2 - arm1            text+behavioural vs text
    artefact  = arm3 - floor           the ESCI-S presence pattern alone
    honest    = naive - artefact

All three are reported. The subtraction assumes the effects are additive, and
CLAUDE.md measured them as super-additive (+0.0166 together against +0.0052 +
+0.0084 apart), so the honest figure is a point estimate under an assumption
rather than a bound. The caveat travels with the number in the JSON.

Selection happens on folds 1-4. `--split test` exists and requires `--final`,
because running it before the configuration has settled is exactly the thing
CLAUDE.md's evaluation discipline forbids.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.bootstrap import Interval, bootstrap_ci, paired_delta_ci
from src.feature_matrix import FEATURE_GROUPS

DEFAULT_OUT = Path("docs/results/coarse-rank.json")

# PROJECT_SPEC.md §5: the cross-encoder fine-tuned on SQD train. The project's
# stated realistic goal.
ESCI_BASELINE = 0.8562

ARMS: dict[str, tuple[str, ...]] = {
    # Ablations 3 and 4's arm 1. ESCI's own text and presence, plus the
    # retrieval scores. No ESCI-S values, no ESCI-S indicators, no image.
    "text": ("text", "esci_indicators", "retrieval"),
    # Ablation 3's treatment arm. The indicator travels with the score.
    "text+image": ("text", "esci_indicators", "retrieval", "image"),
    # Ablation 4's arm 2: values AND their missingness indicators.
    "text+behavioural": (
        "text",
        "esci_indicators",
        "retrieval",
        "behavioural",
        "categorical",
        "attrs",
        "s_indicators",
    ),
    # Ablation 4's arm 3. The artefact on its own.
    "indicators_only": ("s_indicators",),
    # Not an ablation arm - a cross-check against the presence-flag ceilings
    # CLAUDE.md records at +0.0052, +0.0084 and +0.0166 over the floor.
    "esci_presence_only": ("esci_indicators",),
    "both_presence_only": ("esci_indicators", "s_indicators"),
    # The headline, and Ablation 5's feature set.
    "full": tuple(FEATURE_GROUPS),
}

# Ablation 7's learned arm fuses exactly what the fixed weight fuses. Handing
# LambdaMART all 47 features and calling the difference "learned fusion" would
# measure the other 45.
FUSION_FEATURES: tuple[str, ...] = ("dense_sim", "clip_image_sim", "has_image_vector")

ADDITIVITY_CAVEAT = (
    "Subtracting the indicators-only arm assumes the presence effects are "
    "additive. CLAUDE.md measured them as super-additive (+0.0166 for both "
    "presence sets together against +0.0052 + +0.0084 apart), so this is a "
    "point estimate under that assumption, not a bound."
)


def _interval(interval: Interval) -> dict:
    return {"point": interval.point, "low": interval.low, "high": interval.high}


@dataclass(frozen=True)
class ArmResult:
    name: str
    groups: tuple[str, ...]
    n_features: int
    objective: str
    best_iteration: int
    n_queries: int
    ndcg: Interval
    lift_over_floor: Interval
    per_query: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "groups": list(self.groups),
            "n_features": self.n_features,
            "objective": self.objective,
            "best_iteration": self.best_iteration,
            "n_queries": self.n_queries,
            "ndcg": _interval(self.ndcg),
            "lift_over_floor": _interval(self.lift_over_floor),
        }


def qrels_from_frame(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Nested qrels from a feature matrix's own `qrel` column."""
    duplicated = frame.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = frame.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}); building a "
            "dict would silently keep only the last one"
        )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, product_id, qrel in zip(
        frame["query_id"], frame["product_id"], frame["qrel"]
    ):
        qrels.setdefault(str(query_id), {})[str(product_id)] = int(qrel)
    return qrels


def evaluate_arm(
    name: str,
    per_query: Mapping[str, float],
    floor_per_query: Mapping[str, float],
    *,
    groups: Sequence[str],
    n_features: int,
    objective: str,
    best_iteration: int,
    seed: int = 0,
) -> ArmResult:
    """Wrap one arm's per-query NDCG with its interval and its lift."""
    return ArmResult(
        name=name,
        groups=tuple(groups),
        n_features=n_features,
        objective=objective,
        best_iteration=best_iteration,
        n_queries=len(per_query),
        ndcg=bootstrap_ci(per_query, seed=seed),
        lift_over_floor=paired_delta_ci(per_query, floor_per_query, seed=seed),
        per_query=dict(per_query),
    )


def compare(arm: ArmResult, baseline: ArmResult, seed: int = 0) -> dict:
    """Paired bootstrap delta of `arm` against `baseline`."""
    if set(arm.per_query) != set(baseline.per_query):
        raise ValueError(
            f"{arm.name} and {baseline.name} must cover the same queries; "
            f"{len(set(arm.per_query) - set(baseline.per_query))} differ"
        )
    delta = paired_delta_ci(arm.per_query, baseline.per_query, seed=seed)
    return {
        "arm": arm.name,
        "baseline": baseline.name,
        "delta": _interval(delta),
        "significant": not (delta.low <= 0.0 <= delta.high),
    }


def check_ablation_4(arm_names: Sequence[str]) -> None:
    """Refuse to report Ablation 4 without its third arm.

    The failure mode is not computing the subtraction wrong, it is quietly not
    computing it and crediting the behavioural features with an artefact worth
    roughly twice Plan 4's entire image contribution.
    """
    required = {"text", "text+behavioural", "indicators_only"}
    missing = required - set(arm_names)
    if missing:
        raise ValueError(
            f"Ablation 4 needs three arms and {sorted(missing)} are missing. "
            "ESCI-S presence alone ranks at +0.0084 over the floor and is not "
            "available at serving time, so it must be measured and subtracted, "
            "not assumed away."
        )


def honest_behavioural_delta(
    values: Mapping[str, float],
    indicators: Mapping[str, float],
    text: Mapping[str, float],
    floor: Mapping[str, float],
    *,
    seed: int = 0,
) -> dict:
    """Ablation 4's three numbers, each with a paired interval.

    honest = (arm2 - arm1) - (arm3 - floor), computed per query so the
    interval is a real paired bootstrap rather than two intervals subtracted.
    """
    naive = paired_delta_ci(values, text, seed=seed)
    artefact = paired_delta_ci(indicators, floor, seed=seed)
    # mean(a2 + floor) - mean(a1 + a3) == (a2 - a1) - (a3 - floor)
    left = {q: values[q] + floor[q] for q in values}
    right = {q: text[q] + indicators[q] for q in values}
    honest = paired_delta_ci(left, right, seed=seed)
    return {
        "naive_delta": _interval(naive),
        "artefact_lift": _interval(artefact),
        "honest_delta": _interval(honest),
        "significant": not (honest.low <= 0.0 <= honest.high),
        "caveat": ADDITIVITY_CAVEAT,
    }


def format_table(arms: Sequence[ArmResult]) -> str:
    """A markdown table that cannot be printed without the floor."""
    if not arms:
        return "(no arms)"
    lines = [
        "| arm | objective | features | NDCG | 95% CI | lift over floor |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        lift = arm.lift_over_floor
        lines.append(
            f"| {arm.name} | {arm.objective} | {arm.n_features} | "
            f"{arm.ndcg.point:.4f} | [{arm.ndcg.low:.4f}, {arm.ndcg.high:.4f}] | "
            f"{lift.point:+.4f} [{lift.low:+.4f}, {lift.high:+.4f}] |"
        )
    return "\n".join(lines)


def _main() -> int:
    from src.feature_matrix import select_columns
    from src.fixed_weight import (
        best_weight,
        fit_normaliser,
        per_query_scores,
        sweep,
    )
    from src.floor import random_floor
    from src.metrics import ndcg_per_query
    from src.ranker import (
        EARLY_STOP_FOLD,
        OBJECTIVES,
        REPORT_FOLD,
        TRAIN_FOLDS,
        folds,
        predict_run,
        train_ranker,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--final",
        action="store_true",
        help="required with --split test; the test split is predicted ONCE, "
        "after the configuration is frozen",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        raise SystemExit(
            "refusing to touch the test split without --final. Every weight, "
            "round count and feature set is chosen on folds 1-4; test is "
            "measured once, at the end."
        )

    train = pd.read_parquet(args.features_dir / "train.parquet")
    fit = folds(train, TRAIN_FOLDS)
    early = folds(train, [EARLY_STOP_FOLD])
    if args.split == "test":
        # The one test run: every train fold for gradient, the same early-stop
        # fold so the round count is comparable, test for reporting.
        fit = train
        report = pd.read_parquet(args.features_dir / "test.parquet")
    else:
        report = folds(train, [REPORT_FOLD])

    qrels = qrels_from_frame(report)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    print(f"{len(report):,} judgements over {len(qrels):,} queries; "
          f"random floor {floor.mean:.4f} [{floor.low:.4f}, {floor.high:.4f}]")

    results: dict[str, ArmResult] = {}

    def run_arm(name: str, columns: Sequence[str], objective: str,
                groups: Sequence[str]) -> ArmResult:
        ranker = train_ranker(fit, early, list(columns), objective=objective,
                              seed=args.seed)
        per_query = ndcg_per_query(predict_run(ranker, report), qrels)
        arm = evaluate_arm(
            name, per_query, floor.per_query, groups=groups,
            n_features=len(columns), objective=objective,
            best_iteration=ranker.best_iteration, seed=args.seed,
        )
        print(f"  {name:22s} {objective:22s} {arm.ndcg.point:.4f} "
              f"({ranker.best_iteration} rounds, {len(columns)} features)")
        return arm

    print("\ntraining arms:")
    for name, groups in ARMS.items():
        results[name] = run_arm(name, select_columns(groups), "lambdarank", groups)

    # --- Ablation 5: the objective -----------------------------------------
    for objective in OBJECTIVES:
        if objective == "lambdarank":
            continue
        results[f"full/{objective}"] = run_arm(
            f"full/{objective}", select_columns(ARMS["full"]), objective, ARMS["full"]
        )

    # --- Ablation 7: learned fusion vs. one global weight -------------------
    results["learned_fusion"] = run_arm(
        "learned_fusion", FUSION_FEATURES, "lambdarank", ("retrieval", "image")
    )
    normalisers = {
        column: fit_normaliser(fit[column].to_numpy())
        for column in ("dense_sim", "clip_image_sim")
    }
    # Fold 1 in both cases: the weight is a tuned parameter, so it is never
    # chosen on the surface being reported.
    tuning = early
    swept = sweep(
        tuning, qrels_from_frame(tuning), text_column="dense_sim",
        image_column="clip_image_sim", normalisers=normalisers,
    )
    chosen_weight, tuning_ndcg = best_weight(swept)
    fixed_per_query = per_query_scores(
        report, qrels, text_column="dense_sim", image_column="clip_image_sim",
        normalisers=normalisers, weight=chosen_weight,
    )
    results["fixed_weight"] = evaluate_arm(
        "fixed_weight", fixed_per_query, floor.per_query,
        groups=("retrieval", "image"), n_features=2, objective="fixed_weight",
        best_iteration=0, seed=args.seed,
    )
    print(f"  fixed_weight swept on fold {EARLY_STOP_FOLD}: w_text="
          f"{chosen_weight:.2f} ({tuning_ndcg:.4f} there), "
          f"{results['fixed_weight'].ndcg.point:.4f} here")

    ordered = list(results.values())
    print()
    print(format_table(ordered))

    check_ablation_4(list(results))
    ablations = {
        "3_text_vs_image": compare(results["text+image"], results["text"]),
        "4_behavioural": honest_behavioural_delta(
            results["text+behavioural"].per_query,
            results["indicators_only"].per_query,
            results["text"].per_query,
            floor.per_query,
            seed=args.seed,
        ),
        "5_objective": [
            compare(results[f"full/{o}"], results["full"])
            for o in OBJECTIVES
            if o != "lambdarank"
        ],
        "7_learned_fusion": compare(results["learned_fusion"], results["fixed_weight"])
        | {"fixed_weight_w_text": chosen_weight, "sweep": swept},
    }

    print("\n  -- Ablation 3: text+image vs text --")
    d = ablations["3_text_vs_image"]["delta"]
    print(f"  {d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  "
          f"{'significant' if ablations['3_text_vs_image']['significant'] else 'ties'}")

    print("  -- Ablation 4: behavioural, three arms --")
    a4 = ablations["4_behavioural"]
    for key in ("naive_delta", "artefact_lift", "honest_delta"):
        v = a4[key]
        print(f"  {key:16s} {v['point']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}]")

    print("  -- Ablation 5: objective, against lambdarank --")
    for row in ablations["5_objective"]:
        d = row["delta"]
        print(f"  {row['arm']:26s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]  "
              f"{'significant' if row['significant'] else 'ties'}")

    print("  -- Ablation 7: learned fusion vs one global weight --")
    d = ablations["7_learned_fusion"]["delta"]
    print(f"  {d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  "
          f"{'significant' if ablations['7_learned_fusion']['significant'] else 'ties'}")

    # The presence-flag cross-check against CLAUDE.md's recorded ceilings.
    print("\n  -- presence-only cross-check (CLAUDE.md records "
          "+0.0052 / +0.0084 / +0.0166 on test) --")
    for name in ("esci_presence_only", "indicators_only", "both_presence_only"):
        print(f"  {name:22s} {results[name].lift_over_floor.point:+.4f}")

    headline = results["full"]
    print(f"\nheadline: {headline.ndcg.point:.4f} "
          f"[{headline.ndcg.low:.4f}, {headline.ndcg.high:.4f}] against a floor "
          f"of {floor.mean:.4f} and the ESCI_baseline target of {ESCI_BASELINE}")

    payload = {
        "split": args.split,
        "report_fold": None if args.split == "test" else REPORT_FOLD,
        "train_folds": list(TRAIN_FOLDS) if args.split == "train" else "all",
        "early_stop_fold": EARLY_STOP_FOLD,
        "n_queries": len(qrels),
        "n_judgements": len(report),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "esci_baseline_target": ESCI_BASELINE,
        "arms": [arm.to_dict() for arm in ordered],
        "ablations": ablations,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_rank_report.py -q`
Expected: PASS, 18 tests.

- [ ] **Step 5: Produce the fold-0 ablation table**

Run: `python -m src.rank_report`

Expected: ten arms trained (a few minutes in total — 200-round `lambdarank` on
the real shape is 6.8 s), a markdown table, the four ablations, the
presence-only cross-check, and `docs/results/coarse-rank.json`.

Sanity checks against what was measured on 2026-09-21, all on fold 0:

| check | expectation |
|---|---|
| floor | ≈ 0.7440 |
| `text` arm | above `bm25_score` alone (0.8230) and `dense_sim` alone (0.8285) |
| `full` arm | above the fixed-weight oracle of 0.8347 |
| `indicators_only` | just above the floor, roughly +0.005 to +0.015 |
| `fixed_weight` | at or just below 0.8347, with `w_text` near 0.7 |

If `indicators_only` lands near 0.83, an ESCI-S *value* column has leaked into
the indicator group. If `text` lands near the floor, the retrieval scores are
not reaching the matrix.

- [ ] **Step 6: Write the data-marked test**

Append to `tests/test_rank_report.py`:

```python
@pytest.mark.data
def test_the_committed_report_has_every_ablation():
    payload = json.loads(open("docs/results/coarse-rank.json").read())

    names = {arm["name"] for arm in payload["arms"]}
    check_ablation_4(names)                     # Review Focus 4, on the artefact
    assert {"text", "text+image", "full", "fixed_weight", "learned_fusion"} <= names

    assert set(payload["ablations"]) == {
        "3_text_vs_image", "4_behavioural", "5_objective", "7_learned_fusion"
    }
    # Ablation 4 must carry all three numbers and the additivity caveat.
    a4 = payload["ablations"]["4_behavioural"]
    assert {"naive_delta", "artefact_lift", "honest_delta", "caveat"} <= set(a4)
    assert "super-additive" in a4["caveat"]

    # Every arm carries the floor it beats. PROJECT_SPEC.md §2.
    assert 0.70 < payload["floor"]["mean"] < 0.80
    for arm in payload["arms"]:
        assert set(arm["lift_over_floor"]) == {"point", "low", "high"}

    # The headline is a real re-ranker, not a floor-grazing one.
    full = next(a for a in payload["arms"] if a["name"] == "full")
    assert full["ndcg"]["point"] > 0.83
```

Run: `python -m pytest tests/test_rank_report.py -m data -q`
Expected: PASS.

- [ ] **Step 7: Freeze the configuration, then run test exactly once**

Before this step, stop and write down — in the commit message of step 9 or in
`docs/results/coarse-rank.json`'s own `seed` and fold fields — the feature
groups, the objective, the round count and the fixed weight. Nothing may change
after the next command runs.

Run: `python -m src.rank_report --split test --final --out docs/results/coarse-rank-test.json`

Expected: the test floor at ≈0.7467 (`CLAUDE.md`'s recorded value, and a direct
check that the harness still agrees with Plan 1), 8,956 queries, and a headline
NDCG reported against the `ESCI_baseline` of 0.8562. **Beating it and tying it
are both acceptable outcomes.** Do not go back and tune.

- [ ] **Step 8: Update `CLAUDE.md`'s Commands section**

Add to the fenced command block in `CLAUDE.md`, after the retrieval-channel
section:

```bash
# Coarse rank (Plan 5). Pair scores are ~2 min a split, dominated by model loading.
python -m src.pair_scores --split train        # -> data/features/pair-scores-train.parquet
python -m src.pair_scores --split test
python -m src.feature_matrix --split train     # -> data/features/train.parquet
python -m src.feature_matrix --split test
python -m src.rank_report                      # Ablations 3, 4, 5, 7 on fold 0
python -m src.rank_report --split test --final # the one test run
```

Also add a line to the **Data invariants** section, recording what this plan
measured:

```markdown
**The `rerank` image store is redundant.** Measured 2026-09-21: 0 of its
361,875 URLs are absent from `data/embeddings/catalogue/`, and both give the
same judged-product coverage (373,639 products, 77.50%). Plan 5 reads the
catalogue store only; deleting `data/embeddings/rerank/` recovers 0.40 GB
without losing a vector.
```

- [ ] **Step 9: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 10: Commit**

```bash
git add src/rank_report.py tests/test_rank_report.py CLAUDE.md docs/results/coarse-rank.json docs/results/coarse-rank-test.json
git commit -m "Report Ablations 3, 4, 5 and 7 for the LambdaMART coarse ranker"
```

- [ ] **Step 11: Mark the plan gate**

Tick the [Plan Gate](README.md#plan-gate) boxes in `README.md`, and update the
series entry for Plan 5 in `docs/superpowers/plans/README.md` with the landed
numbers, in the style of Plans 1–4: the headline NDCG against the floor and the
0.8562 target, and the four ablation outcomes including any that tie.

```bash
git add docs/superpowers/plans/
git commit -m "Mark the coarse rank plan gate as passed"
```

---

## Phase 3 Gate — and the Plan Gate

This phase's gate *is* the [Plan Gate](README.md#plan-gate). Check it there;
every box must be ticked before Plan 6 is written.

The three that are easiest to skip:

- [ ] Ablation 4 has **three** arms in the committed JSON, and `check_ablation_4` is what enforces it.
- [ ] Exactly **one** test-split number exists, produced after the configuration was frozen.
- [ ] No number in the table came from a LightGBM training log.
