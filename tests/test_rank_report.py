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


def test_the_clean_control_shares_the_text_arms_base():
    # Ablation 4's assumption-free reading needs an arm that differs from
    # text+behavioural ONLY by the ESCI-S value columns.
    assert set(ARMS["text+indicators"]) == set(ARMS["text"]) | {"s_indicators"}
    assert set(ARMS["text+behavioural"]) - set(ARMS["text+indicators"]) == {
        "behavioural", "categorical", "attrs"
    }


def test_values_over_indicators_subtracts_nothing():
    # (arm2 - text+indicators) is a plain paired delta between two real arms,
    # so it needs no additivity assumption at all.
    out = honest_behavioural_delta(
        _per_query([0.86, 0.86]),
        _per_query([0.75, 0.75]),
        _per_query([0.84, 0.84]),
        _per_query([0.74, 0.74]),
        text_indicators=_per_query([0.855, 0.855]),
    )
    assert out["values_over_indicators"]["point"] == pytest.approx(0.005)
    assert out["values_over_indicators_significant"] is True


def test_the_clean_control_is_optional():
    out = honest_behavioural_delta(
        _per_query([0.86]), _per_query([0.75]), _per_query([0.84]), _per_query([0.74])
    )
    assert "values_over_indicators" not in out


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
