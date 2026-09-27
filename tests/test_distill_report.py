import json
from pathlib import Path

import pytest

from src.distill import INITS, TARGETS
from src.distill_report import (
    FREE_ARMS,
    GATE_RULE,
    MODEL_ARMS,
    PILOT_EPOCHS,
    REFERENCE_ARMS,
    check_reference,
    check_split,
    paid_run_gate,
    pilot_arm,
    resolve_model_arms,
    share_of_teacher_gain,
    training_records,
)


def _row(arm, baseline, low, point=None, high=None):
    point = low + 0.001 if point is None else point
    high = point + 0.001 if high is None else high
    return {
        "arm": arm,
        "baseline": baseline,
        "delta": {"point": point, "low": low, "high": high},
        "significant": not (low <= 0.0 <= high),
    }


# --- the pre-registered gate ------------------------------------------------

def test_a_teacher_target_beating_the_labels_from_the_same_start_passes():
    gate = paid_run_gate([
        _row(pilot_arm("landed", "hybrid"), pilot_arm("landed", "labels"), low=0.0004),
    ])
    assert gate["passed"] is True
    assert gate["choice"] == {"init": "landed", "target": "hybrid"}
    assert gate["because"] == ["pilot:landed:hybrid"]


def test_a_tie_does_not_pass():
    # Both pilots measured before this plan was written look like this.
    gate = paid_run_gate([
        _row(pilot_arm("scratch", "llm"), pilot_arm("scratch", "labels"),
             low=-0.0006, point=0.0009, high=0.0022),
        _row(pilot_arm("landed", "llm"), pilot_arm("landed", "labels"),
             low=-0.0031, point=-0.0016, high=-0.0003),
    ])
    assert gate["passed"] is False
    assert gate["choice"] is None
    assert gate["because"] == []


def test_beating_stage_2_is_not_beating_the_labels():
    gate = paid_run_gate([_row(pilot_arm("scratch", "llm"), "stage2", low=0.01)])
    assert gate["passed"] is False


def test_a_win_across_starting_points_does_not_count():
    # The landed start is a better model before any target is applied; its
    # teacher arm beating a from-scratch labels arm measures the start.
    gate = paid_run_gate([
        _row(pilot_arm("landed", "llm"), pilot_arm("scratch", "labels"), low=0.005),
    ])
    assert gate["passed"] is False


def test_only_a_teacher_target_against_the_labels_counts():
    gate = paid_run_gate([
        _row(pilot_arm("scratch", "hybrid"), pilot_arm("scratch", "llm"), low=0.005),
        _row(pilot_arm("scratch", "labels"), pilot_arm("scratch", "labels"), low=0.005),
    ])
    assert gate["passed"] is False


def test_the_choice_is_the_win_with_the_largest_lower_bound():
    gate = paid_run_gate([
        _row(pilot_arm("scratch", "llm"), pilot_arm("scratch", "labels"), low=0.0002),
        _row(pilot_arm("landed", "hybrid"), pilot_arm("landed", "labels"), low=0.0009),
    ])
    assert gate["choice"] == {"init": "landed", "target": "hybrid"}
    assert gate["because"] == ["pilot:landed:hybrid", "pilot:scratch:llm"]


def test_the_gate_records_its_rule():
    assert paid_run_gate([])["rule"] == GATE_RULE
    assert "wholly above zero" in GATE_RULE
    assert "same starting point" in GATE_RULE


def test_the_pilot_arm_name_carries_its_start_and_target():
    assert pilot_arm("landed", "hybrid") == "pilot:landed:hybrid"


def test_each_pilot_start_has_an_epoch_count():
    from src.distill import INITS

    assert set(PILOT_EPOCHS) == set(INITS)


# --- what the report computes -----------------------------------------------

def test_the_share_of_the_teacher_s_gain_is_zero_at_stage_2_and_one_at_the_teacher():
    assert share_of_teacher_gain(0.8579, 0.8579, 0.8855) == 0.0
    assert share_of_teacher_gain(0.8855, 0.8579, 0.8855) == pytest.approx(1.0)
    # The landed cross-encoder on the full test split: +0.0037 of +0.0276.
    assert share_of_teacher_gain(0.8616, 0.8579, 0.8855) == pytest.approx(0.134, abs=1e-3)


def test_a_teacher_that_does_not_beat_stage_2_has_no_gain_to_share():
    with pytest.raises(ValueError, match="no gain"):
        share_of_teacher_gain(0.85, 0.86, 0.86)


def test_the_reference_arms_are_the_three_stages():
    assert REFERENCE_ARMS == ("stage2", "stage2+ce", "stage2+llm")


def test_the_free_arms_need_no_teacher_answers():
    assert set(FREE_ARMS) <= set(MODEL_ARMS)
    assert "stage2+student" in MODEL_ARMS
    assert "stage2+student" not in FREE_ARMS


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        resolve_model_arms(["stage2+magic"])


def test_a_missing_model_names_the_command_that_trains_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.distill"):
        resolve_model_arms(["stage2+ce_windows"], root=tmp_path)


def test_a_present_model_resolves(tmp_path):
    (tmp_path / MODEL_ARMS["stage2+ce_bge"]).mkdir(parents=True)
    assert resolve_model_arms(["stage2+ce_bge"], root=tmp_path) == {
        "stage2+ce_bge": tmp_path / MODEL_ARMS["stage2+ce_bge"]
    }


def test_the_test_split_needs_final():
    with pytest.raises(SystemExit, match="--final"):
        check_split("test", final=False)
    check_split("test", final=True)
    check_split("train", final=False)


_PUBLISHED = {"stage2": 0.8519, "stage2+ce": 0.8587, "stage2+llm": 0.8814}


def test_reference_arms_that_reproduce_plan_6_pass():
    check_reference({"stage2": 0.85192, "stage2+ce": 0.8587, "stage2+llm": 0.88138},
                    _PUBLISHED)


def test_a_drifted_reference_arm_raises():
    # A re-dumped Stage 2 would carve different windows; every delta after
    # that is against a baseline nobody published.
    with pytest.raises(ValueError, match="drifted"):
        check_reference({"stage2": 0.8559, "stage2+ce": 0.8587, "stage2+llm": 0.8814},
                        _PUBLISHED)


def test_every_model_arm_says_what_trained_it(tmp_path):
    path = tmp_path / "windows"
    path.mkdir()
    (path / "training.json").write_text('{"target": "labels", "init": "scratch"}')
    assert training_records({"stage2+ce_windows": path}) == {
        "stage2+ce_windows": {"target": "labels", "init": "scratch"}
    }


def test_a_model_with_no_training_record_is_refused(tmp_path):
    # A directory someone copied in by hand could be any model at all.
    with pytest.raises(FileNotFoundError, match="python -m src.distill"):
        training_records({"stage2+ce_windows": tmp_path})


def _committed(name):
    return json.loads(Path("docs/results", name).read_text(encoding="utf-8"))


def test_the_committed_pilot_decided_its_gate_by_the_written_rule():
    pilot = _committed("distill-pilot.json")
    assert pilot["gate"]["rule"] == GATE_RULE
    assert pilot["gate"] == paid_run_gate(pilot["comparisons"])
    names = {arm["name"] for arm in pilot["arms"]}
    assert {pilot_arm(i, t) for i in INITS for t in TARGETS} <= names
    assert {arm["n_queries"] for arm in pilot["arms"]} == {4130}


def test_the_committed_fold_0_arms_sit_on_plan_6_s_windows():
    arms = {a["name"]: a["ndcg"]["point"] for a in _committed("distill.json")["arms"]}
    published = {a["name"]: a["ndcg"]["point"] for a in _committed("fine-rank.json")["arms"]}
    for name in REFERENCE_ARMS:
        assert arms[name] == pytest.approx(published[name], abs=5e-4)
    assert set(FREE_ARMS) <= set(arms)


def test_the_student_is_exactly_what_the_gate_chose():
    # No gate, no student. A passed gate names one (init, target), and the
    # student's own training record must say it was trained on that.
    gate = _committed("distill-pilot.json")["gate"]
    report = _committed("distill.json")
    arms = {arm["name"] for arm in report["arms"]}
    if not gate["passed"]:
        assert "stage2+student" not in arms
        return
    assert "stage2+student" in arms
    record = report["training"]["stage2+student"]
    assert (record["init"], record["target"]) == (
        gate["choice"]["init"], gate["choice"]["target"]
    )
