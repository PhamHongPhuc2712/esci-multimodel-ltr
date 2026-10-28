"""The Plan 13 report's rules, with no models."""

from pathlib import Path

import pytest

from src.llm_rank_report import (
    CE_ARMS,
    LLM_ARMS,
    OPTIONAL_ARMS,
    available_arms,
    best_llm_arm,
    check_both_precisions,
    check_size_reference,
    comparison_pairs,
    require_final,
)


def _make(root: Path, arm: str) -> None:
    _, path = LLM_ARMS[arm]
    (root / path).mkdir(parents=True)
    (root / path / "config.json").write_text("{}")


def test_hub_models_are_always_available_and_trained_ones_need_their_directory(tmp_path):
    for arm in ("stage2+qwen_point", "stage2+qwen_list"):
        _make(tmp_path, arm)
    arms = available_arms(tmp_path)
    assert "stage2+qwen_point_zs" in arms
    assert "stage2+qwen_point_4b" not in arms


def test_a_missing_required_arm_is_an_error(tmp_path):
    _make(tmp_path, "stage2+qwen_point")
    with pytest.raises(FileNotFoundError, match="qwen_list"):
        available_arms(tmp_path)


def test_only_the_gated_arm_is_optional():
    assert OPTIONAL_ARMS == ("stage2+qwen_point_4b",)


def test_the_bge_arms_must_reproduce_plan_11():
    check_size_reference({"stage2+ce_bge_large": 0.87421}, {"stage2+ce_bge_large": 0.8742})
    with pytest.raises(ValueError, match="drift"):
        check_size_reference({"stage2+ce_bge_large": 0.8750}, {"stage2+ce_bge_large": 0.8742})


def test_every_cross_encoder_is_timed_in_both_precisions():
    complete = {arm: {"fp32": {}, "fp16": {}} for arm in CE_ARMS}
    check_both_precisions(complete)
    partial = dict(complete)
    partial["stage2+ce_bge_large"] = {"fp32": {}}
    with pytest.raises(ValueError, match="fp16"):
        check_both_precisions(partial)


def test_best_llm_arm_ignores_the_zero_shot_and_the_references():
    ndcg = {"stage2+llm": 0.88, "stage2+qwen_point_zs": 0.87,
            "stage2+qwen_point": 0.86, "stage2+qwen_list": 0.865}
    assert best_llm_arm(ndcg) == "stage2+qwen_list"


def test_best_llm_arm_needs_a_trained_arm():
    with pytest.raises(ValueError, match="trained"):
        best_llm_arm({"stage2+qwen_point_zs": 0.87})


def test_every_qwen_arm_is_compared_with_bge_large_and_the_hosted_llm():
    pairs = comparison_pairs(["stage2", "stage2+ce_bge_large", "stage2+llm",
                              "stage2+qwen_point_zs", "stage2+qwen_point", "stage2+qwen_list"])
    for arm in ("stage2+qwen_point", "stage2+qwen_list", "stage2+qwen_point_zs"):
        assert (arm, "stage2+ce_bge_large") in pairs
        assert (arm, "stage2+llm") in pairs
    assert ("stage2+qwen_point", "stage2+qwen_point_zs") in pairs
    assert ("stage2+qwen_list", "stage2+qwen_point") in pairs
    assert not any(a == "stage2+qwen_point_4b" for a, _ in pairs)


def test_the_test_split_needs_final():
    require_final("train", False)
    require_final("test", True)
    with pytest.raises(SystemExit, match="final"):
        require_final("test", False)
