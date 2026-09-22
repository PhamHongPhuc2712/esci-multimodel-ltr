import json

import pytest

from src.fine_rank_report import (
    ARMS,
    TEST_SAMPLE,
    check_same_queries,
    cost_usd,
)
from src.cross_encoder import Latency
from src.llm_rerank import Usage
from src.rank_report import evaluate_arm


def _arm(name, values):
    per = {str(i): v for i, v in enumerate(values)}
    floor = {str(i): 0.74 for i in range(len(values))}
    return evaluate_arm(
        name, per, floor, groups=("retrieval",), n_features=0,
        objective="rerank", best_iteration=0,
    )


# --- the arms ---------------------------------------------------------------

def test_the_baseline_arm_is_stage_2():
    assert ARMS[0] == "stage2"


def test_both_stage_3_variants_are_present():
    # PROJECT_SPEC.md §6 Ablation 6: coarse-only vs +cross-encoder vs +LLM.
    assert "stage2+ce" in ARMS
    assert "stage2+llm" in ARMS


def test_the_zero_shot_control_is_present():
    # Measured: every zero-shot cross-encoder loses to Stage 2, so this arm is
    # what shows the fine-tune earned its GPU hours.
    assert "stage2+ce_zeroshot" in ARMS


def test_the_cascade_arm_is_present():
    assert "stage2+ce+llm" in ARMS


# --- cost -------------------------------------------------------------------

def _usage():
    return Usage(prompt_tokens=1_000_000, completion_tokens=2_000_000, n_calls=100)


def test_cost_is_none_without_prices():
    # Prices change and are not measurable from here.
    assert cost_usd(_usage()) is None


def test_cost_is_none_with_only_one_price():
    assert cost_usd(_usage(), price_in=1.0) is None
    assert cost_usd(_usage(), price_out=1.0) is None


def test_cost_uses_both_rates_per_million_tokens():
    assert cost_usd(_usage(), price_in=2.0, price_out=8.0) == pytest.approx(
        2.0 * 1 + 8.0 * 2
    )


def test_cost_of_no_tokens_is_zero():
    assert cost_usd(Usage(), price_in=1.0, price_out=1.0) == 0.0


# --- comparability ----------------------------------------------------------

def test_arms_over_the_same_queries_pass():
    check_same_queries([_arm("a", [0.8, 0.9]), _arm("b", [0.7, 0.6])])


def test_arms_over_different_queries_are_refused():
    # A 2,000-query LLM number against an 8,956-query cross-encoder number is
    # not a reranker effect.
    a = _arm("a", [0.8, 0.9])
    b = _arm("b", [0.8, 0.9, 0.7])
    with pytest.raises(ValueError, match="same queries"):
        check_same_queries([a, b])


def test_checking_one_arm_is_fine():
    check_same_queries([_arm("a", [0.8])])


def test_checking_nothing_is_fine():
    check_same_queries([])


def test_the_test_sample_is_frozen_and_documented():
    # All 8,956 test queries through the LLM is ~3.3 h and real money.
    assert TEST_SAMPLE == 2000


# --- serialisation ----------------------------------------------------------

def test_latency_and_usage_both_reach_the_payload():
    payload = {
        "latency": Latency(single_ms=40.0, batched_ms=8.0, n_queries=50, n_pairs=500).to_dict(),
        "usage": Usage(prompt_tokens=10, completion_tokens=20).to_dict(),
        "cost_usd": None,
    }
    assert json.loads(json.dumps(payload))["latency"]["single_ms"] == 40.0
    assert payload["latency"]["batched_ms"] == 8.0
    assert payload["usage"]["completion_tokens"] == 20
