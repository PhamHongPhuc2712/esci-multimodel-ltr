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


# --- latency provenance -----------------------------------------------------
# The first fold-0 run published the LLM arm at 5.5 ms/query, derived from the
# wall clock of a fully cached replay. The uncached pass had measured 1,180
# ms/query. Latency is only a latency if calls were actually made.

from src.fine_rank_report import llm_latency  # noqa: E402


def test_a_fully_cached_pass_has_no_latency():
    cached = Usage(seconds=22.6, call_seconds=0.0, n_calls=0, n_cached=4130)
    assert llm_latency(cached, 4130) is None


def test_a_mostly_cached_pass_has_no_latency():
    # The wall clock covers windows that cost nothing, so it times neither the
    # single call nor the throughput.
    partial = Usage(seconds=22.6, call_seconds=7.0, n_calls=6, n_cached=4124)
    assert llm_latency(partial, 4130) is None


def test_a_real_pass_reports_both_numbers():
    real = Usage(seconds=4882.0, call_seconds=19528.0, n_calls=4130, n_cached=0)
    latency = llm_latency(real, 4130)
    assert latency is not None
    # 4.73 s inside one call; 1.18 s/query wall clock at concurrency 4.
    assert latency.single_ms == pytest.approx(4728.0, rel=0.01)
    assert latency.batched_ms == pytest.approx(1182.0, rel=0.01)
    assert latency.single_ms > latency.batched_ms


def test_no_windows_means_no_latency():
    assert llm_latency(Usage(n_calls=5), 0) is None


@pytest.mark.data
def test_the_committed_report_carries_every_ablation_6_column():
    payload = json.loads(open("docs/results/fine-rank.json").read())

    names = {arm["name"] for arm in payload["arms"]}
    assert {"stage2", "stage2+ce", "stage2+llm"} <= names

    # Every arm carries the floor it beats. PROJECT_SPEC.md §2.
    assert 0.70 < payload["floor"]["mean"] < 0.80
    for arm in payload["arms"]:
        assert set(arm["lift_over_floor"]) == {"point", "low", "high"}

    # §6 asks for NDCG, latency AND cost.
    by_name = {arm["name"]: arm for arm in payload["arms"]}
    assert by_name["stage2+ce"]["cost"]["latency"]["single_ms"] > 0
    assert by_name["stage2+ce"]["cost"]["latency"]["batched_ms"] > 0

    # The baseline must reproduce Plan 5.
    assert by_name["stage2"]["ndcg"]["point"] == pytest.approx(0.8519, abs=0.002)

    # Ablation 6 compares every arm against coarse-only.
    assert {row["baseline"] for row in payload["ablation_6"]} == {"stage2"}


@pytest.mark.data
def test_the_llm_arm_reports_latency_from_real_calls_not_cache_replay():
    # The first run published 5.5 ms/query from a cached replay against a real
    # 1,180 ms/query. Either the arm measured its own uncached pass, or the
    # probe carries the number - never the replay.
    payload = json.loads(open("docs/results/fine-rank.json").read())
    by_name = {arm["name"]: arm for arm in payload["arms"]}
    arm = by_name["stage2+llm"]["cost"]
    probe = payload["llm_latency_probe"]

    if arm["latency"] is None:
        assert arm["usage"]["n_cached"] > 0, "null latency needs a cache to explain it"
        assert probe is not None, "a cached arm must carry a probe instead"
        assert probe["latency"]["single_ms"] > 100
    else:
        assert arm["usage"]["n_calls"] >= 0.9 * payload["n_queries"]
        assert arm["latency"]["single_ms"] > 100

    # The LLM is orders of magnitude slower than the cross-encoder, and the
    # report must say so rather than flattering it.
    assert probe["latency"]["single_ms"] > by_name["stage2+ce"]["cost"]["latency"]["single_ms"] * 10
