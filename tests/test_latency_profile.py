"""Latency profiles and cost: pure arithmetic, no timing of real models."""

import pytest

from esci_rerank.serving.latency_profile import (
    MIN_FOR_P95,
    MIN_FOR_P99,
    hosted_cost_per_1k,
    percentile,
    profile,
    self_hosted_cost_per_1k,
    time_single_queries,
)


def test_nearest_rank_percentile():
    values = sorted(float(v) for v in range(1, 101))
    assert percentile(values, 0.50) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 1.00) == 100.0


def test_percentile_refuses_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)


def test_p95_needs_enough_samples():
    small = profile([10.0] * (MIN_FOR_P95 - 1), runtime="hf", precision="bf16")
    assert small.p95_ms is None
    enough = profile([10.0] * MIN_FOR_P95, runtime="hf", precision="bf16")
    assert enough.p95_ms == 10.0
    assert enough.p99_ms is None


def test_p99_needs_a_thousand_samples():
    big = profile([float(i) for i in range(MIN_FOR_P99)], runtime="vllm", precision="w8a8")
    assert big.p99_ms is not None


def test_a_profile_carries_its_runtime_and_precision():
    p = profile([1.0, 2.0, 3.0], runtime="vllm", precision="w4a16")
    assert p.to_dict()["runtime"] == "vllm"
    assert p.to_dict()["precision"] == "w4a16"
    assert p.mean_ms == pytest.approx(2.0)


def test_negative_samples_are_refused():
    with pytest.raises(ValueError, match="negative"):
        profile([1.0, -1.0], runtime="hf", precision="bf16")


def test_time_single_queries_warms_up_first_and_times_only_items():
    calls = []
    p = time_single_queries(calls.append, ["a", "b"], warmup=["w"], runtime="hf", precision="bf16")
    assert calls == ["w", "a", "b"]
    assert p.n == 2


def test_warmup_items_must_not_be_timed_items():
    with pytest.raises(ValueError, match="warm"):
        time_single_queries(lambda x: x, ["a"], warmup=["a"], runtime="hf", precision="bf16")


def test_cost_is_none_without_a_rate():
    assert self_hosted_cost_per_1k(queries_per_second=10.0, gpu_usd_per_hour=None) is None
    assert hosted_cost_per_1k(prompt_tokens=454, completion_tokens=382,
                              usd_per_mtok_in=None, usd_per_mtok_out=1.0) is None


def test_self_hosted_cost_per_1k():
    # 10 queries/s -> 1,000 queries in 100 s = 1/36 h; at $3.60/h that is $0.10.
    assert self_hosted_cost_per_1k(queries_per_second=10.0, gpu_usd_per_hour=3.6) == pytest.approx(0.10)


def test_hosted_cost_per_1k():
    # 1,000 queries x 1,000 prompt tokens = 1M at $1, plus 1M completion at $2.
    assert hosted_cost_per_1k(prompt_tokens=1000, completion_tokens=1000,
                              usd_per_mtok_in=1.0, usd_per_mtok_out=2.0) == pytest.approx(3.0)


def test_zero_throughput_is_refused():
    with pytest.raises(ValueError, match="throughput"):
        self_hosted_cost_per_1k(queries_per_second=0.0, gpu_usd_per_hour=1.0)
