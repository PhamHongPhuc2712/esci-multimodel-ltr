"""Load-test planning and accounting, with a no-op sender."""

import pytest

from esci_rerank.serving.load_test import (
    MIN_SAMPLES,
    RateResult,
    duration_for,
    max_rate_under,
    plan_rates,
    poisson_offsets,
    run_rate,
    should_stop,
)
from esci_rerank.serving.latency_profile import profile
from esci_rerank.rerank.rerank_window import Window


def _windows(n):
    return [Window(query_id=str(i), window=("a",), tail=()) for i in range(n)]


def test_low_rates_run_long_enough_for_a_p95():
    assert duration_for(1) * 1 >= MIN_SAMPLES
    assert duration_for(64) == 60.0


def test_poisson_offsets_are_increasing_and_inside_the_duration():
    offsets = poisson_offsets(10.0, 60.0, seed=0)
    assert offsets == sorted(offsets)
    assert all(0 <= o < 60.0 for o in offsets)
    assert 450 < len(offsets) < 750


def test_rates_draw_disjoint_queries():
    plan = plan_rates(_windows(5000), (1, 2, 4), seed=0)
    seen = [w.query_id for _, ws in plan.values() for w in ws]
    assert len(seen) == len(set(seen))
    for offsets, ws in plan.values():
        assert len(offsets) == len(ws)


def test_not_enough_unique_queries_is_refused():
    with pytest.raises(ValueError, match="unique"):
        plan_rates(_windows(100), (1, 2, 4), seed=0)


def test_run_rate_counts_errors_and_times_only_successes():
    def send(w):
        if w.query_id == "1":
            raise RuntimeError("server error")

    result = run_rate(send, _windows(3), [0.0, 0.01, 0.02], rate=100.0)
    assert (result.n, result.n_ok, result.n_err) == (3, 2, 1)
    assert result.latency.n == 2


def _result(rate, p95, errors=0):
    samples = [p95] * 200
    return RateResult(rate=rate, n=200, n_ok=200 - errors, n_err=errors, seconds=60.0,
                      achieved_qps=rate, latency=profile(samples, runtime="vllm", precision="bf16"))


def test_the_run_stops_past_two_seconds_or_one_percent_errors():
    assert not should_stop(_result(8, 120.0))
    assert should_stop(_result(8, 2500.0))
    assert should_stop(_result(8, 120.0, errors=3))


def test_the_highest_rate_inside_the_slo():
    results = [_result(4, 80.0), _result(8, 150.0), _result(16, 420.0)]
    assert max_rate_under(results) == 8
    assert max_rate_under([_result(4, 900.0)]) is None
