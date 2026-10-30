"""The speed ladder's rules, with fake rankers."""

import pytest

from esci_rerank.serving.llm_speed_report import (
    check_agreement,
    deployed_rung,
    lossless,
    run_rung,
)
from esci_rerank.rerank.rerank_window import Window


def test_a_rung_whose_interval_excludes_zero_is_not_lossless():
    assert lossless({"point": -0.0004, "low": -0.0011, "high": 0.0003})
    assert not lossless({"point": -0.0030, "low": -0.0042, "high": -0.0018})


def test_the_served_model_must_agree_with_the_reference():
    check_agreement({"point": 0.0002, "low": -0.0003, "high": 0.0007})
    with pytest.raises(ValueError, match="agree"):
        check_agreement({"point": -0.0020, "low": -0.0030, "high": -0.0010})
    with pytest.raises(ValueError, match="agree"):
        check_agreement({"point": 0.0015, "low": -0.0001, "high": 0.0031})


def test_latency_is_timed_before_quality_on_unseen_queries():
    windows = [Window(query_id=str(i), window=("a", "b"), tail=()) for i in range(300)]
    warmup = [Window(query_id=f"w{i}", window=("a", "b"), tail=()) for i in range(3)]
    log = []

    def order_one(w):
        log.append(("one", w.query_id))
        return list(w.window)

    def order_many(ws):
        log.append(("many", tuple(w.query_id for w in ws)))
        return {w.query_id: list(w.window) for w in ws}

    qrels = {w.query_id: {"a": 100, "b": 0} for w in windows}
    latency, per_query, qps = run_rung(order_one, order_many, windows, warmup, qrels,
                                       runtime="vllm", precision="bf16")
    timed = [q for kind, q in log if kind == "one" and not q.startswith("w")]
    many = [qs for kind, qs in log if kind == "many"]
    first_many = log.index(next(e for e in log if e[0] == "many"))
    assert len(timed) == 200
    assert all(e[0] == "one" for e in log[:first_many])
    assert not set(timed) & set(many[0])
    assert set(per_query) == {w.query_id for w in windows}
    assert latency.n == 200 and qps > 0


def test_the_deployed_rung_is_the_fastest_lossless_served_rung():
    rungs = {
        "vllm_bf16_prefix": {"lossless": True, "latency": {"p95_ms": 40.0, "mean_ms": 30.0}},
        "vllm_w8a8_prefix": {"lossless": True, "latency": {"p95_ms": 30.0, "mean_ms": 25.0}},
        "vllm_w4a16_prefix": {"lossless": False, "latency": {"p95_ms": 20.0, "mean_ms": 15.0}},
        "hf_bf16_batched": {"lossless": True, "latency": {"p95_ms": 10.0, "mean_ms": 9.0}},
    }
    assert deployed_rung(rungs) == "vllm_w8a8_prefix"


def test_without_a_lossless_quantised_rung_bf16_prefix_is_deployed():
    rungs = {
        "vllm_bf16_prefix": {"lossless": True, "latency": {"p95_ms": 40.0, "mean_ms": 30.0}},
        "vllm_w4a16_prefix": {"lossless": False, "latency": {"p95_ms": 20.0, "mean_ms": 15.0}},
    }
    assert deployed_rung(rungs) == "vllm_bf16_prefix"


def test_no_served_rung_means_nothing_to_deploy():
    with pytest.raises(ValueError, match="vllm_bf16_prefix"):
        deployed_rung({"hf_bf16_batched": {"lossless": True, "latency": {"p95_ms": 1.0}}})
