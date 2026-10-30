"""The chart is built from committed JSON, and every point says how it was timed."""

import json

import pytest

from src.quality_latency_chart import chart_points


def _profile(mean, runtime, precision):
    return {"n": 200, "mean_ms": mean, "p50_ms": mean, "p95_ms": mean * 1.3,
            "p99_ms": None, "runtime": runtime, "precision": precision}


def _interval(point):
    return {"point": point, "low": point - 0.003, "high": point + 0.003}


def _write(tmp_path, qwen_latency=None):
    arms = [
        {"name": "stage2", "ndcg": _interval(0.8579), "latency": None},
        {"name": "stage2+ce", "ndcg": _interval(0.8616), "latency": None},
        {"name": "stage2+ce_bge_groups", "ndcg": _interval(0.8711),
         "latency": {"fp32": _profile(64.0, "hf", "fp32"), "fp16": _profile(25.0, "hf", "fp16")}},
        {"name": "stage2+ce_bge_large", "ndcg": _interval(0.8770),
         "latency": {"fp32": _profile(200.0, "hf", "fp32"), "fp16": _profile(60.0, "hf", "fp16")}},
        {"name": "stage2+llm", "ndcg": _interval(0.8855), "latency": None},
        {"name": "stage2+qwen_point", "ndcg": _interval(0.8800),
         "latency": qwen_latency or {"bf16": _profile(40.0, "hf", "bf16")}},
    ]
    (tmp_path / "llm-rank-test.json").write_text(json.dumps({
        "arms": arms,
        "hosted_llm_latency": {"single_ms": 4712.8, "batched_ms": 1179.1,
                               "runtime": "hosted API", "precision": "unknown"},
    }))
    (tmp_path / "fine-rank-test-full.json").write_text(json.dumps({
        "arms": [{"name": "stage2+ce", "cost": {"latency": {"single_ms": 19.7, "batched_ms": 9.6}}}],
    }))
    (tmp_path / "llm-speed.json").write_text(json.dumps({
        "rungs": {"vllm_w8a8_prefix": {"latency": _profile(15.0, "vllm", "w8a8")}},
        "deployed_rung": "vllm_w8a8_prefix",
        "test_quality": {"vllm_w8a8_prefix": {"ndcg": _interval(0.8798)}},
    }))


def test_every_point_carries_the_committed_ndcg(tmp_path):
    _write(tmp_path)
    by_name = {p.name: p for p in chart_points(tmp_path)}
    assert by_name["stage2+ce_bge_large@fp16"].ndcg == 0.8770
    assert by_name["stage2+qwen_point@bf16"].ndcg == 0.8800
    assert by_name["deployed:vllm_w8a8_prefix"].ndcg == 0.8798
    assert by_name["stage2+llm"].mean_ms == 4712.8
    assert "stage2" not in by_name


def test_both_cross_encoder_precisions_are_plotted(tmp_path):
    _write(tmp_path)
    names = {p.name for p in chart_points(tmp_path)}
    assert {"stage2+ce_bge_large@fp32", "stage2+ce_bge_large@fp16"} <= names


def test_every_chart_point_names_its_runtime_and_precision(tmp_path):
    bad = {"bf16": {"n": 200, "mean_ms": 40.0, "p50_ms": 40.0, "p95_ms": 50.0, "p99_ms": None}}
    _write(tmp_path, qwen_latency=bad)
    with pytest.raises(ValueError, match="runtime"):
        chart_points(tmp_path)


def test_every_label_names_the_card_it_was_timed_on(tmp_path):
    # Plan 6 timed MiniLM on the laptop's RTX 3080; Plan 13 timed the rest on
    # an A6000. A point without its device would compare the two cards.
    _write(tmp_path)
    by_name = {p.name: p for p in chart_points(tmp_path)}
    assert "RTX 3080" in by_name["stage2+ce"].label
    assert "hosted" in by_name["stage2+llm"].label
    assert all(p.device for p in by_name.values())
