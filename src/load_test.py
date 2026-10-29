"""Plan 13: an open-loop load test of the served reranker.

Poisson arrivals at rising rates; latency from each request's scheduled
arrival; no query repeated in one server lifetime. The run stops once p95
passes 2 s or errors pass 1%, and reports the highest rate whose p95 stays
under 300 ms, and the GPU cost of 1,000 queries at that rate if a price is
given.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.latency_profile import LatencyProfile, profile

RATES: tuple[float, ...] = (1, 2, 4, 8, 16, 32, 64)
MIN_SAMPLES = 150
SLO_P95_MS = 300.0
STOP_P95_MS = 2000.0
STOP_ERROR_SHARE = 0.01
OUT = Path("docs/results/llm-load.json")


def duration_for(rate: float) -> float:
    return max(60.0, MIN_SAMPLES / rate)


def poisson_offsets(rate: float, duration_s: float, seed: int) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    offsets, t = [], 0.0
    while True:
        t += float(rng.exponential(1.0 / rate))
        if t >= duration_s:
            return offsets
        offsets.append(t)


def plan_rates(windows_: Sequence, rates: Sequence[float], seed: int = 0) -> dict:
    plan, cursor = {}, 0
    for rate in rates:
        offsets = poisson_offsets(rate, duration_for(rate), seed + int(rate * 1000))
        if cursor + len(offsets) > len(windows_):
            raise ValueError(
                f"rate {rate} needs {len(offsets)} more queries but only "
                f"{len(windows_) - cursor} unique ones remain; drop the highest rate"
            )
        plan[rate] = (offsets, list(windows_[cursor : cursor + len(offsets)]))
        cursor += len(offsets)
    return plan


@dataclass(frozen=True)
class RateResult:
    rate: float
    n: int
    n_ok: int
    n_err: int
    seconds: float
    achieved_qps: float
    latency: LatencyProfile | None

    def to_dict(self) -> dict:
        return {"rate": self.rate, "n": self.n, "n_ok": self.n_ok, "n_err": self.n_err,
                "error_share": self.n_err / self.n if self.n else 0.0,
                "seconds": self.seconds, "achieved_qps": self.achieved_qps,
                "latency": self.latency.to_dict() if self.latency else None}


def _timed(send: Callable, item, scheduled: float) -> tuple[bool, float]:
    try:
        send(item)
        ok = True
    except Exception:  # noqa: BLE001 - every failure is an error to count, not to raise
        ok = False
    return ok, (time.perf_counter() - scheduled) * 1000


def run_rate(send: Callable, items: Sequence, offsets: Sequence[float], *,
             rate: float, max_workers: int = 128,
             runtime: str = "vllm", precision: str = "bf16") -> RateResult:
    start = time.perf_counter()
    futures = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for item, offset in zip(items, offsets):
            scheduled = start + offset
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(pool.submit(_timed, send, item, scheduled))
        outcomes = [f.result() for f in futures]
    seconds = time.perf_counter() - start
    ok = [ms for success, ms in outcomes if success]
    return RateResult(
        rate=rate, n=len(outcomes), n_ok=len(ok), n_err=len(outcomes) - len(ok),
        seconds=seconds, achieved_qps=len(ok) / seconds if seconds > 0 else 0.0,
        latency=profile(ok, runtime=runtime, precision=precision) if ok else None,
    )


def should_stop(result: RateResult) -> bool:
    if result.n and result.n_err / result.n > STOP_ERROR_SHARE:
        return True
    lat = result.latency
    tail = lat.p95_ms if lat and lat.p95_ms is not None else (lat.p50_ms if lat else None)
    return tail is None or tail > STOP_P95_MS


def max_rate_under(results: Sequence[RateResult], slo_ms: float = SLO_P95_MS) -> float | None:
    inside = [r.rate for r in results
              if r.latency and r.latency.p95_ms is not None and r.latency.p95_ms <= slo_ms
              and (r.n_err / r.n if r.n else 0.0) <= STOP_ERROR_SHARE]
    return max(inside) if inside else None


def _main() -> int:
    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    from src.cross_encoder import text_maps
    from src.latency_profile import self_hosted_cost_per_1k
    from src.llm_point import DEFAULT_MAX_LENGTH, warmup_scope
    from src.serve_client import (DEFAULT_URL, ServedRanker, prefix_cache_hits,
                                  server_version, urllib_get)
    from src.stage_signals import scope_windows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--kind", choices=["point", "list"], required=True)
    parser.add_argument("--precision", required=True, help="bf16, w8a8 or w4a16, as served")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--server-flags", default="")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--gpu-usd-per-hour", type=float, default=None)
    parser.add_argument("--price-source", default=None,
                        help="where the hourly rate came from, with the date; required with a rate")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.gpu_usd_per_hour is not None and not args.price_source:
        raise SystemExit("--price-source is required with --gpu-usd-per-hour")

    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "unknown"
    hits = prefix_cache_hits(urllib_get(f"{args.url}/metrics"))
    if hits:
        raise SystemExit(f"the server already has {hits:.0f} prefix-cache hits; restart it")

    fold0, m0 = scope_windows("fold0")
    fold1, m1 = warmup_scope()
    pool_windows = list(fold0) + list(fold1)
    query_text, doc_text = text_maps(pd.concat([m0, m1], ignore_index=True),
                                     Path("data/combined/products.parquet"),
                                     Path("data/combined/judgements.parquet"))
    query_text = {str(k): v for k, v in query_text.items()}
    tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    ranker = ServedRanker(args.url, args.served_name, tok, args.kind,
                          query_text, doc_text, args.max_length)

    plan = plan_rates(pool_windows, RATES)
    results: list[RateResult] = []
    for rate, (offsets, items) in plan.items():
        result = run_rate(ranker.order, items, offsets, rate=rate, precision=args.precision)
        results.append(result)
        lat = result.latency
        print(f"rate {rate:>4}: {result.achieved_qps:6.1f} q/s achieved, "
              f"p50 {lat.p50_ms if lat else float('nan'):7.1f} ms, "
              f"p95 {lat.p95_ms if lat and lat.p95_ms else float('nan'):7.1f} ms, "
              f"errors {result.n_err}")
        if should_stop(result):
            break

    best = max_rate_under(results)
    cost = (self_hosted_cost_per_1k(queries_per_second=best, gpu_usd_per_hour=args.gpu_usd_per_hour)
            if best else None)
    payload = {
        "served_name": args.served_name, "model_dir": args.model_dir, "kind": args.kind,
        "precision": args.precision, "server_flags": args.server_flags, "device": device,
        "vllm": server_version(urllib_get, args.url),
        "prefix_cache_hits_at_start": hits,
        "rates": [r.to_dict() for r in results],
        "slo_p95_ms": SLO_P95_MS, "max_rate_under_slo": best,
        "cost_per_1k_queries_usd": cost,
        "cost_basis": ({"gpu_usd_per_hour": args.gpu_usd_per_hour, "source": args.price_source,
                        "note": f"throughput measured on {device}; the rate prices that card"}
                       if cost is not None else
                       {"reason": "no hourly rate given" if best else "no rate met the SLO"}),
        "imputed_answers": ranker.imputed, "malformed_answers": len(ranker.malformed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"highest rate with p95 under {SLO_P95_MS:.0f} ms: {best}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
