"""Latency as a distribution, and cost per 1,000 queries that is never invented.

Plans 6-11 reported a mean single-query latency. A search engine is held to
its tail, so Plan 13 reports p50 and p95, and p99 when there are enough
samples to mean anything. Every profile names the runtime and precision it
was measured under, because a latency without them cannot be compared.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

MIN_FOR_P95 = 100
MIN_FOR_P99 = 1000


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted sample."""
    if not sorted_values:
        raise ValueError("percentile of an empty sample")
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[rank - 1])


@dataclass(frozen=True)
class LatencyProfile:
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float | None
    p99_ms: float | None
    runtime: str
    precision: str

    def to_dict(self) -> dict:
        return asdict(self)


def profile(samples_ms: Sequence[float], *, runtime: str, precision: str) -> LatencyProfile:
    """Summarise single-query latencies. Tail percentiles need enough samples."""
    if not samples_ms:
        raise ValueError("no latency samples")
    if any(s < 0 for s in samples_ms):
        raise ValueError("negative latency sample")
    ordered = sorted(float(s) for s in samples_ms)
    n = len(ordered)
    return LatencyProfile(
        n=n,
        mean_ms=sum(ordered) / n,
        p50_ms=percentile(ordered, 0.50),
        p95_ms=percentile(ordered, 0.95) if n >= MIN_FOR_P95 else None,
        p99_ms=percentile(ordered, 0.99) if n >= MIN_FOR_P99 else None,
        runtime=runtime,
        precision=precision,
    )


def time_single_queries(
    fn: Callable[[object], object],
    items: Sequence,
    *,
    warmup: Sequence,
    runtime: str,
    precision: str,
) -> LatencyProfile:
    """Run `fn` on each warm-up item untimed, then time it on each item alone.

    Warm-up items must not be timed items: a query timed after it was already
    run can hit a cache a real user never would.
    """
    overlap = set(map(_identity, warmup)) & set(map(_identity, items))
    if overlap:
        raise ValueError(
            f"{len(overlap)} warm-up items are also timed items; warm up on "
            "queries that are never timed"
        )
    for item in warmup:
        fn(item)
    samples = []
    for item in items:
        started = time.perf_counter()
        fn(item)
        samples.append((time.perf_counter() - started) * 1000)
    return profile(samples, runtime=runtime, precision=precision)


def _identity(item) -> object:
    """Windows compare by query id; anything else by value."""
    return getattr(item, "query_id", item)


def self_hosted_cost_per_1k(
    *, queries_per_second: float, gpu_usd_per_hour: float | None
) -> float | None:
    """GPU time for 1,000 queries at a sustained throughput, priced only if a rate is given."""
    if queries_per_second <= 0:
        raise ValueError("throughput must be positive")
    if gpu_usd_per_hour is None:
        return None
    hours = 1000 / queries_per_second / 3600
    return hours * gpu_usd_per_hour


def hosted_cost_per_1k(
    *,
    prompt_tokens: float,
    completion_tokens: float,
    usd_per_mtok_in: float | None,
    usd_per_mtok_out: float | None,
) -> float | None:
    """Token cost of 1,000 queries at per-query token counts, priced only if both rates are given."""
    if usd_per_mtok_in is None or usd_per_mtok_out is None:
        return None
    return 1000 * (
        prompt_tokens / 1e6 * usd_per_mtok_in + completion_tokens / 1e6 * usd_per_mtok_out
    )
