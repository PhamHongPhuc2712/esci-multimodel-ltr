"""Plan 13: the speed ladder for the arm chosen on fold 0.

One rung per invocation, appended to docs/results/llm-speed.json. Each rung
changes one thing from the rung above; each reports a single-query latency
profile, throughput, and fold-0 NDCG paired against the in-process bf16
reference. The served bf16 rung must agree with that reference, or the ladder
stops: the served model is not the model that was evaluated.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from esci_rerank.serving.latency_profile import LatencyProfile, time_single_queries

RUNGS: dict[str, tuple[str, str]] = {
    "hf_bf16_loop": ("hf", "bf16"),
    "hf_bf16_batched": ("hf", "bf16"),
    "vllm_bf16": ("vllm", "bf16"),
    "vllm_bf16_prefix": ("vllm", "bf16"),
    "vllm_w8a8_prefix": ("vllm", "w8a8"),
    "vllm_w4a16_prefix": ("vllm", "w4a16"),
}
REFERENCE_RUNG = "hf_bf16_batched"
AGREEMENT_RUNG = "vllm_bf16"
MAX_AGREEMENT_DRIFT = 0.001
LATENCY_QUERIES = 200
WARMUP_QUERIES = 20
THROUGHPUT_CONCURRENCY = 16
OUT = Path("docs/results/llm-speed.json")
REFERENCE_FILE = {"train": Path("data/features/llm-speed-reference.json"),
                  "test": Path("data/features/llm-speed-reference-test.json")}


def lossless(delta: Mapping[str, float]) -> bool:
    return delta["low"] <= 0.0 <= delta["high"]


def check_agreement(delta: Mapping[str, float]) -> None:
    if not lossless(delta) or abs(delta["point"]) > MAX_AGREEMENT_DRIFT:
        raise ValueError(
            f"the served bf16 model does not agree with the in-process reference "
            f"({delta['point']:+.4f} [{delta['low']:+.4f}, {delta['high']:+.4f}]); "
            "check the template, the token ids and the checkpoint before any other rung"
        )


def run_rung(
    order_one: Callable,
    order_many: Callable[[Sequence], dict],
    windows_: Sequence,
    warmup: Sequence,
    qrels: Mapping,
    *,
    runtime: str,
    precision: str,
) -> tuple[LatencyProfile, dict[str, float], float]:
    """Latency on 200 unseen queries, then throughput on the rest, then quality on all."""
    from esci_rerank.blend.blend_report import per_query_ndcg

    timed, rest = list(windows_[:LATENCY_QUERIES]), list(windows_[LATENCY_QUERIES:])
    latency = time_single_queries(order_one, timed, warmup=warmup,
                                  runtime=runtime, precision=precision)
    started = time.perf_counter()
    orderings = dict(order_many(rest))
    qps = len(rest) / (time.perf_counter() - started)
    orderings.update(order_many(timed))
    return latency, per_query_ndcg(orderings, windows_, qrels), qps


def deployed_rung(rungs: Mapping[str, dict]) -> str:
    if "vllm_bf16_prefix" not in rungs:
        raise ValueError("vllm_bf16_prefix has not been measured; nothing to deploy")
    quantised = [r for r in ("vllm_w8a8_prefix", "vllm_w4a16_prefix")
                 if r in rungs and rungs[r]["lossless"]]
    candidates = ["vllm_bf16_prefix", *quantised]

    def tail(r: str) -> float:
        lat = rungs[r]["latency"]
        return lat["p95_ms"] if lat.get("p95_ms") is not None else lat["mean_ms"]

    return min(candidates, key=tail)


def _interval(i) -> dict:
    return {"point": i.point, "low": i.low, "high": i.high}


def _main() -> int:
    import pandas as pd
    import torch
    import transformers
    from transformers import AutoTokenizer

    from esci_rerank.blend.blend_report import per_query_ndcg
    from esci_rerank.evaluation.bootstrap import bootstrap_ci, paired_delta_ci
    from esci_rerank.rerank.cross_encoder import text_maps
    from esci_rerank.rerank.llm_list import generate_orderings, load_list_model
    from esci_rerank.rerank.llm_point import DEFAULT_MAX_LENGTH, load_point_model, rerank_point, warmup_scope
    from esci_rerank.ranking.rank_report import qrels_from_frame
    from esci_rerank.serving.serve_client import DEFAULT_URL, ServedRanker, server_version, urllib_get
    from esci_rerank.blend.stage_signals import scope_windows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=list(RUNGS), required=True)
    parser.add_argument("--kind", choices=["point", "list"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--served-name", default=None)
    parser.add_argument("--server-flags", default="", help="the vllm serve flags, recorded verbatim")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("refusing the test split without --final")
    runtime, precision = RUNGS[args.rung]
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else None

    windows_, matrix = scope_windows("fold0" if args.split == "train" else "test")
    warm_windows, warm_matrix = warmup_scope()
    both = pd.concat([matrix, warm_matrix], ignore_index=True)
    query_text, doc_text = text_maps(both, Path("data/combined/products.parquet"),
                                     Path("data/combined/judgements.parquet"))
    query_text = {str(k): v for k, v in query_text.items()}
    qrels = qrels_from_frame(matrix)
    counters: dict = {}

    if runtime == "hf":
        loop = args.rung == "hf_bf16_loop"
        if args.kind == "point":
            model, tok = load_point_model(args.model_dir)

            def order_many(ws):
                return rerank_point(model, tok, ws, query_text, doc_text,
                                    max_length=args.max_length, batch_size=1 if loop else 32)

            def order_one(w):
                return rerank_point(model, tok, [w], query_text, doc_text,
                                    max_length=args.max_length,
                                    batch_size=1 if loop else len(w.window))
        else:
            model, tok = load_list_model(args.model_dir)

            def order_many(ws):
                return generate_orderings(model, tok, ws, query_text, doc_text)[0]

            def order_one(w):
                return generate_orderings(model, tok, [w], query_text, doc_text, batch_size=1)[0]
        versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    else:
        if not args.served_name:
            raise SystemExit("--served-name is required for a vllm rung")
        tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
        ranker = ServedRanker(args.url, args.served_name, tok, args.kind,
                              query_text, doc_text, args.max_length)

        def order_one(w):
            return ranker.order(w)

        def order_many(ws):
            with cf.ThreadPoolExecutor(THROUGHPUT_CONCURRENCY) as pool:
                return dict(zip((w.query_id for w in ws), pool.map(ranker.order, ws)))

        versions = {"vllm": server_version(urllib_get, args.url)}
        counters = ranker

    payload = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {"rungs": {}}
    reference_path = REFERENCE_FILE[args.split]

    if args.split == "test":
        orderings = order_many(windows_)
        per_query = per_query_ndcg(orderings, windows_, qrels)
        entry = {"ndcg": _interval(bootstrap_ci(per_query)), "n_queries": len(per_query),
                 "device": device}
        if args.rung == REFERENCE_RUNG:
            reference_path.write_text(json.dumps(per_query))
        else:
            reference = json.loads(reference_path.read_text())
            entry["delta_vs_reference"] = _interval(paired_delta_ci(per_query, reference))
        payload.setdefault("test_quality", {})[args.rung] = entry
    else:
        latency, per_query, qps = run_rung(order_one, order_many, windows_,
                                           warm_windows[:WARMUP_QUERIES], qrels,
                                           runtime=runtime, precision=precision)
        record = {
            "runtime": runtime, "precision": precision, "model_dir": args.model_dir,
            "kind": args.kind, "server_flags": args.server_flags, "versions": versions,
            "device": device,
            "latency": latency.to_dict(), "throughput_qps": qps,
            "ndcg": _interval(bootstrap_ci(per_query)), "n_queries": len(per_query),
        }
        if args.rung == REFERENCE_RUNG:
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_text(json.dumps(per_query))
            record["delta_vs_reference"], record["lossless"] = None, True
        else:
            if not reference_path.exists():
                raise SystemExit(f"run --rung {REFERENCE_RUNG} first")
            reference = json.loads(reference_path.read_text())
            delta = _interval(paired_delta_ci(per_query, reference))
            if args.rung == AGREEMENT_RUNG:
                check_agreement(delta)
            record["delta_vs_reference"], record["lossless"] = delta, lossless(delta)
        if counters:
            record["imputed_answers"] = counters.imputed
            record["malformed_answers"] = len(counters.malformed)
        payload["rungs"][args.rung] = record
        if "vllm_bf16_prefix" in payload["rungs"]:
            payload["deployed_rung"] = deployed_rung(payload["rungs"])
        print(f"{args.rung}: NDCG {record['ndcg']['point']:.4f}, p50 {latency.p50_ms:.1f} ms, "
              f"p95 {latency.p95_ms:.1f} ms, {qps:.1f} queries/s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
