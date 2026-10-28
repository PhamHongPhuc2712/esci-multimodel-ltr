"""Plan 13: every Stage 3 arm on fold 0 or the test split, with latency profiles.

The reference arms are read from the signal frame and must reproduce Plan 6.
The two bge arms are re-scored and must reproduce Plan 11, then timed in fp32
and fp16. The LLM arms are scored and timed in bf16. Selection happens on
fold 0; `--split test` requires `--final`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

REFERENCE_ARMS: tuple[str, ...] = ("stage2", "stage2+ce", "stage2+llm")
CE_ARMS: dict[str, Path] = {
    "stage2+ce_bge_groups": Path("models/cross-encoder/bge-groups/lambda"),
    "stage2+ce_bge_large": Path("models/cross-encoder/bge-large-groups/lambda"),
}
LLM_ARMS: dict[str, tuple[str, str]] = {
    "stage2+qwen_point_zs": ("point", "Qwen/Qwen3-Reranker-0.6B"),
    "stage2+qwen_point": ("point", "models/llm-rank/point-0.6b"),
    "stage2+qwen_list": ("list", "models/llm-rank/list-1.7b"),
    "stage2+qwen_point_4b": ("point", "models/llm-rank/point-4b"),
}
OPTIONAL_ARMS: tuple[str, ...] = ("stage2+qwen_point_4b",)
TRAINED_ARMS: tuple[str, ...] = ("stage2+qwen_point", "stage2+qwen_list", "stage2+qwen_point_4b")
SIZE_REFERENCE = {"train": Path("docs/results/size.json"), "test": Path("docs/results/size-test.json")}
HOSTED_LATENCY_SOURCE = Path("docs/results/fine-rank-test.json")
OUT = {"train": Path("docs/results/llm-rank.json"), "test": Path("docs/results/llm-rank-test.json")}
LATENCY_QUERIES = 200
WARMUP_QUERIES = 20


def _is_local(path: str) -> bool:
    return path.startswith("models/")


def available_arms(root: Path = Path(".")) -> list[str]:
    """LLM arms to score: hub models always, trained ones if their directory exists."""
    arms = []
    for arm, (_, path) in LLM_ARMS.items():
        if not _is_local(path) or (Path(root) / path / "config.json").exists():
            arms.append(arm)
        elif arm not in OPTIONAL_ARMS:
            raise FileNotFoundError(f"{arm} has no model at {path}; train it first (Task 7)")
    return arms


def check_size_reference(
    measured: Mapping[str, float], published: Mapping[str, float], tol: float = 2e-4
) -> None:
    for arm, value in published.items():
        if abs(measured[arm] - value) > tol:
            raise ValueError(
                f"{arm} measures {measured[arm]:.4f} against Plan 11's {value:.4f}; "
                "the windows or the model have drifted"
            )


def check_both_precisions(latency: Mapping[str, Mapping[str, dict]]) -> None:
    """Review Focus 1: a cross-encoder timed in fp32 only would flatter the LLM."""
    for arm in CE_ARMS:
        missing = {"fp32", "fp16"} - set(latency.get(arm, {}))
        if missing:
            raise ValueError(f"{arm} was not timed in {sorted(missing)}")


def comparison_pairs(arms: Sequence[str]) -> list[tuple[str, str]]:
    present = set(arms)
    pairs = []
    for arm in LLM_ARMS:
        if arm not in present:
            continue
        for baseline in ("stage2", "stage2+ce_bge_large", "stage2+llm"):
            if baseline in present:
                pairs.append((arm, baseline))
    for arm, baseline in (("stage2+qwen_point", "stage2+qwen_point_zs"),
                          ("stage2+qwen_list", "stage2+qwen_point"),
                          ("stage2+qwen_point_4b", "stage2+qwen_point")):
        if arm in present and baseline in present:
            pairs.append((arm, baseline))
    return pairs


def best_llm_arm(ndcg: Mapping[str, float]) -> str:
    trained = {arm: v for arm, v in ndcg.items() if arm in TRAINED_ARMS}
    if not trained:
        raise ValueError("no trained LLM arm was scored")
    return max(trained, key=trained.get)


def require_final(split: str, final: bool) -> None:
    if split == "test" and not final:
        raise SystemExit("refusing the test split without --final; every choice is made on fold 0")


def _published_size(split: str) -> dict[str, float]:
    payload = json.loads(SIZE_REFERENCE[split].read_text(encoding="utf-8"))
    return {a["name"]: a["ndcg"]["point"] for a in payload["arms"] if a["name"] in CE_ARMS}


def _hosted_latency() -> dict:
    payload = json.loads(HOSTED_LATENCY_SOURCE.read_text(encoding="utf-8"))
    arm = next(a for a in payload["arms"] if a["name"] == "stage2+llm")
    return {"single_ms": arm["cost"]["latency"]["single_ms"],
            "batched_ms": arm["cost"]["latency"]["batched_ms"],
            "runtime": "hosted API", "precision": "unknown",
            "source": str(HOSTED_LATENCY_SOURCE)}


def _main() -> int:
    import pandas as pd
    import torch
    import transformers

    from src.blend_report import per_query_ndcg, single_stage_orderings
    from src.cross_encoder import load_reranker, rerank, text_maps
    from src.distill_report import _evaluate, _published, _scope, check_reference
    from src.floor import random_floor
    from src.latency_profile import time_single_queries
    from src.llm_list import check_malformed_share, generate_orderings, load_list_model
    from src.llm_point import DEFAULT_MAX_LENGTH, load_point_model, rerank_point, warmup_scope
    from src.rank_report import compare, qrels_from_frame

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    require_final(args.split, args.final)

    _, windows_, matrix, frame, query_text, doc_text = _scope(args.split, args.seed)
    warm_windows, warm_matrix = warmup_scope()
    warm_q, warm_d = text_maps(warm_matrix, Path("data/combined/products.parquet"),
                               Path("data/combined/judgements.parquet"))
    query_text.update({str(k): v for k, v in warm_q.items()})
    doc_text.update(warm_d)
    warm = warm_windows[:WARMUP_QUERIES]
    timed = windows_[:LATENCY_QUERIES]
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=100, seed=args.seed)

    per_query: dict[str, dict[str, float]] = {}
    latency: dict[str, dict[str, dict]] = {}
    precision_checks: dict[str, dict] = {}
    malformed: dict[str, int] = {}

    singles = single_stage_orderings(frame)
    for arm in REFERENCE_ARMS:
        per_query[arm] = per_query_ndcg(singles[arm], windows_, qrels)
    check_reference({a: _mean(per_query[a]) for a in REFERENCE_ARMS}, _published(args.split))

    for arm, path in CE_ARMS.items():
        model = load_reranker(path)
        per_query[arm] = per_query_ndcg(rerank(model, windows_, query_text, doc_text), windows_, qrels)
        one = lambda w, m=model: rerank(m, [w], query_text, doc_text)
        latency[arm] = {"fp32": time_single_queries(one, timed, warmup=warm, runtime="hf",
                                                    precision="fp32").to_dict()}
        model.model.half()
        fp16 = per_query_ndcg(rerank(model, windows_, query_text, doc_text), windows_, qrels)
        latency[arm]["fp16"] = time_single_queries(one, timed, warmup=warm, runtime="hf",
                                                   precision="fp16").to_dict()
        precision_checks[arm] = compare(_evaluate(f"{arm}@fp16", fp16, floor, args.seed),
                                        _evaluate(arm, per_query[arm], floor, args.seed), args.seed)
        del model
        torch.cuda.empty_cache()
    check_size_reference({a: _mean(per_query[a]) for a in CE_ARMS}, _published_size(args.split))
    check_both_precisions(latency)

    training = {}
    for arm in available_arms():
        kind, path = LLM_ARMS[arm]
        if kind == "point":
            model, tok = load_point_model(path)
            orderings = rerank_point(model, tok, windows_, query_text, doc_text,
                                     max_length=args.max_length)
            one = lambda w, m=model, t=tok: rerank_point(m, t, [w], query_text, doc_text,
                                                         max_length=args.max_length,
                                                         batch_size=len(w.window))
        else:
            model, tok = load_list_model(path)
            orderings, bad = generate_orderings(model, tok, windows_, query_text, doc_text)
            check_malformed_share(bad, len(windows_))
            malformed[arm] = len(bad)
            one = lambda w, m=model, t=tok: generate_orderings(m, t, [w], query_text, doc_text,
                                                               batch_size=1)
        per_query[arm] = per_query_ndcg(orderings, windows_, qrels)
        latency[arm] = {"bf16": time_single_queries(one, timed, warmup=warm, runtime="hf",
                                                    precision="bf16").to_dict()}
        record = Path(path) / "training.json"
        if record.exists():
            training[arm] = json.loads(record.read_text(encoding="utf-8"))
        del model
        torch.cuda.empty_cache()

    results = {arm: _evaluate(arm, pq, floor, args.seed) for arm, pq in per_query.items()}
    comparisons = [compare(results[a], results[b], args.seed)
                   for a, b in comparison_pairs(list(results))]
    ndcg = {arm: r.ndcg.point for arm, r in results.items()}
    for arm, r in results.items():
        print(f"  {arm:24s} {r.ndcg.point:.4f} [{r.ndcg.low:.4f}, {r.ndcg.high:.4f}]")
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:24s} - {row['baseline']:22s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]")

    payload = {
        "split": args.split,
        "n_queries": len(qrels),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high},
        "max_length": args.max_length,
        "arms": [r.to_dict() | {"latency": latency.get(r.name)} for r in results.values()],
        "comparisons": comparisons,
        "precision_checks": precision_checks,
        "malformed_answers": malformed,
        "hosted_llm_latency": _hosted_latency(),
        "training": training,
        "models": {arm: LLM_ARMS[arm][1] for arm in available_arms()}
                  | {arm: str(p) for arm, p in CE_ARMS.items()},
        "best_llm_arm": best_llm_arm(ndcg),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "latency_queries": LATENCY_QUERIES,
        "seed": args.seed,
    }
    out = args.out or OUT[args.split]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"best_llm_arm: {payload['best_llm_arm']}  -> {out}")
    return 0


def _mean(per_query: Mapping[str, float]) -> float:
    return sum(per_query.values()) / len(per_query)


if __name__ == "__main__":
    raise SystemExit(_main())
