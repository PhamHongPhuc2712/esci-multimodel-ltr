"""Plan 13: a small open LLM as a pointwise reranker, scored by P("yes").

Qwen3-Reranker is a causal LM trained to answer "yes" or "no" after a fixed
template. Its score for a pair is the softmax over those two next-token
logits, which is a sigmoid of their difference. Task 4 trains on that same
difference, so the training objective and the evaluation score agree.

Two properties the rest of Plan 13 depends on:

  * **The query precedes the document.** Every candidate of a query shares
    every token up to the document, so a server with prefix caching computes
    that part once per query (Phase 3).
  * **One encoding.** `encode_pair` is the only place a pair becomes token
    ids. Training, in-process scoring and the vLLM client all call it, so the
    three cannot drift apart through a template edit or a tokenizer boundary.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence

from pathlib import Path

from src.cross_encoder import _lookup

DEFAULT_BACKBONE = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_MAX_LENGTH = 256

# The instruction is part of the template the model was trained with; the
# wording here is ESCI's definition of an Exact match.
INSTRUCTION = (
    "Given a shopping search query, judge whether the product satisfies "
    "every specification in the query"
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def query_prefix(query: str) -> str:
    """Everything before the document. Identical for all candidates of a query."""
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: "


def encode_pair(
    tokenizer, query: str, document: str, max_length: int = DEFAULT_MAX_LENGTH
) -> tuple[list[int], bool]:
    """Token ids for one pair, and whether the document was cut to fit.

    Only the document is truncated. The prefix carries the query and the
    suffix carries the answer position; cutting either would score a
    different question.
    """
    prefix = tokenizer(query_prefix(query), add_special_tokens=False).input_ids
    suffix = tokenizer(SUFFIX, add_special_tokens=False).input_ids
    budget = max_length - len(prefix) - len(suffix)
    if budget <= 0:
        raise ValueError(
            f"the template and query take {len(prefix) + len(suffix)} tokens, "
            f"leaving nothing of max_length={max_length} for the document"
        )
    body = tokenizer(document, add_special_tokens=False).input_ids
    truncated = len(body) > budget
    return list(prefix) + list(body[:budget]) + list(suffix), truncated


def yes_no_ids(tokenizer) -> tuple[int, int]:
    """The single-token ids of "yes" and "no". Refuses a tokenizer without them."""
    ids = []
    for word in ("yes", "no"):
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"{word!r} is not a single token in this tokenizer; the "
                "P(yes) score needs both answers as one token each"
            )
        ids.append(int(token_id))
    return ids[0], ids[1]


def collate_left(
    ids: Sequence[Sequence[int]], pad_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad to the longest row, so the last column is a real token everywhere."""
    width = max(len(row) for row in ids)
    padded = [[pad_id] * (width - len(row)) + list(row) for row in ids]
    mask = [[0] * (width - len(row)) + [1] * len(row) for row in ids]
    return padded, mask


def p_yes(logit_no: float, logit_yes: float) -> float:
    """softmax([no, yes])[1], written as a sigmoid that cannot overflow."""
    z = logit_yes - logit_no
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def order_by_scores(window: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Best first. Ties break on document id, as src.cross_encoder.rerank does."""
    if len(window) != len(scores):
        raise ValueError(f"{len(scores)} scores for {len(window)} documents")
    by_doc = dict(zip(window, scores))
    return sorted(window, key=lambda d: (-by_doc[d], d))


def score_pairs(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 32,
    device: str = "cuda",
) -> list[float]:
    """P("yes") for every (query, document) pair."""
    import torch

    yes, no = yes_no_ids(tokenizer)
    pad = tokenizer.pad_token_id
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = [
                encode_pair(tokenizer, q, d, max_length)[0]
                for q, d in pairs[start : start + batch_size]
            ]
            ids, mask = collate_left(chunk, pad)
            logits = model(
                input_ids=torch.tensor(ids, device=device),
                attention_mask=torch.tensor(mask, device=device),
                logits_to_keep=1,
            ).logits[:, -1, :].float()
            for row in logits:
                scores.append(p_yes(float(row[no]), float(row[yes])))
    return scores


def rerank_point(
    model,
    tokenizer,
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, list[str]]:
    """A new ordering of each window, best first."""
    pairs: list[tuple[str, str]] = []
    for w in windows_:
        query = _lookup(query_text, w.query_id, "query text")
        pairs.extend((query, _lookup(doc_text, d, "document text")) for d in w.window)
    scores = score_pairs(
        model, tokenizer, pairs, max_length=max_length,
        batch_size=batch_size, device=device,
    )
    out: dict[str, list[str]] = {}
    position = 0
    for w in windows_:
        n = len(w.window)
        out[w.query_id] = order_by_scores(w.window, scores[position : position + n])
        position += n
    return out


def load_point_model(path: str, *, device: str = "cuda", dtype: str = "bfloat16"):
    """A causal LM and its left-padding tokenizer, for scoring."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=getattr(torch, dtype)
    ).to(device)
    return model, tokenizer


# --- the fine-tune --------------------------------------------------------------

FULL_LR = 1e-5
LORA_LR = 1e-4
LOG_EVERY = 200


def point_targets(gains: Sequence[float]) -> list[float]:
    """ESCI's gains as soft targets for P("yes"). They must already lie in [0, 1]."""
    targets = [float(g) for g in gains]
    bad = [g for g in targets if not 0.0 <= g <= 1.0]
    if bad:
        raise ValueError(
            f"gain {bad[0]} is outside [0, 1]; pass the 1.0/0.1/0.01/0.0 gains, "
            "not the integer qrels"
        )
    return targets


def iterate_batches(n: int, batch_size: int, seed: int) -> list[list[int]]:
    """Row indices for one epoch, shuffled by `seed`, in batches."""
    import numpy as np

    order = np.random.default_rng(seed).permutation(n)
    return [order[i : i + batch_size].tolist() for i in range(0, n, batch_size)]


def point_record(
    *,
    backbone: str,
    lora_rank: int | None,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    learning_rate: float,
    max_length: int,
    n_queries: int,
    n_pairs: int,
    updates: int,
    minutes: float,
    peak_reserved_gb: float,
    seed: int,
) -> dict:
    """What trained a pointwise arm, and what a full epoch would take at this pace."""
    pairs_per_update = batch_size * grad_accum
    updates_per_second = updates / (minutes * 60) if minutes > 0 else 0.0
    epoch_updates = math.ceil(n_pairs / pairs_per_update)
    return {
        "style": "pointwise",
        "backbone": backbone,
        "lora_rank": lora_rank,
        "loss": "BCEWithLogits on logit(yes) - logit(no), gain as soft target",
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "pairs_per_update": pairs_per_update,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "n_queries": n_queries,
        "n_pairs": n_pairs,
        "updates": updates,
        "minutes": minutes,
        "updates_per_second": updates_per_second,
        "projected_epoch_hours": (
            epoch_updates / updates_per_second / 3600 if updates_per_second else None
        ),
        "peak_reserved_gb": peak_reserved_gb,
        "seed": seed,
    }


def fine_tune_point(
    matrix,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    out_dir,
    backbone: str = DEFAULT_BACKBONE,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    grad_accum: int = 1,
    epochs: int = 1,
    learning_rate: float | None = None,
    lora_rank: int | None = None,
    seed: int = 0,
    max_updates: int | None = None,
    device: str = "cuda",
) -> dict:
    """Train on folds 2/3/4 and save. With `max_updates`, a probe that saves nothing."""
    from src.cross_encoder import check_training_folds, configure_allocator, pairwise_dataset

    check_training_folds(matrix)
    configure_allocator()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    lr = learning_rate if learning_rate is not None else (LORA_LR if lora_rank else FULL_LR)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(backbone, padding_side="left")
    yes, no = yes_no_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        backbone, dtype=torch.bfloat16 if lora_rank else torch.float32
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if lora_rank:
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        ))

    data = pairwise_dataset(matrix, query_text, doc_text)
    targets = point_targets(data["label"])
    batches = [
        b for epoch in range(epochs) for b in iterate_batches(len(targets), batch_size, seed + epoch)
    ]
    total_updates = math.ceil(len(batches) / grad_accum)
    if max_updates is not None:
        total_updates = min(total_updates, max_updates)
        batches = batches[: total_updates * grad_accum]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(0.05 * total_updates)), total_updates
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def update() -> None:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.time()
    running, updates = 0.0, 0
    for step, batch in enumerate(batches):
        ids, mask = collate_left(
            [encode_pair(tok, data["query"][i], data["doc"][i], max_length)[0] for i in batch],
            tok.pad_token_id,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=torch.tensor(ids, device=device),
                attention_mask=torch.tensor(mask, device=device),
                logits_to_keep=1,
            ).logits[:, -1, :]
        z = (logits[:, yes] - logits[:, no]).float()
        loss = loss_fn(z, torch.tensor([targets[i] for i in batch], device=device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        (loss / grad_accum).backward()
        running += float(loss.detach())
        if (step + 1) % grad_accum == 0:
            update()
            updates += 1
            if updates % LOG_EVERY == 0:
                print(f"  update {updates:,}/{total_updates:,}  "
                      f"loss {running / (LOG_EVERY * grad_accum):.4f}", flush=True)
                running = 0.0
    if len(batches) % grad_accum:
        update()
        updates += 1

    record = point_record(
        backbone=backbone, lora_rank=lora_rank, batch_size=batch_size,
        grad_accum=grad_accum, epochs=epochs, learning_rate=lr, max_length=max_length,
        n_queries=int(matrix["query_id"].nunique()), n_pairs=len(targets),
        updates=updates, minutes=(time.time() - started) / 60,
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30, seed=seed,
    )
    if max_updates is not None:
        return record

    if lora_rank:
        model = model.merge_and_unload()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(target)
    tok.save_pretrained(target)
    (target / "training.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _train_command(args) -> int:
    import pandas as pd

    from src.cross_encoder import check_training_folds, text_maps
    from src.ranker import TRAIN_FOLDS

    matrix = pd.read_parquet("data/features/train.parquet",
                             columns=["query_id", "product_id", "gain", "fold"])
    matrix = matrix.loc[matrix["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    check_training_folds(matrix)
    query_text, doc_text = text_maps(
        matrix, Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    record = fine_tune_point(
        matrix, query_text, doc_text, out_dir=args.out, backbone=args.backbone,
        max_length=args.max_length, batch_size=args.batch_size,
        grad_accum=args.grad_accum, epochs=args.epochs, lora_rank=args.lora_rank,
        max_updates=args.probe,
    )
    print(json.dumps(record, indent=2))
    return 0


def truncation_share(tokenizer, pairs: Sequence[tuple[str, str]], max_length: int) -> float:
    """The share of pairs whose document `encode_pair` had to cut."""
    if not pairs:
        raise ValueError("no pairs")
    cut = sum(encode_pair(tokenizer, q, d, max_length)[1] for q, d in pairs)
    return cut / len(pairs)


def _score_command(args) -> int:
    from pathlib import Path

    import pandas as pd

    from src.blend_report import per_query_ndcg
    from src.cross_encoder import text_maps
    from src.latency_profile import time_single_queries
    from src.rank_report import qrels_from_frame
    from src.stage_signals import scope_windows

    windows_, matrix = scope_windows("fold0")
    warmup_windows, warmup_matrix = warmup_scope()
    both = pd.concat([matrix, warmup_matrix], ignore_index=True)
    query_text, doc_text = text_maps(
        both, Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}
    qrels = qrels_from_frame(matrix)

    model, tok = load_point_model(args.model)
    pairs = [(query_text[w.query_id], doc_text[d]) for w in windows_ for d in w.window]
    share = truncation_share(tok, pairs, args.max_length)
    orderings = rerank_point(model, tok, windows_, query_text, doc_text,
                             max_length=args.max_length, batch_size=args.batch_size)
    per_query = per_query_ndcg(orderings, windows_, qrels)
    ndcg = sum(per_query.values()) / len(per_query)

    def one(w):
        rerank_point(model, tok, [w], query_text, doc_text,
                     max_length=args.max_length, batch_size=len(w.window))

    latency = time_single_queries(one, windows_[:200], warmup=warmup_windows[:20],
                                  runtime="hf", precision="bf16")
    print(f"{args.model}: fold-0 NDCG {ndcg:.4f} over {len(per_query):,} queries")
    print(f"  {share:.1%} of {len(pairs):,} pairs truncated at {args.max_length} tokens")
    print(f"  single query: mean {latency.mean_ms:.1f} ms, p50 {latency.p50_ms:.1f}, "
          f"p95 {latency.p95_ms:.1f}")
    return 0


def warmup_scope() -> tuple[list, "pd.DataFrame"]:
    """Fold-1 windows and their rows, for warm-up only: never scored, never timed.

    Fold 1 is the early-stop fold. Nothing in Plan 13 is selected on it, so
    running a model over a few of its queries to warm the GPU touches no
    surface any number is reported on.
    """
    import pandas as pd

    from src.ranker import EARLY_STOP_FOLD
    from src.rerank_window import windows
    from src.stage2_scores import load_stage2

    matrix = pd.read_parquet("data/features/train.parquet")
    matrix = matrix.loc[matrix["fold"] == EARLY_STOP_FOLD].reset_index(drop=True)
    joined = matrix[["query_id", "product_id"]].merge(
        load_stage2("train"), on=["query_id", "product_id"]
    )
    return windows(joined), matrix


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="fold-0 NDCG, truncation and latency for one model")
    score.add_argument("--model", default=DEFAULT_BACKBONE)
    score.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    score.add_argument("--batch-size", type=int, default=32)
    train = sub.add_parser("train", help="fine-tune on folds 2/3/4, or probe with --probe N")
    train.add_argument("--out", default="models/llm-rank/point-0.6b")
    train.add_argument("--backbone", default=DEFAULT_BACKBONE)
    train.add_argument("--lora-rank", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--grad-accum", type=int, default=1)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train.add_argument("--probe", type=int, default=None,
                       help="stop after N updates, report memory and pace, save nothing")
    args = parser.parse_args()
    if args.command == "train":
        return _train_command(args)
    if args.command == "score":
        return _score_command(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
