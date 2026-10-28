"""Plan 13: a small open LLM as a listwise reranker of the Stage 2 window.

The hosted LLM of Plan 6 answers with a permutation after ~340 tokens of
reasoning. This arm learns to emit the permutation directly: Qwen3-1.7B with
a LoRA adapter, trained on the label-sorted order of Plan 8's out-of-fold
windows, thinking disabled. Its answer is parsed by the same exact-permutation
reader as the hosted arm; anything else keeps the Stage 2 order and is
counted, and more than 0.5% of windows refuses the run.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from src.cross_encoder import _lookup
from src.distill import gains_from_frame, training_windows
from src.llm_point import collate_left, iterate_batches
from src.llm_rerank import parse_permutation

DEFAULT_BACKBONE = "Qwen/Qwen3-1.7B"
DOC_CHARS = 300
MAX_NEW_TOKENS = 64
MAX_MALFORMED_SHARE = 0.005
END_OF_TURN = "<|im_end|>"

SYSTEM = "You rank products for a shopping search engine."
INSTRUCTION = (
    "Rank the {n} products by how well they satisfy the search query: exact "
    "matches (every specification satisfied) first, then substitutes, then "
    "complements, then irrelevant products.\n\nQuery: {query}\n\n{listing}\n\n"
    "Reply with all {n} identifiers in descending relevance, like [3] > [1] > [2]."
)


def user_message(query: str, docs: Sequence[str]) -> str:
    listing = "\n".join(f"[{i}] {str(d)[:DOC_CHARS]}" for i, d in enumerate(docs, start=1))
    return INSTRUCTION.format(n=len(docs), query=query, listing=listing)


def target_permutation(gains: Sequence[float]) -> list[int]:
    """1-based, highest gain first, ties kept in window (Stage 2) order."""
    return [i + 1 for i in sorted(range(len(gains)), key=lambda i: (-gains[i], i))]


def answer_text(permutation: Sequence[int]) -> str:
    return " > ".join(f"[{i}]" for i in permutation)


def encode_prompt(tokenizer, query: str, docs: Sequence[str]) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_message(query, docs)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return list(tokenizer(text, add_special_tokens=False).input_ids)


def encode_example(tokenizer, query: str, docs: Sequence[str], gains: Sequence[float]):
    """Prompt and answer ids, with the prompt masked out of the loss."""
    prompt = encode_prompt(tokenizer, query, docs)
    answer = list(tokenizer(
        answer_text(target_permutation(gains)) + " " + END_OF_TURN,
        add_special_tokens=False,
    ).input_ids)
    return prompt + answer, [-100] * len(prompt) + answer


def collate_right(examples, pad_id: int):
    """Right-pad ids, labels and mask; padding never enters the loss."""
    width = max(len(ids) for ids, _ in examples)
    ids_out, labels_out, mask_out = [], [], []
    for ids, labels in examples:
        pad = width - len(ids)
        ids_out.append(list(ids) + [pad_id] * pad)
        labels_out.append(list(labels) + [-100] * pad)
        mask_out.append([1] * len(ids) + [0] * pad)
    return ids_out, labels_out, mask_out


def list_training_set(
    query_text: Mapping, doc_text: Mapping, *, windows_fn: Callable = training_windows
) -> list[tuple[str, list[str], list[float]]]:
    """(query, documents, gains) for every out-of-fold training window."""
    from src.cross_encoder import check_training_folds

    windows_, matrix = windows_fn()
    check_training_folds(matrix)
    gains = gains_from_frame(matrix)
    out = []
    for w in windows_:
        out.append((
            _lookup(query_text, w.query_id, "query text"),
            [_lookup(doc_text, d, "document text") for d in w.window],
            [gains[(w.query_id, d)] for d in w.window],
        ))
    return out


def check_malformed_share(malformed: Sequence[str], n_windows: int) -> None:
    if n_windows <= 0:
        raise ValueError("no windows")
    share = len(malformed) / n_windows
    if share > MAX_MALFORMED_SHARE:
        raise ValueError(
            f"{len(malformed):,} of {n_windows:,} answers ({share:.2%}) were malformed, "
            f"past {MAX_MALFORMED_SHARE:.1%}; the Stage 2 fallback would be scored "
            "as the model's ranking"
        )


def generate_orderings(
    model, tokenizer, windows_: Sequence, query_text: Mapping, doc_text: Mapping,
    *, batch_size: int = 8, max_new_tokens: int = MAX_NEW_TOKENS, device: str = "cuda",
) -> tuple[dict[str, list[str]], list[str]]:
    """Greedy permutations per window, and the query ids whose answer was malformed."""
    import torch

    out: dict[str, list[str]] = {}
    malformed: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows_), batch_size):
            chunk = list(windows_[start : start + batch_size])
            prompts = [
                encode_prompt(tokenizer, _lookup(query_text, w.query_id, "query text"),
                              [_lookup(doc_text, d, "document text") for d in w.window])
                for w in chunk
            ]
            ids, mask = collate_left(prompts, tokenizer.pad_token_id)
            ids_t = torch.tensor(ids, device=device)
            generated = model.generate(
                input_ids=ids_t, attention_mask=torch.tensor(mask, device=device),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            for w, row in zip(chunk, generated[:, ids_t.shape[1]:]):
                text = tokenizer.decode(row, skip_special_tokens=True)
                permutation = parse_permutation(text, len(w.window))
                if permutation is None:
                    malformed.append(w.query_id)
                    out[w.query_id] = list(w.window)
                else:
                    out[w.query_id] = [w.window[i - 1] for i in permutation]
    return out, malformed


def fine_tune_list(
    examples: Sequence[tuple[str, list[str], list[float]]],
    *,
    out_dir,
    backbone: str = DEFAULT_BACKBONE,
    lora_rank: int = 16,
    batch_size: int = 4,
    grad_accum: int = 4,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 0,
    max_updates: int | None = None,
    device: str = "cuda",
    gradient_checkpointing: bool = True,
) -> dict:
    """LoRA SFT on the answer tokens. With `max_updates`, a probe that saves nothing."""
    from src.cross_encoder import configure_allocator

    configure_allocator()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(backbone)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(backbone, dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    ))

    encoded = [encode_example(tok, q, d, g) for q, d, g in examples]
    batches = [b for e in range(epochs) for b in iterate_batches(len(encoded), batch_size, seed + e)]
    total_updates = math.ceil(len(batches) / grad_accum)
    if max_updates is not None:
        total_updates = min(total_updates, max_updates)
        batches = batches[: total_updates * grad_accum]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(0.05 * total_updates)), total_updates
    )

    def update() -> None:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    started, updates, running = time.time(), 0, 0.0
    for step, batch in enumerate(batches):
        ids, labels, mask = collate_right([encoded[i] for i in batch], tok.pad_token_id)
        loss = model(
            input_ids=torch.tensor(ids, device=device),
            attention_mask=torch.tensor(mask, device=device),
            labels=torch.tensor(labels, device=device),
        ).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        (loss / grad_accum).backward()
        running += float(loss.detach())
        if (step + 1) % grad_accum == 0:
            update()
            updates += 1
            if updates % 50 == 0:
                print(f"  update {updates:,}/{total_updates:,}  loss {running / (50 * grad_accum):.4f}",
                      flush=True)
                running = 0.0
    if len(batches) % grad_accum:
        update()
        updates += 1

    minutes = (time.time() - started) / 60
    per_second = updates / (minutes * 60) if minutes > 0 else 0.0
    epoch_updates = math.ceil(len(encoded) / (batch_size * grad_accum))
    record = {
        "style": "listwise", "backbone": backbone, "lora_rank": lora_rank,
        "loss": "causal LM cross-entropy on the answer tokens",
        "batch_size": batch_size, "grad_accum": grad_accum,
        "windows_per_update": batch_size * grad_accum, "epochs": epochs,
        "learning_rate": learning_rate, "doc_chars": DOC_CHARS,
        "n_windows": len(encoded), "updates": updates, "minutes": minutes,
        "updates_per_second": per_second,
        "projected_epoch_hours": epoch_updates / per_second / 3600 if per_second else None,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "gradient_checkpointing": gradient_checkpointing,
        "device_name": torch.cuda.get_device_name(), "seed": seed,
    }
    if max_updates is not None:
        return record
    model = model.merge_and_unload()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tok.save_pretrained(target)
    (target / "training.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_list_model(path: str, *, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
    return model, tok


def _main() -> int:
    import argparse

    import pandas as pd

    from src.cross_encoder import text_maps
    from src.ranker import TRAIN_FOLDS

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--out", default="models/llm-rank/list-1.7b")
    train.add_argument("--backbone", default=DEFAULT_BACKBONE)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--probe", type=int, default=None)
    train.add_argument("--no-gradient-checkpointing", action="store_true",
                       help="keep activations: faster, for a card with memory to spare")
    args = parser.parse_args()

    matrix = pd.read_parquet("data/features/train.parquet",
                             columns=["query_id", "product_id", "gain", "fold"])
    matrix = matrix.loc[matrix["fold"].isin(TRAIN_FOLDS)]
    query_text, doc_text = text_maps(
        matrix, Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}
    examples = list_training_set(query_text, doc_text)
    record = fine_tune_list(
        examples, out_dir=args.out, backbone=args.backbone,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        epochs=args.epochs, max_updates=args.probe,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
