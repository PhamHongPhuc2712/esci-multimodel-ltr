"""A dense retriever fine-tuned on ESCI, for the recall stage.

Plan 4's dense channel is `all-MiniLM-L12-v2` off the shelf: it never saw an
ESCI query. Fine-tuning it on the training folds - each query against its
Exact products, every other product in the batch a negative - is the standard
lever for retrieval, and measured on validation fold 1 (2026-10-01, full
1,215,854-product catalogue) it is the largest one this project has:

    dense Recall@100       off the shelf 0.4644
                           MiniLM fine-tuned 0.5365, bge-small 0.5436, bge-base 0.5626
                           bge-base reading 256 tokens rather than 128: 0.5680

The product text averages ~440 tokens, so 256 reads twice as much of it. It
won on fold 1 by a little (0.5680 against 0.5626 alone; 0.6211 against 0.6197
fused) at twice the training and embedding time, and fold 1 is where this is
chosen, so 256 is the default.

Two choices were measured and dropped:

  * **Hard negatives.** ESCI's own Irrelevant and Complement judgements are
    natural hard negatives - Amazon's search retrieved them and a human said
    no - but adding one per pair tied in-batch negatives (MiniLM 0.5345 against
    0.5365, bge-small 0.5424 against 0.5436) at three times the training cost.
  * **Plain MultipleNegativesRankingLoss at batch 256 for a base-size model.**
    It does not fit 16 GB without spilling. The cached form (GradCache)
    computes the same gradients in mini-batches, so the batch - and with it
    the number of in-batch negatives - stays at 256.

**The retriever must never see fold 0, fold 1 or the test split.** Fold 1
chooses the fusion settings and fold 0 is reported; a retriever trained on
either is scored on its own training queries. `training_pairs` refuses them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from src.cross_encoder import check_training_folds
from src.ranker import TRAIN_FOLDS

DEFAULT_BASE = "BAAI/bge-base-en-v1.5"
DEFAULT_OUT = Path("models/retriever/bge-base")
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_MAX_SEQ = 256
# GradCache mini-batch: bounds memory, never changes the gradient.
DEFAULT_MINI_BATCH = 32

# Written beside every retriever this module trains.
RETRIEVER_RECORD = "training.json"


def training_pairs(
    judgements: pd.DataFrame,
    doc_text: Mapping[str, str],
    *,
    folds: Sequence[int] = TRAIN_FOLDS,
) -> dict[str, list[str]]:
    """(query, Exact product text) pairs from the training folds, in a stable order.

    `judgements` needs query_id, query, product_id, esci_label and fold. Only
    Exact judgements become pairs: recall is measured on Exact products, and a
    Substitute as a positive would teach the retriever that close is enough.
    """
    check_training_folds(judgements, allowed=folds)
    exact = judgements.loc[judgements["esci_label"] == "E"].sort_values(
        ["query_id", "product_id"], kind="stable"
    )
    if exact.empty:
        raise ValueError("no Exact judgements to train on")
    positives = []
    for product_id in exact["product_id"]:
        text = doc_text.get(product_id)
        if not isinstance(text, str) or not text.strip():
            raise KeyError(
                f"no document text for {product_id!r}; training on an empty "
                "string would teach the retriever that this query matches nothing"
            )
        positives.append(text)
    return {"anchor": [str(q) for q in exact["query"]], "positive": positives}


def retriever_record(
    *,
    base: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq: int,
    mini_batch_size: int,
    seed: int,
    n_pairs: int,
    n_queries: int,
    minutes: float,
) -> dict:
    """What trained a retriever, written beside it."""
    return {
        "base": base,
        "loss": "CachedMultipleNegativesRankingLoss",
        "positives": "Exact",
        "negatives": "in-batch",
        "folds": list(TRAIN_FOLDS),
        "epochs": epochs,
        "batch_size": batch_size,
        "mini_batch_size": mini_batch_size,
        "learning_rate": learning_rate,
        "max_seq": max_seq,
        "seed": seed,
        "n_pairs": n_pairs,
        "n_queries": n_queries,
        "minutes": minutes,
    }


def fine_tune_retriever(
    pairs: Mapping[str, list[str]],
    *,
    base: str = DEFAULT_BASE,
    out_dir: Path = DEFAULT_OUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_seq: int = DEFAULT_MAX_SEQ,
    mini_batch_size: int = DEFAULT_MINI_BATCH,
    device: str = "auto",
    seed: int = 0,
) -> Path:
    """Fine-tune with in-batch negatives and save to `out_dir`."""
    import tempfile

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    from src.clip_encoder import resolve_device

    if not pairs.get("anchor"):
        raise ValueError("an empty training set; nothing to fine-tune on")
    resolved = resolve_device(device)
    model = SentenceTransformer(base, device=resolved)
    model.max_seq_length = max_seq
    with tempfile.TemporaryDirectory() as scratch:
        arguments = SentenceTransformerTrainingArguments(
            output_dir=scratch,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            fp16=resolved.startswith("cuda"),
            # A product that is Exact for two queries in one batch would be
            # its own negative; this sampler keeps every text once per batch.
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="no",
            logging_steps=100,
            report_to=[],
            seed=seed,
            disable_tqdm=True,
        )
        SentenceTransformerTrainer(
            model=model,
            args=arguments,
            train_dataset=Dataset.from_dict(dict(pairs)),
            loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size),
        ).train()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    return out_dir


def _main() -> int:
    import time

    from src.baseline_sbert import product_text
    from src.dense_embed import DEFAULT_FIELDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--mini-batch", type=int, default=DEFAULT_MINI_BATCH)
    parser.add_argument("--judgements", type=Path, default=Path("data/combined/judgements.parquet"))
    parser.add_argument("--products", type=Path, default=Path("data/combined/products.parquet"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    judgements = pd.read_parquet(
        args.judgements,
        columns=["query_id", "query", "product_id", "esci_label", "split", "fold"],
    )
    judgements = judgements.loc[
        (judgements["split"] == "train") & judgements["fold"].isin(TRAIN_FOLDS)
    ].reset_index(drop=True)
    products = pd.read_parquet(args.products, columns=["product_id", *DEFAULT_FIELDS])
    products = products.loc[
        products["product_id"].isin(set(judgements["product_id"]))
    ].drop_duplicates("product_id")
    # The same text the corpus is embedded from, so training and serving agree.
    doc_text = dict(zip(products["product_id"], product_text(products, list(DEFAULT_FIELDS))))

    pairs = training_pairs(judgements, doc_text)
    n_queries = int(judgements.loc[judgements["esci_label"] == "E", "query_id"].nunique())
    print(f"{len(pairs['anchor']):,} (query, Exact product) pairs over {n_queries:,} "
          f"queries from folds {list(TRAIN_FOLDS)}; base {args.base}")
    started = time.time()
    out = fine_tune_retriever(
        pairs, base=args.base, out_dir=args.out, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_seq=args.max_seq, mini_batch_size=args.mini_batch,
        device=args.device, seed=args.seed,
    )
    minutes = (time.time() - started) / 60
    record = retriever_record(
        base=args.base, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, max_seq=args.max_seq,
        mini_batch_size=args.mini_batch, seed=args.seed,
        n_pairs=len(pairs["anchor"]), n_queries=n_queries, minutes=minutes,
    )
    (out / RETRIEVER_RECORD).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"trained in {minutes:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
