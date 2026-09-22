"""Stage 3(a): a cross-encoder fine-tuned on the ESCI train folds.

PROJECT_SPEC.md §4.3 asks for a fine-tuned modern cross-encoder, and §5's
0.8562 ESCI_baseline is exactly that - `ms-marco-MiniLM-L-12-v2` fine-tuned on
SQD train. The fine-tune is not an optimisation of this arm, it IS the arm:
measured zero-shot on a 400-query fold-0 sample where Stage 2 scores 0.8577,
every off-the-shelf reranker loses -

    ms-marco-MiniLM-L6-v2  title        0.8286     7 ms/query
    ms-marco-MiniLM-L6-v2  title+desc   0.8396    40 ms/query
    bge-reranker-base      title        0.8366    33 ms/query
    bge-reranker-base      title+desc   0.8445   205 ms/query

`LambdaLoss` is the default: it optimises a listwise objective against the same
graded labels the metric uses, and Plan 5's Ablation 5 measured listwise
beating pointwise by +0.0085 to +0.0096 on this data. Measured throughput on
the 3080 over 12,519 train-fold queries: 7.2 queries/s at batch 8, 6.22 GB
peak, 29 minutes an epoch. `BinaryCrossEntropyLoss` over pairs is the cheap
comparison at 393 pairs/s and 10.6 minutes.

**The model must never see fold 0 or fold 1.** data/features/train.parquet
holds all five folds; a fine-tune that reads it unfiltered puts the reporting
surface into a 22M-parameter model's weights and then scores on it.
`check_training_folds` is a hard guard, not a convention.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ranker import TRAIN_FOLDS

DEFAULT_BACKBONE = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_MODEL_DIR = Path("models/cross-encoder")
DEFAULT_MAX_LENGTH = 192

# ~4 characters a token, so 192 tokens is roughly 800 characters. Cutting in
# Python first keeps the tokeniser off text it would throw away anyway.
DEFAULT_MAX_CHARS = 800

DEFAULT_BATCH_SIZE = {"lambda": 8, "bce": 128}
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 2e-5

LOSSES: tuple[str, ...] = ("lambda", "bce")

_WHITESPACE = re.compile(r"\s+")


def document_text(
    title: object, description: object, *, max_chars: int = DEFAULT_MAX_CHARS
) -> str:
    """The document side of a pair: title first, then description, truncated.

    Field order is load-bearing. At max_length = 192 tokens a title +
    description document is cut inside the description for most products, so
    whatever leads is what the model actually reads. Plan 3 lost 6 points of
    its semantic gate to a 77-character title slice and Plan 4 ordered its
    dense fields for the same reason; this is the third time.

    Nulls are dropped rather than stringified - "None" as literal product text
    is worse than nothing.
    """
    parts = [
        _WHITESPACE.sub(" ", str(value)).strip()
        for value in (title, description)
        if isinstance(value, str) and value.strip()
    ]
    return " ".join(parts)[:max_chars]


def check_training_folds(
    matrix: pd.DataFrame, allowed: Sequence[int] = TRAIN_FOLDS
) -> None:
    """Raise unless every row is in an allowed training fold.

    A 22M-parameter model over 250,485 pairs memorises. Training on fold 0 and
    then reporting on it returns a high, meaningless number that nothing else
    in the pipeline would flag.
    """
    present = set(int(f) for f in matrix["fold"].unique())
    forbidden = present - set(int(f) for f in allowed)
    if forbidden:
        raise ValueError(
            f"the training frame contains fold(s) {sorted(forbidden)}, which "
            f"are not in {sorted(allowed)}. Fold 0 is the reporting surface, "
            "fold 1 the eval set and -1 the test split; a cross-encoder "
            "trained on any of them is reporting memorisation."
        )


def _lookup(mapping: Mapping, key, kind: str):
    try:
        value = mapping[key]
    except KeyError:
        raise KeyError(
            f"no {kind} for {key!r}; training on an empty string would teach "
            "the model that this query matches nothing"
        ) from None
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"empty {kind} for {key!r}")
    return value


def listwise_dataset(
    matrix: pd.DataFrame,
    query_text: Mapping,
    doc_text: Mapping,
) -> dict[str, list]:
    """One row per query: (query, docs, labels), as LambdaLoss wants it."""
    out: dict[str, list] = {"query": [], "docs": [], "labels": []}
    for query_id, group in matrix.groupby("query_id", sort=True):
        out["query"].append(_lookup(query_text, query_id, "query text"))
        out["docs"].append(
            [_lookup(doc_text, p, "document text") for p in group["product_id"]]
        )
        out["labels"].append([float(g) for g in group["gain"]])
    return out


def pairwise_dataset(
    matrix: pd.DataFrame,
    query_text: Mapping,
    doc_text: Mapping,
) -> dict[str, list]:
    """One row per judgement: (query, doc, label), for BinaryCrossEntropyLoss."""
    return {
        "query": [
            _lookup(query_text, q, "query text") for q in matrix["query_id"]
        ],
        "doc": [
            _lookup(doc_text, p, "document text") for p in matrix["product_id"]
        ],
        "label": [float(g) for g in matrix["gain"]],
    }


def fine_tune(
    matrix: pd.DataFrame,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    loss: str = "lambda",
    backbone: str = DEFAULT_BACKBONE,
    out_dir: Path = DEFAULT_MODEL_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str = "auto",
    seed: int = 0,
) -> Path:
    """Fine-tune and save. `matrix` must contain only training folds."""
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import (
        BinaryCrossEntropyLoss,
        LambdaLoss,
    )

    from src.clip_encoder import resolve_device

    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")
    check_training_folds(matrix)

    resolved = resolve_device(device)
    model = CrossEncoder(
        backbone, num_labels=1, device=resolved, max_length=max_length
    )
    if loss == "lambda":
        dataset = Dataset.from_dict(listwise_dataset(matrix, query_text, doc_text))
        objective = LambdaLoss(model)
    else:
        dataset = Dataset.from_dict(pairwise_dataset(matrix, query_text, doc_text))
        objective = BinaryCrossEntropyLoss(model)

    target = Path(out_dir) / loss
    target.mkdir(parents=True, exist_ok=True)
    arguments = CrossEncoderTrainingArguments(
        output_dir=str(target / "checkpoints"),
        per_device_train_batch_size=batch_size or DEFAULT_BATCH_SIZE[loss],
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=resolved.startswith("cuda"),
        save_strategy="no",
        logging_steps=200,
        report_to=[],
        seed=seed,
    )
    CrossEncoderTrainer(
        model=model, args=arguments, train_dataset=dataset, loss=objective
    ).train()
    model.save_pretrained(str(target))
    return target


def text_maps(
    matrix: pd.DataFrame, products_path: Path, judgements_path: Path
) -> tuple[dict, dict]:
    """Query and document text for every row of `matrix`.

    Public because src/llm_rerank.py and src/fine_rank_report.py both need the
    same two maps, and building them twice is how the two arms end up reading
    different product text.
    """
    judgements = pd.read_parquet(
        judgements_path, columns=["query_id", "query"]
    ).drop_duplicates("query_id")
    query_text = dict(zip(judgements["query_id"], judgements["query"]))

    products = pd.read_parquet(
        products_path, columns=["product_id", "product_title", "description"]
    )
    products = products.loc[
        products["product_id"].isin(set(matrix["product_id"]))
    ].drop_duplicates("product_id")
    doc_text = {
        pid: document_text(title, description)
        for pid, title, description in zip(
            products["product_id"], products["product_title"], products["description"]
        )
    }
    return query_text, doc_text


# --- prediction over the window ---------------------------------------------


@dataclass(frozen=True)
class Latency:
    """Two different questions, both answered.

    `batched_ms` is what an offline re-ranking job costs per query when the
    whole split is in flight. `single_ms` is what one user waits. The
    cross-encoder batches 256 pairs a forward pass and the LLM arm cannot
    batch at all, so reporting only the amortised figure flatters this arm by
    a further 3-5x on top of the real gap. Plan 4 measured the same split on
    its BM25 index: 28 ms/query amortised against 64-88 ms single.
    """

    single_ms: float
    batched_ms: float
    n_queries: int
    n_pairs: int

    def to_dict(self) -> dict:
        return {
            "single_ms": self.single_ms,
            "batched_ms": self.batched_ms,
            "n_queries": self.n_queries,
            "n_pairs": self.n_pairs,
        }


def load_reranker(
    path: Path | str,
    *,
    device: str = "auto",
    max_length: int = DEFAULT_MAX_LENGTH,
):
    """Load a fine-tuned (or off-the-shelf) cross-encoder onto the GPU."""
    from sentence_transformers import CrossEncoder

    from src.clip_encoder import resolve_device

    return CrossEncoder(str(path), device=resolve_device(device), max_length=max_length)


def _window_pairs(
    windows_: Sequence, query_text: Mapping, doc_text: Mapping
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(query, document) pairs for every window document, plus their (qid, doc)."""
    pairs: list[tuple[str, str]] = []
    keys: list[tuple[str, str]] = []
    for w in windows_:
        query = _lookup(query_text, w.query_id, "query text")
        for document in w.window:
            pairs.append((query, _lookup(doc_text, document, "document text")))
            keys.append((w.query_id, document))
    return pairs, keys


def rerank(
    model,
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    batch_size: int = 256,
) -> dict[str, list[str]]:
    """A new ordering of each window, best first.

    Returns orderings rather than scores on purpose: src.rerank_window builds
    the run from rank positions, so the cross-encoder's raw logits never enter
    a run alongside Stage 2's scores. Ties break on document id so two runs of
    the same model produce the same ordering.
    """
    windows_ = list(windows_)
    if not windows_:
        return {}

    pairs, keys = _window_pairs(windows_, query_text, doc_text)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    by_query: dict[str, list[tuple[float, str]]] = {}
    for (query_id, document), score in zip(keys, scores):
        by_query.setdefault(query_id, []).append((float(score), document))
    return {
        query_id: [d for _, d in sorted(scored, key=lambda sd: (-sd[0], sd[1]))]
        for query_id, scored in by_query.items()
    }


def measure_latency(
    model,
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    n_queries: int = 50,
    batch_size: int = 256,
) -> Latency:
    """Time the same work twice: one query at a time, then all at once."""
    import time

    windows_ = list(windows_)[:n_queries]
    if not windows_:
        raise ValueError("latency needs at least one query to measure")

    # One query at a time - what a user waits for.
    started = time.perf_counter()
    n_pairs = 0
    for w in windows_:
        pairs, _ = _window_pairs([w], query_text, doc_text)
        model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        n_pairs += len(pairs)
    single = (time.perf_counter() - started) * 1000 / len(windows_)

    # Everything in flight - what an offline job costs.
    pairs, _ = _window_pairs(windows_, query_text, doc_text)
    started = time.perf_counter()
    model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    batched = (time.perf_counter() - started) * 1000 / len(windows_)

    return Latency(
        single_ms=single,
        batched_ms=batched,
        n_queries=len(windows_),
        n_pairs=n_pairs,
    )


def _main() -> int:
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", default="lambda", choices=list(LOSSES))
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--products", type=Path, default=Path("data/combined/products.parquet"))
    parser.add_argument("--judgements", type=Path, default=Path("data/combined/judgements.parquet"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matrix = pd.read_parquet(
        args.features_dir / "train.parquet",
        columns=["query_id", "product_id", "gain", "fold"],
    )
    matrix = matrix.loc[matrix["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    check_training_folds(matrix)
    print(
        f"{len(matrix):,} pairs over {matrix['query_id'].nunique():,} queries "
        f"from folds {sorted(TRAIN_FOLDS)}"
    )

    query_text, doc_text = text_maps(matrix, args.products, args.judgements)
    started = time.time()
    target = fine_tune(
        matrix,
        query_text,
        doc_text,
        loss=args.loss,
        backbone=args.backbone,
        out_dir=args.out_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )
    print(f"trained in {(time.time() - started) / 60:.1f} min -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
