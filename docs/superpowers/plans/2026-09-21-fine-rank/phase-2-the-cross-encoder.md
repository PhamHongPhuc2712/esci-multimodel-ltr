# Phase 2 — The Cross-Encoder

**Plan 6 of 7 · Phase 2 of 3 · Tasks 3–4.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 1 gate](phase-1-the-window.md#phase-1-gate) passes.

**Delivers:** `src/cross_encoder.py` — a cross-encoder fine-tuned on folds
2/3/4, its predictions over the top-K window, and its latency measured two ways.

**Needs on disk:** `data/features/stage2-{train,test}.parquet` from Phase 1,
`data/features/train.parquet` and `data/combined/products.parquet`. A GPU.

**Owns Review Focus items 3 and 5** (training on the reporting fold,
batch-amortised throughput reported as latency).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **The cross-encoder trains on folds 2/3/4 only.** Fold 0 is the reporting surface, fold 1 the eval set.
- **Report latency as both single-query and batch-amortised**, labelled. They differ by 3–5x.
- **Never tune on test.** The backbone, loss, epoch count, `K` and `max_length` are all chosen on fold 1.
- **Run GPU-capable work on the GPU.** Pass `device` explicitly; fail loudly if `cuda` is unavailable.
- **A method that ties, or loses, is a reportable result.**
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## What was measured, and what it decides

Zero-shot, on an identical 400-query fold-0 sample where **Stage 2 scores
0.8577**:

| model | document text | NDCG | GPU latency (batched) |
|---|---|---|---|
| `ms-marco-MiniLM-L6-v2` | title | 0.8286 | 7 ms/query |
| `ms-marco-MiniLM-L6-v2` | title + description | **0.8396** | 40 ms/query |
| `bge-reranker-base` | title | 0.8366 | 33 ms/query |
| `bge-reranker-base` | title + description | **0.8445** | 205 ms/query |

**Every zero-shot arm loses to Stage 2**, by 0.013 to 0.029. Fine-tuning is the
arm, not an optimisation of it. Adding the description is worth +0.008 to
+0.011 and costs 5–6x the latency, so the default document text is
**title first, then description**, truncated — and the order is load-bearing,
because `max_length` cuts the tail. This is the third time the repo has met
this trap: Plan 3 lost 6 points of its semantic gate to a 77-*character* title
slice, and Plan 4 ordered its dense fields for the same reason.

Fine-tuning throughput on the 3080, over the 12,519 train-fold queries /
250,485 pairs:

| recipe | batch | throughput | peak VRAM | one epoch |
|---|---|---|---|---|
| `BinaryCrossEntropyLoss`, pairs | 128 @ len 192 | 393 pairs/s | 4.12 GB | **10.6 min** |
| `BinaryCrossEntropyLoss`, pairs | 64 @ len 256 | 266 pairs/s | 2.87 GB | 15.7 min |
| **`LambdaLoss`, listwise groups** | **8** | **7.2 queries/s** | **6.22 GB** | **29.0 min** |
| `LambdaLoss`, listwise groups | 4 | 3.9 queries/s | 3.81 GB | 54.1 min |

Both fit in 16 GB with room to spare. **`LambdaLoss` is the default** — it
optimises a listwise objective against the same graded labels the metric uses,
and Plan 5's Ablation 5 measured listwise beating pointwise by +0.0085 to
+0.0096 on exactly this data. `BinaryCrossEntropyLoss` is kept as the cheap
comparison arm, which also gives Ablation 6 a second data point on whether the
listwise loss keeps paying at Stage 3.

**Two new dependencies.** `CrossEncoderTrainer` needs `datasets` and
`accelerate>=1.1.0`; neither ships with `sentence-transformers`. Verified: with
both installed, `LambdaLoss` trains on a `(query, docs, labels)` dataset and
`BinaryCrossEntropyLoss` on a `(query, doc, label)` one.

---

## Task 3: Fine-tune the cross-encoder

**Files:**
- Create: `src/cross_encoder.py`
- Create: `tests/test_cross_encoder.py`
- Modify: `pyproject.toml` (add `datasets` and `accelerate` to the `baselines` extra)

**Interfaces:**
- Consumes: `src.ranker.{TRAIN_FOLDS, EARLY_STOP_FOLD, REPORT_FOLD}` (Plan 5).
- Produces:
  - `src.cross_encoder.DEFAULT_BACKBONE: str` = `"cross-encoder/ms-marco-MiniLM-L6-v2"`
  - `src.cross_encoder.DEFAULT_MODEL_DIR: Path` = `Path("models/cross-encoder")`
  - `src.cross_encoder.DEFAULT_MAX_LENGTH: int` = `192`
  - `src.cross_encoder.LOSSES: tuple[str, ...]` = `("lambda", "bce")`
  - `src.cross_encoder.document_text(title, description, *, max_chars=DEFAULT_MAX_CHARS) -> str`
  - `src.cross_encoder.listwise_dataset(matrix, query_text, doc_text) -> dict[str, list]`
  - `src.cross_encoder.pairwise_dataset(matrix, query_text, doc_text) -> dict[str, list]`
  - `src.cross_encoder.check_training_folds(matrix, allowed=TRAIN_FOLDS) -> None`
  - `src.cross_encoder.fine_tune(...) -> Path`
  - `src.cross_encoder.text_maps(matrix, products_path, judgements_path) -> tuple[dict, dict]`
  - CLI: `python -m src.cross_encoder --loss lambda` writing `models/cross-encoder/<loss>/`

**Review Focus 3 lives in `check_training_folds`.** `data/features/train.parquet`
contains all five folds. A fine-tune that reads it without filtering trains a
22M-parameter model on 419,653 pairs including the 82,751 of fold 0, and then
gets scored on fold 0. The resulting NDCG would be a memorisation score, high
and meaningless, and nothing about the run would look wrong. So the check is a
hard guard rather than a convention: `fine_tune` calls it and raises on any
fold outside 2/3/4.

**`document_text` puts the title first, always.** At `max_length = 192` tokens,
a document of title + description is truncated inside the description for most
products (the coalesced `description` is present for 89.2% of judged products
and averages well over 1,000 characters). Whatever leads is what the model
reads.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cross_encoder.py`:

```python
import pandas as pd
import pytest

from src.cross_encoder import (
    DEFAULT_BACKBONE,
    DEFAULT_MAX_LENGTH,
    LOSSES,
    check_training_folds,
    document_text,
    listwise_dataset,
    pairwise_dataset,
)
from src.ranker import EARLY_STOP_FOLD, REPORT_FOLD, TRAIN_FOLDS


# --- the document side ------------------------------------------------------

def test_the_title_comes_first():
    # max_length truncates the tail, so whatever leads is what the model reads.
    # Plan 3 lost 6 points of its semantic gate to exactly this mistake.
    text = document_text("Steel Water Bottle", "A very long description.")
    assert text.startswith("Steel Water Bottle")


def test_a_missing_description_leaves_just_the_title():
    assert document_text("Steel Water Bottle", None) == "Steel Water Bottle"
    assert document_text("Steel Water Bottle", "") == "Steel Water Bottle"


def test_a_missing_title_does_not_stringify_none():
    # 'None' as literal product text is worse than an empty string.
    assert "None" not in document_text(None, "A description.")


def test_the_document_is_truncated_but_the_title_survives():
    long_description = "x" * 5000
    text = document_text("Steel Water Bottle", long_description, max_chars=100)
    assert len(text) == 100
    assert text.startswith("Steel Water Bottle")


def test_a_title_longer_than_the_budget_is_not_silently_dropped():
    text = document_text("y" * 400, "a description", max_chars=100)
    assert text == "y" * 100


def test_whitespace_is_collapsed():
    assert document_text("A   B", "C\n\nD") == "A B C D"


# --- Review Focus 3: the training folds -------------------------------------

def _matrix(folds=(2, 3, 4)):
    rows = []
    q = 0
    for fold in folds:
        for _ in range(3):
            for p in range(4):
                rows.append(
                    {
                        "query_id": q,
                        "product_id": f"p{q}_{p}",
                        "gain": [0.0, 0.1, 1.0, 0.01][p],
                        "fold": fold,
                    }
                )
            q += 1
    return pd.DataFrame(rows)


def test_check_training_folds_accepts_the_training_folds():
    check_training_folds(_matrix(TRAIN_FOLDS))  # must not raise


def test_check_training_folds_rejects_the_reporting_fold():
    # 22M parameters over 250,485 pairs will memorise. Training on fold 0 and
    # then scoring on it returns an impressively wrong number.
    with pytest.raises(ValueError, match="fold"):
        check_training_folds(_matrix((2, 3, REPORT_FOLD)))


def test_check_training_folds_rejects_the_early_stop_fold():
    with pytest.raises(ValueError, match="fold"):
        check_training_folds(_matrix((2, EARLY_STOP_FOLD)))


def test_check_training_folds_rejects_the_test_split():
    with pytest.raises(ValueError, match="fold"):
        check_training_folds(_matrix((2, -1)))


def test_check_training_folds_names_the_offending_folds():
    with pytest.raises(ValueError) as excinfo:
        check_training_folds(_matrix((2, 0)))
    assert "0" in str(excinfo.value)


# --- the two dataset shapes -------------------------------------------------

def _texts():
    q = {0: "red shoes", 1: "blue hat", 2: "green mug"}
    d = {f"p{i}_{p}": f"product {i}-{p}" for i in range(9) for p in range(4)}
    return q, d


def test_the_listwise_dataset_groups_one_row_per_query():
    matrix = _matrix((2,))
    q, d = _texts()
    ds = listwise_dataset(matrix, q, d)
    assert set(ds) == {"query", "docs", "labels"}
    assert len(ds["query"]) == matrix["query_id"].nunique()
    assert all(len(docs) == len(labels) for docs, labels in zip(ds["docs"], ds["labels"]))


def test_the_listwise_labels_are_the_esci_gains():
    matrix = _matrix((2,))
    q, d = _texts()
    ds = listwise_dataset(matrix, q, d)
    assert sorted(ds["labels"][0]) == [0.0, 0.01, 0.1, 1.0]


def test_the_pairwise_dataset_is_one_row_per_judgement():
    matrix = _matrix((2,))
    q, d = _texts()
    ds = pairwise_dataset(matrix, q, d)
    assert set(ds) == {"query", "doc", "label"}
    assert len(ds["query"]) == len(matrix)


def test_a_query_with_no_text_raises_rather_than_training_on_an_empty_string():
    matrix = _matrix((2,))
    _, d = _texts()
    with pytest.raises(KeyError, match="query text"):
        listwise_dataset(matrix, {}, d)


def test_a_product_with_no_text_raises():
    matrix = _matrix((2,))
    q, _ = _texts()
    with pytest.raises(KeyError, match="document text"):
        listwise_dataset(matrix, q, {})


def test_the_declared_losses_are_the_two_measured_recipes():
    assert LOSSES == ("lambda", "bce")


def test_the_defaults_match_what_was_measured():
    assert DEFAULT_BACKBONE == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert DEFAULT_MAX_LENGTH == 192
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cross_encoder.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.cross_encoder'`.

- [ ] **Step 3: Add the two dependencies**

`CrossEncoderTrainer` needs `datasets` and `accelerate>=1.1.0`, and neither
ships with `sentence-transformers`. Without them the import succeeds and the
*first training step* fails with `ImportError: Using the Trainer with PyTorch
requires accelerate>=1.1.0` — an hour into a run, if it is not caught here.

In `pyproject.toml`, extend the `baselines` extra:

```toml
baselines = [
    "sentence-transformers>=3.0",
    "torch>=2.4",
    "pillow>=10",
    "datasets>=3.0",
    "accelerate>=1.1.0",
]
```

Then: `uv pip install -e ".[dev,baselines,retrieval,ranking]"`
Verify: `python -c "import datasets, accelerate; print(datasets.__version__, accelerate.__version__)"`

- [ ] **Step 4: Write the implementation**

Create `src/cross_encoder.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cross_encoder.py -q`
Expected: PASS, 18 tests.

- [ ] **Step 6: Fine-tune both arms**

Run:

```bash
python -m src.cross_encoder --loss lambda
python -m src.cross_encoder --loss bce
```

Expected: 250,485 pairs over 12,519 queries from folds [2, 3, 4]; roughly
**29 min** for `lambda` at batch 8 and **11 min** for `bce` at batch 128; peak
VRAM 6.2 GB and 4.1 GB. Models land in `models/cross-encoder/{lambda,bce}/`.

If the run reports a query count near 20,888 rather than 12,519, the fold
filter did not apply and `check_training_folds` should have raised — stop and
fix it before scoring anything.

- [ ] **Step 7: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 8: Commit**

```bash
git add src/cross_encoder.py tests/test_cross_encoder.py pyproject.toml
git commit -m "Fine-tune a cross-encoder on the train folds with a listwise loss"
```

---

## Task 4: Predict over the window, and measure latency honestly

**Files:**
- Modify: `src/cross_encoder.py` (add the prediction and latency half)
- Modify: `tests/test_cross_encoder.py`

**Interfaces:**
- Consumes: `src.rerank_window.{Window, DEFAULT_K}` (Task 2), Task 3's saved models.
- Produces:
  - `src.cross_encoder.Latency` (frozen dataclass: `single_ms: float`, `batched_ms: float`, `n_queries: int`, `n_pairs: int`)
  - `src.cross_encoder.load_reranker(path, *, device="auto", max_length=DEFAULT_MAX_LENGTH) -> CrossEncoder`
  - `src.cross_encoder.rerank(model, windows, query_text, doc_text, *, batch_size=256) -> dict[str, list[str]]`
  - `src.cross_encoder.measure_latency(model, windows, query_text, doc_text, *, n_queries=50, batch_size=256) -> Latency`

**Review Focus 5 lives in `measure_latency`.** Ablation 6's deliverable is
"NDCG, latency and cost". The cross-encoder scores 256 pairs in one forward
pass; the LLM arm cannot batch at all. Reporting the cross-encoder's
batch-amortised throughput as "latency" flatters it by a further 3–5x on top of
the genuine 20–600x gap, and the two numbers have different operational
meanings: batched is what an offline re-ranking job costs, single is what a user
waits. Plan 4 hit this exactly — its BM25 index measured 28 ms/query amortised
against 64–88 ms for a single query — so `Latency` carries both fields and
neither can be produced without the other.

`rerank` returns *orderings*, not scores, because `src.rerank_window.spliced_run`
takes orderings and refuses anything that is not a permutation of the window.
Keeping the cross-encoder's raw logits out of the run is what stops the scale
mismatch of Review Focus 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cross_encoder.py`:

```python
import numpy as np

from src.cross_encoder import Latency, measure_latency, rerank
from src.rerank_window import Window


class FakeModel:
    """A CrossEncoder with the one method the reranker uses."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def predict(self, pairs, batch_size=256, show_progress_bar=False, **kwargs):
        pairs = list(pairs)
        self.calls.append(len(pairs))
        return np.array([self.scores.get(d, 0.0) for _, d in pairs], dtype=np.float32)


def _windows():
    return [
        Window(query_id="1", window=("a", "b", "c"), tail=("d",)),
        Window(query_id="2", window=("x", "y"), tail=()),
    ]


def _maps():
    q = {"1": "red shoes", "2": "blue hat"}
    d = {k: f"doc {k}" for k in "abcdxy"}
    return q, d


def test_rerank_orders_the_window_by_model_score():
    q, d = _maps()
    model = FakeModel({"doc a": 0.1, "doc b": 0.9, "doc c": 0.5, "doc x": 0.2, "doc y": 0.8})
    out = rerank(model, _windows(), q, d)
    assert out["1"] == ["b", "c", "a"]
    assert out["2"] == ["y", "x"]


def test_rerank_returns_a_permutation_of_each_window():
    q, d = _maps()
    out = rerank(FakeModel(), _windows(), q, d)
    for w in _windows():
        assert sorted(out[w.query_id]) == sorted(w.window)


def test_rerank_never_touches_the_tail():
    q, d = _maps()
    out = rerank(FakeModel({"doc d": 99.0}), _windows(), q, d)
    assert "d" not in out["1"]


def test_rerank_breaks_ties_deterministically():
    q, d = _maps()
    a = rerank(FakeModel(), _windows(), q, d)
    b = rerank(FakeModel(), _windows(), q, d)
    assert a == b


def test_rerank_scores_every_window_pair_once():
    q, d = _maps()
    model = FakeModel()
    rerank(model, _windows(), q, d)
    assert sum(model.calls) == 5  # 3 + 2 window documents, no tail


def test_rerank_of_nothing_is_nothing():
    assert rerank(FakeModel(), [], *_maps()) == {}


# --- Review Focus 5: two latencies, both measured ---------------------------

def test_latency_reports_single_and_batched_separately():
    # Batched is what an offline job costs; single is what a user waits. They
    # differ by 3-5x, and quoting one as the other flatters whichever arm
    # batches better - the cross-encoder does, the LLM cannot.
    q, d = _maps()
    result = measure_latency(FakeModel(), _windows(), q, d, n_queries=2)
    assert isinstance(result, Latency)
    assert result.single_ms > 0
    assert result.batched_ms > 0
    assert result.n_queries == 2
    assert result.n_pairs == 5


def test_latency_single_mode_calls_the_model_once_per_query():
    q, d = _maps()
    model = FakeModel()
    measure_latency(model, _windows(), q, d, n_queries=2)
    # The single-query pass must issue one call per query, not one big batch,
    # or it is measuring throughput again under a different name.
    assert model.calls[:2] == [3, 2]


def test_latency_serialises_both_numbers():
    q, d = _maps()
    payload = measure_latency(FakeModel(), _windows(), q, d, n_queries=2).to_dict()
    assert set(payload) == {"single_ms", "batched_ms", "n_queries", "n_pairs"}


def test_latency_needs_at_least_one_query():
    with pytest.raises(ValueError, match="at least one"):
        measure_latency(FakeModel(), [], *_maps(), n_queries=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cross_encoder.py -q`
Expected: FAIL with `ImportError: cannot import name 'Latency' from 'src.cross_encoder'`.

- [ ] **Step 3: Write the implementation**

Append to `src/cross_encoder.py`:

```python
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
```

Add `from dataclasses import dataclass` to the imports at the top of the file.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cross_encoder.py -q`
Expected: PASS, 28 tests.

- [ ] **Step 5: Score fold 0 with both fine-tuned arms and the zero-shot control**

Run:

```bash
python - <<'PY'
import pandas as pd
from src.cross_encoder import DEFAULT_BACKBONE, text_maps, load_reranker, measure_latency, rerank
from src.metrics import ndcg_per_query
from src.rank_report import qrels_from_frame
from src.ranker import REPORT_FOLD
from src.rerank_window import DEFAULT_K, spliced_run, stage2_run, windows
from src.stage2_scores import load_stage2

matrix = pd.read_parquet("data/features/train.parquet")
fold0 = matrix.loc[matrix["fold"] == REPORT_FOLD]
scores = load_stage2("train")
joined = fold0[["query_id", "product_id"]].merge(scores, on=["query_id", "product_id"])
ws = windows(joined, k=DEFAULT_K)
qrels = qrels_from_frame(fold0)
qt, dt = text_maps(fold0, "data/combined/products.parquet", "data/combined/judgements.parquet")
qt = {str(k): v for k, v in qt.items()}

base = ndcg_per_query(stage2_run(ws), qrels)
print(f"stage2                {sum(base.values())/len(base):.4f}")
for label, path in [("zero-shot", DEFAULT_BACKBONE),
                    ("fine-tuned (lambda)", "models/cross-encoder/lambda"),
                    ("fine-tuned (bce)", "models/cross-encoder/bce")]:
    model = load_reranker(path)
    per = ndcg_per_query(spliced_run(ws, rerank(model, ws, qt, dt)), qrels)
    lat = measure_latency(model, ws, qt, dt, n_queries=50)
    print(f"{label:21s} {sum(per.values())/len(per):.4f}   "
          f"single {lat.single_ms:.0f} ms/query, batched {lat.batched_ms:.1f} ms/query")
PY
```

Expected: `stage2` reproduces **0.8519**. The zero-shot arm should land near
the measured 0.8396 for this backbone on title+description — **below** Stage 2.
The fine-tuned arms are the question the plan exists to answer; anything at or
below the zero-shot number means the fine-tune failed and the fold filter,
the loss and the learning rate are the places to look.

Record the four numbers; Task 6 reproduces them with intervals.

**Measured 2026-09-22** on all 4,130 fold-0 queries (the plan's zero-shot
figures came from a 400-query sample, so they are not directly comparable):

| arm | NDCG | vs. stage2 | single | batched |
|---|---|---|---|---|
| `stage2` | 0.8519 | — | — | — |
| zero-shot `ms-marco-MiniLM-L6-v2` | 0.8457 | **−0.0062** | 21 ms | 9.7 ms |
| **fine-tuned `lambda`** | **0.8587** | **+0.0068** | 21 ms | 9.9 ms |
| **fine-tuned `bce`** | **0.8594** | **+0.0075** | 20 ms | 10.0 ms |

**The fine-tune is the arm, as the plan argued.** Zero-shot loses to Stage 2
by 0.0062 and both fine-tuned arms beat it; fine-tuning is worth **+0.0130**
over zero-shot on the same backbone, same window, same 4,130 queries.

**`bce` edges `lambda` by +0.0007, which the plan did not predict.** Plan 5's
Ablation 5 measured listwise beating pointwise by +0.0085 to +0.0096 at the
coarse stage, and this phase made `LambdaLoss` the default on that basis. At
Stage 3, over a 10-document window rather than a full candidate list, the gap
reverses and is an order of magnitude smaller than the coarse-stage one — well
inside what a paired interval may not separate. Task 6 reports both with CIs
rather than declaring a winner here. `bce` also trains **5.4x faster**
(5.7 min against 30.6), so if the interval straddles zero the cheap arm wins on
cost.

The latency gap is **2.1x**, not the 3–5x the phase predicted — the window is
only 10 documents, so a single-query call is already near the GPU's efficient
batch size. Both numbers are recorded regardless, which is the point.

Training, measured: `lambda` 30.6 min at 6.92 queries/s (predicted 29.0 at
7.2); `bce` 5.7 min at 771 pairs/s (predicted 10.6 at 393 — nearly 2x faster
than measured when the plan was written).

- [ ] **Step 6: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 7: Commit**

```bash
git add src/cross_encoder.py tests/test_cross_encoder.py
git commit -m "Re-rank the Stage 2 window with the cross-encoder and measure both latencies"
```

---

## Phase 2 Gate

**Passed 2026-09-22.** Both fine-tuned arms beat Stage 2; the loss the phase
chose as default is not the one that won.

- [x] `python -m pytest` passes with no failures and no new skips. — 546 passed (518 before this phase), 23 deselected, 0 skipped.
- [x] `models/cross-encoder/lambda/` and `models/cross-encoder/bce/` exist, trained on 12,519 queries from folds 2/3/4 only. — both 88 MB; the CLI printed `250,485 pairs over 12,519 queries from folds [2, 3, 4]` for each, and 1,565 steps at batch 8 is 12,520 rows of the listwise dataset.
- [x] `check_training_folds` raises on a frame containing fold 0, fold 1 or the test split, and the real fine-tune passed it. — four tests pin the raises; `fine_tune` calls it before touching a model.
- [x] `rerank` returns orderings, never scores, and `spliced_run` accepts every one of them as a permutation. — all 4,130 fold-0 windows spliced without raising, for all three models.
- [x] `measure_latency` reports single-query and batch-amortised figures that differ, both recorded. — 20–21 ms single against 9.7–10.0 ms batched, a 2.1x gap.
- [x] The fine-tuned arms are scored on fold 0 against the zero-shot control, and the four numbers are written down. — see the table in Task 4, Step 5.

Then: [Phase 3 — The LLM and the Ablation](phase-3-the-llm-and-the-ablation.md).
