# Phase 1 — The Code

**Plan 9 · Phase 1 of 2 · Tasks 1–2.** Read [`README.md`](README.md) first. It
carries the goal, the measurements, the comparison grid and the full Global
Constraints.

**Delivers:**
- gradient checkpointing and a `training.json` record in both fine-tunes;
- the `--capacity` preset in `src/distill_report.py`: four arms, three labelled comparisons, and the best arm on fold 0.

**Needs on disk:** nothing new for the tests. The Task 1 smoke step needs a GPU,
`data/features/{train.parquet,stage2-train-oof.parquet}` and
`data/combined/*.parquet`.

**Owns Review Focus items 1–3** (checkpointing that does not switch on, a
comparison that changes more than it says, the headline picked on test).

### Constraints that bite hardest here

- **Checkpointing must raise if it does not switch on.** A spilling run looks like a slow machine, not a bug.
- **Every pair is labelled with what it isolates, and the training commands must agree with the label.**
- **`best_model_arm` is a fold-0 decision.** The test file records one too, but nothing may read it.
- **No API calls.**
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: Gradient checkpointing and training records for both fine-tunes

**Files:**
- Modify: `src/cross_encoder.py` (import `json`; add `LOSS_NAMES`, `TRAINING_RECORD`, `enable_gradient_checkpointing`, `training_record`; `fine_tune(gradient_checkpointing=…)`; `--gradient-checkpointing`; the CLI writes `training.json`)
- Modify: `src/distill.py` (import `TRAINING_RECORD` and `enable_gradient_checkpointing` from `src.cross_encoder`; `fine_tune_windows(gradient_checkpointing=…)`; `--gradient-checkpointing`, recorded)
- Modify: `tests/test_cross_encoder.py`

**Interfaces:**
- Consumes: `sentence_transformers.CrossEncoder`, whose `.model` is the wrapped transformers model with `gradient_checkpointing_enable(gradient_checkpointing_kwargs=…)` and `is_gradient_checkpointing`.
- Produces:
  - `src.cross_encoder.LOSS_NAMES: dict[str, str]` = `{"lambda": "LambdaLoss", "bce": "BinaryCrossEntropyLoss"}`
  - `src.cross_encoder.TRAINING_RECORD: str` = `"training.json"`, re-exported by `src.distill` (Plan 8's `src.distill_report.training_records` imports it from there, unchanged)
  - `src.cross_encoder.enable_gradient_checkpointing(model) -> None` — raises `RuntimeError` if the switch does not take
  - `src.cross_encoder.training_record(*, loss, backbone, epochs, batch_size, seed, gradient_checkpointing, n_queries, n_pairs, minutes) -> dict`
  - `src.cross_encoder.fine_tune(..., gradient_checkpointing=False) -> Path`
  - `src.distill.fine_tune_windows(..., gradient_checkpointing=False)`
  - CLI: `python -m src.cross_encoder ... --gradient-checkpointing` writes `<out-dir>/<loss>/training.json`; `python -m src.distill ... --gradient-checkpointing` records the flag

**Review Focus 1 lives in `enable_gradient_checkpointing`.** Without
checkpointing, bge on whole groups does not raise an out-of-memory error. It
spills into shared memory and runs at under 0.27 queries/s, a 13-hour epoch
that looks like a slow machine. So the function checks
`is_gradient_checkpointing` after switching it on, and raises if it did not
take. The Trainer's own `gradient_checkpointing=True` cannot be used: in
sentence-transformers 6.1 it raises `gradient_checkpointing_enable() got an
unexpected keyword argument 'every_n_layers'`.

**The whole-group CLI now writes a training record.** `src.distill_report`
refuses a model directory without one (Plan 8's `training_records`), and
Plan 6's `python -m src.cross_encoder` never wrote one. The record has the
same shape as `src.distill`'s, so one committed test can read every arm's
recipe.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cross_encoder.py`, extend the import:

```python
from src.cross_encoder import (
    DEFAULT_BACKBONE,
    DEFAULT_MAX_LENGTH,
    LOSSES,
    TRAINING_RECORD,
    check_training_folds,
    document_text,
    enable_gradient_checkpointing,
    listwise_dataset,
    pairwise_dataset,
    training_record,
)
```

and append:

```python
# --- Plan 9: memory, and saying what trained a model ------------------------

class _WrappedModel:
    """Stands in for the transformers model a CrossEncoder wraps."""

    def __init__(self, switches_on=True):
        self.switches_on = switches_on
        self.is_gradient_checkpointing = False
        self.kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.kwargs = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = self.switches_on


class _CrossEncoder:
    def __init__(self, switches_on=True):
        self.model = _WrappedModel(switches_on)


def test_gradient_checkpointing_is_switched_on_the_wrapped_model():
    model = _CrossEncoder()
    enable_gradient_checkpointing(model)
    assert model.model.is_gradient_checkpointing
    assert model.model.kwargs == {"use_reentrant": False}


def test_a_checkpointing_switch_that_does_not_take_raises():
    # Without it bge-reranker-base spills past 16 GB and crawls at under 0.27
    # queries/s instead of failing - a nine-hour run nobody asked for.
    with pytest.raises(RuntimeError, match="checkpointing"):
        enable_gradient_checkpointing(_CrossEncoder(switches_on=False))


def _record(**overrides):
    fields = dict(
        loss="lambda", backbone="BAAI/bge-reranker-base", epochs=1, batch_size=8,
        seed=0, gradient_checkpointing=True, n_queries=12_519, n_pairs=250_485,
        minutes=66.0,
    )
    return training_record(**(fields | overrides))


def test_the_training_record_names_the_recipe():
    record = _record()
    assert record["target"] == "labels"
    assert record["init"] == "BAAI/bge-reranker-base"
    assert record["loss"] == "LambdaLoss"
    assert record["batch_size"] == 8
    assert record["gradient_checkpointing"] is True
    assert "whole query groups" in record["data"]


def test_the_training_record_names_the_pairwise_loss_too():
    assert _record(loss="bce")["loss"] == "BinaryCrossEntropyLoss"


def test_a_training_record_for_an_unknown_loss_is_refused():
    with pytest.raises(ValueError, match="loss"):
        _record(loss="hinge")


def test_both_fine_tunes_write_the_same_record_file():
    # src.distill_report reads this one name from every model directory.
    from src.distill import TRAINING_RECORD as window_record

    assert TRAINING_RECORD == window_record == "training.json"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cross_encoder.py -v`
Expected: FAIL — `ImportError: cannot import name 'TRAINING_RECORD' from 'src.cross_encoder'`.

- [ ] **Step 3: Implement**

In `src/cross_encoder.py`, add `import json` to the standard-library imports,
and replace the `LOSSES` line with:

```python
LOSSES: tuple[str, ...] = ("lambda", "bce")
LOSS_NAMES: dict[str, str] = {"lambda": "LambdaLoss", "bce": "BinaryCrossEntropyLoss"}

# Written beside every model this module's CLI or src.distill's trains, so a
# results file can say what trained each arm. src.distill_report refuses a
# model directory without one.
TRAINING_RECORD = "training.json"
```

Add these two functions directly above `fine_tune`:

```python
def enable_gradient_checkpointing(model) -> None:
    """Recompute activations during backward instead of storing them.

    bge-reranker-base on whole query groups at batch 8 spills past the 3080's
    16 GB and crawls under 0.27 queries/s; with this it peaks at 6.3 GB and
    runs 3.18 queries/s (measured 2026-09-30). The step's arithmetic is
    unchanged - only when activations are computed. It is switched on the
    wrapped transformers model because CrossEncoderTrainingArguments'
    `gradient_checkpointing=True` raises inside sentence-transformers 6.1.
    Non-reentrant, which is what current PyTorch recommends.
    """
    model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not model.model.is_gradient_checkpointing:
        raise RuntimeError(
            "gradient checkpointing did not switch on; training would spill "
            "past GPU memory rather than fail"
        )


def training_record(
    *,
    loss: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    seed: int,
    gradient_checkpointing: bool,
    n_queries: int,
    n_pairs: int,
    minutes: float,
) -> dict:
    """What trained a whole-group fine-tune, in the shape src.distill writes."""
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")
    return {
        "target": "labels",
        "init": backbone,
        "loss": LOSS_NAMES[loss],
        "data": "every judged pair of folds 2/3/4, whole query groups",
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "gradient_checkpointing": gradient_checkpointing,
        "n_queries": n_queries,
        "n_pairs": n_pairs,
        "minutes": minutes,
    }
```

Replace `fine_tune`. It gains the `gradient_checkpointing` parameter and
switches it on straight after the model is built:

```python
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
    gradient_checkpointing: bool = False,
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
    if gradient_checkpointing:
        enable_gradient_checkpointing(model)
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
```

Replace `_main`. It gains `--gradient-checkpointing` and writes the training
record beside the model:

```python
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
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="recompute activations in backward; bge-reranker-base needs it on 16 GB")
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
        gradient_checkpointing=args.gradient_checkpointing,
    )
    minutes = (time.time() - started) / 60
    record = training_record(
        loss=args.loss,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size or DEFAULT_BATCH_SIZE[args.loss],
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        n_queries=int(matrix["query_id"].nunique()),
        n_pairs=len(matrix),
        minutes=minutes,
    )
    (target / TRAINING_RECORD).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"trained in {minutes:.1f} min -> {target}")
    return 0
```

In `src/distill.py`, replace the `src.cross_encoder` import with the one below,
and delete the module's own `TRAINING_RECORD = "training.json"` line and the
comment above it. The name is now imported, so `src.distill.TRAINING_RECORD`
still resolves for `src.distill_report`.

```python
from src.cross_encoder import (
    DEFAULT_BACKBONE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_DIR,
    TRAINING_RECORD,
    _lookup,
    enable_gradient_checkpointing,
)
```

Replace `fine_tune_windows`:

```python
def fine_tune_windows(
    dataset: Mapping[str, list],
    *,
    init: str = DEFAULT_BACKBONE,
    out_dir: Path | None = None,
    epochs: int = WINDOW_EPOCHS,
    batch_size: int = WINDOW_BATCH_SIZE,
    learning_rate: float = WINDOW_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "auto",
    seed: int = 0,
    gradient_checkpointing: bool = False,
):
    """Fine-tune a cross-encoder with RankNet on (query, docs, labels) windows."""
    import tempfile

    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import RankNetLoss

    from src.clip_encoder import resolve_device

    if not dataset.get("query"):
        raise ValueError("an empty training set; nothing to fine-tune on")
    resolved = resolve_device(device)
    model = CrossEncoder(str(init), num_labels=1, device=resolved, max_length=max_length)
    if gradient_checkpointing:
        enable_gradient_checkpointing(model)
    with tempfile.TemporaryDirectory() as scratch:
        arguments = CrossEncoderTrainingArguments(
            output_dir=scratch,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=resolved.startswith("cuda"),
            save_strategy="no",
            logging_steps=200,
            report_to=[],
            seed=seed,
            disable_tqdm=True,
        )
        CrossEncoderTrainer(
            model=model,
            args=arguments,
            train_dataset=Dataset.from_dict(dict(dataset)),
            loss=RankNetLoss(model),
        ).train()
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
    return model
```

Replace `_main`. It gains `--gradient-checkpointing`, passes it on and records
it:

```python
def _main() -> int:
    import time

    from src.cross_encoder import text_maps
    from src.llm_rerank import DEFAULT_CACHE, RerankCache
    from src.stage_signals import check_fallback_share, llm_orderings_from_cache

    parser = argparse.ArgumentParser(description=__doc__)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--target", choices=list(TARGETS))
    which.add_argument(
        "--from-gate", type=Path,
        help="train what the pilot's gate chose (docs/results/distill-pilot.json)")
    parser.add_argument(
        "--init", default="scratch",
        help=f"one of {sorted(INITS)}, or any cross-encoder name or path "
             "(the capacity arm passes BAAI/bge-reranker-base)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=WINDOW_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=WINDOW_BATCH_SIZE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--products", type=Path, default=Path("data/combined/products.parquet"))
    parser.add_argument("--judgements", type=Path, default=Path("data/combined/judgements.parquet"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="recompute activations in backward; bge-reranker-base at batch 16 needs it")
    args = parser.parse_args()

    if args.from_gate is not None:
        args.init, args.target = gate_choice(
            json.loads(args.from_gate.read_text(encoding="utf-8"))
        )

    windows_, matrix = training_windows(args.features_dir)
    query_text, doc_text = text_maps(matrix, args.products, args.judgements)
    query_text = {str(k): v for k, v in query_text.items()}
    gains = gains_from_frame(matrix)

    orderings = None
    if args.target != "labels":
        orderings, missing = llm_orderings_from_cache(
            windows_, query_text, RerankCache(args.cache)
        )
        if len(missing) == len(windows_):
            raise SystemExit(
                "the cache holds no teacher answer for any training window; "
                "Plan 8's paid pass (Phase 2) has not run"
            )
        check_fallback_share(missing, len(windows_))

    dataset, dropped = window_dataset(
        windows_, query_text, doc_text,
        kind=args.target, gains=gains, llm_orderings=orderings,
    )
    print(f"{len(dataset['query']):,} training windows from folds 2/3/4, target "
          f"{args.target!r}, init {args.init!r}; {len(dropped):,} dropped for want "
          "of a teacher answer")
    started = time.time()
    fine_tune_windows(
        dataset, init=INITS.get(args.init, args.init), out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    minutes = (time.time() - started) / 60
    # What trained this model, beside it: the report records it with the
    # arm's numbers, so a results file says which target made which arm.
    (args.out / TRAINING_RECORD).write_text(json.dumps({
        "target": args.target, "init": args.init, "loss": "RankNetLoss",
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "n_windows": len(dataset["query"]), "n_dropped": len(dropped),
        "minutes": minutes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"trained in {minutes:.1f} min -> {args.out}")
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cross_encoder.py tests/test_distill.py tests/test_distill_report.py -v`
Expected: PASS — 37 (31 before this task), 33 and 26 passed, 1 deselected.

Run: `python -m pytest`
Expected: PASS — 793 passed (787 before this plan), 36 deselected.

- [ ] **Step 5: Smoke-test both fine-tunes on the GPU with checkpointing on**

This trains bge for a few steps with each fine-tune, and proves the flag
switches on in the real library, not only against the fake in the tests.

Run:

```bash
python - <<'EOF'
from pathlib import Path
import tempfile
import pandas as pd
from src.cross_encoder import fine_tune, text_maps
from src.distill import fine_tune_windows, gains_from_frame, training_windows, window_dataset
from src.ranker import TRAIN_FOLDS

m = pd.read_parquet("data/features/train.parquet", columns=["query_id", "product_id", "gain", "fold"])
m = m.loc[m.fold.isin(TRAIN_FOLDS)]
small = m.loc[m.query_id.isin(set(m.query_id.drop_duplicates().head(16)))].reset_index(drop=True)
qt, dt = text_maps(small, Path("data/combined/products.parquet"), Path("data/combined/judgements.parquet"))
with tempfile.TemporaryDirectory() as out:
    print(fine_tune(small, qt, dt, loss="lambda", backbone="BAAI/bge-reranker-base",
                    out_dir=Path(out), gradient_checkpointing=True))
ws, matrix = training_windows()
qt, dt = text_maps(matrix, Path("data/combined/products.parquet"), Path("data/combined/judgements.parquet"))
data, _ = window_dataset(ws[:32], {str(k): v for k, v in qt.items()}, dt,
                         kind="labels", gains=gains_from_frame(matrix))
model = fine_tune_windows(data, init="BAAI/bge-reranker-base", batch_size=16, gradient_checkpointing=True)
print("windows, checkpointing on:", model.model.is_gradient_checkpointing)
EOF
```

Expected (about a minute): a `.../lambda` path, then
`windows, checkpointing on: True`. Measured that way on 2026-09-30: 48 s and
12 s.

- [ ] **Step 6: Commit**

```bash
git add src/cross_encoder.py src/distill.py tests/test_cross_encoder.py
git commit -m "Add gradient checkpointing and a training record to both cross-encoder fine-tunes"
```

**Landed 2026-09-30 (fc1c16d).** 37, 33 and 26 passed, and 793 in the suite.
The smoke step printed a `.../lambda` path and `windows, checkpointing on:
True`. It ran while another project's GPU job held 7.8 GB of the card, so its
timings are not comparable to the plan's: 10.4 s and 3.2 s of training inside
2 min 19 s end to end. Peak allocated memory was 5.83 GB on the windows, as
measured before the plan, but 7.03 GB on whole groups against the plan's
6.29 GB. The smoke step's 16 queries are not typical: 13 of them hold 38–40
documents, against a mean of 20 and a 99th percentile of 42. Phase 2's
whole-group run should start on an idle card.

---

## Task 2: The capacity preset

**Files:**
- Modify: `src/distill_report.py`
- Modify: `tests/test_distill_report.py`

**Interfaces:**
- Consumes: `src.rank_report.compare`, and Plan 8's `resolve_model_arms`, `training_records` and `_arms`.
- Produces:
  - `MODEL_ARMS` gains `"stage2+ce_bge_b16": models/cross-encoder/bge-b16` and `"stage2+ce_bge_groups": models/cross-encoder/bge-groups/lambda`, each with a `TRAIN_COMMANDS` entry
  - `src.distill_report.CAPACITY_ARMS: tuple[str, ...]` — the four arms
  - `src.distill_report.CAPACITY_PAIRS: dict[tuple[str, str], str]` — `(arm, baseline)` mapped to what the pair isolates
  - `src.distill_report.CAPACITY_OUT: dict[str, Path]` — `docs/results/capacity{,-test}.json`
  - `src.distill_report.pair_comparisons(by_name, pairs, *, seed=0) -> list[dict]` — `compare` rows plus `"isolates"`
  - `src.distill_report.best_model_arm(points, candidates) -> str`
  - every arms report now records `best_model_arm`
  - CLI: `python -m src.distill_report --capacity [--split test --final]`; `--capacity` refuses a hand-picked `--arms`

**Review Focus 2 and 3 live here.**
- Each pair carries what it isolates, and `test_the_commands_differ_where_the_pairs_say_they_do` holds the training commands to those labels. The committed training records are checked against them in Task 3.
- `best_model_arm` never considers a reference arm: the LLM outscores every cross-encoder and is not a no-API arm. Plan 9 reads it from the fold-0 file only.

- [ ] **Step 1: Write the failing tests**

In `tests/test_distill_report.py`, extend the `src.distill_report` import:

```python
from src.distill_report import (
    CAPACITY_ARMS,
    CAPACITY_PAIRS,
    FREE_ARMS,
    GATE_RULE,
    MODEL_ARMS,
    PILOT_EPOCHS,
    REFERENCE_ARMS,
    TRAIN_COMMANDS,
    best_model_arm,
    check_reference,
    check_split,
    paid_run_gate,
    pair_comparisons,
    pilot_arm,
    resolve_model_arms,
    share_of_teacher_gain,
    training_records,
)
```

and append:

```python
# --- Plan 9: the capacity preset --------------------------------------------

def test_every_model_arm_has_a_command_that_trains_it():
    assert set(TRAIN_COMMANDS) == set(MODEL_ARMS)


def test_the_capacity_arms_are_model_arms_and_not_the_student():
    assert set(CAPACITY_ARMS) <= set(MODEL_ARMS)
    assert "stage2+student" not in CAPACITY_ARMS


def test_every_capacity_comparison_is_between_scored_arms():
    scored = set(CAPACITY_ARMS) | set(REFERENCE_ARMS)
    for (arm, baseline), isolates in CAPACITY_PAIRS.items():
        assert {arm, baseline} <= scored
        assert arm != baseline
        assert isolates


def test_the_commands_differ_where_the_pairs_say_they_do():
    # The pairs claim what differs between two arms; the commands are what
    # actually trains them, so they must agree.
    bge, b16 = TRAIN_COMMANDS["stage2+ce_bge"], TRAIN_COMMANDS["stage2+ce_bge_b16"]
    assert "--batch-size 8" in bge and "--batch-size 16" in b16
    assert "BAAI/bge-reranker-base" in bge and "BAAI/bge-reranker-base" in b16
    groups = TRAIN_COMMANDS["stage2+ce_bge_groups"]
    assert "src.cross_encoder --loss lambda" in groups
    assert "--batch-size" not in groups            # the landed recipe's own 8
    assert "--gradient-checkpointing" in groups and "--gradient-checkpointing" in b16


def _arm(name, per_query):
    from src.rank_report import evaluate_arm

    return evaluate_arm(
        name, per_query, {q: 0.5 for q in per_query}, groups=("retrieval",),
        n_features=0, objective="rerank", best_iteration=0,
    )


def test_a_pair_comparison_records_what_it_isolates():
    by_name = {
        "a": _arm("a", {"1": 0.9, "2": 0.8, "3": 0.7}),
        "b": _arm("b", {"1": 0.8, "2": 0.8, "3": 0.7}),
    }
    [row] = pair_comparisons(by_name, {("a", "b"): "the backbone"})
    assert (row["arm"], row["baseline"]) == ("a", "b")
    assert row["isolates"] == "the backbone"
    assert row["delta"]["point"] == pytest.approx(0.1 / 3)


def test_a_pair_naming_an_unscored_arm_raises():
    with pytest.raises(KeyError, match="not among the scored arms"):
        pair_comparisons({"a": _arm("a", {"1": 0.9})}, {("a", "b"): "anything"})


def test_the_best_model_arm_is_the_highest_scoring_candidate():
    points = {"stage2+ce_bge": 0.865, "stage2+ce_bge_groups": 0.870}
    assert best_model_arm(points, list(points)) == "stage2+ce_bge_groups"


def test_the_best_model_arm_is_never_a_reference_arm():
    # The LLM outscores every cross-encoder, and it is not a no-API arm.
    points = {"stage2+llm": 0.881, "stage2+ce_bge": 0.865}
    assert best_model_arm(points, ["stage2+ce_bge"]) == "stage2+ce_bge"


def test_no_scored_candidate_raises():
    with pytest.raises(ValueError, match="no candidate"):
        best_model_arm({"stage2": 0.85}, ["stage2+ce_bge"])


def test_capacity_refuses_a_hand_picked_arm_list(monkeypatch):
    import sys

    from src.distill_report import _main

    monkeypatch.setattr(sys, "argv", ["distill_report", "--capacity", "--arms", "stage2+ce_bge"])
    with pytest.raises(SystemExit, match="--capacity"):
        _main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'CAPACITY_ARMS' from 'src.distill_report'`.

- [ ] **Step 3: Implement**

In `src/distill_report.py`, add this bullet to the end of the module
docstring's list of modes:

```python
  * ``--capacity``: Plan 9's preset. Four cross-encoder arms whose pairwise
    comparisons each change one thing - backbone, batch or recipe - on fold 0
    (docs/results/capacity.json) or, with ``--split test --final``, on test
    (docs/results/capacity-test.json).
```

Replace `MODEL_ARMS` and `FREE_ARMS` with:

```python
MODEL_ARMS: dict[str, Path] = {
    "stage2+ce_windows": Path("models/cross-encoder/windows"),
    "stage2+ce_bge": Path("models/cross-encoder/bge"),
    "stage2+student": Path("models/cross-encoder/student"),
    "stage2+ce_bge_b16": Path("models/cross-encoder/bge-b16"),
    "stage2+ce_bge_groups": Path("models/cross-encoder/bge-groups/lambda"),
}
FREE_ARMS: tuple[str, ...] = ("stage2+ce_windows", "stage2+ce_bge")

# Plan 9. Plan 8's bge arm beat the landed cross-encoder, but it differed from
# it in backbone, data, loss and batch at once. These four arms, with the
# landed cross-encoder, make every comparison below change exactly one of them.
CAPACITY_ARMS: tuple[str, ...] = (
    "stage2+ce_windows",
    "stage2+ce_bge",
    "stage2+ce_bge_b16",
    "stage2+ce_bge_groups",
)
# (arm, baseline) -> the one thing that differs. Every model arm is also
# compared with Stage 2, the landed cross-encoder and the LLM, which is where
# bge_groups against the landed cross-encoder - the backbone under the landed
# recipe - comes from.
CAPACITY_PAIRS: dict[tuple[str, str], str] = {
    ("stage2+ce_bge_b16", "stage2+ce_windows"): "the backbone, on windows at batch 16",
    ("stage2+ce_bge", "stage2+ce_bge_b16"): "the batch, 8 against 16, for bge on windows",
    ("stage2+ce_bge_groups", "stage2+ce_bge"): (
        "the recipe for bge: whole groups and LambdaLoss against windows and "
        "RankNet, both at batch 8"
    ),
}
CAPACITY_OUT: dict[str, Path] = {
    "train": Path("docs/results/capacity.json"),
    "test": Path("docs/results/capacity-test.json"),
}
```

Replace `TRAIN_COMMANDS` with:

```python
TRAIN_COMMANDS: dict[str, str] = {
    "stage2+ce_windows": (
        "python -m src.distill --target labels --out models/cross-encoder/windows"
    ),
    "stage2+ce_bge": (
        "python -m src.distill --target labels --init BAAI/bge-reranker-base "
        "--batch-size 8 --out models/cross-encoder/bge"
    ),
    "stage2+student": (
        "python -m src.distill --from-gate docs/results/distill-pilot.json "
        "--out models/cross-encoder/student"
    ),
    "stage2+ce_bge_b16": (
        "python -m src.distill --target labels --init BAAI/bge-reranker-base "
        "--batch-size 16 --gradient-checkpointing --out models/cross-encoder/bge-b16"
    ),
    "stage2+ce_bge_groups": (
        "python -m src.cross_encoder --loss lambda --backbone BAAI/bge-reranker-base "
        "--gradient-checkpointing --out-dir models/cross-encoder/bge-groups"
    ),
}
```

Add these two functions directly above `check_split`:

```python
def pair_comparisons(
    by_name: Mapping, pairs: Mapping[tuple[str, str], str], *, seed: int = 0
) -> list[dict]:
    """Paired comparisons between arms, each labelled with what it isolates."""
    from src.rank_report import compare

    rows = []
    for (arm, baseline), isolates in pairs.items():
        missing = [name for name in (arm, baseline) if name not in by_name]
        if missing:
            raise KeyError(
                f"{missing} not among the scored arms; a comparison that "
                "silently drops is a table missing its point"
            )
        rows.append(compare(by_name[arm], by_name[baseline], seed=seed)
                    | {"isolates": isolates})
    return rows


def best_model_arm(points: Mapping[str, float], candidates: Sequence[str]) -> str:
    """The candidate with the highest NDCG.

    Plan 9 reads it from the fold-0 file only: that arm becomes the project's
    no-API Stage 3. The test file records it too, but a choice made there
    would be selection on the test split.
    """
    scored = [name for name in candidates if name in points]
    if not scored:
        raise ValueError("no candidate arm was scored")
    return max(scored, key=lambda name: points[name])
```

Replace `_arms`. The changes:
- `pairs` comes from the preset;
- `pair_comparisons` extends the comparisons;
- `best_model_arm` is recorded;
- the output defaults to `CAPACITY_OUT` under `--capacity`.

```python
def _arms(args) -> int:
    from src.blend_report import per_query_ndcg, single_stage_orderings
    from src.cross_encoder import load_reranker, measure_latency, rerank
    from src.distill import gains_from_frame, pairwise_accuracy
    from src.fine_rank_report import check_same_queries
    from src.floor import random_floor
    from src.rank_report import compare, format_table, qrels_from_frame

    check_split(args.split, args.final)
    models = resolve_model_arms(args.arms)
    pairs = CAPACITY_PAIRS if args.capacity else {}
    training = training_records(models)
    scope, windows_, matrix, frame, query_text, doc_text = _scope(args.split, args.seed)
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    gains = gains_from_frame(matrix)
    print(f"{scope}: {len(windows_):,} windows; floor {floor.mean:.4f}")

    results, latency, accuracy = [], {}, {}
    reference = single_stage_orderings(frame)
    for name in REFERENCE_ARMS:
        results.append(_evaluate(name, per_query_ndcg(reference[name], windows_, qrels),
                                 floor, args.seed))
        accuracy[name] = pairwise_accuracy(windows_, reference[name], gains)
    check_reference({arm.name: arm.ndcg.point for arm in results}, _published(args.split))

    landed = load_reranker(LANDED_MODEL)
    latency["stage2+ce"] = measure_latency(
        landed, windows_, query_text, doc_text, n_queries=50).to_dict()
    del landed

    for name, path in models.items():
        model = load_reranker(path)
        orderings = rerank(model, windows_, query_text, doc_text)
        results.append(_evaluate(name, per_query_ndcg(orderings, windows_, qrels),
                                 floor, args.seed))
        accuracy[name] = pairwise_accuracy(windows_, orderings, gains)
        latency[name] = measure_latency(
            model, windows_, query_text, doc_text, n_queries=50).to_dict()
        del model
        print(f"  {name:22s} {results[-1].ndcg.point:.4f}  "
              f"single {latency[name]['single_ms']:.1f} ms")

    check_same_queries(results)
    by_name = {arm.name: arm for arm in results}
    stage2, ce, llm = (by_name[name] for name in REFERENCE_ARMS)
    comparisons = [compare(arm, stage2, seed=args.seed) for arm in results[1:]]
    for name in models:
        comparisons.append(compare(by_name[name], ce, seed=args.seed))
        comparisons.append(compare(by_name[name], llm, seed=args.seed))
    comparisons.extend(pair_comparisons(by_name, pairs, seed=args.seed))
    shares = {
        name: share_of_teacher_gain(by_name[name].ndcg.point, stage2.ndcg.point,
                                    llm.ndcg.point)
        for name in ("stage2+ce", *models)
    }

    print()
    print(format_table(results))
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:22s} vs {row['baseline']:12s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]")

    payload = {
        "split": args.split,
        "scope": scope,
        "n_queries": len(qrels),
        "n_judgements": len(matrix),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "arms": [arm.to_dict() for arm in results],
        "comparisons": comparisons,
        "share_of_llm_gain": shares,
        "pairwise_accuracy": accuracy,
        "latency": latency,
        "models": {name: str(path) for name, path in models.items()},
        "training": training,
        "best_model_arm": best_model_arm(
            {arm.name: arm.ndcg.point for arm in results}, list(models)
        ),
        "seed": args.seed,
    }
    out = args.out or (CAPACITY_OUT if args.capacity else DEFAULT_OUT)[args.split]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0
```

Replace `_main`. `--arms` now defaults to `None` and resolves to `FREE_ARMS`,
or to `CAPACITY_ARMS` under `--capacity`, which refuses a hand-picked list:

```python
def _main() -> int:
    from src.llm_rerank import DEFAULT_CACHE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="the fold-0 cross-fitted pilot and the gate")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--final", action="store_true", help="required with --split test")
    parser.add_argument("--arms", nargs="+", default=None, choices=list(MODEL_ARMS),
                        help=f"default {list(FREE_ARMS)}; --capacity sets its own")
    parser.add_argument("--capacity", action="store_true",
                        help="Plan 9: the four capacity arms and the comparisons "
                             "that each isolate one factor")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.capacity and args.arms is not None:
        raise SystemExit("--capacity sets its own arms; do not pass --arms with it")
    if args.arms is None:
        args.arms = list(CAPACITY_ARMS if args.capacity else FREE_ARMS)
    if args.pilot:
        if args.split != "train":
            raise SystemExit("the pilot trains on fold 0; it never touches the test split")
        return _pilot(args)
    return _arms(args)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: PASS — 36 passed (26 before this task).

Run: `python -m pytest`
Expected: PASS — 803 passed (787 before this plan), 36 deselected.

Before this plan was written, the preset was smoke-tested on the real fold-0
data with the two new arms pointed at copies of Plan 8's bge model. The three
labelled comparisons came out, the copies compared at exactly +0.0000,
`ce_windows` and `ce_bge` reproduced 0.8555 and 0.8650, and the committed
training-record test rejected the copies' placeholder records, as it should.

- [ ] **Step 5: Commit**

```bash
git add src/distill_report.py tests/test_distill_report.py
git commit -m "Add the capacity preset: four cross-encoder arms whose comparisons each change one thing"
```

**Landed 2026-09-30 (0337d18).** 36 passed, and 803 in the suite.

---

## Phase 1 Gate

**Passed 2026-09-30.**

- [x] `python -m pytest` passes with no failures and no new skips. — 803 passed (787 before this plan), 36 deselected, 0 skipped.
- [x] Task 1's smoke step printed a checkpoint path and `windows, checkpointing on: True`.
- [x] `python -m src.distill_report --capacity --arms stage2+ce_bge` exits refusing the hand-picked list before reading any data. — exit 1 in 0.07 s: `--capacity sets its own arms; do not pass --arms with it`.

Then: [Phase 2 — The Runs and the Writeup](phase-2-the-runs-and-the-writeup.md).
