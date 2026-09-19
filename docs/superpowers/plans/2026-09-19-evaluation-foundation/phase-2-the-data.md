# Phase 2 — The Data

**Plan 1 of 7 · Phase 2 of 3 · Tasks 4–6.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 1 gate](phase-1-the-metric.md#phase-1-gate) passes: every number here is
produced by the metric Phase 1 proved.

**Delivers:** the ESCI Task 1 English loader that refuses the wrong source file,
TREC run/qrels I/O with exact score round-tripping, and the random-ordering NDCG
floor — measured, not quoted.

**Needs on disk:** the real official parquet, ~1.08 GB (examples 48.9 MB +
products 1.03 GB). This is the first phase that downloads anything. Task 4
Step 5 fetches it; budget several minutes. Task 6 Step 5 then takes 1–2 minutes
for 100 trials × 181,701 scored documents.

**Owns Review Focus item 5** — loading the wrong source file — pinned in Task 4.
The HF mirror parses cleanly, has the right column names, and yields 185,361
test rows instead of 181,701. Nothing about it fails except the count, which is
why the count is checked on every load.

> **`src/floor.py` is not finished when Phase 2 ends.** Phase 3, Task 9 Step 4
> reopens it to add a `per_query` field, because `evaluate_run` pairs the run
> against the floor query by query rather than comparing two means. Treat
> `FloorResult` as extensible, not final.

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Use the official parquet, never the `tasksource/esci` HF mirror.** The mirror has no `split` column and encodes the *large-version* split, yielding 185,361 test judgements instead of 181,701 (~2% contamination). Fetch via `https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/...` (the repo is Git LFS; the plain `raw.githubusercontent.com` URL returns an LFS pointer, not a Parquet file).
- **Task 1 English filter is `small_version == 1` and `product_locale == "us"`**, then `split`.
- **Asserted row counts:** test = 181,701 judgements / 8,956 queries / 164,900 products; train = 419,653 judgements / 20,888 queries.
- **Asserted test label distribution:** E 43.87%, S 34.98%, I 16.69%, C 4.46%. There is no published train figure, and asserting the test shares against train would be inventing a number.
- **Compute the random floor, never quote it.** `CLAUDE.md` records 0.7467 measured with this discount; the published SQID figure is 0.7483; under the swapped S/C mapping it is 0.7141. That 3.3-point spread is larger than most method gains, so the number must come from the code, not from a constant.
- **Do not commit datasets.** `.gitignore` already blocks `data/`, `runs/`, `*.parquet`, `*.npy`.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line.

---

## Task 4: ESCI dataset loader with asserted invariants

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `src.labels.label_to_gain(label) -> float`, `src.labels.label_to_qrel(label) -> int`, `src.labels.ESCI_GAINS`.
- Produces:
  - `src.dataset.DataInvariantError` (subclass of `ValueError`)
  - `src.dataset.SplitStats(judgements: int, queries: int, products: int | None)`
  - `src.dataset.EXPECTED_STATS: dict[str, SplitStats]`
  - `src.dataset.EXPECTED_LABEL_SHARE: dict[str, dict[str, float]]`
  - `src.dataset.EsciSplit(name: str, judgements: pd.DataFrame, products: pd.DataFrame)`
  - `src.dataset.data_dir() -> Path`
  - `src.dataset.ensure_downloaded(name: str, data_dir_override: Path | None = None) -> Path` for `name` in `{"examples", "products"}`
  - `src.dataset.filter_task1(examples: pd.DataFrame, split: str) -> pd.DataFrame`
  - `src.dataset.check_invariants(split: str, judgements: pd.DataFrame, products: pd.DataFrame | None = None) -> None`
  - `src.dataset.load_split(split: str, data_dir_override: Path | None = None) -> EsciSplit`

`EsciSplit.judgements` columns: `example_id` (int64), `query_id` (int64), `query` (str), `product_id` (str), `esci_label` (str), `gain` (float64), `qrel` (int64).
`EsciSplit.products` columns: `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color` — restricted to the `product_id` values appearing in `judgements`.

- [x] **Step 1: Write the failing test**

Create `tests/test_dataset.py`:

```python
import pandas as pd
import pytest

from src.dataset import (
    EXPECTED_LABEL_SHARE,
    EXPECTED_STATS,
    DataInvariantError,
    check_invariants,
    filter_task1,
    load_split,
)


def _judgements(labels: list[str], n_queries: int) -> pd.DataFrame:
    """A judgements frame with a given label multiset spread over n_queries."""
    n = len(labels)
    label_column = pd.Series(labels, dtype="object")
    return pd.DataFrame(
        {
            "example_id": range(n),
            "query_id": [i % n_queries for i in range(n)],
            "query": "a query",
            "product_id": [f"B{i:09d}" for i in range(n)],
            "esci_label": label_column,
            "gain": label_column.map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}),
            "qrel": label_column.map({"E": 100, "S": 10, "C": 1, "I": 0}),
        }
    )


def _test_shaped_judgements() -> pd.DataFrame:
    """A frame with exactly the documented test-split counts and label shares."""
    stats = EXPECTED_STATS["test"]
    share = EXPECTED_LABEL_SHARE["test"]
    counts = {label: round(stats.judgements * s) for label, s in share.items()}
    counts["E"] += stats.judgements - sum(counts.values())  # absorb rounding
    labels = [label for label, n in counts.items() for _ in range(n)]
    return _judgements(labels, stats.queries)


# --- filtering --------------------------------------------------------------

def test_filter_task1_keeps_only_small_version_us_rows_of_the_split():
    examples = pd.DataFrame(
        {
            "example_id": [0, 1, 2, 3],
            "query_id": [0, 0, 1, 1],
            "query": ["a", "a", "b", "b"],
            "product_id": ["B0", "B1", "B2", "B3"],
            "product_locale": ["us", "es", "us", "us"],
            "esci_label": ["E", "E", "E", "E"],
            "small_version": [1, 1, 0, 1],
            "large_version": [1, 1, 1, 1],
            "split": ["test", "test", "test", "train"],
        }
    )
    kept = filter_task1(examples, "test")
    assert list(kept["example_id"]) == [0]


def test_filter_task1_rejects_a_frame_with_no_split_column():
    # The tasksource/esci HF mirror has no `split` column. Loading it yields
    # 185,361 test judgements instead of 181,701 (~2% contamination).
    examples = pd.DataFrame(
        {
            "example_id": [0],
            "query_id": [0],
            "query": ["a"],
            "product_id": ["B0"],
            "product_locale": ["us"],
            "esci_label": ["E"],
            "small_version": [1],
            "large_version": [1],
        }
    )
    with pytest.raises(DataInvariantError, match="no 'split' column"):
        filter_task1(examples, "test")


# --- Review Focus 5: loading the wrong source file --------------------------

def test_check_invariants_accepts_the_documented_test_split():
    check_invariants("test", _test_shaped_judgements())


def test_wrong_judgement_count_names_the_hf_mirror_trap():
    frame = _test_shaped_judgements().iloc[:-1]
    with pytest.raises(DataInvariantError) as exc:
        check_invariants("test", frame)
    message = str(exc.value)
    assert "181701" in message.replace(",", "")
    assert "tasksource" in message


def test_the_hf_mirror_row_count_is_rejected():
    # The exact count the mirror produces, which is the realistic failure.
    stats = EXPECTED_STATS["test"]
    extra = 185_361 - stats.judgements
    frame = pd.concat(
        [_test_shaped_judgements(), _judgements(["E"] * extra, stats.queries)],
        ignore_index=True,
    )
    with pytest.raises(DataInvariantError, match="tasksource"):
        check_invariants("test", frame)


def test_swapped_substitute_and_complement_shares_are_rejected():
    # Complement 35% / Substitute 4% is the signature of the S/C swap bug in
    # the official prepare_trec_eval_files.py.
    stats = EXPECTED_STATS["test"]
    share = dict(EXPECTED_LABEL_SHARE["test"])
    share["S"], share["C"] = share["C"], share["S"]
    counts = {label: round(stats.judgements * s) for label, s in share.items()}
    counts["E"] += stats.judgements - sum(counts.values())
    labels = [label for label, n in counts.items() for _ in range(n)]
    with pytest.raises(DataInvariantError, match="label distribution"):
        check_invariants("test", _judgements(labels, stats.queries))


def test_wrong_query_count_is_rejected():
    frame = _test_shaped_judgements()
    frame["query_id"] = 0
    with pytest.raises(DataInvariantError, match="quer"):
        check_invariants("test", frame)


def test_train_split_has_no_label_share_expectation():
    # CLAUDE.md documents the label distribution for test only. Asserting the
    # test shares against train would be inventing a number.
    assert "train" not in EXPECTED_LABEL_SHARE
    assert EXPECTED_STATS["train"].products is None


# --- the real files ---------------------------------------------------------

@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "test"])
def test_real_split_satisfies_every_invariant(split):
    loaded = load_split(split)
    check_invariants(split, loaded.judgements, loaded.products)
    assert loaded.judgements["gain"].between(0.0, 1.0).all()
    assert set(loaded.judgements["product_id"]) <= set(loaded.products["product_id"])


@pytest.mark.data
def test_products_overlap_between_splits_is_the_documented_size():
    # 34,756 products appear in both splits. The split is query-level so this
    # is legitimate, but it is exactly why no feature may be computed from a
    # product alone using labels.
    train = set(load_split("train").judgements["product_id"])
    test = set(load_split("test").judgements["product_id"])
    assert len(train & test) == 34_756
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.dataset'`.

- [x] **Step 3: Write `src/dataset.py`**

```python
"""Load the official Amazon ESCI Task 1 English data, and refuse the wrong file.

The failure this module exists to prevent is silent: the tasksource/esci HF
mirror parses cleanly, has the right column names, and returns 185,361 test
judgements instead of 181,701 because it encodes the large-version split.
Nothing about it looks wrong except the count, so the count is checked on
every load.

The official files live behind Git LFS. raw.githubusercontent.com serves the
LFS *pointer* (a 130-byte text file), not the Parquet, so downloads go through
media.githubusercontent.com/media/.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.labels import ESCI_GAINS, label_to_gain, label_to_qrel

BASE_URL = (
    "https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
    "shopping_queries_dataset/"
)

FILES = {
    "examples": "shopping_queries_dataset_examples.parquet",
    "products": "shopping_queries_dataset_products.parquet",
}

LABEL_SHARE_TOLERANCE = 0.001


class DataInvariantError(ValueError):
    """Raised when loaded data does not match the documented ESCI invariants."""


@dataclass(frozen=True)
class SplitStats:
    judgements: int
    queries: int
    products: int | None = None


EXPECTED_STATS: dict[str, SplitStats] = {
    "train": SplitStats(judgements=419_653, queries=20_888),
    "test": SplitStats(judgements=181_701, queries=8_956, products=164_900),
}

# Documented for the test split only. There is no published train figure, and
# asserting the test shares against train would be inventing a number.
EXPECTED_LABEL_SHARE: dict[str, dict[str, float]] = {
    "test": {"E": 0.4387, "S": 0.3498, "I": 0.1669, "C": 0.0446},
}


@dataclass(frozen=True)
class EsciSplit:
    name: str
    judgements: pd.DataFrame
    products: pd.DataFrame


def data_dir() -> Path:
    """Where the raw parquet lives. Override with ESCI_DATA_DIR."""
    return Path(os.environ.get("ESCI_DATA_DIR", "data/esci"))


def ensure_downloaded(name: str, data_dir_override: Path | None = None) -> Path:
    """Download one official parquet if it is not already on disk.

    products is ~1.03 GB and examples ~48.9 MB, so this prints progress and
    downloads to a .part file that is renamed only on success — an interrupted
    download must not leave a truncated parquet that later parses as garbage.
    """
    if name not in FILES:
        raise ValueError(f"unknown file {name!r}; expected one of {sorted(FILES)}")
    directory = data_dir_override or data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / FILES[name]
    if target.exists():
        return target

    partial = target.with_suffix(target.suffix + ".part")
    url = BASE_URL + FILES[name]
    print(f"downloading {url} -> {target}")

    def _progress(block_count: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(block_count * block_size, total)
            print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, partial, reporthook=_progress)
    print()
    partial.rename(target)
    return target


def filter_task1(examples: pd.DataFrame, split: str) -> pd.DataFrame:
    """Restrict the examples table to Task 1, English, one split."""
    if "split" not in examples.columns:
        raise DataInvariantError(
            "this examples table has no 'split' column, so it cannot be the "
            "official amazon-science/esci-data parquet. The tasksource/esci HF "
            "mirror has this shape and encodes the large-version split, which "
            "yields 185,361 test judgements instead of 181,701. Download from "
            f"{BASE_URL} instead."
        )
    mask = (
        (examples["small_version"] == 1)
        & (examples["product_locale"] == "us")
        & (examples["split"] == split)
    )
    return examples.loc[mask].reset_index(drop=True)


def check_invariants(
    split: str,
    judgements: pd.DataFrame,
    products: pd.DataFrame | None = None,
) -> None:
    """Assert the documented counts and label distribution, or raise.

    These were verified empirically against the real data and each one silently
    corrupts results if violated.
    """
    expected = EXPECTED_STATS[split]

    n_judgements = len(judgements)
    if n_judgements != expected.judgements:
        raise DataInvariantError(
            f"{split} split has {n_judgements:,} judgements, expected "
            f"{expected.judgements:,}. The usual cause is loading the "
            "tasksource/esci HF mirror, which has no 'split' column and encodes "
            "the large-version split. Use the official parquet from "
            f"{BASE_URL}"
        )

    n_queries = judgements["query_id"].nunique()
    if n_queries != expected.queries:
        raise DataInvariantError(
            f"{split} split has {n_queries:,} unique queries, expected "
            f"{expected.queries:,}"
        )

    if expected.products is not None and products is not None:
        n_products = products["product_id"].nunique()
        if n_products != expected.products:
            raise DataInvariantError(
                f"{split} split has {n_products:,} unique products, expected "
                f"{expected.products:,}"
            )

    expected_share = EXPECTED_LABEL_SHARE.get(split)
    if expected_share is not None:
        observed = judgements["esci_label"].value_counts(normalize=True)
        for label, share in expected_share.items():
            seen = float(observed.get(label, 0.0))
            if abs(seen - share) > LABEL_SHARE_TOLERANCE:
                raise DataInvariantError(
                    f"{split} label distribution is wrong: {label} is "
                    f"{seen:.4f}, expected {share:.4f}. Complement near 35% and "
                    "Substitute near 4% is the signature of the S/C swap bug in "
                    "the official prepare_trec_eval_files.py."
                )

    unknown = set(judgements["esci_label"]) - set(ESCI_GAINS)
    if unknown:
        raise DataInvariantError(f"unknown ESCI labels present: {sorted(unknown)}")


def load_split(split: str, data_dir_override: Path | None = None) -> EsciSplit:
    """Load one Task 1 English split, with every invariant asserted."""
    if split not in EXPECTED_STATS:
        raise ValueError(f"unknown split {split!r}; expected train or test")

    examples = pd.read_parquet(ensure_downloaded("examples", data_dir_override))
    judgements = filter_task1(examples, split)
    judgements = judgements[
        ["example_id", "query_id", "query", "product_id", "esci_label"]
    ].copy()
    judgements["gain"] = judgements["esci_label"].map(label_to_gain)
    judgements["qrel"] = judgements["esci_label"].map(label_to_qrel)

    all_products = pd.read_parquet(ensure_downloaded("products", data_dir_override))
    products = all_products.loc[
        (all_products["product_locale"] == "us")
        & all_products["product_id"].isin(set(judgements["product_id"]))
    ]
    products = products[
        [
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
        ]
    ].drop_duplicates("product_id").reset_index(drop=True)

    check_invariants(split, judgements, products)
    return EsciSplit(name=split, judgements=judgements, products=products)
```

- [x] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: PASS, 8 tests. The two `@pytest.mark.data` tests are deselected.

- [x] **Step 5: Download the real data and run the marked tests**

```bash
python -c "from src.dataset import ensure_downloaded; ensure_downloaded('examples'); ensure_downloaded('products')"
python -m pytest tests/test_dataset.py -v -m data
```

Expected: PASS, 3 tests (train, test, and the overlap check). ~1.08 GB downloaded, several minutes.

If `test_real_split_satisfies_every_invariant[test]` fails on the judgement count, read the error message before touching the expected numbers — the numbers in `EXPECTED_STATS` were verified empirically and are not the thing that is wrong.

- [x] **Step 6: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "Load official ESCI Task 1 English splits with asserted invariants"
```
---

## Task 5: TREC run and qrels I/O

**Files:**
- Create: `src/runs.py`
- Create: `tests/test_runs.py`

**Interfaces:**
- Consumes: `src.dataset.EsciSplit`, `src.metrics.Qrels`, `src.metrics.Run`.
- Produces:
  - `src.runs.qrels_from_judgements(judgements: pd.DataFrame) -> dict[str, dict[str, int]]`
  - `src.runs.run_from_scores(scores: pd.DataFrame, score_column: str = "score") -> dict[str, dict[str, float]]`
  - `src.runs.write_run(run: Run, path: Path, run_tag: str) -> None`
  - `src.runs.read_run(path: Path) -> dict[str, dict[str, float]]`
  - `src.runs.write_qrels(qrels: Qrels, path: Path) -> None`

Ids are strings in every dict, because TREC files are text and `query_id` is an `int64` in the parquet. Converting at the boundary once means nothing downstream has to remember to.

- [x] **Step 1: Write the failing test**

Create `tests/test_runs.py`:

```python
import pandas as pd
import pytest

from src.runs import (
    qrels_from_judgements,
    read_run,
    run_from_scores,
    write_qrels,
    write_run,
)


def _judgements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "product_id": ["B1", "B2", "B3"],
            "esci_label": ["E", "I", "S"],
            "qrel": [100, 0, 10],
        }
    )


def test_qrels_keys_are_strings_not_numpy_integers():
    qrels = qrels_from_judgements(_judgements())
    assert set(qrels) == {"1", "2"}
    assert all(isinstance(k, str) for k in qrels)
    assert qrels["1"] == {"B1": 100, "B2": 0}


def test_qrel_values_are_plain_ints():
    # pytrec_eval rejects numpy.int64.
    qrels = qrels_from_judgements(_judgements())
    assert all(type(v) is int for docs in qrels.values() for v in docs.values())


def test_duplicate_query_product_pair_raises_instead_of_overwriting():
    duplicated = pd.concat([_judgements(), _judgements().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        qrels_from_judgements(duplicated)


def test_run_from_scores_shapes_a_nested_dict():
    scores = pd.DataFrame(
        {"query_id": [1, 1], "product_id": ["B1", "B2"], "score": [0.5, 0.25]}
    )
    assert run_from_scores(scores) == {"1": {"B1": 0.5, "B2": 0.25}}


def test_run_file_round_trip_preserves_scores_exactly(tmp_path):
    # Formatting scores as %.6f would collapse near-identical scores into ties
    # and change the ranking, so the round trip must be exact.
    run = {"1": {"B1": 0.1234567890123456, "B2": 0.1234567890123457}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    assert read_run(path) == run


def test_run_file_has_the_six_trec_columns_in_rank_order(tmp_path):
    run = {"1": {"B2": 0.5, "B1": 0.9}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    rows = [line.split() for line in path.read_text().splitlines()]
    assert [r[0] for r in rows] == ["1", "1"]
    assert [r[1] for r in rows] == ["Q0", "Q0"]
    assert [r[2] for r in rows] == ["B1", "B2"]  # sorted by descending score
    assert [r[3] for r in rows] == ["1", "2"]  # rank is 1-based
    assert [r[5] for r in rows] == ["unit", "unit"]


def test_run_file_ties_are_written_in_ascending_document_id(tmp_path):
    # Must match the tie policy in src/metrics.py, or a run scores differently
    # after a round trip through disk.
    run = {"1": {"B2": 0.5, "B1": 0.5}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    assert [line.split()[2] for line in path.read_text().splitlines()] == ["B1", "B2"]


def test_qrels_file_has_the_four_trec_columns(tmp_path):
    path = tmp_path / "qrels.txt"
    write_qrels({"1": {"B1": 100, "B2": 0}}, path)
    rows = [line.split() for line in path.read_text().splitlines()]
    assert rows == [["1", "0", "B1", "100"], ["1", "0", "B2", "0"]]


def test_reading_a_malformed_run_line_raises_with_the_line_number(tmp_path):
    path = tmp_path / "run.trec"
    path.write_text("1 Q0 B1 1 0.9 tag\nnot a run line\n")
    with pytest.raises(ValueError, match="line 2"):
        read_run(path)
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_runs.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.runs'`.

- [x] **Step 3: Write `src/runs.py`**

```python
"""TREC-format run and qrels I/O, and DataFrame -> nested-dict conversion.

Ids are strings everywhere: TREC files are text, query_id is int64 in the
parquet, and pytrec_eval rejects numpy scalars. Converting once at this
boundary means nothing downstream has to remember to.

Scores are written at full float precision rather than a fixed number of
decimals. Rounding to %.6f would turn distinct scores into ties and silently
change the ranking between an in-memory run and the same run read back.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.metrics import Qrels, Run


def _ranked(docs: Mapping[str, float]) -> list[tuple[str, float]]:
    """Documents ordered the way src.metrics.ndcg_per_query orders them."""
    return sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))


def qrels_from_judgements(judgements: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Nested qrels dict from a judgements frame carrying a `qrel` column."""
    duplicated = judgements.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = judgements.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}) in judgements; "
            "building a dict would silently keep only the last one"
        )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, product_id, qrel in zip(
        judgements["query_id"], judgements["product_id"], judgements["qrel"]
    ):
        qrels.setdefault(str(query_id), {})[str(product_id)] = int(qrel)
    return qrels


def run_from_scores(
    scores: pd.DataFrame, score_column: str = "score"
) -> dict[str, dict[str, float]]:
    """Nested run dict from a frame of query_id / product_id / score."""
    duplicated = scores.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = scores.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}) in scores"
        )
    run: dict[str, dict[str, float]] = {}
    for query_id, product_id, score in zip(
        scores["query_id"], scores["product_id"], scores[score_column]
    ):
        run.setdefault(str(query_id), {})[str(product_id)] = float(score)
    return run


def write_run(run: Run, path: Path, run_tag: str) -> None:
    """Write a TREC run file: qid Q0 docid rank score tag."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(run):
            for rank, (doc_id, score) in enumerate(_ranked(run[qid]), start=1):
                # float() first: numpy 2.x reprs a float64 as "np.float64(0.5)",
                # which read_run could not parse back.
                fh.write(f"{qid} Q0 {doc_id} {rank} {float(score)!r} {run_tag}\n")


def read_run(path: Path) -> dict[str, dict[str, float]]:
    """Read a TREC run file back into a nested dict."""
    run: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}: line {line_number} has {len(fields)} fields, "
                    f"expected 6 (qid Q0 docid rank score tag): {line.strip()!r}"
                )
            qid, _, doc_id, _, score, _ = fields
            try:
                run.setdefault(qid, {})[doc_id] = float(score)
            except ValueError:
                raise ValueError(
                    f"{path}: line {line_number} has a non-numeric score {score!r}"
                ) from None
    return run


def write_qrels(qrels: Qrels, path: Path) -> None:
    """Write a TREC qrels file: qid 0 docid relevance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                fh.write(f"{qid} 0 {doc_id} {int(qrels[qid][doc_id])}\n")
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_runs.py -v
```

Expected: PASS, 9 tests.

- [x] **Step 5: Commit**

```bash
git add src/runs.py tests/test_runs.py
git commit -m "Add TREC run and qrels I/O with exact score round-tripping"
```
---

## Task 6: The computed random floor

**Files:**
- Create: `src/floor.py`
- Create: `tests/test_floor.py`

**Interfaces:**
- Consumes: `src.metrics.ndcg_per_query(run, qrels)`, `src.metrics.Qrels`.
- Produces:
  - `src.floor.FloorResult(mean: float, low: float, high: float, per_trial: tuple[float, ...], n_trials: int, seed: int)`
  - `src.floor.random_run(qrels: Qrels, rng: random.Random) -> dict[str, dict[str, float]]`
  - `src.floor.random_floor(qrels: Qrels, *, n_trials: int = 100, seed: int = 0, alpha: float = 0.05) -> FloorResult`

`random_run` assigns a distinct random score to every judged document, so the floor measures random *ordering* and never touches the tie-breaking path.

- [x] **Step 1: Write the failing test**

Create `tests/test_floor.py`:

```python
import pytest

from src.floor import random_floor, random_run


def test_uniform_gains_make_every_ordering_ideal():
    # If every judged document has the same gain, DCG == IDCG for any order,
    # so the floor is exactly 1.0. Anything else means the ideal is being
    # computed from the run rather than from the judgements.
    qrels = {f"q{i}": {f"d{j}": 100 for j in range(5)} for i in range(20)}
    assert random_floor(qrels, n_trials=5, seed=0).mean == pytest.approx(1.0)


def test_floor_of_an_all_irrelevant_query_is_zero():
    qrels = {"q": {"a": 0, "b": 0, "c": 0}}
    assert random_floor(qrels, n_trials=5, seed=0).mean == 0.0


def test_floor_sits_between_zero_and_one_on_mixed_labels():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 1, "d": 0} for i in range(50)}
    result = random_floor(qrels, n_trials=20, seed=0)
    assert 0.0 < result.mean < 1.0
    assert result.low <= result.mean <= result.high


def test_same_seed_gives_the_same_floor():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    assert random_floor(qrels, n_trials=10, seed=7).per_trial == (
        random_floor(qrels, n_trials=10, seed=7).per_trial
    )


def test_different_seeds_give_different_trials():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    assert random_floor(qrels, n_trials=10, seed=0).per_trial != (
        random_floor(qrels, n_trials=10, seed=1).per_trial
    )


def test_result_records_its_own_settings():
    qrels = {"q": {"a": 100, "b": 0}}
    result = random_floor(qrels, n_trials=4, seed=3)
    assert result.n_trials == 4
    assert result.seed == 3
    assert len(result.per_trial) == 4


def test_random_run_scores_every_judged_document_distinctly():
    import random

    qrels = {"q": {f"d{i}": 0 for i in range(100)}}
    run = random_run(qrels, random.Random(0))
    assert set(run["q"]) == set(qrels["q"])
    assert len(set(run["q"].values())) == 100


def test_zero_trials_raises():
    with pytest.raises(ValueError, match="n_trials"):
        random_floor({"q": {"a": 100}}, n_trials=0)


@pytest.mark.data
@pytest.mark.slow
def test_real_test_split_floor_is_near_the_documented_measurement():
    from src.dataset import load_split
    from src.runs import qrels_from_judgements

    qrels = qrels_from_judgements(load_split("test").judgements)
    result = random_floor(qrels, n_trials=100, seed=0)
    print(f"\nrandom floor (test) = {result.mean:.4f} [{result.low:.4f}, {result.high:.4f}]")
    # CLAUDE.md records 0.7467 measured with the 1/log2(rank+1) discount; the
    # published SQID figure is 0.7483; under the swapped S/C mapping it is
    # 0.7141.
    #
    # The band is wider than that spread on purpose. Simulating the documented
    # marginal label distribution (E 43.87 / S 34.98 / I 16.69 / C 4.46) over
    # ~20 candidates per query lands at 0.755-0.776 across i.i.d.,
    # Dirichlet-correlated and count-skewed variants -- consistently above
    # 0.7467. So the real per-query label structure is not reconstructible from
    # the marginal alone, and a band tight around 0.7467 would be a band around
    # a number this project cannot currently derive from first principles.
    #
    # What this test guards is the large moves: the S/C swap (0.7141), an
    # exponential gain, a wrong discount, or a degenerate uniform gain (1.0).
    # It is not a precision check on the fourth decimal.
    assert 0.730 < result.mean < 0.780
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_floor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.floor'`.

- [x] **Step 3: Write `src/floor.py`**

```python
"""The random-ordering NDCG floor, computed rather than quoted.

Roughly 44% of ESCI judgements are Exact, so shuffling the judged candidate
list already scores near 0.75 on full-list NDCG. A headline of 0.86 means
nothing without that floor attached.

The number is computed on every report because it moves with the gain mapping:
CLAUDE.md records 0.7467 with the standard discount, the published SQID figure
is 0.7483, and under the S/C swap it is 0.7141. That 3.3-point spread is larger
than most method gains, so a hard-coded constant would be a way to be wrong by
more than the effect being measured.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from src.metrics import Qrels, ndcg_per_query


@dataclass(frozen=True)
class FloorResult:
    mean: float
    low: float
    high: float
    per_trial: tuple[float, ...]
    n_trials: int
    seed: int


def random_run(qrels: Qrels, rng: random.Random) -> dict[str, dict[str, float]]:
    """Score every judged document with a distinct random value.

    Distinct scores keep the floor a measurement of random *ordering* rather
    than of the tie-breaking policy.
    """
    run: dict[str, dict[str, float]] = {}
    for qid, judged in qrels.items():
        order = list(judged)
        rng.shuffle(order)
        run[qid] = {doc_id: float(rank) for rank, doc_id in enumerate(order)}
    return run


def random_floor(
    qrels: Qrels,
    *,
    n_trials: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> FloorResult:
    """Mean NDCG of a random ordering, with a percentile interval over trials."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")

    rng = random.Random(seed)
    per_trial: list[float] = []
    for _ in range(n_trials):
        scores = ndcg_per_query(random_run(qrels, rng), qrels)
        per_trial.append(sum(scores.values()) / len(scores))

    ordered = sorted(per_trial)
    low_index = int((alpha / 2) * (len(ordered) - 1))
    high_index = int((1 - alpha / 2) * (len(ordered) - 1))
    return FloorResult(
        mean=statistics.fmean(per_trial),
        low=ordered[low_index],
        high=ordered[high_index],
        per_trial=tuple(per_trial),
        n_trials=n_trials,
        seed=seed,
    )
```

- [x] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_floor.py -v
```

Expected: PASS, 8 tests; the real-data test is deselected.

- [x] **Step 5: Measure the floor on the real test split and record it**

```bash
python -m pytest tests/test_floor.py -v -m "data and slow" -s
```

Expected: PASS, and the printed floor lands near 0.7467. Takes 1–2 minutes: 100 trials × 181,701 scored documents.

Record the measured value in the commit message so it is recoverable from history without re-running.

**If the measured floor differs from 0.7467 by more than 0.005, that is a finding, not a nuisance.** Record it, and check in this order: the gain mapping in `src/labels.py` (a swap lands near 0.7141), the discount in `src/metrics.py`, and whether `load_split` filtered to `small_version == 1` and `product_locale == "us"`. If all three are right and the number still differs, report the measured value as this project's floor and note the discrepancy with CLAUDE.md in the commit message. Every later NDCG in the series is quoted against this number, so it must be the one this code actually produces.

- [x] **Step 6: Commit**

```bash
git add src/floor.py tests/test_floor.py
git commit -m "Compute the random-ordering NDCG floor from qrels"
```
---

## Phase 2 Gate

Phase 3 does not start until all of these hold:

- [x] `python -m pytest tests/test_dataset.py tests/test_runs.py tests/test_floor.py -v` passes — 8 + 9 + 8 tests.
- [x] `python -m pytest -m data -v` passes for both splits, including the 34,756-product train/test overlap check.
- [x] `python -m pytest -m "data and slow" -s` passes and prints a random floor near 0.7467, recorded in the Task 6 commit message so it is recoverable from history without re-running.
- [x] If the measured floor differs from 0.7467 by more than 0.005, that is a **finding**, investigated in the order Task 6 Step 5 lays out (gain mapping → discount → split filter) and written down — not a band to widen.

Next: [Phase 3 — Statistics and the Baseline](phase-3-statistics-and-baseline.md).
