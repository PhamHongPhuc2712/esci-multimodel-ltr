# Phase 1 — The Metric and the Lexical Channel

**Plan 4 of 7 · Phase 1 of 3 · Tasks 1–2.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into.

**Delivers:** `src/recall.py` and `src/bm25_index.py` — Recall@k with an
explicit answer for queries that have no relevant product, and a lexical
channel that builds once and memory-maps thereafter.

**Needs on disk:** `data/combined/products.parquet` and
`data/combined/judgements.parquet` from Plan 2. The index build peaks at
14 GB of RAM for about four minutes and writes 1.1 GB.

**Owns Review Focus items 1 and 2** (a query with no relevant product, row
indices mistaken for product ids).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Build once, then memory-map.** In-process build-and-query peaks at 18.3 GB against 23 GB of RAM. Saved and reopened with `mmap=True` it is 1.55 GB and *faster*.
- **`retrieve` returns row indices, not product ids.** The id array is a separate file; if the two are ever built from differently-ordered frames every result is silently wrong.
- **One test query has no Exact product.** `query_id` 45928 is 15 judgements, all S.
- **Recall's baseline is BM25 R@100 = 0.5018**, not the NDCG random floor.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: Recall@k and its ground truth

**Files:**
- Create: `src/recall.py`
- Create: `tests/test_recall.py`

**Interfaces:**
- Consumes: `data/combined/judgements.parquet` (columns `query_id`, `query`, `product_id`, `esci_label`, `split`, `fold`).
- Produces:
  - `src.recall.DEFAULT_JUDGEMENTS: Path`, `RELEVANT_SETS: dict[str, tuple[str, ...]]`, `DEFAULT_KS: tuple[int, ...]` (10, 50, 100, 500, 1000)
  - `src.recall.RecallResult(per_query, n_queries, n_skipped, k)` with `.mean -> float` and `.to_dict()`
  - `src.recall.relevant_sets(judgements, relevance="E") -> dict[int, set[str]]`
  - `src.recall.recall_at_k(retrieved, relevant, k) -> RecallResult`
  - `src.recall.recall_table(retrieved, relevant, ks=DEFAULT_KS) -> dict[int, RecallResult]`
  - `src.recall.load_queries(split, judgements_path=..., folds=None) -> pd.DataFrame` with columns `query_id`, `query`
  - `src.recall.Channel` — a `typing.Protocol` with `.search(queries: Sequence[str], k: int) -> list[list[str]]`

`recall_at_k` takes already-retrieved lists rather than a channel, so the
metric is tested in milliseconds against hand-written lists and no index has
to exist yet. The `Channel` protocol lives here because it is the contract the
metric consumes; Tasks 2, 3 and 4 each implement it.

**The empty-relevant-set decision, made explicitly.** A query with no relevant
product has recall 0/0. This plan **skips** such queries and reports
`n_skipped` alongside the mean, because the alternative — scoring 0.0 — makes
a retriever look worse for a query where no retriever could have scored, and
because 1 of 8,956 test queries hits it under the E definition. Reporting
`n_skipped` is what stops two runs with different denominators being compared
as if they were the same measurement.

- [ ] **Step 1: Write the failing test**

Create `tests/test_recall.py`:

```python
import pandas as pd
import pytest

from src.recall import (
    DEFAULT_KS,
    RELEVANT_SETS,
    recall_at_k,
    recall_table,
    relevant_sets,
)


def _judgements(rows):
    return pd.DataFrame(rows)


# --- the ground truth -------------------------------------------------------

def test_relevant_sets_default_to_exact_only():
    frame = _judgements([
        {"query_id": 1, "product_id": "a", "esci_label": "E"},
        {"query_id": 1, "product_id": "b", "esci_label": "S"},
        {"query_id": 1, "product_id": "c", "esci_label": "I"},
    ])
    assert relevant_sets(frame) == {1: {"a"}}


def test_relevant_sets_can_include_substitutes():
    frame = _judgements([
        {"query_id": 1, "product_id": "a", "esci_label": "E"},
        {"query_id": 1, "product_id": "b", "esci_label": "S"},
        {"query_id": 1, "product_id": "c", "esci_label": "C"},
    ])
    assert relevant_sets(frame, relevance="E+S") == {1: {"a", "b"}}


def test_relevant_sets_rejects_an_unknown_relevance():
    frame = _judgements([{"query_id": 1, "product_id": "a", "esci_label": "E"}])
    with pytest.raises(ValueError, match="relevance"):
        relevant_sets(frame, relevance="everything")


def test_the_relevance_definitions_are_the_spec_s():
    assert RELEVANT_SETS == {"E": ("E",), "E+S": ("E", "S")}


def test_a_query_with_no_relevant_product_is_absent_from_the_sets():
    # query_id 45928 in the real test split: 15 judgements, all S, no E.
    frame = _judgements([
        {"query_id": 45928, "product_id": "a", "esci_label": "S"},
        {"query_id": 45928, "product_id": "b", "esci_label": "S"},
    ])
    assert relevant_sets(frame) == {}
    assert relevant_sets(frame, relevance="E+S") == {45928: {"a", "b"}}


# --- the metric -------------------------------------------------------------

def test_recall_counts_the_share_of_relevant_products_found():
    result = recall_at_k({1: ["x", "a", "y", "b"]}, {1: {"a", "b", "c"}}, k=4)
    assert result.mean == pytest.approx(2 / 3)


def test_recall_respects_k():
    # "b" sits at rank 4 and must not count at k=2.
    result = recall_at_k({1: ["x", "a", "y", "b"]}, {1: {"a", "b"}}, k=2)
    assert result.mean == pytest.approx(0.5)


def test_recall_of_a_perfect_retrieval_is_one():
    result = recall_at_k({1: ["a", "b"]}, {1: {"a", "b"}}, k=10)
    assert result.mean == pytest.approx(1.0)


def test_recall_averages_over_queries_not_over_products():
    # A query with 1 relevant found and a query with 0 of 10 found average to
    # 0.5, not to 1/11. Macro-averaging is what the spec's Recall@k means.
    result = recall_at_k(
        {1: ["a"], 2: []},
        {1: {"a"}, 2: {f"p{i}" for i in range(10)}},
        k=10,
    )
    assert result.mean == pytest.approx(0.5)
    assert result.n_queries == 2


def test_a_duplicate_in_the_retrieved_list_is_not_counted_twice():
    result = recall_at_k({1: ["a", "a"]}, {1: {"a", "b"}}, k=10)
    assert result.mean == pytest.approx(0.5)


# --- Review Focus 1: a query with no relevant product -----------------------

def test_a_query_with_no_relevant_product_is_skipped_not_scored_zero():
    # 0/0 is undefined. Scoring it 0.0 punishes a retriever for a query where
    # no retrieval could have scored, and drags the reported mean down.
    result = recall_at_k({1: ["a"], 2: ["x"]}, {1: {"a"}}, k=10)
    assert result.mean == pytest.approx(1.0)
    assert result.n_queries == 1
    assert result.n_skipped == 1


def test_the_skipped_count_is_reported_so_denominators_are_comparable():
    result = recall_at_k({1: ["a"], 2: ["x"]}, {1: {"a"}}, k=10)
    payload = result.to_dict()
    assert payload["n_queries"] == 1
    assert payload["n_skipped"] == 1
    assert payload["k"] == 10


def test_a_query_whose_relevant_set_is_empty_is_also_skipped():
    result = recall_at_k({1: ["a"]}, {1: set()}, k=10)
    assert result.n_queries == 0
    assert result.n_skipped == 1


def test_every_query_being_skipped_gives_a_nan_mean_not_a_zero():
    # A silent 0.0 here would read as "the retriever found nothing" when the
    # truth is "nothing was measurable".
    import math

    result = recall_at_k({1: ["a"]}, {}, k=10)
    assert result.n_queries == 0
    assert math.isnan(result.mean)


# --- the table --------------------------------------------------------------

def test_recall_table_covers_every_k():
    table = recall_table({1: ["a", "b", "c"]}, {1: {"a", "c"}})
    assert set(table) == set(DEFAULT_KS)
    assert table[10].mean == pytest.approx(1.0)


def test_recall_is_monotonic_in_k():
    retrieved = {1: [f"p{i}" for i in range(200)]}
    relevant = {1: {"p5", "p150"}}
    table = recall_table(retrieved, relevant, ks=(10, 100, 1000))
    assert table[10].mean <= table[100].mean <= table[1000].mean
    assert table[10].mean == pytest.approx(0.5)
    assert table[1000].mean == pytest.approx(1.0)


def test_per_query_scores_are_keyed_for_the_bootstrap():
    # src.bootstrap.paired_delta_ci pairs on the mapping's keys, so the keys
    # must be the query ids and must survive as strings or ints consistently.
    result = recall_at_k({1: ["a"], 2: ["b"]}, {1: {"a"}, 2: {"b"}}, k=10)
    assert set(result.per_query) == {1, 2}
    assert result.per_query[1] == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_recall.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.recall'`.

- [ ] **Step 3: Write `src/recall.py`**

```python
"""Recall@k over the full corpus, and the relevant sets it is measured against.

PROJECT_SPEC.md §2 is explicit that this is a re-ranking task and that recall
is evaluated *separately*: Recall@k over the ~1.2M corpus, NDCG over the judged
candidate list. This module owns the first and knows nothing about ranking.

Two decisions are made here rather than left to callers, because both are
silent when got wrong:

  * **A query with no relevant product is skipped, and the skip is counted.**
    Its recall is 0/0. Scored as 0.0 it punishes a retriever for a query no
    retriever could have scored; dropped without a count, two runs with
    different denominators look comparable. `query_id` 45928 of the real test
    split - 15 judgements, every one Substitute - is why this is not
    hypothetical.
  * **The mean is macro-averaged over queries**, not micro-averaged over
    products, so a query with 40 relevant products does not outvote one with
    three.

`recall_at_k` takes already-retrieved lists rather than a Channel, so the
metric is tested against hand-written lists in milliseconds and no index has
to exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

DEFAULT_JUDGEMENTS = Path("data/combined/judgements.parquet")

# The spec's two reasonable definitions of "relevant" for retrieval. E alone is
# the strict reading; E+S matches what a shopper would accept and is the only
# definition under which every test query has a non-empty set.
RELEVANT_SETS: dict[str, tuple[str, ...]] = {"E": ("E",), "E+S": ("E", "S")}

DEFAULT_KS: tuple[int, ...] = (10, 50, 100, 500, 1000)


@runtime_checkable
class Channel(Protocol):
    """A retrieval channel: queries in, ranked product_ids out."""

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        """One ranked list of product_ids per query, best first, at most k."""


@dataclass(frozen=True)
class RecallResult:
    per_query: dict[int, float] = field(default_factory=dict)
    n_queries: int = 0
    n_skipped: int = 0
    k: int = 0

    @property
    def mean(self) -> float:
        """Macro-average over the queries that had a relevant product.

        NaN rather than 0.0 when nothing was measurable: a silent zero reads
        as "the retriever found nothing" when the truth is "nothing was
        measurable", and the two call for opposite reactions.
        """
        if not self.per_query:
            return math.nan
        return sum(self.per_query.values()) / len(self.per_query)

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "recall": self.mean,
            "n_queries": self.n_queries,
            "n_skipped": self.n_skipped,
        }


def relevant_sets(
    judgements: pd.DataFrame, relevance: str = "E"
) -> dict[int, set[str]]:
    """product_ids counted as relevant, per query_id.

    Queries whose set would be empty are omitted entirely rather than mapped
    to an empty set, so callers cannot accidentally divide by zero.
    """
    if relevance not in RELEVANT_SETS:
        raise ValueError(
            f"unknown relevance {relevance!r}; expected one of "
            f"{tuple(RELEVANT_SETS)}"
        )
    labels = RELEVANT_SETS[relevance]
    matching = judgements.loc[judgements["esci_label"].isin(labels)]
    grouped = matching.groupby("query_id")["product_id"].apply(set)
    return {int(qid): products for qid, products in grouped.items() if products}


def recall_at_k(
    retrieved: Mapping[int, Sequence[str]],
    relevant: Mapping[int, set[str]],
    k: int,
) -> RecallResult:
    """Share of each query's relevant products found in its top k."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    per_query: dict[int, float] = {}
    skipped = 0
    for query_id, ranked in retrieved.items():
        wanted = relevant.get(query_id)
        if not wanted:
            skipped += 1
            continue
        # set() de-duplicates: a channel that returns the same product twice
        # must not be credited twice.
        found = set(ranked[:k]) & wanted
        per_query[query_id] = len(found) / len(wanted)

    return RecallResult(
        per_query=per_query,
        n_queries=len(per_query),
        n_skipped=skipped,
        k=k,
    )


def recall_table(
    retrieved: Mapping[int, Sequence[str]],
    relevant: Mapping[int, set[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[int, RecallResult]:
    """Recall at each k, from one set of retrieved lists."""
    return {k: recall_at_k(retrieved, relevant, k) for k in ks}


def load_queries(
    split: str,
    judgements_path: Path = DEFAULT_JUDGEMENTS,
    folds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """One row per distinct query in `split`, optionally limited to folds.

    `folds` is how model selection stays off the test split: pass the frozen
    validation folds from Plan 1 and tune against those.
    """
    frame = pd.read_parquet(
        judgements_path, columns=["query_id", "query", "split", "fold"]
    )
    frame = frame.loc[frame["split"] == split]
    if folds is not None:
        frame = frame.loc[frame["fold"].isin(list(folds))]
    return (
        frame.drop_duplicates("query_id")[["query_id", "query"]]
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_recall.py -v
```

Expected: PASS, 17 tests.

- [ ] **Step 5: Commit**

```bash
git add src/recall.py tests/test_recall.py
git commit -m "Add Recall@k with explicit handling for queries with no relevant product"
```

---

## Task 2: The lexical channel

**Files:**
- Create: `src/bm25_index.py`
- Create: `tests/test_bm25_index.py`
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `src.recall.Channel`, `src.baseline_sbert.product_text`, `data/combined/products.parquet`.
- Produces:
  - `src.bm25_index.DEFAULT_INDEX_DIR: Path` (`data/bm25`), `DEFAULT_PRODUCTS: Path`, `IDS_NAME: str` (`product_ids.npy`), `DEFAULT_FIELDS: tuple[str, ...]`
  - `src.bm25_index.tokenize_texts(texts, stemmer=None) -> Any`
  - `src.bm25_index.build_index(products, fields=DEFAULT_FIELDS, index_dir=DEFAULT_INDEX_DIR) -> int`
  - `src.bm25_index.Bm25Channel` with `.search(queries, k) -> list[list[str]]` and `.n_products -> int`
  - `src.bm25_index.open_channel(index_dir=DEFAULT_INDEX_DIR) -> Bm25Channel`
  - `python -m src.bm25_index --products ... --index-dir ...`

The index and the id array are written by `build_index` in **one function from
one frame**, and `open_channel` asserts their lengths match. That assertion is
the whole defence against Review Focus 2: `bm25s.retrieve` returns positions
into the matrix, and a mismatched id array turns every result into a
plausible-looking wrong answer.

Measured on the real corpus: tokenize 151 s, index 90 s, peak 14.0 GB with the
corpus strings freed first, 1.1 GB on disk, `mmap` reopen 0.7 s at 1.55 GB RSS,
28 ms/query.

- [ ] **Step 1: Add the lexical dependencies**

In `pyproject.toml`, add a new extra beside the existing ones:

```toml
retrieval = [
    "bm25s>=0.3",
    "PyStemmer>=2.2",
]
```

Then install it:

```bash
uv pip install -e ".[dev,baselines,retrieval]"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_bm25_index.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.bm25_index import (
    DEFAULT_FIELDS,
    IDS_NAME,
    build_index,
    open_channel,
)
from src.recall import Channel


def _products(n_extra: int = 0) -> pd.DataFrame:
    rows = [
        {
            "product_id": "A1",
            "product_title": "red running shoes for men",
            "description": "lightweight mesh trainers",
            "product_bullet_point": "breathable",
        },
        {
            "product_id": "A2",
            "product_title": "blue ceramic coffee mug",
            "description": "holds twelve ounces",
            "product_bullet_point": "dishwasher safe",
        },
        {
            "product_id": "A3",
            "product_title": "stainless steel water bottle",
            "description": "keeps drinks cold",
            "product_bullet_point": "insulated",
        },
    ]
    for i in range(n_extra):
        rows.append({
            "product_id": f"F{i}",
            "product_title": f"filler product {i}",
            "description": "",
            "product_bullet_point": "",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def index_dir(tmp_path):
    directory = tmp_path / "bm25"
    build_index(_products(), index_dir=directory)
    return directory


# --- the happy path ---------------------------------------------------------

def test_build_reports_how_many_products_it_indexed(tmp_path):
    assert build_index(_products(), index_dir=tmp_path / "bm25") == 3


def test_a_query_finds_the_product_whose_text_matches(index_dir):
    channel = open_channel(index_dir)
    assert channel.search(["running shoes"], k=1)[0] == ["A1"]


def test_search_returns_one_list_per_query(index_dir):
    results = open_channel(index_dir).search(["coffee mug", "water bottle"], k=2)
    assert len(results) == 2
    assert results[0][0] == "A2"
    assert results[1][0] == "A3"


def test_search_respects_k(index_dir):
    assert len(open_channel(index_dir).search(["product"], k=2)[0]) <= 2


def test_the_channel_satisfies_the_protocol(index_dir):
    assert isinstance(open_channel(index_dir), Channel)


def test_n_products_matches_what_was_indexed(index_dir):
    assert open_channel(index_dir).n_products == 3


def test_searching_for_nothing_returns_nothing(index_dir):
    assert open_channel(index_dir).search([], k=10) == []


def test_a_query_matching_no_document_returns_no_spurious_hits(index_dir):
    # bm25s scores a no-match query 0 for every document; returning the whole
    # corpus in arbitrary order would poison RRF with meaningless votes.
    results = open_channel(index_dir).search(["zzzzqqqq nonexistent"], k=3)
    assert results[0] == []


# --- Review Focus 2: row indices mistaken for product ids -------------------

def test_the_id_array_is_written_beside_the_index(index_dir):
    ids = np.load(index_dir / IDS_NAME, allow_pickle=False)
    assert list(ids) == ["A1", "A2", "A3"]


def test_opening_an_index_whose_ids_are_the_wrong_length_raises(index_dir):
    # bm25s.retrieve returns positions into the matrix. A truncated or
    # differently-ordered id array returns real-looking products for every
    # query and nothing raises - recall just comes out low, which reads as a
    # weak retriever rather than a broken one.
    np.save(index_dir / IDS_NAME, np.array(["A1", "A2"]))
    with pytest.raises(ValueError, match="ids"):
        open_channel(index_dir)


def test_results_are_product_ids_not_row_numbers(index_dir):
    hits = open_channel(index_dir).search(["ceramic mug"], k=1)[0]
    assert hits == ["A2"]
    assert not any(hit.isdigit() for hit in hits)


def test_ids_survive_a_corpus_larger_than_the_result_set(tmp_path):
    # With filler rows the matching product is no longer row 0, so an
    # off-by-anything in the id mapping shows up here and not in the 3-row case.
    directory = tmp_path / "bm25"
    build_index(_products(n_extra=50), index_dir=directory)
    assert open_channel(directory).search(["insulated water bottle"], k=1)[0] == ["A3"]


# --- configuration ----------------------------------------------------------

def test_the_default_fields_are_the_text_ones():
    assert DEFAULT_FIELDS == (
        "product_title",
        "description",
        "product_bullet_point",
    )


def test_building_with_a_missing_field_raises(tmp_path):
    with pytest.raises(KeyError, match="no such product field"):
        build_index(
            _products(), fields=("product_title", "nope"), index_dir=tmp_path / "b"
        )


def test_opening_an_index_that_does_not_exist_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="bm25"):
        open_channel(tmp_path / "absent")


# --- the real corpus --------------------------------------------------------

@pytest.mark.data
@pytest.mark.slow
def test_the_real_index_round_trips_and_stays_within_its_memory_budget():
    """The plan gate: mmap under 2 GB RSS, a query under 100 ms.

    Skips rather than builds - the build peaks at 14 GB and takes four
    minutes, so it belongs to `python -m src.bm25_index`, not to a test.
    """
    import resource
    import time

    from src.bm25_index import DEFAULT_INDEX_DIR

    if not (DEFAULT_INDEX_DIR / IDS_NAME).exists():
        pytest.skip(f"no index at {DEFAULT_INDEX_DIR}; run python -m src.bm25_index")

    channel = open_channel(DEFAULT_INDEX_DIR)
    assert channel.n_products == 1_215_854

    started = time.time()
    hits = channel.search(["stainless steel water bottle"], k=100)[0]
    elapsed_ms = 1000 * (time.time() - started)

    assert len(hits) == 100
    assert len(set(hits)) == 100, "the same product was returned twice"
    assert elapsed_ms < 100, f"{elapsed_ms:.0f} ms per query"
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    assert rss_gb < 2.0, f"peak RSS {rss_gb:.2f} GB; the index is not mmapped"
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
python -m pytest tests/test_bm25_index.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.bm25_index'`.

- [ ] **Step 4: Write `src/bm25_index.py`**

```python
"""The lexical retrieval channel: BM25 over the full product corpus.

Built once by the CLI and memory-mapped by every consumer afterwards. That
split is not tidiness - measured on the real 1,215,854-product corpus, the
naive build-and-query-in-one-process path peaks at **18.3 GB** on a 23 GB
machine, while the same index saved and reopened with `mmap=True` costs
**1.55 GB** and answers faster (28 ms/query against 37). Build peak with the
corpus strings freed before indexing is 14.0 GB; the index is 1.1 GB on disk.

`bm25s.retrieve` returns *positions into the indexed matrix*, not product ids,
so the id array is written by the same function from the same frame and its
length is asserted on open. A mismatched id array returns real-looking
products for every query and raises nothing: recall simply comes out low,
which reads as a weak retriever rather than a broken one.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baseline_sbert import product_text

DEFAULT_INDEX_DIR = Path("data/bm25")
DEFAULT_PRODUCTS = Path("data/combined/products.parquet")
IDS_NAME = "product_ids.npy"

# The coalesced `description` rather than either source column: Plan 2 measured
# it at 89.2% coverage against ESCI's own 52.2%, and its label confound largely
# cancels (-0.0065, not significant).
DEFAULT_FIELDS = ("product_title", "description", "product_bullet_point")


def _stemmer() -> Any:
    import Stemmer

    return Stemmer.Stemmer("english")


def tokenize_texts(texts: Sequence[str], stemmer: Any | None = None) -> Any:
    """Tokenise with English stopwords and stemming, as the index expects.

    Queries and documents must go through the same function - a query stemmed
    differently from the corpus matches nothing, and nothing errors.
    """
    import bm25s

    return bm25s.tokenize(
        list(texts),
        stopwords="en",
        stemmer=stemmer if stemmer is not None else _stemmer(),
        show_progress=False,
    )


def build_index(
    products: pd.DataFrame,
    fields: Sequence[str] = DEFAULT_FIELDS,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> int:
    """Build and save the index plus its id array. Returns the row count.

    The id array is written here, from this frame, in this order. Nothing else
    may write it.
    """
    import bm25s

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    product_ids = products["product_id"].to_numpy(dtype=object)
    texts = product_text(products, list(fields)).tolist()

    tokens = tokenize_texts(texts)
    del texts  # ~2 GB of strings; freeing them before indexing is the
    # difference between an 14 GB peak and an 18 GB one.

    index = bm25s.BM25()
    index.index(tokens, show_progress=False)
    index.save(str(index_dir), corpus=None)
    np.save(index_dir / IDS_NAME, product_ids.astype(str), allow_pickle=False)
    return len(product_ids)


@dataclass(frozen=True)
class Bm25Channel:
    """A memory-mapped BM25 index. Open it via open_channel."""

    index: Any
    product_ids: np.ndarray
    stemmer: Any

    @property
    def n_products(self) -> int:
        return len(self.product_ids)

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        """Ranked product_ids per query, best first, at most k."""
        queries = list(queries)
        if not queries:
            return []

        k = min(k, self.n_products)
        tokens = tokenize_texts(queries, stemmer=self.stemmer)
        rows, scores = self.index.retrieve(
            tokens, k=k, show_progress=False, n_threads=8
        )

        results: list[list[str]] = []
        for row, score in zip(rows, scores):
            # A zero score is "this document shares no term with the query".
            # Returning those would hand RRF a full-length ranked list of
            # arbitrary documents to vote for.
            hits = [
                str(self.product_ids[position])
                for position, value in zip(row, score)
                if value > 0
            ]
            results.append(hits)
        return results


def open_channel(index_dir: Path = DEFAULT_INDEX_DIR) -> Bm25Channel:
    """Memory-map a built index, checking it agrees with its id array."""
    import bm25s

    index_dir = Path(index_dir)
    ids_path = index_dir / IDS_NAME
    if not ids_path.exists():
        raise FileNotFoundError(
            f"no bm25 index at {index_dir}; run python -m src.bm25_index"
        )

    index = bm25s.BM25.load(str(index_dir), mmap=True, load_corpus=False)
    product_ids = np.load(ids_path, allow_pickle=False)

    n_indexed = int(index.scores["num_docs"])
    if len(product_ids) != n_indexed:
        raise ValueError(
            f"the index holds {n_indexed:,} documents but its ids file has "
            f"{len(product_ids):,} entries; retrieve() returns row positions, "
            "so a mismatch silently returns the wrong products for every query"
        )
    return Bm25Channel(index=index, product_ids=product_ids, stemmer=_stemmer())


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    args = parser.parse_args()

    fields = tuple(args.fields.split(","))
    frame = pd.read_parquet(
        args.products, columns=["product_id", *fields]
    )
    print(f"indexing {len(frame):,} products on {fields}")
    print("this peaks near 14 GB of RAM and takes about four minutes")

    started = time.time()
    n = build_index(frame, fields=fields, index_dir=args.index_dir)
    print(f"indexed {n:,} products in {time.time() - started:.0f}s -> {args.index_dir}")

    channel = open_channel(args.index_dir)
    hits = channel.search(["stainless steel water bottle"], k=5)[0]
    print(f"smoke test: {hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 5: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_bm25_index.py -v
```

Expected: PASS, 15 tests; the `data`/`slow`-marked real-corpus test is deselected.

- [ ] **Step 6: Build the real index and check the gate**

```bash
python -m src.bm25_index
python -m pytest tests/test_bm25_index.py -m "data and slow" -v
```

Expected: the build reports `indexed 1,215,854 products` in roughly four
minutes and writes ~1.1 GB to `data/bm25/`; the smoke test returns five
plausible water-bottle ASINs; the marked test passes, confirming mmap RSS
under 2 GB and a query under 100 ms.

**If the build is OOM-killed**, the machine has less headroom than the 23 GB
this was measured on. Index `product_title` alone (`--fields product_title`),
which is a third of the text, and record the narrower field set in
`docs/results/recall.json` — a smaller index is a legitimate, reportable
configuration; a machine that swaps for four minutes is not.

**If the smoke test returns nothing**, queries and documents are being
tokenised differently. Both must go through `tokenize_texts`.

- [ ] **Step 7: Record the command in `CLAUDE.md`**

Add to the Commands block, below the image-embeddings lines:

````markdown
# Retrieval channels (build once; every consumer memory-maps the result)
python -m src.bm25_index                      # -> data/bm25/, ~4 min, 14 GB peak
````

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/bm25_index.py tests/test_bm25_index.py CLAUDE.md
git commit -m "Add a memory-mapped BM25 channel over the full product corpus"
```

---

## Phase 1 Gate

Phase 2 does not start until all of these hold:

- [ ] `python -m pytest tests/test_recall.py tests/test_bm25_index.py -v` passes — 17 + 15 tests.
- [ ] `python -m pytest` still passes end to end, with Plans 1–3 untouched.
- [ ] `data/bm25/` exists, holds 1,215,854 documents, and its id array is the same length.
- [ ] `python -m pytest tests/test_bm25_index.py -m "data and slow"` passes: mmap RSS under 2 GB, a query under 100 ms.
- [ ] A BM25 Recall@100 measured on the validation folds lands near the **0.5018** measured on 1,000 test queries. A large gap means the index, not the metric — check the id array first.

Next: [Phase 2 — The Learned Channels](phase-2-the-learned-channels.md).
