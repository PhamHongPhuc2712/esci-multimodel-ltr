# Phase 2 — The Matrix and the Ranker

**Plan 5 of 7 · Phase 2 of 3 · Tasks 3–4.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 1 gate](phase-1-the-features.md#phase-1-gate) passes.

**Delivers:** `src/feature_matrix.py` and `src/ranker.py` — one row per
judgement with 47 named features in eight groups, and a LightGBM ranker that
trains on ESCI's own gain mapping rather than LightGBM's.

**Needs on disk:** `data/features/pair-scores-{train,test}.parquet` from
Phase 1, plus `data/combined/`.

**Owns Review Focus items 1 and 2** (LightGBM's NDCG mistaken for the
project's, the group array disagreeing with the row order).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Every reported NDCG comes from `src.metrics`.** LightGBM's `ndcg@k` differs by up to 6.4 points on the same booster.
- **Features are query×product or product-attribute, never product-alone-from-labels.** The matrix carries labels in its own columns; no *feature* column may be derived from one.
- **Absence is NaN, never zero.** Do not fill, do not impute, do not `nan_to_num`.
- **Never tune on test.** The fold protocol is train 2/3/4, early stop 1, report 0.
- **RAM is not free.** Read Parquet with an explicit `columns=` list.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 3: The feature matrix

**Files:**
- Create: `src/feature_matrix.py`
- Create: `tests/test_feature_matrix.py`

**Interfaces:**
- Consumes: `src.features.*` (Task 1), `src.pair_scores.PAIR_SCORE_COLUMNS` (Task 2), `src.labels.ESCI_GAINS` (Plan 1).
- Produces:
  - `src.feature_matrix.FEATURE_GROUPS: dict[str, tuple[str, ...]]` — the eight named column sets
  - `src.feature_matrix.CATEGORICAL_FEATURES: tuple[str, ...]` = `("s_template", "s_type")`
  - `src.feature_matrix.LABEL_CODES: dict[str, int]` = `{"I": 0, "C": 1, "S": 2, "E": 3}`
  - `src.feature_matrix.ALL_FEATURES: tuple[str, ...]` (47 names)
  - `src.feature_matrix.select_columns(groups: Sequence[str]) -> list[str]`
  - `src.feature_matrix.build_matrix(judgements, products, pair_scores, *, categories) -> pd.DataFrame`
  - `src.feature_matrix.sort_for_ranking(frame) -> pd.DataFrame`
  - `src.feature_matrix.group_sizes(query_ids) -> np.ndarray`
  - `src.feature_matrix.category_vocabularies(products) -> dict[str, list[str]]`
  - CLI: `python -m src.feature_matrix --split {train,test}` writing `data/features/{split}.parquet`

**`FEATURE_GROUPS` is the whole point of this module.** Every ablation arm in
Phase 3 is a list of group names, so the arms are defined once, in one place,
and an arm cannot accidentally include a column its name disclaims. The eight
groups and their 47 columns:

| group | n | columns |
|---|---|---|
| `text` | 13 | `query_chars`, `query_words`, `title_chars`, `title_words`, `esci_desc_chars`, `bullet_chars`, `title_overlap`, `title_coverage`, `bullet_coverage`, `esci_desc_coverage`, `brand_match`, `brand_match_all`, `color_match` |
| `esci_indicators` | 4 | `has_brand`, `has_color`, `has_esci_desc`, `has_bullets` |
| `retrieval` | 2 | `bm25_score`, `dense_sim` |
| `image` | 2 | `clip_image_sim`, `has_image_vector` |
| `behavioural` | 6 | `s_stars`, `s_ratings`, `s_price`, `s_price_multi`, `s_bsr_rank`, `s_n_reviews` |
| `categorical` | 5 | `s_template`, `s_type`, `category_depth`, `category_match_depth`, `category_match_frac` |
| `attrs` | 6 | `n_attrs`, `attr_key_overlap`, `attr_value_overlap`, `n_info`, `info_key_overlap`, `info_value_overlap` |
| `s_indicators` | 9 | `has_enrichment`, `has_stars`, `has_ratings`, `has_price`, `has_bsr`, `has_attrs`, `has_info`, `has_category`, `desc_from_esci_s` |

`has_image_vector` sits in `image`, not in `s_indicators`, because the image
similarity cannot be used without it: a model given `clip_image_sim` and no way
to tell a low cosine from an absent one is being asked to read NaN as a number.
Ablation 3 therefore contrasts "no image signal" against "the image signal and
its indicator", which is the unit a production system would actually ship.

**Review Focus 2 lives in `group_sizes`.** LightGBM's `group` is a list of
*sizes over consecutive rows* — it never sees a query id. Hand it a frame
sorted by anything else and it forms groups that straddle queries, trains on
comparisons between products of different queries, and raises nothing at all.
`group_sizes` therefore counts runs and refuses when the run count does not
equal the distinct-query count.

**Categorical codes are positional, and that is a second silent trap.** A
pandas `category` column built from train has 105 templates; one built from
test may have 103, and the codes then mean different things in the two frames.
`category_vocabularies` reads the vocabulary from the *whole* product table —
one two-column read, the same file for both splits — so the codes are
identical by construction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_feature_matrix.py`:

```python
import math

import numpy as np
import pandas as pd
import pytest

from src.feature_matrix import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_GROUPS,
    LABEL_CODES,
    build_matrix,
    category_vocabularies,
    group_sizes,
    select_columns,
    sort_for_ranking,
)
from src.labels import ESCI_GAINS


# --- the groups -------------------------------------------------------------

def test_the_groups_are_disjoint():
    # An arm is a list of group names. If a column were in two groups, an arm
    # that excludes one of them would still carry the column.
    flat = [c for group in FEATURE_GROUPS.values() for c in group]
    assert len(flat) == len(set(flat))


def test_all_features_is_every_group_flattened():
    flat = [c for group in FEATURE_GROUPS.values() for c in group]
    assert sorted(ALL_FEATURES) == sorted(flat)
    assert len(ALL_FEATURES) == 47


def test_select_columns_returns_the_named_groups_in_order():
    assert select_columns(["retrieval"]) == ["bm25_score", "dense_sim"]
    assert select_columns(["retrieval", "image"]) == [
        "bm25_score", "dense_sim", "clip_image_sim", "has_image_vector",
    ]


def test_select_columns_rejects_an_unknown_group():
    with pytest.raises(KeyError, match="behavioral"):
        select_columns(["behavioral"])  # the American spelling is not a group


def test_the_image_indicator_travels_with_the_image_score():
    # A model given clip_image_sim and no way to tell a low cosine from an
    # absent one is being asked to read NaN as a number.
    assert "has_image_vector" in FEATURE_GROUPS["image"]
    assert "has_image_vector" not in FEATURE_GROUPS["s_indicators"]


def test_the_two_presence_groups_stay_apart():
    # CLAUDE.md: ESCI presence is legitimate, ESCI-S presence is an artefact.
    assert set(FEATURE_GROUPS["esci_indicators"]).isdisjoint(
        FEATURE_GROUPS["s_indicators"]
    )


# --- the label codes --------------------------------------------------------

def test_label_codes_are_ordered_by_gain():
    # LightGBM's label_gain is indexed by the integer label, and it requires a
    # non-decreasing list. Codes out of gain order would silently invert the
    # objective.
    gains = [ESCI_GAINS[label] for label, _ in sorted(LABEL_CODES.items(), key=lambda kv: kv[1])]
    assert gains == sorted(gains)
    assert LABEL_CODES == {"I": 0, "C": 1, "S": 2, "E": 3}


# --- group_sizes: Review Focus 2 -------------------------------------------

def test_group_sizes_counts_consecutive_runs():
    assert group_sizes([1, 1, 1, 2, 2]).tolist() == [3, 2]


def test_group_sizes_rejects_a_query_split_across_the_frame():
    # LightGBM's `group` is sizes over consecutive rows - it never sees a query
    # id. A frame sorted any other way trains on comparisons that straddle
    # queries, and nothing raises.
    with pytest.raises(ValueError, match="not contiguous"):
        group_sizes([1, 2, 1])


def test_group_sizes_of_nothing_is_empty():
    assert group_sizes([]).tolist() == []


def test_group_sizes_sum_to_the_row_count():
    ids = [1, 1, 2, 3, 3, 3]
    assert int(group_sizes(ids).sum()) == len(ids)


def test_sort_for_ranking_makes_every_query_contiguous():
    frame = pd.DataFrame(
        {"query_id": [2, 1, 2, 1], "product_id": ["b", "a", "a", "b"]}
    )
    ordered = sort_for_ranking(frame)
    assert ordered["query_id"].tolist() == [1, 1, 2, 2]
    # Deterministic within a query too, so two runs of the same code produce
    # byte-identical matrices.
    assert ordered["product_id"].tolist() == ["a", "b", "a", "b"]
    group_sizes(ordered["query_id"])  # must not raise


# --- categorical vocabularies ----------------------------------------------

def test_category_vocabularies_are_sorted_and_shared():
    products = pd.DataFrame(
        {"s_template": ["kitchen", "apparel", None, "kitchen"], "s_type": ["product", "book", None, "product"]}
    )
    vocab = category_vocabularies(products)
    assert vocab["s_template"] == ["apparel", "kitchen"]
    assert vocab["s_type"] == ["book", "product"]


def test_the_same_vocabulary_gives_the_same_codes_in_two_frames():
    # pandas category codes are positional. A dtype built per split would give
    # "kitchen" code 1 in one frame and code 0 in another, and LightGBM would
    # be told they are the same feature.
    vocab = {"s_template": ["apparel", "book", "kitchen"], "s_type": ["book", "product"]}
    a = build_matrix(*_tiny(template="kitchen"), categories=vocab)
    b = build_matrix(*_tiny(template="kitchen", query_id=9), categories=vocab)
    assert a["s_template"].cat.categories.tolist() == b["s_template"].cat.categories.tolist()
    assert a["s_template"].cat.codes.tolist() == b["s_template"].cat.codes.tolist()


# --- build_matrix -----------------------------------------------------------

def _tiny(template="kitchen", query_id=1):
    judgements = pd.DataFrame(
        {
            "query_id": [query_id, query_id],
            "query": ["steel water bottle", "steel water bottle"],
            "product_id": ["p1", "p2"],
            "esci_label": ["E", "I"],
            "gain": [1.0, 0.0],
            "qrel": [100, 0],
            "split": ["train", "train"],
            "fold": [0, 0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_title": ["Steel Water Bottle", "Plastic Cup"],
            "product_description": ["Holds water.", None],
            "product_bullet_point": ["Insulated", None],
            "product_brand": ["Hydro", None],
            "product_color": ["Silver", None],
            "s_template": [template, None],
            "s_type": ["product", None],
            "s_category": [["Kitchen", "Bottles"], None],
            "s_attrs_json": ['{"Material": "Steel"}', None],
            "s_info_json": ['{"Item Weight": "1 lb"}', None],
            "s_stars": [4.5, None],
            "s_ratings": [10.0, None],
            "s_price": [19.99, None],
            "s_price_multi": [False, None],
            "s_bsr_rank": [1234.0, None],
            "s_n_reviews": [3.0, None],
            "has_enrichment": [True, False],
            "description_source": ["esci", "none"],
        }
    )
    pair_scores = pd.DataFrame(
        {
            "query_id": [query_id, query_id],
            "product_id": ["p1", "p2"],
            "bm25_score": [12.5, 3.0],
            "dense_sim": [0.8, 0.2],
            "clip_image_sim": [0.6, np.nan],
        }
    )
    return judgements, products, pair_scores


def test_build_matrix_emits_every_feature_once():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    for name in ALL_FEATURES:
        assert name in frame.columns, name
    assert len(frame) == 2


def test_build_matrix_carries_the_label_columns():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    assert frame.loc[frame["product_id"] == "p1", "label_code"].iloc[0] == LABEL_CODES["E"]
    assert frame.loc[frame["product_id"] == "p2", "label_code"].iloc[0] == LABEL_CODES["I"]
    assert set(["query_id", "product_id", "esci_label", "gain", "qrel", "split", "fold"]) <= set(frame.columns)


def test_no_feature_column_is_derived_from_a_label():
    # 34,756 products appear in both train and test. A feature computed from
    # labels per product leaks. Flipping every label must not move a feature.
    judgements, products, scores = _tiny()
    flipped = judgements.copy()
    flipped["esci_label"] = ["I", "E"]
    flipped["gain"] = [0.0, 1.0]
    flipped["qrel"] = [0, 100]
    vocab = category_vocabularies(products)

    before = build_matrix(judgements, products, scores, categories=vocab)
    after = build_matrix(flipped, products, scores, categories=vocab)
    pd.testing.assert_frame_equal(
        before[list(ALL_FEATURES)], after[list(ALL_FEATURES)], check_dtype=True
    )


def test_the_image_indicator_follows_the_image_score():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    p1 = frame.loc[frame["product_id"] == "p1"].iloc[0]
    p2 = frame.loc[frame["product_id"] == "p2"].iloc[0]
    assert p1["has_image_vector"] == 1.0
    assert p2["has_image_vector"] == 0.0
    assert math.isnan(p2["clip_image_sim"])


def test_build_matrix_never_fills_a_nan_similarity():
    # Review Focus 5 again, one layer up: the assembly must not impute.
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    assert frame["clip_image_sim"].isna().sum() == 1


def test_build_matrix_raises_when_a_pair_has_no_scores():
    judgements, products, scores = _tiny()
    with pytest.raises(ValueError, match="pair scores"):
        build_matrix(judgements, products, scores.iloc[:1], categories=category_vocabularies(products))


def test_build_matrix_raises_when_a_judged_product_is_missing():
    judgements, products, scores = _tiny()
    with pytest.raises(ValueError, match="no product row"):
        build_matrix(judgements, products.iloc[:1], scores, categories=category_vocabularies(products))


def test_the_matrix_comes_back_sorted_for_ranking():
    judgements, products, scores = _tiny()
    judgements = judgements.iloc[::-1].reset_index(drop=True)
    frame = build_matrix(judgements, products, scores, categories=category_vocabularies(products))
    group_sizes(frame["query_id"])  # must not raise


def test_categorical_columns_have_the_category_dtype():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    for name in CATEGORICAL_FEATURES:
        assert isinstance(frame[name].dtype, pd.CategoricalDtype), name
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_feature_matrix.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.feature_matrix'`.

- [ ] **Step 3: Write the implementation**

Create `src/feature_matrix.py`:

```python
"""One row per judgement, 47 named features, eight named groups.

FEATURE_GROUPS is the point of this module. Every ablation arm in Plan 5 is a
list of group names, so the arms are defined once and an arm cannot quietly
include a column its own name disclaims - which is how "text only" ends up
carrying the enrichment.

Two silent traps are closed here rather than downstream:

  * **LightGBM's `group` is sizes over consecutive rows.** It never sees a
    query id. A frame sorted by anything but the query forms groups that
    straddle queries and trains on comparisons between products of different
    queries, with no exception and no warning. `group_sizes` counts runs and
    refuses when the run count is not the distinct-query count.
  * **pandas category codes are positional.** `s_template` has 105 values over
    the whole corpus but fewer within one split, so a dtype built per split
    gives the same string different codes in train and test.
    `category_vocabularies` reads the vocabulary from the whole product table,
    the same file for both splits, so the codes agree by construction.

The matrix carries `gain`, `qrel`, `esci_label` and `label_code` as *label*
columns. No column in ALL_FEATURES may be derived from any of them: 34,756
products appear in both train and test, so a product-level statistic computed
from labels leaks. tests/test_feature_matrix.py pins this by flipping every
label and asserting the feature block is unchanged.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    ESCI_INDICATORS,
    S_INDICATORS,
    indicator_features,
    pair_features,
    product_features,
    product_terms,
)
from src.pair_scores import DEFAULT_OUT

DEFAULT_PRODUCTS = Path("data/combined/products.parquet")
DEFAULT_JUDGEMENTS = Path("data/combined/judgements.parquet")

# Ordered by gain, because LightGBM's label_gain is indexed by the integer
# label and must be non-decreasing.
LABEL_CODES: dict[str, int] = {"I": 0, "C": 1, "S": 2, "E": 3}

CATEGORICAL_FEATURES: tuple[str, ...] = ("s_template", "s_type")

# The columns build_matrix needs from the product table. Named explicitly
# because the combined table is 27 columns and 1.84 GB.
PRODUCT_COLUMNS: tuple[str, ...] = (
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "s_template",
    "s_type",
    "s_category",
    "s_attrs_json",
    "s_info_json",
    "s_stars",
    "s_ratings",
    "s_price",
    "s_price_multi",
    "s_bsr_rank",
    "s_n_reviews",
    "has_enrichment",
    "description_source",
)

LABEL_COLUMNS: tuple[str, ...] = (
    "query_id",
    "product_id",
    "esci_label",
    "label_code",
    "gain",
    "qrel",
    "split",
    "fold",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # ESCI's own text, and nothing from the scrape. `esci_desc_*` reads
    # product_description rather than the coalesced `description`, which is
    # 37.0% ESCI-S - a "text only" arm built on the coalesce would already
    # carry the enrichment.
    "text": (
        "query_chars",
        "query_words",
        "title_chars",
        "title_words",
        "esci_desc_chars",
        "bullet_chars",
        "title_overlap",
        "title_coverage",
        "bullet_coverage",
        "esci_desc_coverage",
        "brand_match",
        "brand_match_all",
        "color_match",
    ),
    # Legitimate: a production ranker also knows whether a product has a brand.
    "esci_indicators": ESCI_INDICATORS,
    "retrieval": ("bm25_score", "dense_sim"),
    # The indicator travels with the score: a model handed clip_image_sim with
    # no way to tell a low cosine from an absent one is being asked to read
    # NaN as a number.
    "image": ("clip_image_sim", "has_image_vector"),
    "behavioural": (
        "s_stars",
        "s_ratings",
        "s_price",
        "s_price_multi",
        "s_bsr_rank",
        "s_n_reviews",
    ),
    "categorical": (
        "s_template",
        "s_type",
        "category_depth",
        "category_match_depth",
        "category_match_frac",
    ),
    "attrs": (
        "n_attrs",
        "attr_key_overlap",
        "attr_value_overlap",
        "n_info",
        "info_key_overlap",
        "info_value_overlap",
    ),
    # An artefact: "our 2022 scrape failed on this page" is not available at
    # serving time. Ablation 4 subtracts an arm built from these alone.
    "s_indicators": S_INDICATORS,
}

ALL_FEATURES: tuple[str, ...] = tuple(
    column for group in FEATURE_GROUPS.values() for column in group
)


def select_columns(groups: Sequence[str]) -> list[str]:
    """The columns of the named feature groups, in group order."""
    columns: list[str] = []
    for name in groups:
        if name not in FEATURE_GROUPS:
            raise KeyError(
                f"unknown feature group {name!r}; expected one of "
                f"{sorted(FEATURE_GROUPS)}"
            )
        columns.extend(FEATURE_GROUPS[name])
    return columns


def group_sizes(query_ids: Iterable) -> np.ndarray:
    """Sizes of consecutive runs of the same query id, for LightGBM's `group`.

    Raises when a query id appears in more than one run. LightGBM takes sizes
    over consecutive rows and never sees the ids, so a non-contiguous frame
    produces groups that straddle queries - a model trained to rank one
    query's product against another's, with nothing raised.
    """
    ids = np.asarray(list(query_ids))
    if len(ids) == 0:
        return np.zeros(0, dtype=np.int64)

    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    runs = np.diff(np.concatenate([[0], boundaries, [len(ids)]]))
    n_distinct = len(pd.unique(ids))
    if len(runs) != n_distinct:
        raise ValueError(
            f"query ids are not contiguous: {len(runs)} runs over "
            f"{n_distinct} distinct queries. LightGBM's `group` is a list of "
            "sizes over consecutive rows, so this would train on comparisons "
            "that straddle queries. Call sort_for_ranking first."
        )
    return runs.astype(np.int64)


def sort_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort so every query's rows are contiguous, and deterministically so."""
    return frame.sort_values(["query_id", "product_id"], kind="stable").reset_index(
        drop=True
    )


def category_vocabularies(products: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted category values for the categorical columns.

    Built from the whole product table rather than per split, because pandas
    category codes are positional: a dtype fitted on train would give
    "kitchen" a different code in test, and LightGBM would be told the two are
    the same feature.
    """
    return {
        name: sorted({v for v in products[name] if isinstance(v, str) and v})
        for name in CATEGORICAL_FEATURES
    }


def build_matrix(
    judgements: pd.DataFrame,
    products: pd.DataFrame,
    pair_scores: pd.DataFrame,
    *,
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """One row per judgement: labels, then the 47 features, sorted for ranking.

    Product-derived features are computed once per distinct product and
    joined; only the query x product half runs per judgement. There are
    482,105 distinct products against 601,354 judgements, and product_terms is
    the part that parses JSON.
    """
    missing = set(judgements["product_id"]) - set(products["product_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} judged products have no product row "
            f"(for example {sorted(missing)[:3]}); scoring them from defaults "
            "would put a fabricated feature vector in the training data"
        )

    catalogue = products.drop_duplicates("product_id").reset_index(drop=True)
    # One pass over the records: to_dict is not cheap at 482,105 rows.
    records = catalogue.to_dict("records")
    product_frame = pd.DataFrame(
        [product_features(row) | indicator_features(row) for row in records]
    )
    product_frame.insert(0, "product_id", catalogue["product_id"].to_numpy())
    for name in CATEGORICAL_FEATURES:
        product_frame[name] = pd.Categorical(
            catalogue[name], categories=list(categories[name])
        )

    terms = {
        row["product_id"]: product_terms(row) for row in records
    }

    frame = judgements.copy()
    frame["label_code"] = frame["esci_label"].map(LABEL_CODES)
    pair_rows = [
        pair_features(str(query), terms[product_id])
        for query, product_id in zip(frame["query"], frame["product_id"])
    ]
    frame = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(pair_rows)], axis=1
    )

    frame = frame.merge(product_frame, on="product_id", how="left")
    # A merge can widen a category dtype back to object. Re-cast against the
    # same vocabulary so the codes stay comparable across splits.
    for name in CATEGORICAL_FEATURES:
        frame[name] = pd.Categorical(frame[name], categories=list(categories[name]))

    before = len(frame)
    frame = frame.merge(pair_scores, on=["query_id", "product_id"], how="left")
    if len(frame) != before:
        raise ValueError(
            f"joining pair scores changed the row count from {before:,} to "
            f"{len(frame):,}; the pair-score table has duplicate "
            "(query_id, product_id) rows"
        )
    unscored = frame["bm25_score"].isna() & frame["dense_sim"].isna()
    if unscored.any():
        raise ValueError(
            f"{int(unscored.sum()):,} pairs have no pair scores at all; run "
            "python -m src.pair_scores for this split first"
        )

    # Absence, not a score. 20.6% of judged pairs have no image vector.
    frame["has_image_vector"] = frame["clip_image_sim"].notna().astype(float)

    columns = [c for c in LABEL_COLUMNS if c in frame.columns] + list(ALL_FEATURES)
    return sort_for_ranking(frame[columns])


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--judgements", type=Path, default=DEFAULT_JUDGEMENTS)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    judgements = pd.read_parquet(args.judgements)
    judgements = judgements.loc[judgements["split"] == args.split].reset_index(drop=True)

    scores_path = args.features_dir / f"pair-scores-{args.split}.parquet"
    if not scores_path.exists():
        raise SystemExit(
            f"no pair scores at {scores_path}; run "
            f"python -m src.pair_scores --split {args.split}"
        )
    pair_scores = pd.read_parquet(scores_path)

    # The vocabulary comes from the whole table, both splits, so the codes
    # match. Two columns of 1,215,854 rows is a cheap read.
    vocabulary = category_vocabularies(
        pd.read_parquet(args.products, columns=list(CATEGORICAL_FEATURES))
    )
    products = pd.read_parquet(args.products, columns=list(PRODUCT_COLUMNS))
    products = products.loc[
        products["product_id"].isin(set(judgements["product_id"]))
    ].reset_index(drop=True)

    print(f"{len(judgements):,} judgements, {len(products):,} products, "
          f"{len(ALL_FEATURES)} features")
    frame = build_matrix(judgements, products, pair_scores, categories=vocabulary)

    sizes = group_sizes(frame["query_id"])
    print(f"{len(sizes):,} query groups, sizes {sizes.min()}-{sizes.max()}, "
          f"mean {sizes.mean():.2f}")
    coverage = frame[list(ALL_FEATURES)].notna().mean()
    print("least-covered features:")
    for name, share in coverage.sort_values().head(5).items():
        print(f"  {name:22s} {share:.4f}")

    path = args.features_dir / f"{args.split}.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {len(frame):,} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_feature_matrix.py -q`
Expected: PASS, 22 tests.

- [ ] **Step 5: Build both real matrices**

Run:

```bash
python -m src.feature_matrix --split train
python -m src.feature_matrix --split test
```

Expected: 419,653 and 181,701 rows; 20,888 and 8,956 query groups; group sizes
8–188 (train) and 8–95 (test); the least-covered features being `s_price`
(≈0.27), `s_bsr_rank` (≈0.43), `n_info` (≈0.48), `info_*_overlap` (≈0.48).
Each build takes a couple of minutes, almost all of it in `product_terms`.

- [ ] **Step 6: Write the data-marked test**

Append to `tests/test_feature_matrix.py`:

```python
@pytest.mark.data
def test_the_real_matrices_match_the_asserted_invariants():
    train = pd.read_parquet("data/features/train.parquet")
    test = pd.read_parquet("data/features/test.parquet")

    assert len(train) == 419_653
    assert len(test) == 181_701
    assert train["query_id"].nunique() == 20_888
    assert test["query_id"].nunique() == 8_956

    for frame in (train, test):
        sizes = group_sizes(frame["query_id"])   # contiguity, Review Focus 2
        assert int(sizes.sum()) == len(frame)
        assert set(ALL_FEATURES) <= set(frame.columns)

    # Measured 2026-09-21 over the 482,105 judged products; the share over
    # judgements is close but not identical, so the bands are wide.
    coverage = train[list(ALL_FEATURES)].notna().mean()
    assert 0.20 < coverage["s_price"] < 0.35
    assert 0.80 < coverage["s_stars"] < 0.95
    assert 0.70 < coverage["clip_image_sim"] < 0.85
    assert coverage["dense_sim"] == pytest.approx(1.0)
    # Indicators answer "was it there", so they are never themselves missing.
    for name in FEATURE_GROUPS["esci_indicators"] + FEATURE_GROUPS["s_indicators"]:
        assert coverage[name] == pytest.approx(1.0), name


@pytest.mark.data
def test_the_two_splits_agree_on_categorical_codes():
    train = pd.read_parquet("data/features/train.parquet")
    test = pd.read_parquet("data/features/test.parquet")
    for name in CATEGORICAL_FEATURES:
        assert (
            train[name].cat.categories.tolist() == test[name].cat.categories.tolist()
        ), name
```

Run: `python -m pytest tests/test_feature_matrix.py -m data -q`
Expected: PASS.

- [ ] **Step 7: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 8: Commit**

```bash
git add src/feature_matrix.py tests/test_feature_matrix.py
git commit -m "Assemble the judged-pair feature matrix with named feature groups"
```

---

## Task 4: The ranker

**Files:**
- Create: `src/ranker.py`
- Create: `tests/test_ranker.py`

**Interfaces:**
- Consumes: `src.feature_matrix.{group_sizes, sort_for_ranking, LABEL_CODES, CATEGORICAL_FEATURES}` (Task 3), `src.labels.ESCI_GAINS` (Plan 1).
- Produces:
  - `src.ranker.LABEL_ORDER: tuple[str, ...]` = `("I", "C", "S", "E")`
  - `src.ranker.LABEL_GAIN: list[float]` = `[0.0, 0.01, 0.1, 1.0]`
  - `src.ranker.OBJECTIVES: tuple[str, ...]` = `("lambdarank", "pointwise_regression", "pointwise_class")`
  - `src.ranker.TRAIN_FOLDS`, `EARLY_STOP_FOLD`, `REPORT_FOLD`
  - `src.ranker.TrainedRanker` (frozen dataclass: `booster`, `objective`, `features`, `best_iteration`)
  - `src.ranker.train_ranker(train, valid, features, *, objective="lambdarank", params=None, num_boost_round=500, early_stopping_rounds=50, seed=0) -> TrainedRanker`
  - `src.ranker.predict(ranker, frame) -> np.ndarray`
  - `src.ranker.predict_run(ranker, frame) -> dict[str, dict[str, float]]`
  - `src.ranker.folds(frame, which) -> pd.DataFrame`

**Review Focus 1 lives here.** LightGBM will happily report an NDCG, and it
will not be this project's NDCG, for two independent reasons that compound:

1. `lambdarank` defaults to a `2**rel - 1` gain. ESCI's mapping is
   1.0/0.1/0.01/0.0 — not exponential, and not even proportional to one.
2. `eval_at` is a cutoff. This project reports **full-list** NDCG, which is why
   its random floor sits near 0.75 rather than near 0.

Measured on a matrix of the real shape, the **same booster** reports `ndcg@10`
**0.7931** under `label_gain=[0.0, 0.01, 0.1, 1.0]` and **0.8575** under
LightGBM's default. A number lifted from a training log would inflate the whole
ablation table by more than every effect in it. So:

- `LABEL_GAIN` is *derived* from `src.labels.ESCI_GAINS`, never written out, so
  the mapping has one definition in the project.
- `train_ranker` raises if a caller overrides `label_gain`.
- `TrainedRanker` carries no NDCG of any kind. The only way to get one is
  `predict_run` into `src.metrics.ndcg_per_query`.

LightGBM's internal metric still earns its keep: it is a monotone-enough proxy
to early-stop on, and early stopping happens on fold 1, which no reported
number comes from.

**`lambdarank_truncation_level` is set to 200** because the largest train query
has 188 judged candidates and the reported metric has no cutoff. LightGBM's
default of 30 would ignore two thirds of the longest lists.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ranker.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.feature_matrix import LABEL_CODES
from src.labels import ESCI_GAINS
from src.metrics import ndcg_per_query
from src.ranker import (
    EARLY_STOP_FOLD,
    LABEL_GAIN,
    LABEL_ORDER,
    OBJECTIVES,
    REPORT_FOLD,
    TRAIN_FOLDS,
    TrainedRanker,
    folds,
    predict,
    predict_run,
    train_ranker,
)


def _frame(n_queries=40, seed=0, fold=0):
    """A small ranking frame with one informative feature."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        for p in range(8):
            code = int(rng.choice([0, 1, 2, 3], p=[0.17, 0.05, 0.35, 0.43]))
            rows.append(
                {
                    "query_id": q,
                    "product_id": f"p{q}_{p}",
                    "label_code": code,
                    "gain": ESCI_GAINS[LABEL_ORDER[code]],
                    "qrel": round(ESCI_GAINS[LABEL_ORDER[code]] * 100),
                    "fold": fold,
                    "signal": code + rng.normal(scale=0.3),
                    "noise": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


FEATURES = ["signal", "noise"]


# --- Review Focus 1: the gain mapping --------------------------------------

def test_label_gain_is_derived_from_the_projects_own_mapping():
    assert LABEL_ORDER == ("I", "C", "S", "E")
    assert LABEL_GAIN == [ESCI_GAINS[label] for label in LABEL_ORDER]
    assert LABEL_GAIN == [0.0, 0.01, 0.1, 1.0]


def test_the_gain_mapping_is_not_lightgbms_default():
    # LightGBM defaults to 2**rel - 1 = [0, 1, 3, 7]. On a matrix of the real
    # shape the same booster reports ndcg@10 0.7931 under ESCI's gains and
    # 0.8575 under that default - a 6.4-point gap, larger than any effect in
    # the ablation table.
    assert LABEL_GAIN != [0, 1, 3, 7]


def test_label_gain_agrees_with_the_matrix_label_codes():
    for label, code in LABEL_CODES.items():
        assert LABEL_GAIN[code] == ESCI_GAINS[label]


def test_overriding_the_gain_mapping_raises():
    train, valid = _frame(), _frame(seed=1)
    with pytest.raises(ValueError, match="label_gain"):
        train_ranker(train, valid, FEATURES, params={"label_gain": [0, 1, 3, 7]})


def test_the_trained_ranker_reports_no_ndcg_of_its_own():
    # The only NDCG in this project comes from src.metrics. A number lifted
    # from a training log is a different metric wearing the same name.
    ranker = train_ranker(_frame(), _frame(seed=1), FEATURES, num_boost_round=5)
    assert not hasattr(ranker, "ndcg")
    assert not hasattr(ranker, "score")


def test_truncation_covers_the_longest_real_query():
    # The largest train query has 188 judged candidates; LightGBM's default
    # truncation of 30 would ignore two thirds of it.
    from src.ranker import LAMBDARANK_PARAMS

    assert LAMBDARANK_PARAMS["lambdarank_truncation_level"] >= 188


# --- training ---------------------------------------------------------------

def test_lambdarank_learns_the_informative_feature():
    train, valid = _frame(n_queries=120), _frame(n_queries=60, seed=1)
    ranker = train_ranker(train, valid, FEATURES, num_boost_round=60)
    importance = dict(
        zip(ranker.booster.feature_name(), ranker.booster.feature_importance("gain"))
    )
    assert importance["signal"] > importance["noise"]


def test_every_objective_trains_and_predicts_one_score_per_row():
    train, valid = _frame(n_queries=60), _frame(n_queries=30, seed=1)
    for objective in OBJECTIVES:
        ranker = train_ranker(
            train, valid, FEATURES, objective=objective, num_boost_round=10
        )
        scores = predict(ranker, valid)
        assert scores.shape == (len(valid),), objective
        assert np.isfinite(scores).all(), objective


def test_the_multiclass_arm_is_scored_by_expected_gain():
    # A pointwise classifier emits four probabilities. Collapsing them with
    # argmax would throw away the graded structure the whole metric is about;
    # the expected gain keeps it.
    train, valid = _frame(n_queries=60), _frame(n_queries=30, seed=1)
    ranker = train_ranker(
        train, valid, FEATURES, objective="pointwise_class", num_boost_round=10
    )
    scores = predict(ranker, valid)
    assert scores.min() >= -1e-9
    assert scores.max() <= 1.0 + 1e-9


def test_an_unknown_objective_raises():
    with pytest.raises(ValueError, match="objective"):
        train_ranker(_frame(), _frame(seed=1), FEATURES, objective="pairwise")


def test_training_uses_only_the_named_features():
    train, valid = _frame(), _frame(seed=1)
    ranker = train_ranker(train, valid, ["signal"], num_boost_round=5)
    assert ranker.features == ("signal",)
    assert ranker.booster.feature_name() == ["signal"]


def test_training_is_deterministic_for_a_seed():
    train, valid = _frame(n_queries=60), _frame(n_queries=30, seed=1)
    a = predict(train_ranker(train, valid, FEATURES, num_boost_round=20, seed=7), valid)
    b = predict(train_ranker(train, valid, FEATURES, num_boost_round=20, seed=7), valid)
    np.testing.assert_allclose(a, b)


def test_training_sorts_the_frame_before_grouping():
    # Review Focus 2, one layer up: a caller handing over an unsorted frame
    # must get a correct model, not a silently mis-grouped one.
    train = _frame(n_queries=60).sample(frac=1.0, random_state=0)
    valid = _frame(n_queries=30, seed=1).sample(frac=1.0, random_state=0)
    ranker = train_ranker(train, valid, FEATURES, num_boost_round=10)
    assert predict(ranker, valid).shape == (len(valid),)


# --- predict_run ------------------------------------------------------------

def test_predict_run_is_scoreable_by_src_metrics():
    train, valid = _frame(n_queries=120), _frame(n_queries=60, seed=1)
    ranker = train_ranker(train, valid, FEATURES, num_boost_round=60)
    run = predict_run(ranker, valid)

    qrels: dict[str, dict[str, int]] = {}
    for q, p, r in zip(valid["query_id"], valid["product_id"], valid["qrel"]):
        qrels.setdefault(str(q), {})[str(p)] = int(r)

    per_query = ndcg_per_query(run, qrels)
    assert len(per_query) == valid["query_id"].nunique()
    # A model that learned the informative feature must beat a random ordering,
    # whose full-list NDCG on this label mix sits near 0.75.
    assert sum(per_query.values()) / len(per_query) > 0.80


def test_predict_run_keys_are_strings():
    # pytrec_eval and src.metrics both key on strings; query_id is int64.
    train, valid = _frame(), _frame(seed=1)
    run = predict_run(train_ranker(train, valid, FEATURES, num_boost_round=5), valid)
    assert all(isinstance(q, str) for q in run)
    assert all(isinstance(p, str) for docs in run.values() for p in docs)


def test_predict_run_scores_every_judged_pair():
    train, valid = _frame(), _frame(seed=1)
    run = predict_run(train_ranker(train, valid, FEATURES, num_boost_round=5), valid)
    assert sum(len(docs) for docs in run.values()) == len(valid)


# --- the fold protocol ------------------------------------------------------

def test_the_three_fold_roles_are_disjoint():
    # Selecting, early-stopping and reporting on one fold is how a plan ends
    # up reporting the number it optimised.
    assert REPORT_FOLD not in TRAIN_FOLDS
    assert EARLY_STOP_FOLD not in TRAIN_FOLDS
    assert REPORT_FOLD != EARLY_STOP_FOLD


def test_folds_selects_the_requested_folds():
    frame = pd.concat([_frame(fold=f, seed=f) for f in range(5)], ignore_index=True)
    assert set(folds(frame, TRAIN_FOLDS)["fold"]) == set(TRAIN_FOLDS)
    assert set(folds(frame, [REPORT_FOLD])["fold"]) == {REPORT_FOLD}


def test_folds_raises_when_a_requested_fold_is_absent():
    with pytest.raises(ValueError, match="fold"):
        folds(_frame(fold=0), [3])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ranker.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.ranker'`.

- [ ] **Step 3: Write the implementation**

Create `src/ranker.py`:

```python
"""LightGBM for the coarse rank stage, on ESCI's gain mapping rather than its own.

PROJECT_SPEC.md §4.2 asks for `objective: lambdarank` - a real LTR loss, not a
binary classification - and §6's Ablation 5 asks for the pointwise comparison
that shows whether the listwise loss earned its keep.

**LightGBM will report an NDCG, and it is not this project's NDCG.** Two
independent reasons compound:

  1. `lambdarank` defaults to a 2**rel - 1 gain. ESCI's mapping is
     1.0/0.1/0.01/0.0 - not exponential and not proportional to one.
  2. `eval_at` is a cutoff; this project reports full-list NDCG, which is why
     its random floor sits near 0.75 rather than near 0.

Measured on a matrix of the real shape (385,829 rows, 20,888 groups), the same
booster reports ndcg@10 0.7931 under ESCI's gains and 0.8575 under LightGBM's
default - a 6.4-point gap, larger than every effect in the ablation table. So
LABEL_GAIN is derived from src.labels rather than written out, `train_ranker`
refuses to let a caller override it, and TrainedRanker carries no score of any
kind. The only NDCG in this project comes from src.metrics.

LightGBM's internal metric still earns its keep as an early-stopping proxy,
and early stopping happens on fold 1, which no reported number comes from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.feature_matrix import (
    CATEGORICAL_FEATURES,
    group_sizes,
    sort_for_ranking,
)
from src.labels import ESCI_GAINS

# Integer label codes, ordered by gain. LightGBM indexes label_gain by the
# label and requires the list to be non-decreasing.
LABEL_ORDER: tuple[str, ...] = ("I", "C", "S", "E")
LABEL_GAIN: list[float] = [ESCI_GAINS[label] for label in LABEL_ORDER]

OBJECTIVES: tuple[str, ...] = (
    "lambdarank",
    "pointwise_regression",
    "pointwise_class",
)

# Three roles, three folds. Selecting, early-stopping and reporting on one
# fold is how a plan ends up reporting the number it optimised.
TRAIN_FOLDS: tuple[int, ...] = (2, 3, 4)
EARLY_STOP_FOLD: int = 1
REPORT_FOLD: int = 0

_SHARED = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "num_threads": 8,
    "deterministic": True,
}

LAMBDARANK_PARAMS: dict[str, Any] = _SHARED | {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "label_gain": LABEL_GAIN,
    # The largest train query has 188 judged candidates and the reported
    # metric has no cutoff; LightGBM's default of 30 would ignore two thirds
    # of the longest lists.
    "lambdarank_truncation_level": 200,
}

POINTWISE_REGRESSION_PARAMS: dict[str, Any] = _SHARED | {
    "objective": "regression",
    "metric": "l2",
}

POINTWISE_CLASS_PARAMS: dict[str, Any] = _SHARED | {
    "objective": "multiclass",
    "num_class": len(LABEL_ORDER),
    "metric": "multi_logloss",
}

_PARAMS: dict[str, dict[str, Any]] = {
    "lambdarank": LAMBDARANK_PARAMS,
    "pointwise_regression": POINTWISE_REGRESSION_PARAMS,
    "pointwise_class": POINTWISE_CLASS_PARAMS,
}


@dataclass(frozen=True)
class TrainedRanker:
    """A trained booster and the columns it was trained on. Carries no score."""

    booster: Any
    objective: str
    features: tuple[str, ...]
    best_iteration: int


def folds(frame: pd.DataFrame, which: Sequence[int]) -> pd.DataFrame:
    """The rows of `frame` in the requested folds."""
    wanted = list(which)
    available = set(frame["fold"])
    missing = set(wanted) - available
    if missing:
        raise ValueError(
            f"fold(s) {sorted(missing)} are not in this frame, which has "
            f"{sorted(available)}; test rows carry fold = -1 and have none"
        )
    return frame.loc[frame["fold"].isin(wanted)].reset_index(drop=True)


def _dataset(
    frame: pd.DataFrame, features: Sequence[str], objective: str, reference=None
):
    import lightgbm as lgb

    ordered = sort_for_ranking(frame)
    matrix = ordered[list(features)]
    label = (
        ordered["label_code"].to_numpy()
        if objective != "pointwise_regression"
        else ordered["gain"].to_numpy()
    )
    categorical = [c for c in CATEGORICAL_FEATURES if c in features]
    dataset = lgb.Dataset(
        matrix,
        label=label,
        categorical_feature=categorical or "auto",
        free_raw_data=False,
        reference=reference,
    )
    if objective == "lambdarank":
        dataset.set_group(group_sizes(ordered["query_id"]))
    return dataset


def train_ranker(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: Sequence[str],
    *,
    objective: str = "lambdarank",
    params: Mapping[str, Any] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    seed: int = 0,
) -> TrainedRanker:
    """Train one arm. `valid` is for early stopping and nothing else."""
    import lightgbm as lgb

    if objective not in _PARAMS:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of {OBJECTIVES}"
        )

    overrides = dict(params or {})
    if "label_gain" in overrides:
        raise ValueError(
            "label_gain is derived from src.labels.ESCI_GAINS and must not be "
            "overridden. LightGBM's default 2**rel - 1 reports an NDCG 6.4 "
            "points away from this project's on the same booster."
        )

    settings = _PARAMS[objective] | overrides | {"seed": seed, "data_random_seed": seed}
    train_set = _dataset(train, features, objective)
    valid_set = _dataset(valid, features, objective, reference=train_set)

    booster = lgb.train(
        settings,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["early_stop"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return TrainedRanker(
        booster=booster,
        objective=objective,
        features=tuple(features),
        best_iteration=int(booster.best_iteration or num_boost_round),
    )


def predict(ranker: TrainedRanker, frame: pd.DataFrame) -> np.ndarray:
    """One score per row of `frame`, higher is better.

    The multiclass arm emits four probabilities; they are collapsed by
    *expected gain* rather than by argmax. Argmax would throw away the graded
    structure the whole metric is about - an E and an S both become "the top
    class" and the ordering between two S products is lost.
    """
    raw = ranker.booster.predict(
        frame[list(ranker.features)], num_iteration=ranker.best_iteration
    )
    raw = np.asarray(raw)
    if ranker.objective == "pointwise_class":
        return (raw @ np.asarray(LABEL_GAIN)).astype(np.float64)
    return raw.astype(np.float64)


def predict_run(
    ranker: TrainedRanker, frame: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """A nested run dict for src.metrics.ndcg_per_query.

    Ids are stringified here because query_id is int64 in the parquet and both
    src.metrics and pytrec_eval key on strings.
    """
    scores = predict(ranker, frame)
    run: dict[str, dict[str, float]] = {}
    for query_id, product_id, score in zip(
        frame["query_id"], frame["product_id"], scores
    ):
        run.setdefault(str(query_id), {})[str(product_id)] = float(score)
    return run
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ranker.py -q`
Expected: PASS, 19 tests. The training tests take a few seconds in total.

- [ ] **Step 5: Smoke-test the real matrix end to end**

Run:

```bash
python - <<'PY'
import pandas as pd
from src.feature_matrix import ALL_FEATURES
from src.metrics import ndcg_per_query
from src.ranker import EARLY_STOP_FOLD, REPORT_FOLD, TRAIN_FOLDS, folds, predict_run, train_ranker

frame = pd.read_parquet("data/features/train.parquet")
ranker = train_ranker(
    folds(frame, TRAIN_FOLDS), folds(frame, [EARLY_STOP_FOLD]), list(ALL_FEATURES)
)
report = folds(frame, [REPORT_FOLD])
qrels = {}
for q, p, r in zip(report["query_id"], report["product_id"], report["qrel"]):
    qrels.setdefault(str(q), {})[str(p)] = int(r)
per_query = ndcg_per_query(predict_run(ranker, report), qrels)
print("rounds", ranker.best_iteration)
print("fold-0 NDCG", sum(per_query.values()) / len(per_query))
PY
```

Expected: a couple of minutes, and a fold-0 NDCG **above 0.8347** — the
measured fixed-weight control. If it lands below 0.8285 (dense similarity
alone), something is wrong with the matrix rather than with the model: check
that `group_sizes` was applied to a sorted frame and that the NaNs were not
filled. Record the number; Task 6 reproduces it properly with an interval.

- [ ] **Step 6: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 7: Commit**

```bash
git add src/ranker.py tests/test_ranker.py
git commit -m "Train LightGBM lambdarank on the ESCI gain mapping with a three-role fold split"
```

---

## Phase 2 Gate

Phase 3 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including both real-matrix tests.
- [ ] `data/features/train.parquet` has 419,653 rows over 20,888 groups and `test.parquet` 181,701 over 8,956.
- [ ] `group_sizes` raises on a frame that is not sorted by `query_id`, and the real matrices pass it.
- [ ] `LABEL_GAIN` is derived from `src.labels.ESCI_GAINS`, and `train_ranker` refuses an override.
- [ ] `TrainedRanker` exposes no NDCG, and the smoke test got its number from `src.metrics`.
- [ ] The smoke test's fold-0 NDCG is above the 0.8347 fixed-weight control, or the discrepancy is understood before Phase 3 starts.

Then: [Phase 3 — The Ablations](phase-3-the-ablations.md).
