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
