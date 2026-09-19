"""Join ESCI and ESCI-S into the two tables everything downstream reads.

Deliberately two tables, not one. ESCI carries the labels; ESCI-S carries
product attributes. Putting both in one row would place `gain` beside every
product field, and 34,756 products appear in both train and test - so a
product-level target encoding would be one groupby away, which is exactly the
leak CLAUDE.md forbids. Keeping the product table label-free makes that
mistake awkward rather than merely discouraged.

  products.parquet    one row per us product, no labels, joinable by product_id
  judgements.parquet  one row per judgement, labels + split + frozen fold

The `description` column is the practical payoff of the join: ESCI has a
description for 52.2% of products and ESCI-S fills a further 37.0%, taking the
union to 89.2%. `description_source` records which one supplied it so a feature
can condition on provenance instead of pretending they are the same field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.coverage import load_corpus
from src.dataset import ensure_downloaded, load_split
from src.splits import FOLDS_PATH, load_folds

DEFAULT_DEST = Path("data/combined")

# Test rows have no fold: folds are carved from train only, and a fold number
# on a test row is an invitation to tune on it.
NO_FOLD = -1

JUDGEMENT_COLUMNS: tuple[str, ...] = (
    "example_id",
    "query_id",
    "query",
    "product_id",
    "esci_label",
    "gain",
    "qrel",
    "split",
    "fold",
)

# ESCI-S columns, prefixed on the way in: both datasets have `title` and
# `description`, and silently letting one win would be untraceable later.
ENRICHMENT_COLUMNS: tuple[str, ...] = (
    "type",
    "template",
    "title",
    "subtitle",
    "author",
    "description",
    "bullets",
    "image_url",
    "category",
    "attrs_json",
    "info_json",
    "stars",
    "ratings",
    "price",
    "price_multi",
    "bsr_rank",
    "bsr_category",
    "n_reviews",
)

_LABEL_COLUMNS = frozenset(
    {"gain", "qrel", "esci_label", "query_id", "query", "split", "example_id"}
)


def _non_empty(series: pd.Series) -> pd.Series:
    """True where a value is genuinely present - "" and [] are not."""
    if series.dtype == object:
        return series.map(
            lambda v: v is not None
            and not (isinstance(v, float) and pd.isna(v))
            and len(v) > 0
        )
    return series.notna()


def combine_products(
    esci_products: pd.DataFrame, enrichment: pd.DataFrame
) -> pd.DataFrame:
    """One row per ESCI product, with ESCI-S attributes attached where present.

    A left join: products ESCI-S never scraped are kept with null enrichment
    and `has_enrichment` False, because dropping them would silently shrink
    the corpus the recall stage searches.
    """
    renamed = enrichment.rename(
        columns={c: f"s_{c}" for c in ENRICHMENT_COLUMNS}
    ).rename(columns={"asin": "product_id"})
    keep = ["product_id"] + [
        f"s_{c}" for c in ENRICHMENT_COLUMNS if f"s_{c}" in renamed.columns
    ]

    combined = esci_products.drop_duplicates("product_id").merge(
        renamed[keep], on="product_id", how="left"
    )
    combined["has_enrichment"] = combined["product_id"].isin(set(enrichment["asin"]))

    esci_has = _non_empty(combined["product_description"])
    scrape_has = (
        _non_empty(combined["s_description"])
        if "s_description" in combined.columns
        else pd.Series(False, index=combined.index)
    )
    combined["description"] = combined["product_description"].where(
        esci_has, combined.get("s_description")
    )
    combined.loc[~esci_has & ~scrape_has, "description"] = None
    combined["description_source"] = "none"
    combined.loc[~esci_has & scrape_has, "description_source"] = "esci_s"
    combined.loc[esci_has, "description_source"] = "esci"

    leaked = _LABEL_COLUMNS & set(combined.columns)
    if leaked:
        raise ValueError(
            f"the product table must carry no labels, found {sorted(leaked)}; "
            "34,756 products appear in both train and test, so a label here "
            "makes product-level target encoding a one-line mistake"
        )
    return combined.reset_index(drop=True)


def combine_judgements(
    by_split: dict[str, pd.DataFrame], folds: pd.DataFrame
) -> pd.DataFrame:
    """One row per judgement, tagged with its split and its frozen fold."""
    assignment = dict(zip(folds["query_id"], folds["fold"]))
    frames = []
    for split, judgements in by_split.items():
        frame = judgements.copy()
        frame["split"] = split
        if split == "train":
            missing = set(frame["query_id"]) - assignment.keys()
            if missing:
                raise ValueError(
                    f"{len(missing)} train queries have no frozen fold "
                    f"(for example {sorted(missing)[:3]}); regenerate with "
                    "python -m src.cli freeze-splits"
                )
            frame["fold"] = frame["query_id"].map(assignment)
        else:
            frame["fold"] = NO_FOLD
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined[list(JUDGEMENT_COLUMNS)]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--scope",
        default="catalogue",
        choices=["catalogue", "rerank"],
        help="catalogue = every us product (what recall searches); "
        "rerank = only judged products",
    )
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    enrichment = load_corpus(columns=["asin", *ENRICHMENT_COLUMNS])

    by_split = {split: load_split(split).judgements for split in ("train", "test")}
    judgements = combine_judgements(by_split, load_folds(FOLDS_PATH))

    if args.scope == "rerank":
        esci_products = pd.concat(
            [load_split(s).products for s in ("train", "test")], ignore_index=True
        )
    else:
        all_products = pd.read_parquet(ensure_downloaded("products"))
        esci_products = all_products.loc[all_products["product_locale"] == "us"][
            [
                "product_id",
                "product_title",
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
            ]
        ]

    products = combine_products(esci_products, enrichment)

    products_path = args.dest / "products.parquet"
    judgements_path = args.dest / "judgements.parquet"
    products.to_parquet(products_path, index=False, compression="zstd")
    judgements.to_parquet(judgements_path, index=False, compression="zstd")

    print(f"products   {len(products):>9,} rows -> {products_path}")
    print(f"           enriched {products['has_enrichment'].mean():.2%}")
    print(
        "           description "
        + ", ".join(
            f"{k} {v:.2%}"
            for k, v in products["description_source"]
            .value_counts(normalize=True)
            .items()
        )
    )
    print(f"judgements {len(judgements):>9,} rows -> {judgements_path}")
    print(
        "           "
        + ", ".join(
            f"{k} {v:,}" for k, v in judgements["split"].value_counts().items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
