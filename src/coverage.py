"""Join coverage and missingness bias for the ESCI-S enrichment corpus.

Two questions gate Plan 3, and they are different questions. "How much of
ESCI does ESCI-S cover?" is answered by a join against this project's actual
product set - not by the 91.5% headline, which is over all 1,814,925 ESCI
ASINs across every locale. "Is what is missing missing at random?" is answered
by comparing relevance among judgements whose product carries a field against
those whose product does not, resampling whole queries because judgements are
clustered ~20 to a query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.dataset import ensure_downloaded, load_split
from src.esci_s_etl import DEFAULT_DEST

MIN_JOIN_COVERAGE = 0.90


@dataclass(frozen=True)
class CoverageResult:
    scope: str
    n_products: int
    n_matched: int
    coverage: float
    n_with_image: int
    image_coverage: float

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_products": self.n_products,
            "n_matched": self.n_matched,
            "coverage": self.coverage,
            "n_with_image": self.n_with_image,
            "image_coverage": self.image_coverage,
        }


def load_corpus(
    path: Path = DEFAULT_DEST, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read the enrichment corpus, optionally only some columns."""
    return pd.read_parquet(path, columns=columns)


def rerank_product_ids() -> set[str]:
    """Every judged product across train and test - the re-ranking set."""
    ids: set[str] = set()
    for split in ("train", "test"):
        ids |= set(load_split(split).judgements["product_id"])
    return ids


def catalogue_product_ids() -> set[str]:
    """Every us product in the ESCI products parquet - the recall corpus."""
    products = pd.read_parquet(
        ensure_downloaded("products"), columns=["product_id", "product_locale"]
    )
    return set(products.loc[products["product_locale"] == "us", "product_id"])


def join_coverage(
    product_ids: set[str], corpus: pd.DataFrame, *, scope: str
) -> CoverageResult:
    """Share of `product_ids` present in the corpus, and carrying an image.

    Both shares divide by the size of `product_ids`, never by the number of
    matches or the size of the corpus. The corpus holds every us product,
    which is far more than any one scope asks about, and image coverage
    reported over matches rather than over the product set is exactly how the
    91.5% headline gets mistaken for image coverage when it is really ~75%.
    """
    if not product_ids:
        raise ValueError("no products to measure coverage against")

    matched = corpus.loc[corpus["asin"].isin(product_ids)]
    n_matched = int(matched["asin"].nunique())
    with_image = matched.loc[matched["image_url"].notna() & (matched["image_url"] != "")]
    n_with_image = int(with_image["asin"].nunique())
    return CoverageResult(
        scope=scope,
        n_products=len(product_ids),
        n_matched=n_matched,
        coverage=n_matched / len(product_ids),
        n_with_image=n_with_image,
        image_coverage=n_with_image / len(product_ids),
    )
