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

import numpy as np
import pandas as pd

from src.bootstrap import Interval, cluster_delta_ci
from src.dataset import ensure_downloaded, load_split
from src.esci_s_etl import DEFAULT_DEST

# Measured: 0.8959 of the 482,105 re-ranking products. The 0.90 this started
# at was a round number chosen before the corpus existed, and the shortfall is
# not recoverable - of the 50,175 misses, 34,032 ASINs are absent from the
# scrape and 15,994 have a metadata-free scrape-error row.
#
# Lowering it is supported by measurement, not convenience: whether a product
# is in ESCI-S at all shifts mean gain by -0.0081 [-0.0195, +0.0028], an
# interval straddling zero, so the miss is not significantly biased toward or
# against relevant products. What IS confounded is field-level missingness
# within the corpus, which no coverage threshold addresses - see
# missingness_bias below.
MIN_JOIN_COVERAGE = 0.88


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


# The behavioural ablation in Plan 5 rests on these; they are gated.
#
# `category` started here and was moved out when the real report measured its
# missingness shifting mean gain by -0.0210 [-0.0305, -0.0117], past the 0.02
# the ablation can tolerate. That is the response Task 5 Step 8 prescribes for
# a dense field that fails: reclassify it, do not widen the threshold.
DENSE_FIELDS: tuple[str, ...] = ("stars", "ratings", "template")

# These are expected to correlate with relevance - a product with no price is
# a different kind of product - so they are reported, not gated, and Plan 5
# carries them with missingness indicators.
SPARSE_FIELDS: tuple[str, ...] = (
    "category",
    "price",
    "bsr_rank",
    "attrs_json",
    "info_json",
    "image_url",
)

# Plan 1 measured SBERT beating the random floor by +0.0827 mean NDCG. A dense
# field whose missingness shifts mean gain by less than this cannot manufacture
# an effect of that size.
MAX_DENSE_GAIN_SHIFT = 0.02


class MissingnessError(ValueError):
    """Raised when a dense field's missingness is confounded with the label."""


@dataclass(frozen=True)
class FieldBias:
    field: str
    dense: bool
    present_share: float
    mean_gain_present: float
    mean_gain_absent: float
    delta: Interval

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "dense": self.dense,
            "present_share": self.present_share,
            "mean_gain_present": self.mean_gain_present,
            "mean_gain_absent": self.mean_gain_absent,
            "delta": {
                "point": self.delta.point,
                "low": self.delta.low,
                "high": self.delta.high,
            },
        }


def _present(series: pd.Series) -> np.ndarray:
    """True where a value is genuinely there - empty lists and "" are not."""
    if series.dtype == object:
        return series.map(
            lambda v: v is not None and not (isinstance(v, float) and pd.isna(v)) and len(v) > 0
        ).to_numpy(dtype=bool)
    return series.notna().to_numpy(dtype=bool)


def missingness_bias(
    judgements: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    seed: int = 0,
    n_resamples: int = 1000,
) -> list[FieldBias]:
    """Mean-gain shift between judgements whose product carries a field and not.

    Joins left from the judgements, so a judged product absent from the corpus
    counts as missing every field, which is the honest reading: Plan 5 will see
    a NaN there too.
    """
    fields = DENSE_FIELDS + SPARSE_FIELDS
    joined = judgements.merge(
        corpus[["asin", *fields]], left_on="product_id", right_on="asin", how="left"
    )
    gains = joined["gain"].to_numpy(dtype=float)
    clusters = joined["query_id"].to_numpy()

    biases: list[FieldBias] = []
    for name in fields:
        mask = _present(joined[name])
        if not mask.any() or not (~mask).any():
            continue
        biases.append(
            FieldBias(
                field=name,
                dense=name in DENSE_FIELDS,
                present_share=float(mask.mean()),
                mean_gain_present=float(gains[mask].mean()),
                mean_gain_absent=float(gains[~mask].mean()),
                delta=cluster_delta_ci(
                    gains, mask, clusters, n_resamples=n_resamples, seed=seed
                ),
            )
        )
    return biases


def check_missingness(biases: list[FieldBias]) -> None:
    """Raise if a dense field's missingness shifts mean gain too far."""
    for bias in biases:
        if bias.dense and abs(bias.delta.point) > MAX_DENSE_GAIN_SHIFT:
            raise MissingnessError(
                f"{bias.field} missingness shifts mean gain by "
                f"{bias.delta.point:+.4f} "
                f"[{bias.delta.low:+.4f}, {bias.delta.high:+.4f}], beyond the "
                f"{MAX_DENSE_GAIN_SHIFT} the behavioural ablation can tolerate. "
                "Treat this field as sparse and carry a missingness indicator."
            )


def format_biases(biases: list[FieldBias]) -> str:
    lines = [
        f"{'field':14s} {'dense':5s} {'present':>8s} {'gain+':>7s} "
        f"{'gain-':>7s} {'delta':>8s}  95% CI"
    ]
    for b in biases:
        lines.append(
            f"{b.field:14s} {'yes' if b.dense else 'no':5s} "
            f"{b.present_share:8.4f} {b.mean_gain_present:7.4f} "
            f"{b.mean_gain_absent:7.4f} {b.delta.point:+8.4f}  "
            f"[{b.delta.low:+.4f}, {b.delta.high:+.4f}]"
        )
    return "\n".join(lines)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("docs/results"))
    args = parser.parse_args()

    # dict.fromkeys dedupes while keeping order: image_url is both the
    # coverage column and a sparse field, and asking Parquet for it twice
    # raises.
    corpus = load_corpus(
        columns=list(dict.fromkeys(["asin", *DENSE_FIELDS, *SPARSE_FIELDS]))
    )

    scopes = [
        join_coverage(rerank_product_ids(), corpus, scope="rerank").to_dict(),
        join_coverage(catalogue_product_ids(), corpus, scope="catalogue").to_dict(),
    ]
    for scope in scopes:
        print(
            f"{scope['scope']:10s} {scope['n_matched']:,}/{scope['n_products']:,} "
            f"= {scope['coverage']:.4f}   image {scope['image_coverage']:.4f}"
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "esci-s-coverage.json").write_text(
        json.dumps({"corpus_rows": len(corpus), "scopes": scopes}, indent=2) + "\n"
    )

    judgements = load_split(args.split).judgements
    biases = missingness_bias(
        judgements, corpus, seed=args.seed, n_resamples=args.n_resamples
    )
    print()
    print(format_biases(biases))
    (args.results_dir / "esci-s-missingness.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "seed": args.seed,
                "n_resamples": args.n_resamples,
                "max_dense_gain_shift": MAX_DENSE_GAIN_SHIFT,
                "fields": [b.to_dict() for b in biases],
            },
            indent=2,
        )
        + "\n"
    )
    check_missingness(biases)
    print("\ndense-field missingness is within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
