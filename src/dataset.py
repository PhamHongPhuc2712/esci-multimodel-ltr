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
    downloads to a .part file that is renamed only on success - an interrupted
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
