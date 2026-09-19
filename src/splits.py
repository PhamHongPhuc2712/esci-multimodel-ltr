"""The frozen validation split, carved from train by query_id.

ESCI ships no official validation split, and the official train/test split is
query-level. Splitting within a query group would put some of a query's judged
products in train and the rest in validation, which leaks the answer.

Folds come from a salted SHA-256 of the query id rather than a library's
K-fold, so the assignment survives a scikit-learn upgrade and does not depend
on the row order of the frame it was computed from. The result is committed to
splits/val_folds.csv with a checksum: "frozen" has to mean something.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

N_FOLDS = 5
SALT = "esci-multimodel-ltr/val-folds/v1"
FOLDS_PATH = Path("splits/val_folds.csv")


def fold_for_query_id(query_id: int, n_folds: int = N_FOLDS) -> int:
    """Deterministic fold for one query id.

    SHA-256, not hash(): Python salts str.__hash__ per process, so a fold built
    on hash() reshuffles between runs.
    """
    digest = hashlib.sha256(f"{SALT}:{int(query_id)}".encode()).hexdigest()
    return int(digest[:16], 16) % n_folds


def assign_folds(query_ids: Iterable[int], n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Fold assignment for a set of query ids, sorted by query id."""
    unique = sorted({int(q) for q in query_ids})
    return pd.DataFrame(
        {
            "query_id": unique,
            "fold": [fold_for_query_id(q, n_folds) for q in unique],
        }
    )


def freeze_folds(folds: pd.DataFrame, path: Path = FOLDS_PATH) -> str:
    """Write the folds and a sha256 sidecar. Returns the digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = folds.sort_values("query_id").reset_index(drop=True)
    body = ordered.to_csv(index=False, lineterminator="\n")
    path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def load_folds(path: Path = FOLDS_PATH) -> pd.DataFrame:
    """Load the frozen folds, verifying the checksum."""
    path = Path(path)
    body = path.read_text(encoding="utf-8")
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if actual != expected:
            raise ValueError(
                f"{path} does not match its recorded sha256 "
                f"({actual} != {expected}); the frozen validation split has "
                "been modified, which invalidates every tuning decision made "
                "against it"
            )
    return pd.read_csv(path)


def split_train_val(
    judgements: pd.DataFrame, folds: pd.DataFrame, val_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split judgements into (train, validation) at a query-group boundary."""
    available = set(folds["fold"])
    if val_fold not in available:
        raise ValueError(
            f"val_fold {val_fold} is not one of {sorted(available)}"
        )
    assignment = dict(zip(folds["query_id"], folds["fold"]))
    missing = set(judgements["query_id"]) - assignment.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} query ids are not in the frozen folds "
            f"(for example {sorted(missing)[:3]}); they would be dropped from "
            "both sides silently"
        )
    is_val = judgements["query_id"].map(assignment) == val_fold
    return (
        judgements.loc[~is_val].reset_index(drop=True),
        judgements.loc[is_val].reset_index(drop=True),
    )
