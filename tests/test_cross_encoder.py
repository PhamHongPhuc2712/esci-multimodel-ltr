import pandas as pd
import pytest

from src.cross_encoder import (
    DEFAULT_BACKBONE,
    DEFAULT_MAX_LENGTH,
    LOSSES,
    check_training_folds,
    document_text,
    listwise_dataset,
    pairwise_dataset,
)
from src.ranker import EARLY_STOP_FOLD, REPORT_FOLD, TRAIN_FOLDS


# --- the document side ------------------------------------------------------

def test_the_title_comes_first():
    # max_length truncates the tail, so whatever leads is what the model reads.
    # Plan 3 lost 6 points of its semantic gate to exactly this mistake.
    text = document_text("Steel Water Bottle", "A very long description.")
    assert text.startswith("Steel Water Bottle")


def test_a_missing_description_leaves_just_the_title():
    assert document_text("Steel Water Bottle", None) == "Steel Water Bottle"
    assert document_text("Steel Water Bottle", "") == "Steel Water Bottle"


def test_a_missing_title_does_not_stringify_none():
    # 'None' as literal product text is worse than an empty string.
    assert "None" not in document_text(None, "A description.")


def test_the_document_is_truncated_but_the_title_survives():
    long_description = "x" * 5000
    text = document_text("Steel Water Bottle", long_description, max_chars=100)
    assert len(text) == 100
    assert text.startswith("Steel Water Bottle")


def test_a_title_longer_than_the_budget_is_not_silently_dropped():
    text = document_text("y" * 400, "a description", max_chars=100)
    assert text == "y" * 100


def test_whitespace_is_collapsed():
    assert document_text("A   B", "C\n\nD") == "A B C D"


# --- Review Focus 3: the training folds -------------------------------------

def _matrix(folds=(2, 3, 4)):
    rows = []
    q = 0
    for fold in folds:
        for _ in range(3):
            for p in range(4):
                rows.append(
                    {
                        "query_id": q,
                        "product_id": f"p{q}_{p}",
                        "gain": [0.0, 0.1, 1.0, 0.01][p],
                        "fold": fold,
                    }
                )
            q += 1
    return pd.DataFrame(rows)


def test_check_training_folds_accepts_the_training_folds():
    check_training_folds(_matrix(TRAIN_FOLDS))  # must not raise


def test_check_training_folds_rejects_the_reporting_fold():
    # 22M parameters over 250,485 pairs will memorise. Training on fold 0 and
    # then scoring on it returns an impressively wrong number.
    with pytest.raises(ValueError, match="fold"):
        check_training_folds(_matrix((2, 3, REPORT_FOLD)))


def test_check_training_folds_rejects_the_early_stop_fold():
    with pytest.raises(ValueError, match="fold"):
        check_training_folds(_matrix((2, EARLY_STOP_FOLD)))


def test_check_training_folds_rejects_the_test_split():
    with pytest.raises(ValueError, match="fold"):
        check_training_folds(_matrix((2, -1)))


def test_check_training_folds_names_the_offending_folds():
    with pytest.raises(ValueError) as excinfo:
        check_training_folds(_matrix((2, 0)))
    assert "0" in str(excinfo.value)


# --- the two dataset shapes -------------------------------------------------

def _texts():
    q = {0: "red shoes", 1: "blue hat", 2: "green mug"}
    d = {f"p{i}_{p}": f"product {i}-{p}" for i in range(9) for p in range(4)}
    return q, d


def test_the_listwise_dataset_groups_one_row_per_query():
    matrix = _matrix((2,))
    q, d = _texts()
    ds = listwise_dataset(matrix, q, d)
    assert set(ds) == {"query", "docs", "labels"}
    assert len(ds["query"]) == matrix["query_id"].nunique()
    assert all(len(docs) == len(labels) for docs, labels in zip(ds["docs"], ds["labels"]))


def test_the_listwise_labels_are_the_esci_gains():
    matrix = _matrix((2,))
    q, d = _texts()
    ds = listwise_dataset(matrix, q, d)
    assert sorted(ds["labels"][0]) == [0.0, 0.01, 0.1, 1.0]


def test_the_pairwise_dataset_is_one_row_per_judgement():
    matrix = _matrix((2,))
    q, d = _texts()
    ds = pairwise_dataset(matrix, q, d)
    assert set(ds) == {"query", "doc", "label"}
    assert len(ds["query"]) == len(matrix)


def test_a_query_with_no_text_raises_rather_than_training_on_an_empty_string():
    matrix = _matrix((2,))
    _, d = _texts()
    with pytest.raises(KeyError, match="query text"):
        listwise_dataset(matrix, {}, d)


def test_a_product_with_no_text_raises():
    matrix = _matrix((2,))
    q, _ = _texts()
    with pytest.raises(KeyError, match="document text"):
        listwise_dataset(matrix, q, {})


def test_the_declared_losses_are_the_two_measured_recipes():
    assert LOSSES == ("lambda", "bce")


def test_the_defaults_match_what_was_measured():
    assert DEFAULT_BACKBONE == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert DEFAULT_MAX_LENGTH == 192
