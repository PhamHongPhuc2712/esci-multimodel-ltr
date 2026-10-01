import pandas as pd
import pytest

from src.dense_finetune import (
    RETRIEVER_RECORD,
    retriever_record,
    training_pairs,
)
from src.ranker import TRAIN_FOLDS


def _judgements(rows):
    return pd.DataFrame(
        rows, columns=["query_id", "query", "product_id", "esci_label", "split", "fold"]
    )


def _text(*ids):
    return {p: f"text of {p}" for p in ids}


# --- what the retriever learns from ------------------------------------------

def test_only_exact_products_are_positives():
    # Recall is measured on Exact products; a Substitute as a positive would
    # teach the retriever that "close enough" is the target.
    frame = _judgements([
        (1, "steel bottle", "A", "E", "train", 2),
        (1, "steel bottle", "B", "S", "train", 2),
        (1, "steel bottle", "C", "I", "train", 2),
        (1, "steel bottle", "D", "C", "train", 2),
    ])
    pairs = training_pairs(frame, _text("A", "B", "C", "D"))
    assert pairs == {"anchor": ["steel bottle"], "positive": ["text of A"]}


def test_every_exact_judgement_becomes_one_pair_in_a_stable_order():
    frame = _judgements([
        (2, "mug", "M2", "E", "train", 3),
        (1, "bottle", "B1", "E", "train", 2),
        (1, "bottle", "B2", "E", "train", 2),
    ])
    pairs = training_pairs(frame, _text("M2", "B1", "B2"))
    assert pairs["anchor"] == ["bottle", "bottle", "mug"]
    assert pairs["positive"] == ["text of B1", "text of B2", "text of M2"]


@pytest.mark.parametrize("fold", [0, 1])
def test_a_validation_fold_in_the_training_frame_is_refused(fold):
    # Fold 1 chooses the fusion settings and fold 0 is reported; a retriever
    # that trained on either is being scored on its own training queries.
    frame = _judgements([
        (1, "bottle", "B1", "E", "train", 2),
        (9, "lamp", "L1", "E", "train", fold),
    ])
    with pytest.raises(ValueError, match="fold"):
        training_pairs(frame, _text("B1", "L1"))


def test_a_test_row_in_the_training_frame_is_refused():
    frame = _judgements([
        (1, "bottle", "B1", "E", "train", 2),
        (9, "lamp", "L1", "E", "test", -1),
    ])
    with pytest.raises(ValueError, match="fold"):
        training_pairs(frame, _text("B1", "L1"))


def test_the_default_folds_are_the_ranker_s_training_folds():
    frame = _judgements([(1, "bottle", "B1", "E", "train", f) for f in TRAIN_FOLDS])
    frame["product_id"] = ["B1", "B2", "B3"]
    assert len(training_pairs(frame, _text("B1", "B2", "B3"))["anchor"]) == 3


def test_a_positive_with_no_text_raises():
    # An empty document would teach the retriever that the query matches nothing.
    frame = _judgements([(1, "bottle", "B1", "E", "train", 2)])
    with pytest.raises(KeyError, match="B1"):
        training_pairs(frame, {})


def test_no_exact_products_at_all_raises():
    frame = _judgements([(1, "bottle", "B1", "S", "train", 2)])
    with pytest.raises(ValueError, match="no Exact"):
        training_pairs(frame, _text("B1"))


# --- saying what trained a model ---------------------------------------------

def test_the_record_names_the_recipe():
    record = retriever_record(
        base="BAAI/bge-base-en-v1.5", epochs=1, batch_size=256, learning_rate=3e-5,
        max_seq=256, mini_batch_size=32, seed=0, n_pairs=108_627, n_queries=12_519,
        minutes=20.0,
    )
    assert record["base"] == "BAAI/bge-base-en-v1.5"
    assert record["loss"] == "CachedMultipleNegativesRankingLoss"
    assert record["positives"] == "Exact"
    assert record["folds"] == list(TRAIN_FOLDS)
    assert record["max_seq"] == 256
    assert RETRIEVER_RECORD == "training.json"
