import pandas as pd
import pytest

from src.cross_encoder import (
    DEFAULT_BACKBONE,
    DEFAULT_MAX_LENGTH,
    LOSSES,
    TRAINING_RECORD,
    check_training_folds,
    document_text,
    enable_gradient_checkpointing,
    listwise_dataset,
    pairwise_dataset,
    training_record,
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


import numpy as np

from src.cross_encoder import Latency, measure_latency, rerank
from src.rerank_window import Window


class FakeModel:
    """A CrossEncoder with the one method the reranker uses."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def predict(self, pairs, batch_size=256, show_progress_bar=False, **kwargs):
        pairs = list(pairs)
        self.calls.append(len(pairs))
        return np.array([self.scores.get(d, 0.0) for _, d in pairs], dtype=np.float32)


def _windows():
    return [
        Window(query_id="1", window=("a", "b", "c"), tail=("d",)),
        Window(query_id="2", window=("x", "y"), tail=()),
    ]


def _maps():
    q = {"1": "red shoes", "2": "blue hat"}
    d = {k: f"doc {k}" for k in "abcdxy"}
    return q, d


def test_rerank_orders_the_window_by_model_score():
    q, d = _maps()
    model = FakeModel({"doc a": 0.1, "doc b": 0.9, "doc c": 0.5, "doc x": 0.2, "doc y": 0.8})
    out = rerank(model, _windows(), q, d)
    assert out["1"] == ["b", "c", "a"]
    assert out["2"] == ["y", "x"]


def test_rerank_returns_a_permutation_of_each_window():
    q, d = _maps()
    out = rerank(FakeModel(), _windows(), q, d)
    for w in _windows():
        assert sorted(out[w.query_id]) == sorted(w.window)


def test_rerank_never_touches_the_tail():
    q, d = _maps()
    out = rerank(FakeModel({"doc d": 99.0}), _windows(), q, d)
    assert "d" not in out["1"]


def test_rerank_breaks_ties_deterministically():
    q, d = _maps()
    a = rerank(FakeModel(), _windows(), q, d)
    b = rerank(FakeModel(), _windows(), q, d)
    assert a == b


def test_rerank_scores_every_window_pair_once():
    q, d = _maps()
    model = FakeModel()
    rerank(model, _windows(), q, d)
    assert sum(model.calls) == 5  # 3 + 2 window documents, no tail


def test_rerank_of_nothing_is_nothing():
    assert rerank(FakeModel(), [], *_maps()) == {}


# --- Review Focus 5: two latencies, both measured ---------------------------

def test_latency_reports_single_and_batched_separately():
    # Batched is what an offline job costs; single is what a user waits. They
    # differ by 3-5x, and quoting one as the other flatters whichever arm
    # batches better - the cross-encoder does, the LLM cannot.
    q, d = _maps()
    result = measure_latency(FakeModel(), _windows(), q, d, n_queries=2)
    assert isinstance(result, Latency)
    assert result.single_ms > 0
    assert result.batched_ms > 0
    assert result.n_queries == 2
    assert result.n_pairs == 5


def test_latency_single_mode_calls_the_model_once_per_query():
    q, d = _maps()
    model = FakeModel()
    measure_latency(model, _windows(), q, d, n_queries=2)
    # The single-query pass must issue one call per query, not one big batch,
    # or it is measuring throughput again under a different name.
    assert model.calls[:2] == [3, 2]


def test_latency_serialises_both_numbers():
    q, d = _maps()
    payload = measure_latency(FakeModel(), _windows(), q, d, n_queries=2).to_dict()
    assert set(payload) == {"single_ms", "batched_ms", "n_queries", "n_pairs"}


def test_latency_needs_at_least_one_query():
    with pytest.raises(ValueError, match="at least one"):
        measure_latency(FakeModel(), [], *_maps(), n_queries=0)


# --- Plan 7: the scores themselves, for the blend ---------------------------

def test_window_scores_keys_on_query_and_document():
    from src.cross_encoder import window_scores

    q, d = _maps()
    model = FakeModel({"doc b": 0.9})
    scores = window_scores(model, _windows(), q, d)
    assert scores[("1", "b")] == pytest.approx(0.9)
    assert set(scores) == {("1", "a"), ("1", "b"), ("1", "c"), ("2", "x"), ("2", "y")}


def test_rerank_and_window_scores_agree():
    # rerank is a wrapper over window_scores, so an ordering can never
    # disagree with the scores it came from.
    from src.cross_encoder import window_scores

    q, d = _maps()
    scores = window_scores(FakeModel({"doc a": 0.1, "doc b": 0.9, "doc c": 0.5}), _windows(), q, d)
    ordering = rerank(FakeModel({"doc a": 0.1, "doc b": 0.9, "doc c": 0.5}), _windows(), q, d)
    assert ordering["1"] == sorted(
        ["a", "b", "c"], key=lambda doc: (-scores[("1", doc)], doc)
    )


def test_window_scores_of_nothing_is_nothing():
    from src.cross_encoder import window_scores

    assert window_scores(FakeModel(), [], *_maps()) == {}


# --- Plan 9: memory, and saying what trained a model ------------------------

class _WrappedModel:
    """Stands in for the transformers model a CrossEncoder wraps."""

    def __init__(self, switches_on=True):
        self.switches_on = switches_on
        self.is_gradient_checkpointing = False
        self.kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.kwargs = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = self.switches_on


class _CrossEncoder:
    def __init__(self, switches_on=True):
        self.model = _WrappedModel(switches_on)


def test_gradient_checkpointing_is_switched_on_the_wrapped_model():
    model = _CrossEncoder()
    enable_gradient_checkpointing(model)
    assert model.model.is_gradient_checkpointing
    assert model.model.kwargs == {"use_reentrant": False}


def test_a_checkpointing_switch_that_does_not_take_raises():
    # Without it bge-reranker-base spills past 16 GB and crawls at under 0.27
    # queries/s instead of failing - a nine-hour run nobody asked for.
    with pytest.raises(RuntimeError, match="checkpointing"):
        enable_gradient_checkpointing(_CrossEncoder(switches_on=False))


def _record(**overrides):
    fields = dict(
        loss="lambda", backbone="BAAI/bge-reranker-base", epochs=1, batch_size=8,
        seed=0, gradient_checkpointing=True, n_queries=12_519, n_pairs=250_485,
        minutes=66.0,
    )
    return training_record(**(fields | overrides))


def test_the_training_record_names_the_recipe():
    record = _record()
    assert record["target"] == "labels"
    assert record["init"] == "BAAI/bge-reranker-base"
    assert record["loss"] == "LambdaLoss"
    assert record["batch_size"] == 8
    assert record["gradient_checkpointing"] is True
    assert "whole query groups" in record["data"]


def test_the_training_record_names_the_pairwise_loss_too():
    assert _record(loss="bce")["loss"] == "BinaryCrossEntropyLoss"


def test_a_training_record_for_an_unknown_loss_is_refused():
    with pytest.raises(ValueError, match="loss"):
        _record(loss="hinge")


def test_both_fine_tunes_write_the_same_record_file():
    # src.distill_report reads this one name from every model directory.
    from src.distill import TRAINING_RECORD as window_record

    assert TRAINING_RECORD == window_record == "training.json"
