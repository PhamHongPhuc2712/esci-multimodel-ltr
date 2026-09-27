import pandas as pd
import pytest

from src.distill import (
    INITS,
    TARGETS,
    cross_fit_orderings,
    gains_from_frame,
    pairwise_accuracy,
    split_halves,
    window_dataset,
    window_targets,
)
from src.rerank_window import Window

# Window order a, b, c, d. Grades: b and d Exact, c Substitute, a Irrelevant.
GAINS = {
    ("1", "a"): 0.0, ("1", "b"): 1.0, ("1", "c"): 0.1, ("1", "d"): 1.0,
    ("1", "z"): 0.0, ("2", "x"): 1.0, ("2", "y"): 0.0,
}
TEACHER = ["d", "a", "b", "c"]          # the teacher's order, best first
QUERY_TEXT = {"1": "red shoes", "2": "blue hat"}
DOC_TEXT = {d: f"product {d}" for d in "abcdxyz"}


def _window():
    return Window(query_id="1", window=("a", "b", "c", "d"), tail=("z",))


def _windows():
    return [_window(), Window(query_id="2", window=("x", "y"), tail=())]


def _targets(kind):
    return dict(
        zip(_window().window,
            window_targets(_window(), kind, gains=GAINS, llm_ordering=TEACHER))
    )


# --- the targets ------------------------------------------------------------

def test_the_targets_are_the_three_the_pilot_compares():
    assert TARGETS == ("labels", "llm", "hybrid")


def test_the_inits_are_the_two_the_pilot_measured():
    assert set(INITS) == {"scratch", "landed"}


def test_the_label_target_is_the_gain_in_window_order():
    assert window_targets(_window(), "labels", gains=GAINS) == [0.0, 1.0, 0.1, 1.0]


def test_the_teacher_s_first_choice_gets_the_highest_target():
    # Review Focus 4. The loss reads a higher label as more relevant. A target
    # built from the position itself would teach the student to invert the
    # teacher, and the result would read as "distillation does not work".
    targets = _targets("llm")
    assert targets["d"] == 4.0          # the teacher put d first
    assert targets["c"] == 1.0          # and c last
    assert targets["d"] > targets["a"] > targets["b"] > targets["c"]


def test_the_hybrid_target_lets_the_grade_beat_the_teacher():
    # The teacher put Irrelevant a second, above Exact b and Substitute c.
    targets = _targets("hybrid")
    assert targets["b"] > targets["a"]
    assert targets["c"] > targets["a"]


def test_the_hybrid_target_lets_the_teacher_order_a_grade():
    # b and d are both Exact; the labels are silent and the teacher put d first.
    targets = _targets("hybrid")
    assert targets["d"] > targets["b"]


def test_a_teacher_target_without_a_teacher_ordering_raises():
    for kind in ("llm", "hybrid"):
        with pytest.raises(KeyError, match="teacher"):
            window_targets(_window(), kind, gains=GAINS)


def test_a_teacher_ordering_that_is_not_a_permutation_raises():
    with pytest.raises(ValueError, match="permutation"):
        window_targets(_window(), "llm", gains=GAINS, llm_ordering=["d", "d", "b", "c"])


def test_an_unknown_target_raises():
    with pytest.raises(ValueError, match="target"):
        window_targets(_window(), "soft", gains=GAINS)


def test_an_unjudged_window_document_raises():
    gains = {k: v for k, v in GAINS.items() if k != ("1", "b")}
    with pytest.raises(KeyError, match="label"):
        window_targets(_window(), "labels", gains=gains)


def test_gains_are_keyed_by_string_ids():
    # data/features/train.parquet stores query_id as int64; windows hold str.
    frame = pd.DataFrame({"query_id": [1, 1], "product_id": ["a", "b"], "gain": [1.0, 0.0]})
    assert gains_from_frame(frame) == {("1", "a"): 1.0, ("1", "b"): 0.0}


# --- Review Focus 3: a window the teacher did not answer --------------------

def test_a_window_without_a_teacher_answer_is_dropped_and_reported():
    dataset, dropped = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="llm", gains=GAINS,
        llm_orderings={"1": TEACHER},
    )
    assert dropped == ["2"]
    assert dataset["query"] == ["red shoes"]


def test_a_dropped_window_is_never_filled_with_the_stage_2_order():
    # src.llm_rerank falls back to the Stage 2 order on a malformed answer. A
    # student taught that would be learning Stage 2 under the teacher's name.
    dataset, _ = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="hybrid", gains=GAINS,
        llm_orderings={"1": TEACHER},
    )
    assert len(dataset["docs"]) == len(dataset["labels"]) == 1
    assert ["product x", "product y"] not in dataset["docs"]


def test_the_label_target_drops_nothing():
    dataset, dropped = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="labels", gains=GAINS
    )
    assert dropped == []
    assert dataset["labels"] == [[0.0, 1.0, 0.1, 1.0], [1.0, 0.0]]


def test_the_dataset_rows_line_up():
    dataset, _ = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="labels", gains=GAINS
    )
    assert dataset["docs"][0] == ["product a", "product b", "product c", "product d"]
    assert all(len(d) == len(t) for d, t in zip(dataset["docs"], dataset["labels"]))


def test_an_unknown_target_is_refused_before_anything_is_dropped():
    with pytest.raises(ValueError, match="target"):
        window_dataset(_windows(), QUERY_TEXT, DOC_TEXT, kind="soft", gains=GAINS)


# --- Review Focus 2: the fold-0 pilot is cross-fitted -----------------------

def test_the_halves_are_disjoint_and_cover_every_query():
    first, second = split_halves([str(i) for i in range(11)])
    assert not first & second
    assert first | second == {str(i) for i in range(11)}
    assert abs(len(first) - len(second)) <= 1


def test_the_halves_are_fixed_by_the_seed():
    ids = [str(i) for i in range(50)]
    assert split_halves(ids, seed=0) == split_halves(ids, seed=0)
    assert split_halves(ids, seed=0) != split_halves(ids, seed=1)


def test_the_halves_do_not_depend_on_input_order_or_id_type():
    ids = list(range(50))
    assert split_halves(ids) == split_halves([str(i) for i in reversed(ids)])


def test_no_window_is_ordered_by_a_model_that_trained_on_it():
    many = [Window(query_id=str(i), window=("a", "b"), tail=()) for i in range(10)]
    trained_on = []

    def fit(ws):
        ids = frozenset(w.query_id for w in ws)
        trained_on.append(ids)
        return ids                      # the "model" is the set it trained on

    def score(model, ws):
        assert not model & {w.query_id for w in ws}
        return {w.query_id: list(reversed(w.window)) for w in ws}

    halves = split_halves([w.query_id for w in many])
    out = cross_fit_orderings(many, halves, fit=fit, score=score)
    assert set(out) == {w.query_id for w in many}
    assert len(trained_on) == 2
    assert not trained_on[0] & trained_on[1]


def test_overlapping_halves_are_refused():
    many = [Window(query_id=str(i), window=("a",), tail=()) for i in range(4)]
    with pytest.raises(ValueError, match="share"):
        cross_fit_orderings(
            many, ({"0", "1", "2"}, {"2", "3"}),
            fit=lambda ws: None, score=lambda m, ws: {},
        )


def test_a_window_in_neither_half_is_refused():
    many = [Window(query_id=str(i), window=("a",), tail=()) for i in range(4)]
    with pytest.raises(ValueError, match="neither half"):
        cross_fit_orderings(
            many, ({"0"}, {"1"}), fit=lambda ws: None, score=lambda m, ws: {}
        )


def test_a_scorer_that_orders_the_training_half_is_caught():
    many = [Window(query_id=str(i), window=("a",), tail=()) for i in range(4)]
    with pytest.raises(AssertionError, match="outside its half"):
        cross_fit_orderings(
            many, ({"0", "1"}, {"2", "3"}),
            fit=lambda ws: None,
            score=lambda m, ws: {w.query_id: ["a"] for w in many},
        )


# --- the diagnostic ---------------------------------------------------------

def test_pairwise_accuracy_of_the_ideal_order_is_one():
    ideal = {"1": ["b", "d", "c", "a"], "2": ["x", "y"]}
    assert pairwise_accuracy(_windows(), ideal, GAINS) == 1.0


def test_pairwise_accuracy_of_the_reversed_order_is_zero():
    worst = {"1": ["a", "c", "d", "b"], "2": ["y", "x"]}
    assert pairwise_accuracy(_windows(), worst, GAINS) == 0.0


def test_pairwise_accuracy_ignores_pairs_inside_a_grade():
    # b and d are both Exact, so their order is neither right nor wrong.
    one = {"1": ["b", "d", "c", "a"], "2": ["x", "y"]}
    two = {"1": ["d", "b", "c", "a"], "2": ["x", "y"]}
    assert pairwise_accuracy(_windows(), one, GAINS) == pairwise_accuracy(
        _windows(), two, GAINS
    )


def test_pairwise_accuracy_counts_each_mixed_pair_once():
    # Window 1 has five mixed-grade pairs (six, less b~d) and window 2 one.
    # Getting only x above y right is 1 of 6.
    order = {"1": ["a", "c", "d", "b"], "2": ["x", "y"]}
    assert pairwise_accuracy(_windows(), order, GAINS) == pytest.approx(1 / 6)


def test_pairwise_accuracy_needs_an_ordering_for_every_window():
    with pytest.raises(KeyError, match="ordering"):
        pairwise_accuracy(_windows(), {"1": ["b", "d", "c", "a"]}, GAINS)


@pytest.mark.data
def test_the_training_windows_are_carved_out_of_fold():
    # Review Focus 1. Measured 2026-09-27: 12,519 windows, and only 34.5% of
    # them hold the same ten documents as the in-sample ordering's windows.
    from src.distill import training_windows
    from src.rerank_window import windows
    from src.stage2_scores import load_stage2

    windows_, matrix = training_windows()
    assert len(windows_) == 12_519
    assert set(matrix["fold"]) == {2, 3, 4}
    assert min(len(w.window) for w in windows_) >= 8
    in_sample = load_stage2("train")
    in_sample = in_sample.loc[in_sample["in_sample"]]
    carved = {w.query_id: set(w.window) for w in windows(in_sample, k=10)}
    same = sum(carved[w.query_id] == set(w.window) for w in windows_)
    assert same / len(windows_) == pytest.approx(0.345, abs=0.01)
