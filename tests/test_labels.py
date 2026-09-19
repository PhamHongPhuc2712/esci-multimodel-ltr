import pytest

from src.labels import (
    ESCI_GAINS,
    ESCI_QRELS,
    QREL_SCALE,
    label_to_gain,
    label_to_qrel,
)


def test_gains_match_the_spec():
    assert ESCI_GAINS == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def test_substitute_outranks_complement():
    # The official prepare_trec_eval_files.py swaps these two. Under the swap
    # the random floor moves 0.7467 -> 0.7141, which is larger than most
    # method gains, so this ordering is pinned explicitly.
    assert ESCI_GAINS["S"] > ESCI_GAINS["C"]
    assert ESCI_QRELS["S"] > ESCI_QRELS["C"]


def test_integer_qrels_are_the_gains_scaled():
    assert QREL_SCALE == 100
    for label, gain in ESCI_GAINS.items():
        assert ESCI_QRELS[label] == round(gain * QREL_SCALE)


def test_integer_qrels_are_actually_integers():
    # pytrec_eval rejects float relevance values.
    assert all(isinstance(v, int) for v in ESCI_QRELS.values())


def test_lookup_helpers_agree_with_the_tables():
    assert label_to_gain("E") == 1.0
    assert label_to_qrel("E") == 100


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unknown ESCI label"):
        label_to_gain("X")
    with pytest.raises(ValueError, match="unknown ESCI label"):
        label_to_qrel("X")
