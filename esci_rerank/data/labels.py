"""ESCI relevance labels and their gain values.

The gains come from PROJECT_SPEC.md §2. The integer table exists because
pytrec_eval requires integer relevance values; NDCG is invariant to the
scale factor, which tests/test_metrics.py pins.

The official amazon-science/esci-data `prepare_trec_eval_files.py` swaps the
S and C values. Published numbers differ depending on which mapping was used,
so this module is the single place the mapping is defined.
"""

from __future__ import annotations

ESCI_GAINS: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

QREL_SCALE = 100

ESCI_QRELS: dict[str, int] = {
    label: round(gain * QREL_SCALE) for label, gain in ESCI_GAINS.items()
}


def label_to_gain(label: str) -> float:
    """Gain for an ESCI label, as a float on the spec's 1.0/0.1/0.01/0.0 scale."""
    try:
        return ESCI_GAINS[label]
    except KeyError:
        raise ValueError(f"unknown ESCI label {label!r}; expected one of E, S, C, I") from None


def label_to_qrel(label: str) -> int:
    """Gain for an ESCI label as an integer, for pytrec_eval / trec_eval qrels."""
    try:
        return ESCI_QRELS[label]
    except KeyError:
        raise ValueError(f"unknown ESCI label {label!r}; expected one of E, S, C, I") from None
