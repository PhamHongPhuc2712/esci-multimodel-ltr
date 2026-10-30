"""The project site is generated from committed results and quotes nothing else."""

import json
import re
from pathlib import Path

import pytest

from src.rerank_window import Window
from src.site import PAGE, RESULTS, pick_example, render, signed

NUMBER = re.compile(r"\b0\.\d{4}\b")
# Cited, not measured: the published ESCI_baseline the project targets.
PUBLISHED = {0.8562}


def _committed_values():
    values = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            values.add(round(float(node), 4))
            values.add(round(abs(float(node)), 4))

    for path in sorted(RESULTS.glob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return values


def test_the_committed_page_is_up_to_date():
    assert PAGE.read_text(encoding="utf-8") == render(), "run python -m src.site"


def test_every_figure_on_the_page_comes_from_a_committed_result():
    text = PAGE.read_text(encoding="utf-8")
    unsourced = sorted({float(m) for m in NUMBER.findall(text)} - _committed_values() - PUBLISHED)
    assert not unsourced, unsourced


def test_the_page_loads_no_script():
    text = PAGE.read_text(encoding="utf-8")
    assert "<script" not in text
    assert set(re.findall(r'(?:href|src)="(https://[^/"]+)', text)) <= {
        "https://fonts.googleapis.com", "https://fonts.gstatic.com", "https://github.com"}


def test_a_negative_delta_carries_a_true_minus_sign():
    assert signed(-0.0008) == "−0.0008"
    assert signed(0.0078) == "+0.0078"


def test_the_example_is_drawn_by_rule_not_by_eye():
    import pandas as pd

    labels = {"a": "EEE", "b": "ESI", "c": "ESC", "d": "EIS"}
    windows = [Window(query_id=q, window=tuple(f"{q}{i}" for i in range(3)), tail=()) for q in labels]
    matrix = pd.DataFrame([(q, f"{q}{i}", l) for q, ls in labels.items() for i, l in enumerate(ls)],
                          columns=["query_id", "product_id", "esci_label"])
    first, _ = pick_example(windows, matrix, seed=1)
    again, _ = pick_example(list(reversed(windows)), matrix, seed=1)
    assert first.query_id == again.query_id
    assert first.query_id != "a"
    with pytest.raises(ValueError, match="three labels"):
        pick_example(windows[:1], matrix, seed=1)
