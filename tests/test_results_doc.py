import json
import re
from pathlib import Path

import pytest

DOC = Path("docs/RESULTS.md")
RESULTS = Path("docs/results")

# Cited from PROJECT_SPEC.md §5, not measured here. Everything else in the
# writeup must come from a committed results file.
PUBLISHED = {
    0.7483,   # the published random floor - wrong for this discount, quoted only to say so
    0.8107,   # CLIP_text zero-shot
    0.8225,   # CLIP_image zero-shot
    0.8292,   # SBERT_text zero-shot
    0.8562,   # ESCI_baseline, the target
    0.9043,   # KDD Cup 2022 winner
}

NUMBER = re.compile(r"\b0\.\d{4}\b")


def _all_values(node, out):
    if isinstance(node, dict):
        for value in node.values():
            _all_values(value, out)
    elif isinstance(node, list):
        for value in node:
            _all_values(value, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.add(round(float(node), 4))
        out.add(round(abs(float(node)), 4))


def _committed_values():
    values = set()
    for path in sorted(RESULTS.glob("*.json")):
        _all_values(json.loads(path.read_text(encoding="utf-8")), values)
    return values


def test_the_writeup_exists():
    assert DOC.exists(), "PROJECT_SPEC.md §8.8 asks for the writeup"


def test_the_writeup_names_the_target_and_the_measured_floor():
    text = DOC.read_text(encoding="utf-8")
    assert "0.8562" in text          # the target, §5
    assert "0.7467" in text or "0.7454" in text or "0.7468" in text


def test_the_writeup_does_not_quote_the_published_floor_as_this_project_s():
    # CLAUDE.md: compute the floor, never quote it. If 0.7483 appears it must
    # be in a sentence that says it is the published one.
    text = DOC.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "0.7483" in line:
            assert "publish" in line.lower() or "sqid" in line.lower()


@pytest.mark.data
def test_every_figure_in_the_writeup_comes_from_a_committed_result():
    # The writeup is the one artifact read without checking, and this project
    # has met four plausible-but-wrong published numbers already.
    text = DOC.read_text(encoding="utf-8")
    committed = _committed_values() | PUBLISHED
    unsourced = sorted(
        {float(m) for m in NUMBER.findall(text)} - committed
    )
    assert not unsourced, (
        f"figures in docs/RESULTS.md that no committed results file contains: "
        f"{unsourced}. Either quote the file's number or add the measurement."
    )


@pytest.mark.data
def test_the_writeup_states_the_headline_with_its_scope():
    # A 2,000-query sample number presented as a full-test number is the one
    # scope error a reader cannot catch.
    text = DOC.read_text(encoding="utf-8").lower()
    assert "2,000" in text or "2000" in text
    assert "sample" in text
