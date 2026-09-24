import json
import re
from pathlib import Path

import pytest

# The writeup, and the README that a visitor reads first. Both quote the
# headline figures, so both are held to the same sourcing rule.
DOCS = (Path("docs/RESULTS.md"), Path("README.md"))
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


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_the_writeup_exists(doc):
    assert doc.exists(), "PROJECT_SPEC.md §8.8 asks for the writeup"


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_the_writeup_names_the_target_and_the_measured_floor(doc):
    text = doc.read_text(encoding="utf-8")
    assert "0.8562" in text          # the target, §5
    assert "0.7467" in text or "0.7454" in text or "0.7468" in text


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_the_writeup_does_not_quote_the_published_floor_as_this_project_s(doc):
    # CLAUDE.md: compute the floor, never quote it. If 0.7483 appears it must
    # be in a sentence that says it is the published one.
    text = doc.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "0.7483" in line:
            assert "publish" in line.lower() or "sqid" in line.lower()


@pytest.mark.data
@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_every_figure_in_the_writeup_comes_from_a_committed_result(doc):
    # The writeup is the one artifact read without checking, and this project
    # has met four plausible-but-wrong published numbers already.
    text = doc.read_text(encoding="utf-8")
    committed = _committed_values() | PUBLISHED
    unsourced = sorted(
        {float(m) for m in NUMBER.findall(text)} - committed
    )
    assert not unsourced, (
        f"figures in {doc} that no committed results file contains: "
        f"{unsourced}. Either quote the file's number or add the measurement."
    )


@pytest.mark.data
@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_the_writeup_states_the_headline_with_its_scope(doc):
    # The headline is a full-split number, and must say so: the LLM arm was
    # first measured on a 2,000-query sample, and a sample figure passed off as
    # the full split's is the one scope error a reader cannot catch. Both docs
    # name the split's size beside the headline, and name the sample as one.
    text = doc.read_text(encoding="utf-8").lower()
    assert "8,956" in text
    assert "2,000-query sample" in text
