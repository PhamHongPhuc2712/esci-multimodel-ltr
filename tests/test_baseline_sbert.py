import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline_sbert import product_text, score_pairs


def _products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["B1", "B2"],
            "product_title": ["red running shoe", "blue kettle"],
            "product_brand": ["Acme", None],
            "product_color": [None, "blue"],
            "product_description": ["a shoe", None],
            "product_bullet_point": [None, None],
        }
    )


def test_product_text_uses_the_title_by_default():
    assert list(product_text(_products(), ["product_title"])) == [
        "red running shoe",
        "blue kettle",
    ]


def test_product_text_joins_requested_fields_and_skips_nulls():
    text = product_text(_products(), ["product_title", "product_brand", "product_color"])
    assert list(text) == ["red running shoe Acme", "blue kettle blue"]


def test_product_text_never_yields_the_string_none():
    # A naive str() over a null column produces the literal "None" or "nan",
    # which the encoder then embeds as if it were product text.
    text = product_text(_products(), ["product_title", "product_brand"])
    assert not any("None" in t or "nan" in t for t in text)


def test_product_text_rejects_an_unknown_field():
    with pytest.raises(KeyError, match="product_weight"):
        product_text(_products(), ["product_weight"])


def _fake_encode(texts: list[str]) -> np.ndarray:
    """One-hot on the first character; cosine similarity is then 1 or 0."""
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    out = np.zeros((len(texts), len(alphabet)), dtype=np.float32)
    for row, text in enumerate(texts):
        out[row, alphabet.index(text[:1].lower() or " ")] = 1.0
    return out


def test_score_pairs_gives_a_score_to_every_judgement():
    judgements = pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "query": ["red shoe", "red shoe", "blue kettle"],
            "product_id": ["B1", "B2", "B1"],
        }
    )
    scores = score_pairs(judgements, _products(), _fake_encode, ["product_title"])
    assert list(scores.columns) == ["query_id", "product_id", "score"]
    assert len(scores) == 3


def test_score_pairs_rewards_the_matching_product():
    judgements = pd.DataFrame(
        {
            "query_id": [1, 1],
            "query": ["red shoe", "red shoe"],
            "product_id": ["B1", "B2"],
        }
    )
    scores = score_pairs(judgements, _products(), _fake_encode, ["product_title"])
    by_product = dict(zip(scores["product_id"], scores["score"]))
    assert by_product["B1"] > by_product["B2"]


def test_score_pairs_encodes_each_distinct_query_once():
    # 181,701 judgements cover only 8,956 queries. Encoding per judgement
    # would be 20x the work for the same answer.
    seen: list[list[str]] = []

    def counting_encode(texts: list[str]) -> np.ndarray:
        seen.append(texts)
        return _fake_encode(texts)

    judgements = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "query": ["red shoe"] * 3,
            "product_id": ["B1", "B2", "B1"],
        }
    )
    score_pairs(judgements, _products(), counting_encode, ["product_title"])
    queries_encoded, products_encoded = seen
    assert len(queries_encoded) == 1
    assert len(products_encoded) == 2


def test_score_pairs_raises_when_a_judged_product_has_no_metadata():
    judgements = pd.DataFrame(
        {"query_id": [1], "query": ["red shoe"], "product_id": ["B-MISSING"]}
    )
    with pytest.raises(ValueError, match="B-MISSING"):
        score_pairs(judgements, _products(), _fake_encode, ["product_title"])


@pytest.mark.data
@pytest.mark.slow
def test_published_sbert_number_is_reproduced():
    # PROJECT_SPEC.md §5: SBERT_text zero-shot = 0.8292 (all-MiniLM-L12-v2).
    # The cited paper does not state which product fields it encoded, so the
    # band allows a title-only vs. title+brand difference while still failing
    # loudly if the harness itself is wrong.
    payload = json.loads(Path("docs/results/sbert-title-test.json").read_text())
    assert 0.820 <= payload["ndcg"]["point"] <= 0.840
    assert payload["lift_over_floor"]["low"] > 0
