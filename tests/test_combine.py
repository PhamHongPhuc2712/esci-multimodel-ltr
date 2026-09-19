import pandas as pd
import pytest

from src.combine import (
    JUDGEMENT_COLUMNS,
    NO_FOLD,
    combine_judgements,
    combine_products,
)


def _esci_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["B1", "B2", "B3"],
            "product_title": ["red shoe", "blue kettle", "green hat"],
            "product_description": ["a red shoe", None, None],
            "product_bullet_point": ["comfortable", None, None],
            "product_brand": ["Acme", "Kettleco", None],
            "product_color": ["red", "blue", None],
        }
    )


def _enrichment() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asin": ["B1", "B2"],
            "type": ["product", "book"],
            "template": ["shoes", "book"],
            "title": ["red shoe", "blue kettle"],
            "subtitle": [None, "2nd ed"],
            "author": [None, "Someone"],
            "description": ["scraped shoe text", "scraped kettle text"],
            "bullets": [["a"], []],
            "image_url": ["http://i/1.jpg", None],
            "category": [["Shoes"], ["Books"]],
            "attrs_json": ['{"Brand": "Acme"}', None],
            "info_json": ["{}", None],
            "stars": [4.7, None],
            "ratings": [54, None],
            "price": [9.99, None],
            "price_multi": [False, False],
            "bsr_rank": [123, None],
            "bsr_category": ["Shoes", None],
            "n_reviews": [2, 0],
        }
    )


# --- the leakage guard ------------------------------------------------------

def test_product_table_carries_no_label_columns():
    # 34,756 products appear in both train and test. A product table holding a
    # label is how product-level target encoding happens by accident.
    combined = combine_products(_esci_products(), _enrichment())
    for banned in ("gain", "qrel", "esci_label", "query_id", "query", "split"):
        assert banned not in combined.columns


def test_product_table_has_one_row_per_product():
    combined = combine_products(_esci_products(), _enrichment())
    assert len(combined) == 3
    assert combined["product_id"].is_unique


def test_products_without_enrichment_are_kept_and_flagged():
    combined = combine_products(_esci_products(), _enrichment()).set_index("product_id")
    assert bool(combined.loc["B1", "has_enrichment"])
    assert not bool(combined.loc["B3", "has_enrichment"])
    assert pd.isna(combined.loc["B3", "s_stars"])


def test_enrichment_columns_are_prefixed_to_avoid_collision():
    # ESCI and ESCI-S both have a title and a description; they must stay
    # distinguishable.
    combined = combine_products(_esci_products(), _enrichment())
    assert "product_title" in combined.columns
    assert "s_title" in combined.columns
    assert "product_description" in combined.columns
    assert "s_description" in combined.columns


# --- the description coalesce ----------------------------------------------

def test_description_prefers_esci_and_records_the_source():
    combined = combine_products(_esci_products(), _enrichment()).set_index("product_id")
    assert combined.loc["B1", "description"] == "a red shoe"
    assert combined.loc["B1", "description_source"] == "esci"


def test_description_falls_back_to_the_scrape_and_records_that():
    # ESCI leaves 47.8% of products with no description; ESCI-S fills 37% of
    # the catalogue. This column is where that gain is realised.
    combined = combine_products(_esci_products(), _enrichment()).set_index("product_id")
    assert combined.loc["B2", "description"] == "scraped kettle text"
    assert combined.loc["B2", "description_source"] == "esci_s"


def test_description_source_is_none_when_neither_has_one():
    combined = combine_products(_esci_products(), _enrichment()).set_index("product_id")
    assert combined.loc["B3", "description_source"] == "none"
    # pandas 3 stores this column as `str` dtype, so a null reads back as nan
    # rather than the None singleton. It is null either way.
    assert pd.isna(combined.loc["B3", "description"])


# --- judgements -------------------------------------------------------------

def _judgements(split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [1, 2],
            "query_id": [10, 11],
            "query": ["a", "b"],
            "product_id": ["B1", "B2"],
            "esci_label": ["E", "S"],
            "gain": [1.0, 0.1],
            "qrel": [100, 10],
        }
    )


def _folds() -> pd.DataFrame:
    return pd.DataFrame({"query_id": [10, 11], "fold": [0, 3]})


def test_judgements_carry_their_split():
    combined = combine_judgements(
        {"train": _judgements("train"), "test": _judgements("test")}, _folds()
    )
    assert set(combined["split"]) == {"train", "test"}
    assert len(combined) == 4


def test_train_judgements_get_their_frozen_fold():
    combined = combine_judgements({"train": _judgements("train")}, _folds())
    assert sorted(combined["fold"]) == [0, 3]


def test_test_judgements_have_no_fold():
    # Folds are carved from train only. A test row with a fold would invite
    # someone to tune on it.
    combined = combine_judgements({"test": _judgements("test")}, _folds())
    assert set(combined["fold"]) == {NO_FOLD}


def test_judgement_columns_are_exactly_the_declared_set():
    combined = combine_judgements({"test": _judgements("test")}, _folds())
    assert list(combined.columns) == list(JUDGEMENT_COLUMNS)


def test_unknown_train_query_without_a_fold_raises():
    folds = pd.DataFrame({"query_id": [10], "fold": [0]})
    with pytest.raises(ValueError, match="no frozen fold"):
        combine_judgements({"train": _judgements("train")}, folds)
