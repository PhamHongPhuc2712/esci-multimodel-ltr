import pytest

from src.esci_s import CORPUS_COLUMNS, is_book, normalise_record


def _product() -> dict:
    """A real us product record, trimmed to the fields under test."""
    return {
        "asin": "B07WP4RXHY",
        "locale": "us",
        "type": "product",
        "template": "home_improvement",
        "title": "YUEPIN U-Tube Clamp 304 Stainless Steel Hose Pipe Cable Strap Clips",
        "description": "Product Description Specification: Material: 304 Stainless Steel",
        "bullets": ["304 stainless steel", "Rubber cushioned"],
        "image": "https://m.media-amazon.com/images/I/61NemNt5X6L.__AC_SX300_.jpg",
        "category": ["Tools & Home Improvement", "Power & Hand Tools", "Clamps"],
        "attrs": {"Material": "304 stainless steel with rubber", "Brand": "YUEPIN"},
        "info": {"Best Sellers Rank": "#496,038 in Tools & Home Improvement"},
        "stars": "4.7 out of 5 stars",
        "ratings": "54 ratings",
        "price": "$9.99",
        "reviews": [{"stars": "5.0", "text": "great"}, {"stars": "4.0", "text": "ok"}],
    }


def _book() -> dict:
    """A real us book record. Note img/desc/attr, and no info block at all."""
    return {
        "asin": "B00X4WHP55",
        "locale": "us",
        "type": "book",
        "template": "book",
        "title": "The Design of Everyday Things",
        "subtitle": "Revised and Expanded Edition",
        "author": "Don Norman",
        "desc": "Design doesn't have to be complicated.",
        "img": "https://m.media-amazon.com/images/I/41yq6qGPZ1L.jpg",
        "category": ["Books", "Engineering & Transportation"],
        "attr": {"Publisher": "Basic Books", "Language": "English"},
        "stars": "4.6 out of 5 stars",
        "ratings": "6,090 ratings",
        "review": "A classic that every designer should read.",
        "reviews": [{"stars": "5.0", "text": "excellent"}],
    }


# --- Review Focus 1: reading only the product spelling drops every book -----

def test_book_description_comes_from_desc_not_description():
    row = normalise_record(_book())
    assert row["description"] == "Design doesn't have to be complicated."


def test_book_attributes_come_from_attr_not_attrs():
    row = normalise_record(_book())
    assert '"Publisher": "Basic Books"' in row["attrs_json"]


def test_book_image_comes_from_img_not_image():
    row = normalise_record(_book())
    assert row["image_url"] == "https://m.media-amazon.com/images/I/41yq6qGPZ1L.jpg"


def test_book_keeps_its_book_only_fields():
    row = normalise_record(_book())
    assert row["author"] == "Don Norman"
    assert row["subtitle"] == "Revised and Expanded Edition"


def test_product_has_no_author_or_subtitle():
    row = normalise_record(_product())
    assert row["author"] is None
    assert row["subtitle"] is None


def test_is_book_distinguishes_the_two_shapes():
    assert is_book(_book())
    assert not is_book(_product())


# --- Review Focus 2: info is product-only -----------------------------------

def test_book_info_is_none_rather_than_an_empty_dict():
    # 0 of 18,727 us books in the real file carry an info block. The absence
    # is structural, so it must be distinguishable from "a product whose info
    # happened to be empty" downstream.
    row = normalise_record(_book())
    assert row["info_json"] is None


def test_product_info_is_carried_as_json():
    row = normalise_record(_product())
    assert "Best Sellers Rank" in row["info_json"]


# --- the image URL must go through image_url() ------------------------------

def test_retired_cdn_bucket_is_stripped():
    # ~50% of ESCI-S image URLs route through a bucket Amazon retired; they
    # return HTTP 400 until the segment is dropped. Reading record["image"]
    # directly would carry the dead URL into the corpus.
    record = _product()
    record["image"] = (
        "https://m.media-amazon.com/images/W/WEBP_402378-T1/images/I/71abc._AC_UL320_.jpg"
    )
    row = normalise_record(record)
    assert row["image_url"] == "https://m.media-amazon.com/images/I/71abc._AC_UL320_.jpg"


def test_record_with_no_image_yields_none():
    record = _product()
    del record["image"]
    assert normalise_record(record)["image_url"] is None


# --- general shape ----------------------------------------------------------

def test_every_row_has_exactly_the_corpus_columns():
    for record in (_product(), _book()):
        assert tuple(normalise_record(record)) == CORPUS_COLUMNS


def test_bullets_are_a_list_even_for_books_which_have_none():
    assert normalise_record(_book())["bullets"] == []
    assert normalise_record(_product())["bullets"] == [
        "304 stainless steel",
        "Rubber cushioned",
    ]


def test_category_is_a_list_and_survives_intact():
    row = normalise_record(_product())
    assert row["category"] == [
        "Tools & Home Improvement",
        "Power & Hand Tools",
        "Clamps",
    ]


def test_missing_category_becomes_an_empty_list():
    record = _product()
    del record["category"]
    assert normalise_record(record)["category"] == []


def test_review_count_is_kept_but_review_text_is_not():
    # Review text is the bulk of the 3.4 GB and PROJECT_SPEC.md §3.2 marks it
    # optional, so the corpus carries the count by default and the text only
    # on request.
    row = normalise_record(_product())
    assert row["n_reviews"] == 2
    assert "reviews_json" not in row or row["reviews_json"] is None


def test_reviews_are_carried_when_requested():
    row = normalise_record(_product(), with_reviews=True)
    assert "great" in row["reviews_json"]


def test_whitespace_is_stripped_from_text_fields():
    record = _product()
    record["title"] = "  spaced out title  "
    assert normalise_record(record)["title"] == "spaced out title"


def test_empty_string_becomes_none():
    record = _product()
    record["description"] = ""
    assert normalise_record(record)["description"] is None


# --- error rows -------------------------------------------------------------

def test_scrape_error_row_raises_rather_than_yielding_an_empty_row():
    error_row = {
        "asin": "B00BROKEN0",
        "locale": "us",
        "type": "error",
        "error": "404",
        "template": "error",
    }
    with pytest.raises(ValueError, match="scrape-error row"):
        normalise_record(error_row)
