import pytest

from src.esci_s import (
    CORPUS_COLUMNS,
    is_book,
    normalise_record,
    parse_best_sellers_rank,
    parse_price,
    parse_ratings,
    parse_stars,
)


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


# --- stars ------------------------------------------------------------------

def test_parse_stars_reads_the_real_format():
    assert parse_stars("4.7 out of 5 stars") == 4.7


def test_parse_stars_handles_a_whole_number():
    assert parse_stars("4.0 out of 5 stars") == 4.0
    assert parse_stars("5 out of 5 stars") == 5.0


def test_parse_stars_returns_none_for_absent_or_unrecognised():
    assert parse_stars(None) is None
    assert parse_stars("") is None
    assert parse_stars("great product") is None


# --- ratings ----------------------------------------------------------------

def test_parse_ratings_strips_thousands_separators():
    assert parse_ratings("1,116 ratings") == 1116


def test_parse_ratings_handles_the_singular():
    # 5,022 of 444,706 sampled ratings say "1 rating", not "1 ratings".
    assert parse_ratings("1 rating") == 1


def test_parse_ratings_returns_none_for_absent_or_unrecognised():
    assert parse_ratings(None) is None
    assert parse_ratings("no ratings yet") is None


# --- Review Focus 3: multi-price strings ------------------------------------

def test_parse_price_reads_a_plain_price():
    assert parse_price("$9.99") == (9.99, False)


def test_parse_price_strips_thousands_separators():
    assert parse_price("$1,234.56") == (1234.56, False)


def test_parse_price_takes_the_first_of_a_multi_price_string_and_says_so():
    # 66 of 132,846 sampled prices carry several prices - sale or variant
    # pairs, in no reliable order ("$19.99 $7.39" and "$15.28 $12.99" both
    # occur). float(value.strip("$")) raises on every one of them, and a bare
    # regex search would pick one silently. The flag is what lets Plan 5
    # exclude them rather than quietly ranking on an arbitrary pick.
    assert parse_price("$11.53 $12.99") == (11.53, True)
    assert parse_price("$24.95 $29.99 $19.99 $19.99") == (24.95, True)


def test_parse_price_returns_none_for_absent_or_unrecognised():
    assert parse_price(None) == (None, False)
    assert parse_price("see price in cart") == (None, False)


# --- Best Sellers Rank ------------------------------------------------------

def test_parse_bsr_reads_the_overall_rank_and_its_category():
    info = {
        "Best Sellers Rank": (
            "#141,895 in Cell Phones & Accessories "
            "( See Top 100 in Cell Phones & Accessories ) "
            "#10,494 in Flip Cell Phone Cases"
        )
    }
    assert parse_best_sellers_rank(info) == (141895, "Cell Phones & Accessories")


def test_parse_bsr_is_not_confused_by_the_see_top_100_parenthetical():
    # The parenthetical repeats "in <category>" without a leading '#', so a
    # looser pattern would match it and report the wrong rank.
    info = {
        "Best Sellers Rank": (
            "#1,206 in Industrial & Scientific "
            "( See Top 100 in Industrial & Scientific ) "
            "#3 in Material Transport Equipment"
        )
    }
    rank, category = parse_best_sellers_rank(info)
    assert rank == 1206
    assert category == "Industrial & Scientific"


def test_parse_bsr_returns_none_when_the_info_block_is_absent():
    # This is the book case: 0 of 18,727 us books carry info at all.
    assert parse_best_sellers_rank(None) == (None, None)
    assert parse_best_sellers_rank({}) == (None, None)


def test_parse_bsr_returns_none_when_info_has_no_rank_key():
    assert parse_best_sellers_rank({"Manufacturer": "YUEPIN"}) == (None, None)


# --- the parsers are wired into the row -------------------------------------

def test_normalised_product_row_carries_parsed_values():
    row = normalise_record(_product())
    assert row["stars"] == 4.7
    assert row["ratings"] == 54
    assert row["price"] == 9.99
    assert row["price_multi"] is False
    assert row["bsr_rank"] == 496038
    assert row["bsr_category"] == "Tools & Home Improvement"


def test_normalised_book_row_has_no_rank_because_books_have_no_info():
    row = normalise_record(_book())
    assert row["stars"] == 4.6
    assert row["ratings"] == 6090
    assert row["bsr_rank"] is None
    assert row["bsr_category"] is None
