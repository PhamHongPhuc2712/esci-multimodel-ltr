import math

import pytest

from src.features import (
    ESCI_INDICATORS,
    PAIR_FEATURES,
    PRODUCT_FEATURES,
    S_INDICATORS,
    ProductTerms,
    indicator_features,
    pair_features,
    product_features,
    product_terms,
    token_set,
    tokens,
)


# --- the tokeniser ----------------------------------------------------------

def test_tokens_lowercases_and_splits_on_punctuation():
    assert tokens("Sony WH-1000XM4, Black") == ["sony", "wh", "1000xm4", "black"]


def test_tokens_of_nothing_is_empty():
    assert tokens(None) == []
    assert tokens("") == []
    assert tokens("   ") == []


def test_tokens_keeps_digits_attached_to_letters():
    # "a7iii" and "32oz" are the whole query in a lot of ESCI rows.
    assert tokens("sony a7iii 32oz") == ["sony", "a7iii", "32oz"]


def test_token_set_deduplicates():
    assert token_set("red red shirt") == frozenset({"red", "shirt"})


# --- product_terms ----------------------------------------------------------

def _row(**overrides):
    """A judged-product row with every column the features read."""
    row = {
        "product_title": "Amazon Basics Wooden Pencils",
        "product_description": "A pack of pencils.",
        "product_bullet_point": "Pre-sharpened\nNumber 2 lead",
        "product_brand": "Amazon Basics",
        "product_color": "Yellow",
        "s_category": ["Office Products", "Writing Supplies", "Pencils"],
        "s_attrs_json": '{"Brand": "Amazon Basics", "Color": "Yellow"}',
        "s_info_json": '{"Item Weight": "3.2 ounces"}',
        "s_stars": 4.7,
        "s_ratings": 1116.0,
        "s_price": 4.99,
        "s_price_multi": False,
        "s_bsr_rank": 141895.0,
        "s_n_reviews": 8.0,
        "has_enrichment": True,
        "description_source": "esci",
    }
    row.update(overrides)
    return row


def test_product_terms_tokenises_every_text_field():
    terms = product_terms(_row())
    assert "pencils" in terms.title
    assert "sharpened" in terms.bullets
    assert "pack" in terms.description
    assert terms.brand == frozenset({"amazon", "basics"})
    assert terms.color == frozenset({"yellow"})


def test_product_terms_keeps_category_levels_separate():
    # The depth of the match is the feature, so the levels cannot be flattened.
    terms = product_terms(_row())
    assert len(terms.category) == 3
    assert terms.category[2] == frozenset({"pencils"})


def test_product_terms_splits_attribute_keys_from_values():
    terms = product_terms(_row())
    assert "brand" in terms.attr_keys
    assert "yellow" in terms.attr_values
    assert "yellow" not in terms.attr_keys


def test_product_terms_survives_malformed_attribute_json():
    # The corpus is a scrape. A row that will not parse must not kill a
    # 601,354-row pass; it is indistinguishable from having no attributes.
    terms = product_terms(_row(s_attrs_json="{not json"))
    assert terms.attr_keys == frozenset()
    assert terms.attr_values == frozenset()


def test_product_terms_handles_a_missing_everything():
    terms = product_terms({"product_title": "thing"})
    assert terms.title == frozenset({"thing"})
    assert terms.brand == frozenset()
    assert terms.category == ()


def test_product_terms_reads_non_string_attribute_values():
    # ESCI-S values are mostly strings but not always.
    terms = product_terms(_row(s_attrs_json='{"Count": 12}'))
    assert "12" in terms.attr_values


# --- product_features -------------------------------------------------------

def test_product_features_emits_exactly_the_declared_names():
    assert set(product_features(_row())) == set(PRODUCT_FEATURES)


def test_product_features_measures_lengths():
    f = product_features(_row())
    assert f["title_chars"] == len("Amazon Basics Wooden Pencils")
    assert f["title_words"] == 4
    assert f["esci_desc_chars"] == len("A pack of pencils.")


def test_product_features_passes_behavioural_values_through():
    f = product_features(_row())
    assert f["s_stars"] == pytest.approx(4.7)
    assert f["s_ratings"] == pytest.approx(1116.0)
    assert f["s_price"] == pytest.approx(4.99)
    assert f["s_bsr_rank"] == pytest.approx(141895.0)
    assert f["s_price_multi"] == 0.0


def test_a_missing_behavioural_value_is_nan_not_zero():
    # s_price is present for 26.67% of judged products. Zero is a price.
    f = product_features(_row(s_price=None, s_stars=None))
    assert math.isnan(f["s_price"])
    assert math.isnan(f["s_stars"])


def test_a_missing_description_has_nan_length_not_zero():
    # Length 0 means "an empty description was written"; NaN means there is none.
    f = product_features(_row(product_description=None))
    assert math.isnan(f["esci_desc_chars"])


def test_category_depth_is_nan_without_a_category():
    assert product_features(_row())["category_depth"] == 3
    assert math.isnan(product_features(_row(s_category=None))["category_depth"])


def test_attribute_counts_are_nan_without_attributes():
    assert product_features(_row())["n_attrs"] == 2
    assert math.isnan(product_features(_row(s_attrs_json=None))["n_attrs"])


def test_no_product_feature_reads_a_label():
    # Review Focus: 34,756 products appear in both splits. A feature computed
    # from labels per product leaks train labels into test.
    with_labels = _row(esci_label="E", gain=1.0, qrel=100)
    assert product_features(with_labels) == product_features(_row())


# --- pair_features ----------------------------------------------------------

def test_pair_features_emits_exactly_the_declared_names():
    assert set(pair_features("pencils", product_terms(_row()))) == set(PAIR_FEATURES)


def test_title_coverage_is_the_share_of_query_terms_in_the_title():
    f = pair_features("wooden pencils blue", product_terms(_row()))
    assert f["title_overlap"] == 2
    assert f["title_coverage"] == pytest.approx(2 / 3)


def test_an_empty_query_does_not_divide_by_zero():
    f = pair_features("!!!", product_terms(_row()))
    assert f["query_words"] == 0
    assert f["title_coverage"] == 0.0


def test_brand_match_is_the_share_of_brand_terms_the_query_names():
    f = pair_features("amazon pencils", product_terms(_row()))
    assert f["brand_match"] == pytest.approx(0.5)
    assert f["brand_match_all"] == 0.0

    full = pair_features("amazon basics pencils", product_terms(_row()))
    assert full["brand_match"] == pytest.approx(1.0)
    assert full["brand_match_all"] == 1.0


def test_brand_match_is_nan_when_there_is_no_brand():
    # 5.64% of judged products. 0.0 would say "the brand does not match".
    f = pair_features("amazon", product_terms(_row(product_brand=None)))
    assert math.isnan(f["brand_match"])
    assert math.isnan(f["brand_match_all"])


def test_color_match_is_nan_when_there_is_no_colour():
    # 32.18% of judged products carry no product_color.
    f = pair_features("yellow", product_terms(_row(product_color=None)))
    assert math.isnan(f["color_match"])


def test_category_match_depth_is_the_deepest_matching_level():
    terms = product_terms(_row())  # Office Products / Writing Supplies / Pencils
    assert pair_features("pencils", terms)["category_match_depth"] == 3
    assert pair_features("office", terms)["category_match_depth"] == 1
    assert pair_features("stapler", terms)["category_match_depth"] == 0


def test_category_match_fraction_normalises_by_depth():
    f = pair_features("pencils", product_terms(_row()))
    assert f["category_match_frac"] == pytest.approx(1.0)


def test_category_match_is_nan_without_a_category():
    f = pair_features("pencils", product_terms(_row(s_category=None)))
    assert math.isnan(f["category_match_depth"])
    assert math.isnan(f["category_match_frac"])


def test_attribute_overlap_counts_keys_and_values_separately():
    f = pair_features("yellow brand", product_terms(_row()))
    assert f["attr_key_overlap"] == 1     # "brand"
    assert f["attr_value_overlap"] == 1   # "yellow"


def test_attribute_overlap_is_nan_without_attributes():
    # attrs are present for 56.96% of judged products.
    f = pair_features("yellow", product_terms(_row(s_attrs_json=None)))
    assert math.isnan(f["attr_key_overlap"])
    assert math.isnan(f["attr_value_overlap"])


def test_info_overlap_is_nan_without_info():
    # info is product-only: 0 of 61,999 us books carry one, so for a book it is
    # structurally absent rather than missing at random.
    f = pair_features("weight", product_terms(_row(s_info_json=None)))
    assert math.isnan(f["info_key_overlap"])
    assert math.isnan(f["info_value_overlap"])


# --- indicators -------------------------------------------------------------

def test_indicator_features_split_esci_from_esci_s():
    f = indicator_features(_row())
    assert set(f) == set(ESCI_INDICATORS) | set(S_INDICATORS)
    assert set(ESCI_INDICATORS).isdisjoint(S_INDICATORS)


def test_esci_indicators_report_esci_presence():
    assert indicator_features(_row())["has_brand"] == 1.0
    assert indicator_features(_row(product_brand=None))["has_brand"] == 0.0


def test_esci_s_indicators_report_scrape_presence():
    f = indicator_features(_row())
    assert f["has_enrichment"] == 1.0
    assert f["has_stars"] == 1.0
    assert f["has_price"] == 1.0

    absent = indicator_features(_row(has_enrichment=False, s_stars=None, s_price=None))
    assert absent["has_enrichment"] == 0.0
    assert absent["has_stars"] == 0.0
    assert absent["has_price"] == 0.0


def test_description_source_becomes_an_indicator():
    # The coalesce took description coverage from 52.2% to 89.2%. Which side
    # supplied it is an ESCI-S presence fact, so it belongs with the artefacts.
    assert indicator_features(_row(description_source="esci"))["desc_from_esci_s"] == 0.0
    assert indicator_features(_row(description_source="esci_s"))["desc_from_esci_s"] == 1.0
    assert indicator_features(_row(description_source="none"))["desc_from_esci_s"] == 0.0


def test_indicators_are_never_nan():
    # An indicator answers "was it there", which is always answerable. A NaN
    # indicator would be a missingness flag that is itself missing.
    f = indicator_features({"product_title": "thing"})
    assert not any(math.isnan(v) for v in f.values())


def test_no_feature_name_appears_in_two_groups():
    groups = [PRODUCT_FEATURES, PAIR_FEATURES, ESCI_INDICATORS, S_INDICATORS]
    flat = [name for group in groups for name in group]
    assert len(flat) == len(set(flat))
