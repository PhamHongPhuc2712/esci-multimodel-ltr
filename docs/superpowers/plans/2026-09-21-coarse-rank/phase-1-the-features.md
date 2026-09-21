# Phase 1 — The Features

**Plan 5 of 7 · Phase 1 of 3 · Tasks 1–2.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into.

**Delivers:** `src/features.py` and `src/pair_scores.py` — every query×product
feature the coarse ranker will see, split into the part that needs nothing on
disk and the part that needs 3 GB of index and a GPU.

**Needs on disk:** `data/combined/{products,judgements}.parquet` (Plan 2),
`data/bm25/` (Plan 4), `data/embeddings/dense/` (Plan 4),
`data/embeddings/catalogue/` (Plan 3). Task 1 needs none of them.

**Owns Review Focus items 3 and 5** (a query that tokenises to nothing, a
missing image scored as zero).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Absence is NaN, never zero.** 20.6% of judged pairs have no image vector. A 0.0 cosine is a score; NaN is an absence.
- **Features are query×product or product-attribute, never product-alone-from-labels.** Nothing in this phase may read `esci_label`, `gain` or `qrel`.
- **ESCI-S presence is an artefact, ESCI's own presence is not.** Both are computed here; Phase 2 puts them in different groups and Phase 3 subtracts one of them.
- **Queries and documents must go through the same tokeniser** for BM25, or a query matches nothing and nothing errors.
- **RAM is not free.** Read Parquet with an explicit `columns=` list.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: The pure features

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `src.features.tokens(text: str | None) -> list[str]`
  - `src.features.token_set(text: str | None) -> frozenset[str]`
  - `src.features.ProductTerms` (frozen dataclass, ten `frozenset`/tuple fields)
  - `src.features.product_terms(row: Mapping[str, Any]) -> ProductTerms`
  - `src.features.product_features(row: Mapping[str, Any]) -> dict[str, float]`
  - `src.features.pair_features(query: str, terms: ProductTerms) -> dict[str, float]`
  - `src.features.indicator_features(row: Mapping[str, Any]) -> dict[str, float]`
  - `src.features.PRODUCT_FEATURES`, `PAIR_FEATURES`, `ESCI_INDICATORS`, `S_INDICATORS`: `tuple[str, ...]`

Everything here is pure: a `Mapping` in, a `dict[str, float]` out. No pandas, no
I/O, no model. That is what lets 47 feature definitions be tested against
hand-written rows in milliseconds, and it is the same boundary `src/esci_s.py`
draws in Plan 2.

The split into three functions is by *cost*, not by taste. `product_terms`
parses `attrs_json` and `info_json`, which is the expensive part, and there are
482,105 distinct products against 601,354 judgements — so Phase 2 computes the
terms once per product and calls `pair_features` once per judgement.

**The tokeniser is deliberately naive** — lowercase, runs of `[a-z0-9]` — and
is *not* the BM25 stemmer. These are overlap counts, not scores: whether
"bottles" matches "bottle" changes a count by one, while using the index's
stemmer here would couple every feature definition to a retrieval artefact and
make the features untestable without `PyStemmer`. `bm25_score` in Task 2 is the
feature that needs the index's own tokenisation, and it gets it.

**Missing means NaN, not zero.** A product with no brand has `brand_match =
NaN`, not 0.0: 0.0 says "the brand does not match the query", NaN says "there
is no brand". `product_brand` is absent for 5.64% of judged products and
`product_color` for 32.18%, so the difference is not a corner case. LightGBM
splits on NaN natively.

- [ ] **Step 1: Write the failing test**

Create `tests/test_features.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_features.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 3: Write the implementation**

Create `src/features.py`:

```python
"""Query x product features for the coarse ranker, as pure functions.

A Mapping in, a dict[str, float] out. No pandas, no I/O, no model, no index -
so all 47 feature definitions are tested against hand-written rows in
milliseconds, the same boundary src/esci_s.py draws in Plan 2.

Three entry points, split by cost rather than by taste. `product_terms` parses
attrs_json and info_json, which is the expensive part, and there are 482,105
distinct products against 601,354 judgements - so the caller computes terms
once per product and calls `pair_features` once per judgement.

Two rules run through every definition:

  * **Missing is NaN, never 0.0.** A product with no brand has brand_match NaN:
    0.0 says "the brand does not match the query", NaN says "there is no
    brand". product_brand is absent for 5.64% of judged products and
    product_color for 32.18%. LightGBM splits on NaN natively.
  * **Nothing here reads a label.** 34,756 products appear in both train and
    test, so a feature computed from labels per product leaks. Every function
    takes a product row, and the product table has no labels on it by
    construction (src/combine.py raises if one appears).

The tokeniser is deliberately naive and is NOT the BM25 stemmer. These are
overlap counts, not scores; using the index's stemmer would couple every
feature to a retrieval artefact and make the features untestable without
PyStemmer. `bm25_score` in src/pair_scores.py is the feature that needs the
index's own tokenisation, and it gets it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

NAN = float("nan")

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: Any) -> list[str]:
    """Lowercase runs of letters and digits. Anything falsy gives []."""
    if not isinstance(text, str) or not text.strip():
        return []
    return _TOKEN.findall(text.lower())


def token_set(text: Any) -> frozenset[str]:
    """`tokens` as a set."""
    return frozenset(tokens(text))


def _is_missing(value: Any) -> bool:
    """True for None, NaN, "" and []."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if hasattr(value, "__len__") and len(value) == 0:
        return True
    return False


def _loads(value: Any) -> dict[str, Any]:
    """Parse a JSON object column, or {} if it is absent or malformed.

    The corpus is a scrape: a row that will not parse must not kill a
    601,354-row pass, and it is indistinguishable from having no attributes.
    """
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass(frozen=True)
class ProductTerms:
    """Every token set one product contributes, parsed once."""

    title: frozenset[str] = frozenset()
    bullets: frozenset[str] = frozenset()
    description: frozenset[str] = frozenset()
    brand: frozenset[str] = frozenset()
    color: frozenset[str] = frozenset()
    category: tuple[frozenset[str], ...] = ()
    attr_keys: frozenset[str] = frozenset()
    attr_values: frozenset[str] = frozenset()
    info_keys: frozenset[str] = frozenset()
    info_values: frozenset[str] = frozenset()
    has_attrs: bool = False
    has_info: bool = False
    has_category: bool = False


def _mapping_terms(payload: dict[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    """Token sets for a mapping's keys and its values, kept apart.

    Values are str()-ed because ESCI-S values are mostly strings but not
    always - a count comes through as an int.
    """
    keys = frozenset(t for key in payload for t in tokens(key))
    values = frozenset(t for value in payload.values() for t in tokens(str(value)))
    return keys, values


def product_terms(row: Mapping[str, Any]) -> ProductTerms:
    """Parse one product row into its token sets. The expensive half."""
    attrs = _loads(row.get("s_attrs_json"))
    info = _loads(row.get("s_info_json"))
    attr_keys, attr_values = _mapping_terms(attrs)
    info_keys, info_values = _mapping_terms(info)

    raw_category = row.get("s_category")
    category: tuple[frozenset[str], ...] = ()
    if not _is_missing(raw_category):
        category = tuple(token_set(level) for level in raw_category)

    return ProductTerms(
        # ESCI's own text. `description` is ESCI's product_description, not the
        # coalesced column: the coalesce is 37.0% ESCI-S, and a "text only"
        # ablation arm built on it would already carry the enrichment.
        title=token_set(row.get("product_title")),
        bullets=token_set(row.get("product_bullet_point")),
        description=token_set(row.get("product_description")),
        brand=token_set(row.get("product_brand")),
        color=token_set(row.get("product_color")),
        category=category,
        attr_keys=attr_keys,
        attr_values=attr_values,
        info_keys=info_keys,
        info_values=info_values,
        has_attrs=bool(attrs),
        has_info=bool(info),
        has_category=bool(category),
    )


PRODUCT_FEATURES: tuple[str, ...] = (
    "title_chars",
    "title_words",
    "esci_desc_chars",
    "bullet_chars",
    "category_depth",
    "n_attrs",
    "n_info",
    "s_stars",
    "s_ratings",
    "s_price",
    "s_price_multi",
    "s_bsr_rank",
    "s_n_reviews",
)

PAIR_FEATURES: tuple[str, ...] = (
    "query_chars",
    "query_words",
    "title_overlap",
    "title_coverage",
    "bullet_coverage",
    "esci_desc_coverage",
    "brand_match",
    "brand_match_all",
    "color_match",
    "category_match_depth",
    "category_match_frac",
    "attr_key_overlap",
    "attr_value_overlap",
    "info_key_overlap",
    "info_value_overlap",
)

# ESCI's own presence. A production ranker also knows whether a product has a
# brand, so these are legitimate features.
ESCI_INDICATORS: tuple[str, ...] = (
    "has_brand",
    "has_color",
    "has_esci_desc",
    "has_bullets",
)

# ESCI-S presence. "Our 2022 scrape failed on this page" is NOT available at
# serving time, so these are an artefact: Plan 5 carries them (LightGBM needs
# the missingness signal to use the values at all) and Ablation 4 subtracts an
# arm built from them alone.
S_INDICATORS: tuple[str, ...] = (
    "has_enrichment",
    "has_stars",
    "has_ratings",
    "has_price",
    "has_bsr",
    "has_attrs",
    "has_info",
    "has_category",
    "desc_from_esci_s",
)


def _length(value: Any) -> float:
    """Character count, or NaN when the field is absent.

    0 would mean "an empty description was written"; there is a difference.
    """
    return NAN if _is_missing(value) else float(len(value))


def _number(value: Any) -> float:
    """A behavioural value as a float, or NaN. Never 0.0 for absent - 0 is a price."""
    if _is_missing(value):
        return NAN
    try:
        result = float(value)
    except (TypeError, ValueError):
        return NAN
    return NAN if math.isnan(result) else result


def product_features(row: Mapping[str, Any]) -> dict[str, float]:
    """The product-only half: lengths, counts and behavioural values.

    No monotone transforms. A gradient-boosted tree splits on order, so
    log1p(ratings) and ratings produce the same splits; a transform here would
    be cargo cult and would make the feature harder to read in an importance
    table.
    """
    terms = product_terms(row)
    price_multi = row.get("s_price_multi")
    return {
        "title_chars": _length(row.get("product_title")),
        "title_words": float(len(tokens(row.get("product_title")))),
        "esci_desc_chars": _length(row.get("product_description")),
        "bullet_chars": _length(row.get("product_bullet_point")),
        "category_depth": float(len(terms.category)) if terms.has_category else NAN,
        "n_attrs": float(len(_loads(row.get("s_attrs_json")))) if terms.has_attrs else NAN,
        "n_info": float(len(_loads(row.get("s_info_json")))) if terms.has_info else NAN,
        "s_stars": _number(row.get("s_stars")),
        "s_ratings": _number(row.get("s_ratings")),
        "s_price": _number(row.get("s_price")),
        # A flag, but only meaningful where there is a price string at all.
        "s_price_multi": NAN if _is_missing(price_multi) else float(bool(price_multi)),
        "s_bsr_rank": _number(row.get("s_bsr_rank")),
        "s_n_reviews": _number(row.get("s_n_reviews")),
    }


def _coverage(query: frozenset[str], field: frozenset[str]) -> float:
    """Share of the query's terms that the field contains. 0.0 for an empty query."""
    if not query:
        return 0.0
    return len(query & field) / len(query)


def _match(query: frozenset[str], field: frozenset[str]) -> tuple[float, float]:
    """(share of the field's terms the query names, all-of-them flag), or NaN."""
    if not field:
        return NAN, NAN
    hit = len(query & field)
    return hit / len(field), float(hit == len(field))


def pair_features(query: str, terms: ProductTerms) -> dict[str, float]:
    """The query x product half. One call per judgement."""
    q = token_set(query)

    brand_match, brand_match_all = _match(q, terms.brand)
    color_match, _ = _match(q, terms.color)

    if terms.has_category:
        # The DEEPEST matching level, 1-based; 0 when nothing matches. Depth is
        # the signal: matching "Office Products" is weak, matching "Pencils" is
        # not, and a flattened category would score them the same.
        depth = 0
        for level, level_terms in enumerate(terms.category, start=1):
            if q & level_terms:
                depth = level
        match_depth = float(depth)
        match_frac = depth / len(terms.category)
    else:
        match_depth = NAN
        match_frac = NAN

    return {
        "query_chars": float(len(query)) if isinstance(query, str) else NAN,
        "query_words": float(len(q)),
        "title_overlap": float(len(q & terms.title)),
        "title_coverage": _coverage(q, terms.title),
        "bullet_coverage": _coverage(q, terms.bullets),
        "esci_desc_coverage": _coverage(q, terms.description),
        "brand_match": brand_match,
        "brand_match_all": brand_match_all,
        "color_match": color_match,
        "category_match_depth": match_depth,
        "category_match_frac": match_frac,
        "attr_key_overlap": float(len(q & terms.attr_keys)) if terms.has_attrs else NAN,
        "attr_value_overlap": (
            float(len(q & terms.attr_values)) if terms.has_attrs else NAN
        ),
        "info_key_overlap": float(len(q & terms.info_keys)) if terms.has_info else NAN,
        "info_value_overlap": (
            float(len(q & terms.info_values)) if terms.has_info else NAN
        ),
    }


def indicator_features(row: Mapping[str, Any]) -> dict[str, float]:
    """Presence flags, ESCI's own and ESCI-S's, in one dict but two groups.

    Never NaN: an indicator answers "was it there", which is always
    answerable, and a missingness flag that is itself missing is nonsense.
    """
    terms = product_terms(row)
    present = lambda key: float(not _is_missing(row.get(key)))  # noqa: E731
    return {
        "has_brand": present("product_brand"),
        "has_color": present("product_color"),
        "has_esci_desc": present("product_description"),
        "has_bullets": present("product_bullet_point"),
        "has_enrichment": float(bool(row.get("has_enrichment"))),
        "has_stars": present("s_stars"),
        "has_ratings": present("s_ratings"),
        "has_price": present("s_price"),
        "has_bsr": present("s_bsr_rank"),
        "has_attrs": float(terms.has_attrs),
        "has_info": float(terms.has_info),
        "has_category": float(terms.has_category),
        "desc_from_esci_s": float(row.get("description_source") == "esci_s"),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_features.py -q`
Expected: PASS, 30 tests.

- [ ] **Step 5: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 6: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "Add pure query-product features with NaN for absent fields"
```

---

## Task 2: The retrieval pair scores

**Files:**
- Create: `src/pair_scores.py`
- Create: `tests/test_pair_scores.py`
- Modify: `pyproject.toml` (add the `ranking` extra)

**Interfaces:**
- Consumes: `src.bm25_index.open_channel`, `.tokenize_texts` (Plan 4); `src.embedding_store.open_store`, `.lookup` (Plan 3); `src.clip_encoder.load_encoder` (Plan 3); `src.baseline_sbert.DEFAULT_MODEL` (Plan 1).
- Produces:
  - `src.pair_scores.PAIR_SCORE_COLUMNS: tuple[str, ...]` = `("bm25_score", "dense_sim", "clip_image_sim")`
  - `src.pair_scores.bm25_pair_scores(pairs, *, tokenize, score_query, progress=None) -> tuple[np.ndarray, int]`
  - `src.pair_scores.similarity_scores(query_vectors, query_index, keys, store) -> np.ndarray`
  - `src.pair_scores.bm25_token_lists(queries, channel) -> list[list[str]]`
  - `src.pair_scores.DEFAULT_OUT: Path` = `Path("data/features")`
  - CLI: `python -m src.pair_scores --split {train,test}` writing
    `data/features/pair-scores-{split}.parquet` with columns
    `query_id`, `product_id`, `bm25_score`, `dense_sim`, `clip_image_sim`

These are the three §4.2 "Retrieval scores", and they are the only features that
need an artefact on disk. Measured on validation fold 0, each one alone already
re-ranks the judged list well above the 0.7440 floor: BM25 **0.8230**, dense
**0.8285**, CLIP image **0.7905**.

**One function serves both vector channels.** `similarity_scores` cosines a
query vector against whatever the row's key names — a `product_id` for the
dense store, an image URL for the CLIP store. Writing it twice is how two
channels drift apart, which is the same argument `src/vector_search.py` makes in
Plan 4.

**Review Focus 5 lives in `similarity_scores`.** `EmbeddingStore.lookup`
already returns NaN rows plus a presence mask; this function must propagate the
NaN rather than let a `0 * nan` or a `nan_to_num` turn it into a score. 20.6% of
judged pairs have no image vector.

**Review Focus 3 lives in `bm25_pair_scores`.** `bm25s.BM25.get_scores([])`
raises `IndexError: list index out of range` from `query_tokens_single[0]`, and
one of fold 0's 4,130 queries tokenises to nothing once English stopwords and
out-of-vocabulary terms are dropped. It must skip those queries with NaN and count them
rather than abort the pass; 7 of the 20,888 train queries are affected.

**The `Tokenized` -> token-strings conversion is not obvious.**
`bm25_index.tokenize_texts` returns a `bm25s.tokenization.Tokenized` carrying
`.ids` (per query, a list of integer ids) and `.vocab` (a `token -> id` dict),
while `get_scores` wants token *strings*. `bm25_token_lists` inverts the vocab
once and maps the ids through it. Passing `.ids` straight to `get_scores` is
accepted — the signature allows a list of ints — but those ints index a
*per-batch* vocabulary, not the index's, so every score comes back wrong and
nothing raises.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pair_scores.py`:

```python
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.pair_scores import (
    PAIR_SCORE_COLUMNS,
    bm25_pair_scores,
    bm25_token_lists,
    similarity_scores,
)


class FakeStore:
    """An EmbeddingStore with the two methods pair scoring uses."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = 2):
        self.dim = dim
        self._vectors = vectors

    def lookup(self, wanted):
        matrix = np.full((len(wanted), self.dim), np.nan, dtype=np.float32)
        present = np.zeros(len(wanted), dtype=bool)
        for i, key in enumerate(wanted):
            if key in self._vectors:
                matrix[i] = self._vectors[key]
                present[i] = True
        return matrix, present


# --- similarity_scores ------------------------------------------------------

def test_similarity_is_the_dot_product_of_the_two_vectors():
    store = FakeStore({"p1": [1.0, 0.0], "p2": [0.0, 1.0]})
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = similarity_scores(q, [0, 0], ["p1", "p2"], store)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_each_row_uses_its_own_query_vector():
    store = FakeStore({"p1": [1.0, 0.0]})
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = similarity_scores(q, [0, 1], ["p1", "p1"], store)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_a_key_the_store_does_not_have_scores_nan():
    # Review Focus 5. 20.6% of judged pairs have no image vector. A 0.0 cosine
    # is a score meaning "orthogonal to the query"; NaN is an absence.
    store = FakeStore({"p1": [1.0, 0.0]})
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    scores = similarity_scores(q, [0, 0], ["p1", "missing"], store)
    assert scores[0] == pytest.approx(1.0)
    assert math.isnan(scores[1])


def test_an_empty_key_scores_nan_rather_than_raising():
    # Products with no image URL carry "" rather than a key.
    store = FakeStore({"p1": [1.0, 0.0]})
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    assert math.isnan(similarity_scores(q, [0], [""], store)[0])


def test_scoring_nothing_returns_an_empty_array():
    store = FakeStore({})
    q = np.zeros((0, 2), dtype=np.float32)
    assert similarity_scores(q, [], [], store).shape == (0,)


def test_a_query_vector_of_the_wrong_width_raises():
    store = FakeStore({"p1": [1.0, 0.0]}, dim=2)
    q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="width"):
        similarity_scores(q, [0], ["p1"], store)


# --- bm25_pair_scores -------------------------------------------------------

def _pairs():
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "query": ["red shoes", "red shoes", "blue hat", "blue hat"],
            "row": [0, 1, 0, 2],
        }
    )


def test_bm25_reads_each_pairs_own_row_from_the_score_vector():
    scores = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out, empty = bm25_pair_scores(
        _pairs(),
        tokenize=lambda qs: [q.split() for q in qs],
        score_query=lambda toks: scores,
    )
    assert out.tolist() == [10.0, 20.0, 10.0, 30.0]
    assert empty == 0


def test_bm25_scores_each_query_once():
    calls = []

    def score_query(toks):
        calls.append(list(toks))
        return np.zeros(3, dtype=np.float32)

    bm25_pair_scores(
        _pairs(), tokenize=lambda qs: [q.split() for q in qs], score_query=score_query
    )
    # Two distinct queries, four rows. Scoring per row would be 20x the work
    # at ~20 judgements per query.
    assert calls == [["red", "shoes"], ["blue", "hat"]]


def test_a_query_that_tokenises_to_nothing_yields_nan_and_is_counted():
    # Review Focus 3. bm25s.get_scores([]) raises IndexError from
    # query_tokens_single[0]; one of fold 0's 4,130 queries is such a query
    # after stopword removal. A 15.7-minute pass must not die on it.
    def score_query(toks):
        assert toks, "an empty token list must never reach get_scores"
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    pairs = _pairs()
    out, empty = bm25_pair_scores(
        pairs,
        tokenize=lambda qs: [[], ["blue", "hat"]],
        score_query=score_query,
    )
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(1.0)
    assert empty == 1


def test_every_query_tokenising_to_nothing_gives_all_nan():
    out, empty = bm25_pair_scores(
        _pairs(),
        tokenize=lambda qs: [[], []],
        score_query=lambda toks: pytest.fail("must not be called"),
    )
    assert np.isnan(out).all()
    assert empty == 2


def test_bm25_result_is_aligned_to_the_input_rows():
    # The frame is deliberately not grouped by query_id: the returned array
    # must line up with the rows as given, not with some internal ordering.
    pairs = pd.DataFrame(
        {
            "query_id": [1, 2, 1],
            "query": ["a", "b", "a"],
            "row": [0, 1, 2],
        }
    )
    out, _ = bm25_pair_scores(
        pairs,
        tokenize=lambda qs: [q.split() for q in qs],
        score_query=lambda toks: np.array([7.0, 8.0, 9.0], dtype=np.float32),
    )
    assert out.tolist() == [7.0, 8.0, 9.0]


def test_progress_is_reported_per_query():
    seen = []
    bm25_pair_scores(
        _pairs(),
        tokenize=lambda qs: [q.split() for q in qs],
        score_query=lambda toks: np.zeros(3, dtype=np.float32),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_a_missing_row_column_raises_rather_than_scoring_zero():
    with pytest.raises(KeyError, match="row"):
        bm25_pair_scores(
            pd.DataFrame({"query_id": [1], "query": ["a"]}),
            tokenize=lambda qs: [["a"]],
            score_query=lambda toks: np.zeros(1, dtype=np.float32),
        )


# --- the Tokenized -> strings conversion ------------------------------------

def test_token_lists_map_ids_back_through_the_vocabulary():
    # bm25s returns per-batch integer ids and a vocab dict; get_scores wants
    # strings. Passing the ids through is accepted and silently wrong - they
    # index the batch's vocabulary, not the index's.
    channel = SimpleNamespace(stemmer=None)
    tokenized = SimpleNamespace(ids=[[1, 0], [0]], vocab={"shoes": 0, "red": 1})
    lists = bm25_token_lists(
        ["red shoes", "shoes"], channel, tokenize=lambda qs, stemmer: tokenized
    )
    assert lists == [["red", "shoes"], ["shoes"]]


def test_token_lists_handle_a_query_that_lost_every_token():
    channel = SimpleNamespace(stemmer=None)
    tokenized = SimpleNamespace(ids=[[], [0]], vocab={"hat": 0})
    assert bm25_token_lists(
        ["the", "hat"], channel, tokenize=lambda qs, stemmer: tokenized
    ) == [[], ["hat"]]


def test_the_score_columns_are_the_three_retrieval_signals():
    assert PAIR_SCORE_COLUMNS == ("bm25_score", "dense_sim", "clip_image_sim")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pair_scores.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.pair_scores'`.

- [ ] **Step 3: Write the implementation**

Create `src/pair_scores.py`:

```python
"""The three retrieval scores, for judged (query, product) pairs.

PROJECT_SPEC.md §4.2 lists BM25 score, dense text similarity and CLIP image
similarity as the coarse ranker's "Retrieval scores". They are the only
features that need an artefact on disk, which is why they live apart from
src/features.py.

Measured on validation fold 0 (4,130 queries, 82,751 judgements), each one
alone re-ranks the judged candidate list well above the 0.7440 floor:

    bm25_score      0.8230    pair coverage 0.9999
    dense_sim       0.8285    pair coverage 1.0000
    clip_image_sim  0.7905    pair coverage 0.7939

Retrieval, not re-ranking, is what Plan 4's channels do: they return the top k
out of 1,215,854 products. Here the product is given and only its score is
wanted, which is a different call - `get_scores` rather than `retrieve` for
BM25, and a keyed lookup rather than a top-k for the two vector stores.

Two failure modes are handled rather than assumed away:

  * **A query can tokenise to nothing.** After English stopword removal and
    the index's vocabulary filter, 7 of the 20,888 train queries have no tokens
    left (1 of them in fold 0), and `bm25s.BM25.get_scores([])` raises
    IndexError from `query_tokens_single[0]`. Those rows get NaN, the count is
    reported, and the pass finishes.
  * **A product can have no vector.** 20.6% of judged pairs have no image
    embedding. `EmbeddingStore.lookup` returns NaN plus a presence mask, and
    that NaN is propagated: a 0.0 cosine is a score meaning "orthogonal to the
    query", which would rank a never-scraped product below a genuinely bad
    match.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_OUT = Path("data/features")
DEFAULT_PRODUCTS = Path("data/combined/products.parquet")
DEFAULT_JUDGEMENTS = Path("data/combined/judgements.parquet")

PAIR_SCORE_COLUMNS: tuple[str, ...] = ("bm25_score", "dense_sim", "clip_image_sim")

DENSE_DIM = 384
CLIP_DIM = 512


def bm25_token_lists(
    queries: Sequence[str],
    channel: Any,
    *,
    tokenize: Callable[[Sequence[str], Any], Any] | None = None,
) -> list[list[str]]:
    """Token *strings* per query, as `get_scores` wants them.

    `bm25_index.tokenize_texts` returns a Tokenized carrying `.ids` (per
    query, integer ids) and `.vocab` (token -> id). Those ids index the
    *batch's* vocabulary, not the index's, so handing them to `get_scores` -
    which the signature permits - scores the wrong terms and raises nothing.
    """
    if tokenize is None:
        from src.bm25_index import tokenize_texts

        tokenize = tokenize_texts

    tokenized = tokenize(list(queries), channel.stemmer)
    ids = getattr(tokenized, "ids", tokenized)
    vocab = getattr(tokenized, "vocab", None)
    if vocab is None:
        return [list(row) for row in ids]
    inverse = [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    return [[inverse[i] for i in row] for row in ids]


def bm25_pair_scores(
    pairs: pd.DataFrame,
    *,
    tokenize: Callable[[Sequence[str]], Sequence[Sequence[str]]],
    score_query: Callable[[Sequence[str]], np.ndarray],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, int]:
    """BM25 score for every row of `pairs`, aligned to `pairs`.

    `pairs` needs `query_id`, `query` and `row` - the latter being the
    product's position in the indexed matrix, which is what
    `bm25s` scores are indexed by. Returns the scores and the number of
    queries that tokenised to nothing; those queries' rows are NaN.

    Scoring is per distinct query, not per row: there are ~20 judgements per
    query, so per-row scoring would be 20x the work for the same answer.
    """
    for column in ("query_id", "query", "row"):
        if column not in pairs.columns:
            raise KeyError(
                f"pairs has no {column!r} column; bm25 scores are read by "
                "matrix position, and guessing one would score the wrong product"
            )

    out = np.full(len(pairs), np.nan, dtype=np.float32)
    rows = pairs["row"].to_numpy()
    groups = pairs.groupby("query_id", sort=False)
    query_ids = list(groups.groups)
    texts = [str(pairs["query"].iloc[groups.indices[q][0]]) for q in query_ids]

    token_lists = list(tokenize(texts))
    n_empty = 0
    for done, (query_id, query_tokens) in enumerate(zip(query_ids, token_lists), start=1):
        if len(query_tokens) == 0:
            # Every term was a stopword or out of vocabulary. get_scores would
            # raise IndexError here; NaN is the honest score.
            n_empty += 1
        else:
            index = groups.indices[query_id]
            scores = np.asarray(score_query(query_tokens))
            out[index] = scores[rows[index]]
        if progress is not None:
            progress(done, len(query_ids))
    return out, n_empty


def similarity_scores(
    query_vectors: np.ndarray,
    query_index: Sequence[int],
    keys: Sequence[str],
    store: Any,
) -> np.ndarray:
    """Cosine of each row's query vector against its key's stored vector.

    One function for both vector channels: the dense store is keyed by
    product_id and the CLIP store by image URL, and the arithmetic is
    identical. Stored vectors are L2-normalised by Plan 3's contract, so the
    cosine is a plain dot product.

    A key the store does not hold scores NaN. `lookup` already returns a NaN
    row and a presence mask for exactly this reason; the mask is re-applied
    here so that a future lookup that returned zeros could not slip through.
    """
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    if len(keys) == 0:
        return np.zeros(0, dtype=np.float32)
    if query_vectors.ndim != 2:
        raise ValueError(f"expected a 2-D query matrix, got {query_vectors.ndim}-D")
    if query_vectors.shape[1] != store.dim:
        raise ValueError(
            f"query width {query_vectors.shape[1]} does not match the store's "
            f"dim {store.dim}"
        )

    matrix, present = store.lookup(list(keys))
    left = query_vectors[np.asarray(query_index, dtype=np.int64)]
    scores = np.einsum("ij,ij->i", left, matrix).astype(np.float32)
    scores[~present] = np.nan
    return scores


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--judgements", type=Path, default=DEFAULT_JUDGEMENTS)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image-store", type=Path,
                        default=Path("data/embeddings/catalogue"))
    parser.add_argument("--dense-store", type=Path, default=Path("data/embeddings/dense"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from src.baseline_sbert import DEFAULT_MODEL
    from src.bm25_index import open_channel as open_bm25
    from src.clip_encoder import load_encoder, resolve_device
    from src.embedding_store import open_store

    judgements = pd.read_parquet(
        args.judgements, columns=["query_id", "query", "product_id", "split"]
    )
    pairs = judgements.loc[judgements["split"] == args.split].reset_index(drop=True)
    queries = pairs.drop_duplicates("query_id")
    print(f"{len(pairs):,} judged pairs, {len(queries):,} queries, split {args.split}")

    # --- BM25: 31.5 ms/query, ~16 minutes for both splits -------------------
    channel = open_bm25()
    row_of = {pid: i for i, pid in enumerate(channel.product_ids)}
    unknown = set(pairs["product_id"]) - row_of.keys()
    if unknown:
        raise ValueError(
            f"{len(unknown)} judged products are not in the bm25 index "
            f"(for example {sorted(unknown)[:3]}); measured 0 on the real "
            "corpus, so this means the index and the product table disagree"
        )
    pairs["row"] = pairs["product_id"].map(row_of).astype(np.int64)

    started = time.time()
    bm25, n_empty = bm25_pair_scores(
        pairs,
        tokenize=lambda texts: bm25_token_lists(texts, channel),
        score_query=channel.index.get_scores,
        progress=lambda done, total: print(
            f"\r  bm25 {done:,}/{total:,}  {done / max(time.time() - started, 1e-9):.0f}/s",
            end="",
            flush=True,
        ),
    )
    print()
    print(f"  bm25 done in {time.time() - started:.0f}s; "
          f"{n_empty} queries tokenised to nothing")
    del channel, row_of

    query_row = {qid: i for i, qid in enumerate(queries["query_id"])}
    query_index = pairs["query_id"].map(query_row).to_numpy()
    device = resolve_device(args.device)
    print(f"encoding queries on {device}")

    # --- dense text ---------------------------------------------------------
    sbert = SentenceTransformer(DEFAULT_MODEL, device=device)
    dense_queries = sbert.encode(
        queries["query"].astype(str).tolist(),
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dense_store = open_store(args.dense_store, dim=DENSE_DIM)
    dense = similarity_scores(
        dense_queries, query_index, pairs["product_id"].astype(str).tolist(), dense_store
    )
    print(f"  dense_sim covers {np.isfinite(dense).mean():.4f} of pairs")
    del sbert, dense_store

    # --- CLIP image ---------------------------------------------------------
    urls = pd.read_parquet(args.products, columns=["product_id", "s_image_url"])
    url_of = dict(zip(urls["product_id"], urls["s_image_url"]))
    del urls
    encoder = load_encoder(device=args.device)
    clip_queries = encoder.encode_texts(queries["query"].astype(str).tolist())
    image_store = open_store(args.image_store, dim=CLIP_DIM)
    image = similarity_scores(
        clip_queries,
        query_index,
        [str(url_of.get(p) or "") for p in pairs["product_id"]],
        image_store,
    )
    print(f"  clip_image_sim covers {np.isfinite(image).mean():.4f} of pairs")

    frame = pd.DataFrame(
        {
            "query_id": pairs["query_id"].to_numpy(),
            "product_id": pairs["product_id"].to_numpy(),
            "bm25_score": bm25,
            "dense_sim": dense,
            "clip_image_sim": image,
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"pair-scores-{args.split}.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {len(frame):,} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pair_scores.py -q`
Expected: PASS, 15 tests.

- [ ] **Step 5: Add the `ranking` extra to `pyproject.toml`**

LightGBM is a new dependency and Phase 2 needs it. Add it now, with the
channels it sits beside, rather than in the middle of Task 4.

In `pyproject.toml`, after the `retrieval` extra:

```toml
ranking = [
    "lightgbm>=4.5",
]
```

Then install it: `uv pip install -e ".[dev,baselines,retrieval,ranking]"`
Verify: `python -c "import lightgbm; print(lightgbm.__version__)"` → `4.7.0` or later.

- [ ] **Step 6: Run the real pass for both splits**

Run:

```bash
python -m src.pair_scores --split train
python -m src.pair_scores --split test
```

Expected, on the real corpus (**under two minutes per split**; the BM25 pass
is 27 s for train, and model loading dominates the rest):

- `train`: 419,653 pairs, 20,888 queries
- `test`: 181,701 pairs, 8,956 queries
- `dense_sim` covers 1.0000 of pairs
- `clip_image_sim` covers ≈0.79 of pairs
- 7 train queries / a handful of test queries reported as tokenising to nothing

If `clip_image_sim` covers 1.0000, the NaN has been filled somewhere and
Review Focus 5 has been violated. If it covers 0.0000, the store keys and the
`s_image_url` column have diverged.

- [ ] **Step 7: Write the data-marked round trip**

Append to `tests/test_pair_scores.py`:

```python
@pytest.mark.data
def test_the_real_pair_scores_are_shaped_as_the_plan_measured():
    frame = pd.read_parquet("data/features/pair-scores-test.parquet")
    assert len(frame) == 181_701
    assert list(frame.columns) == ["query_id", "product_id", *PAIR_SCORE_COLUMNS]

    # Measured 2026-09-21: dense reaches every judged product, the image store
    # reaches 77.50% of them and ~79% of pairs, BM25 all but a rounding error.
    assert np.isfinite(frame["dense_sim"]).mean() == pytest.approx(1.0)
    assert 0.74 < np.isfinite(frame["clip_image_sim"]).mean() < 0.84
    assert np.isfinite(frame["bm25_score"]).mean() > 0.999

    # Cosines of L2-normalised vectors. Outside [-1, 1] means a store of
    # unnormalised vectors, which silently ranks partly by magnitude.
    for column in ("dense_sim", "clip_image_sim"):
        values = frame[column].to_numpy()
        finite = values[np.isfinite(values)]
        assert finite.min() >= -1.001 and finite.max() <= 1.001
```

Run: `python -m pytest tests/test_pair_scores.py -m data -q`
Expected: PASS.

- [ ] **Step 8: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 9: Commit**

```bash
git add src/pair_scores.py tests/test_pair_scores.py pyproject.toml
git commit -m "Score BM25, dense and CLIP similarity for judged pairs"
```

---

## Phase 1 Gate

Phase 2 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the pair-score round trip.
- [ ] `data/features/pair-scores-train.parquet` has 419,653 rows and `pair-scores-test.parquet` has 181,701.
- [ ] `clip_image_sim` is NaN — not 0.0 — for every pair whose product has no stored image vector, and its coverage is reported near 0.79 rather than 1.0.
- [ ] The count of queries that tokenised to nothing is printed by the CLI and is small (1 in fold 0; a large number means the tokeniser or the vocabulary mapping is wrong).
- [ ] `src/features.py` imports nothing from pandas, LightGBM or any store.

Then: [Phase 2 — The Matrix and the Ranker](phase-2-the-matrix-and-the-ranker.md).
