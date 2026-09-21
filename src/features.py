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
