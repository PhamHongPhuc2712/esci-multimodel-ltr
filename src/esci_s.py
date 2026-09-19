"""One ESCI-S record -> one flat typed row.

ESCI-S is a scrape, and its records come in three shapes that do not share a
schema: products, books and scrape errors. A book is not a product with fields
missing - it spells the same information differently (`img` for `image`,
`desc` for `description`, `attr` for `attrs`), carries `author` and `subtitle`
that products never have, and has no `info` block at all. Reading only the
product spelling drops every book without raising anything.

Measured on a 500,356-record prefix of the real file: 0 of 18,727 `us` books
carry `info`. Best Sellers Rank lives in `info`, so for books it is
structurally absent rather than missing at random, and src/coverage.py is what
keeps that from being read as a behavioural signal.

Pure by design - no I/O, no pandas - so every trap here is testable against
records pasted from the real file with no 3.4 GB read in the way.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.esci_images import image_url, is_scrape_error

CORPUS_COLUMNS: tuple[str, ...] = (
    "asin",
    "type",
    "template",
    "title",
    "subtitle",
    "author",
    "description",
    "bullets",
    "image_url",
    "category",
    "attrs_json",
    "info_json",
    "stars",
    "ratings",
    "price",
    "price_multi",
    "bsr_rank",
    "bsr_category",
    "n_reviews",
    "reviews_json",
)


def is_book(record: dict[str, Any]) -> bool:
    """True for ESCI-S book records, which use a different set of key names."""
    return record.get("type") == "book"


def _text(record: dict[str, Any], *keys: str) -> str | None:
    """First non-empty string among `keys`, stripped. None if there is none.

    Takes several keys because products and books spell the same field
    differently and a record carries only one of the two spellings.
    """
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _mapping_json(record: dict[str, Any], *keys: str) -> str | None:
    """First non-empty mapping among `keys`, as a JSON string. None if absent.

    Stored as JSON rather than a Parquet map because the key sets are open -
    463 distinct Best Sellers Rank categories alone - and Plan 5 parses only
    the handful of keys each feature needs.
    """
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict) and value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return None


# "4.7 out of 5 stars". 40 distinct values from 1.0 to 5.0 in the real file.
_STARS = re.compile(r"^([0-9]+(?:\.[0-9]+)?) out of 5 stars$")

# "1,116 ratings", and "1 rating" for the 1.1% with exactly one.
_RATINGS = re.compile(r"^([0-9,]+) ratings?$")

# "$9.99", and "$11.53 $12.99" for the 0.05% carrying several.
_PRICE = re.compile(r"\$([0-9,]+\.[0-9]{2})")

# "#141,895 in Cell Phones & Accessories ( See Top 100 in ... ) #10,494 in ..."
# The '#' is required: the "See Top 100 in <category>" parenthetical repeats
# the category without one, and matching it would report the wrong rank.
_BSR = re.compile(r"#([0-9,]+)\s+in\s+([^#(]+)")


def parse_stars(value: str | None) -> float | None:
    """Mean star rating from "4.7 out of 5 stars"."""
    if not value:
        return None
    match = _STARS.match(value.strip())
    return float(match.group(1)) if match else None


def parse_ratings(value: str | None) -> int | None:
    """Rating count from "1,116 ratings" or "1 rating"."""
    if not value:
        return None
    match = _RATINGS.match(value.strip())
    return int(match.group(1).replace(",", "")) if match else None


def parse_price(value: str | None) -> tuple[float | None, bool]:
    """Price and whether the source string carried more than one.

    Returns the first price and a flag. The multi-price strings are sale or
    variant pairs in no reliable order, so "first" is a convention, not a
    judgement about which price is right - the flag is there so Plan 5 can
    drop these rather than rank on an arbitrary pick.
    """
    if not value:
        return None, False
    found = _PRICE.findall(value)
    if not found:
        return None, False
    return float(found[0].replace(",", "")), len(found) > 1


def parse_best_sellers_rank(
    info: dict[str, Any] | None,
) -> tuple[int | None, str | None]:
    """Overall Best Sellers Rank and its top-level category, from `info`.

    Returns (None, None) when there is no `info` block at all, which is every
    book: the rank is structurally absent for that record type, not missing.
    """
    if not info:
        return None, None
    raw = info.get("Best Sellers Rank")
    if not isinstance(raw, str):
        return None, None
    found = _BSR.findall(raw)
    if not found:
        return None, None
    rank, category = found[0]
    return int(rank.replace(",", "")), category.strip()


def normalise_record(
    record: dict[str, Any], *, with_reviews: bool = False
) -> dict[str, Any]:
    """Flatten one non-error ESCI-S record into the corpus row schema.

    Raises on a scrape-error row: the ETL filters those out beforehand, so
    reaching this function with one is a caller bug, and a silently-empty row
    would be indistinguishable from a real product carrying no metadata.
    """
    if is_scrape_error(record):
        raise ValueError(
            f"{record.get('asin')!r} is a scrape-error row; filter these out "
            "with src.esci_images.is_scrape_error before normalising"
        )

    price, price_multi = parse_price(record.get("price"))
    bsr_rank, bsr_category = parse_best_sellers_rank(record.get("info"))
    reviews = record.get("reviews") or []
    return {
        "asin": record["asin"],
        "type": record.get("type"),
        "template": _text(record, "template"),
        "title": _text(record, "title"),
        "subtitle": _text(record, "subtitle"),
        "author": _text(record, "author"),
        # Books say `desc`, products say `description`.
        "description": _text(record, "description", "desc"),
        "bullets": list(record.get("bullets") or []),
        # image_url() handles the image/img split and strips the retired CDN
        # bucket. Never read record["image"] directly.
        "image_url": image_url(record),
        "category": list(record.get("category") or []),
        # Books say `attr`, products say `attrs`.
        "attrs_json": _mapping_json(record, "attrs", "attr"),
        # Product-only. None here means "this record type never has one".
        "info_json": _mapping_json(record, "info"),
        "stars": parse_stars(record.get("stars")),
        "ratings": parse_ratings(record.get("ratings")),
        "price": price,
        "price_multi": price_multi,
        "bsr_rank": bsr_rank,
        "bsr_category": bsr_category,
        "n_reviews": len(reviews),
        "reviews_json": (
            json.dumps(reviews, ensure_ascii=False) if with_reviews and reviews else None
        ),
    }
