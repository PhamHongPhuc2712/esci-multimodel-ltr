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
        "stars": None,  # filled in by Task 2
        "ratings": None,  # filled in by Task 2
        "price": None,  # filled in by Task 2
        "price_multi": False,  # filled in by Task 2
        "bsr_rank": None,  # filled in by Task 2
        "bsr_category": None,  # filled in by Task 2
        "n_reviews": len(reviews),
        "reviews_json": (
            json.dumps(reviews, ensure_ascii=False) if with_reviews and reviews else None
        ),
    }
