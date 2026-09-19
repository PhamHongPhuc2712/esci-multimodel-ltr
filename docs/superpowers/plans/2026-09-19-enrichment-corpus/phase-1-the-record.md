# Phase 1 — The Record

**Plan 2 of 7 · Phase 1 of 2 · Tasks 1–2.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into.

**Delivers:** `src/esci_s.py` — one ESCI-S JSON record in, one flat typed row
out, with every record shape in the scrape handled exactly.

**Needs on disk:** nothing. No 3.4 GB read in this phase — that is Phase 2.
`src/esci_s.py` deliberately holds no I/O and no pandas, because the traps in
this data are all *structural* (which key name, which record type) and are
testable against records pasted verbatim from the real file.

**Owns Review Focus items 1–3** (books dropped by the product spelling, `info`
structurally absent for books, multi-price strings).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Products store the main image URL in `image`, books in `img`.** Books likewise use `desc`/`attr` where products use `description`/`attrs`.
- **Always go through `image_url()`** from `src/esci_images.py` — it handles the `image`/`img` split *and* strips the retired CDN bucket. Never read `record["image"]` directly.
- **`info` is product-only**, so Best Sellers Rank is structurally absent for books. Measured: 0 of 18,727 `us` books carry it.
- **0.05% of prices are multi-price strings** like `'$11.53 $12.99'`, in no reliable order.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: Record normalisation and book/product field unification

**Files:**
- Create: `src/esci_s.py`
- Create: `tests/test_esci_s.py`

**Interfaces:**
- Consumes: `src.esci_images.image_url(record) -> str | None`, `src.esci_images.is_scrape_error(record) -> bool`. Both already exist; do not reimplement either.
- Produces:
  - `src.esci_s.is_book(record: dict) -> bool`
  - `src.esci_s.normalise_record(record: dict, *, with_reviews: bool = False) -> dict`
  - `src.esci_s.CORPUS_COLUMNS: tuple[str, ...]` — the row keys, in Parquet column order

`normalise_record` raises `ValueError` on a scrape-error row rather than
returning a mostly-empty dict. The ETL filters those out before calling it, so
reaching this function with one is a bug in the caller, and a silently-empty
row would be indistinguishable from a real product with no metadata.

The six parsed behavioural values (`stars`, `ratings`, `price`,
`price_multi`, `bsr_rank`, `bsr_category`) are filled in by Task 2; this task
emits them as `None`/`False` and Task 2 Step 4 replaces those six placeholder
lines with real calls. That ordering is deliberate: it keeps the structural
unification reviewable on its own, and the two tasks fail for different
reasons.

- [ ] **Step 1: Write the failing test**

Create `tests/test_esci_s.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_esci_s.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.esci_s'`.

- [ ] **Step 3: Write `src/esci_s.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_esci_s.py -v
```

Expected: PASS, 19 tests.

- [ ] **Step 5: Commit**

```bash
git add src/esci_s.py tests/test_esci_s.py
git commit -m "Normalise ESCI-S records with book and product field unification"
```

---

## Task 2: Typed behavioural parsers

**Files:**
- Modify: `src/esci_s.py`
- Modify: `tests/test_esci_s.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `src.esci_s.parse_stars(value: str | None) -> float | None`
  - `src.esci_s.parse_ratings(value: str | None) -> int | None`
  - `src.esci_s.parse_price(value: str | None) -> tuple[float | None, bool]` — `(price, was_ambiguous)`
  - `src.esci_s.parse_best_sellers_rank(info: dict | None) -> tuple[int | None, str | None]` — `(rank, top_category)`

All four return `None` rather than raising on an unrecognised shape. A 1.2M-record
scrape will contain shapes nobody enumerated, and one weird price must not abort
a pass over a file that cannot be resumed. The *structural* traps — book key
names, error rows — are handled exactly, not defensively; these four are the
only place tolerance is the right call.

Validated against 500,356 real records before this plan was written: zero parse
failures across 317,105 `stars`, 318,104 `ratings`, 150,714 Best Sellers Ranks
and 94,813 prices.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_esci_s.py`:

```python
from src.esci_s import (
    parse_best_sellers_rank,
    parse_price,
    parse_ratings,
    parse_stars,
)


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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_esci_s.py -v
```

Expected: FAIL — `ImportError: cannot import name 'parse_stars' from 'src.esci_s'`.

- [ ] **Step 3: Add the parsers to `src/esci_s.py`**

Add `import re` to the imports, then add these above `normalise_record`:

```python
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
```

- [ ] **Step 4: Wire the parsers into `normalise_record`**

Replace the six placeholder lines in the returned dict:

```python
        "stars": None,  # filled in by Task 2
        "ratings": None,  # filled in by Task 2
        "price": None,  # filled in by Task 2
        "price_multi": False,  # filled in by Task 2
        "bsr_rank": None,  # filled in by Task 2
        "bsr_category": None,  # filled in by Task 2
```

with the real calls. Add these two lines just above the `return`:

```python
    price, price_multi = parse_price(record.get("price"))
    bsr_rank, bsr_category = parse_best_sellers_rank(record.get("info"))
```

and use them in the dict:

```python
        "stars": parse_stars(record.get("stars")),
        "ratings": parse_ratings(record.get("ratings")),
        "price": price,
        "price_multi": price_multi,
        "bsr_rank": bsr_rank,
        "bsr_category": bsr_category,
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_esci_s.py -v
```

Expected: PASS, 35 tests (19 from Task 1 plus 16 here).

- [ ] **Step 6: Commit**

```bash
git add src/esci_s.py tests/test_esci_s.py
git commit -m "Parse ESCI-S stars, ratings, price and Best Sellers Rank"
```

---

## Phase 1 Gate

Phase 2 does not start until all of these hold:

- [ ] `python -m pytest tests/test_esci_s.py -v` passes — 35 tests, no skips.
- [ ] `python -m pytest` still passes end to end, with Plan 1's suite untouched.
- [ ] `normalise_record` is never called with a scrape-error row anywhere in the codebase; the ETL in Phase 2 filters before calling.

Next: [Phase 2 — The Corpus](phase-2-the-corpus.md).
