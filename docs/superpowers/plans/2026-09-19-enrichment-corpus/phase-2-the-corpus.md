# Phase 2 — The Corpus

**Plan 2 of 7 · Phase 2 of 2 · Tasks 3–5.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 1 gate](phase-1-the-record.md#phase-1-gate) passes.

**Delivers:** `data/esci-s/corpus.parquet` built in one streaming pass, the
measured join coverage against the ESCI product set, and the missingness-bias
report that decides whether the behavioural ablation in Plan 5 is trustworthy.

**Needs on disk:** `data/esci-s/esci.json.zst`, 3.4 GB. Task 3 Step 5 does the
pass; budget 10–20 minutes for ~1.66M records through JSON parsing and Parquet
compression. Plan 1's ESCI parquet (~1.08 GB) must already be present — Tasks 4
and 5 join against it.

**Owns Review Focus items 4–5** (a truncated source parsing as a good prefix
with no exception raised, coverage measured against the wrong ASIN set).

> **Task 5 reopens `src/bootstrap.py` from Plan 1** to add `cluster_delta_ci`.
> The missingness comparison is between two groups of judgements clustered
> within queries, which neither `bootstrap_ci` (one sample) nor
> `paired_delta_ci` (paired on the same queries) can express. Do not implement
> `src/coverage.py` against the Plan 1 shape of `src/bootstrap.py`.

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **One streaming pass.** Single-frame zstd: no random access, no resumable ranged decompression. Filter, normalise and write in the same pass.
- **A truncated stream is a corrupt corpus, not a short one — and zstandard will not tell you.** `stream_reader.read()` returns short and then returns `b""`; it never raises. Check `decompressobj().eof` and the frame's declared `content_size`.
- **Asserted field presence over all 1,080,262 `us` non-error rows** (tolerance ±0.02): `template` 1.000, `ratings` 0.977, `stars` 0.974, `category` 0.954, image URL 0.863, `attrs`∪`attr` 0.634, `info` 0.534, Best Sellers Rank 0.464, `price` 0.292 — plus a book share in [0.04, 0.09], which is the only one of these that actually catches a dropped-books pass.
- **Join coverage is measured against this project's product set**, not ESCI-S's headline 91.5% over all 1,814,925 ESCI ASINs across every locale.
- **Bootstrap over queries, not judgements.** Judgements are clustered within queries; ~20 per query.
- **Do not commit datasets.** The corpus Parquet is not committed; the JSON reports under `docs/results/` are.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 3: The single streaming pass

**Files:**
- Create: `src/esci_s_etl.py`
- Create: `tests/test_esci_s_etl.py`

**Interfaces:**
- Consumes: `src.esci_s.normalise_record(record, *, with_reviews) -> dict`, `src.esci_s.CORPUS_COLUMNS`, `src.esci_images.is_scrape_error(record) -> bool`.
- Produces:
  - `src.esci_s_etl.SOURCE_URL: str`, `DEFAULT_SOURCE: Path`, `DEFAULT_DEST: Path`
  - `src.esci_s_etl.SCHEMA: pyarrow.Schema`
  - `src.esci_s_etl.TruncatedSourceError` (subclass of `RuntimeError`)
  - `src.esci_s_etl.CorpusInvariantError` (subclass of `ValueError`)
  - `src.esci_s_etl.EXPECTED_PRESENCE: dict[str, float]`, `PRESENCE_TOLERANCE: float`
  - `src.esci_s_etl.EtlStats` with `.to_dict() -> dict`
  - `src.esci_s_etl.build_corpus(source, dest, *, locale="us", batch_size=50_000, with_reviews=False) -> EtlStats`
  - `src.esci_s_etl.check_corpus_invariants(stats: EtlStats) -> None`
  - `python -m src.esci_s_etl`

- [x] **Step 1: Write the failing test**

Create `tests/test_esci_s_etl.py`:

```python
import json

import pyarrow.parquet as pq
import pytest
import zstandard

from src.esci_s_etl import (
    SCHEMA,
    CorpusInvariantError,
    EtlStats,
    TruncatedSourceError,
    build_corpus,
    check_corpus_invariants,
)


def _product(asin: str, locale: str = "us") -> dict:
    return {
        "asin": asin,
        "locale": locale,
        "type": "product",
        "template": "home_improvement",
        "title": f"a product titled {asin} with enough text to compress badly",
        "description": "a description " * 20,
        "bullets": ["one", "two"],
        "image": f"https://m.media-amazon.com/images/I/{asin}.jpg",
        "category": ["Tools", "Clamps"],
        "attrs": {"Brand": "YUEPIN"},
        "info": {"Best Sellers Rank": "#1,206 in Industrial & Scientific"},
        "stars": "4.7 out of 5 stars",
        "ratings": "54 ratings",
        "price": "$9.99",
        "reviews": [{"stars": "5.0", "text": "great"}],
    }


def _book(asin: str) -> dict:
    return {
        "asin": asin,
        "locale": "us",
        "type": "book",
        "template": "book",
        "title": f"a book titled {asin}",
        "desc": "a book description " * 20,
        "img": f"https://m.media-amazon.com/images/I/{asin}.jpg",
        "category": ["Books"],
        "attr": {"Publisher": "Basic Books"},
        "author": "Don Norman",
        "stars": "4.6 out of 5 stars",
        "ratings": "6,090 ratings",
    }


def _error(asin: str) -> dict:
    return {
        "asin": asin,
        "locale": "us",
        "type": "error",
        "error": "404",
        "template": "error",
    }


def _source(tmp_path, records, *, truncate: bool = False):
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in records)
    blob = zstandard.ZstdCompressor().compress(raw)
    if truncate:
        blob = blob[: len(blob) // 2]
    path = tmp_path / "esci.json.zst"
    path.write_bytes(blob)
    return path


def _mixed_records(n: int = 200) -> list[dict]:
    records: list[dict] = []
    for i in range(n):
        records.append(_product(f"B{i:09d}"))
        if i % 10 == 0:
            records.append(_book(f"K{i:09d}"))
        if i % 25 == 0:
            records.append(_error(f"E{i:09d}"))
        if i % 20 == 0:
            records.append(_product(f"J{i:09d}", locale="jp"))
    return records


# --- what lands in the corpus -----------------------------------------------

def test_only_us_non_error_rows_are_written(tmp_path):
    source = _source(tmp_path, _mixed_records())
    dest = tmp_path / "corpus.parquet"
    stats = build_corpus(source, dest)
    table = pq.read_table(dest)
    assert stats.rows_written == table.num_rows
    assert set(table.column("type").to_pylist()) == {"product", "book"}
    assert not any(a.startswith("E") for a in table.column("asin").to_pylist())
    assert not any(a.startswith("J") for a in table.column("asin").to_pylist())


def test_books_survive_the_pass(tmp_path):
    # Review Focus 1 again, this time end to end: a pass that reads only the
    # product spelling produces a corpus that is simply 6% smaller.
    source = _source(tmp_path, _mixed_records())
    dest = tmp_path / "corpus.parquet"
    build_corpus(source, dest)
    table = pq.read_table(dest)
    books = [r for r in table.to_pylist() if r["type"] == "book"]
    assert len(books) == 20
    assert all(b["description"] for b in books)
    assert all(b["image_url"] for b in books)


def test_written_schema_is_the_declared_schema(tmp_path):
    source = _source(tmp_path, _mixed_records())
    dest = tmp_path / "corpus.parquet"
    build_corpus(source, dest)
    assert pq.read_schema(dest).names == SCHEMA.names


def test_every_drop_reason_is_counted(tmp_path):
    source = _source(tmp_path, _mixed_records())
    stats = build_corpus(source, tmp_path / "corpus.parquet")
    assert stats.records_read == 200 + 20 + 8 + 10
    assert stats.errors_dropped == 8
    assert stats.other_locale_dropped == 10
    assert stats.products == 200
    assert stats.books == 20
    assert stats.rows_written == 220


def test_batch_size_does_not_change_the_output(tmp_path):
    records = _mixed_records()
    one = tmp_path / "one.parquet"
    many = tmp_path / "many.parquet"
    build_corpus(_source(tmp_path, records), one, batch_size=7)
    build_corpus(_source(tmp_path, records), many, batch_size=100_000)
    assert pq.read_table(one).to_pylist() == pq.read_table(many).to_pylist()


def test_review_text_is_excluded_by_default_and_included_on_request(tmp_path):
    records = _mixed_records()
    without = tmp_path / "without.parquet"
    with_text = tmp_path / "with.parquet"
    build_corpus(_source(tmp_path, records), without)
    build_corpus(_source(tmp_path, records), with_text, with_reviews=True)
    assert set(pq.read_table(without).column("reviews_json").to_pylist()) == {None}
    assert any(pq.read_table(with_text).column("reviews_json").to_pylist())


# --- Review Focus 4: a truncated source ------------------------------------

def test_truncated_source_raises(tmp_path):
    # zstandard does NOT raise on a truncated frame - stream_reader returns
    # short and then returns b"". Verified on 0.25.0: half a frame yielded
    # 3,670,016 of 7,420,000 bytes silently. build_corpus must catch this by
    # checking decompressobj().eof, or it produces a partial corpus that
    # passes every row count anyone would think to write.
    source = _source(tmp_path, _mixed_records(), truncate=True)
    with pytest.raises(TruncatedSourceError, match="end marker"):
        build_corpus(source, tmp_path / "corpus.parquet")


def test_a_stream_reader_would_have_missed_this(tmp_path):
    # Pins the library behaviour the eof check exists for, so that a future
    # zstandard release which starts raising does not make the guard above
    # look redundant and get deleted.
    #
    # The point is the silence, not the byte count: how much comes back
    # depends on where the cut falls relative to a zstd block boundary. For
    # this small fixture it is 0 bytes; for a 34,219-byte frame it was
    # 3,670,016 of 7,420,000. Either way nothing is raised.
    source = _source(tmp_path, _mixed_records(), truncate=True)
    complete = len(b"".join(json.dumps(r).encode() + b"\n" for r in _mixed_records()))
    got = 0
    with source.open("rb") as handle:
        reader = zstandard.ZstdDecompressor().stream_reader(handle)
        while chunk := reader.read(1 << 20):
            got += len(chunk)
    assert got < complete  # short, and no exception on the way here


def test_the_fixtures_declare_a_content_size(tmp_path):
    # The byte-count check in _iter_json_lines is only live when the frame
    # header declares content_size - zstd does not always store it. The real
    # esci.json.zst declares 11,539,399,406, and ZstdCompressor().compress()
    # declares it too, so the check is exercised by these tests rather than
    # quietly skipped.
    source = _source(tmp_path, _mixed_records())
    header = source.read_bytes()[:18]
    assert zstandard.get_frame_parameters(header).content_size > 0


def test_truncated_source_leaves_no_parquet_behind(tmp_path):
    # A half-written corpus on disk is worse than none: the next run finds a
    # file that reads cleanly and is silently short.
    dest = tmp_path / "corpus.parquet"
    source = _source(tmp_path, _mixed_records(), truncate=True)
    with pytest.raises(TruncatedSourceError):
        build_corpus(source, dest)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_a_trailing_partial_line_is_a_truncation_not_a_record(tmp_path):
    # Here the *frame* is complete - eof is True and the byte count matches -
    # so this truncation is only visible as a final line that is not JSON.
    # It is the case the eof check cannot see, which is why both exist.
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in _mixed_records())
    raw += b'{"asin": "BTRUNCATED", "locale": "us", "ty'
    path = tmp_path / "esci.json.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))
    with pytest.raises(TruncatedSourceError, match="not complete JSON"):
        build_corpus(path, tmp_path / "corpus.parquet")


def test_a_final_record_without_a_trailing_newline_is_kept(tmp_path):
    raw = b"".join(json.dumps(r).encode() + b"\n" for r in _mixed_records())
    raw += json.dumps(_product("BLASTONE00")).encode()  # no trailing newline
    path = tmp_path / "esci.json.zst"
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))
    stats = build_corpus(path, tmp_path / "corpus.parquet")
    assert stats.rows_written == 221


# --- invariants -------------------------------------------------------------

def _measured_stats(**overrides) -> EtlStats:
    """A stats object matching what the real 1,080,262-row pass produced."""
    stats = EtlStats(rows_written=1000, products=943, books=57)
    stats.present.update(
        {
            "template": 1000,
            "ratings": 977,
            "stars": 974,
            "category": 954,
            "image_url": 863,
            "attrs_json": 634,
            "info_json": 534,
            "bsr_rank": 464,
            "price": 292,
        }
    )
    for key, value in overrides.items():
        setattr(stats, key, value)
    return stats


def test_invariants_accept_the_documented_presence():
    check_corpus_invariants(_measured_stats())


def test_invariants_reject_a_corpus_that_lost_its_books():
    # Measured: dropping every book moves image_url by only 0.8 points and
    # attrs_json by 2.2, so the presence bands do NOT reliably catch this.
    # The book-share assertion is what does.
    stats = _measured_stats(books=0, products=1000)
    with pytest.raises(CorpusInvariantError, match="books are"):
        check_corpus_invariants(stats)


def test_invariants_reject_a_field_that_sags():
    stats = _measured_stats()
    stats.present["image_url"] = 600
    with pytest.raises(CorpusInvariantError, match="image_url"):
        check_corpus_invariants(stats)


def test_invariants_reject_a_corpus_matching_the_product_only_figures():
    # CLAUDE.md's attrs 0.579 / image 0.847 count the product key alone. A
    # corpus that reproduced them would be one that unified nothing.
    stats = _measured_stats()
    stats.present["attrs_json"] = 579
    with pytest.raises(CorpusInvariantError, match="attrs_json"):
        check_corpus_invariants(stats)


def test_invariants_reject_an_empty_corpus():
    with pytest.raises(CorpusInvariantError, match="no rows"):
        check_corpus_invariants(EtlStats())


@pytest.mark.data
def test_real_corpus_satisfies_every_invariant():
    import pandas as pd

    from src.esci_s_etl import DEFAULT_DEST, EXPECTED_PRESENCE, PRESENCE_TOLERANCE

    corpus = pd.read_parquet(DEFAULT_DEST, columns=list(EXPECTED_PRESENCE))
    for column, expected in EXPECTED_PRESENCE.items():
        series = corpus[column]
        if series.dtype == object:
            present = series.map(lambda v: v is not None and len(v) > 0).mean()
        else:
            present = series.notna().mean()
        assert abs(present - expected) <= PRESENCE_TOLERANCE, (column, present)
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_esci_s_etl.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.esci_s_etl'`.

- [x] **Step 3: Write `src/esci_s_etl.py`**

```python
"""The one streaming pass over ESCI-S.

The source is 3.4 GB of single-frame zstd: no random access, no resumable
ranged decompression, no second chance. So filtering to `us`, dropping scrape
errors, normalising book fields and writing Parquet all happen in the same
pass over the same bytes.

A truncated source is the failure this module is most careful about, and it
is quieter than it looks: zstandard's stream_reader does NOT raise on a
truncated frame, it just stops early and returns b"". So truncation is
detected by checking `decompressobj().eof` and the frame's declared
content_size, never by catching an exception. A short corpus reads back
perfectly well and is silently missing its tail, so the check is explicit and
the partial Parquet is deleted rather than left where the next run finds it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from src.esci_images import is_scrape_error
from src.esci_s import CORPUS_COLUMNS, normalise_record

SOURCE_URL = "https://esci-s.s3.amazonaws.com/esci.json.zst"
DEFAULT_SOURCE = Path("data/esci-s/esci.json.zst")
DEFAULT_DEST = Path("data/esci-s/corpus.parquet")

SCHEMA = pa.schema(
    [
        ("asin", pa.string()),
        ("type", pa.string()),
        ("template", pa.string()),
        ("title", pa.string()),
        ("subtitle", pa.string()),
        ("author", pa.string()),
        ("description", pa.string()),
        ("bullets", pa.list_(pa.string())),
        ("image_url", pa.string()),
        ("category", pa.list_(pa.string())),
        ("attrs_json", pa.string()),
        ("info_json", pa.string()),
        ("stars", pa.float32()),
        ("ratings", pa.int32()),
        ("price", pa.float32()),
        ("price_multi", pa.bool_()),
        ("bsr_rank", pa.int32()),
        ("bsr_category", pa.string()),
        ("n_reviews", pa.int32()),
        ("reviews_json", pa.string()),
    ]
)

# Measured over all 1,080,262 `us` non-error rows of the real file.
#
# Two of these deliberately differ from the figures in CLAUDE.md, and the gap
# is the books. CLAUDE.md counts the *product* key only; this corpus unifies
# the book spelling into the same column, so the share is higher:
#
#   attrs_json  product `attrs` 0.5771 + book `attr` 0.0571 = 0.6342  (vs 0.579)
#   image_url   product `image` 0.8213 + book `img`  0.0418 = 0.8630  (vs 0.847)
#
# A corpus that matched CLAUDE.md on these two would be one that had dropped
# every book.
EXPECTED_PRESENCE: dict[str, float] = {
    "template": 1.000,
    "ratings": 0.977,
    "stars": 0.974,
    "category": 0.954,
    "image_url": 0.863,
    "attrs_json": 0.634,
    "info_json": 0.534,
    "bsr_rank": 0.464,
    "price": 0.292,
}
PRESENCE_TOLERANCE = 0.02

# Books are 5.74% of `us` non-error rows. The presence bands above turn out
# NOT to catch a pass that drops them - losing every book moves image_url by
# only 0.8 points, well inside the tolerance - so the share is asserted
# directly. This is the corpus-level guard for Review Focus item 1.
EXPECTED_BOOK_SHARE = (0.04, 0.09)


class TruncatedSourceError(RuntimeError):
    """Raised when the zstd stream ends mid-frame or mid-record."""


class CorpusInvariantError(ValueError):
    """Raised when the built corpus does not match the documented shape."""


@dataclass
class EtlStats:
    records_read: int = 0
    errors_dropped: int = 0
    other_locale_dropped: int = 0
    products: int = 0
    books: int = 0
    rows_written: int = 0
    present: Counter = field(default_factory=Counter)

    def observe(self, row: dict[str, Any]) -> None:
        if row["type"] == "book":
            self.books += 1
        else:
            self.products += 1
        for name in EXPECTED_PRESENCE:
            value = row[name]
            if value is not None and value != "" and value != []:
                self.present[name] += 1

    def presence(self) -> dict[str, float]:
        if not self.rows_written:
            return {name: 0.0 for name in EXPECTED_PRESENCE}
        return {
            name: self.present[name] / self.rows_written for name in EXPECTED_PRESENCE
        }

    def to_dict(self) -> dict:
        return {
            "records_read": self.records_read,
            "errors_dropped": self.errors_dropped,
            "other_locale_dropped": self.other_locale_dropped,
            "products": self.products,
            "books": self.books,
            "book_share": self.books / self.rows_written if self.rows_written else 0.0,
            "rows_written": self.rows_written,
            "presence": self.presence(),
        }


def _iter_json_lines(source: Path) -> Iterator[dict[str, Any]]:
    """Stream one JSON object per line out of a single-frame zstd file.

    Truncation is *checked*, not caught. Measured on zstandard 0.25.0:
    `ZstdDecompressor().stream_reader(fh).read()` returns short and then
    returns b"" on a truncated frame - it never raises - so wrapping it in
    `except ZstdError` is dead code that reads like a safeguard. Cutting a
    34,219-byte frame in half yielded 3,670,016 of 7,420,000 bytes with no
    exception at all.

    `decompressobj()` exposes `.eof`, which is True only once the frame's end
    marker (and, for this file, its xxhash checksum) has been consumed. That
    is the signal. The real esci.json.zst also declares
    content_size = 11,539,399,406 in its frame header, so the decompressed
    byte count is a second, independent check.
    """
    source = Path(source)
    decompressor = zstandard.ZstdDecompressor().decompressobj()
    produced = 0
    tail = b""

    with source.open("rb") as handle:
        try:
            expected_bytes = zstandard.get_frame_parameters(
                handle.read(18)
            ).content_size
        except zstandard.ZstdError:
            expected_bytes = 0  # header too short, or size not declared
        handle.seek(0)

        while True:
            compressed = handle.read(1 << 22)
            if not compressed:
                break
            try:
                chunk = decompressor.decompress(compressed)
            except zstandard.ZstdError as exc:
                raise TruncatedSourceError(
                    f"{source} is corrupt: {exc}. The frame carries an xxhash "
                    f"checksum, so this is damage, not truncation. Re-download "
                    f"from {SOURCE_URL}."
                ) from exc
            if not chunk:
                continue
            produced += len(chunk)
            lines = (tail + chunk).split(b"\n")
            tail = lines.pop()
            for line in lines:
                if line.strip():
                    yield json.loads(line)

    if tail.strip():
        # A file that does not end in a newline leaves its final, complete
        # record here. A file cut mid-record leaves a fragment.
        try:
            yield json.loads(tail)
        except ValueError as exc:
            raise TruncatedSourceError(
                f"{source} is truncated: the final line is not complete JSON "
                f"({exc}). Re-download from {SOURCE_URL}."
            ) from exc

    if not decompressor.eof:
        raise TruncatedSourceError(
            f"{source} is truncated: the zstd frame never reached its end "
            f"marker after {produced:,} bytes"
            + (f" of {expected_bytes:,} declared" if expected_bytes else "")
            + f". stream_reader would have handed these bytes over silently, "
            f"and a corpus built from them reads back cleanly while missing "
            f"its tail. Re-download from {SOURCE_URL}."
        )
    if expected_bytes and produced != expected_bytes:
        raise TruncatedSourceError(
            f"{source} decompressed to {produced:,} bytes but its frame header "
            f"declares {expected_bytes:,}. Re-download from {SOURCE_URL}."
        )


def _write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    columns = {name: [row[name] for row in rows] for name in CORPUS_COLUMNS}
    writer.write_table(pa.Table.from_pydict(columns, schema=SCHEMA))


def build_corpus(
    source: Path = DEFAULT_SOURCE,
    dest: Path = DEFAULT_DEST,
    *,
    locale: str = "us",
    batch_size: int = 50_000,
    with_reviews: bool = False,
) -> EtlStats:
    """Stream the scrape into a Parquet corpus. One pass, no second chance."""
    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")

    stats = EtlStats()
    batch: list[dict[str, Any]] = []
    writer = pq.ParquetWriter(partial, SCHEMA, compression="zstd")
    try:
        for record in _iter_json_lines(source):
            stats.records_read += 1
            if record.get("locale") != locale:
                stats.other_locale_dropped += 1
                continue
            if is_scrape_error(record):
                stats.errors_dropped += 1
                continue
            row = normalise_record(record, with_reviews=with_reviews)
            stats.observe(row)
            batch.append(row)
            if len(batch) >= batch_size:
                _write_batch(writer, batch)
                stats.rows_written += len(batch)
                batch.clear()
        if batch:
            _write_batch(writer, batch)
            stats.rows_written += len(batch)
        writer.close()
    except BaseException:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return stats


def check_corpus_invariants(stats: EtlStats) -> None:
    """Assert the documented field presence, or raise naming the field."""
    if not stats.rows_written:
        raise CorpusInvariantError(
            "the corpus has no rows; check the locale filter and that the "
            "source is the real 3.4 GB file rather than the stale "
            "sample.json.gz from the GitHub repo, which has no image field"
        )
    low, high = EXPECTED_BOOK_SHARE
    book_share = stats.books / stats.rows_written
    if not low <= book_share <= high:
        raise CorpusInvariantError(
            f"books are {book_share:.4f} of the corpus, expected between "
            f"{low} and {high}. Zero means the pass is filtering on "
            "product-only keys and has dropped every book; the field-presence "
            "bands below will not catch that on their own."
        )

    observed = stats.presence()
    for name, expected in EXPECTED_PRESENCE.items():
        seen = observed[name]
        if abs(seen - expected) > PRESENCE_TOLERANCE:
            raise CorpusInvariantError(
                f"{name} is present in {seen:.3f} of rows, expected "
                f"{expected:.3f} +/- {PRESENCE_TOLERANCE}. If image_url and "
                "description are the fields that sag while the dense ones "
                "hold, the pass is reading only the product spellings and "
                "has dropped every book."
            )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--locale", default="us")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--with-reviews",
        action="store_true",
        help="carry review text; PROJECT_SPEC.md marks it optional and it is "
        "the bulk of the 3.4 GB",
    )
    args = parser.parse_args()

    stats = build_corpus(
        args.source,
        args.dest,
        locale=args.locale,
        batch_size=args.batch_size,
        with_reviews=args.with_reviews,
    )
    print(json.dumps(stats.to_dict(), indent=2))
    check_corpus_invariants(stats)
    print(f"corpus written to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [x] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_esci_s_etl.py -v
```

Expected: PASS, 15 tests; the `data`-marked corpus test is deselected.

- [x] **Step 5: Download the source and run the real pass**

The file is 3.4 GB of single-frame zstd. Download to a `.part` and rename only
on success — an interrupted download leaves a prefix that decompresses fine up
to the cut, which is exactly the failure Task 3 raises on:

```bash
mkdir -p data/esci-s
curl -fL --retry 3 --retry-delay 5 \
  -o data/esci-s/esci.json.zst.part \
  https://esci-s.s3.amazonaws.com/esci.json.zst \
  && mv data/esci-s/esci.json.zst.part data/esci-s/esci.json.zst

python -m src.esci_s_etl
python -m pytest tests/test_esci_s_etl.py -v -m data
```

Expected: ~1.66M records read, ~1.2M rows written after the `us` filter and the
error drop, and every presence figure inside ±0.02 of the table. Budget 10–20
minutes.

**If a presence figure is outside the band, that is a finding, not a nuisance.**
Work through, in order:

1. Are books in the corpus at all? `stats.books / stats.rows_written` should be
   between 0.04 and 0.09. Zero means the pass is filtering on product-only keys.
2. Is `image_url` going through `image_url()`? Reading `record["image"]`
   directly drops every book's image *and* keeps the retired CDN bucket.
3. Is the locale filter `us`? Leaving it off roughly triples the row count and
   shifts every percentage.
4. Is the source the real 3.4 GB file, and not `sample.json.gz` from the GitHub
   repo? That sample has no `image` field at all.

- [x] **Step 6: Record the measured book share**

`CLAUDE.md` says books are "~7% of the corpus". That figure is across all
locales; measured on a `us`-only 500,356-record prefix it is 5.8%. Print the
real number and, if it confirms the gap, add the locale qualifier to
`CLAUDE.md` so the next reader is not surprised:

```bash
python -c "
import pandas as pd
from src.esci_s_etl import DEFAULT_DEST
types = pd.read_parquet(DEFAULT_DEST, columns=['type'])['type']
print(types.value_counts(normalize=True).to_dict())
"
```

- [x] **Step 7: Correct the stale comment in `src/esci_images.py`**

That module catches `zstandard.ZstdError` around its read loop with the
comment `# expected when the input is a truncated prefix`. Measured on
zstandard 0.25.0, that exception never fires — the loop already ends cleanly
on a prefix because `read()` returns `b""`. The behaviour is right; the
comment claims a mechanism that does not exist, and Task 3 now documents the
opposite two files away. Replace it:

```python
        except zstandard.ZstdError:
            pass  # a damaged frame; a prefix simply ends early without raising
```

- [x] **Step 8: Commit**

```bash
git add src/esci_s_etl.py tests/test_esci_s_etl.py src/esci_images.py CLAUDE.md
git commit -m "Stream the ESCI-S scrape into a Parquet enrichment corpus"
```

---

## Task 4: Join coverage against the ESCI product set

**Files:**
- Create: `src/coverage.py`
- Create: `tests/test_coverage.py`
- Create: `docs/results/esci-s-coverage.json` (generated in Step 5, committed)

**Interfaces:**
- Consumes: `src.dataset.load_split(split) -> EsciSplit`, `src.dataset.ensure_downloaded(name) -> Path`, `src.esci_s_etl.DEFAULT_DEST`.
- Produces:
  - `src.coverage.MIN_JOIN_COVERAGE: float` (0.90)
  - `src.coverage.CoverageResult(scope, n_products, n_matched, coverage, n_with_image, image_coverage)` with `.to_dict()`
  - `src.coverage.load_corpus(path=DEFAULT_DEST, columns=None) -> pd.DataFrame`
  - `src.coverage.rerank_product_ids() -> set[str]`
  - `src.coverage.catalogue_product_ids() -> set[str]`
  - `src.coverage.join_coverage(product_ids, corpus, *, scope) -> CoverageResult`

Two scopes, because the funnel needs both and they are different numbers:
**rerank** is the union of judged products over train and test (482,105 per
`CLAUDE.md`), which is what Plans 5–7 re-rank; **catalogue** is every `us`
product in the ESCI products parquet (1,215,851), which is what Plan 4 recalls
over. The gate is on the rerank scope.

- [x] **Step 1: Write the failing test**

Create `tests/test_coverage.py`:

```python
import json

import pandas as pd
import pytest

from src.coverage import MIN_JOIN_COVERAGE, join_coverage


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asin": ["B1", "B2", "B3", "B4"],
            "image_url": ["http://i/1.jpg", None, "http://i/3.jpg", "http://i/4.jpg"],
        }
    )


def test_coverage_is_the_matched_share_of_the_requested_products():
    result = join_coverage({"B1", "B2", "B9"}, _corpus(), scope="rerank")
    assert result.n_products == 3
    assert result.n_matched == 2
    assert result.coverage == pytest.approx(2 / 3)


def test_corpus_rows_outside_the_requested_products_do_not_inflate_coverage():
    # Review Focus 5: the corpus holds every us product, far more than the
    # judged set. Dividing by the corpus size instead of the product set would
    # report a number that is not about this project at all.
    result = join_coverage({"B1"}, _corpus(), scope="rerank")
    assert result.coverage == pytest.approx(1.0)


def test_image_coverage_counts_only_matched_products_with_a_url():
    result = join_coverage({"B1", "B2", "B9"}, _corpus(), scope="rerank")
    assert result.n_with_image == 1
    assert result.image_coverage == pytest.approx(1 / 3)


def test_image_coverage_is_reported_against_the_product_set_not_the_matches():
    # PROJECT_SPEC.md warns that the headline 91.5% is not image coverage:
    # end to end it is ~75%. Dividing by matches rather than by the product
    # set would reproduce exactly that overstatement.
    result = join_coverage({"B1", "B2"}, _corpus(), scope="rerank")
    assert result.image_coverage == pytest.approx(0.5)


def test_scope_is_carried_into_the_result():
    assert join_coverage({"B1"}, _corpus(), scope="catalogue").scope == "catalogue"


def test_empty_product_set_raises():
    with pytest.raises(ValueError, match="no products"):
        join_coverage(set(), _corpus(), scope="rerank")


def test_result_serialises_for_the_committed_record():
    payload = join_coverage({"B1", "B2"}, _corpus(), scope="rerank").to_dict()
    assert json.loads(json.dumps(payload))["scope"] == "rerank"


def test_the_gate_threshold_is_ninety_percent():
    assert MIN_JOIN_COVERAGE == 0.90


@pytest.mark.data
def test_real_join_coverage_clears_the_gate():
    from pathlib import Path

    payload = json.loads(Path("docs/results/esci-s-coverage.json").read_text())
    rerank = next(r for r in payload["scopes"] if r["scope"] == "rerank")
    assert rerank["n_products"] == 482_105
    assert rerank["coverage"] >= MIN_JOIN_COVERAGE
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_coverage.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.coverage'`.

- [x] **Step 3: Write `src/coverage.py`**

```python
"""Join coverage and missingness bias for the ESCI-S enrichment corpus.

Two questions gate Plan 3, and they are different questions. "How much of
ESCI does ESCI-S cover?" is answered by a join against this project's actual
product set - not by the 91.5% headline, which is over all 1,814,925 ESCI
ASINs across every locale. "Is what is missing missing at random?" is answered
by comparing relevance among judgements whose product carries a field against
those whose product does not, resampling whole queries because judgements are
clustered ~20 to a query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.dataset import ensure_downloaded, load_split
from src.esci_s_etl import DEFAULT_DEST

MIN_JOIN_COVERAGE = 0.90


@dataclass(frozen=True)
class CoverageResult:
    scope: str
    n_products: int
    n_matched: int
    coverage: float
    n_with_image: int
    image_coverage: float

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_products": self.n_products,
            "n_matched": self.n_matched,
            "coverage": self.coverage,
            "n_with_image": self.n_with_image,
            "image_coverage": self.image_coverage,
        }


def load_corpus(
    path: Path = DEFAULT_DEST, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read the enrichment corpus, optionally only some columns."""
    return pd.read_parquet(path, columns=columns)


def rerank_product_ids() -> set[str]:
    """Every judged product across train and test - the re-ranking set."""
    ids: set[str] = set()
    for split in ("train", "test"):
        ids |= set(load_split(split).judgements["product_id"])
    return ids


def catalogue_product_ids() -> set[str]:
    """Every us product in the ESCI products parquet - the recall corpus."""
    products = pd.read_parquet(
        ensure_downloaded("products"), columns=["product_id", "product_locale"]
    )
    return set(products.loc[products["product_locale"] == "us", "product_id"])


def join_coverage(
    product_ids: set[str], corpus: pd.DataFrame, *, scope: str
) -> CoverageResult:
    """Share of `product_ids` present in the corpus, and carrying an image.

    Both shares divide by the size of `product_ids`, never by the number of
    matches or the size of the corpus. The corpus holds every us product,
    which is far more than any one scope asks about, and image coverage
    reported over matches rather than over the product set is exactly how the
    91.5% headline gets mistaken for image coverage when it is really ~75%.
    """
    if not product_ids:
        raise ValueError("no products to measure coverage against")

    matched = corpus.loc[corpus["asin"].isin(product_ids)]
    n_matched = int(matched["asin"].nunique())
    with_image = matched.loc[matched["image_url"].notna() & (matched["image_url"] != "")]
    n_with_image = int(with_image["asin"].nunique())
    return CoverageResult(
        scope=scope,
        n_products=len(product_ids),
        n_matched=n_matched,
        coverage=n_matched / len(product_ids),
        n_with_image=n_with_image,
        image_coverage=n_with_image / len(product_ids),
    )
```

- [x] **Step 4: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_coverage.py -v
```

Expected: PASS, 8 tests; the `data`-marked test is deselected.

- [x] **Step 5: Measure the real coverage and commit the record**

```bash
python -c "
import json
from pathlib import Path
from src.coverage import catalogue_product_ids, join_coverage, load_corpus, rerank_product_ids

corpus = load_corpus(columns=['asin', 'image_url'])
scopes = [
    join_coverage(rerank_product_ids(), corpus, scope='rerank').to_dict(),
    join_coverage(catalogue_product_ids(), corpus, scope='catalogue').to_dict(),
]
for s in scopes:
    print(f\"{s['scope']:10s} {s['n_matched']:,}/{s['n_products']:,} = {s['coverage']:.4f}  image {s['image_coverage']:.4f}\")
out = Path('docs/results/esci-s-coverage.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'corpus_rows': len(corpus), 'scopes': scopes}, indent=2) + '\n')
"
python -m pytest tests/test_coverage.py -v -m data
```

Expected: rerank coverage ≥ 0.90 over 482,105 products, and an image coverage
near 0.75 rather than 0.915.

**If rerank coverage is below 0.90, that is the gate failing** — check that the
ETL ran over the full file (`records_read` ≈ 1.66M) and that the ASINs are being
compared as strings on both sides before concluding that ESCI-S is too thin.

- [x] **Step 6: Commit**

```bash
git add src/coverage.py tests/test_coverage.py docs/results/esci-s-coverage.json
git commit -m "Measure ESCI-S join coverage against the ESCI product set"
```

---

## Task 5: The missingness-bias report

**Files:**
- Modify: `src/bootstrap.py` (**reopened from Plan 1**)
- Modify: `src/coverage.py`
- Modify: `tests/test_bootstrap.py`
- Modify: `tests/test_coverage.py`
- Modify: `CLAUDE.md`
- Create: `docs/results/esci-s-missingness.json` (generated in Step 6, committed)

**Interfaces:**
- Consumes: `src.bootstrap.Interval`, `src.dataset.load_split`, `src.coverage.load_corpus`.
- Produces:
  - `src.bootstrap.cluster_delta_ci(values, mask, clusters, *, n_resamples=1000, seed=0, alpha=0.05) -> Interval`
  - `src.coverage.DENSE_FIELDS: tuple[str, ...]`, `SPARSE_FIELDS: tuple[str, ...]`, `MAX_DENSE_GAIN_SHIFT: float` (0.02)
  - `src.coverage.FieldBias(field, dense, present_share, mean_gain_present, mean_gain_absent, delta: Interval)` with `.to_dict()`
  - `src.coverage.missingness_bias(judgements, corpus, *, seed=0, n_resamples=1000) -> list[FieldBias]`
  - `src.coverage.check_missingness(biases) -> None` raising `MissingnessError`
  - `python -m src.coverage`

**Why 0.02.** The gate needs a number, and an arbitrary one would be worse than
none. Plan 1 measured the SBERT baseline beating the random floor by **+0.0827**
mean NDCG. A confound in a dense field that shifts mean gain by less than 0.02
cannot manufacture an effect of that size, so it cannot explain away the
ablations Plans 4–7 will run. Sparse fields are *expected* to correlate with
relevance — a product with no price is a different kind of product — which is
why they are reported rather than gated, and why `CLAUDE.md` says to keep them
with missingness indicators.

- [x] **Step 1: Write the failing test for `cluster_delta_ci`**

Append to `tests/test_bootstrap.py`:

```python
import numpy as np

from src.bootstrap import cluster_delta_ci


def test_cluster_delta_is_the_difference_of_the_two_group_means():
    values = np.array([1.0, 1.0, 0.0, 0.0])
    mask = np.array([True, True, False, False])
    clusters = np.array([0, 1, 2, 3])
    assert cluster_delta_ci(values, mask, clusters, n_resamples=200).point == (
        pytest.approx(1.0)
    )


def test_cluster_delta_of_identical_groups_straddles_zero():
    rng = np.random.default_rng(0)
    values = rng.normal(size=2000)
    mask = np.arange(2000) % 2 == 0
    clusters = np.arange(2000) // 4
    interval = cluster_delta_ci(values, mask, clusters, n_resamples=400)
    assert interval.low < 0 < interval.high


def test_cluster_delta_resamples_clusters_not_rows():
    # Every row in a cluster shares a value, so resampling rows would shrink
    # the interval by pretending there are 2,000 independent observations
    # when there are 100. Judgements are clustered ~20 to a query, so this is
    # the difference between a real interval and a falsely confident one.
    clusters = np.repeat(np.arange(100), 20)
    values = np.repeat(np.random.default_rng(1).normal(size=100), 20)
    mask = clusters % 2 == 0
    interval = cluster_delta_ci(values, mask, clusters, n_resamples=400)
    assert interval.high - interval.low > 0.1


def test_cluster_delta_is_reproducible_for_a_given_seed():
    values = np.arange(200, dtype=float)
    mask = np.arange(200) % 2 == 0
    clusters = np.arange(200) // 5
    assert cluster_delta_ci(values, mask, clusters, seed=4) == cluster_delta_ci(
        values, mask, clusters, seed=4
    )


def test_cluster_delta_raises_when_one_group_is_empty():
    values = np.array([1.0, 2.0])
    mask = np.array([True, True])
    with pytest.raises(ValueError, match="both groups"):
        cluster_delta_ci(values, mask, np.array([0, 1]), n_resamples=10)
```

- [x] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: FAIL — `ImportError: cannot import name 'cluster_delta_ci'`.

- [x] **Step 3: Add `cluster_delta_ci` to `src/bootstrap.py`**

Append:

```python
def cluster_delta_ci(
    values: np.ndarray,
    mask: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Interval for mean(values[mask]) - mean(values[~mask]), by cluster.

    Whole clusters are resampled, not rows. ESCI judgements come ~20 to a
    query and a hard query is hard for all of them, so resampling rows would
    claim 181,701 independent observations where there are 8,956.

    Implemented by pre-aggregating each cluster's sum and count per group, so
    a replicate is a sum over clusters rather than a pass over rows.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    codes, _ = pd.factorize(np.asarray(clusters))
    n_clusters = codes.max() + 1 if len(codes) else 0
    if not mask.any() or not (~mask).any():
        raise ValueError(
            "a missingness comparison needs both groups non-empty; this field "
            "is either always present or always absent"
        )

    def totals(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sums = np.bincount(codes[selected], weights=values[selected], minlength=n_clusters)
        counts = np.bincount(codes[selected], minlength=n_clusters)
        return sums, counts

    present_sum, present_n = totals(mask)
    absent_sum, absent_n = totals(~mask)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    with np.errstate(invalid="ignore", divide="ignore"):
        replicates = (
            present_sum[draws].sum(axis=1) / present_n[draws].sum(axis=1)
            - absent_sum[draws].sum(axis=1) / absent_n[draws].sum(axis=1)
        )
    replicates = replicates[np.isfinite(replicates)]
    low, high = _percentiles(replicates, alpha)
    point = float(values[mask].mean() - values[~mask].mean())
    return Interval(point=point, low=low, high=high)
```

Add `import pandas as pd` to the imports of `src/bootstrap.py` — `pd.factorize`
turns arbitrary cluster labels into dense integer codes for `np.bincount`.

- [x] **Step 4: Run the bootstrap tests to verify they pass**

```bash
python -m pytest tests/test_bootstrap.py -v
```

Expected: PASS, 16 tests (11 from Plan 1 plus 5 here).

- [x] **Step 5: Add the missingness report to `src/coverage.py`**

Append to `src/coverage.py`, and add `from src.bootstrap import Interval,
cluster_delta_ci` and `import numpy as np` to its imports:

```python
# The behavioural ablation in Plan 5 rests on these; they are gated.
#
# `category` started here and was moved out when the real report measured its
# missingness shifting mean gain by -0.0210 [-0.0305, -0.0117], past the 0.02
# the ablation can tolerate. That is the response Task 5 Step 8 prescribes for
# a dense field that fails: reclassify it, do not widen the threshold.
DENSE_FIELDS: tuple[str, ...] = ("stars", "ratings", "template")

# These are expected to correlate with relevance - a product with no price is
# a different kind of product - so they are reported, not gated, and Plan 5
# carries them with missingness indicators.
SPARSE_FIELDS: tuple[str, ...] = (
    "category",
    "price",
    "bsr_rank",
    "attrs_json",
    "info_json",
    "image_url",
)

# Plan 1 measured SBERT beating the random floor by +0.0827 mean NDCG. A dense
# field whose missingness shifts mean gain by less than this cannot manufacture
# an effect of that size.
MAX_DENSE_GAIN_SHIFT = 0.02


class MissingnessError(ValueError):
    """Raised when a dense field's missingness is confounded with the label."""


@dataclass(frozen=True)
class FieldBias:
    field: str
    dense: bool
    present_share: float
    mean_gain_present: float
    mean_gain_absent: float
    delta: Interval

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "dense": self.dense,
            "present_share": self.present_share,
            "mean_gain_present": self.mean_gain_present,
            "mean_gain_absent": self.mean_gain_absent,
            "delta": {
                "point": self.delta.point,
                "low": self.delta.low,
                "high": self.delta.high,
            },
        }


def _present(series: pd.Series) -> np.ndarray:
    """True where a value is genuinely there - empty lists and "" are not."""
    if series.dtype == object:
        return series.map(
            lambda v: v is not None and not (isinstance(v, float) and pd.isna(v)) and len(v) > 0
        ).to_numpy(dtype=bool)
    return series.notna().to_numpy(dtype=bool)


def missingness_bias(
    judgements: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    seed: int = 0,
    n_resamples: int = 1000,
) -> list[FieldBias]:
    """Mean-gain shift between judgements whose product carries a field and not.

    Joins left from the judgements, so a judged product absent from the corpus
    counts as missing every field, which is the honest reading: Plan 5 will see
    a NaN there too.
    """
    fields = DENSE_FIELDS + SPARSE_FIELDS
    joined = judgements.merge(
        corpus[["asin", *fields]], left_on="product_id", right_on="asin", how="left"
    )
    gains = joined["gain"].to_numpy(dtype=float)
    clusters = joined["query_id"].to_numpy()

    biases: list[FieldBias] = []
    for name in fields:
        mask = _present(joined[name])
        if not mask.any() or not (~mask).any():
            continue
        biases.append(
            FieldBias(
                field=name,
                dense=name in DENSE_FIELDS,
                present_share=float(mask.mean()),
                mean_gain_present=float(gains[mask].mean()),
                mean_gain_absent=float(gains[~mask].mean()),
                delta=cluster_delta_ci(
                    gains, mask, clusters, n_resamples=n_resamples, seed=seed
                ),
            )
        )
    return biases


def check_missingness(biases: list[FieldBias]) -> None:
    """Raise if a dense field's missingness shifts mean gain too far."""
    for bias in biases:
        if bias.dense and abs(bias.delta.point) > MAX_DENSE_GAIN_SHIFT:
            raise MissingnessError(
                f"{bias.field} missingness shifts mean gain by "
                f"{bias.delta.point:+.4f} "
                f"[{bias.delta.low:+.4f}, {bias.delta.high:+.4f}], beyond the "
                f"{MAX_DENSE_GAIN_SHIFT} the behavioural ablation can tolerate. "
                "Treat this field as sparse and carry a missingness indicator."
            )


def format_biases(biases: list[FieldBias]) -> str:
    lines = [
        f"{'field':14s} {'dense':5s} {'present':>8s} {'gain+':>7s} "
        f"{'gain-':>7s} {'delta':>8s}  95% CI"
    ]
    for b in biases:
        lines.append(
            f"{b.field:14s} {'yes' if b.dense else 'no':5s} "
            f"{b.present_share:8.4f} {b.mean_gain_present:7.4f} "
            f"{b.mean_gain_absent:7.4f} {b.delta.point:+8.4f}  "
            f"[{b.delta.low:+.4f}, {b.delta.high:+.4f}]"
        )
    return "\n".join(lines)


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("docs/results"))
    args = parser.parse_args()

    # dict.fromkeys dedupes while keeping order: image_url is both the
    # coverage column and a sparse field, and asking Parquet for it twice
    # raises.
    corpus = load_corpus(
        columns=list(dict.fromkeys(["asin", *DENSE_FIELDS, *SPARSE_FIELDS]))
    )

    scopes = [
        join_coverage(rerank_product_ids(), corpus, scope="rerank").to_dict(),
        join_coverage(catalogue_product_ids(), corpus, scope="catalogue").to_dict(),
    ]
    for scope in scopes:
        print(
            f"{scope['scope']:10s} {scope['n_matched']:,}/{scope['n_products']:,} "
            f"= {scope['coverage']:.4f}   image {scope['image_coverage']:.4f}"
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "esci-s-coverage.json").write_text(
        json.dumps({"corpus_rows": len(corpus), "scopes": scopes}, indent=2) + "\n"
    )

    judgements = load_split(args.split).judgements
    biases = missingness_bias(
        judgements, corpus, seed=args.seed, n_resamples=args.n_resamples
    )
    print()
    print(format_biases(biases))
    (args.results_dir / "esci-s-missingness.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "seed": args.seed,
                "n_resamples": args.n_resamples,
                "max_dense_gain_shift": MAX_DENSE_GAIN_SHIFT,
                "fields": [b.to_dict() for b in biases],
            },
            indent=2,
        )
        + "\n"
    )
    check_missingness(biases)
    print("\ndense-field missingness is within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [x] **Step 6: Write the missingness tests**

Append to `tests/test_coverage.py`:

```python
import numpy as np

from src.coverage import (
    MAX_DENSE_GAIN_SHIFT,
    MissingnessError,
    check_missingness,
    missingness_bias,
)


def _judgements(n_queries: int = 60) -> pd.DataFrame:
    rows = []
    for q in range(n_queries):
        for j in range(4):
            rows.append(
                {"query_id": q, "product_id": f"P{q:03d}{j}", "gain": [1.0, 0.1, 0.01, 0.0][j]}
            )
    return pd.DataFrame(rows)


def _label_independent_mask(n_queries: int = 60) -> np.ndarray:
    """A missingness pattern genuinely independent of the gain.

    `np.arange(len(judgements)) % 2 == 0` looks independent and is not:
    judgements come four to a query in gain order, so index parity is position
    parity, which keeps gains {1.0, 0.01} and drops {0.1, 0.0} - a delta of
    0.455. Alternating the offset per query puts every gain level on both
    sides in equal numbers, so the true delta is exactly zero.
    """
    return np.array(
        [(q + j) % 2 == 0 for q in range(n_queries) for j in range(4)]
    )


def _enrichment(judgements: pd.DataFrame, present: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asin": judgements["product_id"],
            "stars": np.where(present, 4.5, np.nan),
            "ratings": np.where(present, 10, np.nan),
            "category": [["Books"] if p else [] for p in present],
            "template": np.where(present, "book", None),
            "price": np.where(present, 9.99, np.nan),
            "bsr_rank": np.where(present, 12, np.nan),
            "attrs_json": np.where(present, '{"Brand": "X"}', None),
            "info_json": np.where(present, "{}", None),
            "image_url": np.where(present, "http://i/1.jpg", None),
        }
    )


def test_missingness_uncorrelated_with_the_label_reports_near_zero():
    judgements = _judgements()
    present = _label_independent_mask()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    stars = next(b for b in biases if b.field == "stars")
    assert abs(stars.delta.point) < MAX_DENSE_GAIN_SHIFT
    check_missingness(biases)


def test_missingness_aligned_with_the_label_is_caught():
    # Every Exact judgement enriched, every Irrelevant one not: the confound
    # the gate exists to catch.
    judgements = _judgements()
    present = (judgements["gain"] > 0.5).to_numpy()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    with pytest.raises(MissingnessError, match="stars"):
        check_missingness(biases)


def test_dense_and_sparse_fields_are_labelled():
    judgements = _judgements()
    present = _label_independent_mask()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    by_field = {b.field: b.dense for b in biases}
    assert by_field["stars"] is True
    assert by_field["price"] is False


def test_a_sparse_field_confounded_with_the_label_is_reported_not_raised():
    # PROJECT_SPEC.md expects sparse fields to correlate; they are carried
    # with missingness indicators rather than rejected.
    judgements = _judgements()
    present = (judgements["gain"] > 0.5).to_numpy()
    biases = missingness_bias(judgements, _enrichment(judgements, present), n_resamples=200)
    price = next(b for b in biases if b.field == "price")
    assert abs(price.delta.point) > MAX_DENSE_GAIN_SHIFT
    assert price.dense is False


def test_a_judged_product_absent_from_the_corpus_counts_as_missing():
    judgements = _judgements()
    present = np.ones(len(judgements), dtype=bool)
    enrichment = _enrichment(judgements, present).iloc[:100]
    biases = missingness_bias(judgements, enrichment, n_resamples=100)
    stars = next(b for b in biases if b.field == "stars")
    assert stars.present_share == pytest.approx(100 / len(judgements))


@pytest.mark.data
def test_real_dense_fields_are_not_confounded_with_the_label():
    from pathlib import Path

    payload = json.loads(Path("docs/results/esci-s-missingness.json").read_text())
    dense = [f for f in payload["fields"] if f["dense"]]
    # `category` is deliberately absent: the real report measured it at
    # -0.0210 and Task 5 Step 8 reclassified it as sparse.
    assert {f["field"] for f in dense} == {"stars", "ratings", "template"}
    for f in dense:
        assert abs(f["delta"]["point"]) < MAX_DENSE_GAIN_SHIFT, f
```

- [x] **Step 7: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_coverage.py tests/test_bootstrap.py -v
```

Expected: PASS, 13 coverage + 16 bootstrap tests; the two `data`-marked tests
are deselected.

- [x] **Step 8: Run the real report**

```bash
python -m src.coverage --split test
python -m pytest tests/test_coverage.py -v -m data
```

Expected: both JSON records written, rerank coverage ≥ 0.90, and every dense
field's mean-gain shift inside ±0.02.

**If a dense field fails,** that is the Plan 2 gate failing and it is a real
result, not a threshold to widen. Record the number, move that field out of
`DENSE_FIELDS` into `SPARSE_FIELDS`, and say so in the commit message — Plan 5's
behavioural ablation then carries it with a missingness indicator instead of
resting on it.

- [x] **Step 9: Record the commands in `CLAUDE.md`**

Add to the Commands block, above the image-gate line:

````markdown
# Enrichment corpus (one streaming pass over the 3.4 GB ESCI-S zstd)
python -m src.esci_s_etl                      # -> data/esci-s/corpus.parquet
python -m src.coverage --split test           # join coverage + missingness bias
````

- [x] **Step 10: Commit**

```bash
git add src/bootstrap.py src/coverage.py tests/test_bootstrap.py tests/test_coverage.py docs/results/esci-s-missingness.json CLAUDE.md
git commit -m "Report ESCI-S missingness bias against the relevance label"
```

---

## Phase 2 Gate

This is the Plan Gate — Plan 3 does not start until all of these hold. The
canonical copy lives in [`README.md`](README.md#plan-gate).

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the corpus invariants.
- [ ] `data/esci-s/corpus.parquet` exists and its measured field presence matches the Global Constraints table within ±0.02.
- [ ] `docs/results/esci-s-coverage.json` shows rerank coverage **≥ 90%** over 482,105 products, measured — not the 91.5% headline.
- [ ] `docs/results/esci-s-missingness.json` shows a mean-gain shift below 0.02 for each of the four dense fields, with its cluster-bootstrap CI.
- [ ] `CLAUDE.md`'s Commands section lists the ETL and coverage invocations.

Then: Plan 3 — Image Pipeline. See [`../README.md`](../README.md) for the series.
