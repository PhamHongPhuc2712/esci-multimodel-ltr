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
