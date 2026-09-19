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
