# Phase 2 — The Fetch

**Plan 3 of 7 · Phase 2 of 2 · Tasks 3–5.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 1 gate](phase-1-the-embedder.md#phase-1-gate) passes.

**Delivers:** the concurrent fetcher, the streaming pipeline that turns URLs
into stored vectors without the images ever touching disk, and the real run
with its committed coverage and semantic-gate record.

**Needs on disk:** `data/combined/products.parquet` from Plan 2. The run
transfers 3.81 GB (`rerank`) or 9.05 GB (`catalogue`) and writes 383 MB or
908 MB of vectors. Budget 1.8–7.3 hours for `rerank`; it is resumable.

**Owns Review Focus items 1 and 4** (an HTTP 200 that is not an image,
duplicate URLs fetched once per product).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **A 200 is not an image.** Validate by decoding. Measured: 600/600 real images decoded, min short side 252 px — so anything tiny is a placeholder, not a product.
- **Never persist the images.** Embed in flight; keep the vectors only.
- **Fetch is keyed by URL, not by product.** 932,320 products share 887,041 URLs.
- **Retry with backoff on 429 and 5xx**, a real User-Agent, concurrency capped at 14.
- **The GPU is not the bottleneck.** Fetch 3,378–5,516/min against CLIP's 18,140/min, so the pipeline is network-bound by ~5× and the embed is effectively free.
- **Write as you go.** Three measured runs varied 6× in throughput; resume is what makes that survivable.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 3: The concurrent fetcher

**Files:**
- Create: `src/image_fetch.py`
- Create: `tests/test_image_fetch.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `src.image_fetch.USER_AGENT: str`, `DEFAULT_CONCURRENCY: int` (14), `DEFAULT_TIMEOUT: float` (25.0), `DEFAULT_RETRIES: int` (3), `RETRY_STATUSES: frozenset[int]`, `MIN_SIDE: int` (16)
  - `src.image_fetch.FetchResult(url, image, status, attempts)` with `.ok -> bool`
  - `src.image_fetch.decode_image(body: bytes) -> "PIL.Image.Image | None"`
  - `src.image_fetch.fetch_one(url, *, opener, timeout=..., retries=..., backoff=0.5, sleep=time.sleep) -> FetchResult`
  - `src.image_fetch.fetch_many(urls, *, opener=None, concurrency=DEFAULT_CONCURRENCY, **kwargs) -> Iterator[FetchResult]`
  - `src.image_fetch.urlopen_opener(url, timeout) -> tuple[int | str, bytes]`

`fetch_one` takes an `opener` callable and a `sleep` callable so retry, backoff
and status handling are tested in milliseconds with no network and no waiting.
`urlopen_opener` is the real one.

- [x] **Step 1: Write the failing test**

Create `tests/test_image_fetch.py`:

```python
import io

import pytest
from PIL import Image

from src.image_fetch import (
    DEFAULT_CONCURRENCY,
    MIN_SIDE,
    RETRY_STATUSES,
    decode_image,
    fetch_many,
    fetch_one,
)


def _png(size=(300, 300), colour=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _opener(script):
    """An opener driven by a list of (status, body) per url, consumed in order."""
    calls = []

    def opener(url, timeout):
        calls.append(url)
        responses = script[url]
        return responses.pop(0) if len(responses) > 1 else responses[0]

    return opener, calls


# --- Review Focus 1: a 200 that is not an image ----------------------------

def test_an_html_error_page_with_status_200_is_not_a_success():
    # CDNs answer some bad paths with a 200 and an HTML body. Trusting the
    # status code would embed the error page as if it were a product.
    opener, _ = _opener({"u": [(200, b"<html>Not found</html>")]})
    result = fetch_one("u", opener=opener)
    assert result.status == 200
    assert result.image is None
    assert not result.ok


def test_a_truncated_image_body_is_not_a_success():
    opener, _ = _opener({"u": [(200, _png()[:120])]})
    assert not fetch_one("u", opener=opener).ok


def test_an_empty_body_is_not_a_success():
    opener, _ = _opener({"u": [(200, b"")]})
    assert not fetch_one("u", opener=opener).ok


def test_a_placeholder_smaller_than_a_real_product_image_is_rejected():
    # 600 real product images measured: shortest side 252 px, 300x300
    # dominant. A 1x1 tracking pixel decodes perfectly well, so decodability
    # alone is not enough.
    opener, _ = _opener({"u": [(200, _png(size=(1, 1)))]})
    assert not fetch_one("u", opener=opener).ok
    assert MIN_SIDE == 16


def test_a_real_image_decodes_to_rgb():
    opener, _ = _opener({"u": [(200, _png())]})
    result = fetch_one("u", opener=opener)
    assert result.ok
    assert result.image.mode == "RGB"
    assert result.image.size == (300, 300)


def test_decode_image_returns_none_rather_than_raising():
    assert decode_image(b"not an image") is None
    assert decode_image(b"") is None


# --- retry and backoff ------------------------------------------------------

def test_a_429_is_retried_and_can_succeed():
    opener, calls = _opener({"u": [(429, b""), (429, b""), (200, _png())]})
    slept = []
    result = fetch_one("u", opener=opener, sleep=slept.append)
    assert result.ok
    assert result.attempts == 3
    assert len(calls) == 3


def test_backoff_grows_between_attempts():
    opener, _ = _opener({"u": [(503, b""), (503, b""), (200, _png())]})
    slept = []
    fetch_one("u", opener=opener, backoff=0.5, sleep=slept.append)
    assert slept == [0.5, 1.0]


def test_retry_statuses_are_the_transient_ones():
    assert RETRY_STATUSES == frozenset({429, 500, 502, 503, 504})


def test_a_404_is_not_retried():
    # The product is simply gone; retrying wastes a slot in a multi-hour run.
    opener, calls = _opener({"u": [(404, b"")]})
    result = fetch_one("u", opener=opener)
    assert not result.ok
    assert result.attempts == 1
    assert len(calls) == 1


def test_retries_are_capped():
    opener, calls = _opener({"u": [(503, b"")]})
    result = fetch_one("u", opener=opener, retries=2, sleep=lambda _: None)
    assert not result.ok
    assert len(calls) == 2


def test_a_transport_error_is_retried_then_reported():
    def opener(url, timeout):
        raise OSError("connection reset")

    result = fetch_one("u", opener=opener, retries=2, sleep=lambda _: None)
    assert not result.ok
    assert result.status == "OSError"
    assert result.attempts == 2


# --- concurrency ------------------------------------------------------------

def test_fetch_many_yields_one_result_per_url():
    opener, _ = _opener({f"u{i}": [(200, _png())] for i in range(20)})
    results = list(fetch_many([f"u{i}" for i in range(20)], opener=opener))
    assert len(results) == 20
    assert {r.url for r in results} == {f"u{i}" for i in range(20)}
    assert all(r.ok for r in results)


def test_fetch_many_reports_failures_rather_than_dropping_them():
    # A dropped failure is indistinguishable from a URL never attempted, and
    # the run report would overstate coverage.
    script = {"good": [(200, _png())], "bad": [(404, b"")]}
    opener, _ = _opener(script)
    results = {r.url: r for r in fetch_many(["good", "bad"], opener=opener)}
    assert results["good"].ok
    assert not results["bad"].ok
    assert results["bad"].status == 404


def test_default_concurrency_is_the_measured_safe_value():
    # 3,378-5,516 images/minute at 14, with no 429s observed across 1,400
    # live requests.
    assert DEFAULT_CONCURRENCY == 14
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_image_fetch.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.image_fetch'`.

- [x] **Step 3: Write `src/image_fetch.py`**

```python
"""Concurrent image fetch with retry, backoff and decode validation.

This is a static asset fetch, not page scraping: ESCI-S already did the
expensive, block-prone work of discovering the URLs. Measured against the live
CDN at concurrency 14: 3,378-5,516 images/minute, median 8.5 KB, 82% WEBP and
18% JPEG, 600/600 decoded, no 429s across 1,400 requests.

A 200 is not an image. A CDN error page, a placeholder and a truncated body
all answer 200, so success means *decoded to something plausibly a product
photo*, never *the status code was fine*.

`fetch_one` takes `opener` and `sleep` callables so retry and backoff are
tested with no network and no waiting; `urlopen_opener` is the real one.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

USER_AGENT = "Mozilla/5.0 (compatible; esci-multimodel-ltr/0.1)"
DEFAULT_CONCURRENCY = 14
DEFAULT_TIMEOUT = 25.0
DEFAULT_RETRIES = 3
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# 600 real product images measured: shortest side 252 px, 300x300 dominant. A
# 1x1 tracking pixel decodes perfectly well, so decodability alone does not
# separate a product photo from a placeholder.
MIN_SIDE = 16

Opener = Callable[[str, float], tuple[Any, bytes]]


@dataclass(frozen=True)
class FetchResult:
    url: str
    image: Any | None
    status: Any
    attempts: int

    @property
    def ok(self) -> bool:
        return self.image is not None


def decode_image(body: bytes) -> Any | None:
    """Decode to an RGB image, or None if the bytes are not a product photo."""
    if not body:
        return None
    from PIL import Image

    try:
        with Image.open(io.BytesIO(body)) as opened:
            opened.load()  # force the decode; Image.open is lazy
            image = opened.convert("RGB")
    except Exception:
        return None
    if min(image.size) < MIN_SIDE:
        return None
    return image


def urlopen_opener(url: str, timeout: float) -> tuple[Any, bytes]:
    """The real opener: returns (status, body), or (status, b"") on an HTTP error."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


def fetch_one(
    url: str,
    *,
    opener: Opener = urlopen_opener,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    """Fetch and decode one image, retrying transient failures with backoff."""
    status: Any = "unattempted"
    for attempt in range(1, retries + 1):
        try:
            status, body = opener(url, timeout)
        except (urllib.error.URLError, OSError) as error:
            status = type(error).__name__
        else:
            if status == 200:
                image = decode_image(body)
                return FetchResult(url, image, status, attempt)
            if status not in RETRY_STATUSES:
                # 404 and friends: the product is gone, and retrying wastes a
                # slot in a run that already takes hours.
                return FetchResult(url, None, status, attempt)
        if attempt < retries:
            sleep(backoff * (2 ** (attempt - 1)))
    return FetchResult(url, None, status, retries)


def fetch_many(
    urls: Iterable[str],
    *,
    opener: Opener = urlopen_opener,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[FetchResult]:
    """Fetch many images concurrently, yielding every result including failures.

    Failures are yielded, not dropped: a dropped failure is indistinguishable
    from a URL that was never attempted, and the run report would overstate
    coverage.
    """
    urls = list(urls)
    if not urls:
        return

    def one(url: str) -> FetchResult:
        return fetch_one(
            url,
            opener=opener,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
            sleep=sleep,
        )

    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        yield from pool.map(one, urls)
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_image_fetch.py -v
```

Expected: PASS, 15 tests.

- [x] **Step 5: Commit**

```bash
git add src/image_fetch.py tests/test_image_fetch.py
git commit -m "Add concurrent image fetch with retry and decode validation"
```

---

## Task 4: The streaming pipeline

**Files:**
- Create: `src/embed_images.py`
- Create: `tests/test_embed_images.py`

**Interfaces:**
- Consumes: `src.image_fetch.fetch_many`, `src.image_fetch.FetchResult`, `src.embedding_store.open_store`, `src.clip_encoder.load_encoder`, `src.clip_encoder.EMBEDDING_DIM`, `src.clip_encoder.DEFAULT_BATCH_SIZE`, `src.combine.DEFAULT_DEST`.
- Produces:
  - `src.embed_images.DEFAULT_STORE_ROOT: Path` (`data/embeddings`), `DEFAULT_PRODUCTS: Path`, `SCOPES: tuple[str, ...]`
  - `src.embed_images.DEFAULT_CHUNK: int` (512)
  - `src.embed_images.RunStats` with `.to_dict()`
  - `src.embed_images.product_image_urls(scope, products_path=...) -> pd.DataFrame` with columns `product_id`, `url`
  - `src.embed_images.pending_urls(frame, store) -> list[str]`
  - `src.embed_images.embed_urls(urls, store, encode_images, *, fetch=fetch_many, chunk=DEFAULT_CHUNK, batch_size=..., stats=None) -> RunStats`
  - `python -m src.embed_images --scope {rerank,catalogue}`

`embed_urls` takes `encode_images` and `fetch` as callables so the whole
pipeline — dedup, chunking, skip-existing, failure accounting — is tested with
no network, no GPU and no model.

- [x] **Step 1: Write the failing test**

Create `tests/test_embed_images.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.clip_encoder import EMBEDDING_DIM
from src.embed_images import (
    DEFAULT_CHUNK,
    embed_urls,
    pending_urls,
    product_image_urls,
)
from src.embedding_store import open_store
from src.image_fetch import FetchResult


def _image() -> Image.Image:
    return Image.new("RGB", (300, 300), (255, 0, 0))


def _fake_fetch(failing: set[str] | None = None):
    failing = failing or set()
    seen: list[str] = []

    def fetch(urls, **kwargs):
        for url in urls:
            seen.append(url)
            if url in failing:
                yield FetchResult(url, None, 404, 1)
            else:
                yield FetchResult(url, _image(), 200, 1)

    return fetch, seen


def _fake_encode(dim: int = EMBEDDING_DIM):
    batches: list[int] = []

    def encode_images(images, batch_size=None):
        batches.append(len(images))
        return np.ones((len(images), dim), dtype=np.float32) / np.sqrt(dim)

    return encode_images, batches


def _products(tmp_path, rows) -> Path:
    path = tmp_path / "products.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# --- Review Focus 4: duplicate URLs -----------------------------------------

def test_a_url_shared_by_several_products_is_fetched_once():
    # 932,320 catalogue products share 887,041 URLs. Keying on the product
    # would waste 5% of a multi-hour fetch and store the same vector several
    # times under different ids.
    store = open_store_in_memory()
    fetch, seen = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls(["u1", "u1", "u2"], store, encode_images, fetch=fetch)
    assert sorted(seen) == ["u1", "u2"]
    assert len(store) == 2


def test_pending_urls_deduplicates_and_skips_what_is_stored():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls(["u1"], store, encode_images, fetch=fetch)
    frame = pd.DataFrame(
        {"product_id": ["a", "b", "c"], "url": ["u1", "u2", "u2"]}
    )
    assert pending_urls(frame, store) == ["u2"]


# --- the pipeline -----------------------------------------------------------

def test_every_successful_url_lands_in_the_store():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch()
    encode_images, _ = _fake_encode()
    stats = embed_urls(["u1", "u2", "u3"], store, encode_images, fetch=fetch)
    assert stats.embedded == 3
    assert store.known_keys() == {"u1", "u2", "u3"}


def test_failures_are_counted_and_not_stored():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch(failing={"u2"})
    encode_images, _ = _fake_encode()
    stats = embed_urls(["u1", "u2", "u3"], store, encode_images, fetch=fetch)
    assert stats.embedded == 2
    assert stats.failed == 1
    assert "u2" not in store.known_keys()


def test_a_chunk_that_fails_entirely_does_not_break_the_run():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch(failing={"u1", "u2"})
    encode_images, batches = _fake_encode()
    stats = embed_urls(["u1", "u2"], store, encode_images, fetch=fetch, chunk=2)
    assert stats.embedded == 0
    assert batches == []  # nothing to encode, and no empty batch sent
    assert len(store) == 0


def test_the_run_is_chunked_so_progress_is_written_as_it_goes():
    # The run takes hours and will be killed. Appending once at the end would
    # lose everything.
    store = open_store_in_memory()
    fetch, _ = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls([f"u{i}" for i in range(10)], store, encode_images, fetch=fetch, chunk=4)
    assert len(store) == 10


def test_resuming_skips_what_is_already_stored():
    store = open_store_in_memory()
    fetch, seen = _fake_fetch()
    encode_images, _ = _fake_encode()
    embed_urls(["u1", "u2"], store, encode_images, fetch=fetch)
    seen.clear()
    stats = embed_urls(["u1", "u2", "u3"], store, encode_images, fetch=fetch)
    assert seen == ["u3"]
    assert stats.skipped == 2
    assert len(store) == 3


def test_default_chunk_keeps_memory_bounded():
    assert DEFAULT_CHUNK == 512


def test_stats_serialise_for_the_committed_record():
    store = open_store_in_memory()
    fetch, _ = _fake_fetch(failing={"u2"})
    encode_images, _ = _fake_encode()
    stats = embed_urls(["u1", "u2"], store, encode_images, fetch=fetch)
    payload = stats.to_dict()
    assert payload["embedded"] == 1
    assert payload["failed"] == 1
    assert payload["requested"] == 2


# --- reading the product table ---------------------------------------------

def test_product_image_urls_drops_products_with_no_url(tmp_path):
    path = _products(
        tmp_path,
        [
            {"product_id": "a", "s_image_url": "u1"},
            {"product_id": "b", "s_image_url": None},
            {"product_id": "c", "s_image_url": ""},
        ],
    )
    frame = product_image_urls("catalogue", products_path=path)
    assert list(frame["product_id"]) == ["a"]
    assert list(frame.columns) == ["product_id", "url"]


def test_product_image_urls_rejects_an_unknown_scope(tmp_path):
    path = _products(tmp_path, [{"product_id": "a", "s_image_url": "u1"}])
    with pytest.raises(ValueError, match="scope"):
        product_image_urls("everything", products_path=path)
```

Add this helper at the top of the test module, under the imports:

```python
def open_store_in_memory():
    """A store in a throwaway directory, for tests that do not assert on disk."""
    import tempfile

    return open_store(tempfile.mkdtemp(), dim=EMBEDDING_DIM)
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_embed_images.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.embed_images'`.

- [x] **Step 3: Write `src/embed_images.py`**

```python
"""Stream image URLs into CLIP vectors, never writing an image to disk.

One pass: URL -> bytes -> decoded image -> 512-d vector -> store, a chunk at a
time so memory stays bounded and progress is written as it goes. The run takes
between 1.8 and 7.3 hours for the re-ranking scope - three measured samples
varied sixfold in throughput - and will be interrupted, so every chunk is
appended before the next is fetched.

The network is the bottleneck by roughly 5x: 3,378-5,516 images/minute fetched
against 18,140/minute embedded on the RTX 3080. Embedding in flight is
therefore free, which is what makes never persisting the images practical.

`embed_urls` takes `fetch` and `encode_images` as callables, so dedup,
chunking, skip-existing and failure accounting are all tested with no network,
no GPU and no model.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.clip_encoder import DEFAULT_BATCH_SIZE, EMBEDDING_DIM
from src.combine import DEFAULT_DEST as COMBINED_DEST
from src.embedding_store import EmbeddingStore, open_store
from src.image_fetch import DEFAULT_CONCURRENCY, fetch_many

DEFAULT_STORE_ROOT = Path("data/embeddings")
DEFAULT_PRODUCTS = COMBINED_DEST / "products.parquet"

# 512 URLs in flight is ~5 MB of image bytes at the measured 10 KB mean, and
# eight full CLIP batches - enough to keep the GPU fed without holding a
# meaningful fraction of the run in memory.
DEFAULT_CHUNK = 512

SCOPES = ("rerank", "catalogue")


@dataclass
class RunStats:
    requested: int = 0
    skipped: int = 0
    fetched: int = 0
    failed: int = 0
    embedded: int = 0
    seconds: float = 0.0
    status_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        rate = 60 * self.embedded / self.seconds if self.seconds else 0.0
        return {
            "requested": self.requested,
            "skipped": self.skipped,
            "fetched": self.fetched,
            "failed": self.failed,
            "embedded": self.embedded,
            "seconds": round(self.seconds, 1),
            "images_per_minute": round(rate, 1),
            "status_counts": self.status_counts,
        }


def product_image_urls(
    scope: str, products_path: Path = DEFAULT_PRODUCTS
) -> pd.DataFrame:
    """product_id and url for every product that has an image URL."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPES}")

    frame = pd.read_parquet(products_path, columns=["product_id", "s_image_url"])
    urls = frame["s_image_url"]
    has_url = urls.notna() & (urls.astype("str").str.len() > 0)
    frame = frame.loc[has_url].rename(columns={"s_image_url": "url"})

    if scope == "rerank":
        from src.coverage import rerank_product_ids

        frame = frame.loc[frame["product_id"].isin(rerank_product_ids())]
    return frame[["product_id", "url"]].reset_index(drop=True)


def pending_urls(frame: pd.DataFrame, store: EmbeddingStore) -> list[str]:
    """Distinct URLs not already in the store, in first-seen order.

    Distinct because 932,320 catalogue products share 887,041 URLs: keying the
    fetch on the product would repeat 5% of a multi-hour run and write the
    same vector several times.
    """
    known = store.known_keys()
    seen: dict[str, None] = {}
    for url in frame["url"]:
        if url not in known:
            seen.setdefault(url, None)
    return list(seen)


def embed_urls(
    urls: Sequence[str],
    store: EmbeddingStore,
    encode_images: Callable[..., Any],
    *,
    fetch: Callable[..., Iterable[Any]] = fetch_many,
    chunk: int = DEFAULT_CHUNK,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[RunStats], None] | None = None,
) -> RunStats:
    """Fetch, decode, embed and store, one chunk at a time."""
    stats = RunStats(requested=len(urls))

    known = store.known_keys()
    ordered: dict[str, None] = {}
    for url in urls:
        if url in known:
            stats.skipped += 1
        else:
            ordered.setdefault(url, None)
    todo = list(ordered)

    started = time.time()
    for start in range(0, len(todo), chunk):
        batch_urls = todo[start : start + chunk]
        images: list[Any] = []
        keys: list[str] = []
        for result in fetch(batch_urls, concurrency=concurrency):
            status = str(result.status)
            stats.status_counts[status] = stats.status_counts.get(status, 0) + 1
            if result.ok:
                stats.fetched += 1
                images.append(result.image)
                keys.append(result.url)
            else:
                stats.failed += 1

        if images:
            vectors = encode_images(images, batch_size=batch_size)
            store.append(keys, vectors)
            stats.embedded += len(keys)
        stats.seconds = time.time() - started
        if progress is not None:
            progress(stats)

    stats.seconds = time.time() - started
    return stats


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="rerank", choices=list(SCOPES))
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many URLs"
    )
    args = parser.parse_args()

    frame = product_image_urls(args.scope, products_path=args.products)
    store = open_store(args.store_root / args.scope, dim=EMBEDDING_DIM)
    todo = pending_urls(frame, store)
    if args.limit is not None:
        todo = todo[: args.limit]

    print(
        f"scope {args.scope}: {len(frame):,} products with a URL, "
        f"{frame['url'].nunique():,} distinct, {len(store):,} already stored, "
        f"{len(todo):,} to fetch"
    )

    from src.clip_encoder import load_encoder

    encoder = load_encoder(device=args.device)
    print(f"encoding on {encoder.device}, dim {encoder.dim}")

    def report(stats: RunStats) -> None:
        done = stats.embedded + stats.failed
        rate = 60 * done / stats.seconds if stats.seconds else 0.0
        print(
            f"\r  {done:,}/{len(todo):,}  embedded {stats.embedded:,}  "
            f"failed {stats.failed:,}  {rate:,.0f}/min",
            end="",
            flush=True,
        )

    stats = embed_urls(
        todo,
        store,
        encoder.encode_images,
        chunk=args.chunk,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        progress=report,
    )
    print()
    print(json.dumps(stats.to_dict(), indent=2))
    print(f"store now holds {len(store):,} vectors at {args.store_root / args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_embed_images.py -v
```

Expected: PASS, 11 tests.

- [x] **Step 5: Commit**

```bash
git add src/embed_images.py tests/test_embed_images.py
git commit -m "Add the streaming image embedding pipeline with resume"
```

---

## Task 5: The real run, the coverage record and the semantic gate

**Files:**
- Create: `src/image_report.py`
- Create: `tests/test_image_report.py`
- Create: `docs/results/image-embeddings.json` (generated in Step 4, committed)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `src.embed_images.product_image_urls`, `src.embedding_store.open_store`, `src.clip_encoder.load_encoder`, `src.coverage.rerank_product_ids`.
- Produces:
  - `src.image_report.MIN_TOP1: float` (0.60), `GATE_POOL: int` (600)
  - `src.image_report.CoverageRecord(scope, products, with_url, embedded, url_coverage, embedding_coverage)` with `.to_dict()`
  - `src.image_report.image_coverage(frame, store, n_products, scope="rerank") -> CoverageRecord`
  - `src.image_report.SemanticGateError` (subclass of `ValueError`)
  - `src.image_report.top1_accuracy(image_vectors, text_vectors) -> float`
  - `python -m src.image_report --scope rerank`

`top1_accuracy` is a pure function over two matrices, so the gate arithmetic is
tested without a model. The threshold is 0.60 against a **measured** 0.753 on
a 600-title pool; chance is 0.167%. The band is wide because it exists to
catch catastrophic breakage — misaligned ids, unnormalised vectors, the wrong
pooling output — not to police the third decimal.

- [x] **Step 1: Write the failing test**

Create `tests/test_image_report.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.image_report import (
    GATE_POOL,
    MIN_TOP1,
    image_coverage,
    top1_accuracy,
)


class _Store:
    def __init__(self, keys):
        self._keys = set(keys)

    def known_keys(self):
        return set(self._keys)


def test_perfectly_aligned_vectors_score_one():
    vectors = np.eye(5, dtype=np.float32)
    assert top1_accuracy(vectors, vectors) == pytest.approx(1.0)


def test_shuffled_vectors_score_near_chance():
    # This is the failure the gate exists for: embeddings and ids drifting out
    # of alignment. It does not raise anywhere; it just quietly stops meaning
    # anything.
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    shuffled = vectors[rng.permutation(200)]
    assert top1_accuracy(vectors, shuffled) < 0.10


def test_top1_counts_a_tie_as_a_miss():
    # Two identical candidate texts must not be scored as a hit by argmax
    # happening to land on the right index.
    images = np.array([[1.0, 0.0]], dtype=np.float32)
    texts = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert top1_accuracy(images, texts) == 0.0


def test_the_gate_reflects_the_measured_baseline():
    # Measured 0.753 top-1 over a 600-title pool, chance 0.00167.
    assert MIN_TOP1 == 0.60
    assert GATE_POOL == 600


def test_coverage_divides_by_the_product_set_not_the_matches():
    # PROJECT_SPEC.md: report ~75% end to end, not ESCI-S's 91.5% headline.
    # Dividing by products that have a URL rather than by all products is
    # exactly how the headline overstates it.
    frame = pd.DataFrame(
        {"product_id": ["a", "b"], "url": ["u1", "u2"]}
    )
    record = image_coverage(frame, _Store({"u1"}), n_products=4)
    assert record.with_url == 2
    assert record.url_coverage == pytest.approx(0.5)
    assert record.embedded == 1
    assert record.embedding_coverage == pytest.approx(0.25)


def test_coverage_counts_a_shared_url_once_per_product():
    # Two products sharing one URL are both covered by one stored vector.
    frame = pd.DataFrame({"product_id": ["a", "b"], "url": ["u1", "u1"]})
    record = image_coverage(frame, _Store({"u1"}), n_products=2)
    assert record.embedded == 2
    assert record.embedding_coverage == pytest.approx(1.0)


def test_coverage_serialises(tmp_path):
    record = image_coverage(
        pd.DataFrame({"product_id": ["a"], "url": ["u1"]}),
        _Store({"u1"}),
        n_products=1,
    )
    assert record.to_dict()["embedding_coverage"] == pytest.approx(1.0)
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_image_report.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.image_report'`.

- [x] **Step 3: Write `src/image_report.py`**

```python
"""Image coverage and the semantic gate.

Coverage divides by the product set, never by the products that happen to have
a URL - dividing by the latter is exactly how ESCI-S's 91.5% headline gets
mistaken for image coverage when the end-to-end figure is ~77.5%.

The gate is a retrieval check: encode N product images and their own titles,
and ask how often an image's nearest title is its own. Measured on 600 real
pairs: top-1 0.753, top-5 0.927, median rank 1, against a 0.00167 chance rate.
The threshold sits at 0.60 because it is there to catch catastrophic breakage
- ids misaligned with vectors, vectors unnormalised, the wrong pooling output
- not to police the third decimal.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MIN_TOP1 = 0.60
GATE_POOL = 600


class SemanticGateError(ValueError):
    """Raised when image and text embeddings no longer line up."""


@dataclass(frozen=True)
class CoverageRecord:
    scope: str
    products: int
    with_url: int
    embedded: int
    url_coverage: float
    embedding_coverage: float

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "products": self.products,
            "with_url": self.with_url,
            "embedded": self.embedded,
            "url_coverage": self.url_coverage,
            "embedding_coverage": self.embedding_coverage,
        }


def image_coverage(frame: pd.DataFrame, store, n_products: int, scope: str = "rerank"):
    """Coverage of `n_products`, by URL and by stored embedding."""
    if n_products <= 0:
        raise ValueError("n_products must be positive")
    known = store.known_keys()
    with_url = int(frame["product_id"].nunique())
    embedded = int(frame.loc[frame["url"].isin(known), "product_id"].nunique())
    return CoverageRecord(
        scope=scope,
        products=n_products,
        with_url=with_url,
        embedded=embedded,
        url_coverage=with_url / n_products,
        embedding_coverage=embedded / n_products,
    )


def top1_accuracy(image_vectors: np.ndarray, text_vectors: np.ndarray) -> float:
    """Share of images whose nearest text is their own, by cosine.

    Ties count as misses: two identical candidate texts must not be scored as
    a hit because argmax happened to land on the right index.
    """
    similarity = np.asarray(image_vectors, dtype=np.float32) @ np.asarray(
        text_vectors, dtype=np.float32
    ).T
    truth = np.diag(similarity)[:, None]
    better_or_equal = (similarity >= truth).sum(axis=1)
    return float((better_or_equal == 1).mean())


def _main() -> int:
    from src.clip_encoder import load_encoder
    from src.coverage import rerank_product_ids
    from src.embed_images import DEFAULT_PRODUCTS, DEFAULT_STORE_ROOT, product_image_urls
    from src.embedding_store import open_store
    from src.image_fetch import fetch_many

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="rerank", choices=["rerank", "catalogue"])
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--pool", type=int, default=GATE_POOL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("docs/results/image-embeddings.json"))
    args = parser.parse_args()

    frame = product_image_urls(args.scope, products_path=args.products)
    store = open_store(args.store_root / args.scope)
    n_products = (
        len(rerank_product_ids())
        if args.scope == "rerank"
        else len(pd.read_parquet(args.products, columns=["product_id"]))
    )
    record = image_coverage(frame, store, n_products, scope=args.scope)
    print(
        f"{record.scope}: {record.with_url:,}/{record.products:,} have a URL "
        f"({record.url_coverage:.2%}); {record.embedded:,} have an embedding "
        f"({record.embedding_coverage:.2%})"
    )

    # --- the semantic gate --------------------------------------------------
    titles = pd.read_parquet(
        args.products, columns=["product_id", "product_title", "s_image_url"]
    ).dropna(subset=["product_title", "s_image_url"])
    titles = titles.loc[titles["s_image_url"].isin(store.known_keys())]
    titles = titles.drop_duplicates("s_image_url").sample(
        n=min(args.pool, len(titles)), random_state=args.seed
    )

    encoder = load_encoder(device=args.device)
    results = {r.url: r for r in fetch_many(list(titles["s_image_url"]))}
    usable = [
        (results[u].image, str(t)[:77])
        for u, t in zip(titles["s_image_url"], titles["product_title"])
        if u in results and results[u].ok
    ]
    images = [i for i, _ in usable]
    texts = [t for _, t in usable]
    top1 = top1_accuracy(encoder.encode_images(images), encoder.encode_texts(texts))
    print(f"semantic gate: top-1 {top1:.3%} over a {len(usable)}-title pool "
          f"(chance {1 / max(len(usable), 1):.3%}, threshold {MIN_TOP1:.0%})")

    payload = record.to_dict() | {
        "semantic_gate": {
            "pool": len(usable),
            "top1": top1,
            "threshold": MIN_TOP1,
            "seed": args.seed,
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"written to {args.out}")

    if top1 < MIN_TOP1:
        raise SemanticGateError(
            f"top-1 {top1:.3%} is below {MIN_TOP1:.0%}; image and text "
            "embeddings are not lining up. Check that the store's keys match "
            "its rows, that vectors are L2-normalised, and that the encoder "
            "reads .pooler_output rather than .last_hidden_state."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [x] **Step 4: Run the fast tests, then the real pipeline**

```bash
python -m pytest tests/test_image_report.py -v
python -m pytest -q

# the resolution gate, re-run rather than trusted
python -m src.esci_images data/esci-s/esci.json.zst

# the run itself: 1.8-7.3 hours for rerank, resumable
python -m src.embed_images --scope rerank

python -m src.image_report --scope rerank
```

Expected: 7 report tests pass; the resolution gate exits zero; the run reports
~373,776 URLs with a handful of failures; the report prints an embedding
coverage near **0.775** and a top-1 near **0.75**.

**If the run stalls or throughput collapses**, stop it and restart — it resumes
from the store. Three measured runs varied from 850 to 5,516 images/minute at
the same concurrency, so a slow stretch is the CDN, not a bug. Do not raise
`--concurrency` above 14 to compensate; no 429s were observed at 14 across
1,400 requests, and that headroom is the reason.

**If the semantic gate fails**, do not lower `MIN_TOP1`. Work through, in order:

1. Are the store's keys still aligned with its rows? `len(store) == len(store.keys())` and a spot check that `store.lookup([k])` returns the vector that key was appended with.
2. Are the vectors L2-normalised? `np.linalg.norm(store.vectors()[:10].astype("float32"), axis=1)` should be 1.0 to within float16 error.
3. Is the encoder reading `.pooler_output` (512-d) and not `.last_hidden_state` (768-d)?
4. Is the title column the right one? `product_title` from ESCI, not `s_title`.

- [x] **Step 5: Record the commands in `CLAUDE.md`**

Add to the Commands block, above the image-gate line:

````markdown
# Image embeddings (network-bound; resumable, 1.8-7.3 h for the rerank scope)
python -m src.embed_images --scope rerank     # -> data/embeddings/rerank/
python -m src.image_report --scope rerank     # coverage + semantic gate
````

- [x] **Step 6: Commit**

```bash
git add src/image_report.py tests/test_image_report.py docs/results/image-embeddings.json CLAUDE.md
git commit -m "Embed product images with CLIP and record coverage and the semantic gate"
```

---

## Phase 2 Gate

This is the Plan Gate — Plan 4 does not start until all of these hold. The
canonical copy lives in [`README.md`](README.md#plan-gate).

- [x] `python -m pytest` passes with no failures and no new skips.
- [x] `python -m pytest -m data` passes, including the store round-trip on the real run.
- [x] `python -m src.esci_images data/esci-s/esci.json.zst` exits zero.
- [x] `data/embeddings/rerank/` holds one float16 vector per successfully fetched image, and `len(store) == len(store.keys())` exactly.
- [x] `docs/results/image-embeddings.json` records end-to-end coverage against the measured ~77.5%, not 91.5%.
- [x] The semantic gate passes: top-1 ≥ 0.60 over a 600-candidate pool, against a measured 0.753.
- [x] `CLAUDE.md`'s Commands section lists the embedding invocation.

Then: Plan 4 — Recall. See [`../README.md`](../README.md) for the series.
