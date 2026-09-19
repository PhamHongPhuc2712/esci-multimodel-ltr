"""Image URL handling for ESCI-S records.

ESCI-S was scraped in late 2022 / early 2023 and its image URLs carry two traps
that both fail silently if you read the records naively. See `normalize_url` and
`image_url` for the details; `python -m src.esci_images <file.zst>` re-runs the
live resolution check that justifies them.
"""

from __future__ import annotations

import re
from typing import Any

# ~50% of ESCI-S image URLs point into a CDN A/B bucket that Amazon retired after
# the scrape: .../images/W/WEBP_402378-T1/images/I/<id><transform>.jpg
# Those return HTTP 400 (not 404 - the product is fine, the path is not).
# Measured on 600 fresh URLs: 0% resolve as-is, 100% once the segment is dropped.
# URLs without the segment are unaffected (294/294 control URLs unchanged).
_W_BUCKET = re.compile(r"/images/W/[^/]+/images/I/")


def normalize_url(url: str) -> str:
    """Drop the retired CDN bucket segment from an ESCI-S image URL.

    A no-op for URLs that don't carry one.
    """
    return _W_BUCKET.sub("/images/I/", url)


def image_url(record: dict[str, Any]) -> str | None:
    """Return the normalized main image URL for an ESCI-S record, if it has one.

    Products keep the URL in `image`, books in `img`. Reading only `image` drops
    every book, which is ~7% of the corpus.
    """
    raw = record.get("image") or record.get("img")
    return normalize_url(raw) if raw else None


def is_scrape_error(record: dict[str, Any]) -> bool:
    """True for ESCI-S rows that are scrape failures rather than products.

    These carry only asin/locale/error/template and are ~3.7% of the file.
    """
    return record.get("type") == "error"


def _main() -> int:
    """Re-run the resolution gate against a live random sample."""
    import argparse
    import concurrent.futures as cf
    import io
    import json
    import random
    import urllib.error
    import urllib.request

    import zstandard

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="esci.json.zst (a truncated prefix is fine)")
    ap.add_argument("-n", "--sample", type=int, default=500)
    ap.add_argument("--locale", default="us")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="fail below this resolution %% (the project's go/no-go gate)",
    )
    args = ap.parse_args()

    urls: list[str] = []
    tail = b""
    with open(args.path, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        try:
            while chunk := reader.read(1 << 22):
                lines = (tail + chunk).split(b"\n")
                tail = lines.pop()
                for line in lines:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if is_scrape_error(rec) or rec.get("locale") != args.locale:
                        continue
                    if url := image_url(rec):
                        urls.append(url)
        except zstandard.ZstdError:
            pass  # expected when the input is a truncated prefix

    if not urls:
        print("no image URLs found")
        return 1

    random.Random(args.seed).shuffle(urls)
    sample = urls[: args.sample]

    headers = {"User-Agent": "Mozilla/5.0 (compatible; esci-multimodel-ltr/0.1)"}

    def fetch(url: str) -> bool:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as resp:
                return resp.status == 200 and len(resp.read()) > 0
        except (urllib.error.URLError, OSError):
            return False

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        ok = sum(pool.map(fetch, sample))

    rate = 100 * ok / len(sample)
    print(f"{ok}/{len(sample)} resolved = {rate:.1f}%  (threshold {args.threshold:.0f}%)")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(_main())
