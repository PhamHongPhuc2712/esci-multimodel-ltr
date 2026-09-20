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
