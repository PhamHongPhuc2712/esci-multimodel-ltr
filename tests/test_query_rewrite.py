import json

import pytest

from src.query_rewrite import (
    MAX_EXPANSION,
    RewriteCache,
    rewrite_queries,
    validate_rewrite,
)


def _call(responses):
    """A fake LLM: returns the scripted response for each query."""
    seen = []

    def call(query: str) -> str:
        seen.append(query)
        return responses[query]

    return call, seen


# --- Review Focus 5: a rewrite that silently becomes a different query ------

def test_an_empty_rewrite_falls_back_to_the_raw_query():
    assert validate_rewrite("sony wh-1000xm4", "") == "sony wh-1000xm4"
    assert validate_rewrite("sony wh-1000xm4", "   ") == "sony wh-1000xm4"


def test_a_refusal_falls_back_to_the_raw_query():
    refusal = "I'm sorry, but I can't help with that request."
    assert validate_rewrite("sony wh-1000xm4", refusal) == "sony wh-1000xm4"


def test_a_rewrite_that_drops_an_alphanumeric_model_number_is_rejected():
    # "wh-1000xm4" is the entire query. A paraphrase that loses it retrieves a
    # different product, and nothing errors - Ablation 1 would report the
    # damage as a property of query rewriting.
    assert (
        validate_rewrite("sony wh-1000xm4", "sony wireless headphones")
        == "sony wh-1000xm4"
    )


def test_a_rewrite_that_keeps_the_model_number_is_accepted():
    rewritten = "sony wh-1000xm4 wireless noise cancelling headphones"
    assert validate_rewrite("sony wh-1000xm4", rewritten) == rewritten


def test_an_identifier_respaced_by_the_rewrite_is_still_accepted():
    # Measured: "sony a7iii" -> "Sony Alpha a7 III mirrorless camera". The
    # model number survived; only its spacing moved. Comparing raw text would
    # reject a good rewrite, so both sides are flattened before the check.
    rewritten = "Sony Alpha a7 III mirrorless camera"
    assert validate_rewrite("sony a7iii", rewritten) == rewritten


def test_flattening_does_not_let_a_dropped_identifier_through():
    # The guard that matters: flattening compares content, not punctuation, and
    # must still catch a rewrite that loses the model number entirely.
    assert (
        validate_rewrite("sony wh-1000xm4", "sony wireless headphones")
        == "sony wh-1000xm4"
    )


def test_an_empty_completion_falls_back():
    # A reasoning model whose token budget is consumed by reasoning returns an
    # empty string rather than raising. That must read as "no rewrite", not as
    # an empty query that matches the whole corpus.
    assert validate_rewrite("automotive battery block", "") == (
        "automotive battery block"
    )


def test_a_runaway_expansion_is_rejected():
    # An LLM that starts listing everything it can think of turns a query into
    # a bag of words that matches the whole corpus.
    raw = "coffee mug"
    runaway = " ".join(["cup"] * 200)
    assert validate_rewrite(raw, runaway) == raw


def test_the_expansion_cap_is_a_multiple_of_the_raw_length():
    assert MAX_EXPANSION == 4


def test_a_json_wrapped_response_is_unwrapped():
    # Models asked for plain text sometimes answer with JSON anyway.
    wrapped = '{"query": "red running shoes for men"}'
    assert validate_rewrite("red shoes", wrapped) == "red running shoes for men"


def test_a_rewrite_is_stripped_of_surrounding_quotes_and_whitespace():
    assert validate_rewrite("red shoes", '  "red running shoes"  ') == (
        "red running shoes"
    )


# --- the cache --------------------------------------------------------------

def test_the_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "rewrites.json"
    cache = RewriteCache(path)
    cache.set("red shoes", "red running shoes")
    cache.save()

    reopened = RewriteCache(path)
    assert reopened.get("red shoes") == "red running shoes"
    assert len(reopened) == 1


def test_a_missing_cache_file_is_an_empty_cache(tmp_path):
    assert len(RewriteCache(tmp_path / "absent.json")) == 0


def test_a_corrupt_cache_file_does_not_take_the_run_down(tmp_path):
    # The cache is an optimisation. A truncated write from a killed run must
    # cost API calls, not the run.
    path = tmp_path / "rewrites.json"
    path.write_text("{not json", encoding="utf-8")
    assert len(RewriteCache(path)) == 0


def test_the_cache_is_readable_json(tmp_path):
    path = tmp_path / "rewrites.json"
    cache = RewriteCache(path)
    cache.set("a", "b")
    cache.save()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": "b"}


# --- the run ----------------------------------------------------------------

def test_every_query_gets_a_rewrite():
    call, _ = _call({"red shoes": "red running shoes"})
    assert rewrite_queries(["red shoes"], call=call) == {
        "red shoes": "red running shoes"
    }


def test_a_cached_query_is_not_called_again(tmp_path):
    cache = RewriteCache(tmp_path / "c.json")
    cache.set("red shoes", "red running shoes")
    call, seen = _call({})
    result = rewrite_queries(["red shoes"], call=call, cache=cache)
    assert seen == []
    assert result["red shoes"] == "red running shoes"


def test_a_duplicate_query_is_called_once():
    call, seen = _call({"red shoes": "red running shoes"})
    rewrite_queries(["red shoes", "red shoes"], call=call)
    assert seen == ["red shoes"]


def test_an_api_failure_falls_back_to_the_raw_query():
    # One failed call must not lose the other 20,887 rewrites.
    def call(query):
        raise RuntimeError("503 from the API")

    assert rewrite_queries(["red shoes"], call=call) == {"red shoes": "red shoes"}


def test_the_result_covers_every_query_even_when_all_calls_fail():
    def call(query):
        raise RuntimeError("down")

    result = rewrite_queries(["a", "b"], call=call)
    assert result == {"a": "a", "b": "b"}
