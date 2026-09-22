import json

import pytest

from src.llm_rerank import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    MAX_COMPLETION_TOKENS,
    PROMPT,
    RerankCache,
    Usage,
    build_prompt,
    parse_permutation,
    rerank_windows,
    window_key,
)
from src.rerank_window import Window


# --- Review Focus 2: only an exact permutation is accepted ------------------

def test_a_well_formed_permutation_parses():
    assert parse_permutation("[3] > [1] > [2]", 3) == [3, 1, 2]


def test_surrounding_prose_is_tolerated():
    # The model was told not to explain, but a stray "Here you go:" must not
    # cost the whole query.
    assert parse_permutation("Sure! [2] > [1]", 2) == [2, 1]


def test_a_missing_index_is_rejected():
    # Silently drops a candidate from the ranking.
    assert parse_permutation("[1] > [2]", 3) is None


def test_a_repeated_index_is_rejected():
    # Silently gives one product two ranks.
    assert parse_permutation("[1] > [1] > [2]", 3) is None


def test_an_out_of_range_index_is_rejected():
    assert parse_permutation("[1] > [2] > [9]", 3) is None


def test_a_zero_index_is_rejected():
    # The prompt is 1-based; a 0 means the model counted differently.
    assert parse_permutation("[0] > [1] > [2]", 3) is None


def test_an_empty_response_is_rejected():
    assert parse_permutation("", 3) is None
    assert parse_permutation("I cannot help with that.", 3) is None


def test_a_single_candidate_window_parses():
    assert parse_permutation("[1]", 1) == [1]


# --- Review Focus 4: the cache key covers the candidate list ----------------

def test_the_key_changes_with_the_candidate_set():
    assert window_key("shoes", ["a", "b"]) != window_key("shoes", ["a", "c"])


def test_the_key_changes_with_the_candidate_order():
    # A listwise ranking depends on the input order, so two orders are two
    # different questions with two different answers.
    assert window_key("shoes", ["a", "b"]) != window_key("shoes", ["b", "a"])


def test_the_key_changes_with_the_query():
    assert window_key("shoes", ["a", "b"]) != window_key("boots", ["a", "b"])


def test_the_key_is_stable_across_processes():
    # sha256, not hash(): Python salts str.__hash__ per process.
    assert window_key("shoes", ["a", "b"]) == window_key("shoes", ["a", "b"])
    assert len(window_key("shoes", ["a", "b"])) == 64


# --- the cache --------------------------------------------------------------

def test_the_cache_round_trips(tmp_path):
    cache = RerankCache(tmp_path / "c.json")
    cache.set("k", [2, 1])
    cache.save()
    assert RerankCache(tmp_path / "c.json").get("k") == [2, 1]


def test_a_corrupt_cache_is_ignored_rather_than_fatal(tmp_path):
    # The cache is an optimisation; a truncated write from a killed run costs
    # API calls, not the run.
    path = tmp_path / "c.json"
    path.write_text("{not json")
    assert len(RerankCache(path)) == 0


def test_the_cache_is_empty_when_absent(tmp_path):
    assert len(RerankCache(tmp_path / "missing.json")) == 0


# --- the prompt -------------------------------------------------------------

def test_the_prompt_numbers_candidates_from_one():
    prompt = build_prompt("red shoes", ["Shoe A", "Shoe B"])
    assert "[1] Shoe A" in prompt
    assert "[2] Shoe B" in prompt
    assert "red shoes" in prompt


def test_the_prompt_asks_for_every_candidate():
    assert "2" in build_prompt("q", ["a", "b"])


def test_the_completion_budget_leaves_room_for_reasoning():
    # Measured: 339 of 387 completion tokens are reasoning on a top-10 window,
    # and 1,536 of 1,716 on a 40-candidate one. Plan 4's 100-token cap left
    # zero room for the answer and returned empty strings.
    assert MAX_COMPLETION_TOKENS >= 1000


def test_the_defaults_are_the_measured_ones():
    assert DEFAULT_MODEL == "gpt-5.6-luna"
    assert DEFAULT_CONCURRENCY == 4
    assert "{query}" in PROMPT or "Query:" in PROMPT


# --- rerank_windows ---------------------------------------------------------

def _windows():
    return [
        Window(query_id="1", window=("a", "b", "c"), tail=("z",)),
        Window(query_id="2", window=("x", "y"), tail=()),
    ]


def _maps():
    return {"1": "red shoes", "2": "blue hat"}, {k: f"doc {k}" for k in "abcxyz"}


def test_rerank_windows_applies_the_returned_order():
    q, d = _maps()
    out, usage = rerank_windows(
        _windows(), q, d, call=lambda prompt: ("[3] > [1] > [2]", 10, 20, 15)
    )
    assert out["1"] == ["c", "a", "b"]
    assert usage.n_calls == 2


def test_a_malformed_response_falls_back_and_is_counted():
    q, d = _maps()
    out, usage = rerank_windows(
        _windows(), q, d, call=lambda prompt: ("nonsense", 10, 20, 15)
    )
    # Fall back to the window's own order, which spliced_run treats as a no-op.
    assert out["1"] == ["a", "b", "c"]
    assert usage.n_fallback == 2


def test_an_api_failure_falls_back_rather_than_losing_the_run():
    q, d = _maps()

    def boom(prompt):
        raise RuntimeError("503")

    out, usage = rerank_windows(_windows(), q, d, call=boom)
    assert out["1"] == ["a", "b", "c"]
    assert usage.n_fallback == 2


def test_usage_accumulates_tokens():
    q, d = _maps()
    _, usage = rerank_windows(
        _windows(), q, d, call=lambda prompt: ("[1]" if False else "[2] > [1]", 100, 200, 150)
    )
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 400
    assert usage.reasoning_tokens == 300


def test_a_cached_window_is_not_called_again(tmp_path):
    q, d = _maps()
    cache = RerankCache(tmp_path / "c.json")
    calls = []

    def call(prompt):
        # Sized to the window it was asked about: a fixed 3-index reply would
        # be rejected for the 2-document window, so that one would never be
        # cached and this test would be measuring the parser instead.
        calls.append(prompt)
        n = int(prompt.split()[2])
        order = [2, 1, *range(3, n + 1)] if n >= 2 else [1]
        return " > ".join(f"[{i}]" for i in order), 1, 1, 1

    rerank_windows(_windows(), q, d, call=call, cache=cache)
    first = len(calls)
    out, usage = rerank_windows(_windows(), q, d, call=call, cache=cache)
    assert len(calls) == first          # nothing re-issued
    assert usage.n_cached == 2
    assert out["1"] == ["b", "a", "c"]  # the cached order, reapplied


def test_the_result_is_always_a_permutation_of_the_window():
    q, d = _maps()
    out, _ = rerank_windows(
        _windows(), q, d, call=lambda prompt: ("[2] > [1] > [3]", 1, 1, 1)
    )
    for w in _windows():
        assert sorted(out[w.query_id]) == sorted(w.window)


def test_reranking_nothing_is_nothing():
    out, usage = rerank_windows([], {}, {}, call=lambda p: ("", 0, 0, 0))
    assert out == {}
    assert usage.n_calls == 0
