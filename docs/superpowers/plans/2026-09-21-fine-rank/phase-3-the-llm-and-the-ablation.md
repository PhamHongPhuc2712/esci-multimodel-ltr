# Phase 3 — The LLM and the Ablation

**Plan 6 of 7 · Phase 3 of 3 · Tasks 5–6.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 2 gate](phase-2-the-cross-encoder.md#phase-2-gate) passes.

**Delivers:** `src/llm_rerank.py` and `src/fine_rank_report.py` — a cached
listwise reranker, and Ablation 6 with NDCG, latency and cost.

**Needs on disk:** everything Phases 1 and 2 built. Task 5 needs an API key and
is the second paid component in the project after Plan 4's Stage 0.

**Owns Review Focus items 2 and 4** (an ill-formed permutation, a cache key
that ignores the candidate list).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **A malformed LLM ranking falls back to the Stage 2 order, and the fallback is counted.**
- **Cache every call keyed on the query *and* the exact candidate window.**
- **Report latency as both single-query and batch-amortised**, labelled.
- **Do not invent prices.** Report tokens and seconds; fill `cost_usd` only from flags.
- **Every arm is scored on the same query set as the arms it is compared against**, with `n` recorded.
- **Never tune on test.** The test run is a frozen 2,000-query sample, behind `--final`.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## What was measured, and what it decides

`gpt-5.6-luna`, the model Plan 4 used for Stage 0:

| window | prompt tok | completion tok | of which reasoning | latency | well-formed |
|---|---|---|---|---|---|
| whole list (40 candidates) | 1,015 | 1,716 | 1,536 (**89%**) | 17.3 s | 1/1 |
| **top-10** | **386** | **387** | **339 (88%)** | 4.3 s each, **1.3 s/query at concurrency 4** | **8/8** |

Three things follow.

**The window pays for itself twice.** Restricting to the top-10 cuts completion
tokens 4.4x *and* wall-clock 4x, while the oracle headroom measurement says the
top-10 still holds two thirds of everything available (+0.1014 of +0.1481).

**The bill is reasoning, not ranking.** 88% of completion tokens are spent
thinking; the permutation itself is ~40 tokens. This is the same shape Plan 4
recorded for Stage 0, where a 100-token cap left *zero* room for the answer.
`MAX_COMPLETION_TOKENS` is therefore 2,000, not 200 — billing is on tokens
actually generated, so a generous cap costs nothing and a tight one silently
returns an empty string.

**Concurrency is the difference between feasible and not.** 4.3 s/query
sequentially is 4.9 h for fold 0; at concurrency 4 it is ~1.5 h. The test-split
sample of 2,000 queries is ~45 min.

Measured output was 8/8 well-formed, which is *better* than
`PROJECT_SPEC.md` §7.4 warns ("open LLMs often emit ill-formed listwise
output"). The validation is built anyway — one malformed response in 4,130
silently drops a candidate, and the failure is invisible without a check.

---

## Task 5: The LLM listwise reranker

**Files:**
- Create: `src/llm_rerank.py`
- Create: `tests/test_llm_rerank.py`

**Interfaces:**
- Consumes: `src.rerank_window.Window` (Task 2).
- Produces:
  - `src.llm_rerank.DEFAULT_MODEL: str` = `"gpt-5.6-luna"`
  - `src.llm_rerank.MAX_COMPLETION_TOKENS: int` = `2000`
  - `src.llm_rerank.DEFAULT_CONCURRENCY: int` = `4`
  - `src.llm_rerank.DEFAULT_CACHE: Path` = `Path("data/llm-rerank.json")`
  - `src.llm_rerank.PROMPT: str`
  - `src.llm_rerank.window_key(query: str, documents: Sequence[str]) -> str`
  - `src.llm_rerank.build_prompt(query: str, titles: Sequence[str]) -> str`
  - `src.llm_rerank.parse_permutation(text: str, n: int) -> list[int] | None`
  - `src.llm_rerank.RerankCache`
  - `src.llm_rerank.Usage` (frozen dataclass)
  - `src.llm_rerank.rerank_windows(windows, query_text, doc_text, *, call, cache=None, concurrency=DEFAULT_CONCURRENCY, progress=None) -> tuple[dict[str, list[str]], Usage]`
  - CLI: `python -m src.llm_rerank --split train --folds 0`

**Review Focus 2 lives in `parse_permutation`.** It accepts a response only if
the bracketed indices it finds are *exactly* a permutation of `1..n` — every
index present, none repeated, none out of range. Anything else returns `None`
and the caller keeps Stage 2's order for that query and increments
`n_fallback`. A response that drops `[7]` silently removes a candidate from the
ranking; one that repeats `[3]` silently gives it two ranks. Both parse fine
under a lenient reader.

**Review Focus 4 lives in `window_key`.** Plan 4's `RewriteCache` keys on the
raw query, which is right for a rewrite — the same query always rewrites the
same way. A listwise ranking depends on the query, the candidate set **and its
input order**, so the same key shape here returns a permutation computed for a
different list. The indices still parse, they just point at the wrong products.
`window_key` is a SHA-256 over the query and the ordered document ids.

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_rerank.py`:

```python
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
        calls.append(prompt)
        return "[2] > [1] > [3]", 1, 1, 1

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_llm_rerank.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.llm_rerank'`.

- [ ] **Step 3: Write the implementation**

Create `src/llm_rerank.py`:

```python
"""Stage 3(b): RankGPT-style listwise re-ranking of the Stage 2 window.

PROJECT_SPEC.md §4.3 asks for an LLM listwise variant beside the cross-encoder,
and §7.4 points at RankGPT's numbered-passage prompt. This is that, over the
top-K window rather than the whole candidate list.

Measured on gpt-5.6-luna, the model Plan 4 used for Stage 0:

    whole list (40)   1,015 prompt + 1,716 completion (1,536 reasoning)  17.3 s
    top-10               386 prompt +   387 completion (  339 reasoning)   4.3 s

So **88% of the bill is reasoning**, not the ~40-token permutation, which is
why MAX_COMPLETION_TOKENS is 2,000 rather than 200: billing is on tokens
actually generated, a generous cap costs nothing, and a tight one silently
returns an empty string. Plan 4 learned this the expensive way on Stage 0.

Two guards, both for failures that are invisible without them:

  * **Only an exact permutation of 1..n is accepted.** A response that drops an
    index silently removes a candidate from the ranking; one that repeats an
    index silently gives a product two ranks. Both parse fine under a lenient
    reader. Anything else falls back to the Stage 2 order for that query and
    increments n_fallback.
  * **The cache is keyed on the query AND the ordered candidate list.** Plan 4's
    RewriteCache keys on the raw query, which is correct for a rewrite - the
    same query always rewrites the same way. A listwise ranking depends on the
    list and its order too, so that key shape would return a permutation
    computed for different products, whose indices still parse.

`rerank_windows` takes a `call` callable, so every path is tested with no API
key and no network; the real client lives in __main__.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_CACHE = Path("data/llm-rerank.json")

# 88% of completion tokens are reasoning. A cap sized for the answer alone
# returns an empty string; billing is on what is generated, so this is free.
MAX_COMPLETION_TOKENS = 2000

# Measured: 4.3 s/query sequential, 1.3 s/query at 4 in flight - 4.9 h against
# 1.5 h for fold 0.
DEFAULT_CONCURRENCY = 4

# Titles only, capped. The window is already short; full descriptions would
# quadruple the prompt for a signal the cross-encoder arm covers.
TITLE_CHARS = 160

PROMPT = """Rank the {n} products by how well they satisfy the search query.

Query: {query}

{listing}

Reply with the identifiers in descending relevance, like [3] > [1] > [2], \
covering all {n}. No explanation."""

_INDEX = re.compile(r"\[(\d+)\]")

# (text, prompt_tokens, completion_tokens, reasoning_tokens)
Call = Callable[[str], tuple[str, int, int, int]]


def window_key(query: str, documents: Sequence[str]) -> str:
    """Stable key over the query and the ordered candidate list.

    sha256, not hash(): Python salts str.__hash__ per process, so a cache
    written by one run would miss entirely in the next.
    """
    payload = json.dumps([str(query), [str(d) for d in documents]], sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt(query: str, titles: Sequence[str]) -> str:
    """A RankGPT-style numbered listing, 1-based."""
    listing = "\n".join(
        f"[{i}] {str(t)[:TITLE_CHARS]}" for i, t in enumerate(titles, start=1)
    )
    return PROMPT.format(n=len(titles), query=query, listing=listing)


def parse_permutation(text: str, n: int) -> list[int] | None:
    """1-based indices in ranked order, or None if it is not a permutation.

    Exactness is the point: every index present, none repeated, none out of
    range. A lenient reader would accept a response that quietly loses a
    candidate.
    """
    if not isinstance(text, str) or n <= 0:
        return None
    found = [int(x) for x in _INDEX.findall(text)]
    if sorted(found) != list(range(1, n + 1)):
        return None
    return found


class RerankCache:
    """window_key -> permutation, persisted as one JSON object."""

    def __init__(self, path: Path = DEFAULT_CACHE) -> None:
        self.path = Path(path)
        self._entries: dict[str, list[int]] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError:
                # An optimisation, not the run: a truncated write from a killed
                # process costs API calls, not correctness.
                loaded = {}
            if isinstance(loaded, dict):
                self._entries = {
                    str(k): [int(i) for i in v]
                    for k, v in loaded.items()
                    if isinstance(v, list)
                }

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> list[int] | None:
        return self._entries.get(key)

    def set(self, key: str, permutation: Sequence[int]) -> None:
        self._entries[key] = [int(i) for i in permutation]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=0, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class Usage:
    """What the arm cost. Tokens and seconds, never an invented price."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    n_calls: int = 0
    n_cached: int = 0
    n_fallback: int = 0

    def per_query(self, n_queries: int) -> dict:
        n = max(n_queries, 1)
        return {
            "prompt_tokens": self.prompt_tokens / n,
            "completion_tokens": self.completion_tokens / n,
            "reasoning_tokens": self.reasoning_tokens / n,
            "seconds": self.seconds / n,
        }

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "seconds": self.seconds,
            "n_calls": self.n_calls,
            "n_cached": self.n_cached,
            "n_fallback": self.n_fallback,
        }


def rerank_windows(
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    call: Call,
    cache: RerankCache | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, list[str]], Usage]:
    """A new ordering per window, falling back to Stage 2's on any failure."""
    windows_ = list(windows_)
    if not windows_:
        return {}, Usage()

    totals = {"pt": 0, "ct": 0, "rt": 0, "calls": 0, "cached": 0, "fallback": 0}

    def one(w) -> tuple[str, list[str]]:
        documents = list(w.window)
        query = str(query_text[w.query_id])
        key = window_key(query, documents)

        if cache is not None:
            cached = cache.get(key)
            if cached is not None and sorted(cached) == list(
                range(1, len(documents) + 1)
            ):
                totals["cached"] += 1
                return w.query_id, [documents[i - 1] for i in cached]

        titles = [str(doc_text.get(d, "")) for d in documents]
        try:
            text, pt, ct, rt = call(build_prompt(query, titles))
        except Exception:
            # One transient failure must not lose the other 4,129 windows, and
            # the Stage 2 order is always a valid order.
            totals["fallback"] += 1
            totals["calls"] += 1
            return w.query_id, documents

        totals["pt"] += int(pt)
        totals["ct"] += int(ct)
        totals["rt"] += int(rt)
        totals["calls"] += 1

        permutation = parse_permutation(text, len(documents))
        if permutation is None:
            totals["fallback"] += 1
            return w.query_id, documents
        if cache is not None:
            cache.set(key, permutation)
        return w.query_id, [documents[i - 1] for i in permutation]

    started = time.time()
    out: dict[str, list[str]] = {}
    with cf.ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        for done, (query_id, order) in enumerate(pool.map(one, windows_), start=1):
            out[query_id] = order
            if progress is not None:
                progress(done, len(windows_))
    elapsed = time.time() - started

    if cache is not None:
        cache.save()

    return out, Usage(
        prompt_tokens=totals["pt"],
        completion_tokens=totals["ct"],
        reasoning_tokens=totals["rt"],
        seconds=elapsed,
        n_calls=totals["calls"],
        n_cached=totals["cached"],
        n_fallback=totals["fallback"],
    )


def openai_call(model: str = DEFAULT_MODEL, max_tokens: int = MAX_COMPLETION_TOKENS) -> Call:
    """The real client. `max_completion_tokens`, not `max_tokens`: gpt-5.x rejects the older name."""
    import openai

    client = openai.OpenAI()

    def call(prompt: str) -> tuple[str, int, int, int]:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        return (
            response.choices[0].message.content or "",
            int(usage.prompt_tokens),
            int(usage.completion_tokens),
            int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        )

    return call


def _main() -> int:
    import pandas as pd

    from src.cross_encoder import text_maps
    from src.rerank_window import DEFAULT_K, windows
    from src.stage2_scores import load_stage2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--folds", default="0", help="comma-separated, or 'all'")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    args = parser.parse_args()

    matrix = pd.read_parquet(args.features_dir / f"{args.split}.parquet")
    if args.folds != "all":
        wanted = [int(f) for f in args.folds.split(",")]
        matrix = matrix.loc[matrix["fold"].isin(wanted)]
    if args.sample is not None:
        keep = (
            matrix["query_id"].drop_duplicates().sample(n=args.sample, random_state=args.seed)
        )
        matrix = matrix.loc[matrix["query_id"].isin(set(keep))]

    scores = load_stage2(args.split)
    joined = matrix[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    ws = windows(joined, k=args.k)
    query_text, doc_text = text_maps(
        matrix, Path("data/combined/products.parquet"), Path("data/combined/judgements.parquet")
    )
    query_text = {str(k): v for k, v in query_text.items()}

    cache = RerankCache(args.cache)
    print(f"{len(ws):,} windows of up to {args.k}; {len(cache):,} already cached")

    def report(done: int, total: int) -> None:
        print(f"\r  {done:,}/{total:,}", end="", flush=True)

    _, usage = rerank_windows(
        ws,
        query_text,
        doc_text,
        call=openai_call(args.model),
        cache=cache,
        concurrency=args.concurrency,
        progress=report,
    )
    print()
    print(json.dumps(usage.to_dict(), indent=2))
    per = usage.per_query(len(ws))
    print(
        f"per query: {per['prompt_tokens']:.0f} prompt + {per['completion_tokens']:.0f} "
        f"completion ({per['reasoning_tokens']:.0f} reasoning), {per['seconds']:.2f}s"
    )
    print(f"{usage.n_fallback:,} windows fell back to the Stage 2 order")
    print(f"cache written to {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_llm_rerank.py -q`
Expected: PASS, 26 tests.

**Two corrections landed here during execution.**

*The cache test's fake was not a valid responder.* As written,
`test_a_cached_window_is_not_called_again` returns the fixed string
`"[2] > [1] > [3]"` for every window, including the two-document one — where
`parse_permutation` correctly rejects it, so that window is never cached and
*is* re-called. The test failed on its own premise, and the implementation was
right. The fake now reads the candidate count out of the prompt
(`int(prompt.split()[2])`) and answers with a permutation of that size.

*The cache was not resumable, though this plan twice says it is.* `save()` is
called once, after the `ThreadPoolExecutor` drains — so an interruption at 90%
of a two-hour paid run loses **every** answer. Caught 64 windows into the real
fold-0 run, which was stopped and restarted rather than gambling on it.
`rerank_windows` now takes `checkpoint_every` (default `CHECKPOINT_EVERY = 200`)
and saves from the consumer thread as results arrive; `RerankCache` guards
`set`/`save` with a `threading.Lock`, because `json.dumps` over a dict a worker
is mutating raises `dictionary changed size during iteration`; and `save()`
writes to a `.tmp` and renames, so a kill mid-write leaves the previous cache
rather than a truncated file. Two tests pin it: an interrupted run leaves its
answered windows on disk, and no `.tmp` survives a successful write.

- [ ] **Step 5: Warm the cache on fold 0**

Run: `python -m src.llm_rerank --split train --folds 0`

Expected: 4,130 windows, roughly **1.5 h at concurrency 4**, about 1.59M prompt
and 1.60M completion tokens. The run is resumable — every answered window is
cached, so an interruption costs only what was in flight.

Check the fallback count. Measured 8/8 well-formed on the pilot, so a fallback
rate above a few percent means the prompt or the parser has drifted, and a rate
near 100% means the model name or the token cap is wrong.

- [ ] **Step 6: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 7: Commit**

```bash
git add src/llm_rerank.py tests/test_llm_rerank.py
git commit -m "Add LLM listwise re-ranking with permutation validation and a window-keyed cache"
```

---

## Task 6: Ablation 6

**Files:**
- Create: `src/fine_rank_report.py`
- Create: `tests/test_fine_rank_report.py`
- Modify: `CLAUDE.md` (Commands section)
- Create: `docs/results/fine-rank.json` (written by the run)

**Interfaces:**
- Consumes: everything above, plus `src.rank_report.{qrels_from_frame, evaluate_arm, compare, format_table}` (Plan 5), `src.floor.random_floor`, `src.metrics.ndcg_per_query`.
- Produces:
  - `src.fine_rank_report.ARMS: tuple[str, ...]`
  - `src.fine_rank_report.TEST_SAMPLE: int` = `2000`
  - `src.fine_rank_report.ArmCost` (frozen dataclass: `latency`, `usage`, `cost`; serialises `cost` as `cost_usd`)
  - `src.fine_rank_report.cost_usd(usage, *, price_in=None, price_out=None) -> float | None`
  - `src.fine_rank_report.check_same_queries(arms) -> None`
  - CLI: `python -m src.fine_rank_report` and `--split test --final`

### The arms

| arm | what it is |
|---|---|
| `stage2` | Plan 5's LambdaMART alone — the baseline every comparison is paired against |
| `stage2+ce` | the `LambdaLoss` fine-tuned cross-encoder over the window |
| `stage2+ce_bce` | the `BinaryCrossEntropyLoss` fine-tune — does listwise still pay at Stage 3? |
| `stage2+ce_zeroshot` | the off-the-shelf backbone — what the fine-tune bought |
| `stage2+llm` | the LLM listwise reranker over the window |
| `stage2+ce+llm` | the cascade: cross-encoder first, LLM over its re-ordered window |

`stage2+ce+llm` answers whether the two arms are complementary or redundant,
which `PROJECT_SPEC.md` §6 does not ask for but §4.4's blend stage will need
from Plan 7.

### Cost is tokens, latency is two numbers

`cost_usd` returns `None` unless the operator passed both
`--price-per-mtok-in` and `--price-per-mtok-out`. Prices change and are not
measurable from here; the JSON records tokens and seconds, which are, and the
writeup multiplies by whatever rate applied on the day.

`check_same_queries` refuses to emit a comparison whose arms cover different
query sets. The LLM arm on test runs over a frozen 2,000-query sample, so every
other arm must be restricted to that same sample before any of them are
compared — otherwise Ablation 6 compares a 2,000-query LLM number against an
8,956-query cross-encoder number and calls the difference a reranker effect.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fine_rank_report.py`:

```python
import json

import pytest

from src.fine_rank_report import (
    ARMS,
    TEST_SAMPLE,
    check_same_queries,
    cost_usd,
)
from src.cross_encoder import Latency
from src.llm_rerank import Usage
from src.rank_report import evaluate_arm


def _arm(name, values):
    per = {str(i): v for i, v in enumerate(values)}
    floor = {str(i): 0.74 for i in range(len(values))}
    return evaluate_arm(
        name, per, floor, groups=("retrieval",), n_features=0,
        objective="rerank", best_iteration=0,
    )


# --- the arms ---------------------------------------------------------------

def test_the_baseline_arm_is_stage_2():
    assert ARMS[0] == "stage2"


def test_both_stage_3_variants_are_present():
    # PROJECT_SPEC.md §6 Ablation 6: coarse-only vs +cross-encoder vs +LLM.
    assert "stage2+ce" in ARMS
    assert "stage2+llm" in ARMS


def test_the_zero_shot_control_is_present():
    # Measured: every zero-shot cross-encoder loses to Stage 2, so this arm is
    # what shows the fine-tune earned its GPU hours.
    assert "stage2+ce_zeroshot" in ARMS


def test_the_cascade_arm_is_present():
    assert "stage2+ce+llm" in ARMS


# --- cost -------------------------------------------------------------------

def _usage():
    return Usage(prompt_tokens=1_000_000, completion_tokens=2_000_000, n_calls=100)


def test_cost_is_none_without_prices():
    # Prices change and are not measurable from here.
    assert cost_usd(_usage()) is None


def test_cost_is_none_with_only_one_price():
    assert cost_usd(_usage(), price_in=1.0) is None
    assert cost_usd(_usage(), price_out=1.0) is None


def test_cost_uses_both_rates_per_million_tokens():
    assert cost_usd(_usage(), price_in=2.0, price_out=8.0) == pytest.approx(
        2.0 * 1 + 8.0 * 2
    )


def test_cost_of_no_tokens_is_zero():
    assert cost_usd(Usage(), price_in=1.0, price_out=1.0) == 0.0


# --- comparability ----------------------------------------------------------

def test_arms_over_the_same_queries_pass():
    check_same_queries([_arm("a", [0.8, 0.9]), _arm("b", [0.7, 0.6])])


def test_arms_over_different_queries_are_refused():
    # A 2,000-query LLM number against an 8,956-query cross-encoder number is
    # not a reranker effect.
    a = _arm("a", [0.8, 0.9])
    b = _arm("b", [0.8, 0.9, 0.7])
    with pytest.raises(ValueError, match="same queries"):
        check_same_queries([a, b])


def test_checking_one_arm_is_fine():
    check_same_queries([_arm("a", [0.8])])


def test_checking_nothing_is_fine():
    check_same_queries([])


def test_the_test_sample_is_frozen_and_documented():
    # All 8,956 test queries through the LLM is ~3.3 h and real money.
    assert TEST_SAMPLE == 2000


# --- serialisation ----------------------------------------------------------

def test_latency_and_usage_both_reach_the_payload():
    payload = {
        "latency": Latency(single_ms=40.0, batched_ms=8.0, n_queries=50, n_pairs=500).to_dict(),
        "usage": Usage(prompt_tokens=10, completion_tokens=20).to_dict(),
        "cost_usd": None,
    }
    assert json.loads(json.dumps(payload))["latency"]["single_ms"] == 40.0
    assert payload["latency"]["batched_ms"] == 8.0
    assert payload["usage"]["completion_tokens"] == 20
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_fine_rank_report.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'src.fine_rank_report'`.

- [ ] **Step 3: Write the implementation**

Create `src/fine_rank_report.py`:

```python
"""Ablation 6: coarse-only vs. +cross-encoder vs. +LLM listwise.

PROJECT_SPEC.md §6 asks for NDCG, latency and cost across the three arms. This
module produces all three, plus two context arms: the zero-shot cross-encoder
(what the fine-tune bought) and the cascade (whether the two rerankers are
complementary, which Plan 7's blend stage will need).

Three things the report refuses to do:

  * **Quote one latency.** The cross-encoder scores 256 pairs a forward pass
    and the LLM cannot batch at all, so a single amortised figure flatters the
    cross-encoder by 3-5x on top of the real 20-600x gap. Both are recorded.
  * **Invent a price.** Tokens and seconds are measurable from here; dollars
    are not. `cost_usd` stays None unless the operator passes both rates.
  * **Compare arms over different query sets.** The LLM arm on test runs over a
    frozen 2,000-query sample; every other arm is restricted to that sample
    before anything is compared.

Selection happens on fold 0. `--split test` requires `--final`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.cross_encoder import Latency
from src.llm_rerank import Usage
from src.rank_report import compare, evaluate_arm, format_table, qrels_from_frame

DEFAULT_OUT = Path("docs/results/fine-rank.json")

# PROJECT_SPEC.md §5. Plan 5 already matched it at 0.8579 [0.8551, 0.8611].
ESCI_BASELINE = 0.8562

# All 8,956 test queries through the LLM is ~3.3 h and real money, so the
# three-way test comparison runs on a frozen sample. Every arm uses the same
# one, and `n` travels with every number.
TEST_SAMPLE = 2000

ARMS: tuple[str, ...] = (
    "stage2",
    "stage2+ce",
    "stage2+ce_bce",
    "stage2+ce_zeroshot",
    "stage2+llm",
    "stage2+ce+llm",
)


@dataclass(frozen=True)
class ArmCost:
    latency: Latency | None = None
    usage: Usage | None = None
    cost: float | None = None

    def to_dict(self) -> dict:
        return {
            "latency": self.latency.to_dict() if self.latency else None,
            "usage": self.usage.to_dict() if self.usage else None,
            "cost_usd": self.cost,
        }


def cost_usd(
    usage: Usage, *, price_in: float | None = None, price_out: float | None = None
) -> float | None:
    """Dollars, or None when either rate is unknown.

    Rates are per million tokens. Prices change and are not measurable from
    here; a fabricated number in a results file outlives the day it was right.
    """
    if price_in is None or price_out is None:
        return None
    return (
        usage.prompt_tokens / 1e6 * price_in
        + usage.completion_tokens / 1e6 * price_out
    )


def check_same_queries(arms: Sequence) -> None:
    """Refuse a table whose arms cover different query sets."""
    if len(arms) < 2:
        return
    reference = set(arms[0].per_query)
    for arm in arms[1:]:
        if set(arm.per_query) != reference:
            raise ValueError(
                f"{arm.name!r} covers {len(arm.per_query):,} queries but "
                f"{arms[0].name!r} covers {len(reference):,}; arms must be "
                "scored on the same queries or the difference is not a "
                "reranker effect"
            )


def _main() -> int:
    import pandas as pd

    from src.cross_encoder import (
        DEFAULT_BACKBONE,
        DEFAULT_MODEL_DIR,
        text_maps,
        load_reranker,
        measure_latency,
        rerank,
    )
    from src.floor import random_floor
    from src.llm_rerank import DEFAULT_CACHE, RerankCache, openai_call, rerank_windows
    from src.metrics import ndcg_per_query
    from src.ranker import REPORT_FOLD
    from src.rerank_window import DEFAULT_K, Window, spliced_run, stage2_run, windows
    from src.stage2_scores import load_stage2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--price-per-mtok-in", type=float, default=None)
    parser.add_argument("--price-per-mtok-out", type=float, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--final", action="store_true",
                        help="required with --split test")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        raise SystemExit(
            "refusing to touch the test split without --final. K, the loss, "
            "the backbone and the prompt are all chosen on folds 1-4."
        )

    matrix = pd.read_parquet(args.features_dir / f"{args.split}.parquet")
    if args.split == "train":
        matrix = matrix.loc[matrix["fold"] == REPORT_FOLD]
        label = f"fold {REPORT_FOLD}"
    else:
        keep = matrix["query_id"].drop_duplicates().sample(
            n=TEST_SAMPLE, random_state=args.seed
        )
        matrix = matrix.loc[matrix["query_id"].isin(set(keep))]
        label = f"test sample of {TEST_SAMPLE:,}"
    matrix = matrix.reset_index(drop=True)

    scores = load_stage2(args.split)
    joined = matrix[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    ws = windows(joined, k=args.k)
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    print(f"{label}: {len(matrix):,} judgements over {len(qrels):,} queries; "
          f"floor {floor.mean:.4f}")

    query_text, doc_text = text_maps(
        matrix,
        Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}

    results: list = []
    costs: dict[str, ArmCost] = {}

    def record(name: str, run: dict, cost: ArmCost) -> None:
        per_query = ndcg_per_query(run, qrels)
        arm = evaluate_arm(
            name, per_query, floor.per_query, groups=("retrieval",),
            n_features=0, objective="rerank", best_iteration=0, seed=args.seed,
        )
        results.append(arm)
        costs[name] = cost
        print(f"  {name:20s} {arm.ndcg.point:.4f}")

    record("stage2", stage2_run(ws), ArmCost())

    ce_orderings: dict[str, list[str]] = {}
    for name, path in [
        ("stage2+ce", args.model_dir / "lambda"),
        ("stage2+ce_bce", args.model_dir / "bce"),
        ("stage2+ce_zeroshot", DEFAULT_BACKBONE),
    ]:
        model = load_reranker(path)
        orderings = rerank(model, ws, query_text, doc_text)
        if name == "stage2+ce":
            ce_orderings = orderings
        record(
            name,
            spliced_run(ws, orderings),
            ArmCost(latency=measure_latency(model, ws, query_text, doc_text, n_queries=50)),
        )
        del model

    cache = RerankCache(args.cache)
    llm_orderings, usage = rerank_windows(
        ws, query_text, doc_text, call=openai_call(), cache=cache
    )
    record(
        "stage2+llm",
        spliced_run(ws, llm_orderings),
        ArmCost(
            latency=Latency(
                single_ms=usage.seconds * 1000 / max(len(ws), 1),
                batched_ms=usage.seconds * 1000 / max(len(ws), 1),
                n_queries=len(ws),
                n_pairs=sum(len(w.window) for w in ws),
            ),
            usage=usage,
            cost=cost_usd(usage, price_in=args.price_per_mtok_in,
                          price_out=args.price_per_mtok_out),
        ),
    )
    print(f"    {usage.n_fallback:,} windows fell back to the Stage 2 order")

    # The cascade: the LLM re-ranks the cross-encoder's window order.
    cascade_windows = [
        Window(query_id=w.query_id,
               window=tuple(ce_orderings.get(w.query_id, w.window)),
               tail=w.tail)
        for w in ws
    ]
    cascade_orderings, cascade_usage = rerank_windows(
        cascade_windows, query_text, doc_text, call=openai_call(), cache=cache
    )
    record(
        "stage2+ce+llm",
        spliced_run(cascade_windows, cascade_orderings),
        ArmCost(usage=cascade_usage,
                cost=cost_usd(cascade_usage, price_in=args.price_per_mtok_in,
                              price_out=args.price_per_mtok_out)),
    )

    check_same_queries(results)
    print()
    print(format_table(results))

    baseline = results[0]
    comparisons = [compare(arm, baseline, seed=args.seed) for arm in results[1:]]
    print("\n  -- Ablation 6: each arm against coarse-only --")
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:20s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]  "
              f"{'significant' if row['significant'] else 'ties'}")

    print("\n  -- latency and cost --")
    for arm in results:
        c = costs[arm.name]
        if c.latency:
            print(f"  {arm.name:20s} single {c.latency.single_ms:8.1f} ms  "
                  f"batched {c.latency.batched_ms:7.1f} ms")
        if c.usage:
            per = c.usage.per_query(len(ws))
            print(f"  {'':20s} {per['prompt_tokens']:.0f} prompt + "
                  f"{per['completion_tokens']:.0f} completion tok/query"
                  + (f", ${c.cost:.2f} total" if c.cost is not None else
                     ", cost_usd null (no rates given)"))

    payload = {
        "split": args.split,
        "scope": label,
        "k": args.k,
        "n_queries": len(qrels),
        "n_judgements": len(matrix),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "esci_baseline_target": ESCI_BASELINE,
        "arms": [arm.to_dict() | {"cost": costs[arm.name].to_dict()} for arm in results],
        "ablation_6": comparisons,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_fine_rank_report.py -q`
Expected: PASS, 15 tests. **Measured 14** — the block above defines fourteen
tests, so the 15 is a miscount in this plan, not a missing test.

- [ ] **Step 5: Produce the fold-0 Ablation 6 table**

Run: `python -m src.fine_rank_report`

Expected: `stage2` reproduces **0.8519**; `stage2+ce_zeroshot` lands near the
measured **0.8396** for this backbone, i.e. *below* Stage 2; the fine-tuned
arms are the open question. The LLM arm should report a near-zero fallback
count if Task 5's cache is warm.

Sanity checks: if `stage2+ce` and `stage2+ce_zeroshot` are identical, the
fine-tuned model did not load. If every arm equals `stage2`, the splice is
returning the Stage 2 order — check `rerank` is returning orderings and that
`spliced_run` is receiving them.

- [ ] **Step 6: Write the data-marked test**

Append to `tests/test_fine_rank_report.py`:

```python
@pytest.mark.data
def test_the_committed_report_carries_every_ablation_6_column():
    payload = json.loads(open("docs/results/fine-rank.json").read())

    names = {arm["name"] for arm in payload["arms"]}
    assert {"stage2", "stage2+ce", "stage2+llm"} <= names

    # Every arm carries the floor it beats. PROJECT_SPEC.md §2.
    assert 0.70 < payload["floor"]["mean"] < 0.80
    for arm in payload["arms"]:
        assert set(arm["lift_over_floor"]) == {"point", "low", "high"}

    # §6 asks for NDCG, latency AND cost.
    by_name = {arm["name"]: arm for arm in payload["arms"]}
    assert by_name["stage2+ce"]["cost"]["latency"]["single_ms"] > 0
    assert by_name["stage2+ce"]["cost"]["latency"]["batched_ms"] > 0
    assert by_name["stage2+llm"]["cost"]["usage"]["completion_tokens"] > 0

    # The baseline must reproduce Plan 5.
    assert by_name["stage2"]["ndcg"]["point"] == pytest.approx(0.8519, abs=0.002)

    # Ablation 6 compares every arm against coarse-only.
    assert {row["baseline"] for row in payload["ablation_6"]} == {"stage2"}
```

Run: `python -m pytest tests/test_fine_rank_report.py -m data -q`
Expected: PASS.

- [ ] **Step 7: Freeze the configuration, then run test exactly once**

Write down `K`, the backbone, the loss, the epoch count and the prompt before
running. Nothing changes after this command.

Run: `python -m src.fine_rank_report --split test --final --out docs/results/fine-rank-test.json`

Expected: the 2,000-query frozen sample, roughly 45 minutes of LLM time at
concurrency 4 plus the cascade's second pass, a floor near 0.7468 and every arm
scored on the same 2,000 queries. **An arm that ties or loses is reported as
measured** — Plan 5's Ablation 7 lost to its control and was published as a loss.

- [ ] **Step 8: Update `CLAUDE.md`'s Commands section**

Add to the fenced command block, after the coarse-rank section:

```bash
# Fine rank (Plan 6). The LLM arm is the second paid component after Stage 0.
python -m src.stage2_scores --split train      # -> data/features/stage2-train.parquet
python -m src.stage2_scores --split test
python -m src.cross_encoder --loss lambda      # ~29 min GPU -> models/cross-encoder/lambda/
python -m src.cross_encoder --loss bce         # ~11 min GPU
python -m src.llm_rerank --split train --folds 0   # ~1.5 h at concurrency 4, cached
python -m src.fine_rank_report                 # Ablation 6 on fold 0
python -m src.fine_rank_report --split test --final --out docs/results/fine-rank-test.json
```

- [ ] **Step 9: Run the whole fast suite**

Run: `python -m pytest -q`
Expected: PASS with no new failures and no new skips.

- [ ] **Step 10: Commit**

```bash
git add src/fine_rank_report.py tests/test_fine_rank_report.py CLAUDE.md docs/results/fine-rank.json docs/results/fine-rank-test.json
git commit -m "Report Ablation 6 for the cross-encoder and LLM listwise rerankers"
```

- [ ] **Step 11: Mark the plan gate**

Tick the [Plan Gate](README.md#plan-gate) boxes in `README.md`, and update the
series entry for Plan 6 in `docs/superpowers/plans/README.md` with the landed
numbers in the style of Plans 1–5: the headline against the floor and the
0.8562 target, and Ablation 6's outcome including any arm that ties or loses.

```bash
git add docs/superpowers/plans/
git commit -m "Mark the fine rank plan gate as passed"
```

---

## Phase 3 Gate — and the Plan Gate

This phase's gate *is* the [Plan Gate](README.md#plan-gate). Check it there;
every box must be ticked before Plan 7 is written.

The three that are easiest to skip:

- [ ] Both latencies are recorded for every arm that has one, and they differ.
- [ ] The LLM arm's fallback count is in the JSON, not just in the console.
- [ ] Every arm in the test table was scored on the same 2,000 queries.
