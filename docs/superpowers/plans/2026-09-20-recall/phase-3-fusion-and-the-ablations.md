# Phase 3 — Fusion and the Ablations

**Plan 4 of 7 · Phase 3 of 3 · Tasks 5–7.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into. Do not start before the
[Phase 2 gate](phase-2-the-learned-channels.md#phase-2-gate) passes.

**Delivers:** `src/rrf.py`, `src/query_rewrite.py` and `src/recall_report.py` —
Reciprocal Rank Fusion, Stage 0 query rewriting with a cache, and the two
ablations this plan owns reported with bootstrap CIs.

**Needs on disk:** everything Phases 1 and 2 built. Task 6 needs an API key
and is the only part of the project that costs money.

**Owns Review Focus items 4 and 5** (RRF treating absence as a bottom rank, a
rewritten query that silently becomes a different query).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Fusion sums only over channels that actually returned the product.** The image channel covers 77.50%; absence is a scrape artefact, not evidence.
- **Never tune on test.** The RRF constant and `k` are chosen on the frozen validation folds; test is measured once, at the end.
- **Recall's baseline is BM25 R@100 = 0.5018**, not the NDCG random floor.
- **Cache every LLM rewrite to disk**, keyed by the query, so a re-run is free and the ablation is reproducible.
- **A method that ties is a reportable result.** Do not tune until the fusion wins.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 5: Reciprocal Rank Fusion

**Files:**
- Create: `src/rrf.py`
- Create: `tests/test_rrf.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `src.rrf.DEFAULT_K: int` (60)
  - `src.rrf.fuse(ranked_lists, *, k=DEFAULT_K, weights=None, limit=None) -> list[str]`
  - `src.rrf.fuse_batches(channel_results, *, k=DEFAULT_K, weights=None, limit=None) -> list[list[str]]`

RRF scores a document as `Σ_channels w_c / (k + rank_c)`, summing **only over
the channels that returned it**. That is the whole of Review Focus 4: a
product with no image embedding is absent from the image channel, and absence
must contribute nothing rather than contribute a worst-possible rank. The two
are not the same — a worst-possible rank is still a vote.

`k = 60` is the constant from the original Cormack et al. RRF paper and the
usual default. It is a tunable, and Task 7 tunes it on the validation folds.

- [x] **Step 1: Write the failing test**

Create `tests/test_rrf.py`:

```python
import pytest

from src.rrf import DEFAULT_K, fuse, fuse_batches


# --- the happy path ---------------------------------------------------------

def test_a_document_ranked_first_everywhere_wins():
    assert fuse([["a", "b"], ["a", "c"]])[0] == "a"


def test_agreement_across_channels_beats_a_single_strong_hit():
    # "b" is second in both channels; "a" is first in one and absent from the
    # other. Consensus is the entire point of RRF.
    fused = fuse([["a", "b"], ["b", "c"]], k=1)
    assert fused[0] == "b"


def test_every_document_from_every_channel_appears():
    assert set(fuse([["a"], ["b"], ["c"]])) == {"a", "b", "c"}


def test_the_result_has_no_duplicates():
    assert fuse([["a", "b"], ["a", "b"]]) == ["a", "b"]


def test_fusing_one_channel_preserves_its_order():
    assert fuse([["a", "b", "c"]]) == ["a", "b", "c"]


def test_fusing_nothing_returns_nothing():
    assert fuse([]) == []
    assert fuse([[], []]) == []


def test_limit_truncates_the_fused_list():
    assert fuse([["a", "b", "c"]], limit=2) == ["a", "b"]


def test_the_default_constant_is_the_papers():
    assert DEFAULT_K == 60


# --- Review Focus 4: absence is not a bottom rank ---------------------------

def test_a_channel_that_did_not_return_a_document_does_not_vote_on_it():
    # 22.5% of products have no image embedding, so they are absent from the
    # image channel. If absence contributed a worst-possible rank, a scrape
    # artefact would push them down; if it contributed the same as any rank, a
    # channel that retrieved nothing would still vote.
    with_image = fuse([["a", "b"], ["a"]], k=1)
    assert with_image == ["a", "b"]

    # "a" scores 1/(1+1) + 1/(1+1) = 1.0; "b" scores 1/(1+2) = 0.333 from the
    # first channel only. If the second channel voted on "b" at all, "b" would
    # score higher than 0.333.
    scores = _scores([["a", "b"], ["a"]], k=1)
    assert scores["b"] == pytest.approx(1 / 3)


def _scores(ranked_lists, k):
    """Recompute RRF by hand, to pin the arithmetic rather than the order."""
    out: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, doc in enumerate(ranked, start=1):
            out[doc] = out.get(doc, 0.0) + 1 / (k + position)
    return out


def test_a_short_channel_does_not_penalise_what_it_omitted():
    # Channel 2 returned one document. "b" and "c" must keep channel 1's
    # relative order, not be pushed below anything by channel 2's silence.
    # Asserted as an ordering rather than an exact list because "a" and "b"
    # genuinely tie at 1/(60+1) and the tie-break is first appearance.
    fused = fuse([["b", "c"], ["a"]], k=60)
    assert fused.index("b") < fused.index("c")
    assert set(fused) == {"a", "b", "c"}


def test_an_empty_channel_contributes_nothing():
    assert fuse([["a", "b"], []]) == fuse([["a", "b"]])


# --- weights ----------------------------------------------------------------

def test_a_zero_weight_silences_a_channel():
    assert fuse([["a"], ["b"]], weights=[1.0, 0.0]) == ["a", "b"]
    assert fuse([["a"], ["b"]], weights=[1.0, 0.0])[0] == "a"


def test_weights_change_which_channel_wins_a_tie():
    assert fuse([["a"], ["b"]], weights=[2.0, 1.0])[0] == "a"
    assert fuse([["a"], ["b"]], weights=[1.0, 2.0])[0] == "b"


def test_the_wrong_number_of_weights_raises():
    with pytest.raises(ValueError, match="weights"):
        fuse([["a"], ["b"]], weights=[1.0])


# --- batches ----------------------------------------------------------------

def test_fuse_batches_fuses_each_query_independently():
    # One list per channel per query: channel_results[channel][query].
    fused = fuse_batches([[["a"], ["c"]], [["a"], ["d"]]])
    assert fused == [["a"], ["c", "d"]]


def test_fuse_batches_requires_every_channel_to_cover_every_query():
    with pytest.raises(ValueError, match="queries"):
        fuse_batches([[["a"], ["b"]], [["a"]]])


def test_fuse_batches_of_nothing_returns_nothing():
    assert fuse_batches([]) == []
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_rrf.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.rrf'`.

- [x] **Step 3: Write `src/rrf.py`**

```python
"""Reciprocal Rank Fusion over ranked lists.

score(d) = sum over the channels that returned d of  w_c / (k + rank_c(d))

The "channels that returned d" clause is the whole design. The image channel
has no vector for 22.5% of products, so those products are simply *absent*
from its list. Absence must contribute nothing: a worst-possible rank is still
a vote, and it would push products down for a 2022 scrape failure rather than
for anything about the product. `CLAUDE.md`'s evaluation discipline is explicit
that this kind of missingness is an artefact and must not be treated as signal.

k = 60 is the constant from Cormack et al.'s original RRF paper. It is a
tunable; src/recall_report.py tunes it on the frozen validation folds and never
on test.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_K = 60


def fuse(
    ranked_lists: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Fuse one query's ranked lists into a single ranking, best first."""
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            f"{len(weights)} weights for {len(ranked_lists)} channels; "
            "pass one weight per channel or none at all"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for channel, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else float(weights[channel])
        seen: set[str] = set()
        for position, document in enumerate(ranked, start=1):
            # A channel that returns a document twice votes for it once.
            if document in seen:
                continue
            seen.add(document)
            scores[document] = scores.get(document, 0.0) + weight / (k + position)
            if document not in first_seen:
                first_seen[document] = order
                order += 1

    # Ties break on first appearance, so fusion is deterministic across runs
    # rather than dependent on dict iteration order.
    ranking = sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
    return ranking[:limit] if limit is not None else ranking


def fuse_batches(
    channel_results: Sequence[Sequence[Sequence[str]]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[list[str]]:
    """Fuse many queries at once. Indexing is channel_results[channel][query]."""
    if not channel_results:
        return []

    n_queries = len(channel_results[0])
    for channel, results in enumerate(channel_results):
        if len(results) != n_queries:
            raise ValueError(
                f"channel {channel} covers {len(results)} queries but channel "
                f"0 covers {n_queries}; every channel must answer every query"
            )

    return [
        fuse(
            [results[query] for results in channel_results],
            k=k,
            weights=weights,
            limit=limit,
        )
        for query in range(n_queries)
    ]
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_rrf.py -v
```

Expected: PASS, 17 tests.

- [x] **Step 5: Commit**

```bash
git add src/rrf.py tests/test_rrf.py
git commit -m "Add reciprocal rank fusion that does not vote on absent documents"
```

---

## Task 6: Stage 0 — LLM query rewriting

**Files:**
- Create: `src/query_rewrite.py`
- Create: `tests/test_query_rewrite.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `src.query_rewrite.DEFAULT_CACHE: Path` (`data/rewrites.json`), `DEFAULT_MODEL: str` (`gpt-5.6-luna`), `MAX_COMPLETION_TOKENS: int` (1000), `PROMPT: str`, `MAX_EXPANSION: int` (4)
  - `src.query_rewrite.validate_rewrite(raw, rewritten) -> str`
  - `src.query_rewrite.RewriteCache` with `.get(query)`, `.set(query, rewrite)`, `.save()`, `.__len__()`
  - `src.query_rewrite.rewrite_queries(queries, *, call, cache=None, progress=None) -> dict[str, str]`
  - `python -m src.query_rewrite --split train --folds 0`

`rewrite_queries` takes a `call` callable, so every path — cache hits, empty
responses, refusals, dropped brands — is tested with no API key and no network.
The real client lives in `__main__`.

**Validation is not optional.** An LLM asked to expand a query can return an
empty string, a refusal, a JSON wrapper, or a paraphrase that drops the model
number the query was about. Used unchecked, Ablation 1 measures prompt damage
rather than query understanding, and the raw-query arm wins for the wrong
reason. `validate_rewrite` falls back to the raw query whenever the rewrite
looks wrong, and the fallback rate is reported.

- [x] **Step 1: Add the API client dependency**

In `pyproject.toml`, extend the `retrieval` extra added in Phase 1:

```toml
retrieval = [
    "bm25s>=0.3",
    "PyStemmer>=2.2",
    "openai>=1.40",
]
```

Then install it:

```bash
uv pip install -e ".[dev,baselines,retrieval]"
```

- [x] **Step 2: Write the failing test**

Create `tests/test_query_rewrite.py`:

```python
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
```

- [x] **Step 3: Run the test to verify it fails**

```bash
python -m pytest tests/test_query_rewrite.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.query_rewrite'`.

- [x] **Step 4: Write `src/query_rewrite.py`**

```python
"""Stage 0: LLM query rewriting, cached and validated.

PROJECT_SPEC.md §4.0 puts one LLM call per query before retrieval, measured by
Recall@k lift against the raw query. This is the only paid component in the
project, so every rewrite is cached to disk keyed by the raw query: a re-run
costs nothing and Ablation 1 is reproducible.

Every rewrite is validated before use. An LLM asked to expand a query can
return an empty string, a refusal, a JSON wrapper, or a fluent paraphrase that
drops the model number the query was entirely about. Used unchecked, Ablation 1
would measure prompt damage rather than query understanding, and the raw-query
arm would win for the wrong reason. `validate_rewrite` falls back to the raw
query whenever the rewrite looks wrong, and the CLI reports how often it did.

`rewrite_queries` takes a `call` callable so every path is tested with no API
key and no network; the real client lives in __main__.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

DEFAULT_CACHE = Path("data/rewrites.json")
DEFAULT_MODEL = "gpt-5.6-luna"

# gpt-5.6-luna is a reasoning model: it spends completion tokens thinking before
# it emits anything. Measured on this prompt, reasoning alone takes 52-162
# tokens, so the 100 that sufficed for a non-reasoning model left *zero* for the
# answer - the call returned an empty string or failed outright with "Could not
# finish the message because max_tokens ... was reached". Billing is on tokens
# actually generated (76-181 here), so a generous cap costs nothing.
MAX_COMPLETION_TOKENS = 1000

# A rewrite may add context but must not become a bag of words that matches the
# whole corpus: cap it at MAX_EXPANSION times the raw token count.
MAX_EXPANSION = 4

PROMPT = """Rewrite this e-commerce search query to improve product retrieval.

Keep every brand, model number, size and colour from the original - they are
what the shopper is looking for. You may add the product category or an obvious
synonym. Reply with the rewritten query only, as plain text, with no quotes,
no JSON and no explanation.

Query: {query}"""

_REFUSAL = re.compile(
    r"^\s*(i'?m sorry|i cannot|i can'?t|as an ai|unfortunately[, ])", re.I
)
_ALPHANUMERIC = re.compile(r"\b(?=[a-z]*\d)[a-z0-9][a-z0-9-]{2,}\b", re.I)
_FLATTEN = re.compile(r"[^a-z0-9]+")


def validate_rewrite(raw: str, rewritten: str) -> str:
    """The rewrite if it is usable, otherwise the raw query.

    Falling back is always safe: the worst case is that Stage 0 buys nothing
    for that query, which is a real result. Using a broken rewrite is not
    safe - it silently turns Ablation 1 into a measurement of prompt damage.
    """
    if not isinstance(rewritten, str):
        return raw

    candidate = rewritten.strip()

    # Models asked for plain text sometimes answer with JSON anyway.
    if candidate.startswith("{"):
        try:
            payload = json.loads(candidate)
        except ValueError:
            return raw
        if not isinstance(payload, dict):
            return raw
        candidate = str(
            payload.get("query") or payload.get("rewritten") or ""
        ).strip()

    candidate = candidate.strip().strip('"').strip("'").strip()
    if not candidate:
        return raw
    if _REFUSAL.match(candidate):
        return raw

    raw_tokens = raw.split()
    if len(candidate.split()) > MAX_EXPANSION * max(len(raw_tokens), 1):
        return raw

    # Identifiers - model numbers, sizes, SKUs - are usually the whole query.
    # A rewrite that drops one is about a different product. Compared with
    # separators stripped from both sides, so "a7iii" still matches a rewrite
    # that respaced it to "a7 III": the guard is about the identifier
    # surviving, not about how it was punctuated. A rewrite that genuinely
    # drops it - "sony wh-1000xm4" -> "sony wireless headphones" - still fails.
    flattened = _FLATTEN.sub("", candidate.lower())
    for token in _ALPHANUMERIC.findall(raw.lower()):
        if _FLATTEN.sub("", token) not in flattened:
            return raw

    return candidate


class RewriteCache:
    """Raw query -> validated rewrite, persisted as one JSON object."""

    def __init__(self, path: Path = DEFAULT_CACHE) -> None:
        self.path = Path(path)
        self._entries: dict[str, str] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError:
                # The cache is an optimisation; a truncated write from a killed
                # run costs API calls, not the run.
                loaded = {}
            if isinstance(loaded, dict):
                self._entries = {str(k): str(v) for k, v in loaded.items()}

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, query: str) -> str | None:
        return self._entries.get(query)

    def set(self, query: str, rewrite: str) -> None:
        self._entries[query] = rewrite

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=0, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def rewrite_queries(
    queries: Iterable[str],
    *,
    call: Callable[[str], str],
    cache: RewriteCache | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, str]:
    """A usable rewrite for every distinct query, falling back to the raw one."""
    distinct = list(dict.fromkeys(queries))
    out: dict[str, str] = {}

    for done, query in enumerate(distinct, start=1):
        cached = cache.get(query) if cache is not None else None
        if cached is not None:
            out[query] = cached
        else:
            try:
                rewritten = validate_rewrite(query, call(query))
            except Exception:
                # One transient API failure must not lose the other 20,887
                # rewrites, and a raw query is always a valid query.
                rewritten = query
            out[query] = rewritten
            if cache is not None:
                cache.set(query, rewritten)
        if progress is not None:
            progress(done, len(distinct))

    if cache is not None:
        cache.save()
    return out


def _main() -> int:
    from src.recall import load_queries

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--folds", default=None, help="comma-separated validation folds"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS
    )
    args = parser.parse_args()

    folds = (
        [int(f) for f in args.folds.split(",")] if args.folds is not None else None
    )
    frame = load_queries(args.split, folds=folds)
    if args.limit is not None:
        frame = frame.head(args.limit)
    queries = frame["query"].astype(str).tolist()

    cache = RewriteCache(args.cache)
    print(f"{len(queries):,} queries, {len(cache):,} already cached")

    import openai

    client = openai.OpenAI()

    def call(query: str) -> str:
        # `max_completion_tokens`, not `max_tokens`: the gpt-5.x models reject
        # the older parameter name.
        response = client.chat.completions.create(
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            messages=[{"role": "user", "content": PROMPT.format(query=query)}],
        )
        return response.choices[0].message.content or ""

    def report(done: int, total: int) -> None:
        print(f"\r  {done:,}/{total:,}", end="", flush=True)

    rewrites = rewrite_queries(queries, call=call, cache=cache, progress=report)
    print()

    unchanged = sum(1 for q, r in rewrites.items() if q == r)
    print(
        f"{len(rewrites):,} rewrites, {unchanged:,} fell back to the raw query "
        f"({unchanged / max(len(rewrites), 1):.1%})"
    )
    print(f"cache written to {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [x] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest tests/test_query_rewrite.py -v
```

Expected: PASS, 20 tests.

- [ ] **Step 6: Rewrite the validation-fold queries**

Stage 0 is the only paid step. Rewrite **validation fold 0 only** — about
4,200 queries — which is enough for Ablation 1 and costs a fraction of the
full split.

```bash
export OPENAI_API_KEY=...           # or however the key reaches the process
python -m src.query_rewrite --split train --folds 0 --limit 50   # check the output first
python -m src.query_rewrite --split train --folds 0
```

Expected: the limited run prints a fallback rate; inspect
`data/rewrites.json` by eye before spending on the rest.

**If the fallback rate is above ~20%**, the prompt is fighting the validator
rather than the validator catching a rare failure. Read the rejected rewrites
before touching `validate_rewrite` — lowering the bar to make the number look
better is exactly how Ablation 1 stops meaning anything.

This happened on the first run, at **46.7%**, and neither cause was the
validator being too strict:

1. **The token budget starved the model.** Ported from a non-reasoning model,
   `max_completion_tokens=100` was entirely consumed by reasoning, so the call
   returned `""` or failed with a 400. Raising it to 1000 fixed it; billing is
   on tokens generated, so the cap costs nothing.
2. **The identifier check compared punctuation.** `"sony a7iii"` rewritten to
   `"Sony Alpha a7 III"` keeps the model number and only respaces it, but a
   raw substring test rejected it. Both sides are now flattened first.

After both, a 40-query random sample rewrote **40/40** with zero rejections and
zero API errors. Note also that the CLI's "fell back to the raw query" count
includes rewrites the *model* returned unchanged, which is not a rejection —
read the two apart before reacting to the number.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/query_rewrite.py tests/test_query_rewrite.py
git commit -m "Add Stage 0 LLM query rewriting with an on-disk cache and validation"
```

---

## Task 7: Ablations 1 and 2

**Files:**
- Create: `src/recall_report.py`
- Create: `tests/test_recall_report.py`
- Create: `docs/results/recall.json` (generated in Step 4, committed)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `src.recall.*`, `src.rrf.fuse_batches`, `src.bm25_index.open_channel`, `src.dense_embed.open_channel`, `src.image_channel.open_channel`, `src.query_rewrite.RewriteCache`, `src.bootstrap.bootstrap_ci`, `src.bootstrap.paired_delta_ci`, `src.splits.load_folds`.
- Produces:
  - `src.recall_report.DEFAULT_OUT: Path` (`docs/results/recall.json`), `STORAGE_GB: dict[str, float]`, `SPEC_BUDGET_GB: float` (1.0)
  - `src.recall_report.ArmResult(name, table)` with `.to_dict()`
  - `src.recall_report.evaluate_arm(name, retrieved, relevant, ks=DEFAULT_KS) -> ArmResult`
  - `src.recall_report.compare(arm, baseline, k=100) -> dict`
  - `src.recall_report.format_table(arms) -> str`
  - `python -m src.recall_report --split train --folds 0`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recall_report.py`:

```python
import math

import pytest

from src.recall_report import (
    SPEC_BUDGET_GB,
    STORAGE_GB,
    compare,
    evaluate_arm,
    format_table,
)


def _arm(name, retrieved, relevant):
    return evaluate_arm(name, retrieved, relevant, ks=(1, 10))


def test_an_arm_reports_recall_at_every_k():
    arm = _arm("bm25", {1: ["a", "b"]}, {1: {"a", "b"}})
    assert arm.table[1].mean == pytest.approx(0.5)
    assert arm.table[10].mean == pytest.approx(1.0)


def test_an_arm_serialises_with_its_denominator():
    # Two arms with different query coverage are not comparable, so the
    # denominator travels with the number.
    arm = _arm("bm25", {1: ["a"], 2: ["x"]}, {1: {"a"}})
    payload = arm.to_dict()
    assert payload["name"] == "bm25"
    assert payload["recall"]["10"]["n_queries"] == 1
    assert payload["recall"]["10"]["n_skipped"] == 1


def test_compare_reports_a_paired_delta_with_an_interval():
    better = _arm("fused", {1: ["a"], 2: ["b"]}, {1: {"a"}, 2: {"b"}})
    worse = _arm("bm25", {1: ["x"], 2: ["y"]}, {1: {"a"}, 2: {"b"}})
    result = compare(better, worse, k=10)
    assert result["delta"]["point"] == pytest.approx(1.0)
    assert result["baseline"] == "bm25"


def test_compare_pairs_on_queries_not_on_position():
    # paired_delta_ci pairs by key. An arm that answered a different set of
    # queries must not be silently compared position by position.
    a = _arm("a", {1: ["a"], 2: ["b"]}, {1: {"a"}, 2: {"b"}})
    b = _arm("b", {2: ["b"], 1: ["a"]}, {1: {"a"}, 2: {"b"}})
    assert compare(a, b, k=10)["delta"]["point"] == pytest.approx(0.0)


def test_compare_on_a_k_the_arm_does_not_have_raises():
    arm = _arm("a", {1: ["a"]}, {1: {"a"}})
    with pytest.raises(KeyError, match="100"):
        compare(arm, arm, k=100)


def test_the_table_names_every_arm():
    arms = [
        _arm("bm25", {1: ["a"]}, {1: {"a"}}),
        _arm("dense", {1: ["a"]}, {1: {"a"}}),
    ]
    rendered = format_table(arms)
    assert "bm25" in rendered and "dense" in rendered


def test_the_table_renders_a_nan_recall_without_crashing():
    # Every query skipped gives a NaN mean; the report must say so rather
    # than raise or print a misleading 0.000.
    arm = _arm("empty", {1: ["a"]}, {})
    assert math.isnan(arm.table[10].mean)
    assert "nan" in format_table([arm]).lower()


def test_the_storage_overrun_is_recorded_not_hidden():
    # PROJECT_SPEC.md §9 budgets "<1 GB" for embeddings. The three channels
    # total 2.94 GB. The honest number is worth more than a flattering one.
    assert SPEC_BUDGET_GB == 1.0
    assert sum(STORAGE_GB.values()) > SPEC_BUDGET_GB
    assert set(STORAGE_GB) == {"bm25", "dense", "image"}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_recall_report.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.recall_report'`.

- [ ] **Step 3: Write `src/recall_report.py`**

```python
"""Ablations 1 and 2: the recall table this plan exists to produce.

Ablation 1 - raw vs. LLM-rewritten query, on the fused run.
Ablation 2 - dense-only vs. +BM25 vs. +image, all fused with RRF.

Every arm carries its own denominator, because a query with no relevant
product is skipped and two arms that skipped different queries are not
comparable. Every comparison is a paired bootstrap over queries, via
src.bootstrap, which pairs on the query id rather than on position.

Selection happens on the frozen validation folds from Plan 1. `--split test`
exists, but running it before the folds have settled the configuration is
exactly the thing CLAUDE.md's evaluation discipline forbids.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.bootstrap import bootstrap_ci, paired_delta_ci
from src.recall import DEFAULT_KS, RecallResult, recall_table

DEFAULT_OUT = Path("docs/results/recall.json")

# Measured on 2026-09-20. PROJECT_SPEC.md §9 budgets "<1 GB" for embeddings;
# the three channels together do not fit, and that is a reportable fact rather
# than a reason to shrink a channel.
STORAGE_GB = {"bm25": 1.10, "dense": 0.93, "image": 0.91}
SPEC_BUDGET_GB = 1.0


@dataclass(frozen=True)
class ArmResult:
    name: str
    table: dict[int, RecallResult]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "recall": {str(k): r.to_dict() for k, r in sorted(self.table.items())},
        }


def evaluate_arm(
    name: str,
    retrieved: Mapping[int, Sequence[str]],
    relevant: Mapping[int, set[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> ArmResult:
    """Recall@k for one configuration."""
    return ArmResult(name=name, table=recall_table(retrieved, relevant, ks=ks))


def _interval(interval) -> dict:
    """An src.bootstrap.Interval as plain JSON."""
    return {"point": interval.point, "low": interval.low, "high": interval.high}


def compare(arm: ArmResult, baseline: ArmResult, k: int = 100) -> dict:
    """Paired bootstrap delta of `arm` against `baseline` at k."""
    if k not in arm.table:
        raise KeyError(f"arm {arm.name!r} has no recall at k={k}")
    if k not in baseline.table:
        raise KeyError(f"arm {baseline.name!r} has no recall at k={k}")

    delta = paired_delta_ci(arm.table[k].per_query, baseline.table[k].per_query)
    return {
        "arm": arm.name,
        "baseline": baseline.name,
        "k": k,
        "delta": _interval(delta),
        "significant": not (delta.low <= 0.0 <= delta.high),
    }


def format_table(arms: Sequence[ArmResult]) -> str:
    """A markdown table of every arm at every k."""
    if not arms:
        return "(no arms)"
    ks = sorted(arms[0].table)
    header = "| arm | " + " | ".join(f"R@{k}" for k in ks) + " | queries |"
    rule = "|---" * (len(ks) + 2) + "|"
    lines = [header, rule]
    for arm in arms:
        cells = []
        for k in ks:
            mean = arm.table[k].mean
            cells.append("nan" if math.isnan(mean) else f"{mean:.4f}")
        n = arm.table[ks[-1]].n_queries
        lines.append(f"| {arm.name} | " + " | ".join(cells) + f" | {n:,} |")
    return "\n".join(lines)


def _main() -> int:
    from src.bm25_index import open_channel as open_bm25
    from src.dense_embed import open_channel as open_dense
    from src.image_channel import open_channel as open_image
    from src.query_rewrite import RewriteCache
    from src.recall import DEFAULT_JUDGEMENTS, load_queries, relevant_sets
    from src.rrf import DEFAULT_K, fuse_batches

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--folds", default="0", help="comma-separated, or 'all'")
    parser.add_argument("--relevance", default="E", choices=["E", "E+S"])
    parser.add_argument("--k", type=int, default=1000, help="depth to retrieve")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_K)
    parser.add_argument("--image-scope", default="catalogue")
    parser.add_argument("--rewrites", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    folds = None if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    queries = load_queries(args.split, folds=folds)
    judgements = pd.read_parquet(DEFAULT_JUDGEMENTS)
    judgements = judgements.loc[
        judgements["query_id"].isin(set(queries["query_id"]))
    ]
    relevant = relevant_sets(judgements, relevance=args.relevance)
    query_ids = queries["query_id"].tolist()
    raw = queries["query"].astype(str).tolist()
    print(f"{len(raw):,} queries from {args.split} folds {args.folds}; "
          f"{len(relevant):,} have a relevant product under {args.relevance}")

    channels = {
        "bm25": open_bm25(),
        "dense": open_dense(),
        "image": open_image(scope=args.image_scope),
    }
    hits = {}
    for name, channel in channels.items():
        hits[name] = channel.search(raw, args.k)
        print(f"  {name}: searched")

    def as_map(ranked: Sequence[Sequence[str]]) -> dict[int, Sequence[str]]:
        return dict(zip(query_ids, ranked))

    arms = [
        evaluate_arm("bm25", as_map(hits["bm25"]), relevant),
        evaluate_arm("dense", as_map(hits["dense"]), relevant),
        evaluate_arm("image", as_map(hits["image"]), relevant),
    ]

    # --- Ablation 2: dense-only vs. +BM25 vs. +image -----------------------
    combos = [
        ("dense+bm25", ["dense", "bm25"]),
        ("dense+bm25+image", ["dense", "bm25", "image"]),
    ]
    for name, members in combos:
        fused = fuse_batches(
            [hits[m] for m in members], k=args.rrf_k, limit=args.k
        )
        arms.append(evaluate_arm(name, as_map(fused), relevant))

    # --- Ablation 1: raw vs. rewritten query -------------------------------
    ablation1 = None
    if args.rewrites is not None:
        cache = RewriteCache(args.rewrites)
        rewritten = [cache.get(q) or q for q in raw]
        changed = sum(1 for a, b in zip(raw, rewritten) if a != b)
        print(f"  rewrites: {changed:,}/{len(raw):,} queries changed")
        rewritten_hits = {
            name: channel.search(rewritten, args.k)
            for name, channel in channels.items()
        }
        fused = fuse_batches(
            [rewritten_hits[m] for m in ["dense", "bm25", "image"]],
            k=args.rrf_k,
            limit=args.k,
        )
        arm = evaluate_arm("rewritten (dense+bm25+image)", as_map(fused), relevant)
        raw_arm = next(a for a in arms if a.name == "dense+bm25+image")
        arms.append(arm)
        ablation1 = compare(arm, raw_arm, k=100)

    print()
    print(format_table(arms))

    baseline = next(a for a in arms if a.name == "bm25")
    comparisons = [
        compare(arm, baseline, k=100) for arm in arms if arm is not baseline
    ]
    intervals = {
        arm.name: _interval(bootstrap_ci(arm.table[100].per_query))
        for arm in arms
        if 100 in arm.table and arm.table[100].per_query
    }
    print()
    for row in comparisons:
        flag = "significant" if row["significant"] else "ties"
        d = row["delta"]
        print(
            f"  {row['arm']} vs {row['baseline']} @100: "
            f"{d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  {flag}"
        )

    payload = {
        "split": args.split,
        "folds": args.folds,
        "relevance": args.relevance,
        "retrieval_depth": args.k,
        "rrf_k": args.rrf_k,
        "bm25_baseline_r100": 0.5018,
        "arms": [arm.to_dict() for arm in arms],
        "comparisons": comparisons,
        "recall_at_100_intervals": intervals,
        "ablation_1_rewrite": ablation1,
        "storage_gb": STORAGE_GB
        | {
            "total": round(sum(STORAGE_GB.values()), 2),
            "spec_budget": SPEC_BUDGET_GB,
            "over_budget": round(sum(STORAGE_GB.values()) - SPEC_BUDGET_GB, 2),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests, then the real report**

```bash
python -m pytest tests/test_recall_report.py -v
python -m pytest -q

# Ablation 2 on the validation folds
python -m src.recall_report --split train --folds 0

# Ablation 1, once the rewrites are cached
python -m src.recall_report --split train --folds 0 --rewrites data/rewrites.json
```

Expected: 8 report tests pass; the table prints five or six arms; BM25 lands
near its measured **R@100 = 0.5018**; `docs/results/recall.json` is written.

**Interpret the table before tuning.** If the fused arm ties BM25, that is a
reportable result and `CLAUDE.md` says so explicitly. Tune `--rrf-k` on the
validation folds if you tune it at all, record what you tried, and never move
to `--split test` to find a better number.

- [ ] **Step 5: Record the command in `CLAUDE.md`**

Add under the retrieval-channels comment:

````markdown
python -m src.recall_report --split train --folds 0   # Ablations 1 and 2
````

- [ ] **Step 6: Commit**

```bash
git add src/recall_report.py tests/test_recall_report.py \
        docs/results/recall.json CLAUDE.md
git commit -m "Report Recall@k ablations for the three channels and their fusion"
```

---

## Phase 3 Gate

This is the Plan Gate — Plan 5 does not start until all of these hold. The
canonical copy lives in [`README.md`](README.md#plan-gate).

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the index round-trip on the real corpus.
- [ ] `data/bm25/` memory-maps in under 2 GB of RSS and answers under 100 ms per query amortised over a batch.
- [ ] Every channel reports Recall@{10,50,100,500,1000} on the frozen validation folds, and the fused run beats the BM25 R@100 baseline of 0.5018 or reports honestly that it ties, with a paired bootstrap CI.
- [ ] `docs/results/recall.json` records per-channel and fused Recall@k, the query-coverage denominator, and the storage overrun against §9's <1 GB.
- [ ] Ablation 1 (raw vs. rewritten query) and Ablation 2 (dense-only vs. +BM25 vs. +image) are both reported with bootstrap CIs.
- [ ] `CLAUDE.md`'s Commands section lists the index build, the corpus embed and the recall report.

Then: Plan 5 — Coarse Rank. See [`../README.md`](../README.md) for the series.
