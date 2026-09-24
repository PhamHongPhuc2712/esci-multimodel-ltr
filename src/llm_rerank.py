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
import threading
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

# Windows between cache writes. A fold-0 pass is ~2 h of paid calls; saving
# only at the end means an interruption at 90% loses every answer, which is
# the opposite of the resumability the cache exists for. 200 caps the loss at
# ~90 s of calls and costs one small JSON write per 200 windows.
CHECKPOINT_EVERY = 200

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
        # Workers write while the main loop checkpoints, so both go through
        # the lock: json.dumps over a dict another thread is mutating raises
        # "dictionary changed size during iteration" and loses the write.
        self._lock = threading.Lock()
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
        with self._lock:
            self._entries[key] = [int(i) for i in permutation]

    def save(self) -> None:
        with self._lock:
            payload = json.dumps(self._entries, indent=0, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a process killed mid-write leaves the previous
        # cache intact rather than a truncated file worth nothing.
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)


@dataclass(frozen=True)
class Usage:
    """What the arm cost. Tokens and seconds, never an invented price."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    # Time spent inside real API calls, summed across threads. `seconds` is
    # wall clock for the whole pass and collapses to cache-replay time when
    # the cache is warm; this does not, so a single-query latency derived from
    # it stays honest.
    call_seconds: float = 0.0
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
            "call_seconds": self.call_seconds,
            "n_calls": self.n_calls,
            "n_cached": self.n_cached,
            "n_fallback": self.n_fallback,
        }


_SUMMED = ("prompt_tokens", "completion_tokens", "reasoning_tokens",
           "n_calls", "n_cached", "n_fallback")


def append_usage(path: Path, record: Mapping) -> dict:
    """Append one pass's usage to a committed record of runs, and total them.

    A paid pass over thousands of windows can be interrupted; the cache makes
    the resumption cheap, but it is a second run with its own tokens, and the
    bill is the sum. Overwriting would publish only the last leg.
    """
    path = Path(path)
    payload = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    )
    payload["runs"].append(dict(record))
    totals = {
        key: sum(int(run["usage"][key]) for run in payload["runs"]) for key in _SUMMED
    }
    calls = max(totals["n_calls"], 1)
    totals["prompt_tokens_per_call"] = totals["prompt_tokens"] / calls
    totals["completion_tokens_per_call"] = totals["completion_tokens"] / calls
    totals["reasoning_tokens_per_call"] = totals["reasoning_tokens"] / calls
    payload["totals"] = totals
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def rerank_windows(
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    call: Call,
    cache: RerankCache | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[int, int], None] | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> tuple[dict[str, list[str]], Usage]:
    """A new ordering per window, falling back to Stage 2's on any failure."""
    windows_ = list(windows_)
    if not windows_:
        return {}, Usage()

    totals = {"pt": 0, "ct": 0, "rt": 0, "calls": 0, "cached": 0, "fallback": 0}
    totals_seconds = 0.0
    # `d[k] += v` is read-modify-write and the GIL does not make it atomic, so
    # four workers over 4,130 windows can lose token updates silently.
    totals_lock = threading.Lock()

    def one(w) -> tuple[str, list[str]]:
        documents = list(w.window)
        query = str(query_text[w.query_id])
        key = window_key(query, documents)

        if cache is not None:
            cached = cache.get(key)
            if cached is not None and sorted(cached) == list(
                range(1, len(documents) + 1)
            ):
                with totals_lock:
                    totals["cached"] += 1
                return w.query_id, [documents[i - 1] for i in cached]

        titles = [str(doc_text.get(d, "")) for d in documents]
        started_call = time.perf_counter()
        try:
            text, pt, ct, rt = call(build_prompt(query, titles))
        except Exception:
            # One transient failure must not lose the other 4,129 windows, and
            # the Stage 2 order is always a valid order.
            with totals_lock:
                totals["fallback"] += 1
                totals["calls"] += 1
            return w.query_id, documents
        call_elapsed = time.perf_counter() - started_call

        nonlocal totals_seconds
        with totals_lock:
            totals["pt"] += int(pt)
            totals["ct"] += int(ct)
            totals["rt"] += int(rt)
            totals["calls"] += 1
            totals_seconds += call_elapsed

        permutation = parse_permutation(text, len(documents))
        if permutation is None:
            with totals_lock:
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
            # Checkpoint from the consumer thread, not the workers: an
            # interruption must cost the windows in flight, not the run.
            if cache is not None and checkpoint_every > 0 and done % checkpoint_every == 0:
                cache.save()
    elapsed = time.time() - started

    if cache is not None:
        cache.save()

    return out, Usage(
        prompt_tokens=totals["pt"],
        completion_tokens=totals["ct"],
        reasoning_tokens=totals["rt"],
        seconds=elapsed,
        call_seconds=totals_seconds,
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
    parser.add_argument(
        "--usage-out", type=Path, default=None,
        help="append this pass's token usage to a JSON record (for committing the bill)")
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
    if args.usage_out is not None:
        payload = append_usage(args.usage_out, {
            "split": args.split, "folds": args.folds, "sample": args.sample,
            "k": args.k, "model": args.model, "n_windows": len(ws),
            "usage": usage.to_dict(),
        })
        print(f"usage appended to {args.usage_out} "
              f"({len(payload['runs'])} run(s), {payload['totals']['n_calls']:,} calls in total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
