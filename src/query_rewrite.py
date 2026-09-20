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
from collections.abc import Callable, Iterable
from pathlib import Path

DEFAULT_CACHE = Path("data/rewrites.json")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

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
    # A rewrite that drops one is about a different product.
    lowered = candidate.lower()
    for token in _ALPHANUMERIC.findall(raw.lower()):
        if token not in lowered:
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

    import anthropic

    client = anthropic.Anthropic()

    def call(query: str) -> str:
        response = client.messages.create(
            model=args.model,
            max_tokens=100,
            messages=[{"role": "user", "content": PROMPT.format(query=query)}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

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
