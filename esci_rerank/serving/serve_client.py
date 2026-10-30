"""Plan 13: score Stage 3 windows through a vLLM server.

The server is sent token ids built by the same functions training used, so a
template edit or a tokenizer boundary cannot make the served model answer a
different prompt from the one it was evaluated on. Standard library only:
this module runs in the project venv and talks to `.venv-serve` over HTTP.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from collections.abc import Callable, Mapping, Sequence

from esci_rerank.rerank.llm_list import END_OF_TURN, MAX_NEW_TOKENS, encode_prompt
from esci_rerank.rerank.llm_point import encode_pair, order_by_scores, p_yes
from esci_rerank.rerank.llm_rerank import parse_permutation

DEFAULT_URL = "http://127.0.0.1:8000"
TOP_LOGPROBS = 20
TIMEOUT_S = 60

Post = Callable[[str, dict], dict]
Get = Callable[[str], str]

_HITS = re.compile(r"^vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)


def urllib_post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())


def urllib_get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read().decode("utf-8")


def point_payload(model: str, ids: list[list[int]]) -> dict:
    return {"model": model, "prompt": ids, "max_tokens": 1,
            "temperature": 0.0, "logprobs": TOP_LOGPROBS}


def p_yes_from_top(top: Mapping[str, float]) -> tuple[float, bool]:
    yes, no = top.get("yes"), top.get("no")
    if yes is None and no is None:
        raise ValueError("neither 'yes' nor 'no' is among the top log-probabilities")
    imputed = yes is None or no is None
    floor = min(top.values())
    return p_yes(floor if no is None else no, floor if yes is None else yes), imputed


def served_point_ordering(post: Post, url: str, model: str, tokenizer, query: str,
                          docs: Sequence[str], window: Sequence[str],
                          max_length: int) -> tuple[list[str], int]:
    ids = [encode_pair(tokenizer, query, d, max_length)[0] for d in docs]
    response = post(f"{url}/v1/completions", point_payload(model, ids))
    choices = sorted(response["choices"], key=lambda c: c["index"])
    if len(choices) != len(docs):
        raise ValueError(f"{len(choices)} choices for {len(docs)} documents")
    scores, imputed = [], 0
    for choice in choices:
        score, flag = p_yes_from_top(choice["logprobs"]["top_logprobs"][0])
        scores.append(score)
        imputed += int(flag)
    return order_by_scores(window, scores), imputed


def served_list_ordering(post: Post, url: str, model: str, tokenizer, query: str,
                         docs: Sequence[str], window: Sequence[str]) -> tuple[list[str], bool]:
    payload = {"model": model, "prompt": encode_prompt(tokenizer, query, docs),
               "max_tokens": MAX_NEW_TOKENS, "temperature": 0.0, "stop": [END_OF_TURN]}
    text = post(f"{url}/v1/completions", payload)["choices"][0]["text"]
    permutation = parse_permutation(text, len(window))
    if permutation is None:
        return list(window), True
    return [window[i - 1] for i in permutation], False


def server_version(get: Get, url: str) -> str:
    return str(json.loads(get(f"{url}/version"))["version"])


def prefix_cache_hits(metrics_text: str) -> float | None:
    values = [float(v) for v in _HITS.findall(metrics_text)]
    return sum(values) if values else None


class ServedRanker:
    """One window at a time through the server; safe to call from many threads."""

    def __init__(self, url: str, model: str, tokenizer, kind: str, query_text: Mapping,
                 doc_text: Mapping, max_length: int, post: Post = urllib_post) -> None:
        if kind not in ("point", "list"):
            raise ValueError(f"unknown kind {kind!r}")
        self.url, self.model, self.tokenizer, self.kind = url, model, tokenizer, kind
        self.query_text, self.doc_text, self.max_length, self.post = (
            query_text, doc_text, max_length, post)
        self._lock = threading.Lock()
        self.imputed = 0
        self.malformed: list[str] = []

    def order(self, window) -> list[str]:
        query = self.query_text[window.query_id]
        docs = [self.doc_text[d] for d in window.window]
        if self.kind == "point":
            ordering, n = served_point_ordering(self.post, self.url, self.model, self.tokenizer,
                                                query, docs, window.window, self.max_length)
            with self._lock:
                self.imputed += n
        else:
            ordering, bad = served_list_ordering(self.post, self.url, self.model, self.tokenizer,
                                                 query, docs, window.window)
            if bad:
                with self._lock:
                    self.malformed.append(window.query_id)
        return ordering


def _main() -> int:
    import argparse
    from pathlib import Path

    from transformers import AutoTokenizer

    from esci_rerank.rerank.cross_encoder import text_maps
    from esci_rerank.rerank.llm_point import DEFAULT_MAX_LENGTH, warmup_scope

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="score five fold-1 windows through the server")
    smoke.add_argument("--url", default=DEFAULT_URL)
    smoke.add_argument("--served-name", required=True)
    smoke.add_argument("--model-dir", required=True)
    smoke.add_argument("--kind", choices=["point", "list"], required=True)
    smoke.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    windows_, matrix = warmup_scope()
    query_text, doc_text = text_maps(matrix, Path("data/combined/products.parquet"),
                                     Path("data/combined/judgements.parquet"))
    query_text = {str(k): v for k, v in query_text.items()}
    tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    ranker = ServedRanker(args.url, args.served_name, tok, args.kind,
                          query_text, doc_text, args.max_length)
    print("server", server_version(urllib_get, args.url))
    for w in windows_[:5]:
        print(w.query_id, ranker.order(w))
    print(f"imputed {ranker.imputed}, malformed {len(ranker.malformed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
