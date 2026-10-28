"""Plan 13: a small open LLM as a pointwise reranker, scored by P("yes").

Qwen3-Reranker is a causal LM trained to answer "yes" or "no" after a fixed
template. Its score for a pair is the softmax over those two next-token
logits, which is a sigmoid of their difference. Task 4 trains on that same
difference, so the training objective and the evaluation score agree.

Two properties the rest of Plan 13 depends on:

  * **The query precedes the document.** Every candidate of a query shares
    every token up to the document, so a server with prefix caching computes
    that part once per query (Phase 3).
  * **One encoding.** `encode_pair` is the only place a pair becomes token
    ids. Training, in-process scoring and the vLLM client all call it, so the
    three cannot drift apart through a template edit or a tokenizer boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from src.cross_encoder import _lookup

DEFAULT_BACKBONE = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_MAX_LENGTH = 256

# The instruction is part of the template the model was trained with; the
# wording here is ESCI's definition of an Exact match.
INSTRUCTION = (
    "Given a shopping search query, judge whether the product satisfies "
    "every specification in the query"
)
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def query_prefix(query: str) -> str:
    """Everything before the document. Identical for all candidates of a query."""
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: "


def encode_pair(
    tokenizer, query: str, document: str, max_length: int = DEFAULT_MAX_LENGTH
) -> tuple[list[int], bool]:
    """Token ids for one pair, and whether the document was cut to fit.

    Only the document is truncated. The prefix carries the query and the
    suffix carries the answer position; cutting either would score a
    different question.
    """
    prefix = tokenizer(query_prefix(query), add_special_tokens=False).input_ids
    suffix = tokenizer(SUFFIX, add_special_tokens=False).input_ids
    budget = max_length - len(prefix) - len(suffix)
    if budget <= 0:
        raise ValueError(
            f"the template and query take {len(prefix) + len(suffix)} tokens, "
            f"leaving nothing of max_length={max_length} for the document"
        )
    body = tokenizer(document, add_special_tokens=False).input_ids
    truncated = len(body) > budget
    return list(prefix) + list(body[:budget]) + list(suffix), truncated


def yes_no_ids(tokenizer) -> tuple[int, int]:
    """The single-token ids of "yes" and "no". Refuses a tokenizer without them."""
    ids = []
    for word in ("yes", "no"):
        token_id = tokenizer.convert_tokens_to_ids(word)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"{word!r} is not a single token in this tokenizer; the "
                "P(yes) score needs both answers as one token each"
            )
        ids.append(int(token_id))
    return ids[0], ids[1]


def collate_left(
    ids: Sequence[Sequence[int]], pad_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad to the longest row, so the last column is a real token everywhere."""
    width = max(len(row) for row in ids)
    padded = [[pad_id] * (width - len(row)) + list(row) for row in ids]
    mask = [[0] * (width - len(row)) + [1] * len(row) for row in ids]
    return padded, mask


def p_yes(logit_no: float, logit_yes: float) -> float:
    """softmax([no, yes])[1], written as a sigmoid that cannot overflow."""
    z = logit_yes - logit_no
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def order_by_scores(window: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Best first. Ties break on document id, as src.cross_encoder.rerank does."""
    if len(window) != len(scores):
        raise ValueError(f"{len(scores)} scores for {len(window)} documents")
    by_doc = dict(zip(window, scores))
    return sorted(window, key=lambda d: (-by_doc[d], d))


def score_pairs(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 32,
    device: str = "cuda",
) -> list[float]:
    """P("yes") for every (query, document) pair."""
    import torch

    yes, no = yes_no_ids(tokenizer)
    pad = tokenizer.pad_token_id
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = [
                encode_pair(tokenizer, q, d, max_length)[0]
                for q, d in pairs[start : start + batch_size]
            ]
            ids, mask = collate_left(chunk, pad)
            logits = model(
                input_ids=torch.tensor(ids, device=device),
                attention_mask=torch.tensor(mask, device=device),
                logits_to_keep=1,
            ).logits[:, -1, :].float()
            for row in logits:
                scores.append(p_yes(float(row[no]), float(row[yes])))
    return scores


def rerank_point(
    model,
    tokenizer,
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, list[str]]:
    """A new ordering of each window, best first."""
    pairs: list[tuple[str, str]] = []
    for w in windows_:
        query = _lookup(query_text, w.query_id, "query text")
        pairs.extend((query, _lookup(doc_text, d, "document text")) for d in w.window)
    scores = score_pairs(
        model, tokenizer, pairs, max_length=max_length,
        batch_size=batch_size, device=device,
    )
    out: dict[str, list[str]] = {}
    position = 0
    for w in windows_:
        n = len(w.window)
        out[w.query_id] = order_by_scores(w.window, scores[position : position + n])
        position += n
    return out


def load_point_model(path: str, *, device: str = "cuda", dtype: str = "bfloat16"):
    """A causal LM and its left-padding tokenizer, for scoring."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=getattr(torch, dtype)
    ).to(device)
    return model, tokenizer
