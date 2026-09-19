"""Full-list NDCG for ESCI Task 1.

Matches `trec_eval -m ndcg`: gain enters linearly (not 2**rel - 1), the
discount is 1/log2(rank + 1) with rank starting at 1, and there is no cutoff.
The whole judged candidate list is ranked, which is why the random floor for
this task sits near 0.75 rather than near 0.

Data is shaped as nested dicts, qid -> doc_id -> value, which is also
pytrec_eval's interface; tests/test_metrics_vs_pytrec.py uses it as an oracle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

Qrels = Mapping[str, Mapping[str, float]]
Run = Mapping[str, Mapping[str, float]]


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain of an already-ordered list of gains."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_per_query(run: Run, qrels: Qrels) -> dict[str, float]:
    """NDCG for every query in `qrels`.

    Iterates `qrels`, not `run`, so that:
      - documents the run failed to return still count towards the ideal,
      - a query the run skipped entirely is an error rather than a silent
        shrinking of the denominator.

    Documents in the run with no judgement count as gain 0 and still occupy a
    rank, matching trec_eval. Ties are broken by ascending document id so the
    same run file scores identically on any machine. A query whose ideal DCG
    is 0 (no positive gain anywhere) scores 0.0, also matching trec_eval.
    """
    scores: dict[str, float] = {}
    for qid, judged in qrels.items():
        if qid not in run:
            raise KeyError(
                f"query {qid!r} is in qrels but not in the run; "
                "a run must score every judged query"
            )
        ranked = sorted(run[qid].items(), key=lambda kv: (-kv[1], kv[0]))
        dcg = _dcg([float(judged.get(doc_id, 0.0)) for doc_id, _ in ranked])
        idcg = _dcg(sorted((float(g) for g in judged.values()), reverse=True))
        scores[qid] = 0.0 if idcg == 0.0 else dcg / idcg
    return scores


def mean_ndcg(run: Run, qrels: Qrels) -> float:
    """NDCG averaged over queries, weighting every query equally."""
    per_query = ndcg_per_query(run, qrels)
    if not per_query:
        raise ValueError("no queries to score: qrels is empty")
    return sum(per_query.values()) / len(per_query)
