"""TREC-format run and qrels I/O, and DataFrame -> nested-dict conversion.

Ids are strings everywhere: TREC files are text, query_id is int64 in the
parquet, and pytrec_eval rejects numpy scalars. Converting once at this
boundary means nothing downstream has to remember to.

Scores are written at full float precision rather than a fixed number of
decimals. Rounding to %.6f would turn distinct scores into ties and silently
change the ranking between an in-memory run and the same run read back.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from src.metrics import Qrels, Run


def _ranked(docs: Mapping[str, float]) -> list[tuple[str, float]]:
    """Documents ordered the way src.metrics.ndcg_per_query orders them."""
    return sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))


def qrels_from_judgements(judgements: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Nested qrels dict from a judgements frame carrying a `qrel` column."""
    duplicated = judgements.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = judgements.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}) in judgements; "
            "building a dict would silently keep only the last one"
        )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, product_id, qrel in zip(
        judgements["query_id"], judgements["product_id"], judgements["qrel"]
    ):
        qrels.setdefault(str(query_id), {})[str(product_id)] = int(qrel)
    return qrels


def run_from_scores(
    scores: pd.DataFrame, score_column: str = "score"
) -> dict[str, dict[str, float]]:
    """Nested run dict from a frame of query_id / product_id / score."""
    duplicated = scores.duplicated(subset=["query_id", "product_id"])
    if duplicated.any():
        example = scores.loc[duplicated].iloc[0]
        raise ValueError(
            f"duplicate (query_id, product_id) pair "
            f"({example['query_id']}, {example['product_id']}) in scores"
        )
    run: dict[str, dict[str, float]] = {}
    for query_id, product_id, score in zip(
        scores["query_id"], scores["product_id"], scores[score_column]
    ):
        run.setdefault(str(query_id), {})[str(product_id)] = float(score)
    return run


def write_run(run: Run, path: Path, run_tag: str) -> None:
    """Write a TREC run file: qid Q0 docid rank score tag."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(run):
            for rank, (doc_id, score) in enumerate(_ranked(run[qid]), start=1):
                # float() first: numpy 2.x reprs a float64 as "np.float64(0.5)",
                # which read_run could not parse back.
                fh.write(f"{qid} Q0 {doc_id} {rank} {float(score)!r} {run_tag}\n")


def read_run(path: Path) -> dict[str, dict[str, float]]:
    """Read a TREC run file back into a nested dict."""
    run: dict[str, dict[str, float]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}: line {line_number} has {len(fields)} fields, "
                    f"expected 6 (qid Q0 docid rank score tag): {line.strip()!r}"
                )
            qid, _, doc_id, _, score, _ = fields
            try:
                run.setdefault(qid, {})[doc_id] = float(score)
            except ValueError:
                raise ValueError(
                    f"{path}: line {line_number} has a non-numeric score {score!r}"
                ) from None
    return run


def write_qrels(qrels: Qrels, path: Path) -> None:
    """Write a TREC qrels file: qid 0 docid relevance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                fh.write(f"{qid} 0 {doc_id} {int(qrels[qid][doc_id])}\n")
