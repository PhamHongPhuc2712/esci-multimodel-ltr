import random

import pytest

from src.labels import ESCI_QRELS
from src.metrics import ndcg_per_query

pytrec_eval = pytest.importorskip(
    "pytrec_eval",
    reason="install the dev extra: pip install -e '.[dev]'",
)

LABELS = list(ESCI_QRELS)


def _random_case(seed: int, n_queries: int = 40, max_docs: int = 25):
    """Build an ESCI-shaped qrels/run pair with distinct scores.

    Scores are a shuffled range so no two documents tie: trec_eval breaks ties
    by *descending* document id while this project breaks them by ascending id,
    and that difference is deliberate and tested separately. Feeding ties to
    the oracle would compare tie policies rather than NDCG.
    """
    rng = random.Random(seed)
    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q in range(n_queries):
        qid = f"q{q}"
        n_docs = rng.randint(1, max_docs)
        doc_ids = [f"B{q:03d}{d:04d}" for d in range(n_docs)]
        qrels[qid] = {d: ESCI_QRELS[rng.choice(LABELS)] for d in doc_ids}
        scores = list(range(n_docs))
        rng.shuffle(scores)
        run[qid] = {d: float(s) for d, s in zip(doc_ids, scores)}
    return qrels, run


@pytest.mark.parametrize("seed", range(10))
def test_matches_pytrec_eval_on_random_esci_shaped_data(seed):
    qrels, run = _random_case(seed)
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    ours = ndcg_per_query(run, qrels)
    assert set(ours) == set(oracle)
    for qid, expected in oracle.items():
        assert ours[qid] == pytest.approx(expected["ndcg"], abs=1e-9), qid


def test_matches_pytrec_eval_when_the_run_has_unjudged_documents():
    qrels, run = _random_case(seed=99)
    for qid, docs in run.items():
        docs[f"UNJUDGED-{qid}"] = 1e6  # ranked first, no judgement
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    ours = ndcg_per_query(run, qrels)
    for qid, expected in oracle.items():
        assert ours[qid] == pytest.approx(expected["ndcg"], abs=1e-9), qid


def test_matches_pytrec_eval_when_the_run_omits_judged_documents():
    qrels, run = _random_case(seed=7)
    for qid, docs in run.items():
        if len(docs) > 1:
            docs.pop(next(iter(docs)))  # drop one retrieved document
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    ours = ndcg_per_query(run, qrels)
    for qid, expected in oracle.items():
        assert ours[qid] == pytest.approx(expected["ndcg"], abs=1e-9), qid


def test_matches_pytrec_eval_when_a_query_has_no_relevant_document():
    qrels = {"q": {"a": 0, "b": 0}}
    run = {"q": {"a": 2.0, "b": 1.0}}
    oracle = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run)
    assert ndcg_per_query(run, qrels)["q"] == pytest.approx(oracle["q"]["ndcg"], abs=1e-9)
