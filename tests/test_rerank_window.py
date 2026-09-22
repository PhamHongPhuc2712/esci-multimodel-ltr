import numpy as np
import pandas as pd
import pytest

from src.metrics import ndcg_per_query
from src.rerank_window import (
    DEFAULT_K,
    Window,
    splice_scores,
    spliced_run,
    stage2_run,
    windows,
)


def _stage2():
    # Query 1: five candidates, descending score. Query 2: three.
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "product_id": ["a", "b", "c", "d", "e", "x", "y", "z"],
            "stage2_score": [5.0, 4.0, 3.0, 2.0, 1.0, 9.0, 8.0, 7.0],
        }
    )


# --- carving the window -----------------------------------------------------

def test_the_window_is_the_top_k_in_stage_2_order():
    w = {x.query_id: x for x in windows(_stage2(), k=3)}
    assert w["1"].window == ("a", "b", "c")
    assert w["1"].tail == ("d", "e")


def test_the_tail_keeps_stage_2_order():
    w = {x.query_id: x for x in windows(_stage2(), k=2)}
    assert w["1"].tail == ("c", "d", "e")


def test_a_short_query_is_all_window_and_no_tail():
    w = {x.query_id: x for x in windows(_stage2(), k=10)}
    assert w["2"].window == ("x", "y", "z")
    assert w["2"].tail == ()


def test_every_query_gets_exactly_one_window():
    assert len(windows(_stage2(), k=3)) == 2


def test_ties_break_deterministically():
    # Two runs of the same code must carve the same window, or two arms are
    # re-ranking different candidate sets.
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": ["b", "a", "c"],
            "stage2_score": [1.0, 1.0, 0.0],
        }
    )
    first = windows(frame, k=2)[0].window
    second = windows(frame.iloc[::-1].reset_index(drop=True), k=2)[0].window
    assert first == second == ("a", "b")


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        windows(_stage2(), k=0)


def test_the_default_window_is_ten():
    # Measured: an oracle reorder of the top-10 holds +0.1014 of the +0.1481
    # total headroom, at a quarter of the LLM cost of the whole list.
    assert DEFAULT_K == 10


# --- Review Focus 1: the splice --------------------------------------------

def test_the_window_sits_strictly_above_the_tail():
    # A cross-encoder logit can be -8 while a lambdarank score is +3. If the
    # two scales were mixed, a demoted window item could outrank a tail item
    # the reranker never saw.
    scores = splice_scores(["c", "a", "b"], ["d", "e"])
    assert min(scores["c"], scores["a"], scores["b"]) > max(scores["d"], scores["e"])


def test_the_splice_preserves_the_given_window_order():
    scores = splice_scores(["c", "a", "b"], [])
    assert sorted(scores, key=lambda d: -scores[d]) == ["c", "a", "b"]


def test_the_splice_preserves_the_tail_order():
    scores = splice_scores(["a"], ["d", "e", "f"])
    ordered = sorted(scores, key=lambda d: -scores[d])
    assert ordered == ["a", "d", "e", "f"]


def test_the_splice_scores_every_document_once():
    scores = splice_scores(["a", "b"], ["c"])
    assert set(scores) == {"a", "b", "c"}
    assert len(set(scores.values())) == 3


def test_the_splice_rejects_a_document_in_both_halves():
    with pytest.raises(ValueError, match="both the window and the tail"):
        splice_scores(["a", "b"], ["b"])


def test_the_splice_of_nothing_is_nothing():
    assert splice_scores([], []) == {}


# --- the run ----------------------------------------------------------------

def _qrels():
    return {
        "1": {"a": 100, "b": 0, "c": 10, "d": 0, "e": 100},
        "2": {"x": 0, "y": 100, "z": 10},
    }


def test_the_identity_reranking_is_a_no_op():
    # The sharpest test of the splice: re-ranking each window with the order it
    # already had must reproduce Stage 2's NDCG exactly.
    ws = windows(_stage2(), k=3)
    identity = {w.query_id: list(w.window) for w in ws}
    before = ndcg_per_query(stage2_run(ws), _qrels())
    after = ndcg_per_query(spliced_run(ws, identity), _qrels())
    assert before == pytest.approx(after)


def test_a_query_with_no_reranking_keeps_its_stage_2_order():
    ws = windows(_stage2(), k=3)
    run = spliced_run(ws, {"2": ["z", "y", "x"]})
    ordered = sorted(run["1"], key=lambda d: -run["1"][d])
    assert ordered == ["a", "b", "c", "d", "e"]


def test_a_reranking_changes_only_the_window():
    ws = windows(_stage2(), k=3)
    run = spliced_run(ws, {"1": ["c", "b", "a"]})
    ordered = sorted(run["1"], key=lambda d: -run["1"][d])
    assert ordered == ["c", "b", "a", "d", "e"]


def test_a_reranking_that_is_not_a_permutation_raises():
    # Dropping or inventing a candidate silently changes what was ranked.
    ws = windows(_stage2(), k=3)
    with pytest.raises(ValueError, match="permutation"):
        spliced_run(ws, {"1": ["a", "b"]})
    with pytest.raises(ValueError, match="permutation"):
        spliced_run(ws, {"1": ["a", "b", "zzz"]})


def test_a_reranking_for_an_unknown_query_raises():
    ws = windows(_stage2(), k=3)
    with pytest.raises(KeyError, match="99"):
        spliced_run(ws, {"99": ["a"]})


def test_the_run_covers_every_document():
    ws = windows(_stage2(), k=2)
    run = spliced_run(ws)
    assert sum(len(docs) for docs in run.values()) == 8


def test_stage2_run_matches_the_original_scores():
    ws = windows(_stage2(), k=3)
    ordered = sorted(stage2_run(ws)["1"], key=lambda d: -stage2_run(ws)["1"][d])
    assert ordered == ["a", "b", "c", "d", "e"]


@pytest.mark.data
def test_the_identity_splice_reproduces_plan_5_on_the_real_ordering():
    from src.rank_report import qrels_from_frame
    from src.ranker import REPORT_FOLD
    from src.stage2_scores import load_stage2

    matrix = pd.read_parquet("data/features/train.parquet")
    fold0 = matrix.loc[matrix["fold"] == REPORT_FOLD]
    scores = load_stage2("train")
    scores = scores.loc[
        scores["query_id"].isin(set(fold0["query_id"]))
        & scores["product_id"].isin(set(fold0["product_id"]))
    ]
    joined = fold0[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    assert len(joined) == len(fold0)

    ws = windows(joined, k=DEFAULT_K)
    qrels = qrels_from_frame(fold0)
    base = ndcg_per_query(stage2_run(ws), qrels)
    identity = ndcg_per_query(
        spliced_run(ws, {w.query_id: list(w.window) for w in ws}), qrels
    )
    mean_base = sum(base.values()) / len(base)
    assert mean_base == pytest.approx(0.8519, abs=0.0005)
    # Not approx: an identity splice that moves NDCG at all is a broken splice.
    assert base == identity


@pytest.mark.data
def test_the_real_windows_cover_every_judged_pair():
    from src.ranker import REPORT_FOLD
    from src.stage2_scores import load_stage2

    matrix = pd.read_parquet("data/features/train.parquet")
    fold0 = matrix.loc[matrix["fold"] == REPORT_FOLD]
    scores = load_stage2("train")
    joined = fold0[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    ws = windows(joined, k=DEFAULT_K)
    assert len(ws) == fold0["query_id"].nunique()
    assert sum(len(w.documents) for w in ws) == len(fold0)
    # Measured: fold 0 has a median of 16 candidates, so most queries have a
    # tail at k=10 but a meaningful minority do not.
    assert any(w.tail for w in ws)
    assert any(not w.tail for w in ws)
