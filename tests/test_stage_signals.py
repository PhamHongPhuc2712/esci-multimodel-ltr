import json

import numpy as np
import pandas as pd
import pytest

from src.rerank_window import Window
from src.stage_signals import (
    BLEND_FEATURES,
    MAX_FALLBACK_SHARE,
    RRF_K,
    SCOPES,
    SIGNAL_COLUMNS,
    build_signals,
    check_fallback_share,
    llm_orderings_from_cache,
    require_full_coverage,
    scope_windows,
)


class FakeCache:
    def __init__(self, entries):
        self._entries = dict(entries)

    def get(self, key):
        return self._entries.get(key)


def _windows():
    return [
        Window(query_id="1", window=("a", "b", "c"), tail=("z",)),
        Window(query_id="2", window=("x", "y"), tail=()),
    ]


def _labels():
    return pd.DataFrame(
        {
            "query_id": ["1", "1", "1", "1", "2", "2"],
            "product_id": ["a", "b", "c", "z", "x", "y"],
            "label_code": [3, 0, 2, 1, 3, 0],
            "gain": [1.0, 0.0, 0.1, 0.01, 1.0, 0.0],
            "qrel": [100, 0, 10, 1, 100, 0],
            "stage2_score": [3.0, 2.0, 1.0, 0.5, 9.0, 8.0],
        }
    )


def _ce():
    return {("1", "a"): 0.1, ("1", "b"): 0.9, ("1", "c"): 0.5,
            ("2", "x"): 0.2, ("2", "y"): 0.8}


def _llm():
    return {"1": ["c", "a", "b"], "2": ["y", "x"]}


def _frame(**kwargs):
    arguments = dict(ce_scores=_ce(), llm_orderings=_llm()) | kwargs
    return build_signals(_windows(), _labels(), **arguments)


# --- the frame --------------------------------------------------------------

def test_the_frame_has_one_row_per_window_document():
    frame = _frame()
    assert len(frame) == 5          # 3 + 2 window documents; the tail is not scored
    assert list(frame.columns) == list(SIGNAL_COLUMNS)


def test_the_tail_is_not_in_the_frame():
    # Stage 3 never looked at it, so no stage has an opinion to blend.
    assert "z" not in set(_frame()["product_id"])


def test_stage_2_rank_is_the_window_position():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["a", "stage2_rank"] == 1
    assert q1.loc["c", "stage2_rank"] == 3


def test_stage_2_score_travels_from_the_labels():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["b", "stage2_score"] == pytest.approx(2.0)


def test_llm_rank_is_the_position_in_the_llm_ordering():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["c", "llm_rank"] == 1      # the LLM put c first
    assert q1.loc["b", "llm_rank"] == 3


def test_reciprocal_rank_uses_the_stage_1_constant():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["c", "llm_rr"] == pytest.approx(1.0 / (RRF_K + 1))
    assert q1.loc["c", "llm_rr"] > q1.loc["b", "llm_rr"]


def test_labels_travel_with_the_signals():
    q1 = _frame().query("query_id == '1'").set_index("product_id")
    assert q1.loc["a", "label_code"] == 3
    assert q1.loc["a", "qrel"] == 100


def test_every_blend_feature_is_a_column():
    assert set(BLEND_FEATURES) <= set(_frame().columns)


def test_the_answer_is_not_a_blend_feature():
    # gain, qrel and label_code are the label. llm_fallback is bookkeeping:
    # three windows carry it, and a combiner that learned from it would be
    # learning which queries made the LLM malfunction.
    assert not {"gain", "qrel", "label_code", "llm_fallback"} & set(BLEND_FEATURES)


def test_a_missing_cross_encoder_score_raises():
    with pytest.raises(KeyError, match="cross-encoder"):
        _frame(ce_scores={})


def test_a_missing_label_raises():
    # A window document with no judgement would train the blend on a NaN.
    labels = _labels().drop(index=0)
    with pytest.raises(KeyError, match="label"):
        build_signals(_windows(), labels, ce_scores=_ce(), llm_orderings=_llm())


def test_labels_without_a_stage_2_score_raise():
    # The written plan let this through as NaN, and require_full_coverage then
    # rejected every frame - including the one its own test called complete.
    labels = _labels().drop(columns=["stage2_score"])
    with pytest.raises(KeyError, match="stage2_score"):
        build_signals(_windows(), labels, ce_scores=_ce(), llm_orderings=_llm())


# --- Review Focus 2: the cache covers these windows, or it says so ---------

def test_cached_permutations_become_orderings():
    from src.llm_rerank import window_key

    query_text = {"1": "red shoes", "2": "blue hat"}
    cache = FakeCache({
        window_key("red shoes", ["a", "b", "c"]): [3, 1, 2],
        window_key("blue hat", ["x", "y"]): [2, 1],
    })
    orderings, missing = llm_orderings_from_cache(_windows(), query_text, cache)
    assert orderings["1"] == ["c", "a", "b"]
    assert missing == []


def test_a_cache_miss_is_reported_not_papered_over():
    # The cache covers fold 0 and the 2,000-query test sample, nothing else.
    # Falling back silently here would label 6,956 test queries `llm` and
    # report the dilution as a blend effect.
    query_text = {"1": "red shoes", "2": "blue hat"}
    orderings, missing = llm_orderings_from_cache(_windows(), query_text, FakeCache({}))
    assert sorted(missing) == ["1", "2"]
    assert orderings == {}


def test_a_permutation_of_the_wrong_length_counts_as_a_miss():
    from src.llm_rerank import window_key

    query_text = {"1": "red shoes", "2": "blue hat"}
    cache = FakeCache({window_key("red shoes", ["a", "b", "c"]): [2, 1]})
    _, missing = llm_orderings_from_cache(_windows(), query_text, cache)
    assert "1" in missing


def test_a_window_with_no_ordering_and_no_fallback_flag_raises():
    # build_signals itself refuses the silent fallback; the CLI is not the
    # only guard.
    with pytest.raises(KeyError, match="LLM ordering"):
        _frame(llm_orderings={"1": ["c", "a", "b"]})


def test_a_declared_fallback_keeps_the_stage_2_order_and_is_flagged():
    # Exactly how Plan 6 scored a window whose LLM answer was malformed, so
    # the frame reproduces Plan 6's arm rather than a different one.
    frame = _frame(llm_orderings={"1": ["c", "a", "b"]}, llm_fallbacks=["2"])
    q2 = frame.query("query_id == '2'").set_index("product_id")
    assert list(q2["llm_rank"]) == list(q2["stage2_rank"])
    assert q2["llm_fallback"].all()
    assert not frame.query("query_id == '1'")["llm_fallback"].any()


def test_a_query_cannot_be_both_ranked_and_a_fallback():
    with pytest.raises(ValueError, match="fallback"):
        _frame(llm_fallbacks=["1"])


def test_an_llm_ordering_that_is_not_a_permutation_raises():
    with pytest.raises(ValueError, match="permutation"):
        _frame(llm_orderings={"1": ["a", "a", "b"], "2": ["y", "x"]})


def test_the_measured_handful_of_fallbacks_is_accepted():
    check_fallback_share(["55755"], 4130)                 # fold 0, measured
    check_fallback_share(["49855", "66028"], 2000)        # test sample, measured


def test_a_scope_error_is_refused():
    # The full test split against a cache that covers 2,000 of its queries.
    missing = [str(i) for i in range(6956)]
    with pytest.raises(ValueError, match="scope"):
        check_fallback_share(missing, 8956)


def test_the_fallback_ceiling_sits_between_the_measurement_and_a_scope_error():
    assert 2 / 2000 <= MAX_FALLBACK_SHARE < 6956 / 8956


def test_require_full_coverage_passes_a_complete_frame():
    require_full_coverage(_frame())      # must not raise


def test_require_full_coverage_rejects_a_null_signal():
    frame = _frame()
    frame.loc[0, "ce_score"] = np.nan
    with pytest.raises(ValueError, match="incomplete"):
        require_full_coverage(frame)


def test_the_scopes_are_the_two_the_llm_actually_ran_on():
    assert SCOPES == ("fold0", "test-sample")


def test_an_unknown_scope_is_refused_before_anything_is_read(tmp_path):
    with pytest.raises(ValueError, match="scope"):
        scope_windows("test", features_dir=tmp_path)


# --- the real frames --------------------------------------------------------

_PLAN_6_RESULTS = {
    "fold0": "docs/results/fine-rank.json",
    "test-sample": "docs/results/fine-rank-test.json",
}


def _ordering(frame, column, ascending):
    out = {}
    for query_id, group in frame.groupby("query_id", sort=False):
        pairs = sorted(
            zip(group[column], group["product_id"]),
            key=lambda sd: (sd[0] if ascending else -sd[0], sd[1]),
        )
        out[str(query_id)] = [doc for _, doc in pairs]
    return out


@pytest.mark.data
def test_the_real_frames_are_complete():
    from src.stage_signals import load_signals

    for scope, n_queries, n_fallback in [("fold0", 4130, 1), ("test-sample", 2000, 2)]:
        frame = load_signals(scope)
        assert frame["query_id"].nunique() == n_queries
        require_full_coverage(frame)
        # Every window is at most K documents and at least one.
        sizes = frame.groupby("query_id").size()
        assert sizes.max() <= 10
        assert sizes.min() >= 1
        # The windows whose LLM answer was malformed, as Plan 6 counted them.
        assert frame.groupby("query_id")["llm_fallback"].first().sum() == n_fallback
        # The three stages disagree; if any two are identical the frame is
        # carrying one stage's opinion twice under two names.
        assert not frame["ce_score"].equals(frame["stage2_score"])
        assert (frame["llm_rank"] != frame["stage2_rank"]).any()


@pytest.mark.data
def test_the_frames_reproduce_plan_6s_three_arms():
    # The frame is only the stage scores if ordering by each column gives back
    # the NDCG Plan 6 published for that stage. This also pins the test sample:
    # a different 2,000 queries could not land on all three numbers.
    from src.metrics import ndcg_per_query
    from src.rank_report import qrels_from_frame
    from src.rerank_window import spliced_run
    from src.stage_signals import load_signals

    for scope, path in _PLAN_6_RESULTS.items():
        published = {
            arm["name"]: arm["ndcg"]["point"]
            for arm in json.loads(open(path, encoding="utf-8").read())["arms"]
        }
        frame = load_signals(scope)
        windows_, matrix = scope_windows(scope)
        qrels = qrels_from_frame(matrix)
        for arm, column, ascending in [
            ("stage2", "stage2_rank", True),
            ("stage2+ce", "ce_score", False),
            ("stage2+llm", "llm_rank", True),
        ]:
            run = spliced_run(windows_, _ordering(frame, column, ascending))
            per_query = ndcg_per_query(run, qrels)
            measured = sum(per_query.values()) / len(per_query)
            assert measured == pytest.approx(published[arm], abs=5e-4), (scope, arm)
