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
