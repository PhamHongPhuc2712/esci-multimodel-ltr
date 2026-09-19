import pytest

from src.floor import random_floor, random_run


def test_uniform_gains_make_every_ordering_ideal():
    # If every judged document has the same gain, DCG == IDCG for any order,
    # so the floor is exactly 1.0. Anything else means the ideal is being
    # computed from the run rather than from the judgements.
    qrels = {f"q{i}": {f"d{j}": 100 for j in range(5)} for i in range(20)}
    assert random_floor(qrels, n_trials=5, seed=0).mean == pytest.approx(1.0)


def test_floor_of_an_all_irrelevant_query_is_zero():
    qrels = {"q": {"a": 0, "b": 0, "c": 0}}
    assert random_floor(qrels, n_trials=5, seed=0).mean == 0.0


def test_floor_sits_between_zero_and_one_on_mixed_labels():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 1, "d": 0} for i in range(50)}
    result = random_floor(qrels, n_trials=20, seed=0)
    assert 0.0 < result.mean < 1.0
    assert result.low <= result.mean <= result.high


def test_same_seed_gives_the_same_floor():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    assert random_floor(qrels, n_trials=10, seed=7).per_trial == (
        random_floor(qrels, n_trials=10, seed=7).per_trial
    )


def test_different_seeds_give_different_trials():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    assert random_floor(qrels, n_trials=10, seed=0).per_trial != (
        random_floor(qrels, n_trials=10, seed=1).per_trial
    )


def test_result_records_its_own_settings():
    qrels = {"q": {"a": 100, "b": 0}}
    result = random_floor(qrels, n_trials=4, seed=3)
    assert result.n_trials == 4
    assert result.seed == 3
    assert len(result.per_trial) == 4


def test_random_run_scores_every_judged_document_distinctly():
    import random

    qrels = {"q": {f"d{i}": 0 for i in range(100)}}
    run = random_run(qrels, random.Random(0))
    assert set(run["q"]) == set(qrels["q"])
    assert len(set(run["q"].values())) == 100


def test_zero_trials_raises():
    with pytest.raises(ValueError, match="n_trials"):
        random_floor({"q": {"a": 100}}, n_trials=0)


@pytest.mark.data
@pytest.mark.slow
def test_real_test_split_floor_is_near_the_documented_measurement():
    from src.dataset import load_split
    from src.runs import qrels_from_judgements

    qrels = qrels_from_judgements(load_split("test").judgements)
    result = random_floor(qrels, n_trials=100, seed=0)
    print(f"\nrandom floor (test) = {result.mean:.4f} [{result.low:.4f}, {result.high:.4f}]")
    # CLAUDE.md records 0.7467 measured with the 1/log2(rank+1) discount; the
    # published SQID figure is 0.7483; under the swapped S/C mapping it is
    # 0.7141.
    #
    # The band is wider than that spread on purpose. Simulating the documented
    # marginal label distribution (E 43.87 / S 34.98 / I 16.69 / C 4.46) over
    # ~20 candidates per query lands at 0.755-0.776 across i.i.d.,
    # Dirichlet-correlated and count-skewed variants -- consistently above
    # 0.7467. So the real per-query label structure is not reconstructible from
    # the marginal alone, and a band tight around 0.7467 would be a band around
    # a number this project cannot currently derive from first principles.
    #
    # What this test guards is the large moves: the S/C swap (0.7141), an
    # exponential gain, a wrong discount, or a degenerate uniform gain (1.0).
    # It is not a precision check on the fourth decimal.
    assert 0.730 < result.mean < 0.780


def test_per_query_floor_averages_to_the_headline_floor():
    qrels = {f"q{i}": {"a": 100, "b": 10, "c": 0} for i in range(30)}
    result = random_floor(qrels, n_trials=10, seed=0)
    assert set(result.per_query) == set(qrels)
    mean_of_per_query = sum(result.per_query.values()) / len(result.per_query)
    assert mean_of_per_query == pytest.approx(result.mean)
