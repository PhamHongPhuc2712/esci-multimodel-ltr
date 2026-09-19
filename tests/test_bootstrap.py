import numpy as np
import pytest

from src.bootstrap import bootstrap_ci, cluster_delta_ci, paired_delta_ci


def test_point_estimate_is_the_plain_mean():
    per_query = {"q1": 0.2, "q2": 0.4, "q3": 0.9}
    assert bootstrap_ci(per_query, n_resamples=200).point == pytest.approx(0.5)


def test_constant_scores_give_a_zero_width_interval():
    per_query = {f"q{i}": 0.8 for i in range(50)}
    interval = bootstrap_ci(per_query, n_resamples=200)
    assert interval.low == pytest.approx(0.8)
    assert interval.high == pytest.approx(0.8)


def test_interval_brackets_the_point_estimate():
    per_query = {f"q{i}": i / 100 for i in range(100)}
    interval = bootstrap_ci(per_query, n_resamples=500)
    assert interval.low <= interval.point <= interval.high


def test_more_queries_narrow_the_interval():
    small = {f"q{i}": (i % 10) / 10 for i in range(20)}
    large = {f"q{i}": (i % 10) / 10 for i in range(2000)}
    narrow = bootstrap_ci(large, n_resamples=500)
    wide = bootstrap_ci(small, n_resamples=500)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_same_seed_reproduces_the_interval():
    per_query = {f"q{i}": (i % 7) / 7 for i in range(100)}
    assert bootstrap_ci(per_query, n_resamples=300, seed=5) == bootstrap_ci(
        per_query, n_resamples=300, seed=5
    )


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci({}, n_resamples=10)


# --- paired deltas ----------------------------------------------------------

def test_identical_methods_have_a_zero_delta_and_a_zero_width_interval():
    per_query = {f"q{i}": (i % 5) / 5 for i in range(100)}
    interval = paired_delta_ci(per_query, per_query, n_resamples=300)
    assert interval.point == pytest.approx(0.0)
    assert interval.low == pytest.approx(0.0)
    assert interval.high == pytest.approx(0.0)


def test_a_uniform_improvement_gives_an_interval_strictly_above_zero():
    baseline = {f"q{i}": 0.5 for i in range(200)}
    better = {f"q{i}": 0.6 for i in range(200)}
    interval = paired_delta_ci(better, baseline, n_resamples=500)
    assert interval.point == pytest.approx(0.1)
    assert interval.low > 0


def test_a_noisy_tie_gives_an_interval_that_straddles_zero():
    # The spec calls a method that ties the baseline a legitimate, reportable
    # result, which is only expressible if the interval can contain zero.
    baseline = {f"q{i}": (i % 11) / 11 for i in range(400)}
    noisy = {f"q{i}": (i % 11) / 11 + (0.02 if i % 2 else -0.02) for i in range(400)}
    interval = paired_delta_ci(noisy, baseline, n_resamples=500)
    assert interval.low < 0 < interval.high


def test_pairing_is_tighter_than_treating_the_methods_as_independent():
    # Both methods vary a lot across queries but differ by a constant. Pairing
    # cancels the shared variance; without it the interval would be wide.
    baseline = {f"q{i}": (i % 50) / 50 for i in range(500)}
    better = {f"q{i}": (i % 50) / 50 + 0.05 for i in range(500)}
    interval = paired_delta_ci(better, baseline, n_resamples=500)
    assert interval.high - interval.low < 1e-9


def test_mismatched_query_sets_raise():
    a = {"q1": 0.5, "q2": 0.5}
    b = {"q1": 0.5, "q3": 0.5}
    with pytest.raises(ValueError, match="same queries"):
        paired_delta_ci(a, b, n_resamples=10)


def test_cluster_delta_is_the_difference_of_the_two_group_means():
    values = np.array([1.0, 1.0, 0.0, 0.0])
    mask = np.array([True, True, False, False])
    clusters = np.array([0, 1, 2, 3])
    assert cluster_delta_ci(values, mask, clusters, n_resamples=200).point == (
        pytest.approx(1.0)
    )


def test_cluster_delta_of_identical_groups_straddles_zero():
    rng = np.random.default_rng(0)
    values = rng.normal(size=2000)
    mask = np.arange(2000) % 2 == 0
    clusters = np.arange(2000) // 4
    interval = cluster_delta_ci(values, mask, clusters, n_resamples=400)
    assert interval.low < 0 < interval.high


def test_cluster_delta_resamples_clusters_not_rows():
    # Every row in a cluster shares a value, so resampling rows would shrink
    # the interval by pretending there are 2,000 independent observations
    # when there are 100. Judgements are clustered ~20 to a query, so this is
    # the difference between a real interval and a falsely confident one.
    clusters = np.repeat(np.arange(100), 20)
    values = np.repeat(np.random.default_rng(1).normal(size=100), 20)
    mask = clusters % 2 == 0
    interval = cluster_delta_ci(values, mask, clusters, n_resamples=400)
    assert interval.high - interval.low > 0.1


def test_cluster_delta_is_reproducible_for_a_given_seed():
    values = np.arange(200, dtype=float)
    mask = np.arange(200) % 2 == 0
    clusters = np.arange(200) // 5
    assert cluster_delta_ci(values, mask, clusters, seed=4) == cluster_delta_ci(
        values, mask, clusters, seed=4
    )


def test_cluster_delta_raises_when_one_group_is_empty():
    values = np.array([1.0, 2.0])
    mask = np.array([True, True])
    with pytest.raises(ValueError, match="both groups"):
        cluster_delta_ci(values, mask, np.array([0, 1]), n_resamples=10)
