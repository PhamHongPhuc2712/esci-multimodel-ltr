import pandas as pd
import pytest

from src.dataset import (
    EXPECTED_LABEL_SHARE,
    EXPECTED_STATS,
    DataInvariantError,
    check_invariants,
    filter_task1,
    load_split,
)


def _judgements(labels: list[str], n_queries: int) -> pd.DataFrame:
    """A judgements frame with a given label multiset spread over n_queries."""
    n = len(labels)
    label_column = pd.Series(labels, dtype="object")
    return pd.DataFrame(
        {
            "example_id": range(n),
            "query_id": [i % n_queries for i in range(n)],
            "query": "a query",
            "product_id": [f"B{i:09d}" for i in range(n)],
            "esci_label": label_column,
            "gain": label_column.map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}),
            "qrel": label_column.map({"E": 100, "S": 10, "C": 1, "I": 0}),
        }
    )


def _test_shaped_judgements() -> pd.DataFrame:
    """A frame with exactly the documented test-split counts and label shares."""
    stats = EXPECTED_STATS["test"]
    share = EXPECTED_LABEL_SHARE["test"]
    counts = {label: round(stats.judgements * s) for label, s in share.items()}
    counts["E"] += stats.judgements - sum(counts.values())  # absorb rounding
    labels = [label for label, n in counts.items() for _ in range(n)]
    return _judgements(labels, stats.queries)


# --- filtering --------------------------------------------------------------

def test_filter_task1_keeps_only_small_version_us_rows_of_the_split():
    examples = pd.DataFrame(
        {
            "example_id": [0, 1, 2, 3],
            "query_id": [0, 0, 1, 1],
            "query": ["a", "a", "b", "b"],
            "product_id": ["B0", "B1", "B2", "B3"],
            "product_locale": ["us", "es", "us", "us"],
            "esci_label": ["E", "E", "E", "E"],
            "small_version": [1, 1, 0, 1],
            "large_version": [1, 1, 1, 1],
            "split": ["test", "test", "test", "train"],
        }
    )
    kept = filter_task1(examples, "test")
    assert list(kept["example_id"]) == [0]


def test_filter_task1_rejects_a_frame_with_no_split_column():
    # The tasksource/esci HF mirror has no `split` column. Loading it yields
    # 185,361 test judgements instead of 181,701 (~2% contamination).
    examples = pd.DataFrame(
        {
            "example_id": [0],
            "query_id": [0],
            "query": ["a"],
            "product_id": ["B0"],
            "product_locale": ["us"],
            "esci_label": ["E"],
            "small_version": [1],
            "large_version": [1],
        }
    )
    with pytest.raises(DataInvariantError, match="no 'split' column"):
        filter_task1(examples, "test")


# --- Review Focus 5: loading the wrong source file --------------------------

def test_check_invariants_accepts_the_documented_test_split():
    check_invariants("test", _test_shaped_judgements())


def test_wrong_judgement_count_names_the_hf_mirror_trap():
    frame = _test_shaped_judgements().iloc[:-1]
    with pytest.raises(DataInvariantError) as exc:
        check_invariants("test", frame)
    message = str(exc.value)
    assert "181701" in message.replace(",", "")
    assert "tasksource" in message


def test_the_hf_mirror_row_count_is_rejected():
    # The exact count the mirror produces, which is the realistic failure.
    stats = EXPECTED_STATS["test"]
    extra = 185_361 - stats.judgements
    frame = pd.concat(
        [_test_shaped_judgements(), _judgements(["E"] * extra, stats.queries)],
        ignore_index=True,
    )
    with pytest.raises(DataInvariantError, match="tasksource"):
        check_invariants("test", frame)


def test_swapped_substitute_and_complement_shares_are_rejected():
    # Complement 35% / Substitute 4% is the signature of the S/C swap bug in
    # the official prepare_trec_eval_files.py.
    stats = EXPECTED_STATS["test"]
    share = dict(EXPECTED_LABEL_SHARE["test"])
    share["S"], share["C"] = share["C"], share["S"]
    counts = {label: round(stats.judgements * s) for label, s in share.items()}
    counts["E"] += stats.judgements - sum(counts.values())
    labels = [label for label, n in counts.items() for _ in range(n)]
    with pytest.raises(DataInvariantError, match="label distribution"):
        check_invariants("test", _judgements(labels, stats.queries))


def test_wrong_query_count_is_rejected():
    frame = _test_shaped_judgements()
    frame["query_id"] = 0
    with pytest.raises(DataInvariantError, match="quer"):
        check_invariants("test", frame)


def test_train_split_has_no_label_share_expectation():
    # CLAUDE.md documents the label distribution for test only. Asserting the
    # test shares against train would be inventing a number.
    assert "train" not in EXPECTED_LABEL_SHARE
    assert EXPECTED_STATS["train"].products is None


# --- the real files ---------------------------------------------------------

@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "test"])
def test_real_split_satisfies_every_invariant(split):
    loaded = load_split(split)
    check_invariants(split, loaded.judgements, loaded.products)
    assert loaded.judgements["gain"].between(0.0, 1.0).all()
    assert set(loaded.judgements["product_id"]) <= set(loaded.products["product_id"])


@pytest.mark.data
def test_products_overlap_between_splits_is_the_documented_size():
    # 34,756 products appear in both splits. The split is query-level so this
    # is legitimate, but it is exactly why no feature may be computed from a
    # product alone using labels.
    train = set(load_split("train").judgements["product_id"])
    test = set(load_split("test").judgements["product_id"])
    assert len(train & test) == 34_756
