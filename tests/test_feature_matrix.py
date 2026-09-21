import math

import numpy as np
import pandas as pd
import pytest

from src.feature_matrix import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_GROUPS,
    LABEL_CODES,
    build_matrix,
    category_vocabularies,
    group_sizes,
    select_columns,
    sort_for_ranking,
)
from src.labels import ESCI_GAINS


# --- the groups -------------------------------------------------------------

def test_the_groups_are_disjoint():
    # An arm is a list of group names. If a column were in two groups, an arm
    # that excludes one of them would still carry the column.
    flat = [c for group in FEATURE_GROUPS.values() for c in group]
    assert len(flat) == len(set(flat))


def test_all_features_is_every_group_flattened():
    flat = [c for group in FEATURE_GROUPS.values() for c in group]
    assert sorted(ALL_FEATURES) == sorted(flat)
    assert len(ALL_FEATURES) == 47


def test_select_columns_returns_the_named_groups_in_order():
    assert select_columns(["retrieval"]) == ["bm25_score", "dense_sim"]
    assert select_columns(["retrieval", "image"]) == [
        "bm25_score", "dense_sim", "clip_image_sim", "has_image_vector",
    ]


def test_select_columns_rejects_an_unknown_group():
    with pytest.raises(KeyError, match="behavioral"):
        select_columns(["behavioral"])  # the American spelling is not a group


def test_the_image_indicator_travels_with_the_image_score():
    # A model given clip_image_sim and no way to tell a low cosine from an
    # absent one is being asked to read NaN as a number.
    assert "has_image_vector" in FEATURE_GROUPS["image"]
    assert "has_image_vector" not in FEATURE_GROUPS["s_indicators"]


def test_the_two_presence_groups_stay_apart():
    # CLAUDE.md: ESCI presence is legitimate, ESCI-S presence is an artefact.
    assert set(FEATURE_GROUPS["esci_indicators"]).isdisjoint(
        FEATURE_GROUPS["s_indicators"]
    )


# --- the label codes --------------------------------------------------------

def test_label_codes_are_ordered_by_gain():
    # LightGBM's label_gain is indexed by the integer label, and it requires a
    # non-decreasing list. Codes out of gain order would silently invert the
    # objective.
    gains = [ESCI_GAINS[label] for label, _ in sorted(LABEL_CODES.items(), key=lambda kv: kv[1])]
    assert gains == sorted(gains)
    assert LABEL_CODES == {"I": 0, "C": 1, "S": 2, "E": 3}


# --- group_sizes: Review Focus 2 -------------------------------------------

def test_group_sizes_counts_consecutive_runs():
    assert group_sizes([1, 1, 1, 2, 2]).tolist() == [3, 2]


def test_group_sizes_rejects_a_query_split_across_the_frame():
    # LightGBM's `group` is sizes over consecutive rows - it never sees a query
    # id. A frame sorted any other way trains on comparisons that straddle
    # queries, and nothing raises.
    with pytest.raises(ValueError, match="not contiguous"):
        group_sizes([1, 2, 1])


def test_group_sizes_of_nothing_is_empty():
    assert group_sizes([]).tolist() == []


def test_group_sizes_sum_to_the_row_count():
    ids = [1, 1, 2, 3, 3, 3]
    assert int(group_sizes(ids).sum()) == len(ids)


def test_sort_for_ranking_makes_every_query_contiguous():
    frame = pd.DataFrame(
        {"query_id": [2, 1, 2, 1], "product_id": ["b", "a", "a", "b"]}
    )
    ordered = sort_for_ranking(frame)
    assert ordered["query_id"].tolist() == [1, 1, 2, 2]
    # Deterministic within a query too, so two runs of the same code produce
    # byte-identical matrices.
    assert ordered["product_id"].tolist() == ["a", "b", "a", "b"]
    group_sizes(ordered["query_id"])  # must not raise


# --- categorical vocabularies ----------------------------------------------

def test_category_vocabularies_are_sorted_and_shared():
    products = pd.DataFrame(
        {"s_template": ["kitchen", "apparel", None, "kitchen"], "s_type": ["product", "book", None, "product"]}
    )
    vocab = category_vocabularies(products)
    assert vocab["s_template"] == ["apparel", "kitchen"]
    assert vocab["s_type"] == ["book", "product"]


def test_the_same_vocabulary_gives_the_same_codes_in_two_frames():
    # pandas category codes are positional. A dtype built per split would give
    # "kitchen" code 1 in one frame and code 0 in another, and LightGBM would
    # be told they are the same feature.
    vocab = {"s_template": ["apparel", "book", "kitchen"], "s_type": ["book", "product"]}
    a = build_matrix(*_tiny(template="kitchen"), categories=vocab)
    b = build_matrix(*_tiny(template="kitchen", query_id=9), categories=vocab)
    assert a["s_template"].cat.categories.tolist() == b["s_template"].cat.categories.tolist()
    assert a["s_template"].cat.codes.tolist() == b["s_template"].cat.codes.tolist()


# --- build_matrix -----------------------------------------------------------

def _tiny(template="kitchen", query_id=1):
    judgements = pd.DataFrame(
        {
            "query_id": [query_id, query_id],
            "query": ["steel water bottle", "steel water bottle"],
            "product_id": ["p1", "p2"],
            "esci_label": ["E", "I"],
            "gain": [1.0, 0.0],
            "qrel": [100, 0],
            "split": ["train", "train"],
            "fold": [0, 0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_title": ["Steel Water Bottle", "Plastic Cup"],
            "product_description": ["Holds water.", None],
            "product_bullet_point": ["Insulated", None],
            "product_brand": ["Hydro", None],
            "product_color": ["Silver", None],
            "s_template": [template, None],
            "s_type": ["product", None],
            "s_category": [["Kitchen", "Bottles"], None],
            "s_attrs_json": ['{"Material": "Steel"}', None],
            "s_info_json": ['{"Item Weight": "1 lb"}', None],
            "s_stars": [4.5, None],
            "s_ratings": [10.0, None],
            "s_price": [19.99, None],
            "s_price_multi": [False, None],
            "s_bsr_rank": [1234.0, None],
            "s_n_reviews": [3.0, None],
            "has_enrichment": [True, False],
            "description_source": ["esci", "none"],
        }
    )
    pair_scores = pd.DataFrame(
        {
            "query_id": [query_id, query_id],
            "product_id": ["p1", "p2"],
            "bm25_score": [12.5, 3.0],
            "dense_sim": [0.8, 0.2],
            "clip_image_sim": [0.6, np.nan],
        }
    )
    return judgements, products, pair_scores


def test_build_matrix_emits_every_feature_once():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    for name in ALL_FEATURES:
        assert name in frame.columns, name
    assert len(frame) == 2


def test_build_matrix_carries_the_label_columns():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    assert frame.loc[frame["product_id"] == "p1", "label_code"].iloc[0] == LABEL_CODES["E"]
    assert frame.loc[frame["product_id"] == "p2", "label_code"].iloc[0] == LABEL_CODES["I"]
    assert set(["query_id", "product_id", "esci_label", "gain", "qrel", "split", "fold"]) <= set(frame.columns)


def test_no_feature_column_is_derived_from_a_label():
    # 34,756 products appear in both train and test. A feature computed from
    # labels per product leaks. Flipping every label must not move a feature.
    judgements, products, scores = _tiny()
    flipped = judgements.copy()
    flipped["esci_label"] = ["I", "E"]
    flipped["gain"] = [0.0, 1.0]
    flipped["qrel"] = [0, 100]
    vocab = category_vocabularies(products)

    before = build_matrix(judgements, products, scores, categories=vocab)
    after = build_matrix(flipped, products, scores, categories=vocab)
    pd.testing.assert_frame_equal(
        before[list(ALL_FEATURES)], after[list(ALL_FEATURES)], check_dtype=True
    )


def test_the_image_indicator_follows_the_image_score():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    p1 = frame.loc[frame["product_id"] == "p1"].iloc[0]
    p2 = frame.loc[frame["product_id"] == "p2"].iloc[0]
    assert p1["has_image_vector"] == 1.0
    assert p2["has_image_vector"] == 0.0
    assert math.isnan(p2["clip_image_sim"])


def test_build_matrix_never_fills_a_nan_similarity():
    # Review Focus 5 again, one layer up: the assembly must not impute.
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    assert frame["clip_image_sim"].isna().sum() == 1


def test_build_matrix_raises_when_a_pair_has_no_scores():
    judgements, products, scores = _tiny()
    with pytest.raises(ValueError, match="pair scores"):
        build_matrix(judgements, products, scores.iloc[:1], categories=category_vocabularies(products))


def test_build_matrix_raises_when_a_judged_product_is_missing():
    judgements, products, scores = _tiny()
    with pytest.raises(ValueError, match="no product row"):
        build_matrix(judgements, products.iloc[:1], scores, categories=category_vocabularies(products))


def test_the_matrix_comes_back_sorted_for_ranking():
    judgements, products, scores = _tiny()
    judgements = judgements.iloc[::-1].reset_index(drop=True)
    frame = build_matrix(judgements, products, scores, categories=category_vocabularies(products))
    group_sizes(frame["query_id"])  # must not raise


def test_categorical_columns_have_the_category_dtype():
    frame = build_matrix(*_tiny(), categories=category_vocabularies(_tiny()[1]))
    for name in CATEGORICAL_FEATURES:
        assert isinstance(frame[name].dtype, pd.CategoricalDtype), name


@pytest.mark.data
def test_the_real_matrices_match_the_asserted_invariants():
    train = pd.read_parquet("data/features/train.parquet")
    test = pd.read_parquet("data/features/test.parquet")

    assert len(train) == 419_653
    assert len(test) == 181_701
    assert train["query_id"].nunique() == 20_888
    assert test["query_id"].nunique() == 8_956

    for frame in (train, test):
        sizes = group_sizes(frame["query_id"])   # contiguity, Review Focus 2
        assert int(sizes.sum()) == len(frame)
        assert set(ALL_FEATURES) <= set(frame.columns)

    # Measured 2026-09-21 over the 482,105 judged products; the share over
    # judgements is close but not identical, so the bands are wide.
    coverage = train[list(ALL_FEATURES)].notna().mean()
    assert 0.20 < coverage["s_price"] < 0.35
    assert 0.80 < coverage["s_stars"] < 0.95
    assert 0.70 < coverage["clip_image_sim"] < 0.85
    assert coverage["dense_sim"] == pytest.approx(1.0)
    # Indicators answer "was it there", so they are never themselves missing.
    for name in FEATURE_GROUPS["esci_indicators"] + FEATURE_GROUPS["s_indicators"]:
        assert coverage[name] == pytest.approx(1.0), name


@pytest.mark.data
def test_the_two_splits_agree_on_categorical_codes():
    train = pd.read_parquet("data/features/train.parquet")
    test = pd.read_parquet("data/features/test.parquet")
    for name in CATEGORICAL_FEATURES:
        assert (
            train[name].cat.categories.tolist() == test[name].cat.categories.tolist()
        ), name
