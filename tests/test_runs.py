import pandas as pd
import pytest

from src.runs import (
    qrels_from_judgements,
    read_run,
    run_from_scores,
    write_qrels,
    write_run,
)


def _judgements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "product_id": ["B1", "B2", "B3"],
            "esci_label": ["E", "I", "S"],
            "qrel": [100, 0, 10],
        }
    )


def test_qrels_keys_are_strings_not_numpy_integers():
    qrels = qrels_from_judgements(_judgements())
    assert set(qrels) == {"1", "2"}
    assert all(isinstance(k, str) for k in qrels)
    assert qrels["1"] == {"B1": 100, "B2": 0}


def test_qrel_values_are_plain_ints():
    # pytrec_eval rejects numpy.int64.
    qrels = qrels_from_judgements(_judgements())
    assert all(type(v) is int for docs in qrels.values() for v in docs.values())


def test_duplicate_query_product_pair_raises_instead_of_overwriting():
    duplicated = pd.concat([_judgements(), _judgements().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        qrels_from_judgements(duplicated)


def test_run_from_scores_shapes_a_nested_dict():
    scores = pd.DataFrame(
        {"query_id": [1, 1], "product_id": ["B1", "B2"], "score": [0.5, 0.25]}
    )
    assert run_from_scores(scores) == {"1": {"B1": 0.5, "B2": 0.25}}


def test_run_file_round_trip_preserves_scores_exactly(tmp_path):
    # Formatting scores as %.6f would collapse near-identical scores into ties
    # and change the ranking, so the round trip must be exact.
    run = {"1": {"B1": 0.1234567890123456, "B2": 0.1234567890123457}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    assert read_run(path) == run


def test_run_file_has_the_six_trec_columns_in_rank_order(tmp_path):
    run = {"1": {"B2": 0.5, "B1": 0.9}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    rows = [line.split() for line in path.read_text().splitlines()]
    assert [r[0] for r in rows] == ["1", "1"]
    assert [r[1] for r in rows] == ["Q0", "Q0"]
    assert [r[2] for r in rows] == ["B1", "B2"]  # sorted by descending score
    assert [r[3] for r in rows] == ["1", "2"]  # rank is 1-based
    assert [r[5] for r in rows] == ["unit", "unit"]


def test_run_file_ties_are_written_in_ascending_document_id(tmp_path):
    # Must match the tie policy in src/metrics.py, or a run scores differently
    # after a round trip through disk.
    run = {"1": {"B2": 0.5, "B1": 0.5}}
    path = tmp_path / "run.trec"
    write_run(run, path, run_tag="unit")
    assert [line.split()[2] for line in path.read_text().splitlines()] == ["B1", "B2"]


def test_qrels_file_has_the_four_trec_columns(tmp_path):
    path = tmp_path / "qrels.txt"
    write_qrels({"1": {"B1": 100, "B2": 0}}, path)
    rows = [line.split() for line in path.read_text().splitlines()]
    assert rows == [["1", "0", "B1", "100"], ["1", "0", "B2", "0"]]


def test_reading_a_malformed_run_line_raises_with_the_line_number(tmp_path):
    path = tmp_path / "run.trec"
    path.write_text("1 Q0 B1 1 0.9 tag\nnot a run line\n")
    with pytest.raises(ValueError, match="line 2"):
        read_run(path)
