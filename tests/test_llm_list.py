"""Listwise LLM reranker: prompt, target, encoding, parsing. No model."""

import pandas as pd
import pytest

from esci_rerank.rerank.llm_list import (
    MAX_MALFORMED_SHARE,
    answer_text,
    check_malformed_share,
    collate_right,
    encode_example,
    list_training_set,
    target_permutation,
    user_message,
)
from esci_rerank.rerank.llm_rerank import parse_permutation
from esci_rerank.rerank.rerank_window import Window


class ChatTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            pass

        encoded = Encoded()
        encoded.input_ids = [self._id(w) for w in text.split()]
        return encoded

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        assert enable_thinking is False
        return " ".join(m["content"] for m in messages) + " ASSISTANT"


def test_target_keeps_stage2_order_inside_a_grade():
    # window order: S, E, I, E -> Es first in window order, then S, then I.
    assert target_permutation([0.1, 1.0, 0.0, 1.0]) == [2, 4, 1, 3]


def test_the_answer_round_trips_through_the_project_parser():
    permutation = [3, 1, 2]
    assert parse_permutation(answer_text(permutation), 3) == permutation


def test_documents_are_numbered_and_cut():
    message = user_message("shoes", ["a" * 500, "b"])
    assert "[1] " + "a" * 300 in message
    assert "a" * 301 not in message
    assert "[2] b" in message


def test_only_answer_tokens_carry_labels():
    tok = ChatTokenizer()
    ids, labels = encode_example(tok, "shoes", ["red shoe", "blue sock"], [0.0, 1.0])
    assert len(ids) == len(labels)
    answer = tok(answer_text([2, 1]) + " <|im_end|>").input_ids
    assert labels[-len(answer):] == answer
    assert all(label == -100 for label in labels[: -len(answer)])


def test_right_collation_masks_padding_out_of_the_loss():
    ids, labels, mask = collate_right([([5, 6, 7], [-100, 6, 7]), ([8], [8])], pad_id=0)
    assert ids == [[5, 6, 7], [8, 0, 0]]
    assert labels == [[-100, 6, 7], [8, -100, -100]]
    assert mask == [[1, 1, 1], [1, 0, 0]]


def test_too_many_malformed_answers_refuse_the_run():
    n = 1000
    check_malformed_share(["q"] * int(n * MAX_MALFORMED_SHARE), n)
    with pytest.raises(ValueError, match="malformed"):
        check_malformed_share(["q"] * (int(n * MAX_MALFORMED_SHARE) + 1), n)


def test_list_targets_come_from_out_of_fold_windows():
    calls = []

    def fake_windows():
        calls.append("training_windows")
        matrix = pd.DataFrame({"query_id": ["1", "1"], "product_id": ["a", "b"],
                               "gain": [0.0, 1.0], "fold": [2, 2]})
        return [Window(query_id="1", window=("a", "b"), tail=())], matrix

    examples = list_training_set({"1": "shoes"}, {"a": "x", "b": "y"}, windows_fn=fake_windows)
    assert calls == ["training_windows"]
    assert examples == [("shoes", ["x", "y"], [0.0, 1.0])]


def test_a_training_window_from_fold_zero_is_refused():
    def leaky_windows():
        matrix = pd.DataFrame({"query_id": ["1"], "product_id": ["a"],
                               "gain": [1.0], "fold": [0]})
        return [Window(query_id="1", window=("a",), tail=())], matrix

    with pytest.raises(ValueError):
        list_training_set({"1": "q"}, {"a": "x"}, windows_fn=leaky_windows)
