"""The vLLM client, against a fake server."""

import threading

import pytest

from src.llm_point import encode_pair
from src.rerank_window import Window
from src.serve_client import (
    ServedRanker,
    p_yes_from_top,
    point_payload,
    prefix_cache_hits,
    served_list_ordering,
    served_point_ordering,
)


# The same two fakes as tests/test_llm_point.py and tests/test_llm_list.py,
# repeated: this repo's test modules do not import from each other.
class WordTokenizer:
    pad_token_id = 0
    unk_token_id = 1

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "yes": 2, "no": 3}
        self._lock = threading.Lock()

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            pass

        encoded = Encoded()
        with self._lock:
            encoded.input_ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        return encoded

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


class ChatTokenizer(WordTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return " ".join(m["content"] for m in messages) + " ASSISTANT"


def test_both_answers_present_give_the_exact_score():
    p, imputed = p_yes_from_top({"yes": -0.1, "no": -2.4, "maybe": -5.0})
    assert not imputed
    assert p == pytest.approx(1 / (1 + 2.718281828 ** (-2.3)), rel=1e-6)


def test_a_missing_answer_is_imputed_from_the_smallest_logprob_and_flagged():
    p, imputed = p_yes_from_top({"yes": -0.01, "x": -6.0, "y": -9.0})
    assert imputed
    assert p == pytest.approx(1 / (1 + 2.718281828 ** (-(9.0 - 0.01))), rel=1e-6)


def test_neither_answer_present_is_refused():
    with pytest.raises(ValueError, match="neither"):
        p_yes_from_top({"x": -1.0})


def test_the_server_is_sent_the_training_token_ids():
    tok = WordTokenizer()
    sent = {}

    def post(url, payload):
        sent.update(payload)
        return {"choices": [
            {"index": 1, "logprobs": {"top_logprobs": [{"yes": -3.0, "no": -0.1}]}},
            {"index": 0, "logprobs": {"top_logprobs": [{"yes": -0.1, "no": -3.0}]}},
        ]}

    ordering, imputed = served_point_ordering(
        post, "http://x", "point", tok, "shoes", ["red shoe", "sink"], ["a", "b"], 200)
    assert sent["prompt"] == [encode_pair(tok, "shoes", d, 200)[0] for d in ["red shoe", "sink"]]
    assert sent["max_tokens"] == 1 and sent["temperature"] == 0.0
    # choices arrive out of order; index 0 is "red shoe", the confident yes.
    assert ordering == ["a", "b"]
    assert imputed == 0


def test_a_short_server_answer_is_refused():
    def post(url, payload):
        return {"choices": [{"index": 0, "logprobs": {"top_logprobs": [{"yes": -0.1, "no": -2.0}]}}]}

    with pytest.raises(ValueError, match="choices"):
        served_point_ordering(post, "http://x", "point", WordTokenizer(), "q", ["d1", "d2"], ["a", "b"], 200)


def test_a_malformed_listwise_answer_keeps_stage2_order_and_is_flagged():
    def post(url, payload):
        return {"choices": [{"text": "[2] > [2]"}]}

    ordering, bad = served_list_ordering(post, "http://x", "list", ChatTokenizer(), "q",
                                         ["d1", "d2"], ["a", "b"])
    assert bad and ordering == ["a", "b"]


def test_a_wellformed_listwise_answer_is_applied():
    def post(url, payload):
        assert payload["stop"] == ["<|im_end|>"]
        return {"choices": [{"text": "[2] > [1]"}]}

    ordering, bad = served_list_ordering(post, "http://x", "list", ChatTokenizer(), "q",
                                         ["d1", "d2"], ["a", "b"])
    assert not bad and ordering == ["b", "a"]


def test_prefix_cache_hits_are_read_from_metrics():
    text = '# HELP x\nvllm:prefix_cache_hits_total{model_name="point"} 42.0\nother 1\n'
    assert prefix_cache_hits(text) == 42.0
    assert prefix_cache_hits("nothing here\n") is None


def test_counters_survive_concurrent_workers():
    def post(url, payload):
        return {"choices": [{"index": i, "logprobs": {"top_logprobs": [{"yes": -0.1}]}}
                            for i in range(len(payload["prompt"]))]}

    ranker = ServedRanker("http://x", "point", WordTokenizer(), "point",
                          {"q": "shoes"}, {"a": "x", "b": "y"}, 200, post=post)
    window = Window(query_id="q", window=("a", "b"), tail=())
    threads = [threading.Thread(target=lambda: [ranker.order(window) for _ in range(50)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ranker.imputed == 4 * 50 * 2


def test_point_payload_asks_for_one_token_and_top_logprobs():
    payload = point_payload("point", [[1, 2]])
    assert payload["logprobs"] == 20 and payload["max_tokens"] == 1
