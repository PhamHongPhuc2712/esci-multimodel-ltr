import numpy as np
import pytest

from src.clip_encoder import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    encode_in_batches,
    l2_normalise,
    projected,
    resolve_device,
)


def _fake_encode(dim: int = EMBEDDING_DIM):
    """An encoder that records the batches it was handed."""
    seen: list[int] = []

    def encode_batch(items):
        seen.append(len(items))
        # deterministic, non-unit-length rows so normalisation is observable
        return np.arange(len(items) * dim, dtype=np.float32).reshape(len(items), dim) + 1.0

    return encode_batch, seen


# --- normalisation ----------------------------------------------------------

def test_l2_normalise_gives_unit_vectors():
    vectors = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
    assert np.allclose(np.linalg.norm(l2_normalise(vectors), axis=1), 1.0)


def test_l2_normalise_survives_a_zero_row():
    # A zero row would divide by zero and poison the whole matrix with nan.
    out = l2_normalise(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert not np.isnan(out).any()
    assert np.allclose(np.linalg.norm(out[1]), 1.0)


# --- batching ---------------------------------------------------------------

def test_encode_in_batches_respects_the_batch_size():
    encode_batch, seen = _fake_encode()
    encode_in_batches(list(range(10)), encode_batch, batch_size=4)
    assert seen == [4, 4, 2]


def test_encode_in_batches_returns_one_unit_row_per_item():
    encode_batch, _ = _fake_encode()
    out = encode_in_batches(list(range(7)), encode_batch, batch_size=3)
    assert out.shape == (7, EMBEDDING_DIM)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_encode_in_batches_of_nothing_returns_the_right_shape():
    # An empty batch is normal near the end of a resumed run; returning a
    # 0-row matrix keeps np.concatenate downstream from raising.
    encode_batch, seen = _fake_encode()
    out = encode_in_batches([], encode_batch)
    assert out.shape == (0, EMBEDDING_DIM)
    assert seen == []


def test_default_batch_size_is_the_measured_optimum():
    # 302 img/s at 64 on the RTX 3080, against 225 at 32 and 284 at 256.
    assert DEFAULT_BATCH_SIZE == 64


# --- Review Focus 5: the encoder silently changing shape or scale -----------

def test_a_batch_of_the_wrong_width_raises():
    def wrong_width(items):
        return np.ones((len(items), 8), dtype=np.float32)

    with pytest.raises(ValueError, match="512"):
        encode_in_batches([1, 2], wrong_width)


def test_a_batch_of_the_wrong_length_raises():
    def wrong_length(items):
        return np.ones((len(items) + 1, EMBEDDING_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="rows"):
        encode_in_batches([1, 2], wrong_length)


def test_a_one_dimensional_batch_raises():
    def flat(items):
        return np.ones(EMBEDDING_DIM, dtype=np.float32)

    with pytest.raises(ValueError, match="2-D"):
        encode_in_batches([1], flat)


def test_projected_unwraps_the_transformers_5_output():
    # transformers 5 returns BaseModelOutputWithPooling from
    # get_image_features(); transformers 4 returned the tensor itself. Reading
    # the object as a tensor fails with AttributeError: ... has no attribute
    # 'norm', which is how this was found.
    class Output:
        pooler_output = np.array([[1.0, 2.0]])

    assert np.array_equal(projected(Output()), np.array([[1.0, 2.0]]))


def test_projected_passes_a_bare_tensor_through():
    array = np.array([[1.0, 2.0]])
    assert projected(array) is array


# --- device -----------------------------------------------------------------

def test_resolve_device_honours_an_explicit_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_something_torch_accepts():
    assert resolve_device("auto") in {"cuda", "cpu"}


def test_resolve_device_refuses_cuda_when_it_is_unavailable(monkeypatch):
    # A silent CPU fallback turns a 20-minute embed into hours and looks
    # identical in the logs.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="cuda"):
        resolve_device("cuda")


# --- the real model ---------------------------------------------------------

@pytest.mark.slow
def test_real_clip_encodes_images_and_text_into_one_unit_space():
    from PIL import Image

    from src.clip_encoder import load_encoder

    encoder = load_encoder()
    assert encoder.dim == EMBEDDING_DIM

    images = [
        Image.new("RGB", (300, 300), color=(255, 0, 0)),
        Image.new("RGB", (300, 300), color=(0, 0, 255)),
    ]
    image_vectors = encoder.encode_images(images)
    text_vectors = encoder.encode_texts(["a red square", "a blue square"])

    assert image_vectors.shape == (2, EMBEDDING_DIM)
    assert text_vectors.shape == (2, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(image_vectors, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(text_vectors, axis=1), 1.0, atol=1e-5)
    # the two spaces are shared: red matches "red" better than "blue" does
    similarity = image_vectors @ text_vectors.T
    assert similarity[0, 0] > similarity[0, 1]
    assert similarity[1, 1] > similarity[1, 0]
