# Phase 1 — The Embedder

**Plan 3 of 7 · Phase 1 of 2 · Tasks 1–2.** Read [`README.md`](README.md) first —
it carries the goal, the architecture, the full Global Constraints and the File
Structure table this phase writes into.

**Delivers:** `src/clip_encoder.py` and `src/embedding_store.py` — a CLIP
encoder whose shape and scale contracts are pinned, and a vector store that
survives being killed mid-write.

**Needs on disk:** nothing but the CLIP weights (~600 MB, downloaded once by
one `slow`-marked test). No bulk fetch in this phase — that is Phase 2.

**Owns Review Focus items 2, 3 and 5** (absence collapsing into a zero vector,
an interrupted run corrupting the store, the encoder silently changing shape
or scale).

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **`transformers` 5 moved the embedding.** `CLIPModel.get_image_features()` returns a `BaseModelOutputWithPooling`; the projected 512-d vector is `.pooler_output`. transformers 4 returned the tensor directly.
- **Vectors must be L2-normalised.** Unnormalised, cosine becomes a dot product that ranks partly by magnitude, and nothing errors.
- **Store float16.** Measured: round-tripping through float16 left top-1 retrieval identical at 75.333%, max cosine drift 1.22e-04, and halves 1.8 GB to 908 MB.
- **Absence is not a zero vector.** 22.47% of re-ranking products have no image. A zero vector scores 0 against everything, which is a score, not an absence.
- **Pass `device` explicitly and fail loudly** if `cuda` is asked for and missing.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: The CLIP encoder

**Files:**
- Create: `src/clip_encoder.py`
- Create: `tests/test_clip_encoder.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing from earlier plans.
- Produces:
  - `src.clip_encoder.DEFAULT_MODEL: str` (`"openai/clip-vit-base-patch32"`)
  - `src.clip_encoder.EMBEDDING_DIM: int` (512), `DEFAULT_BATCH_SIZE: int` (64)
  - `src.clip_encoder.resolve_device(requested: str) -> str`
  - `src.clip_encoder.l2_normalise(vectors: np.ndarray) -> np.ndarray`
  - `src.clip_encoder.projected(output) -> "torch.Tensor"`
  - `src.clip_encoder.encode_in_batches(items, encode_batch, *, batch_size=DEFAULT_BATCH_SIZE, dim=EMBEDDING_DIM) -> np.ndarray`
  - `src.clip_encoder.ClipEncoder` with `.encode_images(images, batch_size=...) -> np.ndarray` and `.encode_texts(texts, batch_size=...) -> np.ndarray`
  - `src.clip_encoder.load_encoder(model_name=DEFAULT_MODEL, device="auto") -> ClipEncoder`

`encode_in_batches` takes an `encode_batch` callable rather than a model, the
same way `src/baseline_sbert.py` takes `encode`. That keeps batching,
normalisation and the shape contract testable in milliseconds against a fake,
and confines the 600 MB model download to one `slow`-marked test.

Batch size 64 is not arbitrary: measured on the RTX 3080 at 302 img/s against
225 at batch 32 and 284 at batch 256.

- [x] **Step 1: Add `pillow` to the baselines extra**

`src/image_fetch.py` in Phase 2 decodes WEBP and JPEG, and the encoder's real
test needs an image. In `pyproject.toml`, extend the existing extra:

```toml
baselines = [
    "sentence-transformers>=3.0",
    "torch>=2.4",
    "pillow>=10",
]
```

Then install it:

```bash
uv pip install -e ".[dev,baselines]"
```

- [x] **Step 2: Write the failing test**

Create `tests/test_clip_encoder.py`:

```python
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
```

- [x] **Step 3: Run the test to verify it fails**

```bash
python -m pytest tests/test_clip_encoder.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.clip_encoder'`.

- [x] **Step 4: Write `src/clip_encoder.py`**

```python
"""CLIP image and text embeddings in one shared 512-d space.

The batching, normalisation and shape contracts live in `encode_in_batches`,
which takes an `encode_batch` callable rather than a model - the same shape as
src/baseline_sbert.py - so they are tested against a fake in milliseconds and
only the bulk run pays for a 600 MB download.

Two contracts are asserted rather than assumed, because both fail silently:

  * `transformers` 5 returns a BaseModelOutputWithPooling from
    get_image_features(), not a tensor. The projected vector is
    `.pooler_output`. Code written against transformers 4 dies with
    "AttributeError: 'BaseModelOutputWithPooling' object has no attribute
    'norm'", which is how this was found.
  * Vectors are L2-normalised. Unnormalised, cosine similarity silently
    becomes a dot product that ranks partly by magnitude.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# Measured on an RTX 3080 Laptop: 302 img/s at 64, against 225 at 32, 279 at
# 128 and 284 at 256. Peak VRAM 1.40 GB of 17.2 GB.
DEFAULT_BATCH_SIZE = 64

EncodeBatch = Callable[[Sequence[Any]], np.ndarray]


def resolve_device(requested: str) -> str:
    """Pick the device. "auto" means the GPU when there is one.

    Explicit rather than relying on a library default: a silent fall back to
    CPU turns a twenty-minute embed into hours and looks identical in the logs.
    """
    import torch

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(
                f"--device {requested} requested but torch.cuda.is_available() "
                "is False; pass --device cpu to run on CPU deliberately"
            )
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving an all-zero row alone."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def projected(output: Any) -> Any:
    """The projected embedding, across transformers 4 and 5.

    transformers 5 wraps it in BaseModelOutputWithPooling; 4 returned the
    tensor. `.pooler_output` here is post-projection (512-d), not the vision
    tower's 768-d pooled state.
    """
    return getattr(output, "pooler_output", output)


def encode_in_batches(
    items: Sequence[Any],
    encode_batch: EncodeBatch,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """Encode `items` in batches and return one L2-normalised row each."""
    if len(items) == 0:
        return np.zeros((0, dim), dtype=np.float32)

    chunks: list[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        batch = list(items[start : start + batch_size])
        vectors = np.asarray(encode_batch(batch), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"encode_batch returned a {vectors.ndim}-D array; a 2-D "
                "(batch, dim) array is required"
            )
        if vectors.shape[0] != len(batch):
            raise ValueError(
                f"encode_batch returned {vectors.shape[0]} rows for a batch of "
                f"{len(batch)}; rows and items must correspond one to one"
            )
        if vectors.shape[1] != dim:
            raise ValueError(
                f"encode_batch returned width {vectors.shape[1]}, expected "
                f"{dim}; the model or its output field has changed"
            )
        chunks.append(vectors)
    return l2_normalise(np.concatenate(chunks))


@dataclass(frozen=True)
class ClipEncoder:
    model: Any
    processor: Any
    device: str
    dim: int

    def encode_images(
        self, images: Sequence[Any], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        return encode_in_batches(
            images, self._image_batch, batch_size=batch_size, dim=self.dim
        )

    def encode_texts(
        self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        return encode_in_batches(
            texts, self._text_batch, batch_size=batch_size, dim=self.dim
        )

    def _image_batch(self, images: Sequence[Any]) -> np.ndarray:
        import torch

        inputs = self.processor(images=list(images), return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            return projected(self.model.get_image_features(**inputs)).cpu().numpy()

    def _text_batch(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP's context length; longer titles are cut
        ).to(self.device)
        with torch.inference_mode():
            return projected(self.model.get_text_features(**inputs)).cpu().numpy()


def load_encoder(
    model_name: str = DEFAULT_MODEL, device: str = "auto"
) -> ClipEncoder:
    """Load CLIP onto the chosen device, in eval mode."""
    from transformers import CLIPModel, CLIPProcessor

    resolved = resolve_device(device)
    model = CLIPModel.from_pretrained(model_name).to(resolved).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return ClipEncoder(
        model=model,
        processor=processor,
        device=resolved,
        dim=int(model.config.projection_dim),
    )
```

- [x] **Step 5: Run the fast tests to verify they pass**

```bash
python -m pytest tests/test_clip_encoder.py -v
```

Expected: PASS, 14 tests; the `slow`-marked real-model test is deselected.

- [x] **Step 6: Run the real-model test**

```bash
python -m pytest tests/test_clip_encoder.py -v -m slow
```

Expected: PASS, 1 test. First run downloads ~600 MB of weights. If the two
similarity assertions fail, the image and text embeddings are not in the same
space — check that `projected` is returning `.pooler_output` (512-d) and not
`.last_hidden_state` (768-d).

- [x] **Step 7: Commit**

```bash
git add pyproject.toml src/clip_encoder.py tests/test_clip_encoder.py
git commit -m "Add a CLIP encoder with pinned shape and normalisation contracts"
```

---

## Task 2: The embedding store

**Files:**
- Create: `src/embedding_store.py`
- Create: `tests/test_embedding_store.py`

**Interfaces:**
- Consumes: `src.clip_encoder.EMBEDDING_DIM`.
- Produces:
  - `src.embedding_store.STORE_DTYPE: np.dtype` (float16), `VECTORS_NAME`, `KEYS_NAME`, `META_NAME`
  - `src.embedding_store.EmbeddingStore` with `.dim`, `.__len__()`, `.known_keys() -> set[str]`, `.keys() -> list[str]`, `.append(keys, vectors) -> int`, `.vectors() -> np.ndarray`, `.lookup(wanted) -> tuple[np.ndarray, np.ndarray]`
  - `src.embedding_store.open_store(directory, dim=EMBEDDING_DIM) -> EmbeddingStore`

Three files per store, in one directory:

```
vectors.f16   raw little-endian float16, row-major, (n, dim), no header
keys.txt      one key per line, UTF-8; line i is row i
meta.json     {"dim": 512, "dtype": "float16"}
```

`meta.json` exists because the dim cannot be recovered from the other two: a
dim-4 store opened as dim-8 computes `size // 16` rows, gets zero, and
truncates the file to nothing. The width has to be recorded, not inferred.

Line-numbered keys rather than a Parquet index because the store is written
across hours and will be killed: a line count and a file size are both
recoverable without parsing, and `min(rows, lines)` is a deterministic repair
for either write order. Keys are image URLs, not ASINs — 932,320 products share
887,041 URLs, so keying on the product would store the same vector several
times.

`lookup` fills unknown keys with **NaN**, not zero. A zero row has cosine 0
against everything, which is a score rather than an absence; NaN propagates and
is loud. The presence mask is returned alongside so callers never have to infer
it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_embedding_store.py`:

```python
import numpy as np
import pytest

from src.clip_encoder import EMBEDDING_DIM
from src.embedding_store import STORE_DTYPE, open_store


def _vectors(n: int, dim: int = 4, start: float = 0.0) -> np.ndarray:
    return (np.arange(n * dim, dtype=np.float32).reshape(n, dim) + start) / 100.0


# --- the happy path ---------------------------------------------------------

def test_a_new_store_is_empty(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    assert len(store) == 0
    assert store.known_keys() == set()


def test_append_then_read_back(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a", "b"], _vectors(2))
    assert len(store) == 2
    assert store.keys() == ["a", "b"]
    assert np.allclose(store.vectors(), _vectors(2), atol=1e-3)


def test_vectors_are_stored_as_float16(tmp_path):
    # float32 would put the catalogue store at 1.8 GB, past the spec's <1 GB
    # budget. Measured: float16 leaves top-1 retrieval identical at 75.333%
    # with a max cosine drift of 1.22e-04.
    assert STORE_DTYPE == np.float16
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    assert store.vectors().dtype == np.float16


def test_appending_twice_accumulates(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    store.append(["b", "c"], _vectors(2, start=100.0))
    assert len(store) == 3
    assert store.keys() == ["a", "b", "c"]


def test_reopening_sees_everything_written_before(tmp_path):
    # This is resume: the run is killed, and the next one must not refetch.
    directory = tmp_path / "s"
    open_store(directory, dim=4).append(["a", "b"], _vectors(2))
    reopened = open_store(directory, dim=4)
    assert len(reopened) == 2
    assert reopened.known_keys() == {"a", "b"}


# --- Review Focus 3: an interrupted run must not corrupt the store ----------

def test_recovery_truncates_vectors_written_without_their_key(tmp_path):
    # Killed after the vector write, before the key write.
    directory = tmp_path / "s"
    store = open_store(directory, dim=4)
    store.append(["a", "b"], _vectors(2))
    with (directory / "vectors.f16").open("ab") as handle:
        handle.write(np.zeros(4, dtype=np.float16).tobytes())

    recovered = open_store(directory, dim=4)
    assert len(recovered) == 2
    assert recovered.keys() == ["a", "b"]
    assert (directory / "vectors.f16").stat().st_size == 2 * 4 * 2


def test_recovery_drops_keys_written_without_their_vector(tmp_path):
    # Killed after the key write, before the vector write.
    directory = tmp_path / "s"
    store = open_store(directory, dim=4)
    store.append(["a", "b"], _vectors(2))
    with (directory / "keys.txt").open("a", encoding="utf-8") as handle:
        handle.write("c\n")

    recovered = open_store(directory, dim=4)
    assert len(recovered) == 2
    assert recovered.known_keys() == {"a", "b"}


def test_recovery_survives_a_torn_final_vector(tmp_path):
    # Killed mid-write: the file ends part way through a row.
    directory = tmp_path / "s"
    store = open_store(directory, dim=4)
    store.append(["a", "b"], _vectors(2))
    with (directory / "vectors.f16").open("ab") as handle:
        handle.write(b"\x00\x00\x00")  # 3 bytes: not a whole 4-wide f16 row

    recovered = open_store(directory, dim=4)
    assert len(recovered) == 2
    assert np.allclose(recovered.vectors(), _vectors(2), atol=1e-3)


def test_appending_a_key_already_stored_raises(tmp_path):
    # Silent double-writes would grow the store without bound on every resume.
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    with pytest.raises(ValueError, match="already"):
        store.append(["a"], _vectors(1))


def test_appending_a_duplicate_within_one_batch_raises(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="duplicate"):
        store.append(["a", "a"], _vectors(2))


def test_appending_the_wrong_number_of_vectors_raises(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="rows"):
        store.append(["a", "b"], _vectors(3))


def test_appending_the_wrong_width_raises(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="width"):
        store.append(["a"], np.zeros((1, 9), dtype=np.float32))


def test_reopening_with_a_different_dim_raises(tmp_path):
    directory = tmp_path / "s"
    open_store(directory, dim=4).append(["a"], _vectors(1))
    with pytest.raises(ValueError, match="dim"):
        open_store(directory, dim=8)


def test_a_key_containing_a_newline_raises(tmp_path):
    # Line i is row i. A key with a newline would silently shift every row
    # after it by one on the next open.
    store = open_store(tmp_path / "s", dim=4)
    with pytest.raises(ValueError, match="newline"):
        store.append(["a\nb"], _vectors(1))


# --- Review Focus 2: absence is not a zero vector ---------------------------

def test_lookup_returns_nan_and_false_for_an_unknown_key(tmp_path):
    # 22.47% of re-ranking products have no image. A zero row would score
    # cosine 0 against every query - a score, not an absence - and would rank
    # those products as mildly irrelevant rather than unknown.
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    matrix, present = store.lookup(["a", "missing"])
    assert matrix.shape == (2, 4)
    assert present.tolist() == [True, False]
    assert not np.isnan(matrix[0]).any()
    assert np.isnan(matrix[1]).all()


def test_lookup_returns_float32_for_arithmetic(tmp_path):
    # Stored at float16; handed out at float32 so downstream dot products do
    # not silently accumulate half-precision error.
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a"], _vectors(1))
    matrix, _ = store.lookup(["a"])
    assert matrix.dtype == np.float32


def test_lookup_of_nothing_returns_an_empty_matrix(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    matrix, present = store.lookup([])
    assert matrix.shape == (0, 4)
    assert present.shape == (0,)


def test_lookup_preserves_the_requested_order(tmp_path):
    store = open_store(tmp_path / "s", dim=4)
    store.append(["a", "b"], _vectors(2))
    matrix, _ = store.lookup(["b", "a"])
    assert np.allclose(matrix[0], store.vectors()[1].astype(np.float32), atol=1e-3)


def test_the_default_dim_is_clips(tmp_path):
    assert open_store(tmp_path / "s").dim == EMBEDDING_DIM
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_embedding_store.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.embedding_store'`.

- [ ] **Step 3: Write `src/embedding_store.py`**

```python
"""An append-only vector store that survives being killed mid-write.

The bulk embed runs for hours over a flaky network and will be interrupted, so
the store is two plain files rather than one clever format:

    vectors.f16   raw little-endian float16, row-major, (n, dim), no header
    keys.txt      one key per line, UTF-8; line i is row i
    meta.json     the width, which cannot be recovered from the other two

A row count is `size // (dim * 2)` and a key count is a line count, both
recoverable without parsing anything. Whichever write the kill landed between,
`min(rows, keys)` is a deterministic repair, and `open_store` performs it.

Keys are image URLs, not ASINs: 932,320 catalogue products share 887,041
distinct URLs, so keying on the product would fetch and store the same vector
up to several times.

Stored at float16 because float32 would put the catalogue store at 1.8 GB,
past PROJECT_SPEC.md §9's "<1 GB". Measured: the round trip leaves top-1
image-to-title retrieval identical at 75.333%, max cosine drift 1.22e-04.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.clip_encoder import EMBEDDING_DIM

STORE_DTYPE = np.float16
VECTORS_NAME = "vectors.f16"
KEYS_NAME = "keys.txt"
META_NAME = "meta.json"


class EmbeddingStore:
    """Append-only float16 vectors plus their keys. Open it via open_store."""

    def __init__(self, directory: Path, dim: int) -> None:
        self.directory = Path(directory)
        self.dim = int(dim)
        self._vectors_path = self.directory / VECTORS_NAME
        self._keys_path = self.directory / KEYS_NAME
        self._keys: list[str] = []
        self._rows: dict[str, int] = {}

    # --- reading ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._keys)

    def keys(self) -> list[str]:
        return list(self._keys)

    def known_keys(self) -> set[str]:
        return set(self._rows)

    def vectors(self) -> np.ndarray:
        """The stored matrix, (n, dim) float16, memory-mapped."""
        if not len(self._keys):
            return np.zeros((0, self.dim), dtype=STORE_DTYPE)
        return np.memmap(
            self._vectors_path,
            dtype=STORE_DTYPE,
            mode="r",
            shape=(len(self._keys), self.dim),
        )

    def lookup(self, wanted: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Vectors for `wanted`, in order, with a presence mask.

        Unknown keys get a row of NaN, never zeros: a zero row has cosine 0
        against everything, which is a score rather than an absence, and would
        quietly rank an image-less product as mildly irrelevant.
        """
        matrix = np.full((len(wanted), self.dim), np.nan, dtype=np.float32)
        present = np.zeros(len(wanted), dtype=bool)
        if not len(wanted) or not len(self._keys):
            return matrix, present
        stored = self.vectors()
        for position, key in enumerate(wanted):
            row = self._rows.get(key)
            if row is not None:
                matrix[position] = stored[row].astype(np.float32)
                present[position] = True
        return matrix, present

    # --- writing ------------------------------------------------------------

    def append(self, keys: Sequence[str], vectors: np.ndarray) -> int:
        """Append rows. Vectors are written before keys; see open_store."""
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != len(keys):
            raise ValueError(
                f"{vectors.shape[0] if vectors.ndim == 2 else '?'} rows for "
                f"{len(keys)} keys; rows and keys must correspond one to one"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector width {vectors.shape[1]} does not match the store's "
                f"dim {self.dim}"
            )
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate keys within one append batch")
        already = self._rows.keys() & set(keys)
        if already:
            raise ValueError(
                f"{len(already)} keys are already in the store "
                f"(for example {sorted(already)[:3]}); appending them again "
                "would grow it without bound on every resume"
            )
        for key in keys:
            if "\n" in key or "\r" in key:
                raise ValueError(f"key contains a newline: {key!r}")

        if not len(keys):
            return 0

        self.directory.mkdir(parents=True, exist_ok=True)
        meta_path = self.directory / META_NAME
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps({"dim": self.dim, "dtype": np.dtype(STORE_DTYPE).name})
                + "\n",
                encoding="utf-8",
            )
        with self._vectors_path.open("ab") as handle:
            handle.write(np.ascontiguousarray(vectors, dtype=STORE_DTYPE).tobytes())
            handle.flush()
        with self._keys_path.open("a", encoding="utf-8") as handle:
            handle.write("".join(f"{key}\n" for key in keys))
            handle.flush()

        for key in keys:
            self._rows[key] = len(self._keys)
            self._keys.append(key)
        return len(keys)


def open_store(directory: Path, dim: int = EMBEDDING_DIM) -> EmbeddingStore:
    """Open a store, repairing a half-finished write if there is one.

    The dim is checked against meta.json *before* any truncation: the repair
    divides the file size by the row width, so opening a dim-4 store as dim-8
    would compute zero rows and truncate it to nothing.
    """
    directory = Path(directory)
    meta_path = directory / META_NAME
    if meta_path.exists():
        recorded = int(json.loads(meta_path.read_text(encoding="utf-8"))["dim"])
        if recorded != int(dim):
            raise ValueError(
                f"store at {directory} holds dim {recorded} vectors but was "
                f"opened with dim {dim}"
            )

    store = EmbeddingStore(directory, dim)
    vectors_path, keys_path = store._vectors_path, store._keys_path
    if not vectors_path.exists() and not keys_path.exists():
        return store

    row_bytes = dim * np.dtype(STORE_DTYPE).itemsize
    size = vectors_path.stat().st_size if vectors_path.exists() else 0
    keys = (
        keys_path.read_text(encoding="utf-8").splitlines() if keys_path.exists() else []
    )
    kept = min(size // row_bytes, len(keys))

    if size != kept * row_bytes:
        with vectors_path.open("r+b") as handle:
            handle.truncate(kept * row_bytes)
    if len(keys) != kept:
        keys_path.write_text(
            "".join(f"{key}\n" for key in keys[:kept]), encoding="utf-8"
        )

    store._keys = keys[:kept]
    store._rows = {key: row for row, key in enumerate(store._keys)}
    return store
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest tests/test_embedding_store.py -v
```

Expected: PASS, 19 tests.

- [ ] **Step 5: Commit**

```bash
git add src/embedding_store.py tests/test_embedding_store.py
git commit -m "Add a crash-consistent float16 embedding store keyed by image URL"
```

---

## Phase 1 Gate

Phase 2 does not start until all of these hold:

- [ ] `python -m pytest tests/test_clip_encoder.py tests/test_embedding_store.py -v` passes — 14 + 19 tests.
- [ ] `python -m pytest -m slow tests/test_clip_encoder.py` passes: a real CLIP model produces unit-length 512-d vectors, and a red square scores higher against "a red square" than against "a blue square".
- [ ] `python -m pytest` still passes end to end, with Plans 1 and 2 untouched.

Next: [Phase 2 — The Fetch](phase-2-the-fetch.md).
