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
