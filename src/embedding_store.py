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
