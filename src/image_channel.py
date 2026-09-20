"""The image retrieval channel: a text query against CLIP image embeddings.

CLIP puts images and text in one 512-d space, so the query side is
`clip_encoder.encode_texts` and the corpus side is the store Plan 3 filled -
no image is fetched at query time.

The store is keyed by **image URL**, not by product: Plan 3 keyed the fetch on
the URL because 45,279 catalogue products (11,771 in the re-ranking scope)
reuse another product's image. So a hit expands to one *or more* product ids,
and the channel oversamples URLs before truncating the expanded product list
to k. Without the oversample a query whose best URLs are shared returns fewer
than k products and its recall is understated for a reason unrelated to
retrieval.

Coverage is 77.50% of products, and that gap is a 2022 scrape artefact rather
than a property of the products. `CLAUDE.md`'s evaluation discipline requires
it to be reported separately, never quietly imputed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.clip_encoder import EMBEDDING_DIM
from src.embed_images import DEFAULT_PRODUCTS, DEFAULT_STORE_ROOT
from src.embedding_store import EmbeddingStore, open_store
from src.recall import Channel  # noqa: F401  (ImageChannel implements it)

DEFAULT_SCOPE = "catalogue"

# Retrieve OVERSAMPLE x k URLs before expanding to products, so shared images
# cannot starve a query of candidates.
OVERSAMPLE = 2


def url_to_products(products_path: Path = DEFAULT_PRODUCTS) -> dict[str, list[str]]:
    """Every product id that uses each image URL."""
    frame = pd.read_parquet(products_path, columns=["product_id", "s_image_url"])
    urls = frame["s_image_url"]
    frame = frame.loc[urls.notna() & (urls.astype("str").str.len() > 0)]
    grouped = frame.groupby("s_image_url")["product_id"].apply(list)
    return {str(url): [str(p) for p in ids] for url, ids in grouped.items()}


@dataclass(frozen=True)
class ImageChannel:
    store: EmbeddingStore
    encoder: Any
    url_products: dict[str, list[str]]
    device: str

    @property
    def n_urls(self) -> int:
        return len(self.store)

    def coverage(self, n_products: int) -> float:
        """Share of `n_products` reachable through a stored image."""
        if n_products <= 0:
            raise ValueError("n_products must be positive")
        reachable = {
            product
            for url in self.store.known_keys()
            for product in self.url_products.get(url, ())
        }
        return len(reachable) / n_products

    def search(self, queries: Sequence[str], k: int) -> list[list[str]]:
        from src.vector_search import top_k

        queries = list(queries)
        if not queries:
            return []

        vectors = self.encoder.encode_texts(queries)
        url_hits, _ = top_k(
            vectors, self.store, k * OVERSAMPLE, device=self.device
        )

        results: list[list[str]] = []
        for urls in url_hits:
            products: dict[str, None] = {}
            for url in urls:
                # A URL in the store but absent from the product table is
                # stale, not fatal: the store outlives the table it was built
                # from.
                for product in self.url_products.get(url, ()):
                    products.setdefault(product, None)
                    if len(products) >= k:
                        break
                if len(products) >= k:
                    break
            results.append(list(products))
        return results


def open_channel(
    scope: str = DEFAULT_SCOPE,
    store_root: Path = DEFAULT_STORE_ROOT,
    products_path: Path = DEFAULT_PRODUCTS,
    device: str = "auto",
) -> ImageChannel:
    """Open the CLIP image store for `scope` and load the text encoder."""
    from src.clip_encoder import load_encoder

    store = open_store(Path(store_root) / scope, dim=EMBEDDING_DIM)
    if len(store) == 0:
        raise FileNotFoundError(
            f"the image store at {Path(store_root) / scope} is empty; run "
            f"python -m src.embed_images --scope {scope}"
        )
    encoder = load_encoder(device=device)
    return ImageChannel(
        store=store,
        encoder=encoder,
        url_products=url_to_products(products_path),
        device=encoder.device,
    )
