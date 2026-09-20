# Plan 3 — Image Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-embedder.md); do not start a phase before its predecessor's gate passes.

**Goal:** Turn the image URLs in the enrichment corpus into a CLIP embedding store the ranking stages can read — fetched in bulk, embedded in flight on the GPU, resumable, and with the images themselves never written to disk.

**Architecture:** Three small modules and one orchestrator. `src/clip_encoder.py` owns the model and nothing else; `src/embedding_store.py` owns an append-only float16 store that survives being killed mid-write; `src/image_fetch.py` owns concurrent HTTP with retry and decode validation. `src/embed_images.py` wires them into one streaming pass: URL → bytes → decoded image → 512-d vector → store, keeping at most one batch in memory. The pure logic in each module is unit-tested against fakes, so the only step that needs the network or the GPU is the bulk run itself.

**Tech Stack:** Python ≥3.11, `torch` 2.14 + `transformers` 5.17 (CLIP, already installed from Plan 1's `baselines` extra), `pillow` (WEBP/JPEG decode), `numpy`, `pyarrow`, `concurrent.futures` from the standard library for fetch concurrency. One new dependency: `pillow`.

**Spec:** `PROJECT_SPEC.md` (§3.3 Images, §8.3 Build Order, §9 Compute) and `CLAUDE.md` (Data invariants, Scale). The plan argues from both; executors read both.

**Series:** Plan 3 of 7 — see [`../README.md`](../README.md). Gated on Plan 2, whose gate passed with a 1,080,262-row enrichment corpus at 89.59% join coverage.

---

## The two phases

Phase 1 needs no network and no bulk data — it ends with an encoder and a store
that are proven correct against fakes and a handful of real images. Phase 2 is
the only phase that moves gigabytes.

| Phase | File | Tasks | Delivers | Network / GPU? |
|---|---|---|---|---|
| 1 | [**The Embedder**](phase-1-the-embedder.md) | 1–2 | CLIP encoder, crash-consistent embedding store | GPU for one marked test |
| 2 | [**The Fetch**](phase-2-the-fetch.md) | 3–5 | Concurrent fetcher, the streaming pipeline, the real run | Yes, both |

---

## Where these numbers came from

Every figure below was measured against the real corpus and the live CDN on
2026-09-20, before the plan was written. Four of them change the design:

- **The dead CDN bucket is already gone.** 0 of 932,320 corpus image URLs carry
  the `/images/W/<bucket>/images/I/` segment — Plan 2's ETL routed every URL
  through `image_url()`, which strips it. A live sample of 300 corpus URLs
  resolved **300/300**, and a second sample of 600 resolved **600/600** with 0
  undecodable. The repair the spec spends a page on is done; this plan inherits
  clean URLs and only re-runs the gate to confirm nothing regressed.
- **The network is the bottleneck, not the GPU — by roughly 5×.** Fetch runs at
  3,378–5,516 images/minute at concurrency 14; CLIP on the 3080 runs at
  **18,140/minute** at batch 64. The GPU will idle waiting for bytes, which is
  exactly why embedding in flight costs nothing and the images never need to
  touch disk.
- **float16 storage is free.** Round-tripping the embeddings through float16
  left top-1 retrieval **identical** (75.333%) with a maximum cosine drift of
  1.22e-04. So the store is float16: 383 MB for the re-ranking scope, 908 MB
  for the catalogue — both inside §9's "<1 GB" budget, where float32 would blow
  it at 1.8 GB.
- **`transformers` 5 changed the CLIP API.** `CLIPModel.get_image_features()`
  now returns a `BaseModelOutputWithPooling`, not a tensor; the projected 512-d
  embedding is in `.pooler_output`. Code written against transformers 4 fails
  with `AttributeError: 'BaseModelOutputWithPooling' object has no attribute
  'norm'`. Task 1 pins this.

Measured baseline for the semantic gate: over a pool of 600 real
product/title pairs, CLIP matches an image to its own title at **top-1 75.3%,
top-5 92.7%, top-10 95.5%**, median rank 1, against a 0.167% chance rate.
Diagonal cosine +0.3258 vs off-diagonal +0.1536.

---

## Scope: re-ranking or catalogue

The pipeline takes `--scope`, the same way `src/combine.py` does, because the
two downstream needs differ and the cost gap is large:

| scope | products | distinct URLs | transfer | store (fp16) | fetch at 3,400/min |
|---|---|---|---|---|---|
| `rerank` | 482,105 | **373,776** | 3.81 GB | 383 MB | ~1.8 h |
| `catalogue` | 1,215,854 | **887,041** | 9.05 GB | 908 MB | ~4.3 h |

**Default is `rerank`.** Plans 5–7 re-rank judged candidates only and need
nothing more. Plan 4's full-corpus recall needs `catalogue`, and the store is
append-only, so running `rerank` first and `catalogue` later re-fetches only
the difference.

Note 932,320 catalogue products share only 887,041 distinct URLs — 45,279
products reuse another product's image. Fetch is keyed by URL, not by product.

**Throughput varies a lot.** Three runs measured 5,516, 3,378 and 850
images/minute at the same concurrency. Budget for the low end: a `rerank` pass
is between 1.8 and 7.3 hours. The run is resumable, so this is an annoyance,
not a risk.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Run GPU-capable work on the GPU.** An NVIDIA RTX 3080 Laptop (16 GB, sm_86) is present via WSL2. Pass `device` explicitly and fail loudly if `cuda` is requested and unavailable — a silent CPU fallback turns a 20-minute embed into hours and looks identical in the logs. CLIP base/32 peaks at **1.40 GB** VRAM at batch 64, so memory is not a constraint.
- **Never persist the images.** §3.3: embed in flight, keep only the vectors. 3.81–9.05 GB of transfer becomes 383–908 MB of store.
- **Always go through `image_url()`** if reading raw ESCI-S again. The corpus column `s_image_url` is already normalised; raw records are not.
- **Write-as-you-go and skip-existing**, so an interrupted run resumes. §3.3 requires it, and at 1.8–7.3 hours a non-resumable run is not viable.
- **Retry with backoff on 429 and 5xx, send a real User-Agent, cap concurrency.** Measured safe at 14.
- **A 200 response is not an image.** Validate by decoding, not by status code.
- **`type: "error"` rows and products with no URL are not failures.** 22.47% of re-ranking products have no image URL at all; that is expected and must be representable as *absent*, never as a zero vector.
- **Report end-to-end image coverage against the measured ~77.5%**, not ESCI-S's 91.5% headline.
- **Image presence correlates with relevance** at -0.0207 mean gain [-0.0288, -0.0123], and it is a *scrape artefact*, not a product property. Plan 5 must carry an image-presence indicator and report its contribution separately, per `CLAUDE.md`'s evaluation discipline.
- **Do not commit datasets.** `.gitignore` blocks `data/`. The embedding store lives at `data/embeddings/` and is not committed; only the JSON record under `docs/results/` is.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)

---

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **An HTTP 200 that is not an image.** A CDN error page, a placeholder, or a truncated body all return 200 and would be embedded as if they were products — poisoning the store with vectors that are confidently wrong rather than obviously missing. Success means *decoded*, not *fetched*. — pinned in Task 3.
2. **"No image" collapsing into a zero vector.** 22.47% of re-ranking products have no URL. A zero vector has cosine 0 with everything, which is a *score*, not an absence, and it would silently rank those products as mildly irrelevant rather than unknown. The store must represent absence by omission, and callers must get an explicit presence mask. — pinned in Task 2.
3. **An interrupted run corrupting the store.** The run takes hours and will be killed. Vectors and their index are two writes; a kill between them leaves a row with no id or an id with no row, and the next run resumes from a store that silently disagrees with itself. Recovery must be deterministic and must not double-write. — pinned in Task 2.
4. **Duplicate URLs fetched and stored once per product.** 932,320 products share 887,041 URLs. Keying on product rather than URL wastes 5% of a multi-hour fetch and writes the same vector several times under different ids. — pinned in Task 4.
5. **The encoder silently changing shape or scale.** `transformers` 5 already moved the embedding to `.pooler_output`; a future version could change it again, and unnormalised vectors turn cosine into a dot product that ranks by magnitude. Both fail quietly — the pipeline still runs and the numbers still look plausible. — pinned in Task 1.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as
`src.<name>` and run as `python -m src.<name>`. This plan follows it.

| File | Responsibility | Written in |
|---|---|---|
| `src/clip_encoder.py` | Load CLIP, resolve the device, encode images and texts to L2-normalised float32. No I/O, no fetching | Phase 1, Task 1 |
| `src/embedding_store.py` | Append-only float16 vector store plus its id index; crash-consistent recovery and resume | Phase 1, Task 2 |
| `src/image_fetch.py` | Concurrent HTTP with retry, backoff and decode validation. No model, no store | Phase 2, Task 3 |
| `src/embed_images.py` | The streaming pass and its CLI: `python -m src.embed_images` | Phase 2, Task 4 |
| `docs/results/image-embeddings.json` | Committed coverage and semantic-gate record | Phase 2, Task 5 |
| `tests/test_clip_encoder.py`, `tests/test_embedding_store.py`, `tests/test_image_fetch.py`, `tests/test_embed_images.py` | One test module per source module | Every task |

`src/clip_encoder.py` takes an `encode_batch` callable in its pure functions,
the same way `src/baseline_sbert.py` does, so the batching, normalisation and
shape contracts are tested in milliseconds against a fake and only the bulk run
loads a real model.

---

## Plan Gate

Plan 4 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the store round-trip on the real run.
- [ ] `python -m src.esci_images data/esci-s/esci.json.zst` exits zero — the resolution gate, re-run rather than trusted.
- [ ] `data/embeddings/rerank/` holds one float16 vector per successfully fetched image, and its index row count equals the vector row count exactly.
- [ ] `docs/results/image-embeddings.json` records end-to-end coverage against the measured ~77.5%, not 91.5%.
- [ ] The semantic gate passes: top-1 image→title retrieval **≥ 0.60** over a 600-candidate pool, against a measured 0.753 and a 0.00167 chance rate.
- [ ] `CLAUDE.md`'s Commands section lists the embedding invocation.

Then: Plan 4 — Recall. See [`../README.md`](../README.md) for the series.
