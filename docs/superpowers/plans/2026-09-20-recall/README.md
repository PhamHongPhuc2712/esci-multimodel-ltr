# Plan 4 — Recall

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-metric-and-the-lexical-channel.md); do not start a phase before its predecessor's gate passes.

**Goal:** Retrieve candidates from the full 1,215,854-product corpus over three channels — lexical, dense text and CLIP image — fuse them with Reciprocal Rank Fusion, and measure Recall@k honestly enough that Ablations 1 and 2 mean something.

**Architecture:** One metric module and one module per channel, all behind a single `Channel` protocol that returns ranked `product_id`s for a query. `src/recall.py` owns the ground truth and the metric and nothing else; `src/bm25_index.py` owns the lexical channel; `src/vector_search.py` owns top-k cosine over any `EmbeddingStore`, which both the dense-text and the image channels are thin wrappers around; `src/rrf.py` fuses ranked lists; `src/query_rewrite.py` is Stage 0. The pure logic in each is unit-tested against fakes, so only the index builds and the bulk embed need the corpus or the GPU.

**Tech Stack:** Python ≥3.11, `bm25s` 0.3 + `PyStemmer` (lexical), `sentence-transformers` (dense text, already installed from Plan 1's `baselines` extra), Plan 3's `src/clip_encoder.py` and `src/embedding_store.py` (image), `numpy`, `pandas`, `pyarrow`. Two new dependencies: `bm25s`, `PyStemmer`.

**Spec:** `PROJECT_SPEC.md` (§4.0 Stage 0, §4.1 Stage 1, §6 Ablations 1–2, §8.4 Build Order, §9 Compute) and `CLAUDE.md` (Evaluation discipline, Scale). The plan argues from both; executors read both.

**Series:** Plan 4 of 7 — see [`../README.md`](../README.md). Gated on Plan 3, whose gate passed with 361,875 CLIP vectors at 77.50% end-to-end image coverage and a 76.0% semantic gate.

---

## The three phases

Phase 1 delivers the metric before any channel exists, for the same reason
Plan 1 delivered NDCG before any ranker did: §8 states *"Nothing else until the
metric is trustworthy"*, and a Recall@k that is quietly wrong makes every
number in Ablation 2 meaningless.

| Phase | File | Tasks | Delivers | Network / GPU? |
|---|---|---|---|---|
| 1 | [**The Metric and the Lexical Channel**](phase-1-the-metric-and-the-lexical-channel.md) | 1–2 | Recall@k with its ground truth, BM25 channel | No (CPU, 18 GB peak once) |
| 2 | [**The Learned Channels**](phase-2-the-learned-channels.md) | 3–4 | Dense-text channel and its corpus embed, CLIP image channel | GPU + network |
| 3 | [**Fusion and the Ablations**](phase-3-fusion-and-the-ablations.md) | 5–7 | RRF, Stage 0 query rewriting, Ablations 1 and 2 | API key for Task 6 |

---

## Where these numbers came from

Every figure below was measured against the real corpus on 2026-09-20, before
the plan was written. Five of them change the design:

- **Building the BM25 index in memory peaks at 18.3 GB on a 23 GB machine.**
  Tokenising 1,215,854 concatenated title+description+bullets strings (mean
  1,748 chars) takes 151 s and indexing another 90 s, and the naive
  build-and-query-in-one-process path came within 5 GB of the OOM killer.
  Saving the index and reopening it with `mmap=True` costs **1.55 GB** and is
  *faster* — 28 ms/query against 37, amortised over a batch. So the index is
  built once by a CLI, saved to `data/bm25/`, and every consumer memory-maps
  it. **The build still peaks at 18.0 GB** even with the corpus strings freed,
  because the caller's DataFrame stays live throughout — measured end to end,
  not in a probe — and takes about five minutes. On disk it is 1.1 GB.
  Single-query latency against the mmapped index is 64–88 ms; the 28 ms figure
  is per query within a batch, which is the real workload.
- **A test query exists with no Exact product at all.** `query_id` 45928,
  `'glass water bottle hot and cold 32'`, has 15 judgements and every one is
  **S**. Under an E-only definition its recall is 0/0. Counting it as 0.0 would
  drag the mean; dropping it silently would make two runs with different
  denominators look comparable. Task 1 makes the choice explicit and records
  the denominator. Under E+S no test query is empty.
- **The dense encoder only reads 128 tokens.** `all-MiniLM-L12-v2` has
  `max_seq_length = 128`, and the concatenated product text averages 1,748
  characters — roughly 440 tokens. About 70% of the text never reaches the
  model, so *field order decides what survives*. This is the same trap that
  cost Plan 3's semantic gate 6 points when titles were sliced at 77
  characters; it is measured here rather than assumed.
- **The corpus embed is cheap and the image embed is not.** SBERT runs at 821
  docs/s on the 3080 — **25 minutes** for all 1,215,854 products, 0.93 GB at
  float16 and 384 dimensions. The CLIP image store already holds the 361,875
  re-ranking vectors; extending it to the catalogue is **525,166 more URLs,
  about 3.6 hours** at Plan 3's measured 2,463 img/min.
- **Every judged product is in the corpus.** 100.0000% of both splits'
  judgements name a `product_id` present in `data/combined/products.parquet`,
  so Recall@k over the 1.2M corpus has a well-defined ceiling of 1.0 and a
  miss is always the retriever's fault, never the table's.

Measured BM25 baseline, over 1,000 randomly sampled test queries against the
full 1,215,854-product index — the number Phase 3's fusion has to beat:

| relevant set | R@10 | R@50 | R@100 | R@500 | R@1000 |
|---|---|---|---|---|---|
| **E** | 0.2338 | 0.4192 | **0.5018** | 0.6665 | 0.7297 |
| **E+S** | 0.1524 | 0.3238 | 0.4065 | 0.5849 | 0.6536 |

Relevant products per test query: **E** median 7, mean 8.9; **E+S** median 15,
mean 16.0. Corpus field presence: `product_title` 100%, `description` 88.4%,
`product_bullet_point` 85.3%, `product_brand` 94.1%.

---

## The storage budget is exceeded, and that is a reportable fact

`PROJECT_SPEC.md` §9 budgets "Embeddings <1 GB". The three channels together
do not fit:

| artifact | size | scope |
|---|---|---|
| BM25 index (`data/bm25/`) | 1.10 GB | catalogue |
| Dense text vectors, fp16 384-d | 0.93 GB | catalogue |
| CLIP image vectors, fp16 512-d | 0.91 GB | catalogue (0.38 GB at `rerank` today) |
| **total** | **2.94 GB** | |

§9 was written before the channels were scoped. Do not shrink a channel to
hit a number the spec set for a smaller design — record the overrun in
`docs/results/recall.json` and state it in the writeup. Disk is free here;
the honest figure is worth more than a flattering one. What §9's budget
*does* still constrain is RAM at query time, which is why the BM25 index is
memory-mapped and the vector stores are opened with `np.memmap`.

---

## Scope: what "recall" is measured over

`CLAUDE.md` is explicit that the task is re-ranking and that recall is
evaluated **separately**: Recall@k over the corpus, NDCG over the judged
candidate list. This plan owns the first and touches neither the second nor
any ranker.

Concretely, for each test query the channels retrieve from all 1,215,854
products, and Recall@k asks what share of that query's relevant products came
back. The judged candidate list is *not* a filter — retrieving only judged
products would make recall trivially 1.0 and measure nothing.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Never tune on test.** `k`, the RRF constant, the field set and the rewrite prompt are all chosen on the frozen validation folds from Plan 1 (`splits/val_folds.csv`, `src.splits.load_folds`). Test is measured once, at the end of Phase 3.
- **Features are query×product, never product-alone-from-labels.** 34,756 products appear in both splits; the split is query-level, so anything derived from labels per product leaks.
- **Always report against a baseline with bootstrap CIs over queries.** Use `src.bootstrap.bootstrap_ci` and `src.bootstrap.paired_delta_ci`, which are paired by query. A channel that ties BM25 is a legitimate, reportable result.
- **Recall's baseline is BM25 at R@100 = 0.5018**, not the NDCG random floor. The floor is a ranking quantity and does not apply here; quoting 0.7467 next to a recall number is a category error.
- **Run GPU-capable work on the GPU.** RTX 3080 Laptop (16 GB, sm_86) via WSL2. Pass `device` explicitly and fail loudly if `cuda` is requested and unavailable.
- **Build the BM25 index once, then memory-map it.** In-process build-and-query peaks at 18.3 GB against 23 GB of RAM.
- **Absence is not a zero score.** The image channel covers 77.50% of products and the dense channel 100%; a product missing from a channel must be representable as absent, never as a bottom-ranked hit. Plan 3's `EmbeddingStore.lookup` already returns a presence mask — use it.
- **Do not commit datasets.** `.gitignore` blocks `data/`. Indexes and vectors live under `data/`; only the JSON records under `docs/results/` are committed.
- **Stage 0 costs money.** It is the only paid component in the project. Cache every rewrite to disk, keyed by query, so a re-run is free and the ablation is reproducible.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)

---

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **A query with no relevant product.** `query_id` 45928 has 15 judgements, all S, so its E-recall is 0/0. Scored as 0.0 it silently drags the mean; dropped silently it makes runs with different denominators look comparable. The metric must define the case and report how many queries it covered. — pinned in Task 1.
2. **Row indices mistaken for product ids.** `bm25s.retrieve` returns positions into the indexed matrix, not `product_id`s. The id array is a separate file written by a separate line of code; if the two are ever built from differently-ordered frames, every result is wrong and *nothing raises* — recall simply comes out low, which looks like a weak retriever rather than a broken one. — pinned in Task 2.
3. **Truncation deciding the result.** SBERT reads 128 tokens of a 440-token string, so whichever fields come first are the only ones embedded. A field order chosen without measuring is a silent, large effect — Plan 3's 77-character title slice cost 6 points of top-1. — pinned in Task 3.
4. **RRF treating "absent from this channel" as "ranked last".** The image channel has no vector for 22.5% of products. If absence contributes a real reciprocal-rank term, those products are pushed down by a scrape artefact rather than by evidence; if it contributes the same as rank ∞, a channel that retrieved nothing still votes. Fusion must sum only over channels that actually returned the product. — pinned in Task 5.
5. **A rewritten query that silently becomes a different query.** Stage 0 is an LLM call: it can return an empty string, a refusal, a JSON wrapper, or a paraphrase that drops the brand or model number the query was about. Used unchecked, Ablation 1 measures prompt damage rather than query understanding, and the raw-query arm looks artificially good. Rewrites must be validated and fall back to the raw query. — pinned in Task 6.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as
`src.<name>` and run as `python -m src.<name>`. This plan follows it.

| File | Responsibility | Written in |
|---|---|---|
| `src/recall.py` | Relevant sets from judgements, Recall@k, the `Channel` protocol. No retrieval | Phase 1, Task 1 |
| `src/bm25_index.py` | Build, save, memory-map and query the lexical index; `python -m src.bm25_index` | Phase 1, Task 2 |
| `src/vector_search.py` | Top-k cosine over any `EmbeddingStore`, with a presence mask. No model, no I/O | Phase 2, Task 3 |
| `src/dense_embed.py` | SBERT corpus embedding run; `python -m src.dense_embed` | Phase 2, Task 3 |
| `src/image_channel.py` | CLIP query-text → image-store search, built on `vector_search` | Phase 2, Task 4 |
| `src/rrf.py` | Reciprocal Rank Fusion over ranked lists | Phase 3, Task 5 |
| `src/query_rewrite.py` | Stage 0 LLM rewriting with an on-disk cache and validation | Phase 3, Task 6 |
| `src/recall_report.py` | Ablations 1 and 2 as a table with bootstrap CIs; `python -m src.recall_report` | Phase 3, Task 7 |
| `docs/results/recall.json` | Committed Recall@k record for every channel and the fusion | Phase 3, Task 7 |
| `tests/test_recall.py`, `tests/test_bm25_index.py`, `tests/test_vector_search.py`, `tests/test_dense_embed.py`, `tests/test_image_channel.py`, `tests/test_rrf.py`, `tests/test_query_rewrite.py`, `tests/test_recall_report.py` | One test module per source module | Every task |

`src/vector_search.py` exists so the dense-text and image channels are the
same code with a different store: both encode a query, cosine it against a
float16 matrix and take the top k. Writing that twice is how the two channels
drift apart.

---

## What this plan reuses rather than rebuilds

| From | What | Why it matters |
|---|---|---|
| Plan 1 | `src.dataset.load_split`, `src.splits.load_folds`, `src.bootstrap.*` | Validation folds are already frozen; model selection has somewhere legitimate to happen |
| Plan 1 | `src.baseline_sbert.product_text` | Field joining that drops nulls rather than embedding the literal `"None"` |
| Plan 3 | `src.embedding_store.open_store`, `.lookup` | Crash-consistent float16 storage with a presence mask, keyed by an arbitrary string — `product_id` here, image URL there |
| Plan 3 | `src.clip_encoder.load_encoder`, `.encode_texts` | The image channel's query side is CLIP text into the same 512-d space |
| Plan 3 | `src.embed_images` | `--scope catalogue` extends the image store; append-only, so it re-fetches only the difference |
| Plan 2 | `data/combined/products.parquet` | One table, 1,215,854 products, coalesced `description` at 89.2% |

---

## Plan Gate

Plan 5 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the index round-trip on the real corpus.
- [ ] `data/bm25/` memory-maps in under 2 GB of RSS and answers under 100 ms per query amortised over a batch.
- [ ] Every channel reports Recall@{10,50,100,500,1000} on the frozen validation folds, and the fused run beats the **BM25 R@100 baseline of 0.5018** or reports honestly that it ties, with a paired bootstrap CI.
- [ ] `docs/results/recall.json` records per-channel and fused Recall@k, the query-coverage denominator, and the storage overrun against §9's <1 GB.
- [ ] Ablation 1 (raw vs. rewritten query) and Ablation 2 (dense-only vs. +BM25 vs. +image) are both reported with bootstrap CIs.
- [ ] `CLAUDE.md`'s Commands section lists the index build, the corpus embed and the recall report.

Then: Plan 5 — Coarse Rank. See [`../README.md`](../README.md) for the series.
