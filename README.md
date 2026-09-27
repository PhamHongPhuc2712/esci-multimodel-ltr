# Multi-Stage Multimodal Product Search Ranking on Amazon ESCI

A full product-search funnel — query rewriting → hybrid recall → LambdaMART
coarse rank → cross-encoder / LLM fine rank → blend — built on the Amazon ESCI
Shopping Queries Dataset. ESCI-S adds images, prices, ratings, categories and
attributes. Every stage is measured with an ablation against a computed random
floor, with bootstrap intervals over queries.

**Full writeup: [`docs/RESULTS.md`](docs/RESULTS.md).** Every number on this
page is quoted from a committed file in [`docs/results/`](docs/results/), and
a test fails if one is not.

## Headline

ESCI Task 1 (English) is a **re-ranking** task: each query's judged candidates
are ranked, scored by full-list NDCG with gains E 1.0 / S 0.1 / C 0.01 / I 0.0.

| ranker | NDCG | scope |
|---|---|---|
| random ordering (computed floor) | 0.7467 | full test split, 8,956 queries |
| zero-shot SBERT (published 0.8292, reproduced here) | 0.8294 | full test split |
| `ESCI_baseline`, a fine-tuned cross-encoder (published, the target) | 0.8562 | full test split |
| **coarse LambdaMART, 47 features** | **0.8579 [0.8551, 0.8611]** | full test split |
| + fine-tuned cross-encoder on its top 10 | 0.8616 | full test split |
| + a larger fine-tuned cross-encoder (`bge-reranker-base`) on its top 10 | 0.8695 [0.8667, 0.8724] | full test split |
| **+ LLM listwise re-ranking of its top 10** | **0.8855 [0.8827, 0.8883]** | full test split |

**The funnel beats the published baseline on the same 8,956 queries.** The
LLM arm's interval sits wholly above 0.8562, at the price of an API call that
takes 4.7 s a query. Without the LLM, the coarse ranker alone **matches** the
target: its interval contains 0.8562. A larger cross-encoder on its top 10
**beats** it with no API call, at 72.5 ms a query. The LLM arm was first measured on a frozen
2,000-query sample (0.8855 [0.8799, 0.8910]); running the whole split once
left the point estimate unchanged and halved the interval.

## What the ablations found

| | finding | delta |
|---|---|---|
| images, coarse rank | CLIP image features help the ranker | +0.0075 [+0.0059, +0.0090] NDCG |
| images, recall | …but barely help retrieval | +0.0048 Recall@100 |
| behavioural features | help a little, once the scrape's presence artefact is controlled | +0.0036 [+0.0023, +0.0051] NDCG |
| learning to rank | `lambdarank` beats pointwise classification | +0.0085 NDCG |
| fine rank | the LLM buys about seven times the cross-encoder's gain at over two hundred times its latency | +0.0276 vs +0.0037 NDCG |
| query rewriting | an LLM rewrite of the query **ties** | +0.0048 [−0.0007, +0.0107] Recall@100 |
| learned fusion | **ties** a single hand-tuned text/image weight | −0.0011 [−0.0023, +0.0000] NDCG |
| blending | **no blend beats the LLM alone** — all three lose to it, though a per-query oracle would add +0.0257 | best: −0.0006 [−0.0009, −0.0003] NDCG |
| distillation | **the LLM is not a better teacher than the labels**, so the paid labelling was not run | ties from both starting points on fold 0 |
| model size, fine rank | a larger cross-encoder (`bge-reranker-base`) beats the small one, keeping 42% of the LLM's gain at 72.5 ms | +0.0079 [+0.0062, +0.0097] NDCG |

Each row's scope, `n` and source are in the full table in
[`docs/RESULTS.md`](docs/RESULTS.md#4-the-ablation-table). The ties and losses
are reported as results.

## The funnel

| stage | what | measured by |
|---|---|---|
| 0 | LLM query rewriting (`gpt-5.6-luna`), cached | Recall@100 lift |
| 1 | BM25 + dense text (SBERT) + CLIP image, fused with RRF, over the 1.2M-product catalogue | Recall@100 |
| 2 | LightGBM `lambdarank` over 47 query × product features: retrieval scores, text overlap, image similarity, ESCI-S behavioural values, categories, attributes | NDCG |
| 3 | fine-tuned `ms-marco-MiniLM-L6-v2` cross-encoder, or an LLM listwise call, re-ranking Stage 2's top 10 | NDCG, latency, tokens |
| 4 | fixed-weight RRF, a learned combiner, or a per-query selector over the three stages | NDCG against each stage alone |

Recall is evaluated on its own, over the whole catalogue. Stages 2–4 rank only
the judged candidates, as the task defines.

## Things that were wrong in the data, and would have corrupted results

- **The Hugging Face mirror of ESCI is the wrong split.** It encodes the
  large-version split: 185,361 test judgements instead of 181,701. The loader
  asserts the official counts on every load.
- **The official eval script swaps the S and C gains**, which moves the random
  floor by over three points — more than most method gains. The floor is
  computed on every report, never quoted.
- **About half the ESCI-S image URLs point at a retired CDN bucket** and return
  HTTP 400. Stripping one path segment fixes them: 600 of 600 sampled URLs
  resolve afterwards.
- **Books store their image under a different key** (`img`, not `image`).
  Reading only `image` silently drops every book.
- **"Has an image" and "has a price" are themselves ranking signals** — the
  2022 scrape failing on a page — and are not available at serving time. The
  behavioural ablation controls for them rather than crediting them.

## Reproducing

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -e ".[dev,baselines,retrieval,ranking]"
python -m pytest                      # fast suite; -m data needs the datasets below
```

The ESCI parquet files download themselves on first use. ESCI-S is a single
3.4 GB file:

```bash
mkdir -p data/esci-s
curl -o data/esci-s/esci.json.zst https://esci-s.s3.amazonaws.com/esci.json.zst
```

Then, in order. The validation folds are already frozen in `splits/`. A GPU is
used where one is present. The steps marked paid need `OPENAI_API_KEY`, and
every LLM call is cached.

```bash
python -m src.esci_s_etl && python -m src.combine        # enrichment corpus, joined tables
python -m src.embed_images --scope catalogue             # CLIP vectors, network-bound, ~6 h
python -m src.bm25_index && python -m src.dense_embed    # retrieval indexes
python -m src.query_rewrite --split train --folds 0      # Stage 0 (paid)
python -m src.recall_report --split train --folds 0 --rewrites data/rewrites.json
for s in train test; do python -m src.pair_scores --split $s; python -m src.feature_matrix --split $s; done
python -m src.rank_report                                # Stage 2 ablations on validation fold 0
python -m src.rank_report --split test --final --out docs/results/coarse-rank-test.json
for s in train test; do python -m src.stage2_scores --split $s; done
python -m src.cross_encoder --loss lambda && python -m src.cross_encoder --loss bce
python -m src.llm_rerank --split train --folds 0         # Stage 3 LLM (paid)
python -m src.fine_rank_report --llm-latency-probe 40
python -m src.fine_rank_report --split test --final --llm-latency-probe 40 --out docs/results/fine-rank-test.json
python -m src.llm_rerank --split test --folds all --usage-out docs/results/llm-rerank-test-full.json   # (paid, ~2.3 h)
python -m src.fine_rank_report --split test --final --sample 0 --skip-cascade --out docs/results/fine-rank-test-full.json
for s in fold0 test-sample test; do python -m src.stage_signals --scope $s; done
python -m src.blend_report --scope fold0
python -m src.blend_report --scope test-sample --final --out docs/results/blend-test.json
python -m src.blend_report --scope test --final --out docs/results/blend-test-full.json
python -m src.error_analysis && python -m src.ablation_table
python -m src.stage2_scores --split train --out-of-fold  # Plan 8: training windows from a Stage 2 that never saw them
python -m src.distill --target labels --out models/cross-encoder/windows
python -m src.distill --target labels --init BAAI/bge-reranker-base --batch-size 8 --out models/cross-encoder/bge
python -m src.distill_report && python -m src.distill_report --pilot
python -m src.distill_report --split test --final --arms stage2+ce_windows stage2+ce_bge --out docs/results/distill-test.json
```

Every configuration is chosen on a validation fold carved from train by query.
Each test measurement sits behind `--final` and was run once.

## Layout

| path | what |
|---|---|
| `src/` | one module per step, each runnable as `python -m src.<name>` |
| `tests/` | one test module per source module; `-m data` runs against the real datasets |
| `docs/results/` | every committed measurement, as JSON |
| `docs/RESULTS.md` | the writeup |
| `docs/superpowers/plans/` | the eight implementation plans, with what each measured and corrected |
| `PROJECT_SPEC.md` | motivation, funnel design, baselines and the ablation plan |
| `splits/` | the frozen validation folds and their checksum |
