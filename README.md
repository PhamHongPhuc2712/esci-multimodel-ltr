# Multimodal Product Re-Ranking on Amazon ESCI (Task 1)

A multi-stage re-ranker for **ESCI Task 1**, the query–product ranking task of
the Amazon Shopping Queries Dataset: given a query and its judged candidate
products, order them so the relevant ones come first. A LightGBM LambdaMART
ranker over 47 text, image and behavioural features orders each query's list.
A fine-rank stage then re-orders its top 10: a fine-tuned cross-encoder, a
fine-tuned open LLM served by vLLM, or a hosted LLM. A blend stage tests
whether combining them helps. ESCI-S adds images, prices, ratings, categories
and attributes.

**Project site: [phamhongphuc2712.github.io/esci-multimodel-ltr](https://phamhongphuc2712.github.io/esci-multimodel-ltr/).**
**Full writeup: [`docs/RESULTS.md`](docs/RESULTS.md).** Every number on this
page is quoted from a committed file in [`docs/results/`](docs/results/), and
a test fails if one is not.

## Headline

Ranking is scored as full-list NDCG over each query's judged list, with gains
E 1.0 / S 0.1 / C 0.01 / I 0.0, on all 8,956 queries of the test split.

| ranker | NDCG | p95 latency added |
|---|---|---|
| random ordering (computed floor) | 0.7467 | — |
| zero-shot SBERT (published 0.8292, reproduced here) | 0.8294 | — |
| `ESCI_baseline`, a fine-tuned cross-encoder (published, the target) | 0.8562 | — |
| **coarse LambdaMART, 47 features** | **0.8579 [0.8551, 0.8611]** | — |
| + fine-tuned MiniLM cross-encoder on its top 10 | 0.8616 | — |
| + fine-tuned `bge-reranker-base` on its top 10 | 0.8711 [0.8681, 0.8742] | 31.6 ms (fp16, RTX A6000) |
| + fine-tuned `bge-reranker-large` on its top 10 | 0.8770 [0.8742, 0.8801] | 46.1 ms (fp16, RTX A6000) |
| **+ fine-tuned `Qwen3-Reranker-4B` on its top 10, served by vLLM in INT8** | **0.8843 [0.8815, 0.8872]** | **146.0 ms (RTX A6000)** |
| + hosted LLM listwise re-ranking of its top 10 | 0.8855 [0.8827, 0.8883] | a 4.7 s API call |

**Every arm from the coarse ranker up beats or matches the published
baseline on the same queries.** The coarse ranker alone **matches** the
0.8562 target: its interval contains it. Each fine-rank arm lies wholly above
it.

**The recommended re-ranker is the fine-tuned `Qwen3-Reranker-4B`.** It beats
`bge-reranker-large` by +0.0078 and **ties** the hosted LLM, self-served on
one GPU, with no per-call cost. Served by vLLM in INT8 it answers at a
146.0 ms p95 and holds a p95 under 300 ms at 4 queries a second. Where a GPU
that size is not available, `bge-reranker-large` in fp16 is three times
faster for −0.0078.

The hosted LLM arm was first measured on a frozen 2,000-query sample (0.8855
[0.8799, 0.8910]); running the whole split once left the point estimate
unchanged and halved the interval.

![Stage 3 rerankers: quality against latency](docs/figures/quality-vs-latency.svg)

## What the ablations found

| | finding | delta |
|---|---|---|
| images | CLIP image features help the coarse ranker | +0.0075 [+0.0059, +0.0090] NDCG |
| behavioural features | help a little, once the scrape's presence artefact is controlled | +0.0036 [+0.0023, +0.0051] NDCG |
| learning to rank | `lambdarank` beats pointwise classification | +0.0085 NDCG |
| fine rank | the hosted LLM buys about seven times the MiniLM cross-encoder's gain | +0.0276 vs +0.0037 NDCG |
| learned fusion | **ties** a single hand-tuned text/image weight | −0.0011 [−0.0023, +0.0000] NDCG |
| blending | **no blend beats the LLM alone**, though a per-query oracle would add +0.0257 | best: −0.0006 [−0.0009, −0.0003] NDCG |
| distillation | in a fold-0 pilot the LLM's ordering was **not a better training target than the labels**, so the paid labelling was not run | ties from both starting points |
| model size | with everything else held fixed, the larger backbone (`bge-reranker-base` against MiniLM) is what pays | +0.0112 [+0.0094, +0.0130] NDCG |
| one size up | `bge-reranker-large` against -base, same recipe, pays again | +0.0059 [+0.0044, +0.0074] NDCG |
| self-served LLM | a fine-tuned 4B open LLM beats `bge-reranker-large`… | +0.0078 [+0.0061, +0.0096] NDCG |
| | …and **ties** the hosted LLM | −0.0008 [−0.0025, +0.0013] NDCG |

Every row with an interval is in [`docs/results/ablation-table.json`](docs/results/ablation-table.json),
read from the file that measured it, and a test checks that. The scope, `n`
and source of each are in the full table in
[`docs/RESULTS.md`](docs/RESULTS.md#4-the-ablation-table). The ties and
losses are reported as results.

## The pipeline

Stages keep the numbers the code and results files use.

| stage | what | measured by |
|---|---|---|
| input | each query's judged candidates, as ESCI Task 1 defines them | — |
| 2 | LightGBM `lambdarank` over 47 query × product features: BM25 and dense similarity, text overlap, CLIP image similarity, ESCI-S behavioural values, categories, attributes | NDCG |
| 3 | re-ranking Stage 2's top 10: a fine-tuned cross-encoder (`ms-marco-MiniLM-L6-v2`, `bge-reranker-base`, `bge-reranker-large`), a fine-tuned open LLM (`Qwen3-Reranker-4B`) served by vLLM, or a hosted LLM listwise call | NDCG, latency, load |
| 4 | fixed-weight RRF, a learned combiner, or a per-query selector over the stages | NDCG against each stage alone |

An earlier version also had LLM query rewriting and a recall stage over the
1.2M-product catalogue. Retrieval is not part of any ESCI task, so it was
removed; it lives at the `v1-with-retrieval` tag.

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
uv pip install -e ".[dev,baselines,retrieval,ranking,llm,report]"
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
python -m src.embed_images --scope rerank                # CLIP vectors for the judged products, network-bound
python -m src.bm25_index && python -m src.dense_embed    # the BM25 and dense similarity features
for s in train test; do python -m src.pair_scores --split $s; python -m src.feature_matrix --split $s; done
python -m src.rank_report                                # Stage 2 ablations on validation fold 0
python -m src.rank_report --split test --final --out docs/results/coarse-rank-test.json
for s in train test; do python -m src.stage2_scores --split $s; done
python -m src.cross_encoder --loss lambda && python -m src.cross_encoder --loss bce
python -m src.llm_rerank --split train --folds 0         # Stage 3 hosted LLM (paid)
python -m src.fine_rank_report --llm-latency-probe 40
python -m src.fine_rank_report --split test --final --llm-latency-probe 40 --out docs/results/fine-rank-test.json
python -m src.llm_rerank --split test --folds all --usage-out docs/results/llm-rerank-test-full.json   # (paid, ~2.3 h)
python -m src.fine_rank_report --split test --final --sample 0 --skip-cascade --out docs/results/fine-rank-test-full.json
for s in fold0 test-sample test; do python -m src.stage_signals --scope $s; done
python -m src.blend_report --scope fold0
python -m src.blend_report --scope test-sample --final --out docs/results/blend-test.json
python -m src.blend_report --scope test --final --out docs/results/blend-test-full.json
python -m src.error_analysis && python -m src.ablation_table
python -m src.stage2_scores --split train --out-of-fold  # training windows from a Stage 2 that never saw them
python -m src.distill --target labels --out models/cross-encoder/windows
python -m src.distill --target labels --init BAAI/bge-reranker-base --batch-size 8 --out models/cross-encoder/bge
python -m src.distill_report && python -m src.distill_report --pilot
python -m src.distill_report --split test --final --arms stage2+ce_windows stage2+ce_bge --out docs/results/distill-test.json
python -m src.distill --target labels --init BAAI/bge-reranker-base --batch-size 16 --gradient-checkpointing --out models/cross-encoder/bge-b16
python -m src.cross_encoder --loss lambda --backbone BAAI/bge-reranker-base --gradient-checkpointing --out-dir models/cross-encoder/bge-groups
python -m src.distill_report --capacity && python -m src.distill_report --capacity --split test --final
python -m src.cross_encoder --loss lambda --backbone BAAI/bge-reranker-large --gradient-checkpointing --batch-size 4 --grad-accum 2 --out-dir models/cross-encoder/bge-large-groups
python -m src.distill_report --size && python -m src.distill_report --size --split test --final
```

The self-served LLM rerankers, their serving ladder and load test ran on a
rented RTX A6000; `CLAUDE.md` lists those commands and `scripts/cloud/` holds
the scripts that ran there. `python -m src.quality_latency_chart` regenerates
the chart from the committed JSON.

Every configuration is chosen on a validation fold carved from train by query.
Each test measurement sits behind `--final` and was run once.

## Layout

| path | what |
|---|---|
| `src/` | one module per step, each runnable as `python -m src.<name>` |
| `tests/` | one test module per source module; `-m data` runs against the real datasets |
| `docs/results/` | every committed measurement, as JSON |
| `docs/RESULTS.md` | the writeup |
| `docs/figures/` | the quality-against-latency chart, generated from `docs/results/` |
| `site/` | the project site, generated from `docs/results/` and served by GitHub Pages |
| `scripts/cloud/` | the scripts that ran the LLM training and serving on a rented GPU |
| `PROJECT_SPEC.md` | the original motivation, design, baselines and ablation plan |
| `splits/` | the frozen validation folds and their checksum |
