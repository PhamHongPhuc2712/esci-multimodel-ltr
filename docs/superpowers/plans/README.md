# Plan Series — Multi-Stage Multimodal Product Search Ranking on Amazon ESCI

`PROJECT_SPEC.md` is the source of truth. This file decomposes it into a series
of implementation plans, each of which produces working, testable software on
its own and is gated on the one before it.

The decomposition follows the spec's own §8 Build Order. The ordering is not a
preference — §8 states *"Nothing else until the metric is trustworthy"*, and
every downstream number is meaningless if Plan 1's harness is wrong.

## Why seven plans and not one

The spec covers seven independent subsystems. A single plan would have to fix
the Parquet schema for the coarse ranker before the ETL that produces it has
been written, and the feature vector layout before the retrieval channels that
fill it exist. Each plan below ends with an artifact whose schema the next plan
consumes, so each plan is written *after* its predecessor lands and its real
interfaces are known.

## The series

| # | Plan | Delivers | Gate to pass before the next plan | Spec |
|---|---|---|---|---|
| 1 | [**Evaluation Foundation**](2026-09-19-evaluation-foundation/) | ESCI loader with asserted invariants, full-list NDCG, computed random floor, bootstrap CIs, frozen validation folds, one reproduced published baseline | Zero-shot SBERT lands in `[0.820, 0.840]`; NDCG matches `pytrec_eval` exactly on the real test split | §2, §3.1, §8.1 |
| 2 | [**Enrichment Corpus**](2026-09-19-enrichment-corpus/) | Single-pass streaming ETL of ESCI-S (3.4 GB zstd) to Parquet; join coverage and missingness-bias report | Join coverage ≥ 88% of the re-ranking products (measured 0.8959); dense-field missingness under 0.02 mean gain | §3.2, §8.2 |
| 3 | [**Image Pipeline**](2026-09-20-image-pipeline/) | Resolution gate re-run, bulk async fetch, in-flight CLIP embedding, embedding store **(landed)** | `python -m src.esci_images` ≥ 90%; end-to-end image coverage reported against the ~75% figure, not 91.5% | §3.3, §8.3 |
| 4 | [**Recall**](2026-09-20-recall/) | BM25 + dense + CLIP channels, RRF fusion, Recall@k harness, Stage 0 LLM query rewriting **(landed)** | Ablations 1 and 2 produce a table with bootstrap CIs | §4.0, §4.1, §8.4 |
| 5 | [**Coarse Rank**](2026-09-21-coarse-rank/) | Query×product feature extraction, LightGBM `lambdarank`, pointwise A/B | Ablations 3, 4, 5, 7 produce a table with bootstrap CIs | §4.2, §8.5 |
| 6 | [**Fine Rank**](2026-09-21-fine-rank/) | Fine-tuned cross-encoder and LLM listwise reranker, head to head on Stage 2 top-K | Ablation 6 produces NDCG + latency + cost | §4.3, §8.6, §8.7 |
| 7 | **Blend and Report** | Stage 4 learned combiner, full ablation table, per-category error analysis, writeup | Beats the 0.8562 `ESCI_baseline` target, or reports honestly that it ties | §4.4, §5, §6, §8.8 |

A written plan gets its own directory and is split into phase files when a single file stops being readable end to end. Six are written so far, all split that way:

- [`2026-09-19-evaluation-foundation/`](2026-09-19-evaluation-foundation/) — Plan 1, an index plus three phases: the metric, the data, then statistics and the baseline. **Landed.** Its gate passed with zero-shot SBERT at NDCG 0.8294 against a published 0.8292, a measured random floor of 0.7467, and a lift of +0.0827 [+0.0802, +0.0855].
- [`2026-09-19-enrichment-corpus/`](2026-09-19-enrichment-corpus/) — Plan 2, an index plus two phases: the record, then the corpus. Written against the real `esci.json.zst`, not its README: the field-presence figures, the book/product key split, the multi-price strings and zstandard's silence on truncation were all measured first. **Landed.** 1,080,262-row corpus at 89.59% join coverage.
- [`2026-09-20-image-pipeline/`](2026-09-20-image-pipeline/) — Plan 3, an index plus two phases: the embedder, then the fetch. Written against the live CDN and the real GPU: 300/300 and 600/600 URLs resolved, CLIP measured at 18,140 img/min against a 3,378–5,516 img/min fetch, float16 storage shown lossless for retrieval, and CLIP's image→title top-1 measured at 75.3% to set the gate. **Landed.** 361,875 vectors (382 MB) over 362,005 distinct URLs, 77.50% end-to-end image coverage against the ~77.5% target, semantic gate 76.0% against a 0.60 threshold.

- [`2026-09-20-recall/`](2026-09-20-recall/) — Plan 4, an index plus three phases: the metric and the lexical channel, the learned channels, then fusion and the ablations. Written against the real corpus: the in-memory BM25 build measured at 18.3 GB peak against 23 GB of RAM (memory-mapped, 1.55 GB and faster), BM25 Recall@100 measured at 0.5018 to set the baseline, SBERT at 821 docs/s for a 25-minute corpus pass, and `query_id` 45928 found to have no Exact product at all. **Landed.** On validation fold 0 (4,130 queries, E-relevance, full 1.2M corpus): BM25 R@100 0.4890, dense 0.4608, image 0.1971, and RRF fusion 0.5551 — **+0.0661 [+0.0598, +0.0727]** over BM25. Ablation 2's ladder puts +BM25-over-dense at +0.0895 and **+image-over-(dense+bm25) at only +0.0048 [+0.0008, +0.0088]** — real but marginal. Ablation 1's LLM rewriting **ties**: +0.0048 [-0.0007, +0.0107], and it costs R@10 (0.2657 -> 0.2571).

- [`2026-09-21-coarse-rank/`](2026-09-21-coarse-rank/) — Plan 5, an index plus
  three phases: the features, the matrix and the ranker, then the ablations.
  Written against the real matrix: LightGBM's own `ndcg@10` measured **6.4
  points** away from this project's on the same booster (0.7931 under ESCI's
  gains, 0.8575 under the default `2**rel - 1`), `get_scores([])` found to raise
  `IndexError` on the queries that tokenise to nothing (7 of 20,888 in train),
  the `rerank` image store found to be a strict subset of the catalogue one
  (0 URLs missing, identical 77.50% judged coverage), and each §4.2 retrieval
  signal measured alone on fold 0: BM25 0.8230, dense 0.8285, CLIP image 0.7905.
  **Landed.** 47 features in eight groups over 419,653 train and 181,701 test
  judgements; headline **NDCG 0.8579 [0.8551, 0.8611]** on test against a
  measured floor of 0.7468 — the CI contains the 0.8562 `ESCI_baseline` target,
  so this *matches* it rather than significantly beating it. Ablation 3
  **+0.0075 [+0.0059, +0.0090]** (images help). Ablation 4 needed a **fourth**
  arm — subtracting the indicators-only arm over-corrects to −0.0107, while the
  assumption-free `text+indicators` control puts the behavioural contribution at
  **+0.0036 [+0.0023, +0.0051]**. Ablation 5: `lambdarank` beats pointwise
  regression by **+0.0096** and 4-class pointwise by **+0.0085**. Ablation 7
  **fails its hypothesis** — learned fusion over the same two signals loses to a
  single hand-tuned global weight on fold 0 (−0.0023 [−0.0040, −0.0008]) and
  ties on test (−0.0011 [−0.0023, +0.0000]).

- [`2026-09-21-fine-rank/`](2026-09-21-fine-rank/) — Plan 6, an index plus three
  phases: the window, the cross-encoder, then the LLM and the ablation. Written
  against the real Stage 2 output: an oracle reorder of Plan 5's own top-10
  measured at **0.9533** against its 0.8519, so there is **+0.1014** of headroom
  and only 3.3% of fold-0 queries are already perfect. Every **zero-shot**
  cross-encoder measured *below* Stage 2 on an identical 400-query sample
  (0.8286–0.8445 against 0.8577), so the fine-tune is the arm rather than an
  optimisation of it; fine-tuning measured at 29 min/epoch with `LambdaLoss` and
  10.6 min with `BinaryCrossEntropyLoss`, both inside 16 GB. The LLM listwise
  arm returned **8/8** well-formed permutations on `gpt-5.6-luna`, with **88% of
  completion tokens spent on reasoning** — 386 prompt + 387 completion per
  top-10 window against 1,015 + 1,716 for the full list, which is why the window
  exists. **Phases 1 and 2 landed 2026-09-22**; Phase 3 is written, not yet
  executed. Phase 2's fine-tune is the arm, as predicted: zero-shot
  `ms-marco-MiniLM-L6-v2` scores **0.8457** on fold 0 against Stage 2's 0.8519,
  while the fine-tuned arms reach **0.8587** (`LambdaLoss`) and **0.8594**
  (`BinaryCrossEntropyLoss`) — **+0.0130** for the fine-tune. The listwise loss
  the phase made default does *not* win at Stage 3, reversing Plan 5's
  Ablation 5 by a small margin, and `bce` trains 5.4x faster. Phase 1 persisted Plan 5's ordering to
  `data/features/stage2-{train,test}.parquet`, reproducing **0.8519** on fold 0
  and **0.8579** on test exactly, and added the top-K splice whose identity
  re-ranking leaves per-query NDCG untouched. It corrected one thing the plan
  had wrong: Plan 5's *test* headline was fitted on every train fold, not on
  folds 2/3/4, and fitting the narrow set for both lands 0.0020 low.

Start at a plan's `README.md`.

## Outside the plan series

`src/combine.py` (`python -m src.combine`) joins Plan 1's ESCI loader and Plan
2's ESCI-S corpus into `data/combined/{products,judgements}.parquet` — the
tables Plans 4 and 5 read. It is not a plan deliverable; it was built when the
two halves first existed and both were needed in one place. Its `description`
coalesce takes description coverage from 52.2% to 89.2%, which is the single
largest text gain the enrichment provides.

## Ablation ownership

Every row of the spec's §6 table is owned by exactly one plan. No plan is
"done" while an ablation it owns is unreported.

| Ablation | Owner |
|---|---|
| 1 — Raw vs. LLM-rewritten query | Plan 4 |
| 2 — Dense-only vs. +BM25 vs. +image (RRF) | Plan 4 |
| 3 — Text-only vs. text+image features | Plan 5 |
| 4 — With vs. without behavioural features | Plan 5 |
| 5 — Pointwise vs. `lambdarank` | Plan 5 |
| 6 — Coarse-only vs. +cross-encoder vs. +LLM listwise | Plan 6 |
| 7 — Learned fusion vs. fixed global weight | Plan 5 |

## Constraints that outlive any single plan

These come from `PROJECT_SPEC.md` and `CLAUDE.md`. Every plan in the series
inherits them; they are restated in each plan's Global Constraints section.

- **Never tune on test.** Model selection uses the frozen validation folds from
  Plan 1 only.
- **Features are query×product, never product-alone-from-labels.** 34,756
  products appear in both train and test; the split is query-level, so
  product-level target encoding leaks train labels into test.
- **Always report against the computed random floor, with bootstrap CIs over
  queries.** A method that ties the baseline is a legitimate, reportable result.
- **Commit messages are one line, and never mention Claude, Claude Code,
  Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:`
  trailer, not as a "Generated with" line. This applies to pull request
  descriptions too. (`CLAUDE.md`; it overrides any harness default that would
  add an attribution trailer.)
