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
| 7 | [**Blend and Report**](2026-09-23-blend-and-report/) | Stage 4 learned combiner, full ablation table, per-category error analysis, writeup **(landed)** | Beats the 0.8562 `ESCI_baseline` target, or reports honestly that it ties | §4.4, §5, §6, §8.8 |

A written plan gets its own directory and is split into phase files when a single file stops being readable end to end. All seven are written, each split that way:

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
  exists. **Landed 2026-09-22.** Headline **`stage2+llm` 0.8855 [0.8799,
  0.8910]** on a frozen 2,000-query test sample against Stage 2's 0.8576 and a
  floor of 0.7454 — **+0.0278 [+0.0236, +0.0325]**, the best ranker in the
  project so far. The fine-tune is the arm, as predicted: zero-shot
  `ms-marco-MiniLM-L6-v2` is a significant *loss* against Stage 2 on both
  splits, while the fine-tuned arms gain +0.0043 (`LambdaLoss`) and +0.0040
  (`BinaryCrossEntropyLoss`, which **ties**) on test. Ablation 6's real content
  is a trade-off, not a winner: the LLM buys ~6.5x the cross-encoder's gain for
  **~150x** its latency (4,713 ms against 22 ms single-query) plus 453 + 381
  tokens a query. Two negatives Plan 7 inherits — **the cascade is redundant**
  (`stage2+ce+llm` never separates from `stage2+llm`), and **the listwise loss
  does not earn its keep at Stage 3**, reversing Plan 5's Ablation 5, while
  `bce` trains 5.4x faster. Malformed permutations: **2 in 2,000**. One defect
  was caught and fixed in this plan's own code — the first Ablation 6 timed a
  cached replay and published the LLM arm at 5.5 ms/query against a real 1,180.

- [`2026-09-23-blend-and-report/`](2026-09-23-blend-and-report/) — Plan 7, the
  last: an index plus three phases — the signals and the blend, the blend
  report, then the analysis and the writeup. Written against the real fold-0
  output, and **four of its measurements say its own central hypothesis will
  fail**: the oracle best-of-three reaches **0.9076** against the best single
  arm's 0.8814, so **+0.0262** of headroom exists, but every swept rank fusion
  loses (best **0.8800** at 1:1:4) and a cross-fitted learned combiner loses
  too (**0.8793**), with `llm_rank` dominating its feature importance. The
  three arms genuinely disagree (Kendall τ 0.34–0.41) and no arm is strictly
  best more than **41.6%** of the time, which is why the plan adds a per-query
  **selector** — the only strategy shaped like the oracle's advantage — and
  reports the ceiling beside every arm so a tie reads as missed headroom
  rather than absent headroom. Also measured for the error analysis: the LLM
  arm **damages 26.4%** of queries (mean −0.0607) while helping 59.0% (mean
  +0.0772); `s_category` is a path with 49 top-level values of which only 13
  clear 100 queries; per-query image coverage averages 0.78. **No new paid API
  calls** — fold 0 and the test sample are already cached, bar three windows
  whose LLM answer was malformed, which Plan 6 scored in Stage 2 order and
  this plan carries the same way, flagged. Six defects were corrected before
  any task ran — the cache-coverage claim, a selector trained on the fold it
  reported, Ablation 2 read as fusion-vs-BM25 rather than as its ladder, and
  three smaller ones; see its README's "Corrected before execution".
  **Landed 2026-09-23, and Stage 4 does not beat its best input.** On the
  frozen 2,000-query test sample the fixed-weight, combiner and selector arms
  score **0.8843, 0.8849 and 0.8853** against the LLM's **0.8855** — all three
  tie — while the oracle reaches **0.9101**; on fold 0 all three lose
  significantly. The selector, corrected during execution to stop `argmax`
  handing every tie to the weakest arm, routes 93.2% of queries to the LLM,
  and its 281 departures hurt more often than they help (146 worse, 88
  better): nothing label-free predicts which queries the LLM gets wrong. The
  error analysis finds the LLM's gain significant in all 13 large categories;
  images helping even where every candidate is imaged (+0.0065 [+0.0019,
  +0.0112]), so through the signal and not only its presence; and the LLM's
  damage concentrated on queries Stage 2 already ranked well. The §6 table
  has ten rows — Ablations 2, 5 and 6 each split into the comparisons they
  measured — and [`docs/RESULTS.md`](../../RESULTS.md) is the writeup, every
  four-decimal figure in it checked against a committed results file. **The
  project's best number is `stage2+llm` 0.8855 [0.8799, 0.8910] on the
  2,000-query sample, above the 0.8562 `ESCI_baseline`; on the full test split
  the coarse ranker's 0.8579 matches it.**

Start at a plan's `README.md`.

## Outside the plan series

`src/combine.py` (`python -m src.combine`) joins Plan 1's ESCI loader and Plan
2's ESCI-S corpus into `data/combined/{products,judgements}.parquet` — the
tables Plans 4 and 5 read. It is not a plan deliverable; it was built when the
two halves first existed and both were needed in one place. Its `description`
coalesce takes description coverage from 52.2% to 89.2%, which is the single
largest text gain the enrichment provides.

**The full-test-split LLM run, 2026-09-24.** After the series closed, the LLM
arm — measured by Plan 6 on a frozen 2,000-query sample — was run once over
all 8,956 test queries with its configuration unchanged: 6,958 paid calls in
2.3 hours, 3.16M prompt and 2.66M completion tokens
(`docs/results/llm-rerank-test-full.json`). It lands at **0.8855 [0.8827,
0.8883]**, the sample's point estimate with half its interval, and so **beats
the 0.8562 `ESCI_baseline` on the same queries**
(`docs/results/fine-rank-test-full.json`). Stage 4 on the whole split turns
the sample's ties into significant losses — every blend below the LLM alone
(`docs/results/blend-test-full.json`) — and Ablation 6 now shares the test
split with Ablations 3, 4, 5 and 7. It added `--sample 0` and
`--skip-cascade` to `src.fine_rank_report`, a `test` scope to
`src.stage_signals`, and `--usage-out` to `src.llm_rerank`.

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
