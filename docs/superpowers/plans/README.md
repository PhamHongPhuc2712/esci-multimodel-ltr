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
| 2 | **Enrichment Corpus** | Single-pass streaming ETL of ESCI-S (3.4 GB zstd) to Parquet; join coverage and missingness-bias report | Join coverage ≥ 90% of ESCI ASINs; missingness not confounded with label | §3.2, §8.2 |
| 3 | **Image Pipeline** | Resolution gate re-run, bulk async fetch, in-flight CLIP embedding, embedding store | `python -m src.esci_images` ≥ 90%; end-to-end image coverage reported against the ~75% figure, not 91.5% | §3.3, §8.3 |
| 4 | **Recall** | BM25 + dense + CLIP channels, RRF fusion, Recall@k harness, Stage 0 LLM query rewriting | Ablations 1 and 2 produce a table with bootstrap CIs | §4.0, §4.1, §8.4 |
| 5 | **Coarse Rank** | Query×product feature extraction, LightGBM `lambdarank`, pointwise A/B | Ablations 3, 4, 5, 7 produce a table with bootstrap CIs | §4.2, §8.5 |
| 6 | **Fine Rank** | Fine-tuned cross-encoder and LLM listwise reranker, head to head on Stage 2 top-K | Ablation 6 produces NDCG + latency + cost | §4.3, §8.6, §8.7 |
| 7 | **Blend and Report** | Stage 4 learned combiner, full ablation table, per-category error analysis, writeup | Beats the 0.8562 `ESCI_baseline` target, or reports honestly that it ties | §4.4, §5, §6, §8.8 |

A written plan gets its own directory and is split into phase files when a single file stops being readable end to end. Plan 1 is split that way: [`2026-09-19-evaluation-foundation/`](2026-09-19-evaluation-foundation/) holds an index plus three phases — the metric, the data, then statistics and the baseline — each gated on the one before it. Start at its `README.md`.

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
