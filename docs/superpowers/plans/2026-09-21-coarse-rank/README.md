# Plan 5 — Coarse Rank

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-features.md); do not start a phase before its predecessor's gate passes.

**Goal:** Turn each judged (query, product) pair into a feature vector, train LightGBM `lambdarank` on the real train split, and settle Ablations 3, 4, 5 and 7 with bootstrap CIs against the measured random floor.

**Architecture:** Four layers with a hard boundary between pure and impure. `src/features.py` is pure — one query and one product row in, a dict of named floats out, no I/O, no model, no index — so every feature definition is unit-tested in milliseconds. `src/pair_scores.py` owns the three retrieval signals that *do* need the index and the stores (BM25, dense text, CLIP image), because each is a different lookup against a different artefact and all three must return NaN rather than a score when they have nothing. `src/feature_matrix.py` joins those two into one row-per-judgement Parquet and owns `FEATURE_GROUPS`, the named column sets the ablation arms are built from. `src/ranker.py` owns LightGBM — the ESCI gain mapping, the group array, the fold protocol — and `src/rank_report.py` produces the table.

**Tech Stack:** Python ≥3.11, `lightgbm` 4.7 (new dependency), numpy, pandas, pyarrow, pytest. Plan 1's `src.metrics`, `src.floor` and `src.bootstrap`; Plan 4's `src.bm25_index`; Plan 3's `src.embedding_store` and `src.clip_encoder`; Plan 2's `data/combined/`. No new GPU work beyond encoding 29,844 queries twice.

**Spec:** `PROJECT_SPEC.md` (§2 the metric, §4.2 Stage 2 feature groups, §6 Ablations 3/4/5/7, §7.3 LTR methods, §8.5 Build Order) and `CLAUDE.md` (Evaluation discipline, Data invariants). The plan argues from both; executors read both.

**Series:** Plan 5 of 7 — see [`../README.md`](../README.md). Gated on Plan 4, whose gate passed with RRF fusion at R@100 0.5551, +0.0661 [+0.0598, +0.0727] over BM25.

---

## The three phases

Phase 1 delivers every feature before any model exists, for the same reason
Plan 1 delivered NDCG before any ranker and Plan 4 delivered Recall@k before
any channel: a feature that is quietly wrong produces a model that trains
cleanly, scores plausibly and means nothing.

| Phase | File | Tasks | Delivers | Network / GPU? |
|---|---|---|---|---|
| 1 | [**The Features**](phase-1-the-features.md) | 1–2 | Pure query×product features; the three retrieval pair scores | GPU for Task 2's encode |
| 2 | [**The Matrix and the Ranker**](phase-2-the-matrix-and-the-ranker.md) | 3–4 | `data/features/*.parquet`, `FEATURE_GROUPS`, LightGBM `lambdarank` + pointwise | No (CPU, minutes) |
| 3 | [**The Ablations**](phase-3-the-ablations.md) | 5–6 | Fixed-weight fusion control, Ablations 3/4/5/7, `docs/results/coarse-rank.json` | No |

---

## Where these numbers came from

Every figure below was measured against the real corpus on 2026-09-21, before
the plan was written. Six of them change the design.

- **LightGBM's own NDCG is not this project's NDCG, and the gap is bigger than
  every effect in the ablation table.** Two independent reasons compound.
  First, `lambdarank` defaults to a `2**rel - 1` gain; ESCI's mapping is
  1.0/0.1/0.01/0.0, which is *not* exponential. Second, LightGBM reports
  `ndcg@k` at a cutoff while this project reports full-list NDCG. Measured on
  a matrix of the real shape (385,829 rows, 20,888 groups, 40 features, 25%
  NaN), the **same model** reports `ndcg@10` **0.7931** with ESCI's gains and
  **0.8575** with LightGBM's default — a 6.4-point difference, against an
  Ablation 2 image effect of +0.0048. `label_gain=[0.0, 0.01, 0.1, 1.0]` is
  accepted, so the training objective can be made to match; the *reported*
  number must still come from `src.metrics`.
- **A BM25 score for a given pair is cheap, and crashes on an empty query.**
  `bm25s.BM25.get_scores` walks the CSC postings and returns a dense
  1,215,854-float array, from which the ~20 candidate rows are read. But
  `get_scores([])` raises `IndexError: list index out of range` — queries whose
  every term is a stopword or out of vocabulary. The pass must skip them with
  NaN rather than die partway through.

  > **Corrected during execution.** This first read "31.5 ms/query, so all
  > 29,844 queries take 15.7 minutes", from a pre-plan probe that timed the
  > first 200 calls against a cold page cache. The real pass is **27 s for all
  > 20,888 train queries** — 1.3 ms/query, ~35x faster — and the whole
  > `pair_scores` run for both splits is under two minutes, dominated by
  > loading the models rather than by BM25. Measured 2026-09-21. Nothing in the
  > design depends on the pass being slow, so only the prose changed.
  > **7 train queries tokenised to nothing**, consistent with the 1 found in
  > fold 0.
- **The `rerank` image store is fully redundant.** Measured: **0** of its
  361,875 URLs are absent from the catalogue store, and both give the *same*
  judged-product coverage — 373,639 products, **77.50%**. Plan 5 therefore
  reads `data/embeddings/catalogue/` only, and the 0.40 GB `rerank` store can
  be deleted. This corrects the note in `CLAUDE.md` that calls `rerank` "the
  sensible default for Plans 5–7": it is not wrong, it is merely a second copy.
- **Every judged product is reachable from every artefact except images.**
  482,105/482,105 judged products are in the BM25 index and 482,105/482,105
  have a dense vector. Only the image signal is partial, at 77.50% of products
  and **79.39% of judged pairs**.
- **One retrieval signal alone already ranks well, and a hand-tuned global
  weight does better.** Measured on validation fold 0 (4,130 queries, 82,751
  judgements), scoring the judged candidate list by a single feature:

  | single signal | NDCG | vs. the fold-0 floor 0.7440 | pair coverage |
  |---|---|---|---|
  | BM25 score | 0.8230 | +0.0790 | 0.9999 |
  | dense text similarity | 0.8285 | +0.0845 | 1.0000 |
  | CLIP image similarity | 0.7905 | +0.0465 | 0.7939 |
  | **fixed global weight, `0.7·text + 0.3·image`** | **0.8347** | **+0.0907** | — |

  That last row is **Ablation 7's control arm**, swept at 0.1 intervals: 0.8094
  at pure image, 0.8285 at pure text, peaking at 0.8347. It is the number the
  learned fusion has to beat, and it is already within 0.022 of the
  `ESCI_baseline` target of 0.8562. Note it is an **oracle** value — the sweep
  and the report were both on fold 0. Phase 3 tunes the weight on fold 1 and
  reports on fold 0, so the honest control lands at or just below 0.8347;
  beating the oracle is the conservative bar.
- **Training is cheap enough to run every arm.** 200 rounds of `lambdarank` on
  the real shape takes **6.8 s**; pointwise regression **4.3 s**. The whole
  ablation table is minutes of CPU, so no arm has to be skipped for cost.

Supporting measurements, over the 482,105 judged products:

| source | presence | source | presence |
|---|---|---|---|
| `product_title` | 1.0000 | `s_stars` | 0.8762 |
| `product_brand` | 0.9436 | `s_ratings` | 0.8792 |
| `product_color` | 0.6782 | `s_price` | 0.2667 |
| `description` (coalesced) | 0.8922 | `s_bsr_rank` | 0.4272 |
| `s_template` / `s_type` | 0.8959 | `s_attrs_json` | 0.5696 |
| `s_category` | 0.8586 | `s_info_json` | 0.4823 |
| `s_image_url` | 0.7753 | image vector | 0.7750 |

`s_category` is 4.40 levels deep on average (median 4, max 9); `s_attrs_json`
carries a median of 5 keys and at most 14; `s_info_json` a median of 13.
`s_template` has 105 distinct values and `s_type` 2 — both usable as LightGBM
categoricals. `product_brand` has **118,123** distinct values and
`product_color` **80,463**, so neither can be a categorical feature; they enter
as *match* features against the query, exactly as §4.2 says.

Frozen fold sizes, from `splits/val_folds.csv`:

| fold | queries | judgements |
|---|---|---|
| 0 | 4,130 | 82,751 |
| 1 | 4,239 | 86,417 |
| 2 | 4,213 | 84,619 |
| 3 | 4,137 | 82,384 |
| 4 | 4,169 | 83,482 |

---

## The fold protocol, and why it has three roles

Model selection needs a surface to select on, early stopping needs a surface to
stop on, and the ablation table needs a surface to report on. Using one fold
for all three is how a plan ends up reporting the number it optimised.

| role | folds | used for |
|---|---|---|
| **train** | 2, 3, 4 | 12,519 queries / 250,485 rows of gradient |
| **early stop** | 1 | LightGBM's `valid_sets`, and only that |
| **report** | 0 | every number in the ablation table |
| **test** | — | measured **once**, at the very end of Phase 3, configuration frozen |

The final test run retrains on all five train folds at the round count learned
on fold 1, then predicts test. That is the only time test is touched, and
`CLAUDE.md`'s evaluation discipline forbids going back to change anything
afterwards.

---

## The four ablations this plan owns

| # | Ablation | Arms | Baseline to beat |
|---|---|---|---|
| **3** | Text-only vs. text+image | `text+retrieval` vs. `text+retrieval+image` | the text-only arm |
| **4** | With vs. without behavioural | **three**: text only; text + values + indicators; **indicators only** | honest contribution = arm 2 − arm 3 |
| **5** | Pointwise vs. `lambdarank` | `lambdarank`, pointwise regression on gain, pointwise 4-class scored by expected gain | the `lambdarank` arm |
| **7** | Learned fusion vs. fixed global weight | LambdaMART over `{dense_sim, clip_image_sim}` vs. `w·text + (1−w)·image` | **0.8347** at `w_text = 0.7` |

**Ablation 4 has three arms because two would be dishonest.** `CLAUDE.md`
measured that ESCI-S *presence patterns alone* rank at +0.0084 over the floor,
against +0.0052 for ESCI's own presence flags — and "our 2022 scrape failed on
this page" is not available at serving time. So the indicators-only arm is
subtracted, not assumed away. The effects are super-additive (+0.0166 for both
together, against +0.0052 + +0.0084), so the bias cannot be bounded by adding
the parts, which is precisely why arm 3 is measured rather than reasoned about.

**`rank_xendcg` is deliberately not an arm.** `PROJECT_SPEC.md` §7.3 lists it
as a cheap A/B against `lambdarank`, but §6's Ablation 5 is specifically
*pointwise versus listwise*, and a second listwise objective answers a
different question. It is a one-line addition to `src.ranker.OBJECTIVES` if
Plan 7 wants it.

**Ablation 7's learned arm uses only the two signals the fixed weight uses.**
Handing LambdaMART forty features and calling the difference "learned fusion"
would measure the other thirty-eight. The full model is reported separately, as
context, not as the ablation.

### What Ablation 4 does *not* isolate

The BM25 index and the dense store were both built over the **coalesced**
`description` column, which `CLAUDE.md` records as 52.2% ESCI plus 37.0%
ESCI-S. So `bm25_score` and `dense_sim` carry ESCI-S *text* in every arm,
including the "text only" one. Ablation 4 therefore isolates the **behavioural,
categorical and attribute** contribution — which is what `PROJECT_SPEC.md` §6
asks of it ("Behavioural contribution") — and not the whole ESCI-S
contribution. Say so in the writeup rather than letting the arm name imply more
than it measures.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Never tune on test.** Every threshold, hyperparameter, feature set and round count is chosen on folds 1–4 and reported on fold 0. Test is predicted once, at the end of Phase 3.
- **Features are query×product or product-attribute, never product-alone-from-labels.** 34,756 products appear in both train and test (re-measured 2026-09-21); the split is query-level, so any statistic computed from labels per product leaks train labels into test. `src/feature_matrix.py` asserts the provenance of every column.
- **Every reported NDCG comes from `src.metrics.ndcg_per_query`, never from LightGBM.** LightGBM's `ndcg@k` differs by up to 6.4 points on the same model — cutoff and gain mapping both differ. LightGBM's metric is for early stopping and nothing else.
- **Always report against the computed random floor, with bootstrap CIs over queries.** Use `src.floor.random_floor` and `src.bootstrap.paired_delta_ci`. A method that ties is a legitimate, reportable result — do not tune until an arm wins.
- **Absence is NaN, never zero.** 20.6% of judged pairs have no image vector; a 0.0 cosine is a *score* meaning "mildly irrelevant", while NaN is an absence LightGBM splits on natively. Never `np.nan_to_num` a similarity.
- **ESCI-S presence is an artefact; ESCI's own presence is not.** Carry per-field missingness indicators, keep them in their own feature group, and subtract the indicators-only arm from any behavioural result.
- **Run GPU-capable work on the GPU.** RTX 3080 Laptop (16 GB, sm_86) via WSL2. Pass `device` explicitly and fail loudly if `cuda` is requested and unavailable. Task 2 encodes 29,844 queries with SBERT and with CLIP text.
- **Build the pair scores once and cache them.** Nothing in Phases 2 or 3 may recompute them; the matrix reads `data/features/pair-scores-{split}.parquet`.
- **RAM is not free.** Measured 2026-09-21: 23 GB total with ~5 GB available on a working machine. Read Parquet with an explicit `columns=` list; the combined product table is 1.84 GB on disk and 27 columns wide.
- **Do not commit datasets.** `.gitignore` blocks `data/`. The feature matrices live in `data/features/`; only the JSON under `docs/results/` is committed.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)

---

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **LightGBM's reported NDCG mistaken for the project's NDCG.** The default `label_gain` is `2**rel - 1` and `eval_at` is a cutoff, so the same booster reports 0.7931 under ESCI's gains and 0.8575 under the default — a 6.4-point gap, larger than any effect being measured. A plausible-looking number lifted from a training log would silently inflate the whole ablation table. — pinned in Task 4.
2. **The `group` array disagreeing with the row order.** LightGBM's `group` is a list of *sizes* over consecutive rows, not query ids: it assumes the matrix is already sorted so each query's rows are contiguous. Feed it a frame sorted any other way and it trains on groups that straddle queries — no exception, no warning, just a model that learned to rank across query boundaries. — pinned in Task 3.
3. **A query that tokenises to nothing.** `bm25s.get_scores([])` raises `IndexError` from `query_tokens_single[0]`; one of fold 0's 4,130 queries is such a query after stopword removal. It must yield NaN for that query's pairs and let the other 29,843 finish. — pinned in Task 2.
4. **Ablation 4 reported as two arms.** ESCI-S presence alone ranks at +0.0084 and is unavailable at serving time; reporting arm 2 − arm 1 credits the behavioural features with an artefact worth roughly twice Plan 4's entire image contribution. The report must refuse to emit Ablation 4 without the indicators-only arm. — pinned in Task 6.
5. **A missing image scored as zero similarity.** 20.6% of judged pairs have no image vector. `EmbeddingStore.lookup` already returns NaN plus a presence mask; anything that fills those with 0.0 turns "we never scraped this" into "this image is orthogonal to the query" and ranks the product below a genuinely bad match. — pinned in Task 2.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as
`src.<name>` and run as `python -m src.<name>`. This plan follows it.

| File | Responsibility | Written in |
|---|---|---|
| `src/features.py` | Pure query×product feature functions. No I/O, no model, no index | Phase 1, Task 1 |
| `src/pair_scores.py` | BM25, dense and CLIP scores for judged pairs; `python -m src.pair_scores` | Phase 1, Task 2 |
| `src/feature_matrix.py` | Assemble, validate and persist the matrix; owns `FEATURE_GROUPS`; `python -m src.feature_matrix` | Phase 2, Task 3 |
| `src/ranker.py` | LightGBM `lambdarank` and pointwise training, the ESCI gain mapping, the fold protocol | Phase 2, Task 4 |
| `src/fixed_weight.py` | Ablation 7's control arm: sweep a single global text/image weight | Phase 3, Task 5 |
| `src/rank_report.py` | Ablations 3, 4, 5, 7 with bootstrap CIs; `python -m src.rank_report` | Phase 3, Task 6 |
| `docs/results/coarse-rank.json` | The committed record of every arm | Phase 3, Task 6 |
| `tests/test_features.py`, `tests/test_pair_scores.py`, `tests/test_feature_matrix.py`, `tests/test_ranker.py`, `tests/test_fixed_weight.py`, `tests/test_rank_report.py` | One test module per source module | Every task |

`src/features.py` and `src/pair_scores.py` are separate because one is pure and
the other needs a 1.1 GB index, a 0.95 GB store, a 0.98 GB store and a GPU.
Merging them would make every feature test wait on the index.

---

## What this plan reuses rather than rebuilds

| From | What | Why it matters |
|---|---|---|
| Plan 1 | `src.metrics.ndcg_per_query`, `src.floor.random_floor` | The only trustworthy NDCG in the project, cross-checked against `pytrec_eval` |
| Plan 1 | `src.bootstrap.bootstrap_ci`, `.paired_delta_ci` | Intervals paired by query id, which is what makes two arms comparable |
| Plan 1 | `src.splits.load_folds`, `splits/val_folds.csv` | The folds are frozen with a checksum; selection has somewhere legitimate to happen |
| Plan 1 | `src.labels.ESCI_GAINS` | The single definition of the gain mapping, and the source of `label_gain` |
| Plan 2 | `data/combined/products.parquet`, `judgements.parquet` | 1,215,854 products with no labels on them; 601,354 judgements carrying `split` and `fold` |
| Plan 3 | `src.embedding_store.open_store`, `.lookup` | float16 vectors with a presence mask — the mask is Review Focus 5's whole defence |
| Plan 3 | `src.clip_encoder.load_encoder`, `.encode_texts` | The image feature's query side, in the same 512-d space as the stored vectors |
| Plan 4 | `src.bm25_index.open_channel`, `.tokenize_texts` | The mmapped index and, crucially, the *same* tokeniser the corpus was built with |

---

## Plan Gate

**Passed 2026-09-21.** Headline **0.8579 [0.8551, 0.8611]** on test against a
measured floor of 0.7468 and the `ESCI_baseline` target of 0.8562.

- [x] `python -m pytest` passes with no failures and no new skips. — 485 passed.
- [x] `python -m pytest -m data` passes, including the feature-matrix build on the real corpus. — 17 passed.
- [x] `python -m src.feature_matrix --split train` and `--split test` assert 419,653 and 181,701 rows and write `data/features/{train,test}.parquet`.
- [x] Ablation 3 (text vs. text+image) is reported on fold 0 with a paired bootstrap CI. — **+0.0060 [+0.0039, +0.0081]** on fold 0, **+0.0075 [+0.0059, +0.0090]** on test. Significant: images help.
- [x] Ablation 4 is reported with **three** arms, and the honest contribution arm 2 − arm 3 is stated alongside the naive arm 2 − arm 1. — reported with **four**; see the correction in [Phase 3](phase-3-the-ablations.md#review-focus-4-ablation-4-arithmetic). The quotable figure is `values_over_indicators` = **+0.0036 [+0.0023, +0.0051]**.
- [x] Ablation 5 (`lambdarank` vs. two pointwise objectives) is reported with paired bootstrap CIs. — `lambdarank` beats pointwise regression by **+0.0096** and the 4-class pointwise arm by **+0.0085** on test, both significant. The listwise loss earns its keep.
- [x] Ablation 7 beats, or honestly ties, the measured fixed-weight control of **0.8347** at `w_text = 0.7`. — **it does not beat it.** Learned fusion over the same two signals scores 0.8321 against the control's 0.8345 on fold 0 (**−0.0023 [−0.0040, −0.0008]**, a significant *loss*) and 0.8377 against 0.8388 on test (**−0.0011 [−0.0023, +0.0000]**, a tie). Reported as measured.
- [x] Every reported NDCG carries the fold-0 random floor beside it, and no number in the table came from a LightGBM training log.
- [x] `docs/results/coarse-rank.json` records every arm, its interval, the floor, the feature groups it used and the round count.
- [x] Exactly one test-split number exists, produced after the configuration was frozen, reported against the 0.8562 `ESCI_baseline` target. — 0.8579, whose 95% CI contains 0.8562, so this **matches** the target rather than significantly beating it.
- [x] `CLAUDE.md`'s Commands section lists the pair-score pass, the matrix build and the rank report.

Then: Plan 6 — Fine Rank. See [`../README.md`](../README.md) for the series.
