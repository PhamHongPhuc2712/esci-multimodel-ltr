# Plan 6 — Fine Rank

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-window.md); do not start a phase before its predecessor's gate passes.

**Goal:** Re-rank the top-K of Plan 5's ordering with a fine-tuned cross-encoder and with an LLM listwise reranker, and settle Ablation 6 — coarse-only vs. +cross-encoder vs. +LLM — on NDCG, latency and cost.

**Architecture:** A window, then two rerankers behind one interface. `src/stage2_scores.py` freezes Plan 5's configuration and persists its scores, because Plan 5 reported numbers but never wrote an ordering to disk and both Stage 3 arms need the same one. `src/rerank_window.py` is pure: it carves the top-K out of a Stage 2 ordering and splices a new ordering of that window back above an untouched tail, which is the whole of the "re-rank the top-K" contract and the only place scale mismatches can corrupt a run. `src/cross_encoder.py` and `src/llm_rerank.py` are the two arms — one a GPU fine-tune, one a cached API call — and neither knows about the window. `src/fine_rank_report.py` produces Ablation 6.

**Tech Stack:** Python ≥3.11, `sentence-transformers` 6.1 (`CrossEncoderTrainer`, `LambdaLoss` — already installed from Plan 1's `baselines` extra), `torch` on the RTX 3080, `openai` (already installed from Plan 4's `retrieval` extra), numpy, pandas, pytest. Plan 1's `src.metrics`, `src.floor`, `src.bootstrap`; Plan 5's `src.ranker` and `data/features/`. **Two new dependencies**, both required by `CrossEncoderTrainer` and neither shipped with `sentence-transformers`: `datasets` and `accelerate>=1.1.0`.

**Spec:** `PROJECT_SPEC.md` (§4.3 Stage 3, §5 baselines, §6 Ablation 6, §7.4 fine-rank methods, §8.6–§8.7 Build Order, §9 Compute) and `CLAUDE.md` (Evaluation discipline). The plan argues from both; executors read both.

**Series:** Plan 6 of 7 — see [`../README.md`](../README.md). Gated on Plan 5, whose gate passed with a test headline of NDCG 0.8579 [0.8551, 0.8611] against the 0.8562 `ESCI_baseline` target.

---

## The three phases

| Phase | File | Tasks | Delivers | Network / GPU? |
|---|---|---|---|---|
| 1 | [**The Window**](phase-1-the-window.md) | 1–2 | Persisted Stage 2 scores; the top-K splice | GPU briefly for Task 1 |
| 2 | [**The Cross-Encoder**](phase-2-the-cross-encoder.md) | 3–4 | Fine-tune on folds 2/3/4, predict, latency | GPU, ~29 min/epoch |
| 3 | [**The LLM and the Ablation**](phase-3-the-llm-and-the-ablation.md) | 5–6 | Cached listwise reranker, Ablation 6 | API key; ~1.5 h for fold 0 |

Phase 1 comes first because both arms re-rank *the same window of the same
ordering*. If each arm carved its own, Ablation 6 would be comparing two
different candidate sets and calling the difference a reranker effect.

---

## Where these numbers came from

Every figure below was measured against the real data on 2026-09-21, after
Plan 5 landed and before this plan was written. Five of them change the design.

- **There is real headroom, and it is concentrated in the top few ranks.**
  Perfectly reordering Plan 5's own output on validation fold 0, leaving the
  tail untouched:

  | oracle reorder of | NDCG | headroom over Stage 2's 0.8519 |
  |---|---|---|
  | top-5 | 0.9079 | +0.0560 |
  | **top-10** | **0.9533** | **+0.1014** |
  | top-20 | 0.9897 | +0.1378 |
  | the whole list | 1.0000 | +0.1481 |

  Only **3.3%** of fold-0 queries are already perfect, so Stage 3 has
  something to do on 96.7% of them. `K = 10` is the plan's default: it holds
  two thirds of the total headroom and costs the LLM arm a quarter of what the
  full list does.

- **Zero-shot cross-encoders are *worse* than Plan 5's coarse ranker.** On an
  identical 400-query fold-0 sample where **Stage 2 scores 0.8577**:

  | zero-shot model | input | NDCG | GPU latency |
  |---|---|---|---|
  | `ms-marco-MiniLM-L6-v2` | title | 0.8286 | 7 ms/query |
  | `ms-marco-MiniLM-L6-v2` | title + description | 0.8396 | 40 ms/query |
  | `bge-reranker-base` | title | 0.8366 | 33 ms/query |
  | `bge-reranker-base` | title + description | 0.8445 | 205 ms/query |

  All four lose, by 0.013 to 0.029. **Fine-tuning is not an optimisation here,
  it is the whole arm** — which is consistent with `PROJECT_SPEC.md` §5, where
  the 0.8562 `ESCI_baseline` is a cross-encoder *fine-tuned on SQD train*.
  Adding the description is worth +0.008 to +0.011 and costs 5–6x the latency.

- **Fine-tuning fits comfortably on the 3080.** Measured over the 12,519
  train-fold queries / 250,485 pairs, `ms-marco-MiniLM-L6-v2`:

  | recipe | batch | throughput | peak VRAM | one epoch |
  |---|---|---|---|---|
  | `BinaryCrossEntropyLoss`, pairs | 128 @ len 192 | 393 pairs/s | 4.12 GB | **10.6 min** |
  | `BinaryCrossEntropyLoss`, pairs | 64 @ len 256 | 266 pairs/s | 2.87 GB | 15.7 min |
  | **`LambdaLoss`, listwise groups** | **8** | **7.2 queries/s** | **6.22 GB** | **29.0 min** |
  | `LambdaLoss`, listwise groups | 4 | 3.9 queries/s | 3.81 GB | 54.1 min |

  Of 16 GB, so there is room for a larger backbone if the small one
  disappoints. `LambdaLoss` is the default — it optimises a listwise objective
  against the same graded labels the metric uses, and Plan 5's Ablation 5
  measured listwise beating pointwise by +0.0085 to +0.0096 on this data.
  Verified end to end: `LambdaLoss` trains on a `(query, docs, labels)` dataset
  and `BinaryCrossEntropyLoss` on a `(query, doc, label)` one.

- **The LLM emits well-formed permutations, and reasoning tokens dominate the
  bill.** `gpt-5.6-luna`, the model Plan 4 used for Stage 0:

  | window | prompt tok | completion tok | of which reasoning | latency | well-formed |
  |---|---|---|---|---|---|
  | whole list (40 cands) | 1,015 | 1,716 | 1,536 (**89%**) | 17.3 s | 1/1 |
  | **top-10** | **386** | **387** | **339 (88%)** | 4.3 s, **1.3 s at concurrency 4** | **8/8** |

  Restricting to a top-10 window cuts completion tokens **4.4x**. At
  concurrency 4 that is **~1.5 h for all 4,130 fold-0 queries** (1.59M prompt +
  1.60M completion) and ~3.3 h for all 8,956 test queries. This is the same
  lesson Plan 4 recorded for Stage 0 — on a reasoning model the answer is a
  rounding error and the thinking is the cost — so the budget is sized on
  completion tokens, not on the 40-token permutation it returns.

- **No sliding window is needed, and that is a measurement, not a shortcut.**
  `PROJECT_SPEC.md` §7.4 records RankGPT's sliding window and the practical
  chunking fix for listwise context limits — "50 products ≈ 28k tokens → 5
  chunks of 10, take top 2 each, merge". At `K = 10` the whole window is a
  single 386-token prompt, well inside any context limit, so there is exactly
  one chunk and no cross-window blindness to mitigate. If a later plan raises
  `K` past ~30 the sliding window comes back; at 10 it would be machinery
  around a problem that does not exist.
- **The two arms differ in latency by three orders of magnitude.** 7–205
  ms/query for the cross-encoder on GPU against 1.3–17.3 s/query for the LLM.
  That ratio, not the NDCG delta, is the practical content of Ablation 6's
  latency column.

Fold-0 candidate counts, for sizing the window: median 16, mean 20.0, max 69.

---

## What Plan 5 left for this plan to build

Plan 5 reported NDCG but **never persisted an ordering**. `src/rank_report.py`
trains, scores, prints and writes JSON summaries; the per-pair predictions live
only inside that process. Both Stage 3 arms need the same Stage 2 ordering, and
re-deriving it independently in two places is how two arms end up re-ranking
two different windows.

Task 1 therefore adds `src/stage2_scores.py`, which trains the **frozen** Plan 5
configuration — all 47 features, `lambdarank`, `label_gain` from `src.labels`,
train on folds 2/3/4, early-stop on fold 1 — and writes
`data/features/stage2-{split}.parquet`. It is a new module rather than a flag on
`rank_report` because `rank_report` trains eleven arms and only one of them is
the thing Stage 3 sits on top of.

---

## The fold protocol, unchanged

Plan 5's three roles carry over exactly, and the cross-encoder fine-tune obeys
them too:

| role | folds | used for |
|---|---|---|
| **train** | 2, 3, 4 | LightGBM gradient, and the cross-encoder fine-tune |
| **early stop** | 1 | `valid_sets`, the cross-encoder's eval set, `K`, the prompt |
| **report** | 0 | every number in Ablation 6's fold-0 table |
| **test** | — | measured **once**, at the end of Phase 3 |

**The cross-encoder must not see fold 0 or the test split during training.** It
is a 22M-parameter model with 250,485 training pairs; training on all of train
would put the reporting fold in its weights, and the resulting NDCG would be a
memorisation score.

---

## Ablation 6, and how cost is reported

`PROJECT_SPEC.md` §6 asks for **NDCG, latency and cost** across three arms:

| arm | what it is |
|---|---|
| `stage2` | Plan 5's LambdaMART alone — 0.8519 fold 0, 0.8579 test |
| `stage2+ce` | the fine-tuned cross-encoder re-ranking the top-K |
| `stage2+llm` | the LLM listwise reranker re-ranking the top-K |

Two reference arms come along for context: `stage2+ce_zeroshot` (the measured
0.8286–0.8445, which shows what the fine-tune bought) and `stage2+ce+llm` (the
cascade, which answers whether the two arms are complementary or redundant).

**Latency is reported twice, and cost is reported in tokens.** Plan 4 measured
its BM25 index at "28 ms/query amortised over a batch, against 37" single —
and single-query latency at 64–88 ms. A batch-amortised figure is throughput,
not latency, and quoting one as the other flatters whichever arm batches
better. Ablation 6 reports **both**, labelled.

For cost, the report emits **prompt tokens, completion tokens and wall-clock
seconds per query**. It does *not* invent a dollar figure: prices change and
are not measurable from here. The CLI takes optional `--price-per-mtok-in` and
`--price-per-mtok-out` flags and fills a `cost_usd` field only when both are
given; otherwise that field is `null` and the writeup multiplies by whatever
rate applies on the day.

**The test-split LLM arm runs on a frozen sample.** All 8,956 test queries
through the LLM is ~3.3 h and real money. Ablation 6's test table therefore
uses a frozen random sample of **2,000 test queries** (seed 0), and **every
arm is scored on that same sample** so the comparison stays paired. The
full-test Stage 2 headline remains Plan 5's 8,956-query 0.8579; the sample is
for the three-way comparison only, and its `n` is recorded beside every number.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Never tune on test.** `K`, the loss, the backbone, the prompt, the epoch count and the concurrency are all chosen on folds 1–4. Test is predicted once, behind an explicit `--final` flag.
- **The cross-encoder trains on folds 2/3/4 only.** Fold 0 is the reporting surface and fold 1 is the eval set; a model that trained on either is reporting memorisation.
- **Every reported NDCG comes from `src.metrics.ndcg_per_query`, paired with the computed floor** (`src.floor.random_floor`) and a bootstrap CI over queries (`src.bootstrap`). Never from a training log.
- **The re-ranked window must sit strictly above the untouched tail.** Stage 3 scores are on a different scale from Stage 2's; writing them into a run naively lets a tail item outrank a reranked one. `src/rerank_window.py` owns this and nothing else may write a spliced run.
- **A malformed LLM ranking falls back to the Stage 2 order for that query, and the fallback is counted.** An ill-formed permutation that is silently accepted drops or duplicates candidates and quietly changes the denominator.
- **Cache every LLM call, keyed on the query *and* the exact candidate window.** Plan 4's rewrite cache keyed on the query alone, which is correct for a rewrite and wrong here: the answer depends on the list and its order.
- **Run GPU-capable work on the GPU.** RTX 3080 Laptop (16 GB, sm_86) via WSL2. Pass `device` explicitly and fail loudly if `cuda` is requested and unavailable.
- **Report latency as both single-query and batch-amortised**, labelled. They differ by 3–5x and quoting one as the other flatters whichever arm batches better.
- **Do not invent prices.** Report tokens and seconds; fill `cost_usd` only from flags the operator passed.
- **A method that ties, or loses, is a reportable result.** Plan 5's Ablation 7 lost to its control and was reported as a loss. Do not tune until an arm wins.
- **Do not commit datasets.** `.gitignore` blocks `data/`. Models go to `models/` (also ignored); only JSON under `docs/results/` is committed.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)

---

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **The untouched tail outranking the re-ranked window.** Stage 3 scores live on their own scale — a cross-encoder logit can be −8, a Stage 2 score +3. Writing Stage 3 scores into the run for the window and leaving Stage 2 scores on the tail interleaves the two orderings, so a product the reranker demoted can still land above one it never saw. Nothing raises; NDCG simply moves for a reason unrelated to the reranker. — pinned in Task 2.
2. **An ill-formed or lossy LLM permutation.** `PROJECT_SPEC.md` §7.4 records that open LLMs "often emit ill-formed listwise output". Measured 8/8 valid here, but a dropped index silently loses a candidate and a duplicated one silently promotes it twice. The parser must accept only an exact permutation of `1..n` and fall back to Stage 2's order otherwise, counting the fallbacks. — pinned in Task 5.
3. **The cross-encoder trained on the reporting fold.** 22M parameters over 250,485 pairs will memorise. The default `data/features/train.parquet` contains folds 0–4, so a fine-tune that reads it without filtering trains on the surface it is about to be scored on, and returns an impressively wrong number. — pinned in Task 3.
4. **A cache key that ignores the candidate list.** Plan 4's `RewriteCache` keys on the raw query, which is right for a rewrite. A listwise ranking depends on the query, the candidate set *and* its input order; reusing that key shape returns a permutation computed for a different list, and the indices still parse. — pinned in Task 5.
5. **Batch-amortised throughput reported as per-query latency.** Ablation 6's deliverable is explicitly "NDCG, latency and cost". The cross-encoder batches 256 pairs at a time and the LLM does not batch at all, so a single amortised number would flatter the cross-encoder by 3–5x on top of the real 20–600x gap. Both figures must be measured and labelled. — pinned in Task 4.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as
`src.<name>` and run as `python -m src.<name>`. This plan follows it.

| File | Responsibility | Written in |
|---|---|---|
| `src/stage2_scores.py` | Train the frozen Plan 5 config, persist its ordering; `python -m src.stage2_scores` | Phase 1, Task 1 |
| `src/rerank_window.py` | Carve the top-K window, splice a new ordering above an untouched tail. Pure | Phase 1, Task 2 |
| `src/cross_encoder.py` | Fine-tune, predict, measure latency; `python -m src.cross_encoder` | Phase 2, Tasks 3–4 |
| `src/llm_rerank.py` | Listwise prompting, permutation parsing, window-keyed cache; `python -m src.llm_rerank` | Phase 3, Task 5 |
| `src/fine_rank_report.py` | Ablation 6 with NDCG, latency and cost; `python -m src.fine_rank_report` | Phase 3, Task 6 |
| `docs/results/fine-rank.json` | The committed record of every arm | Phase 3, Task 6 |
| `tests/test_stage2_scores.py`, `tests/test_rerank_window.py`, `tests/test_cross_encoder.py`, `tests/test_llm_rerank.py`, `tests/test_fine_rank_report.py` | One test module per source module | Every task |

`src/rerank_window.py` is separate from both arms because it is the one piece
of logic they share, and the one place a scale mismatch can corrupt a run.
Writing the splice twice is how the two arms end up incomparable.

---

## What this plan reuses rather than rebuilds

| From | What | Why it matters |
|---|---|---|
| Plan 5 | `src.ranker.{train_ranker, predict, TRAIN_FOLDS, EARLY_STOP_FOLD, REPORT_FOLD, folds}` | The frozen Stage 2 configuration, including the `label_gain` guard |
| Plan 5 | `src.feature_matrix.ALL_FEATURES`, `data/features/{train,test}.parquet` | 47 features already joined, 601,354 rows, sorted for ranking |
| Plan 5 | `src.rank_report.{qrels_from_frame, evaluate_arm, compare, format_table}` | Arm bookkeeping and paired comparisons, already tested |
| Plan 1 | `src.metrics.ndcg_per_query`, `src.floor.random_floor`, `src.bootstrap.*` | The only trustworthy NDCG, and intervals paired by query |
| Plan 4 | `src.query_rewrite.RewriteCache` (as a *pattern*, not an import) | On-disk JSON cache with validate-then-fall-back; the key shape changes |
| Plan 2 | `data/combined/products.parquet` | Titles and the coalesced `description` for the reranker's document side |

---

## Plan Gate

Plan 7 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the Stage 2 score round trip and the window splice on the real ordering.
- [ ] `data/features/stage2-{train,test}.parquet` exist and reproduce Plan 5's fold-0 NDCG of 0.8519 to within 0.0005.
- [ ] The fine-tuned cross-encoder **beats its own zero-shot arm** (0.8286–0.8445 measured), or the plan reports honestly that fine-tuning did not help.
- [ ] Ablation 6 reports all three arms with NDCG + bootstrap CI, **single-query and batch-amortised latency**, and prompt/completion tokens per query.
- [ ] Every arm's NDCG is scored on the same query set as the arms it is compared against, and the `n` is recorded beside it.
- [ ] The LLM arm records how many queries fell back to the Stage 2 order for a malformed ranking.
- [ ] `docs/results/fine-rank.json` records every arm, its interval, the floor, `K`, the backbone, the loss and the epoch count.
- [ ] Exactly one test-split run exists, on the frozen 2,000-query sample, produced after the configuration was frozen.
- [ ] `CLAUDE.md`'s Commands section lists the Stage 2 score dump, the fine-tune and the fine-rank report.

Then: Plan 7 — Blend and Report. See [`../README.md`](../README.md) for the series.
