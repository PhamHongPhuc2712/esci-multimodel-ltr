# Plan 7 — Blend and Report

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-signals-and-the-blend.md); do not start a phase before its predecessor's gate passes.

**Goal:** Combine the stage scores into a final ordering and measure it against every single stage alone; then assemble the seven-row ablation table, the per-category and failure-case analysis, and the writeup that is this project's deliverable.

**Architecture:** One frame, then three readers. `src/stage_signals.py` persists every stage's opinion of every window document — Stage 2's score and rank, the cross-encoder's logit, the LLM's rank — because the blend, the error analysis and the writeup all need the same one and deriving it three times is how three sections quote three different numbers. `src/blend.py` is pure: it turns that frame into an ordering by fixed weights, by a learned combiner, or by a per-query selector, and it knows nothing about files. `src/blend_report.py` produces Stage 4's table, `src/error_analysis.py` the per-category and failure-case breakdowns, and `src/ablation_table.py` reads the committed JSON of Plans 4–7 and emits §6 as one table. `docs/RESULTS.md` is written from those files and quotes nothing that is not in them.

**Tech Stack:** Python ≥3.11, LightGBM 4.7 (`lambdarank`, the same `label_gain` guard as Plan 5), pandas, numpy, pytest. Plan 1's `src.metrics`, `src.floor`, `src.bootstrap`; Plan 5's `src.ranker` and `src.rank_report`; Plan 6's `src.rerank_window`, `src.stage2_scores`, `src.cross_encoder`, `src.llm_rerank`. **No new dependencies.** **No new paid API calls** — see below.

**Spec:** `PROJECT_SPEC.md` (§4.4 Stage 4, §5 Baselines to Beat, §6 Experiments/Ablations including the error analysis, §8.8 Build Order) and `CLAUDE.md` (Data invariants, Evaluation discipline). The plan argues from both; executors read both.

**Series:** Plan 7 of 7, the last — see [`../README.md`](../README.md). Gated on Plan 6, whose gate passed with `stage2+llm` at 0.8855 [0.8799, 0.8910] on a frozen 2,000-query test sample.

---

## The three phases

| Phase | File | Tasks | Delivers | Network / GPU? |
|---|---|---|---|---|
| 1 | [**The Signals and the Blend**](phase-1-the-signals-and-the-blend.md) | 1–2 | Persisted per-stage signals; the three blend strategies | GPU briefly for Task 1 |
| 2 | [**The Blend Report**](phase-2-the-blend-report.md) | 3 | Stage 4 against every stage alone, fold 0 then test | No |
| 3 | [**The Analysis and the Writeup**](phase-3-the-analysis-and-the-writeup.md) | 4–6 | Error analysis, the §6 table, `docs/RESULTS.md` | No |

Phase 1 comes first because all three Phase 3 readers consume the same signal
frame. If the error analysis rebuilt it, the writeup's "the LLM hurts 26% of
queries" and the blend's feature matrix could disagree and nothing would say so.

---

## Where these numbers came from

Every figure below was measured against the real fold-0 output on 2026-09-23,
after Plan 6 landed and before this plan was written. **Four of them say this
plan's central hypothesis is likely to fail**, which is why the plan is built
to measure that honestly rather than to chase a win.

- **The headroom is real and large.** Taking, per query, the best of the three
  stages' own orderings — an oracle no method can beat:

  | | NDCG on fold 0 |
  |---|---|
  | `stage2` alone | 0.8519 |
  | `stage2+ce` alone | 0.8587 |
  | **`stage2+llm` alone (best single)** | **0.8814** |
  | **oracle best-of-three** | **0.9076** |

  That is **+0.0262** over the best single arm. No arm dominates: the LLM is
  *strictly* best on **41.6%** of queries, the cross-encoder on **21.3%**,
  Stage 2 on **17.7%**, and **19.4%** are tied at the top (11.2% two-way,
  8.2% all three). So Stage 4 has something real to aim at. (This bullet
  first read "Stage 2 31.7%, cross-encoder 26.7%", which is `argmax` handing
  every tie to the first-listed arm — the same artefact Task 2 found in the
  selector's labels.)

- **The three arms genuinely disagree.** Mean Kendall τ over the window:
  `stage2`~`ce` **0.338**, `stage2`~`llm` **0.410**, `ce`~`llm` **0.374**.
  This is not three views of one ranking, which is what makes the oracle gap
  believable rather than an artifact.

- **Rank fusion cannot reach it — every weighting loses to the best single
  arm.** Measured with RRF (`k = 60`) over the three orderings:

  | fusion (s2 : ce : llm) | NDCG |
  |---|---|
  | 1 : 1 : 1 | 0.8721 |
  | 1 : 1 : 2 | 0.8770 |
  | 1 : 1 : 3 | 0.8786 |
  | **1 : 1 : 4 (best swept)** | **0.8800** |
  | 0 : 1 : 2 | 0.8793 |
  | `stage2+llm` alone | **0.8814** |

  As the LLM's weight rises the fusion converges on the LLM *from below*. This
  is the third time the project has met this shape: Plan 5's Ablation 7 found
  learned fusion losing to a single hand-tuned weight, and Plan 6 found the
  `ce`→`llm` cascade indistinguishable from the LLM alone.

- **A learned combiner over the scores also loses.** Piloted with LightGBM
  `lambdarank` on features `stage2_score`, `stage2_rank`, `ce_score`,
  `llm_rank`, `llm_rr`, cross-fitted within fold 0 so no query trains on
  itself: **0.8793**, against the LLM's 0.8814. Gain importance is dominated
  by `llm_rank` (19,132) over `ce_score` (3,420) and `stage2_score` (1,338) —
  the combiner mostly relearns the LLM's ordering and loses a little in the
  translation.

  **So Stage 4's two obvious forms are already measured as losses.** The plan
  builds both anyway, because §4.4 asks for them and a measured loss is the
  answer, and it adds a third arm — a **per-query selector** — which is the
  only form that targets the oracle's structure directly: the oracle wins by
  *choosing between* arms, not by averaging them, and no arm is strictly best
  more than 42% of the time.

- **The LLM arm hurts a quarter of queries, and that is the failure analysis
  §6 asks for.** Against Stage 2, per query: **better on 59.0%**, **worse on
  26.4%**, identical on 14.6%. Mean gain when it helps **+0.0772**; mean loss
  when it hurts **−0.0607**. So roughly 1,090 fold-0 queries are actively
  damaged by the best arm in the project — a real population to characterise,
  not a handful of anecdotes.

- **The category field supports the breakdown, at the top level only.**
  `s_category` is a path (mean depth 3.7, max 9), present on **86.7%** of
  fold-0 judgements, **49 distinct** top-level values. Only **13** have ≥100
  queries, and those cover **86.3%** of queries — so the per-category table is
  13 rows plus an "other" bucket, not 49.

- **Image coverage varies enough to stratify by.** Per query, the share of
  judged candidates carrying an image vector averages **0.78**; **1.7%** of
  queries have none and **25.4%** have all of them. That is the axis for §6's
  "where images help vs. hurt".

- **No new paid API calls are needed.** `data/llm-rerank.json` already holds
  the LLM's permutation for **4,129 of 4,130** fold-0 windows and **1,998 of
  2,000** test-sample windows. (This bullet first said "all 2,000"; checked
  against the cache it is not.) The three missing windows — fold-0 query
  55755 and test queries 49855 and 66028, two TV-series titles and a book —
  are ones whose LLM answer was malformed, which is never cached; fold 0's
  has failed twice. Plan 6 scored all three in Stage 2 order and counted
  them in `n_fallback`, so this plan carries them the same way, flagged
  `llm_fallback`, and its single-stage arms reproduce Plan 6's 0.8814 and
  0.8855 exactly. The marginal API cost is **zero**. Nothing in it should
  issue a call.

---

## What Plan 6 left for this plan to build

Plan 6 produced orderings and NDCG, but the cross-encoder's **scores** live
only inside `src/cross_encoder.rerank`, which deliberately returns orderings so
that a logit can never enter a run beside a Stage 2 score. A blend needs the
scores themselves. Task 1 therefore adds a public
`src.cross_encoder.window_scores` beside `rerank` — same model, same pairs,
returning `(query_id, product_id) -> float` — and `src/stage_signals.py` is the
only module that reads it. `rerank` keeps returning orderings, and
`src/rerank_window.spliced_run` still refuses anything that is not a
permutation, so Plan 6's Review Focus 1 protection is untouched.

---

## Corrected before execution

Reading this plan against the landed code on 2026-09-23, before any task ran,
found six things wrong. Each is fixed in the phase that owns it, and every
code block in all three phases now passes its own fast tests.

| # | What the plan said | What is true | Fixed in |
|---|---|---|---|
| 1 | The LLM cache holds all 2,000 test-sample windows; `stage_signals` should see no misses | 1 fold-0 and 2 test-sample windows are missing (malformed answers are never cached), so Task 1 would have stopped on both scopes | Task 1: flagged `llm_fallback`, capped by `check_fallback_share` |
| 2 | Fold-0 learned arms are cross-fitted | Only the combiner was; the selector was trained on all of fold 0 and scored on it | Task 2 `cross_fit_select`; Task 3 |
| 3 | Ablation 2 = full fusion vs. BM25 (+0.0661) | That credits images with the whole fusion gain, which Plan 4 stored `ablation_2_ladder` to avoid; the image step is +0.0048 | Task 5: rows 2a and 2b |
| 4 | `build_signals` guards Review Focus 2 | It silently fell back to the window order; only the CLI checked | Task 1: raises on an undeclared miss |
| 5 | Task 1's tests pass against Task 1's code | Its test labels had no `stage2_score`, so the "complete" frame carried NaN and `require_full_coverage` rejected it | Task 1 |
| 6 | The error analysis retrains Ablation 3's arms on folds 2/3/4 | Correct for fold 0, but a test-sample run would repeat Plan 5's two-fit mismatch (0.0020 low) | Task 4: fit set follows the scope |

Smaller: the signal frames are 41,255 and 19,974 rows, not "roughly 33,000
and 16,000"; the error analysis has 19 tests, not 20; and the scope-to-windows
code existed twice (Task 1 and Task 3's `rebuild_windows`) and is now
`src.stage_signals.scope_windows` alone.

---

## The fold protocol, and why Stage 4 trains on fold 0

Plans 5 and 6 used three roles: train 2/3/4, early-stop 1, report 0, test once.
Stage 4 cannot follow that, because **the LLM's orderings exist only for fold 0
and the test sample** — running it on folds 2/3/4 would cost ~4 h and real
money for a combiner with five features.

| role | queries | used for |
|---|---|---|
| **train** | fold 0 (4,130) | the combiner's gradient and the selector's |
| **report** | the frozen 2,000-query test sample | every headline number in Stage 4 |
| **fold-0 self-estimate** | fold 0, **cross-fitted** | the fold-0 column only, no query training on itself — for the combiner **and** the selector, over one shared query partition |

This is legitimate and is not tuning on test: fold 0 is a *validation* fold, the
test sample was frozen by Plan 6 before this plan existed, and the combiner sees
test only at prediction time. It does mean **fold 0 stops being a clean
reporting surface for Stage 4 arms** — hence the cross-fit, and hence every
fold-0 learned-arm number is labelled `cross-fit` in the output. The fixed
1:1:4 weights were chosen by a sweep on fold 0 itself, so the fixed arm's
fold-0 number is tuned in-sample; the JSON says so, and it lost anyway.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Never tune on test.** The blend's features, weights, round count and the selector's threshold are chosen on fold 0. The test sample is predicted once per arm.
- **No new API calls.** Every usable LLM ordering this plan needs is already in `data/llm-rerank.json`. A task that issues a call is a bug. `stage_signals` reads the cache, carries the three measured malformed-answer windows in Stage 2 order *with a flag* (as Plan 6 scored them), and *fails* when misses pass 0.5% of windows — a scope error, not a malfunction.
- **Every reported NDCG comes from `src.metrics.ndcg_per_query`**, paired with `src.floor.random_floor` and a bootstrap CI over queries from `src.bootstrap`. Never from a training log, and never from a number typed into the writeup by hand.
- **Arms are only ever compared over the same queries.** `src.fine_rank_report.check_same_queries` exists for this; Stage 4 reuses it rather than reimplementing it.
- **The random floor is computed, never quoted.** Measured 0.7467 on test; the published 0.7483 is wrong for this discount and the swapped-label figure 0.7141 is wronger. (`CLAUDE.md`.)
- **Scope travels with every number.** Ablations 1–2 are Recall@k on fold 0, 3–5 and 7 are NDCG on the full 8,956-query test split, 6 and Stage 4 are NDCG on the 2,000-query test sample. A table that omits this invites reading them as comparable.
- **A method that ties, or loses, is a reportable result.** Four separate measurements above say Stage 4 will not beat its best input. Plan 5's Ablation 7 was published as a loss; so is this, if that is what it measures. **Do not tune until an arm wins.**
- **Do not commit datasets.** `.gitignore` blocks `data/` and `models/`; only JSON under `docs/results/` and the Markdown writeup are committed.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)

---

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **A learned arm trained on the queries it reports.** The LLM's orderings exist only for fold 0 and the test sample, so fold 0 is the only training set available — and it is also the surface every earlier plan reported on. A combiner fitted on all of fold 0 and scored on all of fold 0 will look excellent and mean nothing, and five features over 4,130 queries is more than enough to memorise. The same holds for the selector, whose target — which arm scored best on each query — is the label in another form; this plan as first written cross-fitted the combiner and not the selector. `cross_fit_predict` and `cross_fit_select` must guarantee no query is ever scored or routed by a model that trained on it. — pinned in Task 2.
2. **Blending a stage that never ran on those queries.** `data/llm-rerank.json` covers fold 0 and the 2,000-query test sample — not the other 6,956 test queries. Asking for signals over the full test split would silently give 6,956 queries the Stage 2 order under the name `llm`, and the blend would report the resulting dilution as a Stage 4 effect. `stage_signals` must return its misses, refuse more than 0.5% of windows, and flag the three measured malformed-answer fallbacks rather than hide them; `build_signals` must raise for any window with neither an ordering nor a declared fallback. — pinned in Task 1.
3. **A §6 table that mixes scopes silently.** Recall@k on 4,130 fold-0 queries, NDCG on 8,956 test queries and NDCG on a 2,000-query sample are three different measurements. Printed as seven rows of one table with no scope column, they read as comparable, and the writeup's summary sentence inherits the error. — pinned in Task 5.
4. **Per-category NDCG over categories with too few queries to mean anything.** 49 top-level categories, only 13 with ≥100 queries. A category with 4 queries will show a ±0.15 swing from noise alone, and "images hurt in Musical Instruments" is exactly the sentence a reader will quote. Categories below the threshold must collapse into one bucket, and every row must carry its `n`. — pinned in Task 4.
5. **A writeup that quotes a number no file contains.** `CLAUDE.md` records that the published floor (0.7483), the published product count (1,215,851) and the swapped label distribution are all wrong for this project, and that third-party write-ups disagree with the measured data. The writeup is the one artifact a reader will trust without checking, so every figure in it must come from a committed `docs/results/*.json`, and a test must verify the headline ones do. — pinned in Task 6.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as
`src.<name>` and run as `python -m src.<name>`. This plan follows it.

| File | Responsibility | Written in |
|---|---|---|
| `src/stage_signals.py` | Every stage's opinion of every window document, persisted; `scope_windows`, the one place a scope becomes query ids, read by Phases 2 and 3 too; `python -m src.stage_signals` | Phase 1, Task 1 |
| `src/cross_encoder.py` | *Modified*: add public `window_scores` beside `rerank` | Phase 1, Task 1 |
| `src/blend.py` | Fixed-weight, learned combiner, selector, oracle. Pure | Phase 1, Task 2 |
| `src/blend_report.py` | Stage 4 against every stage alone; `python -m src.blend_report` | Phase 2, Task 3 |
| `src/error_analysis.py` | Per-category, image strata, LLM failure cases; `python -m src.error_analysis` | Phase 3, Task 4 |
| `src/ablation_table.py` | The §6 table from committed JSON; `python -m src.ablation_table` | Phase 3, Task 5 |
| `docs/results/blend.json`, `blend-test.json`, `error-analysis.json`, `ablation-table.json` | The committed record | Phases 2–3 |
| `docs/RESULTS.md` | The writeup | Phase 3, Task 6 |
| `tests/test_stage_signals.py`, `tests/test_blend.py`, `tests/test_blend_report.py`, `tests/test_error_analysis.py`, `tests/test_ablation_table.py`, `tests/test_results_doc.py` | One test module per source module | Every task |

`src/blend.py` is pure and separate from `src/blend_report.py` for the same
reason `src/rerank_window.py` was separate from the two Stage 3 arms: the
strategies are the thing worth testing exhaustively, and a module that also
reads parquet and trains LightGBM cannot be.

---

## What this plan reuses rather than rebuilds

| From | What | Why it matters |
|---|---|---|
| Plan 6 | `src.rerank_window.{Window, windows, spliced_run, stage2_run, DEFAULT_K}` | The window and the splice; Stage 4 orderings go through the same permutation guard |
| Plan 6 | `src.stage2_scores.load_stage2`, `src.llm_rerank.{RerankCache, window_key}` | The persisted Stage 2 ordering and the LLM's cached permutations |
| Plan 6 | `src.fine_rank_report.{check_same_queries, cost_usd, ArmCost}` | Comparability enforcement and the cost block, already tested |
| Plan 5 | `src.rank_report.{qrels_from_frame, evaluate_arm, compare, format_table}` | Arm bookkeeping and paired comparisons |
| Plan 5 | `src.ranker.LABEL_GAIN` | The ESCI gain mapping LightGBM must be told about explicitly |
| Plan 1 | `src.metrics.ndcg_per_query`, `src.floor.random_floor`, `src.bootstrap.*` | The only trustworthy NDCG, and intervals paired by query |
| Plan 2 | `data/combined/products.parquet` (`s_category`) | The category path the breakdown groups by |

---

## Plan Gate

This is the last plan; its gate is the project's.

**Passed 2026-09-23.** No Stage 4 strategy beats the LLM listwise arm it
contains: on the frozen 2,000-query test sample the fixed-weight, combiner and
selector arms tie with it (0.8843, 0.8849, 0.8853 against 0.8855), while the
per-query oracle reaches 0.9101. The project's best number stays
`stage2+llm` **0.8855 [0.8799, 0.8910]** on that sample, above the 0.8562
`ESCI_baseline`; on the full test split the coarse ranker's 0.8579
[0.8551, 0.8611] matches it.

- [x] `python -m pytest` passes with no failures and no new skips. — 705 passed (594 before this plan), 32 deselected, 0 skipped.
- [x] `python -m pytest -m data` passes, including the signal-frame coverage assertion and the writeup's figure check. — 31 passed in 2m49s (24 before this plan).
- [x] `data/features/stage-signals-{fold0,test-sample}.parquet` exist, every row carries a Stage 2 score, a cross-encoder score and an LLM rank, the only Stage 2 fallbacks are the 1 + 2 flagged windows Plan 6 counted, and ordering by each column reproduces Plan 6's `stage2`, `stage2+ce` and `stage2+llm` NDCG. — to float precision, all six arm × scope pairs.
- [x] Stage 4 is reported against **each single stage alone**, as §4.4 requires, with paired bootstrap CIs, on fold 0 (cross-fitted) and on the frozen test sample. — `blend.json`, `blend-test.json`; the test sample was run once, behind `--final`.
- [x] All three Stage 4 strategies are reported — fixed weight, learned combiner, per-query selector — **and the oracle ceiling beside them**, or the plan states which was dropped and why. — none dropped; the selector's routes are recorded too (93.2% to the LLM).
- [x] If no Stage 4 arm beats the best single stage, that is written down as the result, in the table and in the writeup's summary sentence. — `any_blend_beats_best_single: false` in both files; `docs/RESULTS.md` §5 opens with it.
- [x] `docs/results/ablation-table.json` carries all seven ablations — Ablation 2 as its two ladder rungs — each with its **scope, `n`, metric and interval**, and no two rows are presented as comparable when their scopes differ. — ten rows: Ablations 5 and 6 were split the same way during execution, each comparison found by arm name; the scope warning is part of the table.
- [x] The per-category breakdown reports only categories with ≥100 queries, collapses the rest, and carries `n` on every row. — 13 categories plus `(other)`, each with a paired interval on the LLM's gain; the 55 uncategorised queries are counted.
- [x] The LLM failure-case analysis characterises the ~26% of queries the arm damages, not a hand-picked sample. — all 1,089 (26.4%), contrasted on six attributes: the damage tracks a strong Stage 2 baseline (0.873 against 0.829), not a query type.
- [x] `docs/RESULTS.md` exists, states the project's best number with its scope and CI, compares it to the 0.8562 `ESCI_baseline` and the **measured** floor, and every headline figure in it is traceable to a committed `docs/results/*.json`. — `tests/test_results_doc.py` checks every four-decimal figure against the committed JSON or the published §5 allowlist.
- [x] `CLAUDE.md`'s Commands section lists the signal dump, the blend report, the error analysis and the ablation table.

Then: the series is complete. See [`../README.md`](../README.md).
