# Plan 8 — Distil the LLM into the Cross-Encoder

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-free-arms.md); do not start a phase before its predecessor's gate passes. **Phase 2 spends money and needs the user's go-ahead in the session that runs it.**

**Goal:** Find out whether the LLM listwise arm's gain can be moved into a cross-encoder that answers in about 25 ms. On the full test split that gain is +0.0276 NDCG, at 4.7 s a query. The plan spends the ~4 h of paid teacher calls only if free measurements first say they could pay.

**Architecture:** Four steps, each feeding the next.
1. The student's training windows are carved by an **out-of-fold** Stage 2. `src/stage2_scores.py` gains `out_of_fold_scores`, which scores each of folds 2/3/4 with a model fitted on the other two, because the persisted ordering of those folds is in-sample and much easier than what a student meets at test time.
2. `src/distill.py` turns a window into training targets and fine-tunes a cross-encoder with RankNet. There are three targets:
   - the labels;
   - the teacher's ordering;
   - a hybrid: the labels, with the teacher's order breaking ties inside a grade.

   It also cross-fits the fold-0 pilot in halves, because fold 0 holds the only free teacher answers and is also the reporting surface.
3. `src/distill_report.py` runs that pilot and decides a **pre-registered gate**. It also scores every trained arm beside Stage 2, the landed cross-encoder and the LLM.
4. Only if the gate passes, `src/llm_rerank.py` labels the 12,519 training windows (Phase 2) and a student is trained on them (Phase 3).

**Tech Stack:** Python ≥3.11, sentence-transformers 6.1 (`CrossEncoderTrainer`, `RankNetLoss`), PyTorch on the RTX 3080 (16 GB), pandas, numpy, LightGBM 4.7 for the out-of-fold Stage 2, pytest. It reuses Plans 1 and 5–7:
- Plan 1: `src.metrics`, `src.floor`, `src.bootstrap`.
- Plan 5: `src.ranker`, `src.rank_report`.
- Plan 6: `src.rerank_window`, `src.cross_encoder`, `src.llm_rerank`, `src.fine_rank_report`.
- Plan 7: `src.stage_signals`, `src.blend_report`.

**No new dependencies.** Paid API calls happen **only in Task 4**, and only behind the gate.

**Spec:** `PROJECT_SPEC.md` §7.4 (Stage 3 fine rank) is what this plan argues from. It carries two entries that pull opposite ways:
- *Margin-MSE distillation* (Walmart, 2025): a 7B teacher and a BERT-base student on 170M teacher-labelled pairs, where the student matched the teacher.
- The **negative result** of arXiv:2507.08336: *"single-stage contrastive fine-tuning with hard negatives matched or beat distillation-augmented pipelines … Do the simple thing first."*

Also read `docs/RESULTS.md` §8, where this experiment was first proposed, and `CLAUDE.md`'s Data invariants and Evaluation discipline sections. Executors read all three.

**Series:** Plan 8, after the seven-plan series closed. See [`../README.md`](../README.md). It is not a §6 ablation. It answers a question the writeup left open: can the project keep most of the LLM's +0.0276 without paying 4.7 s a query for it?

---

## The three phases

| Phase | File | Tasks | Delivers | Network / GPU? |
|---|---|---|---|---|
| 1 | [**The Free Arms**](phase-1-the-free-arms.md) | 1–3 | Out-of-fold windows; window fine-tuning; the fold-0 pilot, its gate, and two free arms | GPU, about 2 h. **No API calls** |
| 2 | [**The Teacher**](phase-2-the-teacher.md) | 4 | The LLM's ordering of every training window, if the gate passed | **Paid**: about 4.2 h, 5.7M + 4.8M tokens |
| 3 | [**The Student and the Report**](phase-3-the-student-and-the-report.md) | 5–6 | The distilled student, if Phase 2 ran; every arm on the full test split once; the writeup | GPU, about 30 min |

Phase 1 comes first because it is free and decides whether Phase 2 happens at
all. Task 5 is skipped with Phase 2. Task 6 runs either way, because the free
arms are a result whether or not a student exists.

---

## Where these numbers came from

Every figure below was measured on 2026-09-27, before this plan was written, in
scratch code against the real fold-0 and training-fold data. **No API call was
made.** The GPU was shared throughout with another project's 4 GB index build,
so every time and every latency below is an upper bound.

**Two separate pilots say this plan's paid phase will not pay.** The plan is
built to measure that honestly, commit it, and stop — not to find a
configuration that wins.

- **The student's training windows must be carved out-of-fold.** Stage 2 was
  trained on folds 2/3/4, so its persisted ordering of those folds is
  in-sample:

  | folds 2/3/4 carved by | Stage 2 NDCG | Exact on top | window oracle | window already perfect |
  |---|---|---|---|---|
  | *fold 0, for reference (out-of-sample)* | *0.8519* | *71.1%* | *0.9533* | *12.2%* |
  | the persisted, **in-sample** ordering | **0.9035** | **87.7%** | 0.9655 | 17.7% |
  | an **out-of-fold** ordering (each fold fitted on the other two) | **0.8534** | **71.2%** | 0.9553 | 11.4% |

  A student trained on the in-sample windows would learn from a distribution
  where the top document is already Exact seven times in eight. It would meet
  the fold-0 distribution at test time, with 0.062 of headroom where there is
  in fact 0.102. And the teacher would be asked about the wrong products: only
  **34.5%** of in-sample windows hold the same ten documents as the out-of-fold
  ones (mean Jaccard 0.842). The out-of-fold fits cost 20 s of LightGBM, and
  fitting on two folds rather than three weakens Stage 2 by only 0.0016 on fold
  0 (0.8503 against 0.8519).

- **The teacher is better than every stage, and still noisy.** On fold 0's
  4,129 answered windows:

  | ordering | mixed-grade pairs right (83,190) | Exact above Irrelevant (17,441) |
  |---|---|---|
  | Stage 2 | 0.6373 | 0.6611 |
  | the landed cross-encoder | 0.6555 | 0.6738 |
  | **the LLM** | **0.7404** | **0.8001** |

  The labels are right on every one of those pairs by definition, so a student
  given the teacher's ordering is taught the wrong way round on about a quarter
  of the pairs the labels would teach correctly. The teacher's only structural
  advantage is the **4.3%** of training windows (539 of 12,519) whose documents
  all share one grade. There the labels say nothing and the teacher still gives
  an order.

- **Pilot 1 — the same student, the same queries, three targets.** Cross-fitted
  on fold 0 in halves of about 2,065 queries: train on one half, score the
  other, swap. `ms-marco-MiniLM-L6-v2` from its public checkpoint, `RankNetLoss`,
  2 epochs:

  | target | NDCG (4,129 queries) | vs. the labels target |
  |---|---|---|
  | labels | 0.8493 | — |
  | the teacher's ordering | 0.8501 | +0.0009 [−0.0006, +0.0022] — **ties** |
  | hybrid | 0.8499 | +0.0007 [−0.0005, +0.0017] — **ties** |

  All three sit *below* Stage 2 on the same queries (0.8520). With 2,065
  queries the student barely learns, so this pilot alone could be hiding a
  difference between targets. Hence pilot 2.

- **Pilot 2 — the same, starting from the landed cross-encoder.** The landed
  model has already learned folds 2/3/4 from labels. Continue for 1 epoch on
  one fold-0 half with each target and score the other half:

  | target | NDCG | vs. the labels target | Kendall τ with the teacher |
  |---|---|---|---|
  | *the landed model, unchanged* | *0.8588* | *+0.0003 [−0.0007, +0.0012]* | *0.374* |
  | labels | 0.8585 | — | 0.362 |
  | the teacher's ordering | 0.8569 | **−0.0016 [−0.0031, −0.0003]** — a significant **loss** | 0.387 |
  | hybrid | 0.8576 | −0.0009 [−0.0020, +0.0001] — ties | 0.382 |

  The student does move toward the teacher (τ 0.374 → 0.387), but it picks up
  the teacher's errors faster than its insight. And 2,065 more queries of
  *labels* do not move the landed model at all. That points at capacity, not
  data, as what holds the cross-encoder back — which is what the free capacity
  arm tests.

  **So the pre-registered gate is expected to fail, and the paid phase not to
  run.** The plan still writes Phases 2 and 3 in full: the gate is decided by
  the committed pilot, not by this scratch one, and the whole point of
  pre-registering it is that the plan does not know the answer in advance.

- **Hard negatives — the spec's "simple thing first" — lose to the landed
  recipe.** Fine-tuning the same MiniLM on only the 12,519 out-of-fold
  training windows, with labels and 1 epoch, and scoring fold 0:

  | recipe | NDCG | vs. the landed cross-encoder | fold-0 queries |
  |---|---|---|---|
  | windows only, `LambdaLoss` | 0.8556 | −0.0031 [−0.0043, −0.0019] | 4,129 (the answered ones) |
  | windows only, `RankNetLoss` | 0.8555 | −0.0031 [−0.0045, −0.0018] | all 4,130 |

  The loss does not matter. Training on the window alone does: it keeps
  125,080 of the 250,485 training pairs, all of them the hard ones. This
  measurement cannot say whether the cost comes from losing the easy
  negatives or from losing half the pairs. Either way, the spec's "simple
  thing first" loses to the recipe that already landed.

- **The capacity arm has to train on windows, not whole query groups.**
  Measured throughput on the 3080, with training batches of windows of 8–10
  documents:

  | backbone | batch | queries/s | one epoch, 12,519 windows | peak VRAM | latency, single / batched |
  |---|---|---|---|---|---|
  | `ms-marco-MiniLM-L6-v2` (the landed student) | 16 | 22.1 | 9.4 min | 3.54 GB | 24.8 / 7.0 ms |
  | `ms-marco-MiniLM-L12-v2` | 16 | 12.8 | 16.3 min | 6.85 GB | 34.8 / 15.8 ms |
  | **`bge-reranker-base`** | **8** | **3.3** | **62.3 min** | **11.07 GB** | **88.2 / 30.1 ms** |
  | `bge-reranker-base`, whole query groups (the landed recipe) | 4 | < 0.2 | > 17 h | the card, full | — |

  The whole-group run was stopped after 20 minutes without finishing 240
  queries: next to the other project's 4 GB it spills out of GPU memory. So the
  capacity arm is `bge-reranker-base` on the **same windows, loss and labels**
  as the MiniLM windows arm. Against that arm, the backbone is the only
  difference. Against the landed cross-encoder, it pays the windows-only
  penalty measured above.

- **What the paid pass would cost.** No out-of-fold training window is in the
  cache (0 of 12,519). At the full test split's measured rates — 454 prompt,
  382 completion and 334 reasoning tokens a call, and 1.217 s a call at
  concurrency 4 — that is **5,679,799 prompt + 4,783,476 completion tokens
  (4,183,753 reasoning), about 4.2 h.** This is the figure Task 4's dry run
  prints before anything is spent.

---

## The fold protocol

| role | queries | used for |
|---|---|---|
| **train** | folds 2/3/4 (12,519), windows carved **out-of-fold** | every at-scale fine-tune: the windows arm, the capacity arm, the student |
| **early-stop** | fold 1 (4,239) | Stage 2's early-stop set in every out-of-fold fit; nothing else |
| **pilot** | fold 0 (4,130), **cross-fitted in halves** | the gate only; its numbers are never a headline |
| **report** | fold 0 (4,130) | every arm's selection-surface number |
| **test** | all 8,956 test queries, **once**, behind `--final` | the headline |

The pilot trains on fold 0, which every earlier plan kept as a pure reporting
surface. That is legitimate for the same reason Plan 7's cross-fitted
combiner was. No fold-0 query is ever scored by a model that trained on it —
`cross_fit_orderings` refuses otherwise. The pilot's models are thrown away.
And the arms reported on fold 0 train on folds 2/3/4 only
(`check_training_folds`).

---

## The gate, written before the committed pilot runs

> **Pay for the teacher's answers on folds 2/3/4 only if, in the fold-0
> cross-fitted pilot, a teacher target (`llm` or `hybrid`) beats the `labels`
> target from the same starting point with a paired 95% interval wholly above
> zero.**

It is code, not prose: `src.distill_report.paid_run_gate`, with its own tests
(Task 3). It does not count:
- a teacher target beating Stage 2, which says nothing about teacher versus
  labels;
- a win across starting points, which measures the start, not the target;
- a tie.

If it passes, its `choice` — the win with the largest lower bound — is the
student Phase 3 trains, read from the committed JSON by
`python -m src.distill --from-gate`. Nobody picks the student by hand.

**Why this rule and not a weaker one.** Distillation's only lever over the
labels here is target quality per query. The labels already exist for every
one of the 12,519 training queries, and the teacher adds no new ones. If the
teacher's target does not beat the labels on the same queries at pilot scale,
paying for it at 6x the queries buys more of a target that is no better than
one already free.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Never tune on test.** The target, the starting point, the backbone, the loss and the epochs are fixed by this plan or chosen on fold 0. The test split is touched once, in Task 6, behind `--final`.
- **Paid calls happen in Task 4 only, only if the Phase 1 gate passed, and only after the user approves the spend in that session with the dry run's numbers in front of them.** No other task issues a call. The pilot and the report read the cache. The pass is resumable and appends its bill to `docs/results/llm-rerank-train-oof.json`.
- **Training windows come from the out-of-fold Stage 2, never the persisted in-sample one.** `src.distill.training_windows` reads `load_stage2("train-oof")` and nothing else.
- **The student and every free arm train on folds 2/3/4 only.** `check_training_folds` is called on every training frame. The pilot trains on fold-0 halves, and no query is scored by a model that trained on it.
- **A window the teacher did not answer is dropped from a teacher-target training set, never filled with Stage 2's order.** Drops are capped at 0.5% by `src.stage_signals.check_fallback_share`. Past that, the scope is wrong.
- **Every reported NDCG comes from `src.metrics.ndcg_per_query`**, with `src.floor.random_floor` and a bootstrap CI over queries, and arms are compared only over the same queries (`check_same_queries`). The reference arms must reproduce Plan 6's published numbers (`check_reference`) or the report stops.
- **Latency is reported single-query and batched** (`measure_latency`), because latency is this plan's whole reason. Time it with nothing else **training** on the GPU. Another project's job may hold ~4 GB of the 16 GB; say so beside the number when it does.
- **Run GPU work on the GPU.** `bge-reranker-base` at batch 8 needs ~11 GB; check `nvidia-smi` for at least 12 GB free before starting it.
- **A method that ties, or loses, is a reportable result.** Both scratch pilots say the gate will fail. If it does, that is the result. **Do not add targets, losses or epochs until one wins.**
- **Do not commit datasets or models.** `.gitignore` blocks `data/` and `models/`. Only JSON under `docs/results/`, code, tests and Markdown are committed.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)

---

## Review Focus

These are the five failure modes the spec implies but no task's happy path
exercises. Each is pinned to a test inside the task that owns the code.

1. **Training windows carved by the in-sample Stage 2.** The tempting call is `load_stage2("train")`, which already exists, is already on disk and covers folds 2/3/4. It is also wrong: 0.9035 against 0.8519, 87.7% Exact-on-top against 71.1%, and only a third of its windows hold the documents a test-like window would. Nothing downstream would flag it — the student trains, scores, and reports a number. `out_of_fold_scores` must never score a fold with a model fitted on it, and `training_windows` must read only the out-of-fold file. — pinned in Task 1 (`test_no_fold_is_scored_by_a_model_that_trained_on_it`, the data test) and Task 2.
2. **A pilot model scoring the queries it trained on.** Fold 0 holds 4,129 free teacher answers, and it is the reporting surface. A pilot that trains on all of fold 0 and scores all of fold 0 would read as a clear teacher win and mean nothing. `cross_fit_orderings` must refuse overlapping halves, stray windows and a scorer that orders its own training half. — pinned in Task 2.
3. **A malformed teacher answer taught as the teacher's opinion.** `rerank_windows` falls back to Stage 2's order on a malformed answer and returns it like any other ordering. A student trained on the return value would learn Stage 2 under the teacher's name, on exactly the windows the teacher found hardest. Training targets must come from `llm_orderings_from_cache`, which returns misses rather than filling them. `window_dataset` must drop and report those misses. — pinned in Task 2.
4. **A permutation turned into targets read backwards.** The loss reads a higher label as more relevant, and the teacher's ordering puts the best document at position 0. A target built as the position rather than `n − position` teaches the student to invert the teacher. The result would read as "distillation does not work", not as a bug. — pinned in Task 2 (`test_the_teacher_s_first_choice_gets_the_highest_target`).
5. **A paid pass over the wrong windows, or one the gate did not earn.** The cache key is the query and the *ordered* window, so a pass over in-sample windows, or over fold 0 or fold 1, spends four hours answering questions no student will ask. `check_oof_scope` must refuse anything but training folds 2/3/4 before anything is read. The dry run must count what the cache already holds with the same acceptance test the pass uses. And `paid_run_gate` must count only a teacher target against the labels from the same start. — pinned in Task 4 and Task 3.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as
`src.<name>` and run as `python -m src.<name>`. This plan follows it.

| File | Responsibility | Written in |
|---|---|---|
| `src/stage2_scores.py` | *Modified*: `out_of_fold_scores`, `stage2_ndcg`, `window_shift`, `OOF_SPLIT`, `--out-of-fold` | Phase 1, Task 1 |
| `src/distill.py` | Window targets, the window dataset, the cross-fit, pairwise accuracy, the RankNet fine-tune, the out-of-fold training windows, the `training.json` beside every model; `python -m src.distill` | Phase 1, Task 2 (`gate_choice` and `--from-gate` added in Task 3, beside the gate they read) |
| `src/distill_report.py` | The pilot and its gate; every arm beside the three stages on fold 0 or test; `python -m src.distill_report` | Phase 1, Task 3 |
| `src/llm_rerank.py` | *Modified*: `check_oof_scope`, `pass_plan`, `project_usage`, `--oof`, `--dry-run` | Phase 2, Task 4 |
| `docs/results/stage2-oof.json`, `distill-pilot.json`, `distill.json`, `distill-test.json`, `llm-rerank-train-oof.json` | The committed record (the last only if Phase 2 runs) | Tasks 1–6 |
| `docs/RESULTS.md`, `README.md`, `docs/superpowers/plans/README.md`, `CLAUDE.md` | The writeup, the headline table, the series entry and the commands | Phase 3, Task 6 |
| `tests/test_stage2_scores.py`, `tests/test_distill.py`, `tests/test_distill_report.py`, `tests/test_llm_rerank.py` | One test module per source module | Every task |

`src/distill.py` and `src/distill_report.py` are separate for the reason
`src/blend.py` and `src/blend_report.py` were. The targets and the cross-fit
are the parts worth testing exhaustively. A module that also loads models,
reads the signal frame and writes JSON cannot be tested that way.

---

## What this plan reuses rather than rebuilds

| From | What | Why it matters |
|---|---|---|
| Plan 5 | `src.stage2_scores.score_split`, `src.ranker.{TRAIN_FOLDS, EARLY_STOP_FOLD, REPORT_FOLD}` | The frozen Stage 2 configuration, and its `in_sample` flag derived from the rows actually fitted |
| Plan 6 | `src.cross_encoder.{rerank, load_reranker, measure_latency, text_maps, check_training_folds, DEFAULT_BACKBONE}` | One scoring path, one latency definition, one text map, one fold guard |
| Plan 6 | `src.llm_rerank.{RerankCache, window_key, rerank_windows, append_usage}` | The teacher, its cache and its bill |
| Plan 6 | `src.fine_rank_report.check_same_queries`, `docs/results/fine-rank{,-test-full}.json` | Comparability, and the published reference numbers `check_reference` holds the arms to |
| Plan 7 | `src.stage_signals.{scope_windows, load_signals, llm_orderings_from_cache, check_fallback_share}` | The windows and the three stages' orderings, exactly as published; the teacher's answers with their misses |
| Plan 7 | `src.blend_report.{single_stage_orderings, per_query_ndcg}` | Each stage's ordering from the signal frame, scored through the permutation-checking splice |
| Plans 1, 5 | `src.metrics`, `src.floor`, `src.bootstrap`, `src.rank_report.{evaluate_arm, compare, format_table, qrels_from_frame}` | The only trustworthy NDCG, and intervals paired by query |

---

## Plan Gate

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the out-of-fold ordering's reproduction of 0.8534 and, if Phase 2 ran, the teacher's coverage of the training windows.
- [ ] `docs/results/stage2-oof.json` records how far the in-sample training windows sit from the out-of-fold ones.
- [ ] `docs/results/distill-pilot.json` holds all six pilot arms over the same 4,130 fold-0 queries, and its `gate` block equals `paid_run_gate` applied to its own `comparisons`.
- [ ] The paid pass ran **only if** that gate passed, and only after the user approved the dry run's figures. Its bill is in `docs/results/llm-rerank-train-oof.json`. Or the gate did not pass, and no call was made.
- [ ] `docs/results/distill.json` and `distill-test.json` report every trained arm beside Stage 2, the landed cross-encoder and the LLM, with paired intervals against all three, the share of the LLM's gain each keeps, pairwise accuracy, and single-query and batched latency. The reference arms reproduce Plan 6's published numbers.
- [ ] The test split was touched once, behind `--final`.
- [ ] `docs/RESULTS.md` reports the outcome, whether or not a student was trained, and every figure in it is in a committed results file (`tests/test_results_doc.py`).
- [ ] `CLAUDE.md`'s Commands section lists the out-of-fold dump, the window fine-tunes, both report modes and — if it ran — the paid pass.

Then: record Plan 8 in [`../README.md`](../README.md).
