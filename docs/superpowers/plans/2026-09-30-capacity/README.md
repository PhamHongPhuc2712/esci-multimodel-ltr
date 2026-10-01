# Plan 9 — Capacity, One Factor at a Time

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-code.md); do not start a phase before its predecessor's gate passes.

**Goal:** Find out why `bge-reranker-base` beat the landed cross-encoder in Plan 8. Is it the bigger backbone, the batch size, or the training recipe? Then name the project's best Stage 3 that needs no API call, choosing it on fold 0.

**Architecture:**
- Plan 8's bge arm scored 0.8695 on the full test split against the landed MiniLM's 0.8616. But it differed from that model in four things at once: backbone, data (windows vs. whole query groups), loss (RankNet vs. LambdaLoss) and batch (8 vs. 16). Plan 8's review flagged this.
- This plan adds two arms, so that each comparison changes exactly one thing:
  - `stage2+ce_bge_groups` is bge trained with the landed recipe unchanged;
  - `stage2+ce_bge_b16` is bge on the windows at the MiniLM windows arm's batch of 16.
- Both need gradient checkpointing to fit the 16 GB card. It only changes when activations are computed, not what a step computes. `src/cross_encoder.py` and `src/distill.py` gain the flag, and the whole-group fine-tune starts writing the `training.json` record the report already requires.
- `src/distill_report.py` gains a `--capacity` preset: the four arms, the three extra comparisons each labelled with what it isolates, and the best arm on fold 0.

**Tech Stack:** Python ≥3.11, sentence-transformers 6.1, PyTorch on the RTX 3080 (16 GB), pytest. It reuses Plan 6's `src.cross_encoder`, Plan 8's `src.distill` and `src.distill_report`, and Plan 7's signal frames. **No new dependencies. No API calls.**

**Spec:** `PROJECT_SPEC.md` §7.4 (Stage 3) and §7.5 (the KDD Cup winners' larger backbones), with `docs/RESULTS.md` §8–9, where this question was left open. Also read `CLAUDE.md`'s Evaluation discipline section.

**Series:** Plan 9, after Plan 8. See [`../README.md`](../README.md). It is not a §6 ablation. It settles the attribution of Plan 8's result, and it fixes which cross-encoder the close-out plan promotes into the headline.

---

## The two phases

| Phase | File | Tasks | Delivers | GPU? |
|---|---|---|---|---|
| 1 | [**The Code**](phase-1-the-code.md) | 1–2 | Gradient checkpointing and training records in both fine-tunes; the `--capacity` preset | Briefly, for a smoke test |
| 2 | [**The Runs and the Writeup**](phase-2-the-runs-and-the-writeup.md) | 3–4 | Two trained arms; fold 0; the test split once; the writeup | About 2 h |

---

## Where these numbers came from

Every figure below was measured on 2026-09-30, on an otherwise idle RTX 3080,
before this plan was written.

- **bge cannot train on whole query groups without gradient checkpointing.**
  The landed recipe (`LambdaLoss`, batch 8, `max_length` 192, every judged pair
  of folds 2/3/4) over the 12,519 training queries. Groups average 20.0
  documents, with a 99th percentile of 42 and a maximum of 109.

  | configuration | queries/s | one epoch | peak memory, allocated / reserved |
  |---|---|---|---|
  | batch 8 | < 0.27 | > 13 h | spills; 320 queries did not finish in 20 min |
  | batch 2, 4 accumulation steps | 1.41 | 148 min | 15.10 / 16.17 GB — over the card |
  | batch 1, 8 accumulation steps | 0.44 | 479 min | 14.97 / 16.23 GB — over the card |
  | **batch 8, gradient checkpointing** | **3.18** | **66 min** | **6.29 / 7.12 GB** |

  Accumulation does not help: the peak barely moves with the batch, and both
  accumulated runs reserve more than the card holds. Checkpointing recomputes
  activations during backward instead of storing them. The step's arithmetic
  is unchanged, so this *is* the landed recipe. The Trainer's own
  `gradient_checkpointing=True` raises inside sentence-transformers 6.1
  (`gradient_checkpointing_enable() got an unexpected keyword argument
  'every_n_layers'`), so it is switched on the wrapped transformers model.

- **bge on the windows at batch 16 needs it too, and runs fast with it.**
  With checkpointing: 9.63 windows/s, **22 min an epoch**, 5.84 GB peak. Plan
  8's bge arm ran batch 8 without it in 21.9 min.

- **What Plan 8 measured, which this plan re-scores and must reproduce:**

  | arm | recipe | fold 0 | test |
  |---|---|---|---|
  | `stage2+ce` (landed) | MiniLM, whole groups, `LambdaLoss`, batch 8 | 0.8587 | 0.8616 |
  | `stage2+ce_windows` | MiniLM, windows, `RankNetLoss`, batch 16 | 0.8555 | 0.8581 |
  | `stage2+ce_bge` | bge, windows, `RankNetLoss`, batch 8 | 0.8650 | 0.8695 |

  For MiniLM, windows-only training cost 0.0032 on fold 0. Whether bge pays the
  same penalty is exactly what `stage2+ce_bge_groups` measures. If it does,
  bge on whole groups should beat Plan 8's 0.8650. Nothing here predicts it.

---

## The comparison grid

With the landed cross-encoder and Plan 8's two arms, the two new arms make
every comparison below change exactly one thing. The training records in
`capacity.json` — not this table — are what a test holds each arm to.

| comparison | changes only | where it comes from |
|---|---|---|
| `bge_groups` − `ce` (landed) | the backbone, under the landed recipe | the standard "vs `stage2+ce`" row |
| `bge_b16` − `ce_windows` | the backbone, on windows at batch 16 | `CAPACITY_PAIRS` |
| `bge` − `bge_b16` | the batch, 8 against 16 | `CAPACITY_PAIRS` |
| `bge_groups` − `bge` | the recipe: whole groups and `LambdaLoss` against windows and `RankNetLoss`, both at batch 8 | `CAPACITY_PAIRS` |

"The backbone" means everything `bge-reranker-base` brings: its size, its
tokenizer and its own reranking pre-training. This plan does not separate
those. Checkpointing differs between some pairs; it changes memory, not
arithmetic.

**The no-API headline is chosen on fold 0, and written down before anything
is scored.** `best_model_arm` in `docs/results/capacity.json` — the
highest-scoring of the four capacity arms on fold 0 — is the cross-encoder the
close-out plan promotes. Every arm goes to the test split once, whatever it
scored. The test file records its own `best_model_arm` too, but nothing reads
it.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **No API calls.** Nothing in this plan calls the LLM, and the LLM arm is read from Plan 7's signal frame.
- **Never tune on test.** The four recipes are fixed by this plan, and the headline arm is chosen on fold 0. The test split is touched once, in Task 4, behind `--final`.
- **Every at-scale fine-tune trains on folds 2/3/4 only.** `check_training_folds` guards the whole-group fine-tune, and `training_windows` the window one.
- **Every reported NDCG comes from `src.metrics.ndcg_per_query`** with the computed floor and a bootstrap CI over queries, and arms are compared only over the same queries. The reference arms must reproduce Plan 6 (`check_reference`), and Plan 8's two arms must reproduce Plan 8.
- **Latency is reported single-query and batched**, timed with nothing else training on the GPU.
- **A tie is a reportable result.** If the backbone does not separate from the recipe, say so. **Do not add arms or retrain until one wins.**
- **Do not commit datasets or models.** Only JSON under `docs/results/`, code, tests and Markdown are committed.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`.)

---

## Review Focus

These are the five failure modes most likely to bite, each pinned to a test in
the task that owns the code.

1. **Checkpointing that silently does not switch on.** Without it, bge on whole groups does not fail. It spills into shared memory and runs at under 0.27 queries/s, a 13-hour epoch that looks like a slow machine. `enable_gradient_checkpointing` must raise when the switch does not take. — pinned in Task 1.
2. **A comparison that changes more than one thing, reported as if it changed one.** That is the finding this plan exists to correct. Each pair must be labelled with what it isolates, and the training commands must differ exactly where the labels say they do. — pinned in Task 2 (`test_the_commands_differ_where_the_pairs_say_they_do`) and Task 3 (`test_the_capacity_arms_were_trained_as_their_pairs_claim`, against the committed training records).
3. **The headline arm picked on test.** Four arms and one test run make it tempting to promote whichever scored best there. `best_model_arm` must be computed from fold 0 and read from `capacity.json` only. — pinned in Task 2 and Task 3.
4. **A model directory that holds something other than what its arm claims.** It could be a leftover smoke run, a hand-copied model, or Plan 8's bge sitting under the new name. `training_records` refuses a directory with no record, and the committed test checks each record's init, loss and batch. — pinned in Task 3.
5. **Plan 8's arms drifting when re-scored.** The capacity report re-scores `ce_windows` and `ce_bge`, and every pair built on them assumes they are the same models on the same windows. They must reproduce `distill.json` and `distill-test.json` to 2e-4. — pinned in Tasks 3 and 4.

---

## File Structure

| File | Responsibility | Written in |
|---|---|---|
| `src/cross_encoder.py` | *Modified*: `enable_gradient_checkpointing`, `training_record`, `TRAINING_RECORD`, `LOSS_NAMES`; `fine_tune(gradient_checkpointing=…)`; `--gradient-checkpointing`; the CLI writes `training.json` | Phase 1, Task 1 |
| `src/distill.py` | *Modified*: `fine_tune_windows(gradient_checkpointing=…)`, `--gradient-checkpointing`, recorded; `TRAINING_RECORD` now imported from `src.cross_encoder` | Phase 1, Task 1 |
| `src/distill_report.py` | *Modified*: two model arms, `CAPACITY_ARMS`, `CAPACITY_PAIRS`, `CAPACITY_OUT`, `pair_comparisons`, `best_model_arm`, `--capacity` | Phase 1, Task 2 |
| `docs/results/capacity.json`, `capacity-test.json` | The committed record | Tasks 3–4 |
| `docs/RESULTS.md`, `README.md`, `docs/superpowers/plans/README.md`, `CLAUDE.md` | The writeup, the headline table, the series entry and the commands | Task 4 |
| `tests/test_cross_encoder.py`, `tests/test_distill_report.py` | One test module per source module | Every task |

---

## Plan Gate

**Passed 2026-10-01. The backbone is what paid; the no-API Stage 3 is bge on whole query groups.**

- [x] `python -m pytest` passes with no failures and no new skips. — 808 passed (787 before this plan), 36 deselected, 0 skipped.
- [x] `python -m pytest -m data` passes. — 35 passed, 0 skipped. One data test had been skipping since 2026-09-30, because it still named the deleted `rerank` store. It now checks the catalogue store (fae9c3b).
- [x] `docs/results/capacity.json` holds the four capacity arms beside the three reference arms on fold 0. Plan 8's arms reproduce `distill.json`. Every pair in the grid has a paired interval labelled with what it isolates. The training records match the recipes the pairs claim. `best_model_arm` is recorded. — The reference arms are 0.8519 / 0.8587 / 0.8814, and Plan 8's are 0.8555 / 0.8650. The pairs:
  - backbone: +0.0098 [+0.0074, +0.0124];
  - batch: −0.0004 [−0.0016, +0.0010], a tie;
  - recipe: +0.0011 [−0.0004, +0.0027], a tie.

  `best_model_arm` is `stage2+ce_bge_groups`, at 0.8661.
- [x] `docs/results/capacity-test.json` holds the same arms on all 8,956 test queries, measured once. Plan 8's arms reproduce `distill-test.json`. — 0.8581 and 0.8695. The pairs:
  - backbone: +0.0112 [+0.0094, +0.0130];
  - batch: +0.0002 [−0.0006, +0.0010], a tie;
  - recipe: +0.0016 [+0.0006, +0.0026].

  `stage2+ce_bge_groups` scores 0.8711 [0.8681, 0.8742].
- [x] `docs/RESULTS.md` says what each comparison isolated and measured, names the no-API Stage 3 chosen on fold 0 with its test number, and quotes nothing that is not in a committed results file. — §8 *Which part of the larger model paid*; `tests/test_results_doc.py` passes.
- [x] `CLAUDE.md` lists the two new training commands and the capacity report.

Then: record Plan 9 in [`../README.md`](../README.md).
