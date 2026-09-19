# Plan 1 — Evaluation Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-metric.md); do not start a phase before its predecessor's gate passes.

**Goal:** Build a trustworthy NDCG evaluation harness for Amazon ESCI Task 1 — asserted data invariants, a computed random floor, bootstrap confidence intervals, a frozen validation split — and prove it by reproducing a published zero-shot baseline.

**Architecture:** Pure functions over TREC-shaped nested dicts (`qid -> doc_id -> score`), which is `pytrec_eval`'s own interface, so the reference implementation can be used as a test oracle without shaping data twice. The dataset loader asserts documented row counts and the label distribution on every load and fails loudly rather than proceeding, because the most likely failure here is silently loading the wrong source file. Everything numeric is seeded and reproducible.

**Tech Stack:** Python ≥3.11, pandas + pyarrow (Parquet), numpy, `pytrec-eval-terrier` (test oracle only), `sentence-transformers` + torch (baseline only, optional extra), pytest.

**Spec:** `PROJECT_SPEC.md` (§2 The Task, §3.1 Amazon ESCI, §5 Baselines, §8.1 Build Order) and `CLAUDE.md` (Data invariants, Evaluation discipline). The plan argues from both; executors read both.

**Series:** Plan 1 of 7 — see [`../README.md`](../README.md).

---

## The three phases

The split is by dependency, not by size. Phase 1 needs nothing on disk and ends
with the metric proven against a reference implementation. Phase 2 is the first
phase that touches the real ~1 GB parquet. Phase 3 turns a correct metric over
correct data into a *reportable* number and spends it on the baseline that
proves the whole harness.

| Phase | File | Tasks | Delivers | Real data? |
|---|---|---|---|---|
| 1 | [**The Metric**](phase-1-the-metric.md) | 1–3 | Label gain tables, full-list NDCG, `pytrec_eval` oracle cross-check | No |
| 2 | [**The Data**](phase-2-the-data.md) | 4–6 | ESCI loader with asserted invariants, TREC run/qrels I/O, the computed random floor | Yes (~1.08 GB) |
| 3 | [**Statistics and the Baseline**](phase-3-statistics-and-baseline.md) | 7–10 | Bootstrap CIs, frozen validation folds, report + CLI, reproduced SBERT baseline | Yes |

**One cross-phase edit.** Task 9 (Phase 3, Step 4) reopens `src/floor.py` from
Phase 2 to add a `per_query` field, because `evaluate_run` pairs the run against
the floor query by query. Phase 2 must not treat `FloorResult` as final, and
Phase 3 must not implement `report.py` against the Phase 2 shape. Both phase
files carry this note.

---
## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **Use the official parquet, never the `tasksource/esci` HF mirror.** The mirror has no `split` column and encodes the *large-version* split, yielding 185,361 test judgements instead of 181,701 (~2% contamination). Fetch via `https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/...` (the repo is Git LFS; the plain `raw.githubusercontent.com` URL returns an LFS pointer, not a Parquet file).
- **Task 1 English filter is `small_version == 1` and `product_locale == "us"`**, then `split`.
- **Asserted row counts:** test = 181,701 judgements / 8,956 queries / 164,900 products; train = 419,653 judgements / 20,888 queries.
- **Asserted test label distribution:** E 43.87%, S 34.98%, I 16.69%, C 4.46%. Any source reporting Complement ≈ 35% has the known S/C swap bug in the official `prepare_trec_eval_files.py`.
- **Gains are E=1.0, S=0.1, C=0.01, I=0.0.** Integer qrels are the same values ×100 (E=100, S=10, C=1, I=0) so they can be handed to `pytrec_eval`, which requires integers. NDCG is invariant to this scaling; Task 2 tests that it is.
- **NDCG is full-list, no cutoff**, with gain used linearly (not `2^rel - 1`) and discount `1/log2(rank+1)` with rank starting at 1. This matches `trec_eval -m ndcg`, the official ESCI evaluation route.
- **Compute the random floor, never quote it.** `CLAUDE.md` records 0.7467 measured with this discount; the published SQID figure is 0.7483; under the swapped S/C mapping it is 0.7141. That 3.3-point spread is larger than most method gains, so the number must come from the code, not from a constant.
- **There is no official validation split.** Carve one from train by `query_id` and freeze it before tuning anything. Never split within a query group.
- **Never tune on test.**
- **Features must be query×product, never product-alone-from-labels.** 34,756 products appear in both train and test; the official split is query-level, so this is legitimate, but product-level target encoding leaks train labels into test.
- **Report bootstrap confidence intervals over queries, not bare point estimates.** A method that ties the baseline is a legitimate, reportable result.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)
- **Do not commit datasets.** `.gitignore` already blocks `data/`, `runs/`, `*.parquet`, `*.npy`. Frozen split artifacts therefore live at `splits/` in the repo root, deliberately outside `data/`, so they can be committed without fighting the ignore rules.

---

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **A run that omits judged products for a query.** A ranker that returns only its top 10 of 20 judged candidates must be scored against the ideal over *all* judged candidates. Dividing by the ideal of only what was returned silently inflates every score. — pinned in Task 2 (Phase 1).
2. **A run containing products with no judgement for that query.** The spec's funnel retrieves from a ~1.2M corpus, so unjudged products will appear. They must count as gain 0 and still consume a rank slot, per `trec_eval` convention — not raise, and not be dropped as if they were never returned. — pinned in Task 2 (Phase 1).
3. **A query where every judgement is Irrelevant.** IDCG is 0, and the naive ratio is a division by zero that produces `nan`, which then silently poisons the mean over 8,956 queries into `nan` or, worse, gets dropped and shifts the denominator. `trec_eval` scores such a query 0.0. — pinned in Task 2 (Phase 1).
4. **A scorer that returns identical scores for every candidate.** Tie order must be deterministic, or the same run file scores differently on two machines. A fully-tied run must land near the random floor, never at 1.0. — pinned in Task 2 (Phase 1).
5. **Loading the wrong source file.** The HF mirror parses cleanly, has the right column names, and yields 185,361 test rows. Nothing about it fails except the count, so the count must be checked on every load, and the error must name the mirror trap rather than just printing two numbers. — pinned in Task 4 (Phase 2).


---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as `src.<name>` and run as `python -m src.<name>` (`src/esci_images.py`, documented in `CLAUDE.md`). This plan follows that pattern rather than introducing a nested package, so the documented command keeps working.

| File | Responsibility | Written in |
|---|---|---|
| `pyproject.toml` | Dependencies, extras, pytest configuration and markers | Phase 1, Task 1 |
| `src/labels.py` | ESCI label → gain and label → integer qrel mappings; nothing else | Phase 1, Task 1 |
| `src/metrics.py` | `ndcg_per_query`, `mean_ndcg`. Pure, no I/O, no pandas | Phase 1, Task 2 |
| `src/dataset.py` | Download, load, filter and *assert invariants on* the official ESCI parquet | Phase 2, Task 4 |
| `src/runs.py` | TREC run-file read/write; DataFrame → qrels/run dict conversion | Phase 2, Task 5 |
| `src/floor.py` | Random-ordering floor, computed from qrels | Phase 2, Task 6 — **reopened in Phase 3, Task 9** |
| `src/bootstrap.py` | `bootstrap_ci`, `paired_delta_ci` over per-query scores | Phase 3, Task 7 |
| `src/splits.py` | Deterministic query-level fold assignment; freeze and load | Phase 3, Task 8 |
| `src/report.py` | Assemble and format an evaluation report; JSON serialisation | Phase 3, Task 9 |
| `src/cli.py` | `python -m src.cli {evaluate,floor,freeze-splits}` | Phase 3, Task 9 |
| `src/baseline_sbert.py` | Zero-shot SBERT run generation (optional extra) | Phase 3, Task 10 |
| `splits/val_folds.csv` | The frozen validation folds, committed | Phase 3, Task 8 |
| `docs/results/*.json` | Committed evaluation records | Phase 3, Tasks 9 and 10 |
| `tests/*.py` | One test module per source module | Every task |

`src/metrics.py` holds no pandas and no I/O on purpose: it is the one file whose correctness everything downstream rests on, and it must be testable against `pytrec_eval` without any data loading in the way.

---
## Plan Gate

Plan 2 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no skips in `tests/test_metrics_vs_pytrec.py`.
- [ ] `python -m pytest -m data` passes for both splits.
- [ ] The random floor, measured not quoted, is recorded in `docs/results/` and lands near 0.7467.
- [ ] `docs/results/sbert-title-test.json` shows NDCG in `[0.820, 0.840]` with a lift interval strictly above zero.
- [ ] `splits/val_folds.csv` and its `.sha256` are committed, covering all 20,888 train queries.
- [ ] `CLAUDE.md`'s Commands section lists the test and evaluation invocations.

---

## Self-Review

Run against `PROJECT_SPEC.md` and `CLAUDE.md` after writing the plan.

**1. Spec coverage.** §2 (task, labels, metric, S/C bug) → Tasks 1, 2, 3. §3.1 (dataset, splits, counts) → Task 4. §5 (baselines, the 0.8562 target) → Task 10 establishes the measuring apparatus; beating 0.8562 is Plan 7's gate. §8.1 (the build-order gate) → Task 10. `CLAUDE.md` evaluation discipline (train/test product overlap, GroupKFold by query_id, bootstrap CIs) → Tasks 4, 7, 8.

Deliberately out of scope for this plan, and owned elsewhere: §3.2 and §3.3 (ESCI-S, images) → Plans 2 and 3; §4 (the funnel stages) → Plans 4–7; §6 (the ablation table) → Plans 4–7; §7 (methods) → Plans 4–6; §9 (compute) → Plans 3 and 6, where GPU work first appears.

**2. Placeholder scan.** No "TBD", no "add appropriate error handling", no "similar to Task N". Every code step carries the code. The one repeated idea — the ascending-document-id tie policy — is written out in full in both `src/metrics.py` and `src/runs.py` and cross-tested, rather than one referring to the other.

**3. Type consistency.** `Qrels` and `Run` are both `Mapping[str, Mapping[str, float]]`, defined once in `src/metrics.py` and imported by `src/runs.py` and `src/report.py`. Ids are `str` everywhere past `src/runs.py`. `Interval` is defined once in `src/bootstrap.py` and used by `src/report.py`. `FloorResult.per_query` is added in Task 9 Step 4 with a covering test, and Task 9's Interfaces block names it as a dependency so nobody implements `report.py` against the Task 6 shape.

**4. Review Focus.** All five items have tests in the task that owns the code: partial runs, unjudged documents and zero-ideal queries in `tests/test_metrics.py` (Task 2) and again against `pytrec_eval` (Task 3); ties in `tests/test_metrics.py` and `tests/test_runs.py` (Tasks 2 and 5); the wrong source file in `tests/test_dataset.py` (Task 4), including the mirror's exact 185,361 row count and the S/C-swapped label distribution.
