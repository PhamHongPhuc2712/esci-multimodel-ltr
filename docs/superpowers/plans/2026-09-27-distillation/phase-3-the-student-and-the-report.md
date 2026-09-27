# Phase 3 — The Student and the Report

**Plan 8 · Phase 3 of 3 · Tasks 5–6.** Read [`README.md`](README.md) first. It
carries the goal, the measurements, the full Global Constraints, and the Plan
Gate this phase closes.

**Delivers:**
- Task 5 — the distilled student and its fold-0 number. **Only if Phase 2 ran.**
- Task 6 — every trained arm on all 8,956 test queries, once, and the writeup. **Always.**

**Needs on disk:**
- `models/cross-encoder/{windows,bge}/` (Task 3);
- `data/features/stage-signals-test.parquet` (Plan 7);
- `docs/results/distill{,-pilot}.json` (Task 3);
- for Task 5, the cache holding the teacher's answers for the training windows (Task 4).

**No API calls.** The student reads the cache Phase 2 filled; the report reads
the signal frame.

### Constraints that bite hardest here

- **The student is exactly what the gate chose.** `--from-gate` reads it from the committed pilot. No target, start, loss or epoch count is picked by hand, and nothing is re-run because the student's fold-0 number disappoints.
- **Every arm measured on fold 0 goes to test, and nothing else does.** A committed test checks the two arm sets are equal.
- **The test split is touched once**, behind `--final`.
- **Time the arms with nothing else training on the GPU.** Say beside the latency if another project's job was resident.
- **Every four-decimal figure in the writeup comes from a committed results file.** `tests/test_results_doc.py` enforces it for `docs/RESULTS.md` and `README.md` both.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 5: Train the student the gate chose

**Skip this task if the [Phase 1 gate](phase-1-the-free-arms.md#phase-1-gate)
recorded `not passed`.** Mark it `Skipped: the Phase 1 gate did not pass` and
go to Task 6.

**Files:**
- No source changes. Every piece of code this task runs landed in Tasks 2 and 3.
- Create (by running): `models/cross-encoder/student/` (gitignored), and a new `docs/results/distill.json` that includes the student.

**Interfaces:**
- Consumes: `python -m src.distill --from-gate` and `src.distill.gate_choice` (Task 3), `src.distill.training_windows` (Task 2), the teacher's answers in `data/llm-rerank.json` (Task 4), `python -m src.distill_report --arms` (Task 3).
- Produces: `models/cross-encoder/student/` with its `training.json`, and the `stage2+student` arm in `docs/results/distill.json`. Task 6 reads both.

**Nothing new is tested here, because nothing new is written.** The committed
test `test_the_student_is_exactly_what_the_gate_chose` (Task 3) now takes its
other branch. It requires the student arm in `distill.json`, and it requires
the student's own training record to name the gate's `(init, target)`.

- [ ] **Step 1: Train the student**

Run: `python -m src.distill --from-gate docs/results/distill-pilot.json --out models/cross-encoder/student`
Expected (about 10 min):
- a first line naming the gate's target and init;
- a drop count of at most 62, the malformed teacher answers;
- `trained in … min -> models/cross-encoder/student`.

If the drop count is above 62, `check_fallback_share` has already stopped the
run, and Phase 2's pass is incomplete. Re-run it; do not lower the cap.

- [ ] **Step 2: Report it on fold 0 beside the free arms**

Run with no training on the GPU: `python -m src.distill_report --arms stage2+ce_windows stage2+ce_bge stage2+student`
Expected: the three reference arms reproduce Plan 6, the two free arms
reproduce their Task 3 numbers exactly (scoring is deterministic), and
`stage2+student` appears with its intervals, share of the LLM's gain,
pairwise accuracy, latency and training record. Overwrites
`docs/results/distill.json`.

- [ ] **Step 3: Run the committed tests**

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: PASS — 24 passed. `test_the_student_is_exactly_what_the_gate_chose` now checks the student branch.

- [ ] **Step 4: Commit**

```bash
git add docs/results/distill.json
git commit -m "Train the student the pilot's gate chose and report it on fold 0"
```

---

## Task 6: The test split, once, and the writeup

**Files:**
- Modify: `tests/test_distill_report.py` (two tests that pin the test report)
- Create (by running): `docs/results/distill-test.json`
- Modify: `docs/RESULTS.md`, `README.md`, `docs/superpowers/plans/README.md`, `CLAUDE.md`, and this plan's [`README.md`](README.md) Plan Gate

**Interfaces:**
- Consumes: `python -m src.distill_report --split test --final` (Task 3); every committed `docs/results/*.json`.
- Produces: `docs/results/distill-test.json`, and the writeup's Plan 8 section. No Python API.

**The test report scores exactly the arms the fold-0 report scored.** An arm
dropped after a bad fold-0 number, or added after a good test number, is
selection on the test split. `test_the_test_report_scores_exactly_the_fold_0_arms`
makes that a failure rather than a judgement call.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_distill_report.py`:

```python


def test_the_test_report_scores_exactly_the_fold_0_arms():
    # An arm dropped after a bad fold-0 number, or added after a good test
    # number, is selection on the test split.
    fold0 = {arm["name"] for arm in _committed("distill.json")["arms"]}
    test = {arm["name"] for arm in _committed("distill-test.json")["arms"]}
    assert test == fold0


def test_the_test_arms_sit_on_the_published_test_windows():
    report = _committed("distill-test.json")
    arms = {arm["name"]: arm["ndcg"]["point"] for arm in report["arms"]}
    published = {
        arm["name"]: arm["ndcg"]["point"]
        for arm in _committed("fine-rank-test-full.json")["arms"]
    }
    assert report["n_queries"] == 8956
    for name in REFERENCE_ARMS:
        assert arms[name] == pytest.approx(published[name], abs=5e-4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: FAIL — `FileNotFoundError: … docs/results/distill-test.json`.

- [ ] **Step 3: Score the test split, once**

The `--arms` list is every model arm in `docs/results/distill.json`. That is
`stage2+ce_windows stage2+ce_bge`, plus `stage2+student` if Task 5 ran. Run
with no training on the GPU:

Run: `python -m src.distill_report --split test --final --arms stage2+ce_windows stage2+ce_bge` (add `stage2+student` if Task 5 ran)
Expected (about 6 min; bge scores 8,956 windows at about 30 ms each):
- `test: 8,956 windows`;
- the reference arms reproduce Plan 6 — `stage2` 0.8579, `stage2+ce` 0.8616, `stage2+llm` 0.8855 — or `check_reference` stops the run;
- each model arm with its intervals.

Writes `docs/results/distill-test.json`. **This is the only run of this command.**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: PASS — 26 passed.

- [ ] **Step 5: Commit the test report**

```bash
git add tests/test_distill_report.py docs/results/distill-test.json
git commit -m "Score the distillation arms on the full test split"
```

- [ ] **Step 6: Write it up**

Write from the committed JSON only. Open each file and quote it. Do not retype
a number from this plan: its figures were measured in scratch code before the
committed runs, and the pilot's halves differ.

**`docs/RESULTS.md`.** Insert a new `## 8. Distilling the LLM` before the
current §8, and renumber that one to `## 9. What would come next`. The new
section, in this order:

1. **The answer, in its first sentence.** Take one of these two branches.
   - If the gate did not pass: the paid labelling was not run, because on
     fold 0 the teacher's ordering taught a cross-encoder no more than the
     labels did, from either starting point. Quote the four comparisons from
     `distill-pilot.json`.
   - If it passed: the student's test NDCG and interval from
     `distill-test.json`, its delta against `stage2+ce` and against
     `stage2+llm`, its share of the LLM's gain, and its single-query latency
     against the LLM's 4.7 s.
2. **Why the windows had to be out-of-fold**, from `stage2-oof.json`:
   in-sample against out-of-fold NDCG, Exact on top, and the share of windows
   that keep their documents.
3. **The teacher is noisy**, from `distill.json`'s `pairwise_accuracy`: Stage
   2, the landed cross-encoder and the LLM on mixed-grade window pairs.
4. **The free arms**, from `distill-test.json`, as a table. One row per arm,
   with these columns:
   - NDCG and interval;
   - the delta against `stage2+ce`;
   - the share of the LLM's gain it keeps;
   - single-query and batched latency.

   Then one sentence each on:
   - hard negatives against the landed recipe — the spec's "simple thing first", measured;
   - `bge-reranker-base` against the MiniLM windows arm — the same windows, loss and labels, so the backbone is the only difference.
5. **What it cost**:
   - if Phase 2 ran, the tokens and hours from `llm-rerank-train-oof.json`;
   - if not, the dry run's projection from this plan's README (5.7M prompt + 4.8M completion tokens, about 4.2 h), described as a projection that was not spent.

In §7, "What did not work", add one bullet for each loss or tie measured
here. In §9, replace the "Distil the LLM into the cross-encoder" bullet with
what this plan's result points to. If the capacity arm moved and the teacher
did not, that is a bigger student on labels.

**`README.md`.** Add one row to "What the ablations found" for this plan's
headline comparison, quoted from `distill-test.json` with its interval. Add
the Plan 8 commands to "Reproducing". Change the layout row's "the seven
implementation plans" to "the eight implementation plans".

**`docs/superpowers/plans/README.md`.** Add a Plan 8 row to the series table and a
paragraph in the style of the others: what was measured before writing, what
landed, and the gate's verdict.

**`CLAUDE.md`.** Add to the Commands section:

```bash
# Distillation (Plan 8). The pilot decides whether the paid pass runs.
python -m src.stage2_scores --split train --out-of-fold   # -> data/features/stage2-train-oof.parquet, ~20 s
python -m src.distill --target labels --out models/cross-encoder/windows   # ~10 min
python -m src.distill --target labels --init BAAI/bge-reranker-base --batch-size 8 --out models/cross-encoder/bge   # ~62 min, 11 GB
python -m src.distill_report                      # every arm on fold 0
python -m src.distill_report --pilot              # the cross-fitted pilot and the gate, ~31 min
python -m src.distill_report --split test --final --arms stage2+ce_windows stage2+ce_bge \
    --out docs/results/distill-test.json
```

and, only if Phase 2 ran:

```bash
python -m src.llm_rerank --split train --folds 2,3,4 --oof --dry-run        # the count and the bill, no call
python -m src.llm_rerank --split train --folds 2,3,4 --oof \
    --usage-out docs/results/llm-rerank-train-oof.json                     # paid, ~4.2 h
python -m src.distill --from-gate docs/results/distill-pilot.json --out models/cross-encoder/student
```

- [ ] **Step 7: Run every test**

Run: `python -m pytest`
Expected: PASS — 785 passed if the gate did not pass (795 if Phase 2 ran), 36 deselected (37).

Run: `python -m pytest -m data`
Expected: PASS, including `tests/test_results_doc.py`, which fails on any
four-decimal figure in `docs/RESULTS.md` or `README.md` that no committed
results file contains.

- [ ] **Step 8: Tick the Plan Gate and commit**

Tick each box in this plan's [`README.md`](README.md#plan-gate) with the number that
satisfies it, as Plans 1–7 did.

```bash
git add docs/RESULTS.md README.md docs/superpowers/plans/README.md CLAUDE.md docs/superpowers/plans/2026-09-27-distillation/README.md
git commit -m "Write up the distillation experiment and close Plan 8"
```
