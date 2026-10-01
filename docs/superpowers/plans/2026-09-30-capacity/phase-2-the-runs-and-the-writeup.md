# Phase 2 — The Runs and the Writeup

**Plan 9 · Phase 2 of 2 · Tasks 3–4.** Read [`README.md`](README.md) first. Do
not start before the [Phase 1 gate](phase-1-the-code.md#phase-1-gate) passes.

**Delivers:** two trained bge arms; `docs/results/capacity.json` (fold 0);
`docs/results/capacity-test.json` (the test split, once); and the writeup.

**Needs on disk:**
- `data/features/{train,test}.parquet`, `data/features/stage2-train-oof.parquet` (Plan 8);
- `data/features/stage-signals-{fold0,test}.parquet` (Plan 7);
- `models/cross-encoder/{lambda,windows,bge}/` (Plans 6 and 8);
- a GPU with about 8 GB free — about 90 min of training, then the reports.

**Owns Review Focus items 2–5** (in their committed form: the training records,
the fold-0 choice, and Plan 8's arms reproducing).

### Constraints that bite hardest here

- **Nothing else trains on the GPU during the reports.** They time every arm.
- **The headline arm is `best_model_arm` from `capacity.json`.** It is written into this file before Task 4 runs.
- **The test split is touched once**, with all four arms.
- **Every four-decimal figure in the writeup comes from a committed results file.**
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 3: Train the two arms and report them on fold 0

**Files:**
- Create (by running): `models/cross-encoder/bge-b16/`, `models/cross-encoder/bge-groups/lambda/` (gitignored), `docs/results/capacity.json`
- Modify: `tests/test_distill_report.py`

**Interfaces:**
- Consumes: the two `TRAIN_COMMANDS` entries and `python -m src.distill_report --capacity` (Tasks 1–2).
- Produces: `docs/results/capacity.json`, whose `best_model_arm` Task 4 and the close-out plan read.

- [ ] **Step 1: Train bge on the windows at batch 16**

Run: `python -m src.distill --target labels --init BAAI/bge-reranker-base --batch-size 16 --gradient-checkpointing --out models/cross-encoder/bge-b16`
Expected (about 22 min):

```
12,519 training windows from folds 2/3/4, target 'labels', init 'BAAI/bge-reranker-base'; 0 dropped for want of a teacher answer
trained in … min -> models/cross-encoder/bge-b16
```

- [ ] **Step 2: Train bge with the landed recipe**

Run: `python -m src.cross_encoder --loss lambda --backbone BAAI/bge-reranker-base --gradient-checkpointing --out-dir models/cross-encoder/bge-groups`
Expected (about 66 min):

```
250,485 pairs over 12,519 queries from folds [2, 3, 4]
trained in … min -> models/cross-encoder/bge-groups/lambda
```

Check that `models/cross-encoder/bge-groups/lambda/training.json` says
`"loss": "LambdaLoss"`, `"batch_size": 8` and `"gradient_checkpointing": true`.

- [ ] **Step 3: Report the four arms on fold 0**

Run with nothing else on the GPU: `python -m src.distill_report --capacity`
Expected (about 6 min):
- the reference arms reproduce Plan 6 — 0.8519 / 0.8587 / 0.8814 — or `check_reference` stops the run;
- `stage2+ce_windows` and `stage2+ce_bge` reproduce Plan 8's 0.8555 and 0.8650;
- the two new arms are unmeasured, and whatever they score is the result;
- the last three comparison lines are the labelled pairs.

Writes `docs/results/capacity.json`.

- [ ] **Step 4: Pin the committed fold-0 results**

Append to `tests/test_distill_report.py`:

```python


def _points(name):
    return {arm["name"]: arm["ndcg"]["point"] for arm in _committed(name)["arms"]}


def test_the_capacity_arms_were_trained_as_their_pairs_claim():
    # Review Focus 2 and 4: each pair changes one thing, and the training
    # records - not the plan - are what say so.
    record = _committed("capacity.json")["training"]
    bge, b16, groups = (
        record["stage2+ce_bge"], record["stage2+ce_bge_b16"], record["stage2+ce_bge_groups"]
    )
    for arm in (bge, b16, groups):
        assert arm["init"] == "BAAI/bge-reranker-base"
        assert arm["target"] == "labels"
    assert (bge["loss"], bge["batch_size"]) == ("RankNetLoss", 8)
    assert (b16["loss"], b16["batch_size"]) == ("RankNetLoss", 16)
    assert (groups["loss"], groups["batch_size"]) == ("LambdaLoss", 8)
    assert b16["gradient_checkpointing"] and groups["gradient_checkpointing"]
    assert record["stage2+ce_windows"]["batch_size"] == 16


def test_the_capacity_file_re_scores_plan_8_s_arms_exactly():
    # Review Focus 5. The same models on the same windows; a drift here means
    # the windows or a model directory changed underneath the comparison.
    capacity, plan8 = _points("capacity.json"), _points("distill.json")
    for name in (*REFERENCE_ARMS, "stage2+ce_windows", "stage2+ce_bge"):
        assert capacity[name] == pytest.approx(plan8[name], abs=2e-4), name


def test_the_fold_0_file_names_its_best_arm_by_its_own_numbers():
    # Review Focus 3: the no-API headline is chosen here, on fold 0.
    report = _committed("capacity.json")
    points = _points("capacity.json")
    assert set(CAPACITY_ARMS) <= set(points)
    assert report["best_model_arm"] == max(CAPACITY_ARMS, key=points.__getitem__)
    isolated = {row["isolates"] for row in report["comparisons"] if "isolates" in row}
    assert isolated == set(CAPACITY_PAIRS.values())
```

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: PASS — 39 passed.

- [ ] **Step 5: Record the choice, and commit**

Write the fold-0 `best_model_arm`, and its NDCG, into this file under this
step. Task 4 must not start until it is written here.

**The fold-0 choice, recorded 2026-10-01: `stage2+ce_bge_groups`, NDCG 0.8661
[0.8612, 0.8707]** (`docs/results/capacity.json`). It is +0.0074 [+0.0051,
+0.0100] over the landed cross-encoder.

**Landed 2026-10-01.** Both arms trained while another project's GPU job
(an index build) shared the card, so their recorded minutes are not the
plan's:
- `bge-b16` took 35.1 min against the plan's 22. Its record says
  `RankNetLoss`, batch 16, checkpointing on, 12,519 windows and none dropped.
- `bge-groups` took 56.7 min against the plan's 66, at 3.72 queries/s. Its
  record says `LambdaLoss`, batch 8, checkpointing on, 12,519 queries and
  250,485 pairs. GPU memory held between 8.9 and 12.5 GB *including* the
  other job, so the run never spilled.

The report ran on an idle card in about 10 minutes:
- **Reproduced.** The reference arms gave 0.8519 / 0.8587 / 0.8814, and
  Plan 8's arms 0.8555 / 0.8650. Tests: 39 passed.
- **The pairs, on fold 0:**
  - backbone: +0.0098 [+0.0074, +0.0124];
  - batch: −0.0004 [−0.0016, +0.0010], a tie;
  - recipe: +0.0011 [−0.0004, +0.0027], a tie.
- **One latency is off.** The landed cross-encoder timed 31.2 ms single
  against Plan 8's 19.0. Its batched figure matched, 9.5 against 10.0 ms, and
  every bge arm timed as fast as or faster than before. The landed model is
  the first one the report times, so this looks like warm-up. The writeup
  quotes the landed arm's test latency, not this one.

```bash
git add tests/test_distill_report.py docs/results/capacity.json docs/superpowers/plans/2026-09-30-capacity/phase-2-the-runs-and-the-writeup.md
git commit -m "Report the four capacity arms on fold 0 and choose the no-API cross-encoder there"
```

---

## Task 4: The test split, once, and the writeup

**Files:**
- Modify: `tests/test_distill_report.py`
- Create (by running): `docs/results/capacity-test.json`
- Modify: `docs/RESULTS.md`, `README.md`, `docs/superpowers/plans/README.md`, `CLAUDE.md`, and this plan's [`README.md`](README.md) Plan Gate

**Interfaces:**
- Consumes: `python -m src.distill_report --capacity --split test --final`; every committed `docs/results/*.json`.
- Produces: `docs/results/capacity-test.json` and the writeup. No Python API.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_distill_report.py`:

```python


def test_the_capacity_test_file_scores_the_fold_0_arms_and_nothing_else():
    assert set(_points("capacity-test.json")) == set(_points("capacity.json"))


def test_the_capacity_test_file_re_scores_plan_8_s_test_arms_exactly():
    capacity, plan8 = _points("capacity-test.json"), _points("distill-test.json")
    assert _committed("capacity-test.json")["n_queries"] == 8956
    for name in (*REFERENCE_ARMS, "stage2+ce_windows", "stage2+ce_bge"):
        assert capacity[name] == pytest.approx(plan8[name], abs=2e-4), name
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: FAIL — `FileNotFoundError: … docs/results/capacity-test.json`.

- [ ] **Step 3: Score the test split, once**

Run with nothing else on the GPU: `python -m src.distill_report --capacity --split test --final`
Expected (about 15 min — three bge arms over 8,956 windows):
- `test: 8,956 windows`;
- the reference arms reproduce 0.8579 / 0.8616 / 0.8855;
- `stage2+ce_windows` and `stage2+ce_bge` reproduce Plan 8's 0.8581 and 0.8695.

Writes `docs/results/capacity-test.json`. **This is the only run of this command.**

- [ ] **Step 4: Run the tests to verify they pass, and commit**

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: PASS — 41 passed.

```bash
git add tests/test_distill_report.py docs/results/capacity-test.json
git commit -m "Score the four capacity arms on the full test split"
```

- [ ] **Step 5: Write it up**

Write from the committed JSON only; open each file and quote it.

**`docs/RESULTS.md` §8.** After the bullets on the two free arms, add a short
subsection titled *Which part of the larger model paid*. It holds:

1. **A table from `capacity-test.json`** with one row per pair in the comparison grid:
   - the `stage2+ce_bge_groups` − `stage2+ce` row, from the standard comparisons;
   - the three `isolates` rows.

   Each row gives what changed, the paired delta with its interval, and whether
   it is significant.
2. **The same four deltas on fold 0**, from `capacity.json`, in one sentence or a
   second column, because the choice below was made there.
3. **The no-API Stage 3.** Name the fold-0 `best_model_arm`, then give its test
   NDCG with interval, its delta against the landed cross-encoder, its share of
   the LLM's gain, and its single-query latency.
4. **Anything that tied, stated as a tie.** If the backbone and the recipe do
   not separate, the section says so and does not rank them.

Then update the parts of the writeup the answer changes:
- **§2 Headline:** the no-API sentence names the fold-0 choice and its test number.
- **§3 Stage 3 table:** its bge row becomes the fold-0 choice.
- **§7:** a bullet for any comparison that lost or tied.
- **§9:** the "larger student" bullet. Both of its untried things are now tried. Replace it with what the grid points to next, or remove it.

**`README.md`.**
- Change the bge row of the headline table to the fold-0 choice, with its test number.
- Change the "model size, fine rank" row of "What the ablations found" to the comparison that isolates the backbone, with its interval.
- Add the three Plan 9 commands to "Reproducing".
- Change the layout row's "the eight implementation plans" to "the nine implementation plans".

**`docs/superpowers/plans/README.md`.** Add a Plan 9 row and a paragraph in the
style of Plan 8's.

**`CLAUDE.md`.** Add to the Commands section:

```bash
# Capacity (Plan 9, 2026-09-30). Gradient checkpointing lets bge fit 16 GB.
python -m src.distill --target labels --init BAAI/bge-reranker-base --batch-size 16 \
    --gradient-checkpointing --out models/cross-encoder/bge-b16              # ~22 min
python -m src.cross_encoder --loss lambda --backbone BAAI/bge-reranker-base \
    --gradient-checkpointing --out-dir models/cross-encoder/bge-groups       # ~66 min
python -m src.distill_report --capacity           # fold 0 -> docs/results/capacity.json
python -m src.distill_report --capacity --split test --final   # -> capacity-test.json
```

- [ ] **Step 6: Run every test**

Run: `python -m pytest`
Expected: PASS — 808 passed (787 before this plan), 36 deselected.

Run: `python -m pytest -m data`
Expected: PASS, including `tests/test_results_doc.py`.

- [ ] **Step 7: Tick the Plan Gate and commit**

Tick each box in this plan's [`README.md`](README.md#plan-gate) with the number
that satisfies it.

```bash
git add docs/RESULTS.md README.md docs/superpowers/plans/README.md CLAUDE.md docs/superpowers/plans/2026-09-30-capacity/README.md
git commit -m "Write up which part of the larger cross-encoder paid and close Plan 9"
```
