# Phase 2 — The Teacher

**Plan 8 · Phase 2 of 3 · Task 4.** Read [`README.md`](README.md) first. It
carries the goal, the cost projection and the full Global Constraints. **Do not
open this phase unless the [Phase 1 gate](phase-1-the-free-arms.md#phase-1-gate)
recorded `passed`.** If it recorded `not passed`, mark Task 4 skipped and go to
[Task 6](phase-3-the-student-and-the-report.md#task-6-the-test-split-once-and-the-writeup).

**Delivers:** the LLM's ordering of every one of the 12,519 out-of-fold
training windows, in `data/llm-rerank.json`, and the bill for it in
`docs/results/llm-rerank-train-oof.json`.

**Needs on disk:** `data/features/stage2-train-oof.parquet` (Task 1),
`data/llm-rerank.json`, `docs/results/llm-rerank-test-full.json` (the rates
the dry run projects from). `OPENAI_API_KEY` in the environment for the paid
step only.

**This is the only phase that spends money.** At the full test split's
measured rates, that is 5,679,799 prompt + 4,783,476 completion tokens
(4,183,753 reasoning), about 4.2 h at concurrency 4.

**Owns Review Focus item 5** (a paid pass over the wrong windows).

### Constraints that bite hardest here

- **The user approves the spend in the session that runs it,** with the dry run's printed figures in front of them. The Phase 1 gate passing is necessary, not sufficient.
- **Only folds 2/3/4, only their out-of-fold windows.** `check_oof_scope` refuses anything else before a file is read.
- **The pass is resumable.** The cache checkpoints every 200 windows. An interrupted pass is re-run with the same command, and `--usage-out` appends each leg, so the committed bill is the sum.
- **Nothing here trains or scores.** The student is Task 5.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 4: Label the training windows with the teacher

**Files:**
- Modify: `src/llm_rerank.py` (add `check_oof_scope`, `pass_plan`, `project_usage`; add `--oof`, `--dry-run`, `--projection-from` to `_main`; record `oof` in the usage record)
- Modify: `tests/test_llm_rerank.py`
- Modify: `tests/test_distill.py` (the teacher-coverage data test)
- Create (by running): `docs/results/llm-rerank-train-oof.json`

**Interfaces:**
- Consumes: `src.stage2_scores.{OOF_SPLIT, load_stage2}` (Task 1), `src.ranker.TRAIN_FOLDS`, `src.distill.training_windows` (Task 2), `src.llm_rerank.{RerankCache, window_key, rerank_windows, append_usage}` (Plan 6).
- Produces:
  - `src.llm_rerank.check_oof_scope(split, folds) -> None`
  - `src.llm_rerank.pass_plan(windows_, query_text, cache) -> dict` — `{"n_windows", "n_cached", "n_uncached"}`
  - `src.llm_rerank.project_usage(n_calls, record) -> dict` — `{"n_calls", "prompt_tokens", "completion_tokens", "reasoning_tokens", "hours"}`
  - CLI: `python -m src.llm_rerank --split train --folds 2,3,4 --oof [--dry-run] [--usage-out PATH]`
  - The cache holds a teacher ordering for at least 99.5% of the out-of-fold training windows. This is what `python -m src.distill --from-gate` reads in Task 5.

**Review Focus 5 lives here.** The cache is keyed on the query and the
*ordered* window, so there are three expensive ways to get this wrong, and
none of them raises:
- a pass over the in-sample windows (`load_stage2("train")`, the existing default);
- a pass over fold 0 or fold 1;
- a pass whose count nobody looked at first.

Every one spends hours answering questions no student will ask.
`check_oof_scope` runs before any file is read. `pass_plan` counts cached
windows with the same acceptance test `rerank_windows` applies, so its count
of calls is exactly the count the pass will make. `--dry-run` prints that
count and its projected bill, then exits without building a client.

- [ ] **Step 1: Write the failing tests**

In `tests/test_llm_rerank.py`, extend the import:

```python
from src.llm_rerank import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MODEL,
    MAX_COMPLETION_TOKENS,
    PROMPT,
    RerankCache,
    Usage,
    append_usage,
    build_prompt,
    check_oof_scope,
    parse_permutation,
    pass_plan,
    project_usage,
    rerank_windows,
    window_key,
)
```

and append:

```python
# --- Plan 8: the teacher's pass over the training windows -------------------
# Review Focus 5. Four hours of paid calls over the wrong windows cannot be
# taken back, so the scope is checked before anything is read and the count is
# shown before anything is spent.


def test_the_out_of_fold_pass_accepts_the_training_folds():
    check_oof_scope("train", [2, 3, 4])
    check_oof_scope("train", [3])


def test_the_out_of_fold_pass_refuses_the_reporting_and_early_stop_folds():
    for folds in ([0], [1], [0, 2, 3, 4], [1, 2]):
        with pytest.raises(ValueError, match="reporting surface"):
            check_oof_scope("train", folds)


def test_the_out_of_fold_pass_refuses_the_test_split():
    with pytest.raises(ValueError, match="train split"):
        check_oof_scope("test", [2, 3, 4])


def test_the_out_of_fold_pass_refuses_no_folds():
    with pytest.raises(ValueError, match="no folds"):
        check_oof_scope("train", [])


class _Cache:
    def __init__(self, entries):
        self._entries = entries

    def get(self, key):
        return self._entries.get(key)


def test_pass_plan_counts_what_the_cache_already_answers():
    ws = [
        Window(query_id="1", window=("a", "b"), tail=()),
        Window(query_id="2", window=("x", "y", "z"), tail=()),
    ]
    text = {"1": "red shoes", "2": "blue hat"}
    cache = _Cache({window_key("red shoes", ["a", "b"]): [2, 1]})
    assert pass_plan(ws, text, cache) == {"n_windows": 2, "n_cached": 1, "n_uncached": 1}


def test_pass_plan_counts_a_malformed_cache_entry_as_a_call():
    # rerank_windows re-asks when the cached permutation does not fit the
    # window, so the plan must count it as a call too.
    ws = [Window(query_id="1", window=("a", "b", "c"), tail=())]
    cache = _Cache({window_key("red shoes", ["a", "b", "c"]): [2, 1]})
    assert pass_plan(ws, {"1": "red shoes"}, cache)["n_uncached"] == 1


def test_pass_plan_is_keyed_on_the_window_order():
    # The same products in another order are another question.
    ws = [Window(query_id="1", window=("b", "a"), tail=())]
    cache = _Cache({window_key("red shoes", ["a", "b"]): [2, 1]})
    assert pass_plan(ws, {"1": "red shoes"}, cache)["n_cached"] == 0


def _record():
    return {
        "runs": [{"usage": {"seconds": 1200.0}}, {"usage": {"seconds": 800.0}}],
        "totals": {
            "n_calls": 1000,
            "prompt_tokens_per_call": 450.0,
            "completion_tokens_per_call": 380.0,
            "reasoning_tokens_per_call": 330.0,
        },
    }


def test_project_usage_scales_the_committed_per_call_figures():
    projected = project_usage(500, _record())
    assert projected["n_calls"] == 500
    assert projected["prompt_tokens"] == 225_000
    assert projected["completion_tokens"] == 190_000
    assert projected["reasoning_tokens"] == 165_000
    # 2,000 s over 1,000 calls is 2 s a call, so 500 calls is 1,000 s.
    assert projected["hours"] == pytest.approx(1000 / 3600)


def test_nothing_to_call_costs_nothing():
    projected = project_usage(0, _record())
    assert projected["prompt_tokens"] == 0 and projected["hours"] == 0


def test_the_committed_full_split_record_projects_the_training_pass():
    # Measured before Plan 8 was written: 12,519 out-of-fold windows, none
    # cached, at the full test split's own rates.
    record = json.loads(open("docs/results/llm-rerank-test-full.json", encoding="utf-8").read())
    projected = project_usage(12_519, record)
    assert projected["prompt_tokens"] == pytest.approx(5.68e6, rel=0.005)
    assert projected["completion_tokens"] == pytest.approx(4.78e6, rel=0.005)
    assert projected["hours"] == pytest.approx(4.2, abs=0.1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_llm_rerank.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_oof_scope' from 'src.llm_rerank'`.

- [ ] **Step 3: Implement**

In `src/llm_rerank.py`, add after `append_usage`:

```python
def check_oof_scope(split: str, folds: Sequence[int]) -> None:
    """Refuse an out-of-fold pass over anything but the training folds.

    Plan 8's teacher labels the windows a student trains on. Fold 0 is the
    reporting surface and fold 1 the early-stop set; a teacher answer there
    is a training label on queries the plan reports, and the out-of-fold
    ordering does not cover them anyway.
    """
    from src.ranker import TRAIN_FOLDS

    if split != "train":
        raise ValueError("out-of-fold windows exist only for the train split")
    folds = [int(f) for f in folds]
    if not folds:
        raise ValueError("no folds given")
    outside = sorted(set(folds) - set(TRAIN_FOLDS))
    if outside:
        raise ValueError(
            f"fold(s) {outside} have no out-of-fold windows; only "
            f"{list(TRAIN_FOLDS)} do. Fold 0 is the reporting surface and fold 1 "
            "the early-stop set."
        )


def pass_plan(windows_: Sequence, query_text: Mapping, cache: RerankCache) -> dict:
    """How many windows a pass would answer from the cache, and how many it would pay for.

    Uses the same acceptance test as rerank_windows, so a cached entry of the
    wrong length counts as a call, exactly as the pass would treat it.
    """
    n_cached = 0
    for w in windows_:
        documents = list(w.window)
        cached = cache.get(window_key(str(query_text[w.query_id]), documents))
        if cached is not None and sorted(cached) == list(range(1, len(documents) + 1)):
            n_cached += 1
    return {
        "n_windows": len(windows_),
        "n_cached": n_cached,
        "n_uncached": len(windows_) - n_cached,
    }


def project_usage(n_calls: int, record: Mapping) -> dict:
    """What `n_calls` should cost, from a committed record of real calls.

    Per-call tokens are the record's own totals; hours use the record's wall
    clock over its calls, so the concurrency it ran at is built in.
    """
    totals = record["totals"]
    seconds = sum(float(run["usage"]["seconds"]) for run in record["runs"])
    per_call_seconds = seconds / max(int(totals["n_calls"]), 1)
    return {
        "n_calls": n_calls,
        "prompt_tokens": round(n_calls * totals["prompt_tokens_per_call"]),
        "completion_tokens": round(n_calls * totals["completion_tokens_per_call"]),
        "reasoning_tokens": round(n_calls * totals["reasoning_tokens_per_call"]),
        "hours": n_calls * per_call_seconds / 3600,
    }
```

In `_main`, extend the imports:

```python
    from src.cross_encoder import text_maps
    from src.ranker import TRAIN_FOLDS
    from src.rerank_window import DEFAULT_K, windows
    from src.stage2_scores import OOF_SPLIT, load_stage2
```

add three arguments after `--usage-out`:

```python
    parser.add_argument(
        "--oof", action="store_true",
        help="carve windows from the out-of-fold Stage 2 (Plan 8's training "
             "windows); train split, folds 2/3/4 only")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="count the windows and project the bill; make no call")
    parser.add_argument(
        "--projection-from", type=Path,
        default=Path("docs/results/llm-rerank-test-full.json"),
        help="the committed usage record a dry run projects from")
```

replace the fold filter and the Stage 2 load with:

```python
    matrix = pd.read_parquet(args.features_dir / f"{args.split}.parquet")
    if args.oof:
        wanted = (
            list(TRAIN_FOLDS) if args.folds == "all"
            else [int(f) for f in args.folds.split(",")]
        )
        check_oof_scope(args.split, wanted)
        matrix = matrix.loc[matrix["fold"].isin(wanted)]
    elif args.folds != "all":
        wanted = [int(f) for f in args.folds.split(",")]
        matrix = matrix.loc[matrix["fold"].isin(wanted)]
    if args.sample is not None:
        keep = (
            matrix["query_id"].drop_duplicates().sample(n=args.sample, random_state=args.seed)
        )
        matrix = matrix.loc[matrix["query_id"].isin(set(keep))]

    scores = load_stage2(OOF_SPLIT if args.oof else args.split)
    joined = matrix[["query_id", "product_id"]].merge(
        scores, on=["query_id", "product_id"]
    )
    if args.oof and len(joined) != len(matrix):
        raise SystemExit(
            f"the out-of-fold ordering covers {len(joined):,} of {len(matrix):,} rows; "
            "re-run python -m src.stage2_scores --split train --out-of-fold"
        )
    ws = windows(joined, k=args.k)
```

and replace the `cache = RerankCache(...)` / `print(...)` pair with:

```python
    cache = RerankCache(args.cache)
    plan = pass_plan(ws, query_text, cache)
    print(f"{plan['n_windows']:,} windows of up to {args.k}: {plan['n_cached']:,} "
          f"answered from the cache, {plan['n_uncached']:,} to call")
    if args.dry_run:
        record = json.loads(args.projection_from.read_text(encoding="utf-8"))
        projected = project_usage(plan["n_uncached"], record)
        print(f"projected from {args.projection_from}: "
              f"{projected['prompt_tokens']:,} prompt + "
              f"{projected['completion_tokens']:,} completion tokens "
              f"({projected['reasoning_tokens']:,} reasoning), about "
              f"{projected['hours']:.1f} h")
        print("dry run: no call made")
        return 0
```

and record which windows a leg covered, so the committed bill says so:

```python
        payload = append_usage(args.usage_out, {
            "split": args.split, "folds": args.folds, "oof": args.oof, "sample": args.sample,
            "k": args.k, "model": args.model, "n_windows": len(ws),
            "usage": usage.to_dict(),
        })
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_llm_rerank.py -v`
Expected: PASS — 42 passed (32 before this task).

- [ ] **Step 5: Commit the code**

```bash
git add src/llm_rerank.py tests/test_llm_rerank.py
git commit -m "Guard the teacher's pass to the out-of-fold training windows and add a dry run"
```

- [ ] **Step 6: Dry run, and ask**

Run: `python -m src.llm_rerank --split train --folds 2,3,4 --oof --dry-run`
Expected, with no API call and no key needed:

```
12,519 windows of up to 10: 0 answered from the cache, 12,519 to call
projected from docs/results/llm-rerank-test-full.json: 5,679,799 prompt + 4,783,476 completion tokens (4,183,753 reasoning), about 4.2 h
dry run: no call made
```

**Stop here and show the user these three lines.** Ask whether to spend them.
Do not continue without an explicit yes in this session. An earlier approval,
a passed gate, or this plan saying the step exists is not one.

Check the other refusals while here. Each must stop before reading a file:
`--folds 0 --oof --dry-run` and `--folds 1,2 --oof --dry-run` raise
`ValueError: fold(s) [...] have no out-of-fold windows`.

- [ ] **Step 7: The paid pass**

Only after the user's yes:

Run: `python -m src.llm_rerank --split train --folds 2,3,4 --oof --usage-out docs/results/llm-rerank-train-oof.json`
Expected: about 4.2 h. The fallback count should be around the full test
split's 12 in 6,958 calls. If the pass is interrupted, re-run the same command.
The cache answers what is done, and the second leg is appended to the same
record.

- [ ] **Step 8: Pin the teacher's coverage**

Append to `tests/test_distill.py`:

```python


@pytest.mark.data
def test_the_teacher_answered_the_training_windows():
    # Phase 2's paid pass. A malformed answer is never cached, so a handful of
    # windows may be missing - never more than check_fallback_share allows.
    from pathlib import Path

    from src.cross_encoder import text_maps
    from src.distill import training_windows
    from src.llm_rerank import RerankCache
    from src.stage_signals import check_fallback_share, llm_orderings_from_cache

    windows_, matrix = training_windows()
    query_text, _ = text_maps(
        matrix,
        Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}
    orderings, missing = llm_orderings_from_cache(windows_, query_text, RerankCache())
    assert len(orderings) + len(missing) == 12_519
    check_fallback_share(missing, len(windows_))
```

Run: `python -m pytest -m data tests/test_distill.py -v`
Expected: PASS. At most 62 of the 12,519 windows (0.5%) lack a teacher answer.

- [ ] **Step 9: Commit the bill**

```bash
git add tests/test_distill.py docs/results/llm-rerank-train-oof.json
git commit -m "Label the out-of-fold training windows with the LLM teacher"
```

---

## Phase 2 Gate

- [ ] `python -m pytest` passes with no failures and no new skips. — 793 passed, 37 deselected.
- [ ] `python -m pytest -m data` passes, including `test_the_teacher_answered_the_training_windows`.
- [ ] The user's approval of the dry run's figures is recorded in this file, with the date.
- [ ] `docs/results/llm-rerank-train-oof.json` holds every leg of the pass and its totals.
- [ ] No window outside folds 2/3/4's out-of-fold set was called: every leg in the record says `"oof": true`, `"folds": "2,3,4"` and `"n_windows": 12519`.

Then: [Phase 3 — The Student and the Report](phase-3-the-student-and-the-report.md).
