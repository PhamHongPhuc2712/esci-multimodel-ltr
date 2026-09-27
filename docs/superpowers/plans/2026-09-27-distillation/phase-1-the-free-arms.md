# Phase 1 — The Free Arms

**Plan 8 · Phase 1 of 3 · Tasks 1–3.** Read [`README.md`](README.md) first. It
carries the goal, the measurements this phase re-takes in committed code, the
full Global Constraints, and the gate this phase decides.

**Delivers:**
- `src/stage2_scores.out_of_fold_scores`, and the out-of-fold ordering of folds 2/3/4 on disk;
- `src/distill.py`: window targets, the cross-fit and the RankNet fine-tune;
- `src/distill_report.py`, and with it two free arms on fold 0 — hard negatives, and a bigger backbone on the same windows;
- the fold-0 pilot, and **the decision on whether Phase 2 spends money**.

**Needs on disk:**
- `data/features/{train,test}.parquet` and `data/features/stage2-{train,test}.parquet` (Plans 5–6);
- `data/features/stage-signals-{fold0,test}.parquet` (Plan 7);
- `models/cross-encoder/lambda/` (Plan 6);
- `data/llm-rerank.json` (Plan 6);
- `data/combined/{products,judgements}.parquet`;
- a GPU with at least 12 GB free for the capacity arm.

**No API calls.** The pilot reads the fold-0 teacher answers already in the
cache.

**Owns Review Focus items 1–4** (in-sample training windows, a pilot model
scoring its own training queries, a malformed answer taught as the teacher's,
a permutation read backwards). It also owns the gate half of item 5.

### Constraints that bite hardest here

The full list is in [`README.md`](README.md#global-constraints); these are the
ones this phase's code can get wrong:

- **Training windows come from the out-of-fold Stage 2, never `load_stage2("train")`.**
- **No fold-0 query is scored by a pilot model that trained on it.** Every arm reported on fold 0 trains on folds 2/3/4 only.
- **A window without a teacher answer is dropped, never filled with Stage 2's order.**
- **The gate is the code in `paid_run_gate`, decided by the committed pilot.** Do not change the rule, the targets, the losses or the epochs after seeing the pilot.
- **No API calls in this phase.**
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool.**

---

## Task 1: Score the training folds out-of-fold

**Files:**
- Modify: `src/stage2_scores.py` (import `json`, and `REPORT_FOLD` at module level; add `OOF_SPLIT`, `out_of_fold_scores`, `stage2_ndcg`, `window_shift`; make `load_stage2`'s error name the right command; add `--out-of-fold` and `--results-out` to `_main`)
- Modify: `tests/test_stage2_scores.py`
- Create (by running): `docs/results/stage2-oof.json`

**Interfaces:**
- Consumes: `src.stage2_scores.score_split` (Plan 6), `src.ranker.{folds, train_ranker, predict, TRAIN_FOLDS, EARLY_STOP_FOLD, REPORT_FOLD}` (Plan 5).
- Produces:
  - `src.stage2_scores.OOF_SPLIT: str` = `"train-oof"`
  - `src.stage2_scores.out_of_fold_scores(train, *, fit_folds=TRAIN_FOLDS, features=None, seed=0) -> pd.DataFrame` — `STAGE2_COLUMNS`, one row per training-fold judgement, `in_sample` all False
  - `src.stage2_scores.stage2_ndcg(joined) -> float` — mean full-list NDCG of the `stage2_score` ordering; `joined` needs `query_id`, `product_id`, `qrel`, `stage2_score`
  - `src.stage2_scores.window_shift(matrix, out_of_fold, in_sample, *, k=None) -> dict` — `ndcg_*`, `exact_on_top_*` for both orderings, `same_window_share`, `n_queries`, `k`
  - `load_stage2(OOF_SPLIT)` reads `data/features/stage2-train-oof.parquet`
  - CLI: `python -m src.stage2_scores --split train --out-of-fold`, which also commits `docs/results/stage2-oof.json` — the in-sample/out-of-fold comparison, so the writeup can quote it

**Review Focus 1 lives here.** The persisted `stage2-train.parquet` already
covers folds 2/3/4, and it is the wrong thing to train a student on. Its model
trained on those folds, so it scores them 0.9035 against fold 0's 0.8519.
`out_of_fold_scores` fits each fold's scorer on the *other two* training folds,
with fold 1 still the early-stop set. That lands at 0.8534, which is fold 0's
distribution. `score_split` already derives `in_sample` from the rows it
actually fitted, so an out-of-fold row can never be flagged in-sample. The
spy test checks the fits themselves anyway, because a derived flag is only as
good as what it is derived from.

- [ ] **Step 1: Write the failing tests**

In `tests/test_stage2_scores.py`, extend the import:

```python
from src.stage2_scores import (
    DEFAULT_DIR,
    OOF_SPLIT,
    STAGE2_COLUMNS,
    load_stage2,
    out_of_fold_scores,
    require_out_of_sample,
    score_split,
    stage2_ndcg,
    window_shift,
)
```

and add this block above the first `@pytest.mark.data` test:

```python
# --- out-of-fold scores for the training folds (Plan 8) ---------------------
# A student trained on Stage 2's window must see the window it meets at test
# time. The persisted ordering of folds 2/3/4 is in-sample - 0.9035 against
# fold 0's 0.8519 - so the training windows come from these instead.


def test_out_of_fold_scores_cover_exactly_the_training_folds():
    frame = _frame()
    out = out_of_fold_scores(frame, features=FEATURES)
    expected = frame.loc[frame["fold"].isin(TRAIN_FOLDS)]
    assert list(out.columns) == list(STAGE2_COLUMNS)
    assert len(out) == len(expected)
    assert set(zip(out["query_id"], out["product_id"])) == set(
        zip(expected["query_id"], expected["product_id"])
    )


def test_no_fold_is_scored_by_a_model_that_trained_on_it(monkeypatch):
    # Review Focus 1. `in_sample` is derived from the fit, so check the fits
    # themselves: record the folds each model trained on and the folds it
    # then scored.
    import src.stage2_scores as module

    real_train, real_predict = module.train_ranker, module.predict
    fitted: dict[int, set] = {}
    calls: list[tuple[set, set]] = []

    def spy_train(fit, early_stop, columns, **kwargs):
        ranker = real_train(fit, early_stop, columns, **kwargs)
        fitted[id(ranker)] = set(fit["fold"])
        return ranker

    def spy_predict(ranker, target):
        calls.append((fitted[id(ranker)], set(target["fold"])))
        return real_predict(ranker, target)

    monkeypatch.setattr(module, "train_ranker", spy_train)
    monkeypatch.setattr(module, "predict", spy_predict)
    out_of_fold_scores(_frame(), features=FEATURES)

    assert len(calls) == len(TRAIN_FOLDS)
    for fit_folds, scored_folds in calls:
        assert not fit_folds & scored_folds
        assert REPORT_FOLD not in fit_folds
        assert EARLY_STOP_FOLD not in fit_folds
    assert set().union(*(scored for _, scored in calls)) == set(TRAIN_FOLDS)


def test_every_out_of_fold_row_is_out_of_sample():
    out = out_of_fold_scores(_frame(), features=FEATURES)
    assert not out["in_sample"].any()
    assert len(require_out_of_sample(out)) == len(out)


def test_out_of_fold_scores_are_not_the_in_sample_scores():
    frame = _frame()
    oof = out_of_fold_scores(frame, features=FEATURES)
    in_sample = score_split(
        frame, frame.loc[frame["fold"].isin(TRAIN_FOLDS)], features=FEATURES
    )
    merged = oof.merge(in_sample, on=["query_id", "product_id"], suffixes=("_oof", "_in"))
    assert len(merged) == len(oof)
    assert not np.allclose(merged["stage2_score_oof"], merged["stage2_score_in"])


def test_the_reporting_fold_is_refused():
    with pytest.raises(ValueError, match="reporting surface"):
        out_of_fold_scores(_frame(), fit_folds=(REPORT_FOLD, 2, 3), features=FEATURES)


def test_the_early_stop_fold_is_refused():
    with pytest.raises(ValueError, match=rf"\[{EARLY_STOP_FOLD}\]"):
        out_of_fold_scores(_frame(), fit_folds=(EARLY_STOP_FOLD, 2), features=FEATURES)


def test_one_fold_cannot_be_scored_out_of_fold():
    with pytest.raises(ValueError, match="at least two"):
        out_of_fold_scores(_frame(), fit_folds=(2,), features=FEATURES)


def test_a_missing_out_of_fold_file_names_the_command_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--out-of-fold"):
        load_stage2(OOF_SPLIT, directory=tmp_path)


def test_stage2_ndcg_is_one_for_an_ideal_ordering():
    joined = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": ["a", "b", "c"],
            "qrel": [100, 10, 0],
            "stage2_score": [3.0, 2.0, 1.0],
        }
    )
    assert stage2_ndcg(joined) == pytest.approx(1.0)


def _scores(frame, sign=1.0):
    return pd.DataFrame(
        {
            "query_id": frame["query_id"],
            "product_id": frame["product_id"],
            "stage2_score": sign * frame["signal"],
        }
    )


def test_identical_orderings_show_no_shift():
    frame = _frame(n_queries=5, folds=(2,))
    shift = window_shift(frame, _scores(frame), _scores(frame), k=3)
    assert shift["same_window_share"] == 1.0
    assert shift["ndcg_in_sample"] == shift["ndcg_out_of_fold"]
    assert shift["exact_on_top_in_sample"] == shift["exact_on_top_out_of_fold"]
    assert shift["n_queries"] == 5


def test_opposite_orderings_share_no_window():
    # Six products a query and windows of three: reversing the order leaves
    # the two windows disjoint.
    frame = _frame(n_queries=5, folds=(2,))
    shift = window_shift(frame, _scores(frame), _scores(frame, sign=-1.0), k=3)
    assert shift["same_window_share"] == 0.0
    assert shift["ndcg_out_of_fold"] > shift["ndcg_in_sample"]


def test_exact_on_top_reads_the_first_window_document():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "product_id": ["a", "b", "c", "d"],
            "qrel": [100, 0, 0, 100],
            "signal": [2.0, 1.0, 2.0, 1.0],
        }
    )
    shift = window_shift(frame, _scores(frame), _scores(frame, sign=-1.0), k=1)
    assert shift["exact_on_top_out_of_fold"] == 0.5     # query 1 only
    assert shift["exact_on_top_in_sample"] == 0.5       # query 2 only


def test_a_score_frame_that_misses_rows_is_refused():
    frame = _frame(n_queries=5, folds=(2,))
    with pytest.raises(ValueError, match="cover"):
        window_shift(frame, _scores(frame).iloc[1:], _scores(frame), k=3)
```

and this data test at the end of the file:

```python
@pytest.mark.data
def test_the_out_of_fold_ordering_looks_like_fold_0_not_its_own_training_folds():
    # Measured 2026-09-27: 0.8534 out-of-fold against fold 0's 0.8519, where
    # the in-sample ordering of the same three folds scores 0.9035.
    scores = load_stage2(OOF_SPLIT)
    assert len(scores) == 250_485
    assert not scores["in_sample"].any()
    matrix = pd.read_parquet(
        "data/features/train.parquet",
        columns=["query_id", "product_id", "qrel", "fold"],
    )
    joined = matrix.merge(scores, on=["query_id", "product_id"])
    assert len(joined) == 250_485
    assert set(joined["fold"]) == set(TRAIN_FOLDS)
    assert joined["query_id"].nunique() == 12_519
    assert stage2_ndcg(joined) == pytest.approx(0.8534, abs=0.0005)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_stage2_scores.py -v`
Expected: FAIL — `ImportError: cannot import name 'OOF_SPLIT' from 'src.stage2_scores'`.

- [ ] **Step 3: Implement**

In `src/stage2_scores.py`, add `import json` to the standard-library imports,
and replace the `src.ranker` import and the `DEFAULT_DIR` line with:

```python
from src.ranker import (
    EARLY_STOP_FOLD,
    REPORT_FOLD,
    TRAIN_FOLDS,
    folds,
    predict,
    train_ranker,
)

DEFAULT_DIR = Path("data/features")

# Folds 2/3/4, each scored by a Stage 2 that never saw it. Read with
# load_stage2(OOF_SPLIT); written by `--split train --out-of-fold`.
OOF_SPLIT = "train-oof"
```

After `score_split`, add `out_of_fold_scores`, `stage2_ndcg` and
`window_shift`, and replace `load_stage2` with the version below. The only
change there is that a missing out-of-fold file names `--out-of-fold` rather
than a `--split` that does not exist:

```python
def out_of_fold_scores(
    train: pd.DataFrame,
    *,
    fit_folds: Sequence[int] = TRAIN_FOLDS,
    features: Sequence[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Each training fold scored by a Stage 2 fitted on the other training folds.

    A student fine-tuned on Stage 2's window has to see the window it will
    meet at test time, and the persisted ordering cannot supply it for folds
    2/3/4: the model trained on them. Measured 2026-09-27, the in-sample
    ordering scores 0.9035 on its own folds against fold 0's 0.8519, with an
    Exact on top of 87.7% of windows against 71.1%. Fitting on the other two
    folds lands at 0.8534 and 71.2% - fold 0's distribution - and only 34.5%
    of windows keep the same ten documents, so a teacher asked about the
    in-sample windows would be answering about the wrong products.

    Every fit early-stops on fold 1, as score_split's always does.
    """
    fit_folds = [int(f) for f in fit_folds]
    reserved = {REPORT_FOLD, EARLY_STOP_FOLD} & set(fit_folds)
    if reserved:
        raise ValueError(
            f"fold(s) {sorted(reserved)} cannot be scored out-of-fold: fold "
            f"{REPORT_FOLD} is the reporting surface and fold {EARLY_STOP_FOLD} "
            "the early-stop set, and neither may enter a fit"
        )
    if len(fit_folds) < 2:
        raise ValueError(
            f"out-of-fold scoring needs at least two training folds, got {fit_folds}"
        )
    parts = [
        score_split(
            train,
            folds(train, [fold]),
            features=features,
            fit_folds=[f for f in fit_folds if f != fold],
            seed=seed,
        )
        for fold in fit_folds
    ]
    scores = pd.concat(parts, ignore_index=True)
    if scores["in_sample"].any():
        raise AssertionError(
            "an out-of-fold row is flagged in-sample: a fold was scored by a "
            "model that trained on it"
        )
    return scores


def stage2_ndcg(joined: pd.DataFrame) -> float:
    """Mean full-list NDCG of the `stage2_score` ordering over `joined`'s queries.

    `joined` needs query_id, product_id, qrel and stage2_score.
    """
    from src.metrics import ndcg_per_query

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    for q, p, r, s in zip(
        joined["query_id"], joined["product_id"], joined["qrel"], joined["stage2_score"]
    ):
        qrels.setdefault(str(q), {})[str(p)] = int(r)
        run.setdefault(str(q), {})[str(p)] = float(s)
    per_query = ndcg_per_query(run, qrels)
    return sum(per_query.values()) / len(per_query)


def window_shift(
    matrix: pd.DataFrame,
    out_of_fold: pd.DataFrame,
    in_sample: pd.DataFrame,
    *,
    k: int | None = None,
) -> dict:
    """How far the in-sample windows of the training folds sit from out-of-fold ones.

    `matrix` needs query_id, product_id and qrel for exactly the rows both
    score frames cover. Measured 2026-09-27: NDCG 0.9035 in-sample against
    0.8534 out-of-fold, an Exact on top of 87.7% of windows against 71.2%, and
    34.5% of windows holding the same documents in both.
    """
    from src.labels import label_to_qrel
    from src.rerank_window import DEFAULT_K, windows

    k = DEFAULT_K if k is None else k
    exact = label_to_qrel("E")
    out: dict = {}
    carved: dict[str, dict[str, set]] = {}
    for name, scores in (("out_of_fold", out_of_fold), ("in_sample", in_sample)):
        joined = matrix.merge(
            scores[["query_id", "product_id", "stage2_score"]],
            on=["query_id", "product_id"],
        )
        if len(joined) != len(matrix):
            raise ValueError(
                f"the {name} scores cover {len(joined):,} of {len(matrix):,} rows"
            )
        qrel = {
            (str(q), str(p)): int(r)
            for q, p, r in zip(joined["query_id"], joined["product_id"], joined["qrel"])
        }
        ws = windows(joined[["query_id", "product_id", "stage2_score"]], k=k)
        carved[name] = {w.query_id: set(w.window) for w in ws}
        out[f"ndcg_{name}"] = stage2_ndcg(joined)
        out[f"exact_on_top_{name}"] = sum(
            qrel[(w.query_id, w.window[0])] == exact for w in ws
        ) / len(ws)
    same = sum(
        carved["in_sample"][q] == documents
        for q, documents in carved["out_of_fold"].items()
    )
    out["same_window_share"] = same / len(carved["out_of_fold"])
    out["n_queries"] = len(carved["out_of_fold"])
    out["k"] = k
    return out


def load_stage2(split: str, directory: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Read a persisted Stage 2 ordering."""
    path = Path(directory) / f"stage2-{split}.parquet"
    if not path.exists():
        command = (
            "--split train --out-of-fold" if split == OOF_SPLIT else f"--split {split}"
        )
        raise FileNotFoundError(
            f"no Stage 2 scores at {path}; run "
            f"python -m src.stage2_scores {command}"
        )
    return pd.read_parquet(path)
```

Replace `_main` with the version below. The `--split` path is unchanged except
that it computes its check through `stage2_ndcg` instead of building the run
inline, and `REPORT_FOLD` now comes from the module-level import:

```python
def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-of-fold", action="store_true",
        help="score folds 2/3/4 each with a model fitted on the other two - the "
             "windows Plan 8's student trains on - and write stage2-train-oof.parquet")
    parser.add_argument(
        "--results-out", type=Path, default=Path("docs/results/stage2-oof.json"),
        help="where --out-of-fold commits its comparison with the in-sample ordering")
    args = parser.parse_args()

    train = pd.read_parquet(args.features_dir / "train.parquet")

    if args.out_of_fold:
        if args.split != "train":
            raise SystemExit("--out-of-fold scores the training folds; use --split train")
        print(f"scoring folds {list(TRAIN_FOLDS)} out-of-fold")
        scores = out_of_fold_scores(train, seed=args.seed)
        check = train.merge(scores, on=["query_id", "product_id"])
        measured = stage2_ndcg(check)
        # Measured 2026-09-27: 0.8534 here, fold 0 0.8519, and the in-sample
        # ordering of these same folds 0.9035.
        print(f"  folds {list(TRAIN_FOLDS)} out-of-fold NDCG {measured:.4f} "
              "(fold 0: 0.8519; the in-sample ordering of these folds: 0.9035)")
        if abs(measured - 0.8519) > 0.01:
            print("  WARNING: more than 0.01 from fold 0; these windows would not "
                  "look like the ones a student meets at test time")
        path = args.features_dir / f"stage2-{OOF_SPLIT}.parquet"
        scores.to_parquet(path, index=False, compression="zstd")
        print(f"wrote {len(scores):,} rows over {scores['query_id'].nunique():,} "
              f"queries to {path}")

        # The shift from the persisted, in-sample ordering of the same folds,
        # committed so the writeup can quote it.
        persisted = load_stage2("train", args.features_dir)
        rows = train.loc[train["fold"].isin(TRAIN_FOLDS)]
        shift = window_shift(rows, scores, persisted)
        fold0 = train.loc[train["fold"] == REPORT_FOLD].merge(
            persisted, on=["query_id", "product_id"]
        )
        record = {"fold0_ndcg": stage2_ndcg(fold0), **shift,
                  "train_folds": list(TRAIN_FOLDS), "seed": args.seed}
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"  in-sample {shift['ndcg_in_sample']:.4f}, Exact on top "
              f"{shift['exact_on_top_in_sample']:.1%} in-sample against "
              f"{shift['exact_on_top_out_of_fold']:.1%}; "
              f"{shift['same_window_share']:.1%} of windows unchanged -> {args.results_out}")
        return 0

    target = (
        train
        if args.split == "train"
        else pd.read_parquet(args.features_dir / "test.parquet")
    )
    print(f"scoring {len(target):,} rows of {args.split} with the frozen Stage 2 model")

    # Plan 5's own per-split fit; see the module docstring.
    scores = score_split(
        train,
        target,
        fit_folds=None if args.split == "test" else TRAIN_FOLDS,
        seed=args.seed,
    )

    # Reproduce Plan 5's headline as a check that nothing drifted. Fold 0 for
    # the train split, the whole thing for test.
    check = target.merge(scores, on=["query_id", "product_id"])
    if args.split == "train":
        check = check.loc[check["fold"] == REPORT_FOLD]
        expected, label = 0.8519, f"fold {REPORT_FOLD}"
    else:
        expected, label = 0.8579, "test"
    measured = stage2_ndcg(check)
    print(f"  {label} NDCG {measured:.4f} (Plan 5 reported {expected:.4f})")
    if abs(measured - expected) > 0.0005:
        print(
            f"  WARNING: {abs(measured - expected):.4f} away from Plan 5's number; "
            "the Stage 2 configuration has drifted and Ablation 6 would sit on "
            "a different baseline than the one Plan 5 reported"
        )

    path = args.features_dir / f"stage2-{args.split}.parquet"
    scores.to_parquet(path, index=False, compression="zstd")
    print(f"wrote {len(scores):,} rows ({int(scores['in_sample'].sum()):,} in-sample) "
          f"to {path}")
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_stage2_scores.py -v`
Expected: PASS — 26 passed (13 before this task), 4 deselected.

- [ ] **Step 5: Dump the out-of-fold ordering**

Run: `python -m src.stage2_scores --split train --out-of-fold`
Expected (about 20 s of LightGBM):

```
scoring folds [2, 3, 4] out-of-fold
  folds [2, 3, 4] out-of-fold NDCG 0.8534 (fold 0: 0.8519; the in-sample ordering of these folds: 0.9035)
wrote 250,485 rows over 12,519 queries to data/features/stage2-train-oof.parquet
  in-sample 0.9035, Exact on top 87.7% in-sample against 71.2%; 34.5% of windows unchanged -> docs/results/stage2-oof.json
```

This exact output was produced by this task's code against the real data on
2026-09-27, into a scratch directory.

No `WARNING` line. If the NDCG is not 0.8534 to four places, the Stage 2
configuration has drifted from Plan 5's. Stop and find out why before training
anything on these windows.

- [ ] **Step 6: Pin the committed comparison, and run the data test**

Append to `tests/test_stage2_scores.py`:

```python


def test_the_committed_window_shift_says_the_in_sample_windows_are_easier():
    # docs/results/stage2-oof.json, written by --out-of-fold. The out-of-fold
    # windows look like fold 0's; the in-sample ones are far easier and mostly
    # different documents.
    import json
    from pathlib import Path

    shift = json.loads(Path("docs/results/stage2-oof.json").read_text(encoding="utf-8"))
    assert shift["n_queries"] == 12_519
    assert abs(shift["ndcg_out_of_fold"] - shift["fold0_ndcg"]) < 0.01
    assert shift["ndcg_in_sample"] > shift["ndcg_out_of_fold"] + 0.03
    assert shift["exact_on_top_in_sample"] > shift["exact_on_top_out_of_fold"]
    assert shift["same_window_share"] < 0.5
```

Run: `python -m pytest tests/test_stage2_scores.py -v`
Expected: PASS — 27 passed, 4 deselected.

Run: `python -m pytest tests/test_stage2_scores.py -m data -v`
Expected: PASS, including `test_the_out_of_fold_ordering_looks_like_fold_0_not_its_own_training_folds`.

- [ ] **Step 7: Commit**

```bash
git add src/stage2_scores.py tests/test_stage2_scores.py docs/results/stage2-oof.json
git commit -m "Score the training folds out-of-fold so a student trains on test-like windows"
```

---

## Task 2: Window targets and the window fine-tune

**Files:**
- Create: `src/distill.py`
- Create: `tests/test_distill.py`

**Interfaces:**
- Consumes: `src.cross_encoder.{DEFAULT_BACKBONE, DEFAULT_MAX_LENGTH, DEFAULT_MODEL_DIR, _lookup, check_training_folds, text_maps}`, `src.rerank_window.{Window, windows, DEFAULT_K}`, `src.stage2_scores.{OOF_SPLIT, load_stage2}` (Task 1), `src.llm_rerank.{RerankCache, DEFAULT_CACHE}`, `src.stage_signals.{llm_orderings_from_cache, check_fallback_share}`, `src.clip_encoder.resolve_device`.
- Produces:
  - `src.distill.TARGETS: tuple[str, ...]` = `("labels", "llm", "hybrid")`
  - `src.distill.INITS: dict[str, str]` = `{"scratch": DEFAULT_BACKBONE, "landed": "models/cross-encoder/lambda"}`
  - `src.distill.{WINDOW_BATCH_SIZE, WINDOW_EPOCHS, WINDOW_LEARNING_RATE}` = `16, 1, 2e-5`
  - `src.distill.gains_from_frame(matrix) -> dict[tuple[str, str], float]`
  - `src.distill.window_targets(window, kind, *, gains, llm_ordering=None) -> list[float]`
  - `src.distill.window_dataset(windows_, query_text, doc_text, *, kind, gains, llm_orderings=None) -> tuple[dict[str, list], list[str]]` — the dataset and the dropped query ids
  - `src.distill.split_halves(query_ids, *, seed=0) -> tuple[frozenset[str], frozenset[str]]`
  - `src.distill.cross_fit_orderings(windows_, halves, *, fit, score) -> dict[str, list[str]]`
  - `src.distill.pairwise_accuracy(windows_, orderings, gains) -> float`
  - `src.distill.fine_tune_windows(dataset, *, init=DEFAULT_BACKBONE, out_dir=None, epochs=1, batch_size=16, learning_rate=2e-5, max_length=192, device="auto", seed=0)` — returns the model; saves it when `out_dir` is given
  - `src.distill.training_windows(features_dir=Path("data/features")) -> tuple[list[Window], pd.DataFrame]`
  - `src.distill.TRAINING_RECORD: str` = `"training.json"` — written beside every model the CLI trains: target, init, loss, epochs, batch size, seed, windows used and dropped, minutes
  - CLI: `python -m src.distill --target {labels,llm,hybrid} [--init scratch|landed|<model>] [--batch-size N] --out DIR`

**Review Focus 2, 3 and 4 live here.**
- `cross_fit_orderings` is the only way the pilot orders fold 0. It refuses overlapping halves, a window in neither half, and a scorer that returns orderings for its own training half.
- `window_dataset` drops a window with no teacher answer and returns its id. The caller decides whether the drops are a handful (`check_fallback_share`) or a scope error.
- `window_targets` gives the teacher's first choice `n` and its last `1`. The test that pins it says why in its comment, because it is the single easiest line in the plan to get backwards.

**One loss, RankNet, for every window-trained arm.** It is what Sun et al.
distilled RankGPT's permutations with. Using it for the labels arms too means
the target is the only difference between a labels arm and a teacher arm
trained on the same windows. The scratch measurements in the README show the
loss does not move the hard-negative arm (0.8556 with `LambdaLoss`, 0.8555
with `RankNetLoss`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distill.py`:

```python
import pandas as pd
import pytest

from src.distill import (
    INITS,
    TARGETS,
    cross_fit_orderings,
    gains_from_frame,
    pairwise_accuracy,
    split_halves,
    window_dataset,
    window_targets,
)
from src.rerank_window import Window

# Window order a, b, c, d. Grades: b and d Exact, c Substitute, a Irrelevant.
GAINS = {
    ("1", "a"): 0.0, ("1", "b"): 1.0, ("1", "c"): 0.1, ("1", "d"): 1.0,
    ("1", "z"): 0.0, ("2", "x"): 1.0, ("2", "y"): 0.0,
}
TEACHER = ["d", "a", "b", "c"]          # the teacher's order, best first
QUERY_TEXT = {"1": "red shoes", "2": "blue hat"}
DOC_TEXT = {d: f"product {d}" for d in "abcdxyz"}


def _window():
    return Window(query_id="1", window=("a", "b", "c", "d"), tail=("z",))


def _windows():
    return [_window(), Window(query_id="2", window=("x", "y"), tail=())]


def _targets(kind):
    return dict(
        zip(_window().window,
            window_targets(_window(), kind, gains=GAINS, llm_ordering=TEACHER))
    )


# --- the targets ------------------------------------------------------------

def test_the_targets_are_the_three_the_pilot_compares():
    assert TARGETS == ("labels", "llm", "hybrid")


def test_the_inits_are_the_two_the_pilot_measured():
    assert set(INITS) == {"scratch", "landed"}


def test_the_label_target_is_the_gain_in_window_order():
    assert window_targets(_window(), "labels", gains=GAINS) == [0.0, 1.0, 0.1, 1.0]


def test_the_teacher_s_first_choice_gets_the_highest_target():
    # Review Focus 4. The loss reads a higher label as more relevant. A target
    # built from the position itself would teach the student to invert the
    # teacher, and the result would read as "distillation does not work".
    targets = _targets("llm")
    assert targets["d"] == 4.0          # the teacher put d first
    assert targets["c"] == 1.0          # and c last
    assert targets["d"] > targets["a"] > targets["b"] > targets["c"]


def test_the_hybrid_target_lets_the_grade_beat_the_teacher():
    # The teacher put Irrelevant a second, above Exact b and Substitute c.
    targets = _targets("hybrid")
    assert targets["b"] > targets["a"]
    assert targets["c"] > targets["a"]


def test_the_hybrid_target_lets_the_teacher_order_a_grade():
    # b and d are both Exact; the labels are silent and the teacher put d first.
    targets = _targets("hybrid")
    assert targets["d"] > targets["b"]


def test_a_teacher_target_without_a_teacher_ordering_raises():
    for kind in ("llm", "hybrid"):
        with pytest.raises(KeyError, match="teacher"):
            window_targets(_window(), kind, gains=GAINS)


def test_a_teacher_ordering_that_is_not_a_permutation_raises():
    with pytest.raises(ValueError, match="permutation"):
        window_targets(_window(), "llm", gains=GAINS, llm_ordering=["d", "d", "b", "c"])


def test_an_unknown_target_raises():
    with pytest.raises(ValueError, match="target"):
        window_targets(_window(), "soft", gains=GAINS)


def test_an_unjudged_window_document_raises():
    gains = {k: v for k, v in GAINS.items() if k != ("1", "b")}
    with pytest.raises(KeyError, match="label"):
        window_targets(_window(), "labels", gains=gains)


def test_gains_are_keyed_by_string_ids():
    # data/features/train.parquet stores query_id as int64; windows hold str.
    frame = pd.DataFrame({"query_id": [1, 1], "product_id": ["a", "b"], "gain": [1.0, 0.0]})
    assert gains_from_frame(frame) == {("1", "a"): 1.0, ("1", "b"): 0.0}


# --- Review Focus 3: a window the teacher did not answer --------------------

def test_a_window_without_a_teacher_answer_is_dropped_and_reported():
    dataset, dropped = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="llm", gains=GAINS,
        llm_orderings={"1": TEACHER},
    )
    assert dropped == ["2"]
    assert dataset["query"] == ["red shoes"]


def test_a_dropped_window_is_never_filled_with_the_stage_2_order():
    # src.llm_rerank falls back to the Stage 2 order on a malformed answer. A
    # student taught that would be learning Stage 2 under the teacher's name.
    dataset, _ = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="hybrid", gains=GAINS,
        llm_orderings={"1": TEACHER},
    )
    assert len(dataset["docs"]) == len(dataset["labels"]) == 1
    assert ["product x", "product y"] not in dataset["docs"]


def test_the_label_target_drops_nothing():
    dataset, dropped = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="labels", gains=GAINS
    )
    assert dropped == []
    assert dataset["labels"] == [[0.0, 1.0, 0.1, 1.0], [1.0, 0.0]]


def test_the_dataset_rows_line_up():
    dataset, _ = window_dataset(
        _windows(), QUERY_TEXT, DOC_TEXT, kind="labels", gains=GAINS
    )
    assert dataset["docs"][0] == ["product a", "product b", "product c", "product d"]
    assert all(len(d) == len(t) for d, t in zip(dataset["docs"], dataset["labels"]))


def test_an_unknown_target_is_refused_before_anything_is_dropped():
    with pytest.raises(ValueError, match="target"):
        window_dataset(_windows(), QUERY_TEXT, DOC_TEXT, kind="soft", gains=GAINS)


# --- Review Focus 2: the fold-0 pilot is cross-fitted -----------------------

def test_the_halves_are_disjoint_and_cover_every_query():
    first, second = split_halves([str(i) for i in range(11)])
    assert not first & second
    assert first | second == {str(i) for i in range(11)}
    assert abs(len(first) - len(second)) <= 1


def test_the_halves_are_fixed_by_the_seed():
    ids = [str(i) for i in range(50)]
    assert split_halves(ids, seed=0) == split_halves(ids, seed=0)
    assert split_halves(ids, seed=0) != split_halves(ids, seed=1)


def test_the_halves_do_not_depend_on_input_order_or_id_type():
    ids = list(range(50))
    assert split_halves(ids) == split_halves([str(i) for i in reversed(ids)])


def test_no_window_is_ordered_by_a_model_that_trained_on_it():
    many = [Window(query_id=str(i), window=("a", "b"), tail=()) for i in range(10)]
    trained_on = []

    def fit(ws):
        ids = frozenset(w.query_id for w in ws)
        trained_on.append(ids)
        return ids                      # the "model" is the set it trained on

    def score(model, ws):
        assert not model & {w.query_id for w in ws}
        return {w.query_id: list(reversed(w.window)) for w in ws}

    halves = split_halves([w.query_id for w in many])
    out = cross_fit_orderings(many, halves, fit=fit, score=score)
    assert set(out) == {w.query_id for w in many}
    assert len(trained_on) == 2
    assert not trained_on[0] & trained_on[1]


def test_overlapping_halves_are_refused():
    many = [Window(query_id=str(i), window=("a",), tail=()) for i in range(4)]
    with pytest.raises(ValueError, match="share"):
        cross_fit_orderings(
            many, ({"0", "1", "2"}, {"2", "3"}),
            fit=lambda ws: None, score=lambda m, ws: {},
        )


def test_a_window_in_neither_half_is_refused():
    many = [Window(query_id=str(i), window=("a",), tail=()) for i in range(4)]
    with pytest.raises(ValueError, match="neither half"):
        cross_fit_orderings(
            many, ({"0"}, {"1"}), fit=lambda ws: None, score=lambda m, ws: {}
        )


def test_a_scorer_that_orders_the_training_half_is_caught():
    many = [Window(query_id=str(i), window=("a",), tail=()) for i in range(4)]
    with pytest.raises(AssertionError, match="outside its half"):
        cross_fit_orderings(
            many, ({"0", "1"}, {"2", "3"}),
            fit=lambda ws: None,
            score=lambda m, ws: {w.query_id: ["a"] for w in many},
        )


# --- the diagnostic ---------------------------------------------------------

def test_pairwise_accuracy_of_the_ideal_order_is_one():
    ideal = {"1": ["b", "d", "c", "a"], "2": ["x", "y"]}
    assert pairwise_accuracy(_windows(), ideal, GAINS) == 1.0


def test_pairwise_accuracy_of_the_reversed_order_is_zero():
    worst = {"1": ["a", "c", "d", "b"], "2": ["y", "x"]}
    assert pairwise_accuracy(_windows(), worst, GAINS) == 0.0


def test_pairwise_accuracy_ignores_pairs_inside_a_grade():
    # b and d are both Exact, so their order is neither right nor wrong.
    one = {"1": ["b", "d", "c", "a"], "2": ["x", "y"]}
    two = {"1": ["d", "b", "c", "a"], "2": ["x", "y"]}
    assert pairwise_accuracy(_windows(), one, GAINS) == pairwise_accuracy(
        _windows(), two, GAINS
    )


def test_pairwise_accuracy_counts_each_mixed_pair_once():
    # Window 1 has five mixed-grade pairs (six, less b~d) and window 2 one.
    # Getting only x above y right is 1 of 6.
    order = {"1": ["a", "c", "d", "b"], "2": ["x", "y"]}
    assert pairwise_accuracy(_windows(), order, GAINS) == pytest.approx(1 / 6)


def test_pairwise_accuracy_needs_an_ordering_for_every_window():
    with pytest.raises(KeyError, match="ordering"):
        pairwise_accuracy(_windows(), {"1": ["b", "d", "c", "a"]}, GAINS)


@pytest.mark.data
def test_the_training_windows_are_carved_out_of_fold():
    # Review Focus 1. Measured 2026-09-27: 12,519 windows, and only 34.5% of
    # them hold the same ten documents as the in-sample ordering's windows.
    from src.distill import training_windows
    from src.rerank_window import windows
    from src.stage2_scores import load_stage2

    windows_, matrix = training_windows()
    assert len(windows_) == 12_519
    assert set(matrix["fold"]) == {2, 3, 4}
    assert min(len(w.window) for w in windows_) >= 8
    in_sample = load_stage2("train")
    in_sample = in_sample.loc[in_sample["in_sample"]]
    carved = {w.query_id: set(w.window) for w in windows(in_sample, k=10)}
    same = sum(carved[w.query_id] == set(w.window) for w in windows_)
    assert same / len(windows_) == pytest.approx(0.345, abs=0.01)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_distill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill'`.

- [ ] **Step 3: Implement**

Create `src/distill.py`:

```python
"""Training a cross-encoder on Stage 2's window, from the labels or from the LLM.

Plan 8 asks whether the LLM listwise arm's gain - +0.0276 NDCG on the full
test split, at 4.7 s a query - can be moved into a cross-encoder that answers
in about 25 ms. This module owns the three things distillation needs:

  * **the training windows**, carved by the out-of-fold Stage 2
    (src.stage2_scores.out_of_fold_scores). The in-sample ordering of folds
    2/3/4 scores 0.9035 against fold 0's 0.8519, and only 34.5% of its windows
    hold the same ten documents as the out-of-fold ones;
  * **the targets**, one per window document, higher meaning better: the
    labels, the teacher's ordering, or the labels with the teacher breaking
    ties inside a grade;
  * **the cross-fit** the fold-0 pilot needs, because fold 0 holds the only
    free teacher answers and is also the reporting surface.

A window the teacher did not answer is dropped from a teacher-target training
set, never filled with Stage 2's order. src.llm_rerank falls back to that order
on a malformed answer, and a student taught it would be learning Stage 2 under
the teacher's name.

Every window-trained arm uses one loss, RankNet - what Sun et al. distilled
RankGPT's permutations with - so the target is the only thing that differs
between them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.cross_encoder import (
    DEFAULT_BACKBONE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_DIR,
    _lookup,
)

TARGETS: tuple[str, ...] = ("labels", "llm", "hybrid")

# Where a fine-tune starts. "landed" is Plan 6's LambdaLoss arm, already
# trained on every judged pair of folds 2/3/4.
INITS: dict[str, str] = {
    "scratch": DEFAULT_BACKBONE,
    "landed": str(DEFAULT_MODEL_DIR / "lambda"),
}

WINDOW_BATCH_SIZE = 16
WINDOW_EPOCHS = 1
WINDOW_LEARNING_RATE = 2e-5

# Written beside every model `python -m src.distill` trains.
TRAINING_RECORD = "training.json"


def gains_from_frame(matrix: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(query_id, product_id) -> gain, with both ids as strings, as windows hold them."""
    return {
        (str(q), str(p)): float(g)
        for q, p, g in zip(matrix["query_id"], matrix["product_id"], matrix["gain"])
    }


def _gain(gains: Mapping, query_id: str, document: str) -> float:
    try:
        return float(gains[(query_id, document)])
    except KeyError:
        raise KeyError(
            f"no label for ({query_id!r}, {document!r}); a window document "
            "without a judgement cannot be trained on or scored"
        ) from None


def window_targets(
    window,
    kind: str,
    *,
    gains: Mapping,
    llm_ordering: Sequence[str] | None = None,
) -> list[float]:
    """One target per window document, in window order. Higher is better.

    `labels` is the gain. `llm` is n for the teacher's first choice down to 1
    for its last: the loss reads a higher label as more relevant, so a target
    built from the position itself would teach the student to invert the
    teacher. `hybrid` orders by grade first and lets the teacher order the
    documents inside a grade - the only place the labels are silent.
    """
    if kind not in TARGETS:
        raise ValueError(f"unknown target {kind!r}; expected one of {TARGETS}")
    documents = list(window.window)
    labels = [_gain(gains, window.query_id, d) for d in documents]
    if kind == "labels":
        return labels
    if llm_ordering is None:
        raise KeyError(
            f"query {window.query_id}: no teacher ordering for a {kind!r} "
            "target. Drop the window; never substitute Stage 2's order."
        )
    if sorted(llm_ordering) != sorted(documents):
        raise ValueError(
            f"query {window.query_id}: the teacher ordering is not a "
            "permutation of its window"
        )
    position = {d: i for i, d in enumerate(llm_ordering)}
    n = len(documents)
    if kind == "llm":
        return [float(n - position[d]) for d in documents]
    grade = dict(zip(documents, labels))
    order = sorted(documents, key=lambda d: (-grade[d], position[d]))
    rank = {d: i for i, d in enumerate(order)}
    return [float(n - rank[d]) for d in documents]


def window_dataset(
    windows_: Sequence,
    query_text: Mapping,
    doc_text: Mapping,
    *,
    kind: str,
    gains: Mapping,
    llm_orderings: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, list], list[str]]:
    """(query, docs, labels) per window, and the query ids dropped for want of a teacher.

    A `labels` target drops nothing. A teacher target drops every window the
    teacher did not answer and says which - Review Focus 3.
    """
    if kind not in TARGETS:
        raise ValueError(f"unknown target {kind!r}; expected one of {TARGETS}")
    orderings = dict(llm_orderings or {})
    out: dict[str, list] = {"query": [], "docs": [], "labels": []}
    dropped: list[str] = []
    for w in windows_:
        ordering = orderings.get(w.query_id)
        if kind != "labels" and ordering is None:
            dropped.append(w.query_id)
            continue
        out["query"].append(_lookup(query_text, w.query_id, "query text"))
        out["docs"].append([_lookup(doc_text, d, "document text") for d in w.window])
        out["labels"].append(
            window_targets(w, kind, gains=gains, llm_ordering=ordering)
        )
    return out, dropped


def split_halves(
    query_ids: Iterable, *, seed: int = 0
) -> tuple[frozenset[str], frozenset[str]]:
    """Two disjoint halves of a query set, fixed by the seed alone."""
    ids = sorted({str(q) for q in query_ids})
    if len(ids) < 2:
        raise ValueError("a cross-fit needs at least two queries")
    shuffled = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    middle = len(ids) // 2
    return frozenset(shuffled[:middle]), frozenset(shuffled[middle:])


def cross_fit_orderings(
    windows_: Sequence,
    halves: tuple[Iterable[str], Iterable[str]],
    *,
    fit: Callable[[list], object],
    score: Callable[[object, list], Mapping[str, Sequence[str]]],
) -> dict[str, list[str]]:
    """Every window ordered by a model that never trained on it.

    Train on one half, order the other, swap. Fold 0 is the only train fold
    with free teacher answers and also the reporting surface, so the pilot
    trains there - and this is the guard that keeps it honest.
    """
    first, second = (frozenset(str(q) for q in half) for half in halves)
    shared = first & second
    if shared:
        raise ValueError(
            f"the halves share {len(shared)} queries; a query in both would "
            "be scored by a model that trained on it"
        )
    stray = {w.query_id for w in windows_} - (first | second)
    if stray:
        raise ValueError(
            f"{len(stray)} windows are in neither half, so no model would score them"
        )
    out: dict[str, list[str]] = {}
    for train_ids, score_ids in ((first, second), (second, first)):
        model = fit([w for w in windows_ if w.query_id in train_ids])
        scored = score(model, [w for w in windows_ if w.query_id in score_ids])
        outside = set(scored) - score_ids
        if outside:
            raise AssertionError(
                f"the scorer ordered {len(outside)} queries outside its half"
            )
        out.update({q: list(order) for q, order in scored.items()})
    return out


def pairwise_accuracy(
    windows_: Sequence, orderings: Mapping[str, Sequence[str]], gains: Mapping
) -> float:
    """Share of mixed-grade window pairs an ordering puts the right way round.

    Pairs inside one grade are skipped: NDCG does not care how they fall.
    Measured on fold 0 before this plan was written: the LLM 0.7404, the
    cross-encoder 0.6555, Stage 2 0.6373.
    """
    right = total = 0
    for w in windows_:
        if w.query_id not in orderings:
            raise KeyError(f"query {w.query_id}: no ordering to judge")
        position = {d: i for i, d in enumerate(orderings[w.query_id])}
        documents = list(w.window)
        for i, a in enumerate(documents):
            for b in documents[i + 1:]:
                grade_a = _gain(gains, w.query_id, a)
                grade_b = _gain(gains, w.query_id, b)
                if grade_a == grade_b:
                    continue
                better, worse = (a, b) if grade_a > grade_b else (b, a)
                right += position[better] < position[worse]
                total += 1
    if total == 0:
        raise ValueError("no mixed-grade pairs to judge")
    return right / total


def fine_tune_windows(
    dataset: Mapping[str, list],
    *,
    init: str = DEFAULT_BACKBONE,
    out_dir: Path | None = None,
    epochs: int = WINDOW_EPOCHS,
    batch_size: int = WINDOW_BATCH_SIZE,
    learning_rate: float = WINDOW_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str = "auto",
    seed: int = 0,
):
    """Fine-tune a cross-encoder with RankNet on (query, docs, labels) windows."""
    import tempfile

    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import RankNetLoss

    from src.clip_encoder import resolve_device

    if not dataset.get("query"):
        raise ValueError("an empty training set; nothing to fine-tune on")
    resolved = resolve_device(device)
    model = CrossEncoder(str(init), num_labels=1, device=resolved, max_length=max_length)
    with tempfile.TemporaryDirectory() as scratch:
        arguments = CrossEncoderTrainingArguments(
            output_dir=scratch,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=resolved.startswith("cuda"),
            save_strategy="no",
            logging_steps=200,
            report_to=[],
            seed=seed,
            disable_tqdm=True,
        )
        CrossEncoderTrainer(
            model=model,
            args=arguments,
            train_dataset=Dataset.from_dict(dict(dataset)),
            loss=RankNetLoss(model),
        ).train()
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
    return model


def training_windows(features_dir: Path = Path("data/features")) -> tuple[list, pd.DataFrame]:
    """The out-of-fold windows of folds 2/3/4, and the judged rows behind them."""
    from src.cross_encoder import check_training_folds
    from src.ranker import TRAIN_FOLDS
    from src.rerank_window import DEFAULT_K, windows
    from src.stage2_scores import OOF_SPLIT, load_stage2

    features_dir = Path(features_dir)
    matrix = pd.read_parquet(
        features_dir / "train.parquet",
        columns=["query_id", "product_id", "gain", "fold"],
    )
    matrix = matrix.loc[matrix["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    check_training_folds(matrix)
    scores = load_stage2(OOF_SPLIT, features_dir)
    joined = matrix[["query_id", "product_id"]].merge(
        scores[["query_id", "product_id", "stage2_score"]],
        on=["query_id", "product_id"],
    )
    if len(joined) != len(matrix):
        raise ValueError(
            f"the out-of-fold ordering covers {len(joined):,} of {len(matrix):,} "
            "training rows; re-run python -m src.stage2_scores --split train --out-of-fold"
        )
    return windows(joined, k=DEFAULT_K), matrix


def _main() -> int:
    import time

    from src.cross_encoder import text_maps
    from src.llm_rerank import DEFAULT_CACHE, RerankCache
    from src.stage_signals import check_fallback_share, llm_orderings_from_cache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    parser.add_argument(
        "--init", default="scratch",
        help=f"one of {sorted(INITS)}, or any cross-encoder name or path "
             "(the capacity arm passes BAAI/bge-reranker-base)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=WINDOW_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=WINDOW_BATCH_SIZE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--products", type=Path, default=Path("data/combined/products.parquet"))
    parser.add_argument("--judgements", type=Path, default=Path("data/combined/judgements.parquet"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    windows_, matrix = training_windows(args.features_dir)
    query_text, doc_text = text_maps(matrix, args.products, args.judgements)
    query_text = {str(k): v for k, v in query_text.items()}
    gains = gains_from_frame(matrix)

    orderings = None
    if args.target != "labels":
        orderings, missing = llm_orderings_from_cache(
            windows_, query_text, RerankCache(args.cache)
        )
        if len(missing) == len(windows_):
            raise SystemExit(
                "the cache holds no teacher answer for any training window; "
                "Plan 8's paid pass (Phase 2) has not run"
            )
        check_fallback_share(missing, len(windows_))

    dataset, dropped = window_dataset(
        windows_, query_text, doc_text,
        kind=args.target, gains=gains, llm_orderings=orderings,
    )
    print(f"{len(dataset['query']):,} training windows from folds 2/3/4, target "
          f"{args.target!r}, init {args.init!r}; {len(dropped):,} dropped for want "
          "of a teacher answer")
    started = time.time()
    fine_tune_windows(
        dataset, init=INITS.get(args.init, args.init), out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
    )
    minutes = (time.time() - started) / 60
    # What trained this model, beside it: the report records it with the
    # arm's numbers, so a results file says which target made which arm.
    (args.out / TRAINING_RECORD).write_text(json.dumps({
        "target": args.target, "init": args.init, "loss": "RankNetLoss",
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "n_windows": len(dataset["query"]), "n_dropped": len(dropped),
        "minutes": minutes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"trained in {minutes:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_distill.py -v`
Expected: PASS — 28 passed, 1 deselected.

Run: `python -m pytest -m data tests/test_distill.py -v`
Expected: PASS — `test_the_training_windows_are_carved_out_of_fold` finds 12,519
windows of 8–10 documents, 34.5% of them the same set as the in-sample
ordering's windows.

- [ ] **Step 5: Commit**

```bash
git add src/distill.py tests/test_distill.py
git commit -m "Add window targets, the cross-fit and the RankNet window fine-tune for distillation"
```

---

## Task 3: The free arms, the pilot and the gate

**Files:**
- Create: `src/distill_report.py`
- Create: `tests/test_distill_report.py`
- Modify: `src/distill.py` (add `gate_choice` and `--from-gate`: the gate's choice, read back)
- Modify: `tests/test_distill.py`
- Create (by running): `docs/results/distill-pilot.json`, `docs/results/distill.json`

**Interfaces:**
- Consumes: `src.distill.*` (Task 2), `src.stage_signals.{scope_windows, load_signals, llm_orderings_from_cache, check_fallback_share}`, `src.blend_report.{single_stage_orderings, per_query_ndcg}`, `src.cross_encoder.{rerank, load_reranker, measure_latency, text_maps}`, `src.fine_rank_report.check_same_queries`, `src.rank_report.{evaluate_arm, compare, format_table, qrels_from_frame}`, `src.floor.random_floor`, `src.llm_rerank.{RerankCache, DEFAULT_CACHE}`.
- Produces:
  - `src.distill_report.REFERENCE_ARMS` = `("stage2", "stage2+ce", "stage2+llm")`
  - `src.distill_report.MODEL_ARMS: dict[str, Path]` — `stage2+ce_windows`, `stage2+ce_bge`, `stage2+student`
  - `src.distill_report.FREE_ARMS` = `("stage2+ce_windows", "stage2+ce_bge")`
  - `src.distill_report.{TRAIN_COMMANDS, LANDED_MODEL, PILOT_EPOCHS, PUBLISHED_REFERENCE, DEFAULT_OUT, GATE_RULE}`
  - `src.distill_report.pilot_arm(init, target) -> str`
  - `src.distill_report.paid_run_gate(comparisons) -> dict` — `{"rule", "passed", "choice", "because"}`; Phase 3's `gate_choice` reads `choice`
  - `src.distill_report.share_of_teacher_gain(arm, stage2, teacher) -> float`
  - `src.distill_report.resolve_model_arms(names, root=Path(".")) -> dict[str, Path]`
  - `src.distill_report.check_split(split, final) -> None`
  - `src.distill_report.check_reference(measured, published, tol=5e-4) -> None`
  - `src.distill_report.training_records(models) -> dict[str, dict]` — each model arm's `training.json`; refuses a model directory without one
  - `src.distill.gate_choice(payload) -> tuple[str, str]` — the `(init, target)` a passed gate chose; refuses a gate that did not pass
  - CLI: `python -m src.distill_report --pilot`; `python -m src.distill_report [--arms …]`; `python -m src.distill_report --split test --final --arms …` (Task 6); `python -m src.distill --from-gate docs/results/distill-pilot.json --out DIR` (Task 5)

**The gate half of Review Focus 5 lives in `paid_run_gate`.** Three wrong wins
are easy to count by accident, and each has a test:
- a teacher arm beating Stage 2;
- a teacher arm from the landed start beating a labels arm from scratch — which measures the start, not the target;
- a hybrid arm beating the `llm` arm.

`choice` is the win with the largest lower bound. `gate_choice` is the only
thing that reads it: `python -m src.distill --from-gate` trains exactly that
pair, so Phase 3 never picks a student by hand. It lands here, beside the
gate, so it exists whichever way the gate goes.

**Every model arm must say what trained it.** `training_records` reads the
`training.json` that `python -m src.distill` writes beside each model and puts
it in the results file. A model directory without one — copied in by hand,
say — is refused. The committed test then checks that the student's record
names exactly the gate's choice.

**`check_reference` stops the report if a reference arm drifts.** The arms
here are scored on the windows in Plan 7's signal frame. The report re-derives
Stage 2, the landed cross-encoder and the LLM from that frame, and refuses to
continue unless all three reproduce Plan 6's published NDCG to 5e-4. If
`stage2-train.parquet` were ever re-dumped with a different fit, every delta
below would be against a baseline nobody published.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_distill_report.py`:

```python
import pytest

from src.distill_report import (
    FREE_ARMS,
    GATE_RULE,
    MODEL_ARMS,
    PILOT_EPOCHS,
    REFERENCE_ARMS,
    check_reference,
    check_split,
    paid_run_gate,
    pilot_arm,
    resolve_model_arms,
    share_of_teacher_gain,
    training_records,
)


def _row(arm, baseline, low, point=None, high=None):
    point = low + 0.001 if point is None else point
    high = point + 0.001 if high is None else high
    return {
        "arm": arm,
        "baseline": baseline,
        "delta": {"point": point, "low": low, "high": high},
        "significant": not (low <= 0.0 <= high),
    }


# --- the pre-registered gate ------------------------------------------------

def test_a_teacher_target_beating_the_labels_from_the_same_start_passes():
    gate = paid_run_gate([
        _row(pilot_arm("landed", "hybrid"), pilot_arm("landed", "labels"), low=0.0004),
    ])
    assert gate["passed"] is True
    assert gate["choice"] == {"init": "landed", "target": "hybrid"}
    assert gate["because"] == ["pilot:landed:hybrid"]


def test_a_tie_does_not_pass():
    # Both pilots measured before this plan was written look like this.
    gate = paid_run_gate([
        _row(pilot_arm("scratch", "llm"), pilot_arm("scratch", "labels"),
             low=-0.0006, point=0.0009, high=0.0022),
        _row(pilot_arm("landed", "llm"), pilot_arm("landed", "labels"),
             low=-0.0031, point=-0.0016, high=-0.0003),
    ])
    assert gate["passed"] is False
    assert gate["choice"] is None
    assert gate["because"] == []


def test_beating_stage_2_is_not_beating_the_labels():
    gate = paid_run_gate([_row(pilot_arm("scratch", "llm"), "stage2", low=0.01)])
    assert gate["passed"] is False


def test_a_win_across_starting_points_does_not_count():
    # The landed start is a better model before any target is applied; its
    # teacher arm beating a from-scratch labels arm measures the start.
    gate = paid_run_gate([
        _row(pilot_arm("landed", "llm"), pilot_arm("scratch", "labels"), low=0.005),
    ])
    assert gate["passed"] is False


def test_only_a_teacher_target_against_the_labels_counts():
    gate = paid_run_gate([
        _row(pilot_arm("scratch", "hybrid"), pilot_arm("scratch", "llm"), low=0.005),
        _row(pilot_arm("scratch", "labels"), pilot_arm("scratch", "labels"), low=0.005),
    ])
    assert gate["passed"] is False


def test_the_choice_is_the_win_with_the_largest_lower_bound():
    gate = paid_run_gate([
        _row(pilot_arm("scratch", "llm"), pilot_arm("scratch", "labels"), low=0.0002),
        _row(pilot_arm("landed", "hybrid"), pilot_arm("landed", "labels"), low=0.0009),
    ])
    assert gate["choice"] == {"init": "landed", "target": "hybrid"}
    assert gate["because"] == ["pilot:landed:hybrid", "pilot:scratch:llm"]


def test_the_gate_records_its_rule():
    assert paid_run_gate([])["rule"] == GATE_RULE
    assert "wholly above zero" in GATE_RULE
    assert "same starting point" in GATE_RULE


def test_the_pilot_arm_name_carries_its_start_and_target():
    assert pilot_arm("landed", "hybrid") == "pilot:landed:hybrid"


def test_each_pilot_start_has_an_epoch_count():
    from src.distill import INITS

    assert set(PILOT_EPOCHS) == set(INITS)


# --- what the report computes -----------------------------------------------

def test_the_share_of_the_teacher_s_gain_is_zero_at_stage_2_and_one_at_the_teacher():
    assert share_of_teacher_gain(0.8579, 0.8579, 0.8855) == 0.0
    assert share_of_teacher_gain(0.8855, 0.8579, 0.8855) == pytest.approx(1.0)
    # The landed cross-encoder on the full test split: +0.0037 of +0.0276.
    assert share_of_teacher_gain(0.8616, 0.8579, 0.8855) == pytest.approx(0.134, abs=1e-3)


def test_a_teacher_that_does_not_beat_stage_2_has_no_gain_to_share():
    with pytest.raises(ValueError, match="no gain"):
        share_of_teacher_gain(0.85, 0.86, 0.86)


def test_the_reference_arms_are_the_three_stages():
    assert REFERENCE_ARMS == ("stage2", "stage2+ce", "stage2+llm")


def test_the_free_arms_need_no_teacher_answers():
    assert set(FREE_ARMS) <= set(MODEL_ARMS)
    assert "stage2+student" in MODEL_ARMS
    assert "stage2+student" not in FREE_ARMS


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        resolve_model_arms(["stage2+magic"])


def test_a_missing_model_names_the_command_that_trains_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.distill"):
        resolve_model_arms(["stage2+ce_windows"], root=tmp_path)


def test_a_present_model_resolves(tmp_path):
    (tmp_path / MODEL_ARMS["stage2+ce_bge"]).mkdir(parents=True)
    assert resolve_model_arms(["stage2+ce_bge"], root=tmp_path) == {
        "stage2+ce_bge": tmp_path / MODEL_ARMS["stage2+ce_bge"]
    }


def test_the_test_split_needs_final():
    with pytest.raises(SystemExit, match="--final"):
        check_split("test", final=False)
    check_split("test", final=True)
    check_split("train", final=False)


_PUBLISHED = {"stage2": 0.8519, "stage2+ce": 0.8587, "stage2+llm": 0.8814}


def test_reference_arms_that_reproduce_plan_6_pass():
    check_reference({"stage2": 0.85192, "stage2+ce": 0.8587, "stage2+llm": 0.88138},
                    _PUBLISHED)


def test_a_drifted_reference_arm_raises():
    # A re-dumped Stage 2 would carve different windows; every delta after
    # that is against a baseline nobody published.
    with pytest.raises(ValueError, match="drifted"):
        check_reference({"stage2": 0.8559, "stage2+ce": 0.8587, "stage2+llm": 0.8814},
                        _PUBLISHED)


def test_every_model_arm_says_what_trained_it(tmp_path):
    path = tmp_path / "windows"
    path.mkdir()
    (path / "training.json").write_text('{"target": "labels", "init": "scratch"}')
    assert training_records({"stage2+ce_windows": path}) == {
        "stage2+ce_windows": {"target": "labels", "init": "scratch"}
    }


def test_a_model_with_no_training_record_is_refused(tmp_path):
    # A directory someone copied in by hand could be any model at all.
    with pytest.raises(FileNotFoundError, match="python -m src.distill"):
        training_records({"stage2+ce_windows": tmp_path})
```

In `tests/test_distill.py`, add `gate_choice` to the `from src.distill import (…)`
list, and append:

```python
# --- the student trains what the gate chose (Phase 3) -----------------------

def _pilot_payload(passed, choice):
    return {"gate": {"passed": passed, "choice": choice, "because": [], "rule": "..."}}


def test_the_student_trains_what_the_gate_chose():
    payload = _pilot_payload(True, {"init": "landed", "target": "hybrid"})
    assert gate_choice(payload) == ("landed", "hybrid")


def test_a_gate_that_did_not_pass_trains_no_student():
    with pytest.raises(ValueError, match="free arms are the result"):
        gate_choice(_pilot_payload(False, None))


def test_a_gate_without_a_gate_block_trains_no_student():
    with pytest.raises(ValueError, match="did not pass"):
        gate_choice({})


def test_a_gate_choice_that_is_not_a_teacher_target_is_refused():
    with pytest.raises(ValueError, match="teacher target"):
        gate_choice(_pilot_payload(True, {"init": "landed", "target": "labels"}))
    with pytest.raises(ValueError, match="teacher target"):
        gate_choice(_pilot_payload(True, {"init": "bge", "target": "llm"}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_distill_report.py tests/test_distill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.distill_report'`, and `ImportError: cannot import name 'gate_choice' from 'src.distill'`.

- [ ] **Step 3: Implement**

Create `src/distill_report.py`:

```python
"""Plan 8's report: can the LLM teach a cross-encoder, and what is free?

Three modes, one module, because all three score the same windows the same way:

  * ``--pilot``: fold 0, cross-fitted in halves. The same student is trained
    with the labels, the teacher's ordering and the hybrid target on the same
    queries, from two starting points, and the pre-registered gate on the paid
    teacher pass is decided (docs/results/distill-pilot.json).
  * default: every trained arm on fold 0, beside the three stages it sits
    between - Stage 2, the landed cross-encoder and the LLM
    (docs/results/distill.json).
  * ``--split test --final``: the same arms on all 8,956 test queries, once
    (docs/results/distill-test.json).

The reference arms are read from Plan 7's signal frame and checked against
the NDCG Plan 6 published for them, so every arm here is scored on exactly the
windows the published numbers came from.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

REFERENCE_ARMS: tuple[str, ...] = ("stage2", "stage2+ce", "stage2+llm")

MODEL_ARMS: dict[str, Path] = {
    "stage2+ce_windows": Path("models/cross-encoder/windows"),
    "stage2+ce_bge": Path("models/cross-encoder/bge"),
    "stage2+student": Path("models/cross-encoder/student"),
}
FREE_ARMS: tuple[str, ...] = ("stage2+ce_windows", "stage2+ce_bge")

TRAIN_COMMANDS: dict[str, str] = {
    "stage2+ce_windows": (
        "python -m src.distill --target labels --out models/cross-encoder/windows"
    ),
    "stage2+ce_bge": (
        "python -m src.distill --target labels --init BAAI/bge-reranker-base "
        "--batch-size 8 --out models/cross-encoder/bge"
    ),
    "stage2+student": (
        "python -m src.distill --from-gate docs/results/distill-pilot.json "
        "--out models/cross-encoder/student"
    ),
}

# The landed cross-encoder's own directory, loaded only to time it beside the
# new arms on the same machine state.
LANDED_MODEL = Path("models/cross-encoder/lambda")

# The pilot trains on ~2,065 queries a half. From the ms-marco checkpoint that
# is two epochs; from the landed arm, which has already seen folds 2/3/4, one.
PILOT_EPOCHS: dict[str, int] = {"scratch": 2, "landed": 1}

PUBLISHED_REFERENCE: dict[str, Path] = {
    "train": Path("docs/results/fine-rank.json"),
    "test": Path("docs/results/fine-rank-test-full.json"),
}
DEFAULT_OUT: dict[str, Path] = {
    "pilot": Path("docs/results/distill-pilot.json"),
    "train": Path("docs/results/distill.json"),
    "test": Path("docs/results/distill-test.json"),
}

GATE_RULE = (
    "Pay for the teacher's answers on folds 2/3/4 only if, in the fold-0 "
    "cross-fitted pilot, a teacher target (llm or hybrid) beats the labels "
    "target from the same starting point with a paired 95% interval wholly "
    "above zero. Written into Plan 8 before the committed pilot ran."
)


def pilot_arm(init: str, target: str) -> str:
    """The pilot arm's name: `pilot:<init>:<target>`."""
    return f"pilot:{init}:{target}"


def paid_run_gate(comparisons: Sequence[Mapping]) -> dict:
    """The pre-registered gate on Phase 2's paid pass.

    Only a teacher target against the labels target from the same init
    counts. A teacher target beating Stage 2, or beating a labels arm that
    started somewhere else, says nothing about whether the teacher is a
    better target than the labels. `choice` is the win with the largest
    lower bound, so Phase 3's student is not picked by hand.
    """
    wins = []
    for row in comparisons:
        arm = str(row["arm"]).split(":")
        baseline = str(row["baseline"]).split(":")
        if len(arm) != 3 or len(baseline) != 3:
            continue
        if arm[0] != "pilot" or baseline[0] != "pilot" or arm[1] != baseline[1]:
            continue
        if arm[2] not in ("llm", "hybrid") or baseline[2] != "labels":
            continue
        if row["delta"]["low"] > 0:
            wins.append((row["delta"]["low"], arm[1], arm[2], str(row["arm"])))
    wins.sort(reverse=True)
    return {
        "rule": GATE_RULE,
        "passed": bool(wins),
        "choice": {"init": wins[0][1], "target": wins[0][2]} if wins else None,
        "because": [name for *_, name in wins],
    }


def share_of_teacher_gain(arm: float, stage2: float, teacher: float) -> float:
    """How much of the teacher's gain over Stage 2 an arm keeps: 0 at Stage 2, 1 at the teacher."""
    if teacher <= stage2:
        raise ValueError(
            "the teacher does not beat Stage 2 on these queries, so it has no "
            "gain to share"
        )
    return (arm - stage2) / (teacher - stage2)


def resolve_model_arms(names: Sequence[str], root: Path = Path(".")) -> dict[str, Path]:
    """Model directory per requested arm; a missing one names the command that trains it."""
    out: dict[str, Path] = {}
    for name in names:
        if name not in MODEL_ARMS:
            raise ValueError(f"unknown arm {name!r}; expected one of {sorted(MODEL_ARMS)}")
        path = Path(root) / MODEL_ARMS[name]
        if not path.exists():
            raise FileNotFoundError(
                f"no model for {name} at {path}; train it with: {TRAIN_COMMANDS[name]}"
            )
        out[name] = path
    return out


def training_records(models: Mapping[str, Path]) -> dict[str, dict]:
    """What trained each model arm, from the record `python -m src.distill` writes beside it."""
    from src.distill import TRAINING_RECORD

    out: dict[str, dict] = {}
    for name, path in models.items():
        record = Path(path) / TRAINING_RECORD
        if not record.exists():
            raise FileNotFoundError(
                f"{name} at {path} has no {TRAINING_RECORD}, so nothing says what "
                f"trained it; retrain with: {TRAIN_COMMANDS[name]}"
            )
        out[name] = json.loads(record.read_text(encoding="utf-8"))
    return out


def check_split(split: str, final: bool) -> None:
    """The test split is touched once, behind --final."""
    if split == "test" and not final:
        raise SystemExit(
            "refusing to touch the test split without --final. The target, the "
            "starting point, the backbone and the epochs are all chosen on fold 0."
        )


def check_reference(
    measured: Mapping[str, float], published: Mapping[str, float], tol: float = 5e-4
) -> None:
    """Raise unless each reference arm reproduces the NDCG Plan 6 published.

    If Stage 2's parquet or the signal frame were ever re-dumped with a
    different fit, the arms here would be scored on windows the published
    numbers never saw, and every delta would be against the wrong baseline.
    """
    for arm in REFERENCE_ARMS:
        if arm not in published:
            raise KeyError(f"no published NDCG for {arm!r}")
        if abs(measured[arm] - published[arm]) > tol:
            raise ValueError(
                f"{arm} measures {measured[arm]:.4f} here against a published "
                f"{published[arm]:.4f}; the windows have drifted"
            )


def _published(split: str) -> dict[str, float]:
    payload = json.loads(PUBLISHED_REFERENCE[split].read_text(encoding="utf-8"))
    return {arm["name"]: arm["ndcg"]["point"] for arm in payload["arms"]}


def _scope(split: str, seed: int):
    """Windows, judged matrix, signal frame and text maps for fold 0 or the test split."""
    from src.cross_encoder import text_maps
    from src.stage_signals import load_signals, scope_windows

    scope = "fold0" if split == "train" else "test"
    windows_, matrix = scope_windows(scope, seed=seed)
    frame = load_signals(scope)
    query_text, doc_text = text_maps(
        matrix,
        Path("data/combined/products.parquet"),
        Path("data/combined/judgements.parquet"),
    )
    query_text = {str(k): v for k, v in query_text.items()}
    return scope, windows_, matrix, frame, query_text, doc_text


def _evaluate(name, per_query, floor, seed):
    from src.rank_report import evaluate_arm

    return evaluate_arm(
        name, per_query, floor.per_query, groups=("retrieval",),
        n_features=0, objective="rerank", best_iteration=0, seed=seed,
    )


def _pilot(args) -> int:
    from src.blend_report import per_query_ndcg, single_stage_orderings
    from src.cross_encoder import rerank
    from src.distill import (
        INITS, TARGETS, cross_fit_orderings, fine_tune_windows,
        gains_from_frame, split_halves, window_dataset,
    )
    from src.fine_rank_report import check_same_queries
    from src.floor import random_floor
    from src.llm_rerank import RerankCache
    from src.rank_report import compare, format_table, qrels_from_frame
    from src.stage_signals import check_fallback_share, llm_orderings_from_cache

    _, windows_, matrix, frame, query_text, doc_text = _scope("train", args.seed)
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    gains = gains_from_frame(matrix)

    # The teacher's real answers. The malformed one (query 55755) is missing
    # here rather than carried in Stage 2 order, and every target trains on
    # the same answered windows so the target is the only difference.
    teacher, missing = llm_orderings_from_cache(windows_, query_text, RerankCache(args.cache))
    check_fallback_share(missing, len(windows_))
    halves = split_halves([w.query_id for w in windows_], seed=args.seed)
    print(f"fold 0: {len(windows_):,} windows, halves of {len(halves[0]):,} and "
          f"{len(halves[1]):,}; {len(missing)} without a teacher answer {missing}")

    results = []
    reference = single_stage_orderings(frame)
    for name in REFERENCE_ARMS:
        results.append(_evaluate(name, per_query_ndcg(reference[name], windows_, qrels),
                                 floor, args.seed))

    def score(model, ws):
        return rerank(model, ws, query_text, doc_text)

    for init, path in INITS.items():
        for target in TARGETS:
            def fit(ws, init=init, path=path, target=target):
                answered = [w for w in ws if w.query_id in teacher]
                dataset, _ = window_dataset(
                    answered, query_text, doc_text,
                    kind=target, gains=gains, llm_orderings=teacher,
                )
                return fine_tune_windows(
                    dataset, init=path, epochs=PILOT_EPOCHS[init], seed=args.seed
                )

            orderings = cross_fit_orderings(windows_, halves, fit=fit, score=score)
            arm = _evaluate(pilot_arm(init, target),
                            per_query_ndcg(orderings, windows_, qrels), floor, args.seed)
            results.append(arm)
            print(f"  {arm.name:24s} {arm.ndcg.point:.4f}")

    check_same_queries(results)
    by_name = {arm.name: arm for arm in results}
    comparisons = [
        compare(by_name[pilot_arm(init, target)], by_name[pilot_arm(init, "labels")],
                seed=args.seed)
        for init in INITS for target in ("llm", "hybrid")
    ]
    gate = paid_run_gate(comparisons)

    print()
    print(format_table(results))
    print("\n  -- the teacher's target against the labels, same queries, same start --")
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:24s} {d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  "
              f"{'significant' if row['significant'] else 'ties'}")
    print(f"\n  gate: {'PASSED' if gate['passed'] else 'not passed'} - {gate['rule']}")

    payload = {
        "scope": "fold 0, cross-fitted in halves",
        "n_queries": len(qrels),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "halves": {"seed": args.seed, "sizes": [len(halves[0]), len(halves[1])]},
        "teacher_missing": sorted(missing),
        "config": {
            "loss": "RankNetLoss", "inits": INITS, "epochs": PILOT_EPOCHS,
            "targets": list(TARGETS),
        },
        "arms": [arm.to_dict() for arm in results],
        "comparisons": comparisons,
        "gate": gate,
        "seed": args.seed,
    }
    out = args.out or DEFAULT_OUT["pilot"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def _arms(args) -> int:
    from src.blend_report import per_query_ndcg, single_stage_orderings
    from src.cross_encoder import load_reranker, measure_latency, rerank
    from src.distill import gains_from_frame, pairwise_accuracy
    from src.fine_rank_report import check_same_queries
    from src.floor import random_floor
    from src.rank_report import compare, format_table, qrels_from_frame

    check_split(args.split, args.final)
    models = resolve_model_arms(args.arms)
    training = training_records(models)
    scope, windows_, matrix, frame, query_text, doc_text = _scope(args.split, args.seed)
    qrels = qrels_from_frame(matrix)
    floor = random_floor(qrels, n_trials=args.floor_trials, seed=args.seed)
    gains = gains_from_frame(matrix)
    print(f"{scope}: {len(windows_):,} windows; floor {floor.mean:.4f}")

    results, latency, accuracy = [], {}, {}
    reference = single_stage_orderings(frame)
    for name in REFERENCE_ARMS:
        results.append(_evaluate(name, per_query_ndcg(reference[name], windows_, qrels),
                                 floor, args.seed))
        accuracy[name] = pairwise_accuracy(windows_, reference[name], gains)
    check_reference({arm.name: arm.ndcg.point for arm in results}, _published(args.split))

    landed = load_reranker(LANDED_MODEL)
    latency["stage2+ce"] = measure_latency(
        landed, windows_, query_text, doc_text, n_queries=50).to_dict()
    del landed

    for name, path in models.items():
        model = load_reranker(path)
        orderings = rerank(model, windows_, query_text, doc_text)
        results.append(_evaluate(name, per_query_ndcg(orderings, windows_, qrels),
                                 floor, args.seed))
        accuracy[name] = pairwise_accuracy(windows_, orderings, gains)
        latency[name] = measure_latency(
            model, windows_, query_text, doc_text, n_queries=50).to_dict()
        del model
        print(f"  {name:22s} {results[-1].ndcg.point:.4f}  "
              f"single {latency[name]['single_ms']:.1f} ms")

    check_same_queries(results)
    by_name = {arm.name: arm for arm in results}
    stage2, ce, llm = (by_name[name] for name in REFERENCE_ARMS)
    comparisons = [compare(arm, stage2, seed=args.seed) for arm in results[1:]]
    for name in models:
        comparisons.append(compare(by_name[name], ce, seed=args.seed))
        comparisons.append(compare(by_name[name], llm, seed=args.seed))
    shares = {
        name: share_of_teacher_gain(by_name[name].ndcg.point, stage2.ndcg.point,
                                    llm.ndcg.point)
        for name in ("stage2+ce", *models)
    }

    print()
    print(format_table(results))
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:22s} vs {row['baseline']:12s} {d['point']:+.4f} "
              f"[{d['low']:+.4f}, {d['high']:+.4f}]")

    payload = {
        "split": args.split,
        "scope": scope,
        "n_queries": len(qrels),
        "n_judgements": len(matrix),
        "floor": {"mean": floor.mean, "low": floor.low, "high": floor.high,
                  "n_trials": floor.n_trials},
        "arms": [arm.to_dict() for arm in results],
        "comparisons": comparisons,
        "share_of_llm_gain": shares,
        "pairwise_accuracy": accuracy,
        "latency": latency,
        "models": {name: str(path) for name, path in models.items()},
        "training": training,
        "seed": args.seed,
    }
    out = args.out or DEFAULT_OUT[args.split]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


def _main() -> int:
    from src.llm_rerank import DEFAULT_CACHE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="the fold-0 cross-fitted pilot and the gate")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--final", action="store_true", help="required with --split test")
    parser.add_argument("--arms", nargs="+", default=list(FREE_ARMS),
                        choices=list(MODEL_ARMS))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--floor-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.pilot:
        if args.split != "train":
            raise SystemExit("the pilot trains on fold 0; it never touches the test split")
        return _pilot(args)
    return _arms(args)


if __name__ == "__main__":
    raise SystemExit(_main())
```

In `src/distill.py`, add `gate_choice` above `training_windows`:

```python
def gate_choice(payload: Mapping) -> tuple[str, str]:
    """The (init, target) the pilot's pre-registered gate chose for the student.

    The student is not picked by hand: Phase 3 trains exactly what the gate
    named, from the committed docs/results/distill-pilot.json.
    """
    gate = payload.get("gate") or {}
    if not gate.get("passed"):
        raise ValueError(
            "the pilot's gate did not pass, so there is no student to train; "
            "the free arms are the result"
        )
    choice = gate.get("choice") or {}
    if choice.get("init") not in INITS or choice.get("target") not in ("llm", "hybrid"):
        raise ValueError(
            f"the gate chose {choice!r}, which is not a teacher target from a "
            f"known start ({sorted(INITS)})"
        )
    return choice["init"], choice["target"]
```

In its `_main`, replace the `--target` argument with a required choice between
naming a target and reading the gate's:

```python
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--target", choices=list(TARGETS))
    which.add_argument(
        "--from-gate", type=Path,
        help="train what the pilot's gate chose (docs/results/distill-pilot.json)")
```

and, straight after `args = parser.parse_args()`:

```python
    if args.from_gate is not None:
        args.init, args.target = gate_choice(
            json.loads(args.from_gate.read_text(encoding="utf-8"))
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_distill_report.py tests/test_distill.py -v`
Expected: PASS — 21 and 32 passed, 1 deselected.

Run: `python -m pytest`
Expected: PASS — 780 passed (713 before this plan), 36 deselected.

- [ ] **Step 5: Commit the code**

```bash
git add src/distill_report.py src/distill.py tests/test_distill_report.py tests/test_distill.py
git commit -m "Add the distillation report: the fold-0 pilot, its gate, and every arm beside the three stages"
```

- [ ] **Step 6: Train the hard-negative arm**

Run: `python -m src.distill --target labels --out models/cross-encoder/windows`
Expected (about 10 min):

```
12,519 training windows from folds 2/3/4, target 'labels', init 'scratch'; 0 dropped for want of a teacher answer
trained in 9.5 min -> models/cross-encoder/windows
```

- [ ] **Step 7: Train the capacity arm**

Check that the card has room first: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`.
It needs about 11 GB. Another project's index build has been seen holding about 4 GB.
Do not start this beside any other training.

Run: `python -m src.distill --target labels --init BAAI/bge-reranker-base --batch-size 8 --out models/cross-encoder/bge`
Expected (about 62 min):

```
12,519 training windows from folds 2/3/4, target 'labels', init 'BAAI/bge-reranker-base'; 0 dropped for want of a teacher answer
trained in 62.3 min -> models/cross-encoder/bge
```

- [ ] **Step 8: Report the free arms on fold 0**

Run with no training on the GPU, since this step times every arm: `python -m src.distill_report`
Expected (a few minutes): the reference arms reproduce Plan 6 — `stage2`
0.8519, `stage2+ce` 0.8587, `stage2+llm` 0.8814 — or `check_reference` stops
the run. `stage2+ce_windows` lands near the measured 0.8555, about −0.0031
against `stage2+ce`. `stage2+ce_bge` has not been measured; whatever it
scores is the result. Writes `docs/results/distill.json`.

- [ ] **Step 9: Run the pilot and decide the gate**

Run: `python -m src.distill_report --pilot`
Expected (about 31 min — six targets, two halves each):
- halves of 2,065 and 2,065;
- one window without a teacher answer, `['55755']`;
- six `pilot:` arms;
- the four teacher-against-labels comparisons, then the gate line.

The two scratch pilots in the README predict every comparison ties or loses
and the line reads `gate: not passed`. They split 4,129 answered ids rather
than all 4,130, so the committed halves differ and the numbers will not match
them to four places. **Whatever the line says is the decision.**
Writes `docs/results/distill-pilot.json`.

- [ ] **Step 10: Pin the committed results**

Add to the imports at the top of `tests/test_distill_report.py`:

```python
import json
from pathlib import Path

from src.distill import INITS, TARGETS
```

and append:

```python
def _committed(name):
    return json.loads(Path("docs/results", name).read_text(encoding="utf-8"))


def test_the_committed_pilot_decided_its_gate_by_the_written_rule():
    pilot = _committed("distill-pilot.json")
    assert pilot["gate"]["rule"] == GATE_RULE
    assert pilot["gate"] == paid_run_gate(pilot["comparisons"])
    names = {arm["name"] for arm in pilot["arms"]}
    assert {pilot_arm(i, t) for i in INITS for t in TARGETS} <= names
    assert {arm["n_queries"] for arm in pilot["arms"]} == {4130}


def test_the_committed_fold_0_arms_sit_on_plan_6_s_windows():
    arms = {a["name"]: a["ndcg"]["point"] for a in _committed("distill.json")["arms"]}
    published = {a["name"]: a["ndcg"]["point"] for a in _committed("fine-rank.json")["arms"]}
    for name in REFERENCE_ARMS:
        assert arms[name] == pytest.approx(published[name], abs=5e-4)
    assert set(FREE_ARMS) <= set(arms)


def test_the_student_is_exactly_what_the_gate_chose():
    # No gate, no student. A passed gate names one (init, target), and the
    # student's own training record must say it was trained on that.
    gate = _committed("distill-pilot.json")["gate"]
    report = _committed("distill.json")
    arms = {arm["name"] for arm in report["arms"]}
    if not gate["passed"]:
        assert "stage2+student" not in arms
        return
    assert "stage2+student" in arms
    record = report["training"]["stage2+student"]
    assert (record["init"], record["target"]) == (
        gate["choice"]["init"], gate["choice"]["target"]
    )
```

Run: `python -m pytest tests/test_distill_report.py -v`
Expected: PASS — 24 passed. `test_the_student_is_exactly_what_the_gate_chose`
passes on either branch: with the gate not passed, it checks no student was
reported. Task 5 re-runs it on the other branch.

- [ ] **Step 11: Commit the results**

```bash
git add tests/test_distill_report.py docs/results/distill.json docs/results/distill-pilot.json
git commit -m "Report the free distillation arms and decide the pilot's gate on fold 0"
```

---

## Phase 1 Gate

- [ ] `python -m pytest` passes with no failures and no new skips. — 783 passed (713 before this plan), 36 deselected.
- [ ] `python -m pytest -m data` passes, including the out-of-fold ordering's 0.8534 and the training windows' 34.5% overlap with the in-sample ones.
- [ ] `data/features/stage2-train-oof.parquet` holds 250,485 rows over 12,519 queries, none flagged in-sample, and `docs/results/stage2-oof.json` records the shift from the in-sample ordering.
- [ ] `docs/results/distill.json` reports `stage2+ce_windows` and `stage2+ce_bge` beside the three reference arms. The reference arms reproduce Plan 6. Each new arm has paired intervals against all three, its share of the LLM's gain, its pairwise accuracy, its latency and its training record.
- [ ] `docs/results/distill-pilot.json` holds six pilot arms over the same 4,130 queries, and its `gate` equals `paid_run_gate(comparisons)`.
- [ ] **The gate's verdict is recorded here, in this file, before Phase 2 is opened:**
  - **not passed** → Phase 2 and Task 5 are skipped. Mark them `Skipped: the Phase 1 gate did not pass (docs/results/distill-pilot.json)` and go straight to [Task 6](phase-3-the-student-and-the-report.md#task-6-the-test-split-once-and-the-writeup).
  - **passed** → [Phase 2](phase-2-the-teacher.md), whose first step is showing the user the dry run.

Then: [Phase 2 — The Teacher](phase-2-the-teacher.md), or Task 6.
