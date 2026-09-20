# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit messages

- **One line only.** No body, no bullet list, no trailing paragraphs.
- **Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the
  subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line.
  This applies to pull request descriptions too.

## Project

A multi-stage product search ranking funnel (recall → coarse rank → fine rank →
blend) on the Amazon ESCI Shopping Queries Dataset, enriched with images and
behavioural features from ESCI-S. `PROJECT_SPEC.md` is the source of truth for
motivation, funnel design, baselines, and the ablation table — read it first.

The task is **re-ranking**, not end-to-end retrieval: only judged candidates are
ranked. Recall is evaluated separately (Recall@k over the corpus) from ranking
(NDCG over the judged candidate list).

## Commands

```bash
# Environment (the venv has no pip; use uv, or python -m pip in a pip-bearing venv)
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # add ".[dev,baselines]" for the SBERT baseline

# Tests. Bare pytest deselects the `data` and `slow` markers (pyproject addopts),
# so it stays fast enough to run after every edit.
python -m pytest                    # fast suite
python -m pytest -m data            # needs the real ESCI parquet on disk
python -m pytest -m slow            # minutes-long, e.g. the SBERT baseline
python -m pytest -m "data and slow" -s   # the 100-trial random floor, printed

# Evaluation. Every headline NDCG carries the measured floor beside it.
python -m src.cli floor --split test
python -m src.cli evaluate --run runs/<name>.trec --split test --tag <name>
python -m src.cli freeze-splits

# Enrichment corpus (one streaming pass over the 3.4 GB ESCI-S zstd)
python -m src.esci_s_etl                      # -> data/esci-s/corpus.parquet
python -m src.coverage --split test           # join coverage + missingness bias
python -m src.combine                         # -> data/combined/{products,judgements}.parquet

# Image embeddings (network-bound; resumable, 1.8-7.3 h for the rerank scope)
python -m src.embed_images --scope rerank     # -> data/embeddings/rerank/
python -m src.image_report --scope rerank     # coverage + semantic gate

# Image URL resolution gate - samples live URLs, exits non-zero below 90%
python -m src.esci_images <esci.json.zst>   # a truncated prefix of the file is fine
```

There is no linter configured yet.

## Data invariants

These were verified empirically against the real data and each one silently
corrupts results if violated. Do not take the numbers in third-party write-ups
(or older parts of `PROJECT_SPEC.md`) over these.

**Use the official parquet, not the HF mirror.** `tasksource/esci` has no `split`
column — it encodes the *large-version* split, yielding 185,361 test judgements
instead of 181,701 (~2% contamination). The official file is Git LFS, so fetch via
`media.githubusercontent.com/media/amazon-science/esci-data/...`. Assert
**181,701 judgements / 8,956 queries / 164,900 products** (test) and
**419,653 / 20,888** (train) on every load.

**Label distribution (test):** E 43.87%, S 34.98%, I 16.69%, C 4.46%. Any source
reporting Complement ≈ 35% has the known S/C swap bug in the official
`prepare_trec_eval_files.py`. Gains are E=1.0, S=0.1, C=0.01, I=0.0.

**Compute the random floor, never quote it.** Measured 0.7467 with standard
`1/log2(rank+1)` discount; the published SQID figure is 0.7483. Under the swapped
mapping it is 0.7141 — a 3.3-point difference, larger than most method gains.

**ESCI-S image URLs need repairing before use.** ~50% route through a retired CDN
bucket and return HTTP 400. `src/esci_images.py:normalize_url` strips it (0% → 100%
resolution; a no-op on clean URLs). Always go through `image_url()`, which also
handles the fact that **products store the URL in `image` but books store it in
`img`** — reading only `image` drops every book silently. Books are 7.4% of
all-locale records but **5.74% of the `us` non-error corpus** (measured: 61,999 of
1,080,262), so quote the locale with the number.
Books likewise use `desc`/`attr` where products use `description`/`attrs`.

**Real image coverage is ~75%**, not the headline 91.5%: 91.5% of ASINs in ESCI-S
× ~84.7% carrying a URL × ~97.5% resolving. Coverage is even across train (84.8%)
and test (84.6%) and roughly label-balanced, so the both-splits premise holds.
**Measured end to end: 0.7753** over the 482,105 re-ranking products, 0.7668 over
the 1,215,854-product catalogue — both confirming ~75%, not 91.5%.

**Measured ESCI-S join coverage is 89.59%** of the 482,105 Task 1 English
re-ranking products (431,930 matched), and 88.85% of the catalogue. The 91.5%
headline counts ASINs *present in the file*, error rows included; usable
enrichment is lower. Of the 50,175 misses: 34,032 are absent from the scrape
entirely, 15,994 have a `us` scrape-error row carrying no metadata, 276 appear
only under `es`/`jp`. Counting the error rows as covered would give 92.91%,
which is how the headline arises. The catalogue is **1,215,854** unique `us`
products, not the 1,215,851 recorded elsewhere in this file.

**Drop `type: "error"` rows** (~3.7% of ESCI-S) — scrape failures carrying only
`asin`/`locale`/`error`/`template`.

**Behavioural features are uneven.** Dense and fully parseable: `stars` 97.4%,
`ratings` 97.7%, `category` 95.4%, `template` 100%. Sparse: `price` 29.1%,
Best Sellers Rank 46.2%, `attrs` 57.9%, `info` 53.3%. LightGBM handles the NaNs
natively — keep sparse features with missingness indicators, but the behavioural
ablation rests on the dense ones.

**Those sparse percentages count the *product* key only.** Once the book
spellings are unified into the same column, the shares rise: `attrs`∪`attr` is
63.4% (product 57.7 + book 5.7) and the image URL is 86.3% (product 82.1 + book
4.2). A corpus reproducing 57.9% / 84.7% is one that dropped every book. Verified
against all 1,080,262 `us` non-error rows.

**`info` is product-only — 0 of 61,999 `us` books carry one.** Best Sellers Rank
lives in `info`, so for books it is *structurally* absent, not missing at random.
Averaging over that NaN treats a page-layout difference as a behavioural signal.

**Field-presence bands do not detect a dropped-books pass.** Losing every book
moves the image URL share by 0.8 points and `attrs` by 2.2 — both inside any
sane tolerance. Assert the book share directly; `src/esci_s_etl.py` does.

**Enrichment missingness is mildly confounded with relevance, and it is not the
join.** Whether a product is in ESCI-S *at all* shifts mean gain by only
-0.0081 [-0.0195, +0.0028] — a cluster bootstrap over queries whose interval
straddles zero, so the 10.4% join miss is not significantly more or less
relevant. But *field-level* missingness within the corpus is confounded, and
more strongly: restricted to in-corpus products, `ratings` shifts -0.0548,
`category` -0.0501, `stars` -0.0220. The sign is consistent everywhere —
products **missing** a field are slightly **more** relevant. Plan 5 must carry
per-field missingness indicators and must not read Ablation 4 as clean without
checking the model is not simply exploiting them.

**`category` is a sparse field, not a dense one.** Over all test judgements its
missingness shifts mean gain -0.0210 [-0.0305, -0.0117], past the 0.02 the
behavioural ablation tolerates, so it was reclassified. The dense set is
`stars`, `ratings`, `template`.

**The GitHub `sample.json.gz` in shuttie/esci-s has no `image` field at all.** It is
stale against the real 3.4 GB file. Never validate the image pipeline against it.

**The combined tables are two files, not one, on purpose.** `python -m src.combine`
writes `data/combined/products.parquet` (1,215,854 `us` products, no labels) and
`data/combined/judgements.parquet` (601,354 rows with `split` and the frozen
`fold`; test rows carry `fold = -1`). Keeping labels out of the product table is
what stops product-level target encoding being one groupby away — 34,756
products appear in both splits. `combine_products` raises if a label column ever
appears there. ESCI-S columns are prefixed `s_` because both datasets have a
`title` and a `description`.

**Use the coalesced `description` column, not either source.** ESCI has a
description for 52.2% of judged products and ESCI-S fills a further 37.0%,
taking the union to **89.2%**. The two are confounded with the label in
*opposite* directions (+0.0198 and -0.0257 mean gain, both significant) and the
coalesce largely cancels it: the combined column measures **-0.0065
[-0.0178, +0.0040]**, not significant. `description_source` records which side
supplied it.

## Evaluation discipline

- **34,756 products appear in both train and test.** The official split is
  query-level, so this is legitimate — but any product-level target encoding leaks
  train labels into test. Features must be query×product, never product-alone-from-labels.
- There is no official validation split. Carve one from train by `query_id`
  (GroupKFold) and freeze it before tuning anything. Never split within a query group.
- Always report against the random floor with bootstrap CIs over queries. A method
  that ties the baseline is a legitimate, reportable result.
- **Field *presence* is itself a ranker, and only some of it is legitimate.**
  Measured by scoring test judgements on presence patterns alone, fitted on
  train (pattern-optimal, so a true ceiling):

  | presence flags only | NDCG | lift over the 0.7467 floor |
  |---|---|---|
  | ESCI's own (`product_brand`, `product_color`, …) | 0.7519 | +0.0052 |
  | ESCI-S's (`s_*`) | 0.7552 | **+0.0084** |
  | both | 0.7633 | +0.0166 |
  | *SBERT zero-shot, for scale* | *0.8294* | *+0.0827* |

  **ESCI presence is legitimate** — a production ranker also knows whether a
  product has a brand. **ESCI-S presence is an artefact**: "our 2022 scrape
  failed on this page" is not available at serving time. So the ~+0.0084 must be
  subtracted from any behavioural-feature result, not assumed away. Ablation 4
  therefore needs three arms — text only, text + values + indicators, and
  **indicators only** — and the honest contribution is arm 2 − arm 3.
  The effects are super-additive (+0.0166 > +0.0052 + +0.0084), so the combined
  bias cannot be bounded by adding the parts.
- **Correlation size and ranking value are different questions.** `product_brand`
  missingness carries the largest label correlation in the dataset (+0.0567
  [+0.0382, +0.0755]) yet all four ESCI presence flags together buy only +0.0052
  NDCG, because brand is present 94.9% of the time. Report the NDCG, not the
  correlation. §4.2's planned brand-match and colour-match features inherit this.

## Scale

482,105 unique products for re-ranking (~362K images, ~3.6 GB); 1,215,851 for
full-corpus recall (~912K images, ~9.1 GB). Downloads: ESCI examples 48.9 MB +
products 1.03 GB, ESCI-S 3.37 GB. ESCI-S is **single-frame zstd** — no random
access, no resumable ranged decompression, so filter to `us`, drop error rows,
normalise book fields, and write Parquet all in one streaming pass.
