# Plan 2 — Enrichment Corpus

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan phase by phase, task by task. Steps use checkbox (`- [ ]`) syntax for tracking. Start at [Phase 1](phase-1-the-record.md); do not start a phase before its predecessor's gate passes.

**Goal:** Turn the 3.4 GB single-frame ESCI-S zstd scrape into a queryable Parquet corpus in one streaming pass, then prove the join against ESCI is wide enough to build on and that what is missing from it is not confounded with the relevance label.

**Architecture:** Two layers with a hard boundary. `src/esci_s.py` is pure: one JSON record in, one flat typed row out, no I/O, so every structural trap in the scrape is unit-tested in milliseconds against records pasted from the real file. `src/esci_s_etl.py` owns the single pass — zstd stream → locale filter → error-row drop → normalise → Parquet row groups — because the source is single-frame zstd with no random access, so there is exactly one opportunity to read it. `src/coverage.py` then answers the two questions the gate asks, reusing Plan 1's loader, labels and bootstrap rather than recomputing any of them.

**Tech Stack:** Python ≥3.11, `zstandard` (streaming decompression), `pyarrow` (`ParquetWriter`, explicit schema, zstd compression), pandas, numpy, pytest. No new dependencies — all four are already in `pyproject.toml` from Plan 1.

**Spec:** `PROJECT_SPEC.md` (§3.2 ESCI-S, §4.2 Stage 2 feature groups, §8.2 Build Order) and `CLAUDE.md` (Data invariants, Scale). The plan argues from both; executors read both.

**Series:** Plan 2 of 7 — see [`../README.md`](../README.md). Gated on Plan 1, whose Plan Gate passed at NDCG 0.8294 against a published 0.8292.

---

## The two phases

The split is by dependency. Phase 1 needs nothing on disk and ends with every
record shape in the scrape handled exactly. Phase 2 is the first phase that
touches the 3.4 GB file, and it only gets one pass at it.

| Phase | File | Tasks | Delivers | Real data? |
|---|---|---|---|---|
| 1 | [**The Record**](phase-1-the-record.md) | 1–2 | Book/product field unification, typed behavioural parsers | No |
| 2 | [**The Corpus**](phase-2-the-corpus.md) | 3–5 | Streaming ETL to Parquet, join coverage, missingness-bias report | Yes (3.4 GB) |

**One cross-plan edit.** Task 5 reopens `src/bootstrap.py` from Plan 1 to add
`cluster_delta_ci`, because the missingness comparison is between two groups of
*judgements* that are clustered within queries, which neither `bootstrap_ci`
(one sample) nor `paired_delta_ci` (paired on the same queries) can express.
The step carries both the function and its covering test. Do not implement
`src/coverage.py` against the Plan 1 shape of `src/bootstrap.py`.

---

## Where these numbers came from

Every figure asserted in this plan was measured against the real
`esci.json.zst` before the plan was written, by streaming a 400,000-record
prefix and counting. They are not quoted from the ESCI-S README, and three of
them contradict what a reader would reasonably assume:

- **`info` is product-only.** Zero of 11,350 sampled books carry an `info`
  block. Best Sellers Rank lives in `info`, so BSR is *structurally* absent for
  books, not missing at random. Task 5 is what stops that from being read as a
  behavioural signal.
- **Books carry a singular `review` string as well as the `reviews` list.**
  44% of books have it. `CLAUDE.md` documents `desc`/`attr`/`img` but not this.
- **0.05% of prices are multi-price strings** like `'$11.53 $12.99'` or
  `'$24.95 $29.99 $19.99 $19.99'` — sale or variant pairs, in no reliable
  order (the first is sometimes higher, sometimes lower). 66 of 132,846
  sampled prices. `float(price.strip("$"))` raises on every one of them.
- **zstandard does not raise on a truncated frame.** Cutting a frame in half
  and reading it with `stream_reader` returned 3,670,016 of 7,420,000 bytes
  and no exception (zstandard 0.25.0). Truncation is detectable only via
  `decompressobj().eof`, or by comparing against the `content_size` the frame
  header declares — 11,539,399,406 bytes for the real file, which also carries
  an xxhash checksum. `src/esci_images.py` carries a comment asserting the
  opposite; Task 3 corrects it.

The field-presence percentages in Global Constraints reproduce `CLAUDE.md`'s
documented values to within 0.3 points on that prefix, which is why they are
asserted rather than recomputed from scratch.

---

## Global Constraints

- **Python ≥3.11.** Development machine runs 3.13.13.
- **ESCI-S is single-frame zstd.** No random access, no resumable ranged decompression. Filter to `us`, drop error rows, normalise book fields, and write Parquet **all in one streaming pass**. Source: `https://esci-s.s3.amazonaws.com/esci.json.zst` (3,620,467,456 bytes, declaring 11,539,399,406 decompressed; a Google Drive mirror exists and is slower).
- **Drop `type: "error"` rows.** Scrape failures carrying only `asin`/`locale`/`error`/`template`. Measured 3.8% of all records, 3.4% of `us` records.
- **Products store the main image URL in `image`, books in `img`.** Books likewise use `desc`/`attr` where products use `description`/`attrs`. Reading only the product spelling drops every book silently.
- **Always go through `image_url()`** from `src/esci_images.py`, which handles the `image`/`img` split *and* strips the retired CDN bucket (`/images/W/<bucket>/images/I/` → `/images/I/`). Roughly half of ESCI-S image URLs carry that segment and return HTTP 400 with it.
- **Asserted field presence over `us` non-error records** (tolerance ±0.02 absolute): `template` 1.000, `ratings` 0.977, `stars` 0.974, `category` 0.954, image URL 0.847, `attrs`∪`attr` 0.579, `info` 0.533, Best Sellers Rank 0.462, `price` 0.291.
- **Behavioural features are uneven, and the ablation rests on the dense ones.** Dense: `stars`, `ratings`, `category`, `template`. Sparse: `price`, Best Sellers Rank, `attrs`, `info`. LightGBM handles NaN natively — keep sparse features with missingness indicators.
- **Real image coverage is ~75% end to end**, not the headline 91.5%: 91.5% of ASINs in ESCI-S × ~84.7% carrying a URL × ~97.5% resolving. Do not report 91.5% as image coverage.
- **The GitHub `sample.json.gz` in shuttie/esci-s has no `image` field at all.** It is stale against the real 3.4 GB file. Never validate against it.
- **Features must be query×product, never product-alone-from-labels.** 34,756 products appear in both train and test; the official split is query-level, so this is legitimate, but product-level target encoding leaks train labels into test. This plan produces *product* attributes, which are inputs to query×product features — never targets.
- **Report bootstrap confidence intervals over queries, not bare point estimates.** Judgements are clustered within queries; resample the query, not the judgement.
- **Do not commit datasets.** `.gitignore` blocks `data/`, `*.parquet`, `*.zst`. The corpus Parquet lives at `data/esci-s/corpus.parquet` and is *not* committed. Only the JSON reports under `docs/results/` are.
- **Commit messages are one line. Never mention Claude, Claude Code, Anthropic, or any AI tool** — not in the subject, not as a `Co-Authored-By:` trailer, not as a "Generated with" line. (`CLAUDE.md`; this overrides any harness default that would add an attribution trailer.)

---

## Review Focus

The five failure modes the spec implies but that no task's happy path exercises. Each is pinned to a test inside the task that owns the code.

1. **Reading only the product spelling drops every book.** Books are ~6% of `us` non-error records and use `img`/`desc`/`attr` where products use `image`/`description`/`attrs`. Nothing raises — the corpus is simply 6% smaller and systematically missing one product type, which is 5.1% of the page templates. — pinned in Task 1.
2. **`info` is absent for every book, so Best Sellers Rank is structurally missing.** A NaN that means "this record type never has this field" is not the same as a NaN that means "this product has no rank", and averaging over them treats a page-layout difference as a behavioural signal. — pinned in Task 2 (the parser returns `None` for a book) and quantified in Task 5.
3. **Multi-price strings.** 0.05% of prices are `'$11.53 $12.99'`. `float(p.strip("$"))` raises; a bare `re.search` silently picks one of several prices and records nothing about having done so. The row must carry both the chosen value and the fact that it was ambiguous. — pinned in Task 2.
4. **A truncated download parses as a perfectly good prefix, and the library stays silent.** This was measured, not assumed: on zstandard 0.25.0, `stream_reader.read()` over half a frame returned 3,670,016 of 7,420,000 bytes and never raised. Catching `ZstdError` — which `src/esci_images.py` does, with a comment saying truncation is "expected" — is dead code. The ETL must check `decompressobj().eof` and the frame's declared `content_size`, fail loudly, and leave no Parquet behind. — pinned in Task 3.
5. **Join coverage measured against the wrong ASIN set.** ESCI-S's headline 91.5% is against all 1,814,925 ESCI ASINs across every locale. This project re-ranks 482,105 `us` Task 1 products and recalls over ~1.2M. Quoting 91.5% instead of measuring against the actual product set would hide a coverage hole in exactly the subset that matters. — pinned in Task 4.

---

## File Structure

The repo's established pattern is flat modules under `src/`, imported as
`src.<name>` and run as `python -m src.<name>`. This plan follows it.

| File | Responsibility | Written in |
|---|---|---|
| `src/esci_s.py` | One ESCI-S record → one flat typed row. Book/product unification and the behavioural parsers. Pure: no I/O, no pandas | Phase 1, Tasks 1–2 |
| `src/esci_s_etl.py` | The single streaming pass and the Parquet schema. `python -m src.esci_s_etl` | Phase 2, Task 3 |
| `src/coverage.py` | Join coverage and the missingness-bias report. `python -m src.coverage` | Phase 2, Tasks 4–5 |
| `src/bootstrap.py` | **Reopened in Task 5** to add `cluster_delta_ci` | Plan 1, Phase 3, Task 7 |
| `docs/results/esci-s-coverage.json` | Committed join-coverage record | Phase 2, Task 4 |
| `docs/results/esci-s-missingness.json` | Committed missingness-bias record | Phase 2, Task 5 |
| `data/esci-s/corpus.parquet` | The corpus itself — **not committed** | Phase 2, Task 3 |
| `tests/test_esci_s.py`, `tests/test_esci_s_etl.py`, `tests/test_coverage.py` | One test module per source module | Every task |

`src/esci_s.py` holds no I/O on purpose: the traps in this data are all
*structural* (which key name, which record type), and they are testable against
records pasted verbatim from the real file with no 3.4 GB read in the way.

---

## Plan Gate

Plan 3 does not start until all of these hold:

- [ ] `python -m pytest` passes with no failures and no new skips.
- [ ] `python -m pytest -m data` passes, including the corpus invariants.
- [ ] `data/esci-s/corpus.parquet` exists, covers every `us` non-error record, and its measured field presence matches the Global Constraints table within ±0.02.
- [ ] `docs/results/esci-s-coverage.json` shows join coverage **≥ 90%** of the 482,105 Task 1 English re-ranking products, measured — not the 91.5% headline.
- [ ] `docs/results/esci-s-missingness.json` shows, for each of the four dense fields, a mean-gain shift below 0.02 with a reported cluster-bootstrap CI.
- [ ] `CLAUDE.md`'s Commands section lists the ETL and coverage invocations.

Then: Plan 3 — Image Pipeline. See [`../README.md`](../README.md) for the series.
