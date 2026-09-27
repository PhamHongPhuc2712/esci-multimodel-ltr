# Results — Multi-Stage Multimodal Product Search Ranking on Amazon ESCI

Every measured figure below is quoted from a committed file under
`docs/results/`, named beside it. The only exceptions are the published
baselines, cited from `PROJECT_SPEC.md` §5 and marked as published.
`tests/test_results_doc.py` checks every four-decimal figure in this file
against those sources.

## 1. What this is

A five-stage product search funnel on the Amazon ESCI Shopping Queries Dataset
(Task 1, English): LLM query rewriting, hybrid recall (BM25 + dense text + CLIP
image, fused with RRF), a LightGBM LambdaMART coarse ranker over 47 features,
fine re-ranking of its top 10 by a fine-tuned cross-encoder or an LLM listwise
call, and a blend stage. ESCI is enriched with images, prices, ratings,
categories and attributes from the ESCI-S scrape.

The task is **re-ranking**: only judged candidates are ranked, so ranking is
scored as full-list NDCG over each query's judged list (gains E 1.0, S 0.1,
C 0.01, I 0.0). Recall is measured separately, as Recall@100 over the whole
1.2M-product catalogue. Every NDCG sits beside a random floor computed on the
same queries, with a 95% bootstrap interval over queries.

## 2. Headline

**The best ranker in the project is the LLM listwise re-ranking of the coarse
ranker's top 10: NDCG 0.8855 [0.8827, 0.8883] over all 8,956 queries of the
test split**, against a random floor of 0.7468 and the coarse ranker's own
0.8579 — a gain of +0.0276 [+0.0255, +0.0296] (`fine-rank-test-full.json`).

That interval lies entirely above the **0.8562 `ESCI_baseline`** (a
cross-encoder fine-tuned on the train split; published, `PROJECT_SPEC.md` §5),
the target this project set itself, and it is measured on the same 8,956
queries. **The funnel beats the published baseline.** The caveat is cost: the
baseline is one cross-encoder pass, while this is a three-stage funnel ending
in an API call that takes 4.7 s for a single query.

The LLM arm was first measured on a frozen random sample of 2,000 test
queries, because each query is a paid call: 0.8855 [0.8799, 0.8910]
(`fine-rank-test.json`). The whole split was then run once, with the
configuration unchanged. The point estimate did not move, and the interval
halved.

Without the LLM, the coarse LambdaMART ranker alone scores **0.8579 [0.8551,
0.8611]** (`coarse-rank-test.json`). Its interval contains 0.8562, so it
**matches** the target rather than beating it. Re-ranking its top 10 with a
larger fine-tuned cross-encoder, `bge-reranker-base`, reaches **0.8695
[0.8667, 0.8724]** with no API call at all (`distill-test.json`, Section 8).
That also lies wholly above the target, at 72.5 ms a query.

The floor is computed every time, never quoted: 0.7467 on the full test split
with this project's gains and discount (`sbert-title-test.json`), against the
published SQID figure of 0.7483.

| baseline (published, §5) | NDCG | here |
|---|---|---|
| Random — the published SQID figure | 0.7483 | measured 0.7467 on the same split |
| CLIP text, zero-shot | 0.8107 | not re-run |
| CLIP image, zero-shot | 0.8225 | not re-run |
| SBERT text, zero-shot | 0.8292 | **reproduced: 0.8294 [0.8265, 0.8325]** |
| `ESCI_baseline`, fine-tuned cross-encoder | 0.8562 | the target |
| KDD Cup 2022 winner (context, not a target) | 0.9043 | — |

## 3. The funnel, stage by stage

**Evaluation foundation.** Zero-shot SBERT reproduces the published 0.8292 at
0.8294 [0.8265, 0.8325], +0.0827 [+0.0802, +0.0855] over the floor — the check
that the loader, the metric and the floor agree with the literature before
anything else was built (`sbert-title-test.json`).

**Enrichment.** ESCI-S covers 0.8959 of the 482,105 re-ranking products
(`esci-s-coverage.json`). Only 0.7753 have an image URL and 0.7750 an image
embedding (`image-embeddings.json`): about three quarters, not the 91.5% the
scrape's headline suggests.

**Stages 0–1, recall** — fold 0 (4,130 validation queries) against the full
catalogue, `recall.json`:

| channel | Recall@100 |
|---|---|
| BM25 | 0.4890 |
| dense text (SBERT) | 0.4608 |
| CLIP image | 0.1971 |
| RRF of all three | **0.5551** |
| RRF of all three, LLM-rewritten query | 0.5599 (ties; see Ablation 1) |

**Stage 2, coarse rank.** LightGBM `lambdarank` over 47 query × product features
in eight groups — retrieval scores, text overlap, image similarity, ESCI-S
behavioural values, categories, attributes and presence flags — trained with
ESCI's own gain mapping. **0.8579 [0.8551, 0.8611]** on the full test split
(`coarse-rank-test.json`).

**Stage 3, fine rank** — re-ranking the coarse ranker's top 10 over the whole
test split, `fine-rank-test-full.json`:

| arm | NDCG | vs coarse-only | latency, one query |
|---|---|---|---|
| coarse only | 0.8579 | | — |
| + cross-encoder, fine-tuned | 0.8616 | +0.0037 [+0.0016, +0.0058] | 20 ms |
| + larger cross-encoder (`bge-reranker-base`), fine-tuned — Section 8 | 0.8695 | +0.0116 [+0.0095, +0.0138] | 72.5 ms |
| **+ LLM listwise** | **0.8855** | **+0.0276 [+0.0255, +0.0296]** | 4,713 ms |

The LLM's single-query latency comes from the uncached run on the sample
(`fine-rank-test.json`); the full-split report reads a warm cache and so
cannot time it. The paid pass over the split (`llm-rerank-test-full.json`)
made 6,958 calls in 2.3 hours at four in flight. They cost 3.16M prompt and
2.66M completion tokens — 454 and 382 a call, 334 of them reasoning. Seven
windows got no usable ranking on either attempt and keep the coarse order,
counted. The LLM buys about seven times the cross-encoder's gain for over two
hundred times its latency.

**Stage 4, blend.** Nothing beats the LLM alone — Section 5.

## 4. The ablation table

`ablation-table.json`, verbatim. Every row is "A − B": a negative number means
B won.

| # | ablation | metric | scope | n | delta | |
|---|---|---|---|---|---|---|
| 1 | LLM-rewritten − raw query | Recall@100 | fold 0, full corpus | 4,130 | +0.0048 [-0.0007, +0.0107] | ties |
| 2a | dense+BM25 − dense-only (RRF) | Recall@100 | fold 0, full corpus | 4,130 | +0.0895 [+0.0831, +0.0958] | significant |
| 2b | dense+BM25+image − dense+BM25 (RRF) | Recall@100 | fold 0, full corpus | 4,130 | +0.0048 [+0.0008, +0.0088] | significant |
| 3 | text+image − text-only features | NDCG | test | 8,956 | +0.0075 [+0.0059, +0.0090] | significant |
| 4 | behavioural values − text+presence indicators | NDCG | test | 8,956 | +0.0036 [+0.0023, +0.0051] | significant |
| 5a | pointwise classifier − lambdarank | NDCG | test | 8,956 | -0.0085 [-0.0101, -0.0069] | significant |
| 5b | pointwise regression − lambdarank | NDCG | test | 8,956 | -0.0096 [-0.0112, -0.0079] | significant |
| 6a | coarse+cross-encoder − coarse-only | NDCG | test | 8,956 | +0.0037 [+0.0016, +0.0058] | significant |
| 6b | coarse+LLM listwise − coarse-only | NDCG | test | 8,956 | +0.0276 [+0.0255, +0.0296] | significant |
| 7 | learned fusion − fixed global weight | NDCG | test | 8,956 | -0.0011 [-0.0023, +0.0000] | ties |

> Rows 1–2 and 3–7 are **not comparable to each other**: they were measured on different query populations with different metrics. The scope and n columns say which.

The two contributions this project set out to measure:

- **Images help (Ablation 3).** Image features add +0.0075 to the coarse
  ranker. That was an open question in prior work, which only combined
  zero-shot CLIP scores with a hand-set weight. In recall the image channel
  helps far less (+0.0048, Ablation 2b).
- **Behavioural features help, a little (Ablation 4).** ESCI-S's values —
  price, stars, ratings, rank, categories, attributes — add +0.0036 over a
  control that already carries their *presence* flags. The flags alone are an
  artefact: "the 2022 scrape failed on this page" is not known at serving
  time, and on their own they are worth +0.0155 over the floor.

## 5. Stage 4 — the blend

**No blend beats the best stage it contains.** On the whole test split all
three strategies lose to the LLM alone, significantly, as they did on fold 0
(`blend-test-full.json`, `blend.json`). On the 2,000-query sample they only
tied (`blend-test.json`); the larger split resolves that tie as a loss.

| arm | fold 0 (n=4,130) | vs LLM alone | test split (n=8,956) | vs LLM alone |
|---|---|---|---|---|
| coarse only | 0.8519 | | 0.8579 | |
| + cross-encoder | 0.8587 | | 0.8616 | |
| **+ LLM** | **0.8814** | | **0.8855** | |
| fixed-weight RRF, 1:1:4 | 0.8800 | −0.0014 [−0.0023, −0.0006] | 0.8844 | −0.0011 [−0.0016, −0.0004] |
| learned combiner | 0.8793 | −0.0021 [−0.0033, −0.0010] | 0.8842 | −0.0013 [−0.0020, −0.0006] |
| per-query selector | 0.8805 | −0.0009 [−0.0016, −0.0002] | 0.8849 | −0.0006 [−0.0009, −0.0003] |
| *oracle: best arm per query* | *0.9076* | *+0.0262* | *0.9112* | *+0.0257 [+0.0243, +0.0269]* |

Fold-0 numbers for the learned arms are cross-fitted, so no query is scored by
a model that trained on it. The fixed weights were chosen on fold 0 itself, so
that fold-0 row is optimistic — and it still lost.

The headroom is real. Picking the best arm per query with the labels (the
oracle, a ceiling rather than a method) would add +0.0257. The selector is the
only strategy shaped to capture it, and it cannot (`blend-test-full.json`,
`selector_routes`). It sends 97.9% of test queries to the LLM. On the 184 it
sends elsewhere it is better 53 times and worse 105, a mean of −0.0278; fold 0
shows the same pattern (`blend.json`). Nothing visible without the labels —
window size, how far apart each arm's scores are, how much the arms agree —
predicts which queries the LLM will get wrong. So the best a selector can
learn is "trust the LLM", and that *is* the LLM.

## 6. Error analysis

All on fold 0 (4,130 queries), from `error-analysis.json`.

**By category.** Each query takes the most common top-level category of its
judged products. Categories with fewer than 100 queries are pooled into
"(other)", because a four-query row swings by noise alone; 55 queries have no
category. The LLM's gain over the coarse ranker is significant in every
category, and the intervals overlap, so no category is a failure mode of its
own.

| category | n | coarse | +cross-encoder | +LLM | LLM − coarse |
|---|---|---|---|---|---|
| Clothing, Shoes & Jewelry | 868 | 0.8598 | 0.8593 | 0.8875 | +0.0277 [+0.0207, +0.0347] |
| Home & Kitchen | 520 | 0.8481 | 0.8491 | 0.8751 | +0.0270 [+0.0189, +0.0349] |
| Electronics | 319 | 0.8411 | 0.8642 | 0.8793 | +0.0382 [+0.0273, +0.0504] |
| Sports & Outdoors | 228 | 0.8613 | 0.8710 | 0.8873 | +0.0260 [+0.0098, +0.0428] |
| Books | 224 | 0.8606 | 0.8668 | 0.8809 | +0.0203 [+0.0062, +0.0346] |
| Toys & Games | 219 | 0.8503 | 0.8718 | 0.8883 | +0.0380 [+0.0252, +0.0529] |
| Health & Household | 217 | 0.8262 | 0.8320 | 0.8491 | +0.0229 [+0.0073, +0.0374] |
| Tools & Home Improvement | 211 | 0.8447 | 0.8575 | 0.8841 | +0.0394 [+0.0272, +0.0509] |
| Beauty & Personal Care | 189 | 0.8676 | 0.8623 | 0.8921 | +0.0245 [+0.0120, +0.0380] |
| Automotive | 146 | 0.8617 | 0.8608 | 0.8800 | +0.0184 [+0.0051, +0.0320] |
| Patio, Lawn & Garden | 131 | 0.8345 | 0.8583 | 0.8683 | +0.0337 [+0.0183, +0.0513] |
| Cell Phones & Accessories | 124 | 0.8505 | 0.8598 | 0.8931 | +0.0426 [+0.0218, +0.0654] |
| Office Products | 119 | 0.8439 | 0.8508 | 0.8722 | +0.0283 [+0.0137, +0.0434] |
| (other) | 560 | 0.8568 | 0.8671 | 0.8885 | +0.0317 [+0.0244, +0.0390] |

**Where images help.** This is Ablation 3's own delta, split by the share of
each query's candidates that carry an image:

| image coverage | n | text+image − text-only |
|---|---|---|
| under half | 583 | +0.0033 [−0.0017, +0.0087] — ties |
| half to 90% | 1,549 | +0.0070 [+0.0036, +0.0106] |
| 90% to under 100% | 949 | +0.0054 [+0.0012, +0.0099] |
| **every candidate** | **1,049** | **+0.0065 [+0.0019, +0.0112]** |

The last row is the clean test. Whether a product *has* an image is itself a
weak ranking signal, but it can only reorder a query whose candidates differ
in it. Where every candidate has an image it is constant, and images still
help — so the gain comes from the image similarity itself, not from its
presence. Only the lowest-coverage stratum ties. The intervals overlap, so
"images help more where there are more images" is consistent with the data
but not established by it.

**Where the LLM fails.** Against the coarse ranker, query by query, the LLM
is better on 2,437 queries (59.0%), worse on 1,089 (26.4%) and identical on
604. It gains +0.0772 on average where it helps and loses −0.0607 where it
hurts. The damaged queries are ones the coarse ranker had already ranked well
(its NDCG averaged 0.873 on them, against 0.829 where the LLM helped). They
barely differ in candidate count, window size, image coverage, query length
(3.99 words against 3.94) or share of Exact products in the window. The damage
comes with a strong baseline rather than with a kind of query — which is
exactly why Section 5's selector found nothing to route on.

## 7. What did not work

Measured losses and ties, reported as results:

- **LLM query rewriting (Ablation 1) ties** on Recall@100, +0.0048
  [−0.0007, +0.0107], and *costs* precision at the top: Recall@10 falls from
  0.2657 to 0.2571 (`recall.json`).
- **Images barely help recall.** Adding the image channel to dense + BM25 buys
  +0.0048 at Recall@100 (Ablation 2b), against +0.0895 for adding BM25 to dense.
- **Learned fusion does not beat a hand-tuned weight (Ablation 7).** It ties on
  test, −0.0011 [−0.0023, +0.0000], and loses on fold 0, −0.0023 [−0.0040,
  −0.0008] (`coarse-rank.json`). The prior work's single global weight matched
  or beat the learned fusion on both.
- **Zero-shot cross-encoders lose to the coarse ranker**, −0.0063 [−0.0085,
  −0.0041] on the test split. Fine-tuning is not an optimisation of that arm;
  it is the whole arm.
- **The listwise loss does not earn its keep at Stage 3.** Fine-tuned with
  `LambdaLoss` the cross-encoder scores 0.8616; with plain binary cross-entropy,
  which trains several times faster, 0.8612 (`fine-rank-test-full.json`). That
  reverses Ablation 5, where lambdarank beat pointwise by +0.0085 to +0.0096 at
  the coarse stage.
- **The cross-encoder → LLM cascade is redundant**: 0.8847 against the LLM
  alone's 0.8855 on the 2,000-query sample (`fine-rank-test.json`). It doubles
  the LLM's calls, so the full-split run left it out.
- **No blend beats its best input (Section 5)**, while +0.0257 of headroom goes
  unclaimed.
- **The LLM's ordering did not beat the labels as a training target, at pilot
  scale (Section 8).** Trained on the same fold-0 queries, a cross-encoder given
  the LLM's ordering ties one given the labels from both starting points, so
  the paid distillation was not run. The pilot is weak evidence beyond that:
  no target lifted its student at all.
- **The windows-only recipe loses.** The top-10 windows alone, with `RankNetLoss`
  at batch 16, costs the MiniLM cross-encoder −0.0035 [−0.0045, −0.0026]
  against the landed recipe: every judged pair, `LambdaLoss`, batch 8.
- **The obvious reading of the behavioural ablation was wrong.** Subtracting
  the presence-flags-only arm reports the behavioural features at −0.0107,
  worse than useless (`coarse-rank-test.json`). That subtraction assumes the
  presence effects add up, and they do not. The +0.0036 in the table comes from
  a control arm instead.
- **Storage is over budget.** The BM25 index and the three embedding stores
  take 3.48 GB against the spec's 1.0 GB (`recall.json`), 0.40 GB of it a
  redundant image store.

## 8. Distilling the LLM

**The distillation was not run. In a fold-0 pilot, the LLM's ordering taught
a cross-encoder no more than the labels did. What did move the needle was free:
a larger cross-encoder trained on the labels.** Plan 8 set out to keep most of
the LLM's +0.0276 at cross-encoder latency. Doing that means paying the LLM to
rank the 12,519 training-fold windows, which is about 4.2 hours of calls
(projected, 5.7M prompt plus 4.8M completion tokens).

It spent that only behind a gate, written into code before the committed
pilot ran. A teacher target had to beat the true labels on the same queries
with an interval wholly above zero. The rule was not written blind: two
scratch pilots of the same comparison, run while the plan was being designed,
had already tied or lost, and the plan says so. Fold 0 already had the LLM's answers cached, so
the pilot trained the same student there, cross-fitted in halves: the labels,
the LLM's ordering, or the labels with the LLM breaking ties inside a grade.
It did this from the public checkpoint and from the cross-encoder already
trained (`distill-pilot.json`):

| starting from | LLM ordering − labels | labels, LLM tie-breaks − labels |
|---|---|---|
| the public checkpoint | +0.0007 [−0.0009, +0.0020] | +0.0006 [−0.0006, +0.0017] |
| the trained cross-encoder | −0.0009 [−0.0023, +0.0004] | −0.0000 [−0.0010, +0.0010] |

All four tie, so no paid call was made. That decides the spend, not the
general question. At 2,065 training queries a half, no target lifted its
student: every arm from the public checkpoint (0.8493–0.8499) sits below the
coarse ranker's 0.8519, and every arm from the trained model sits below that
model unchanged, 0.8587. A pilot where nothing helps cannot rank targets
finely, and a teacher could still pay off at a scale this one did not test.
The teacher is better than every stage but still noisy. On mixed-grade window pairs the LLM orders 0.7451 the
right way round, against 0.6589 for the cross-encoder and 0.6442 for the
coarse ranker (`distill-test.json`). The labels are right on all of them, and
they already exist for every training query. The teacher adds a noisier
target, not more data.

**The training windows had to be carved afresh.** The coarse ranker was
trained on the same folds a student would train on. Its own ordering of those
folds scores 0.9035, puts an Exact product on top of 87.7% of windows, and
shares only 34.5% of its windows with an ordering from models that never saw
those folds. That ordering scores 0.8534 with 71.2% Exact on top, close to
fold 0's 0.8519 (`stage2-oof.json`). A student trained on the in-sample
windows would have learned from an easier problem than the one it meets. Both
free arms below train on the out-of-fold windows.

On all 8,956 test queries, measured once (`distill-test.json`):

| arm | NDCG | vs the cross-encoder | share of the LLM's gain | latency, single / batched |
|---|---|---|---|---|
| coarse only | 0.8579 [0.8551, 0.8611] | | 0% | — |
| + cross-encoder (MiniLM, whole query groups) | 0.8616 [0.8588, 0.8647] | | 13% | 19.7 / 9.5 ms |
| + MiniLM, top-10 windows only | 0.8581 [0.8552, 0.8612] | −0.0035 [−0.0045, −0.0026] | 1% | 21.5 / 11.3 ms |
| **+ `bge-reranker-base`, top-10 windows only** | **0.8695 [0.8667, 0.8724]** | **+0.0079 [+0.0062, +0.0097]** | **42%** | **72.5 / 53.4 ms** |
| + LLM listwise | 0.8855 [0.8827, 0.8883] | | 100% | 4,713 ms |

- **The windows-only recipe loses.** The MiniLM backbone trained on only the
  coarse ranker's top-10 windows ties the coarse ranker (+0.0002 [−0.0019,
  +0.0024]) and loses to the landed recipe. That comparison changes three
  things at once — the windows, the loss (`RankNetLoss` against `LambdaLoss`)
  and the batch (16 against 8) — so it does not isolate the windows. A scratch
  run before the plan put `LambdaLoss` on the same windows at the same score,
  but that run is not committed. It is also not the contrastive hard-negative
  training that `PROJECT_SPEC.md` §7.4 cites as matching distillation. It only
  shows that the obvious cheap version of it does not help here.
- **A larger backbone pays.** `bge-reranker-base` trains on the same windows,
  with the same loss and labels, at batch 8 rather than 16. It lands +0.0116
  [+0.0095, +0.0138] over the coarse ranker: three times the MiniLM
  cross-encoder's gain and 42% of the LLM's, at 72.5 ms rather than 4.7 s. It
  was measured on fold 0 first: 0.8650, +0.0063 [+0.0038, +0.0089] over the
  cross-encoder (`distill.json`). Its gain over the MiniLM windows arm mixes
  the backbone with a batch half the size, and so twice the optimiser steps;
  this plan does not separate the two.

It trained in 22 minutes on the RTX 3080. It could not use the whole-query-group
recipe: a scratch attempt ran under 0.2 queries a second while another job held
4 GB of the card, and it was not retried on the idle card. Whether the bge arm
pays the same windows-only penalty as the MiniLM is untested.

## 9. What would come next

- **Predict when the LLM is wrong, from the LLM.** Section 5 shows label-free
  features cannot. A signal from the model itself — ask twice with the window
  shuffled and measure how much the two answers agree — is the natural
  candidate. It doubles the LLM's cost: about another 454 + 382 tokens and
  1.2 s of batched time a query.
- **A larger student, on labels, on whole query groups.** In Section 8 a
  larger backbone paid where the teacher did not, at the scale measured.
  `bge-reranker-base` keeps 42% of the LLM's gain while training on windows
  alone. Two things are untried: training it on every judged pair, which needs
  the card to itself or a bigger one, and matching its batch to the MiniLM's,
  which would separate backbone from recipe. Past that, `bge-reranker-large` is
  the next size up.
- **Delete the redundant image store.** `data/embeddings/rerank/` is a strict
  subset of the catalogue store and costs 0.40 GB.
