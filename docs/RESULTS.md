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
ranker's top 10: NDCG 0.8855 [0.8799, 0.8910] on a frozen random sample of
2,000 test queries**, against a random floor of 0.7454 and the coarse ranker's
own 0.8576 on the same sample — a gain of +0.0278 [+0.0236, +0.0325]
(`fine-rank-test.json`).

That interval lies entirely above the **0.8562 `ESCI_baseline`** (a
cross-encoder fine-tuned on the train split; published, `PROJECT_SPEC.md` §5),
the target this project set itself. Two caveats keep it from being a clean
win:

- **Scope.** The baseline was reported over all 8,956 test queries; this is a
  2,000-query sample, because running the LLM over the rest of the split costs
  more than two hours of paid calls (Section 8). The sample is representative
  on the one arm measured both ways: the coarse ranker scores 0.8576 on it and
  0.8579 on the full split.
- **Cost.** The baseline is one cross-encoder pass. This is a three-stage
  funnel ending in an API call that takes 4.7 s for a single query.

**On the full 8,956-query test split** the best measured number is the coarse
LambdaMART ranker at **0.8579 [0.8551, 0.8611]** against a floor of 0.7468
(`coarse-rank-test.json`). Its interval contains 0.8562, so it **matches** the
target rather than beating it.

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

**Stage 3, fine rank** — re-ranking the coarse ranker's top 10 on the
2,000-query sample, `fine-rank-test.json`:

| arm | NDCG | vs coarse-only | latency, one query |
|---|---|---|---|
| coarse only | 0.8576 | | — |
| + cross-encoder, fine-tuned | 0.8619 | +0.0043 [+0.0002, +0.0086] | 22 ms |
| **+ LLM listwise** | **0.8855** | **+0.0278 [+0.0236, +0.0325]** | 4,713 ms |

The LLM costs 453 prompt and 381 completion tokens a query, 333 of them
reasoning. It buys about six times the cross-encoder's gain for about two
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
| 6a | coarse+cross-encoder − coarse-only | NDCG | test sample | 2,000 | +0.0043 [+0.0002, +0.0086] | significant |
| 6b | coarse+LLM listwise − coarse-only | NDCG | test sample | 2,000 | +0.0278 [+0.0236, +0.0325] | significant |
| 7 | learned fusion − fixed global weight | NDCG | test | 8,956 | -0.0011 [-0.0023, +0.0000] | ties |

> These rows are **not comparable to each other**: they were measured on three different query populations with two different metrics. The scope and n columns say which.

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

**No blend beats the best stage it contains.** On the test sample all three
strategies tie with the LLM alone; on fold 0 all three lose to it
significantly (`blend-test.json`, `blend.json`).

| arm | fold 0 (n=4,130) | vs LLM alone | test sample (n=2,000) | vs LLM alone |
|---|---|---|---|---|
| coarse only | 0.8519 | | 0.8576 | |
| + cross-encoder | 0.8587 | | 0.8619 | |
| **+ LLM** | **0.8814** | | **0.8855** | |
| fixed-weight RRF, 1:1:4 | 0.8800 | −0.0014 [−0.0023, −0.0006] | 0.8843 | −0.0012 [−0.0026, +0.0000] |
| learned combiner | 0.8793 | −0.0021 [−0.0033, −0.0010] | 0.8849 | −0.0006 [−0.0019, +0.0007] |
| per-query selector | 0.8805 | −0.0009 [−0.0016, −0.0002] | 0.8853 | −0.0002 [−0.0007, +0.0002] |
| *oracle: best arm per query* | *0.9076* | *+0.0262* | *0.9101* | *+0.0246 [+0.0222, +0.0272]* |

Fold-0 numbers for the learned arms are cross-fitted, so no query is scored by
a model that trained on it. The fixed weights were chosen on fold 0 itself, so
that fold-0 row is optimistic — and it still lost.

The headroom is real. Picking the best arm per query with the labels (the
oracle, a ceiling rather than a method) would add +0.0246. The selector is the
only strategy shaped to capture it, and fold 0 shows why it cannot
(`blend.json`, `selector_routes`). It sends 93.2% of queries to the LLM. On the
281 it sends elsewhere it is better 88 times and worse 146, a mean of −0.0128.
Nothing visible without the labels — window size, how far apart each arm's
scores are, how much the arms agree — predicts which queries the LLM will get
wrong. So the best a selector can learn is "trust the LLM", and that *is* the
LLM.

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
- **Zero-shot cross-encoders lose to the coarse ranker**, −0.0059 [−0.0097,
  −0.0019] on the test sample. Fine-tuning is not an optimisation of that arm;
  it is the whole arm.
- **The listwise loss does not earn its keep at Stage 3.** Fine-tuned with
  `LambdaLoss` the cross-encoder gains +0.0043 [+0.0002, +0.0086]; with plain
  binary cross-entropy +0.0040 [−0.0001, +0.0086], which trains several times
  faster. That reverses Ablation 5, where lambdarank beat pointwise by +0.0085
  to +0.0096 at the coarse stage.
- **The cross-encoder → LLM cascade is redundant**: 0.8847 against the LLM
  alone's 0.8855.
- **No blend beats its best input (Section 5)**, while +0.0246 of headroom goes
  unclaimed.
- **The obvious reading of the behavioural ablation was wrong.** Subtracting
  the presence-flags-only arm reports the behavioural features at −0.0107,
  worse than useless (`coarse-rank-test.json`). That subtraction assumes the
  presence effects add up, and they do not. The +0.0036 in the table comes from
  a control arm instead.
- **Storage is over budget.** The BM25 index and the three embedding stores
  take 3.48 GB against the spec's 1.0 GB (`recall.json`), 0.40 GB of it a
  redundant image store.

## 8. What would come next

- **Predict when the LLM is wrong, from the LLM.** Section 5 shows label-free
  features cannot. A signal from the model itself — ask twice with the window
  shuffled and measure how much the two answers agree — is the natural
  candidate. It doubles the LLM's cost: another 453 + 381 tokens and about
  1.2 s of batched time a query.
- **Distil the LLM into the cross-encoder**, to keep most of its gain at 22 ms.
  That needs LLM orderings for the 12,519 training-fold queries: about four
  hours of calls at the measured 1,179 ms a query (batched), and roughly 5.7M
  prompt plus 4.8M completion tokens.
- **Run the LLM over the full test split**, so the headline is directly
  comparable with 0.8562: the remaining 6,956 queries, about 2.3 hours and 3.2M
  prompt plus 2.7M completion tokens.
- **Delete the redundant image store.** `data/embeddings/rerank/` is a strict
  subset of the catalogue store and costs 0.40 GB.
