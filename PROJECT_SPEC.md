# Multi-Stage Multimodal Product Search Ranking on Amazon ESCI

A portfolio project building a full recall → coarse rank → fine rank → blend
search funnel on the Amazon Shopping Queries Dataset, enriched with images and
behavioural features.

---

## 1. Motivation

Most public e-commerce search projects stop at "embed the titles, do cosine
similarity, show top 10." Real ranking systems are multi-stage funnels that fuse
lexical, semantic, visual and behavioural signals, and they are evaluated with
graded relevance metrics rather than eyeballing.

This project builds the full funnel, and evaluates each stage with ablations
against published baselines.

Two things make it more than a re-implementation:

1. **Behavioural features.** The original ESCI dataset is text-only. Joining
   ESCI-S adds price, star rating, review count, category hierarchy and
   structured attributes — the numerical/categorical features real rankers use
   but public ESCI work mostly ignores.
2. **Multimodal + learned fusion.** Published multimodal work on this task
   (Al Ghossein et al., SIGIR eCom '24) showed images help, but only via
   zero-shot CLIP embeddings combined by a single hand-tuned global weight,
   and only over the test split. Those authors explicitly listed fine-tuning
   and learned combination as out of scope / future work. Because ESCI-S
   provides image URLs across **both** splits, this project can train a
   LambdaMART model to fuse text + image + behavioural signals on the real
   train split — directly targeting that gap.

---

## 2. The Task

**ESCI Task 1 — Query-Product Ranking.** Given a query and its ~20 candidate
products, order them so relevant products come first.

Important framing: this is a **re-ranking** task, not end-to-end retrieval. Only
products with judgement labels are ranked. The recall stage is therefore
evaluated separately (Recall@k over the full ~1.2M product corpus) from the
ranking stages (NDCG over the judged candidate list).

### Labels

| Label | Meaning | Gain |
|---|---|---|
| **E** — Exact | Satisfies all query specifications | 1.0 |
| **S** — Substitute | Somewhat relevant; usable functional substitute | 0.1 |
| **C** — Complement | Doesn't fulfil query but complements an exact item | 0.01 |
| **I** — Irrelevant | Fails a central aspect of the query | 0.0 |

### Metric

NDCG with the gain mapping above, computed with `trec_eval` / `pytrec_eval`
(the official ESCI evaluation route), averaged over all test queries.

**Always report against the random-ordering floor.** Random scores ~0.748 on
this metric because ~44% of judgements are Exact. A headline number of "0.86"
means nothing without that floor attached. Report bootstrap confidence
intervals over queries, not bare point estimates.

> ⚠️ **Known bug in the official eval script.** Al Ghossein et al. (2024) found
> that `prepare_trec_eval_files.py` in `amazon-science/esci-data` swaps the
> relevance scores for **S** and **C** (line ~48). Use the corrected mapping
> above and cite this, since published numbers differ depending on which
> mapping was used. The
> Vespa blog's reported label distribution (Complement 35%, Substitute 4%) also
> looks consistent with this swap — Complement being 9× more common than
> Substitute is implausible for real e-commerce data. **Verify the label
> distribution yourself on load before trusting any downstream number.**

---

## 3. Datasets

Only two datasets are used. Both are keyed on **ASIN** (Amazon Standard
Identification Number), which is what makes the join possible.

### 3.1 Amazon ESCI (Shopping Queries Dataset) — the spine

- **Source:** https://github.com/amazon-science/esci-data
- **Paper:** Reddy et al., 2022 — arXiv:2206.06588
- **Role:** provides the queries and the relevance judgements. Without this
  there is no task.
- **Note:** hosted via Git LFS; also mirrored on Hugging Face.

Schema (two joined tables):

| Table | Fields |
|---|---|
| Judgements | `example_id`, `query`, `query_id`, `product_id`, `product_locale`, `esci_label`, `small_version`, `large_version`, `split` |
| Products | `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, `product_locale` |

**English split statistics** (`small_version=1`, Task 1):

| Split | Queries | Judgements |
|---|---|---|
| Train | 20,888 | 419,653 |
| Test | 8,956 | 181,701 |

~20 judgements per query on average; ~9 relevant (Exact) products per query.
~1.2M unique products indexed across both English splits.

### 3.2 ESCI-S — product metadata enrichment

- **Source:** https://github.com/shuttie/esci-s
- **License:** Apache-2.0
- **Size:** 3.4 GB Zstd-compressed, single file (S3 or GDrive)
- **Coverage:** 1,661,908 ASINs = **91.5%** of ESCI's 1,814,925 ASINs,
  **across both train and test splits**
- **Role:** everything ESCI leaves out. The author's stated motivation is that
  ESCI "leaves behavioral, numerical and categorical ranking features on the
  side."

Fields added per product:

| Field | Type | Use in this project |
|---|---|---|
| `image` | URL | CLIP image embeddings (Stage 1, 2) |
| `stars` | string, e.g. "4.3 out of 5 stars" | behavioural feature (parse to float) |
| `ratings` | string, e.g. "1,116 ratings" | behavioural feature (parse to int) |
| `price` | string | behavioural feature (parse; often empty) |
| `category` | list, top→bottom | category-hierarchy match depth |
| `attrs` | map | attribute-overlap features |
| `info` | map | extended attributes, incl. Best Sellers Rank |
| `bullets` | list | BM25 field |
| `description` | string | BM25 field |
| `reviews` | list of {stars, title, date, text} | optional review-text signal |
| `type` | "product" \| "book" | categorical feature (7% are books) |
| `template` | page layout | proxy for top-level category |
| `formats` | map | optional |

Top categories by page template: apparel 15.1%, kitchen 8.3%, home 5.9%,
home_improvement 5.3%, sports 5.2%, book 5.1%, beauty 4.7%, shoes 4.4%.

### 3.3 Images

Image URLs come from ESCI-S. They point at Amazon's CDN with transform
parameters baked in, e.g.:

```
https://m.media-amazon.com/images/I/81bdoltQWVL.__AC_SY300_SX300_QL70_FMwebp_.jpg
```

`_AC_SY300_SX300_QL70_FMwebp_` = max 300px per side, quality 70, WebP. So each
image is ~10–30 KB, served from a CDN edge.

**This is a static asset fetch, not page scraping.** The expensive, block-prone
step (loading and parsing product pages to *discover* the URL) was already done
by ESCI-S. Measured throughput: ~1,000–4,200 images/minute at 10–14 concurrency,
bounded by local bandwidth.

#### ⚠️ The URLs do not work as shipped — strip the retired CDN bucket

About **half** of ESCI-S image URLs route through a CDN A/B-test bucket that
Amazon retired after the 2023 scrape:

```
https://m.media-amazon.com/images/W/WEBP_402378-T1/images/I/61BRLXQ4CqL.__AC_SX300_..._.jpg
                          ^^^^^^^^^^^^^^^^^^^^^^^^ dead since the scrape
```

These return HTTP **400**, not 404 — the path is malformed, the product is fine.
Dropping the segment fixes every one of them:

```python
re.sub(r"/images/W/[^/]+/images/I/", "/images/I/", url)   # src/esci_images.py
```

Measured over three independent random samples (verified 2026-09-19):

| Sample | n | Resolves as-is | After the fix |
|---|---|---|---|
| Products only | 400 | 51.0% | 97.5% (residual = local timeouts) |
| Products + books, disjoint records | 600 | 49.0% | **100.0%** |
| — of those carrying the bucket segment | 306 | **0.0%** | **100.0%** |
| — of those without it (control) | 294 | 100.0% | 100.0% |

All 600 decoded as real images (WEBP/JPEG, median 8.9 KB, 300×300 dominant, no
placeholders). The substitution is a no-op on URLs that don't carry the segment,
so it is safe to apply unconditionally.

**This matters because the naive rate (≈51%) sits below this project's own
go/no-go threshold.** Without the fix, the multimodal premise looks dead on a
false negative.

#### Field names and real coverage

- **Products store the URL in `image`; books store it in `img`.** Books are ~7%
  of the corpus, and reading only `image` drops all of them silently. Books also
  use `desc`/`attr` where products use `description`/`attrs`.
- **~3.7% of ESCI-S rows are `type: "error"`** — scrape failures carrying only
  `asin`, `locale`, `error`, `template`. Filter them explicitly.
- **End-to-end image coverage is ~75%, not 91.5%:** 91.5% of ASINs present in
  ESCI-S × ~84.7% of those carrying an image URL × ~97.5%+ resolving. Report the
  75% figure, not the headline.
- Coverage is **even across splits** — train 84.8%, test 84.6% — and roughly
  label-balanced (83–85% for E/I/S). The "images on both splits" premise holds,
  and missingness is not obviously confounded with relevance.

Practical notes:
- ~150K images ≈ 1.5 GB transfer at ~10 KB each. If only embeddings are needed,
  embed in-flight and never persist the images (156K CLIP embeddings ≈ <500 MB).
- Write-as-you-go + skip-existing so an interrupted run resumes.
- Retry with backoff on 429/5xx, real User-Agent, cap concurrency.
- **The resolution gate is executable:** `python -m src.esci_images <file.zst>`
  samples live URLs and exits non-zero below 90%. Re-run it before any bulk
  fetch rather than trusting these numbers to keep holding.
- The GitHub `sample.json.gz` has **no `image` field at all** — it is stale
  against the real 3.4 GB file. Never validate the image pipeline against it.

### 3.4 What is deliberately NOT used

| Dataset | Why excluded |
|---|---|
| SQID | Precomputed CLIP embeddings for test-split products only (164,900). Superseded: ESCI-S gives image URLs across both splits, so images are generated in-house from one consistent pipeline. Its *paper* is still cited as a baseline (§5). |
| TREC Product Search | Images at scale, but pooled (sparser) judgements and a heavier eval toolchain. |
| Amazon-C4 | Complex natural-language queries, but no images, no graded labels, no train split. |
| Amazon Reviews 2023 | Same ASIN space, has user histories — relevant only if extending to personalised recsys. |
| ABO / Shopee | Catalogues without query-relevance judgements. Can't compute NDCG against search intent. |

---

## 4. The Funnel

### Stage 0 — Query Understanding

LLM-based rewrite / expansion before retrieval. Maps to the "intent analysis"
and "prompt optimisation" language in the target JDs. One LLM call per query.

**Measured by:** Recall@k lift vs. raw query.

### Stage 1 — Recall (hybrid)

Three channels, fused with Reciprocal Rank Fusion:

| Channel | Signal | Catches |
|---|---|---|
| BM25 over title + description + bullets | lexical | brands, model numbers, exact tokens |
| Dense text embeddings | semantic | paraphrase, synonymy |
| CLIP image embeddings | visual | attribute mismatches text misses |

**Measured by:** Recall@k over the ~1.2M corpus, per-channel and fused.

### Stage 2 — Coarse Rank (LambdaMART)

LightGBM with `objective: lambdarank` — a real LTR loss, not binary
classification. Requires a `group` array giving the number of candidates per
query.

Feature groups:

- **Retrieval scores:** BM25 score, dense text similarity, CLIP image similarity
- **Behavioural (ESCI-S):** `price`, `stars`, `ratings`, Best Sellers Rank
- **Categorical (ESCI-S):** category-hierarchy match depth, `template`, `type`
- **Attribute overlap:** query-term ∩ `attrs` / `info` keys and values
- **Text statistics:** title length, description length, brand match, colour match

### Stage 3 — Fine Rank (cross-encoder / LLM)

Two variants, compared head to head on the top-K from Stage 2:

- **(a) Fine-tuned modern cross-encoder** (BGE-reranker / mxbai-rerank family)
- **(b) LLM listwise reranking** (RankGPT-style sliding window)

### Stage 4 — Blend

Final ordering: weighted combination, or a small learned combiner over stage
scores. Compare against each single stage alone.

---

## 5. Baselines to Beat

All on the English test split, same metric. These are **published prior-work
numbers cited for comparison** — no external dataset is consumed to obtain them.
Where a baseline is cheap to rerun in-house (e.g. zero-shot SBERT/CLIP), rerun
it on the same pipeline rather than quoting, so the comparison is apples to
apples.

| Method | NDCG | Note |
|---|---|---|
| Random | 0.7483 | The floor. Always report this. |
| CLIP_text (zero-shot) | 0.8107 | Al Ghossein et al. 2024 |
| CLIP_image (zero-shot) | 0.8225 | Al Ghossein et al. 2024 — images alone beat CLIP text |
| SBERT_text (zero-shot) | 0.8292 | Al Ghossein et al. 2024, `all-MiniLM-L12-v2` |
| Best zero-shot text+image combo | ~0.83 | +2.41% over CLIP_text, single hand-tuned global weight |
| **ESCI_baseline** | **0.8562** | `ms-marco-MiniLM-L-12-v2` cross-encoder, fine-tuned on SQD train. **Primary target.** |
| KDD Cup 2022 winner | 0.9043 | Ensembles of fine-tuned multilingual transformers + augmentation + self-distillation. Cite as context, **not** as a target. |

**Realistic goal: beat 0.8562.** Reaching ~0.90 requires large fine-tuned
transformer ensembles, which is out of scope on free compute. Beating the
standard baseline with a well-engineered funnel using features nobody else
combined is both achievable and honest.

---

## 6. Experiments / Ablations

The ablations are the deliverable. Each isolates one contribution:

| # | Ablation | Isolates | Metric |
|---|---|---|---|
| 1 | Raw query vs. LLM-rewritten query | Stage 0 value | Recall@k |
| 2 | Dense-only vs. +BM25 vs. +image (RRF) | Hybrid recall value | Recall@k |
| 3 | Text-only vs. text+image features | **Multimodal contribution** (open question in prior work) | NDCG |
| 4 | With vs. without ESCI-S behavioural features | **Behavioural contribution** (not published on ESCI) | NDCG |
| 5 | Pointwise classifier vs. `lambdarank` | LTR objective value | NDCG |
| 6 | Coarse-only vs. +cross-encoder vs. +LLM listwise | Per-stage value | NDCG, latency, cost |
| 7 | Learned fusion vs. a single fixed global text/image weight | **Learned-combination contribution** | NDCG |

Plus: bootstrap CIs over queries; per-category error analysis; where images help
vs. hurt; failure cases for the LLM reranker.

---

## 7. Methods Worth Trying (researched)

### 7.1 Stage 0 — Query understanding

| Method | Source | Notes |
|---|---|---|
| **HyDE** (Hypothetical Document Embeddings) | Gao et al. | LLM drafts an "ideal product description", embed that instead of the query. Consistently improves zero-shot retrieval. |
| **Multi-Query expansion** | LangChain pattern | Generate 4–8 paraphrases, retrieve union. |
| **Doc2Query** (document-side expansion) | Nogueira et al.; `pyterrier_doc2query` | Expand *products* with predicted queries before indexing. **Caveat:** prone to hallucination; filtering poor generated queries with a relevance model improved retrieval by up to 16% while cutting index size 48%. |
| **Elastic's QR taxonomy** | Elastic Search Labs | Generic rephrasing / pseudo-answer / noise reduction / entity enrichment / typo fixing. Evaluated with NDCG@10 + Recall@10/50 — a good template for ablation 1. |
| **Hint-Augmented Re-ranking** | arXiv:2511.13994 | LLM query decomposition feeding reranking, product-search specific. **Closest prior work to this project — read before building.** Also documents the practical chunking fix for listwise context limits (50 products ≈ 28k tokens → 5 chunks of 10, take top 2 each, merge). |

### 7.2 Stage 1 — Recall / retrieval

| Method | Source | Notes |
|---|---|---|
| **RRF hybrid fusion** | standard | BM25 + dense. Baseline hybrid. |
| **Late fusion via linear projection** | arXiv:2403.11593 (fashion product matching) | Train a linear projection over *precomputed* CLIP text+image embeddings with large-batch contrastive loss. Full training <10 min on a single GPU. **Highest value-per-hour option here.** Also found CLIP encoders beat DINOv2 (visual) and multilingual USE (text). |
| **MLP-based late fusion** | arXiv:2508.20013 | CLIP + MLP late fusion beat early fusion and attention fusion. Notably, **attention-based fusion underperformed despite higher complexity** — don't assume fancier is better. |
| **Mixture-of-modality-experts fusion** | arXiv:2603.04836 (Target) | Two-tower with lightweight MoE fusion over CLIP text/image; argues two-stage alignment + domain fine-tuning both matter. |
| **Gated cross-modal fusion** | UniECS, arXiv:2508.13843 | Unified multimodal e-commerce search framework. |

### 7.3 Stage 2 — Coarse rank / LTR

| Method | Source | Notes |
|---|---|---|
| **LambdaMART via LightGBM** | Burges 2010; Ke et al. 2017 | `objective: lambdarank`, `group` = candidates per query, eval `ndcg@k`. **Use Group K-Fold for CV** — never split within a query group. |
| **`rank_xendcg`** | LightGBM | Alternative listwise objective; cheap A/B against `lambdarank`. |
| **Vespa ESCI LTR series** | blog.vespa.ai/improving-product-search-with-ltr | Same dataset, same task, GBDT + neural features. Directly comparable prior work — read for feature design. |
| **GBDT vs. transformer LTR** | arXiv:2507.20753 | Industry finding: transformer LTR can beat GBDT offline but online results are mixed. Useful framing for the writeup. |

### 7.4 Stage 3 — Fine rank / reranking

| Method | Source | Notes |
|---|---|---|
| **RankGPT listwise prompting** | Sun et al., 2023 | The canonical zero-shot listwise recipe; sliding window with bubble-sort pattern to mitigate cross-window blindness. |
| **RankZephyr / RankVicuna** | arXiv:2312.02724 | Open-weight listwise rerankers; exact prompt template published. |
| **Rank-without-GPT** | Zhang et al., ECIR 2025 | Listwise rerankers without GPT dependency; beats GPT-3.5-based by 13%, reaches 97% of GPT-4-based. Note: **out-of-the-box open LLMs often emit ill-formed listwise output** — fine-tuning is usually needed for valid rankings. |
| **Setwise / TourRank** | `llmranker` pypi | Alternative LLM ranking strategies with different call/accuracy trade-offs. TourRank is most robust to candidate input order. |
| **Contrastive learning vs. knowledge distillation** | arXiv:2507.08336 | **Important negative result:** single-stage contrastive fine-tuning with hard negatives matched or beat distillation-augmented pipelines — multi-stage/distillation gave *no consistent improvement*. Do the simple thing first. |
| **Margin-MSE distillation** | Walmart, 2025 | 7B LLM teacher → BERT-base student on 170M teacher-labelled pairs; student matched/slightly beat teacher on NDCG. Relevant if pursuing distillation anyway. |
| **Existing ESCI reranker (reference point)** | HF `albertobarnabo/ecommerce-product-search-reranker` | 33M params, reranks 50 candidates in ~45 ms on laptop CPU, fine-tuned on 427,655 ESCI judgements. Reports pool NDCG@10 0.7601 vs. random floor 0.574 — **a good model of how to report results honestly.** |

### 7.5 Techniques from KDD Cup 2022 winners

Worth knowing, mostly out of compute scope, but cheap versions exist:

- Multilingual/English PLM backbones (RoBERTa-large, DeBERTa-v3-large, XLM-R)
- Self-distillation, pseudo-labelling, label smoothing
- **FGM adversarial training** (cheap regularisation, worth an ablation)
- Continued MLM pre-training on domain corpus before fine-tuning
- Ranking loss instead of regression loss — the winners note regression can only
  fit four fixed constants for four classes, which wastes the graded structure
- Model ensembling (took 1st place from 0.9022 → 0.9057 public LB)

One winner's own stated future direction: *"this challenge is an NLP competition
and uses text only, and we can also continue to study how to use multimodal
information in product search ranking."* — which is exactly this project.

---

## 8. Build Order

Get a working end-to-end baseline before optimising anything.

1. Load ESCI, verify label distribution, implement NDCG + random floor, reproduce
   a published zero-shot number (e.g. SBERT_text ≈ 0.8292). **Nothing else until
   the metric is trustworthy.**
2. Download ESCI-S, join on ASIN, measure join coverage and missingness bias.
3. Sample-test image URLs (`python -m src.esci_images` — gate passes at ~100%
   once the retired CDN bucket is stripped, §3.3); then bulk-fetch + embed.
4. BM25 baseline → dense baseline → RRF hybrid (Recall@k).
5. LightGBM `lambdarank` with text features only → add behavioural → add image.
6. Cross-encoder fine-tune on top-K.
7. LLM listwise variant + cost/latency comparison.
8. Ablation table, bootstrap CIs, error analysis, writeup.

**Discipline:** hold out a validation fold for all model selection. Do not tune
on test. With ~9K test queries and a random floor of 0.748, the gap between a
real improvement and noise can be a couple of NDCG points — a method that ties
the baseline is a legitimate, reportable result.

---

## 9. Compute & Cost

| Item | Plan |
|---|---|
| Image embedding | Kaggle/Colab free GPU (CLIP inference only) |
| LightGBM | CPU, seconds to minutes |
| Cross-encoder fine-tune | Colab free T4, LoRA/QLoRA if needed |
| LLM reranking + query rewriting | API calls (the only real $ cost) |
| Storage | Embeddings <1 GB; ESCI-S 3.4 GB compressed |
| Hosting (optional demo) | HF Spaces free CPU tier |

---

## 10. References

- Reddy et al. (2022). *Shopping Queries Dataset: A Large-Scale ESCI Benchmark
  for Improving Product Search.* arXiv:2206.06588
- Al Ghossein, Chen & Tang (2024). *Shopping Queries Image Dataset (SQID).*
  SIGIR eCom '24, arXiv:2405.15190 — *cited for baseline NDCG numbers and the
  S/C eval-script bug; its data files are not used in this project*
- `shuttie/esci-s` — Extended metadata for Amazon ESCI (Apache-2.0)
- Sun et al. (2023). *RankGPT.*
- Pradeep et al. (2023). *RankZephyr.* arXiv:2312.02724
- Zhang et al. (2025). *Rank-without-GPT.* ECIR 2025
- Xu et al. (2025). *Distillation versus Contrastive Learning: How to Train Your
  Rerankers.* arXiv:2507.08336
- *Hint-Augmented Re-ranking.* arXiv:2511.13994
- *End-to-end multi-modal product matching in fashion e-commerce.* arXiv:2403.11593
- Lin et al. (2022). *A Winning Solution of KDD Cup 2022 ESCI Challenge.*
- Zhang et al. (2022). *A Semantic Alignment System for Multilingual
  Query-Product Retrieval.* arXiv:2208.02958
- Bergum, J.K. *Improving Product Search with Learning to Rank* (Vespa blog series)
