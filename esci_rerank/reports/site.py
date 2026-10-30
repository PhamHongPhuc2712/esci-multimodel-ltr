"""The project site: one static page generated from the committed results.

    python -m esci_rerank.reports.site example   # once, on a GPU: one validation query re-ranked by the 4B model
    python -m esci_rerank.reports.site           # site/index.html from docs/results/ only
    python -m esci_rerank.reports.site --check   # regenerate in memory and compare with the committed page

Like esci_rerank.reports.quality_latency_chart, nothing here types a number: every figure on
the page is read from a file under docs/results/, and tests/test_site.py holds
the page to that. The example query is drawn at random from validation fold
0 among those whose window mixes at least three labels, with a fixed seed, so
it is reproducible and not chosen for looking good.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from esci_rerank.reports.quality_latency_chart import chart_points

RESULTS = Path("docs/results")
EXAMPLE = RESULTS / "site-example.json"
PAGE = Path("site/index.html")
REPO = "https://github.com/PhamHongPhuc2712/esci-multimodel-ltr"
EXAMPLE_SEED = 13
EXAMPLE_ARM = "models/llm-rank/point-4b"
LABEL_NAMES = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}


# --- the example query (GPU, run once) ------------------------------------------

def pick_example(windows_, matrix, seed: int = EXAMPLE_SEED):
    """A fold-0 window with at least three distinct labels, drawn at random."""
    import random

    labels = {(str(q), p): l for q, p, l in
              zip(matrix["query_id"], matrix["product_id"], matrix["esci_label"])}
    eligible = sorted(
        (w for w in windows_ if len({labels[(w.query_id, d)] for d in w.window}) >= 3),
        key=lambda w: w.query_id,
    )
    if not eligible:
        raise ValueError("no fold-0 window mixes three labels")
    return random.Random(seed).choice(eligible), labels


def _example() -> int:
    from esci_rerank.blend.blend_report import per_query_ndcg
    from esci_rerank.rerank.cross_encoder import text_maps
    from esci_rerank.rerank.llm_point import load_point_model, rerank_point
    from esci_rerank.ranking.rank_report import qrels_from_frame
    from esci_rerank.blend.stage_signals import scope_windows

    windows_, matrix = scope_windows("fold0")
    window, labels = pick_example(windows_, matrix)
    query_text, doc_text = text_maps(matrix, Path("data/combined/products.parquet"),
                                     Path("data/combined/judgements.parquet"))
    query = {str(k): v for k, v in query_text.items()}[window.query_id]
    import pandas as pd

    titles = pd.read_parquet("data/combined/products.parquet", columns=["product_id", "product_title"])
    titles = dict(zip(titles["product_id"], titles["product_title"]))
    model, tok = load_point_model(EXAMPLE_ARM)
    # Every fold-0 window, so the one example sits beside how often the model
    # helps, hurts or leaves a query alone.
    query_text = {str(k): v for k, v in query_text.items()}
    orderings = rerank_point(model, tok, windows_, query_text, doc_text)
    qrels = qrels_from_frame(matrix)
    coarse = per_query_ndcg({w.query_id: list(w.window) for w in windows_}, windows_, qrels)
    model_ndcg = per_query_ndcg(orderings, windows_, qrels)
    deltas = [model_ndcg[q] - coarse[q] for q in coarse]
    reranked = orderings[window.query_id]
    before, after = coarse[window.query_id], model_ndcg[window.query_id]
    payload = {
        "scope": "validation fold 0", "seed": EXAMPLE_SEED, "model": EXAMPLE_ARM,
        "rule": "drawn at random among fold-0 windows mixing at least three labels",
        "query_id": window.query_id, "query": query,
        "products": {d: {"title": str(titles.get(d, "")), "label": labels[(window.query_id, d)]}
                     for d in window.window},
        "coarse_order": list(window.window), "reranked_order": reranked,
        "ndcg": {"coarse": before, "reranked": after},
        "tail": len(window.tail),
        "fold0": {"n_queries": len(deltas),
                  "improved": sum(d > 1e-9 for d in deltas),
                  "worsened": sum(d < -1e-9 for d in deltas),
                  "unchanged": sum(abs(d) <= 1e-9 for d in deltas),
                  "coarse_ndcg": sum(coarse.values()) / len(coarse),
                  "reranked_ndcg": sum(model_ndcg.values()) / len(model_ndcg)},
    }
    EXAMPLE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"query {query!r}: NDCG {before:.4f} -> {after:.4f}  -> {EXAMPLE}")
    return 0



# --- formatting ------------------------------------------------------------------

def _load(name: str, results_dir: Path = RESULTS) -> dict:
    return json.loads((Path(results_dir) / name).read_text(encoding="utf-8"))


def f4(x: float) -> str:
    return f"{x:.4f}"


def interval(i: dict) -> str:
    return f"[{i['low']:.4f}, {i['high']:.4f}]"


def signed(x: float) -> str:
    """+0.0078 or −0.0008, with a true minus sign."""
    return f"{x:+.4f}".replace("-", "−")


def delta_text(d: dict) -> str:
    return f"{signed(d['point'])} [{signed(d['low'])}, {signed(d['high'])}]"


def ms(x: float) -> str:
    return f"{x:,.1f}"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


# --- the hero: one query, before and after ----------------------------------------

ROW_H = 54


def rerank_figure(ex: dict) -> str:
    """Two ranked lists with a connector per product, coloured by its ESCI label."""
    before, after = ex["coarse_order"], ex["reranked_order"]
    n = len(before)
    where = {d: i for i, d in enumerate(after)}
    height = n * ROW_H
    paths = []
    for i, d in enumerate(before):
        j = where[d]
        y0, y1 = i * ROW_H + ROW_H / 2, j * ROW_H + ROW_H / 2
        label = ex["products"][d]["label"]
        paths.append(
            f'<path class="link l-{label}" style="--i:{i}" '
            f'd="M0 {y0:.1f} C50 {y0:.1f} 50 {y1:.1f} 100 {y1:.1f}"/>'
        )

    def column(order, side):
        rows = []
        for rank, d in enumerate(order, 1):
            prod = ex["products"][d]
            label = prod["label"]
            rows.append(
                f'<li class="item"><span class="rank">{rank}</span>'
                f'<span class="chip l-{label}" title="{LABEL_NAMES[label]}">{label}</span>'
                f'<span class="title">{esc(prod["title"])}</span></li>'
            )
        return f'<ol class="list {side}">' + "".join(rows) + "</ol>"

    f0 = ex["fold0"]
    change = ex["ndcg"]["reranked"] - ex["ndcg"]["coarse"]
    verdict = ("did worse, lifting Complements above Substitutes" if change < 0 and _c_over_s(ex)
               else "did worse" if change < 0 else "did better" if change > 0 else "changed nothing")
    return f"""
<figure class="rerank" aria-labelledby="rerank-caption">
  <div class="rerank-head">
    <p class="query">Query <q>{esc(ex["query"])}</q></p>
    <div class="cols-head">
      <p>Coarse ranker <span class="score">NDCG {f4(ex["ndcg"]["coarse"])}</span></p>
      <p>Re-ranked by the 4B model <span class="score">NDCG {f4(ex["ndcg"]["reranked"])}</span></p>
    </div>
  </div>
  <div class="rerank-body">
    {column(before, "left")}
    <svg class="links" viewBox="0 0 100 {height}" preserveAspectRatio="none" aria-hidden="true">{"".join(paths)}</svg>
    {column(after, "right")}
  </div>
  <figcaption id="rerank-caption">
    <span class="legend">{"".join(f'<span><span class="chip l-{k}">{k}</span> {v}</span>' for k, v in LABEL_NAMES.items())}</span>
    One validation query, {esc(ex["rule"])} with a fixed seed, not picked to flatter the model.
    On this one the 4B model {verdict}. Across all {f0["n_queries"]:,} validation queries it improved {f0["improved"]:,},
    worsened {f0["worsened"]:,} and left {f0["unchanged"]:,} unchanged, taking mean NDCG from
    {f4(f0["coarse_ndcg"])} to {f4(f0["reranked_ndcg"])}.
  </figcaption>
</figure>"""


def _c_over_s(ex: dict) -> bool:
    """Whether the re-ranking put a Complement above a Substitute the coarse order had below it."""
    def pairs(order):
        labels = [ex["products"][d]["label"] for d in order]
        return {(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))
                if labels[i] == "C" and labels[j] == "S"}
    return bool(pairs(ex["reranked_order"]) - pairs(ex["coarse_order"]))


# --- quality against latency ---------------------------------------------------------

SHORT = {
    "stage2+ce": "MiniLM cross-encoder (RTX 3080)",
    "stage2+ce_bge_groups@fp32": "bge-reranker-base, fp32",
    "stage2+ce_bge_groups@fp16": "bge-reranker-base, fp16",
    "stage2+ce_bge_large@fp32": "bge-reranker-large, fp32",
    "stage2+ce_bge_large@fp16": "bge-reranker-large, fp16",
    "stage2+qwen_point_zs@bf16": "Qwen3 0.6B, zero-shot",
    "stage2+qwen_point@bf16": "Qwen3 0.6B, fine-tuned",
    "stage2+qwen_list@bf16": "Qwen3 1.7B listwise",
    "stage2+qwen_point_4b@bf16": "Qwen3 4B, in-process",
    "stage2+llm": "hosted LLM (API)",
}


def _shape(name: str) -> str:
    if name.startswith("deployed:"):
        return "served"
    if name == "stage2+llm":
        return "hosted"
    if name.startswith("stage2+ce"):
        return "ce"
    return "llm"


def _marker(shape: str, x: float, y: float) -> str:
    if shape == "ce":
        return f'<circle class="m-ce" cx="{x:.1f}" cy="{y:.1f}" r="5"/>'
    if shape == "llm":
        return f'<rect class="m-llm" x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9"/>'
    if shape == "served":
        return (f'<path class="m-served" d="M{x:.1f} {y - 7:.1f} L{x + 7:.1f} {y:.1f} '
                f'L{x:.1f} {y + 7:.1f} L{x - 7:.1f} {y:.1f} Z"/>')
    return f'<circle class="m-hosted" cx="{x:.1f}" cy="{y:.1f}" r="5.5"/>'


def quality_chart(points) -> str:
    W, H, L, R, T, B = 960, 470, 64, 24, 18, 52
    xs = [p.mean_ms for p in points]
    lo, hi = math.log10(min(xs) / 1.5), math.log10(max(xs) * 1.5)
    ylo = math.floor((min(p.low for p in points) - 0.001) * 200) / 200
    yhi = math.ceil((max(p.high for p in points) + 0.001) * 200) / 200

    def X(v):
        return L + (math.log10(v) - lo) / (hi - lo) * (W - L - R)

    def Y(v):
        return T + (yhi - v) / (yhi - ylo) * (H - T - B)

    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Test NDCG against single-query latency for every Stage 3 re-ranker">']
    y = ylo
    while y <= yhi + 1e-9:
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}"/>'
                   f'<text class="tick" x="{L - 8}" y="{Y(y) + 4:.1f}" text-anchor="end">{y:.3f}</text>')
        y += 0.005
    for t in (10, 30, 100, 300, 1000, 3000):
        if 10 ** lo <= t <= 10 ** hi:
            out.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{H - B}"/>'
                       f'<text class="tick" x="{X(t):.1f}" y="{H - B + 18}" text-anchor="middle">{t:,} ms</text>')
    out.append(f'<text class="axis" x="{(L + W - R) / 2:.0f}" y="{H - 10}" text-anchor="middle">'
               f'mean single-query latency, log scale</text>')
    placed: list[tuple[float, float, float, float]] = []
    markers = [(X(p.mean_ms), Y(p.ndcg)) for p in points]
    for p in sorted(points, key=lambda q: (-q.ndcg, q.mean_ms)):
        x, yy = X(p.mean_ms), Y(p.ndcg)
        shape = _shape(p.name)
        out.append(f'<g class="pt"><title>{esc(p.label)}: NDCG {f4(p.ndcg)}, '
                   f'{ms(p.mean_ms)} ms</title>'
                   f'<line class="ci" x1="{x:.1f}" x2="{x:.1f}" y1="{Y(p.high):.1f}" y2="{Y(p.low):.1f}"/>'
                   f'{_marker(shape, x, yy)}</g>')
        text = "Qwen3 4B, served INT8" if shape == "served" else SHORT.get(p.name, p.name)
        width = len(text) * 6.6
        spots = [(dx, dy, a) for dy in (-6, 15, -20, 29, -34, 43)
                 for dx, a in ((10, "start"), (-10, "end"))]
        choice = None
        for strict in (True, False):
            for dx, dy, anchor in spots:
                x0 = x + dx if anchor == "start" else x + dx - width
                box = (x0, yy + dy - 12, x0 + width, yy + dy + 3)
                inside = L <= box[0] and box[2] <= W - R and T <= box[1] and box[3] <= H - B
                clash = any(not (box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1])
                            for b in placed)
                near = strict and any(box[0] - 4 < mx < box[2] + 4 and box[1] - 4 < my < box[3] + 4
                                      for mx, my in markers if (mx, my) != (x, yy))
                if inside and not clash and not near:
                    choice = (dx, dy, anchor, box)
                    break
            if choice:
                break
        if choice is None:
            raise ValueError(f"no room for the chart label {text!r}")
        dx, dy, anchor, box = choice
        placed.append(box)
        cls = "lbl strong" if shape == "served" else "lbl"
        out.append(f'<text class="{cls}" x="{x + dx:.1f}" y="{yy + dy:.1f}" '
                   f'text-anchor="{anchor}">{esc(text)}</text>')
    out.append("</svg>")
    return "".join(out)


# --- what each part is worth: a forest plot ---------------------------------------

def forest(rows) -> str:
    span = max(max(abs(r["delta"]["low"]), abs(r["delta"]["high"])) for r in rows) * 1.08

    def X(v):
        return 50 + v / span * 48

    lines = []
    for r in rows:
        d = r["delta"]
        tie = d["low"] <= 0 <= d["high"]
        cls = "tie" if tie else ("gain" if d["point"] > 0 else "loss")
        lines.append(
            f'<li class="f-row {cls}"><span class="f-name">{esc(r["name"])}</span>'
            f'<svg class="f-bar" viewBox="0 0 100 12" preserveAspectRatio="none" aria-hidden="true">'
            f'<line class="zero" x1="50" x2="50" y1="0" y2="12"/>'
            f'<line class="ci" x1="{X(d["low"]):.2f}" x2="{X(d["high"]):.2f}" y1="6" y2="6"/>'
            f'<line class="dot" x1="{X(d["point"]):.2f}" x2="{X(d["point"]):.2f}" y1="0.5" y2="11.5"/></svg>'
            f'<span class="f-num">{delta_text(d)}</span></li>'
        )
    return '<ol class="forest">' + "".join(lines) + "</ol>"


# --- the page ---------------------------------------------------------------------

RUNG_TEXT = {
    "hf_bf16_loop": "one pair at a time",
    "hf_bf16_batched": "the window as one batch (reference)",
    "vllm_bf16": "served by vLLM",
    "vllm_bf16_prefix": "+ prefix caching",
    "vllm_w8a8_prefix": "+ INT8 weights and activations",
    "vllm_w4a16_prefix": "4-bit weights instead",
}
RESULT_ROWS = (
    ("stage2+ce_bge_groups", "fp16", "bge-reranker-base, fine-tuned"),
    ("stage2+ce_bge_large", "fp16", "bge-reranker-large, fine-tuned"),
    ("stage2+qwen_point_zs", "bf16", "Qwen3-Reranker-0.6B, zero-shot"),
    ("stage2+qwen_point", "bf16", "Qwen3-Reranker-0.6B, fine-tuned"),
    ("stage2+qwen_list", "bf16", "Qwen3-1.7B listwise, fine-tuned"),
    ("stage2+qwen_point_4b", "bf16", "Qwen3-Reranker-4B, fine-tuned"),
)


def render(results_dir: Path = RESULTS) -> str:
    from esci_rerank.evaluation.report import ESCI_BASELINE

    rank = _load("llm-rank-test.json", results_dir)
    speed = _load("llm-speed.json", results_dir)
    load = _load("llm-load.json", results_dir)
    table = _load("ablation-table.json", results_dir)
    ex = _load("site-example.json", results_dir)
    arms = {a["name"]: a for a in rank["arms"]}
    comps = {(c["arm"], c["baseline"]): c["delta"] for c in rank["comparisons"]}
    deployed = speed["deployed_rung"]
    served = speed["test_quality"][deployed]
    served_lat = speed["rungs"][deployed]["latency"]
    hosted = rank["hosted_llm_latency"]
    best = "stage2+qwen_point_4b"

    rows = [
        f'<tr><td>coarse ranker, LambdaMART over 47 features</td><td class="n">{f4(arms["stage2"]["ndcg"]["point"])}</td>'
        f'<td class="n ci-cell">{interval(arms["stage2"]["ndcg"])}</td><td class="n">—</td></tr>'
    ]
    for arm, prec, name in RESULT_ROWS:
        a = arms[arm]
        lat = a["latency"][prec]
        rows.append(f'<tr><td>{esc(name)}</td><td class="n">{f4(a["ndcg"]["point"])}</td>'
                    f'<td class="n ci-cell">{interval(a["ndcg"])}</td><td class="n">{ms(lat["p95_ms"])} ms</td></tr>')
    rows.append(f'<tr class="best"><td>Qwen3-Reranker-4B, served by vLLM in INT8</td>'
                f'<td class="n">{f4(served["ndcg"]["point"])}</td><td class="n ci-cell">{interval(served["ndcg"])}</td>'
                f'<td class="n">{ms(served_lat["p95_ms"])} ms</td></tr>')
    rows.append(f'<tr><td>hosted LLM, listwise</td><td class="n">{f4(arms["stage2+llm"]["ndcg"]["point"])}</td>'
                f'<td class="n ci-cell">{interval(arms["stage2+llm"]["ndcg"])}</td>'
                f'<td class="n">{ms(hosted["single_ms"])} ms</td></tr>')

    rung_rows = []
    for name, text in RUNG_TEXT.items():
        r = speed["rungs"][name]
        lat = r["latency"]
        d = r["delta_vs_reference"]
        cls = ' class="best"' if name == deployed else ""
        rung_rows.append(
            f'<tr{cls}><td>{esc(text)}</td><td class="n">{ms(lat["p50_ms"])}</td><td class="n">{ms(lat["p95_ms"])}</td>'
            f'<td class="n">{r["throughput_qps"]:.1f}</td><td class="n ci-cell">{"reference" if d is None else delta_text(d)}</td></tr>'
        )
    load_rows = "".join(
        f'<tr><td class="n">{r["rate"]:g}</td><td class="n">{r["achieved_qps"]:.1f}</td>'
        f'<td class="n">{ms(r["latency"]["p50_ms"])}</td><td class="n">{ms(r["latency"]["p95_ms"])}</td></tr>'
        for r in load["rates"]
    )
    forest_rows = [r for r in table["ablations"] + table["findings"] if r["delta"]]
    vs_large, vs_hosted = comps[(best, "stage2+ce_bge_large")], comps[(best, "stage2+llm")]
    floor = rank["floor"]["mean"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESCI Product Re-Ranking</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231F7A55'/%3E%3Cpath d='M9 10h14M9 16h10M9 22h6' stroke='white' stroke-width='3' stroke-linecap='round'/%3E%3C/svg%3E">
<meta name="description" content="A multi-stage re-ranker for Amazon ESCI Task 1, measured against a computed floor with bootstrap intervals, ending in a fine-tuned 4B open model that ties a hosted LLM.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<header class="top">
  <a class="brand" href="./">ESCI re-ranking</a>
  <nav><a href="{REPO}/blob/main/docs/RESULTS.md">Writeup</a><a href="{REPO}">Code</a></nav>
</header>
<main>
<section class="hero">
  <h1>Re-ranking Amazon product search, ending in a 4B open model that ties a hosted LLM</h1>
  <p class="lede">A multi-stage re-ranker for ESCI Task 1: given a shopper's query and the products
  judged for it, put the relevant ones first. A gradient-boosted ranker orders each list; a
  fine-tuned model then re-orders its top 10. Every number on this page comes from a committed
  results file, measured once on all {rank["n_queries"]:,} test queries.</p>
  {rerank_figure(ex)}
</section>

<section>
  <h2>Results on the test split</h2>
  <p>NDCG over each query's whole judged list, with a 95% bootstrap interval over queries. A
  random order scores {f4(floor)}; the published fine-tuned cross-encoder baseline scores
  {ESCI_BASELINE:.4f}. Latency is the p95 one query adds, on an RTX A6000 unless noted.</p>
  <div class="scroll"><table class="results">
    <thead><tr><th>re-ranker on the coarse top 10</th><th class="n">NDCG</th><th class="n">95% interval</th><th class="n">p95</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
  <p>The fine-tuned 4B model beats <code>bge-reranker-large</code> by {delta_text(vs_large)} and
  ties the hosted LLM at {delta_text(vs_hosted)}, while answering in {ms(served_lat["p95_ms"])} ms
  at p95 against the API's {ms(hosted["single_ms"])} ms for a single query. The hosted figure is
  timed through its API; the coarse ranker's own latency is not part of this table.</p>
  <div class="scroll chart-wrap">{quality_chart(chart_points(results_dir))}</div>
  <p class="note"><span class="key"><svg viewBox="0 0 14 14"><circle class="m-ce" cx="7" cy="7" r="5"/></svg> cross-encoder</span>
  <span class="key"><svg viewBox="0 0 14 14"><rect class="m-llm" x="2.5" y="2.5" width="9" height="9"/></svg> open LLM, in-process</span>
  <span class="key"><svg viewBox="0 0 16 16"><path class="m-served" d="M8 1 L15 8 L8 15 L1 8 Z"/></svg> open LLM, served</span>
  <span class="key"><svg viewBox="0 0 14 14"><circle class="m-hosted" cx="7" cy="7" r="5"/></svg> hosted LLM</span></p>
  <p class="note">Vertical bars are 95% intervals. MiniLM was timed on a laptop RTX 3080, the hosted
  LLM through its API, and the rest on an RTX A6000.</p>
</section>

<section>
  <h2>How the pipeline works</h2>
  <ol class="stages">
    <li><h3>Coarse rank</h3><p>LightGBM LambdaMART over 47 query–product features: BM25 and
    dense text similarity, CLIP image similarity, text overlap, and prices, ratings, categories
    and attributes from the ESCI-S scrape. It orders each query's whole judged list:
    {f4(arms["stage2"]["ndcg"]["point"])}, level with the published baseline.</p></li>
    <li><h3>Fine rank</h3><p>A slower model re-orders the coarse top 10. Cross-encoders fine-tuned
    on the training folds, open LLMs fine-tuned the same way, and a hosted LLM asked for a
    permutation were each measured, and each lies above the baseline.</p></li>
    <li><h3>Blend</h3><p>Fixed-weight fusion, a learned combiner and a per-query selector over the
    stages. None beats the best stage it contains, so the deployed path stops at fine rank.</p></li>
  </ol>
</section>

<section>
  <h2>Serving the 4B model</h2>
  <p>One change a rung, each paired against the in-process reference on the validation queries.
  Latency is single-query on queries the server had not seen; throughput runs 16 queries at a
  time. No rung lost quality, so the fastest one, INT8, is deployed.</p>
  <div class="scroll"><table>
    <thead><tr><th>rung</th><th class="n">p50 ms</th><th class="n">p95 ms</th><th class="n">queries/s</th><th class="n">NDCG change vs reference</th></tr></thead>
    <tbody>{"".join(rung_rows)}</tbody>
  </table></div>
  <p>Under an open-loop load test with Poisson arrivals, the deployed rung holds a p95 under
  300 ms up to {load["max_rate_under_slo"]:g} queries a second on one GPU, about
  ${load["cost_per_1k_queries_usd"]:.4f} per 1,000 queries at the ${load["cost_basis"]["gpu_usd_per_hour"]:.2f}/hr
  the card was rented for. Past that the queue grows without bound.</p>
  <div class="scroll"><table class="narrow">
    <thead><tr><th class="n">offered q/s</th><th class="n">achieved</th><th class="n">p50 ms</th><th class="n">p95 ms</th></tr></thead>
    <tbody>{load_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>What each part is worth</h2>
  <p>Paired differences in test NDCG, "A − B", with 95% intervals. A bar crossing the centre line
  is a tie, and ties and losses are reported as results.</p>
  {forest(forest_rows)}
</section>

<section>
  <h2>What the data got wrong</h2>
  <ul class="pitfalls">
    <li><strong>The Hugging Face mirror is the wrong split.</strong> It encodes the large-version
    split, 185,361 test judgements instead of 181,701. The loader asserts the official counts on
    every load.</li>
    <li><strong>The official evaluation script swaps the Substitute and Complement gains,</strong>
    which moves the random floor by over three points, more than most method gains. The floor is
    computed on every report, never quoted.</li>
    <li><strong>About half the ESCI-S image URLs point at a retired CDN bucket.</strong> Stripping
    one path segment makes them resolve.</li>
    <li><strong>Books store their image under a different key,</strong> so reading only
    <code>image</code> silently drops every book.</li>
    <li><strong>"Has an image" and "has a price" are ranking signals by accident:</strong> they
    record the 2022 scrape failing on a page, which a live system would not know. The
    behavioural ablation controls for them.</li>
  </ul>
</section>

<section>
  <h2>How it was measured</h2>
  <p>Validation folds were carved from the training queries by query and frozen before anything
  was tuned. Every model, prompt and serving rung was chosen on them; each test measurement sits
  behind a <code>--final</code> flag and was run once. Every NDCG is reported beside a random
  floor computed on the same queries, with bootstrap intervals over queries, and a test fails if
  the writeup quotes a figure that no committed results file contains.</p>
  <p><a class="more" href="{REPO}/blob/main/docs/RESULTS.md">Read the full writeup</a>
  <a class="more" href="{REPO}">Browse the code</a></p>
</section>
</main>
<footer>
  <p>Data: the Amazon Shopping Queries Dataset (ESCI) and the ESCI-S enrichment. Generated by
  <code>python -m esci_rerank.reports.site</code> from <code>docs/results/</code>.</p>
</footer>
</body>
</html>
"""


CSS = """
:root{--paper:#EEF1F5;--panel:#F8FAFC;--ink:#1B2333;--soft:#525D70;--rule:#C8D0DB;
--e:#1F7A55;--s:#B7801A;--c:#6A55C9;--i:#98A2B1;--accent:#1F7A55;--best:#E1EFE7;
--font:"Schibsted Grotesk",system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#141A24;--panel:#1B2230;
--ink:#E6EAF1;--soft:#A3ADBD;--rule:#334055;--e:#4CC48F;--s:#E2AE47;--c:#A595F2;--i:#6E7889;
--accent:#4CC48F;--best:#1C3328}}
:root[data-theme="dark"]{--paper:#141A24;--panel:#1B2230;--ink:#E6EAF1;--soft:#A3ADBD;--rule:#334055;
--e:#4CC48F;--s:#E2AE47;--c:#A595F2;--i:#6E7889;--accent:#4CC48F;--best:#1C3328}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:400 17px/1.6 var(--font)}
a{color:inherit;text-decoration-color:var(--accent);text-underline-offset:3px;text-decoration-thickness:2px}
a:hover{color:var(--accent)}
a:focus-visible{outline:3px solid var(--accent);outline-offset:3px;border-radius:2px}
code{font-family:inherit;font-weight:500;background:var(--panel);padding:0 .25em;border-radius:3px}
.top,main,footer{max-width:1080px;margin:0 auto;padding:0 24px}
.top{display:flex;justify-content:space-between;align-items:baseline;padding-top:22px;padding-bottom:10px}
.brand{font-weight:800;text-decoration:none;letter-spacing:-.01em}
.top nav a{margin-left:20px;font-weight:500}
section{padding:40px 0;border-top:1px solid var(--rule)}
section.hero{border-top:0;padding-top:28px}
h1{font-size:clamp(30px,4.6vw,50px);line-height:1.08;font-weight:800;letter-spacing:-.025em;
max-width:20ch;margin:.2em 0 .4em}
h2{font-size:28px;line-height:1.2;font-weight:800;letter-spacing:-.015em;margin:0 0 .5em}
h3{font-size:18px;margin:0 0 .25em;font-weight:700}
p,li{max-width:68ch}
.lede{font-size:19px;color:var(--soft);max-width:62ch}
.rerank{margin:34px 0 0;background:var(--panel);border:1px solid var(--rule);border-radius:10px;
padding:22px 22px 18px}
.query{margin:0 0 14px;font-size:15px;color:var(--soft)}
.query q{font-size:24px;font-weight:700;color:var(--ink);quotes:"\\201C" "\\201D";margin-left:.35em}
.cols-head{display:grid;grid-template-columns:1fr 110px 1fr;font-size:14px;color:var(--soft)}
.cols-head p:last-child{grid-column:3}
.cols-head p{margin:0 0 8px}
.score{display:block;color:var(--ink);font-weight:700;font-size:16px}
.rerank-body{display:grid;grid-template-columns:1fr 110px 1fr}
.list{list-style:none;margin:0;padding:0}
.item{height:54px;display:grid;grid-template-columns:22px 26px 1fr;gap:8px;align-items:center;
border-bottom:1px solid var(--rule);max-width:none}
.item:last-child{border-bottom:0}
.rank{color:var(--soft);font-size:13px;text-align:right}
.title{font-size:14px;line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
overflow:hidden}
.chip{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:5px;font-size:13px;
font-weight:800;color:#fff}
.chip.l-E{background:var(--e)}.chip.l-S{background:var(--s)}.chip.l-C{background:var(--c)}
.chip.l-I{background:var(--i)}
.links{width:100%;height:540px;overflow:visible}
.link{fill:none;stroke-width:2.6;vector-effect:non-scaling-stroke;stroke-dasharray:240;
stroke-dashoffset:240;animation:draw .9s cubic-bezier(.3,.7,.2,1) forwards;animation-delay:calc(var(--i)*70ms + .2s)}
.link.l-E{stroke:var(--e)}.link.l-S{stroke:var(--s)}.link.l-C{stroke:var(--c)}.link.l-I{stroke:var(--i)}
@keyframes draw{to{stroke-dashoffset:0}}
@media (prefers-reduced-motion:reduce){.link{animation:none;stroke-dashoffset:0}}
figcaption{margin-top:16px;font-size:14px;color:var(--soft);max-width:none}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-bottom:10px;color:var(--ink)}
.legend .chip{width:20px;height:20px;font-size:11px;margin-right:4px}
.scroll{overflow-x:auto;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:15px}
table.narrow{width:auto;min-width:340px}
th{text-align:left;font-weight:500;color:var(--soft);font-size:13px;border-bottom:2px solid var(--ink);
padding:6px 12px 6px 0}
td{padding:9px 12px 9px 0;border-bottom:1px solid var(--rule)}
.n{text-align:right}
.ci-cell{color:var(--soft);white-space:nowrap}
tr.best td{background:var(--best);font-weight:700}
tr.best td:first-child{box-shadow:inset 3px 0 0 var(--accent);padding-left:10px}
.chart{width:100%;height:auto;margin:22px 0 4px;font-family:var(--font)}
.grid{stroke:var(--rule);stroke-width:1}
.tick,.axis{fill:var(--soft);font-size:12px}
.lbl{fill:var(--ink);font-size:12.5px}
.lbl.strong{font-weight:800;fill:var(--accent)}
.ci{stroke:var(--soft);stroke-width:1.5}
.m-ce{fill:var(--ink)}.m-llm{fill:var(--paper);stroke:var(--ink);stroke-width:2}
.m-served{fill:var(--accent)}.m-hosted{fill:none;stroke:var(--soft);stroke-width:2}
.note{font-size:14px;color:var(--soft);max-width:none}
.key{display:inline-flex;align-items:center;gap:6px;margin-right:16px;color:var(--ink)}
.key svg{width:14px;height:14px}
.stages{list-style:none;counter-reset:s;padding:0;display:grid;grid-template-columns:repeat(3,1fr);gap:28px}
.stages li{counter-increment:s;border-top:3px solid var(--ink);padding-top:12px}
.stages h3::before{content:counter(s) ". ";color:var(--soft);font-weight:500}
.stages p{font-size:15.5px;margin:0}
.forest{list-style:none;padding:0;margin:20px 0 0}
.f-row{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr) 190px;gap:16px;align-items:center;
padding:7px 0;border-bottom:1px solid var(--rule);max-width:none}
.f-name{font-size:15px}
.f-bar{width:100%;height:14px}
.f-bar .zero{stroke:var(--soft);stroke-width:1;vector-effect:non-scaling-stroke}
.f-bar .ci{stroke-width:2.2;vector-effect:non-scaling-stroke}
.f-bar .dot{stroke-width:3.5;vector-effect:non-scaling-stroke}
.f-row.gain .ci,.f-row.gain .dot{stroke:var(--e);fill:var(--e)}
.f-row.loss .ci,.f-row.loss .dot{stroke:var(--s);fill:var(--s)}
.f-row.tie .ci,.f-row.tie .dot{stroke:var(--i);fill:var(--i)}
.f-num{font-size:14px;text-align:right;white-space:nowrap}
.pitfalls{padding-left:1.1em}
.pitfalls li{margin:.45em 0}
.more{display:inline-block;margin-right:22px;font-weight:700}
footer{padding-top:28px;padding-bottom:48px;border-top:1px solid var(--rule);font-size:14px;color:var(--soft)}
@media (max-width:760px){body{font-size:16px}
.rerank{padding:16px 12px}.cols-head,.rerank-body{grid-template-columns:1fr 34px 1fr}
.item{grid-template-columns:20px 1fr;grid-template-rows:auto auto;height:54px;gap:0 6px}
.item .rank{display:none}.title{font-size:12px;-webkit-line-clamp:2}.chip{width:20px;height:20px;font-size:11px}
.query q{font-size:20px;display:block;margin:4px 0 0}
.stages{grid-template-columns:1fr;gap:18px}
.scroll table{min-width:600px}.scroll table.narrow{min-width:340px}
.chart-wrap .chart{min-width:720px}
.f-row{grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"name name" "bar num";gap:4px 12px}
.f-num{font-size:13px}
.f-name{grid-area:name}.f-bar{grid-area:bar}.f-num{grid-area:num}}
"""


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["build", "example"], default="build")
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and compare with the committed page")
    args = parser.parse_args()
    if args.command == "example":
        return _example()
    page = render()
    if args.check:
        if not PAGE.exists() or PAGE.read_text(encoding="utf-8") != page:
            raise SystemExit(f"{PAGE} is out of date; run python -m esci_rerank.reports.site")
        print(f"{PAGE} matches the committed results")
        return 0
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page, encoding="utf-8")
    (PAGE.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"-> {PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
