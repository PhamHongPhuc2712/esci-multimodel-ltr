"""Plan 13: every Stage 3 system on one quality-against-latency chart.

Read from committed results files only, like src.ablation_table: no number is
typed here. Every point names the runtime, precision and card it was timed under, and
the chart refuses one without a runtime or precision. The card matters here:
Plan 6 timed MiniLM on the laptop's RTX 3080, Plan 13 timed everything else on
a rented RTX A6000, and the hosted LLM ran on whatever its API runs on.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.report import ESCI_BASELINE

RESULTS = Path("docs/results")
FIGURE = Path("docs/figures/quality-vs-latency.svg")
LABELS = {
    "stage2+ce": "MiniLM cross-encoder",
    "stage2+ce_bge_groups": "bge-reranker-base",
    "stage2+ce_bge_large": "bge-reranker-large",
    "stage2+llm": "hosted LLM",
    "stage2+qwen_point_zs": "Qwen3-Reranker-0.6B, zero-shot",
    "stage2+qwen_point": "Qwen3-Reranker-0.6B, fine-tuned",
    "stage2+qwen_list": "Qwen3-1.7B listwise, fine-tuned",
    "stage2+qwen_point_4b": "Qwen3-Reranker-4B, fine-tuned",
}
# Plan 6's committed MiniLM timing has no precision or device field;
# load_reranker loads float32, and Plan 6 ran on the laptop.
MINILM_TIMING = ("hf", "fp32", "RTX 3080 laptop")
HOSTED_DEVICE = "hosted API"
UNRECORDED = "device unrecorded"


@dataclass(frozen=True)
class Point:
    name: str
    label: str
    ndcg: float
    low: float
    high: float
    mean_ms: float
    p95_ms: float | None
    runtime: str
    precision: str
    device: str
    source: str


def _load(results_dir: Path, name: str) -> dict:
    return json.loads((Path(results_dir) / name).read_text(encoding="utf-8"))


def _card(device: str | None) -> str:
    return (device or UNRECORDED).removeprefix("NVIDIA ")


def _from_profile(name, label, ndcg, prof, device, source) -> Point:
    for field in ("runtime", "precision"):
        if not prof.get(field):
            raise ValueError(f"{name} has no {field} on its latency; it cannot be compared")
    card = _card(device)
    return Point(name, f"{label} [{card}]", ndcg["point"], ndcg["low"], ndcg["high"],
                 prof["mean_ms"], prof.get("p95_ms"), prof["runtime"], prof["precision"],
                 card, source)


def chart_points(results_dir: Path = RESULTS) -> list[Point]:
    rank = _load(results_dir, "llm-rank-test.json")
    speed = _load(results_dir, "llm-speed.json")
    fine = _load(results_dir, "fine-rank-test-full.json")
    arms = {a["name"]: a for a in rank["arms"]}
    points: list[Point] = []

    minilm = next(a for a in fine["arms"] if a["name"] == "stage2+ce")["cost"]["latency"]
    runtime, precision, card = MINILM_TIMING
    points.append(_from_profile(
        "stage2+ce", f"{LABELS['stage2+ce']}, {precision}", arms["stage2+ce"]["ndcg"],
        {"mean_ms": minilm["single_ms"], "runtime": runtime, "precision": precision},
        card, "fine-rank-test-full.json"))

    for arm, entry in arms.items():
        if arm in ("stage2", "stage2+ce", "stage2+llm") or not entry.get("latency"):
            continue
        for prec, prof in entry["latency"].items():
            points.append(_from_profile(f"{arm}@{prec}", f"{LABELS.get(arm, arm)}, {prec}",
                                        entry["ndcg"], prof, rank.get("device"),
                                        "llm-rank-test.json"))

    hosted = rank["hosted_llm_latency"]
    points.append(_from_profile(
        "stage2+llm", LABELS["stage2+llm"], arms["stage2+llm"]["ndcg"],
        {"mean_ms": hosted["single_ms"], "runtime": hosted["runtime"],
         "precision": hosted["precision"]}, HOSTED_DEVICE, "llm-rank-test.json"))

    deployed = speed["deployed_rung"]
    points.append(_from_profile(
        f"deployed:{deployed}", f"{LABELS.get(rank.get('best_llm_arm'), 'LLM')}, served\n{deployed}",
        speed["test_quality"][deployed]["ndcg"], speed["rungs"][deployed]["latency"],
        speed["rungs"][deployed].get("device"), "llm-speed.json"))
    return points


FAMILIES = {
    "cross-encoder": "#2a78b5",
    "LLM, in-process": "#d1702a",
    "LLM, served (vLLM)": "#2c9a5a",
    "hosted LLM": "#7a7a7a",
}
# Where a label may sit around its point, tried in order: (dx, dy) in points.
LABEL_SPOTS = ((7, 3), (7, -3), (-7, 3), (-7, -3), (7, 12), (7, -12), (-7, 12), (-7, -12),
               (7, 24), (7, -24), (-7, 24), (-7, -24))


def _family(name: str) -> str:
    if name.startswith("deployed:"):
        return "LLM, served (vLLM)"
    if name == "stage2+llm":
        return "hosted LLM"
    if name.startswith("stage2+ce"):
        return "cross-encoder"
    return "LLM, in-process"


def _place_labels(fig, ax, points) -> None:
    """Each label in the first spot that stays inside the axes and covers no other
    label and no other point's marker."""
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    inside = ax.get_window_extent(renderer)
    markers = {}
    for p in points:
        x, y = ax.transData.transform((p.mean_ms, p.ndcg))
        markers[p.name] = Bbox([[x - 6, y - 6], [x + 6, y + 6]])
    placed = []
    for p in sorted(points, key=lambda q: (-q.ndcg, q.mean_ms)):
        obstacles = [b for name, b in markers.items() if name != p.name]
        for dx, dy in LABEL_SPOTS:
            text = ax.annotate(p.label, (p.mean_ms, p.ndcg), textcoords="offset points",
                               xytext=(dx, dy), fontsize=8,
                               ha="left" if dx > 0 else "right",
                               va="bottom" if dy > 0 else "top")
            box = text.get_window_extent(renderer)
            fits = (inside.x0 <= box.x0 and box.x1 <= inside.x1
                    and inside.y0 <= box.y0 and box.y1 <= inside.y1)
            if fits and not any(box.overlaps(other) for other in placed + obstacles):
                placed.append(box)
                break
            text.remove()
        else:
            raise ValueError(f"no free spot for the label of {p.name}")


def draw(points, out: Path, coarse_ndcg: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "plan-13"
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xscale("log")
    xs = [p.mean_ms for p in points]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    seen: set[str] = set()
    for p in points:
        family = _family(p.name)
        ax.errorbar(p.mean_ms, p.ndcg, yerr=[[p.ndcg - p.low], [p.high - p.ndcg]],
                    fmt="o", capsize=3, color=FAMILIES[family],
                    label=None if family in seen else family)
        seen.add(family)
    # Reference lines are labelled against the axis edge, in axes x-coordinates,
    # so the labels stay put whatever the data's x-range.
    edge = ax.get_yaxis_transform()
    ax.axhline(coarse_ndcg, linestyle=":", linewidth=1, color="0.4")
    ax.text(0.005, coarse_ndcg, " coarse ranker only", va="bottom", fontsize=8,
            color="0.3", transform=edge)
    ax.axhline(ESCI_BASELINE, linestyle="--", linewidth=1, color="0.4")
    ax.text(0.005, ESCI_BASELINE, " published target", va="top", fontsize=8,
            color="0.3", transform=edge)
    _place_labels(fig, ax, points)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_xlabel("single-query latency added by Stage 3, ms (log scale)")
    ax.set_ylabel("NDCG, all 8,956 test queries, 95% interval")
    ax.set_title("Stage 3 rerankers: quality against latency")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", metadata={"Date": None})
    plt.close(fig)


def _coarse(results_dir: Path) -> float:
    rank = _load(results_dir, "llm-rank-test.json")
    return next(a for a in rank["arms"] if a["name"] == "stage2")["ndcg"]["point"]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="regenerate to a temporary file and compare with the committed chart")
    args = parser.parse_args()
    points = chart_points()
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "chart.svg"
            draw(points, fresh, _coarse(RESULTS))
            if fresh.read_bytes() != FIGURE.read_bytes():
                raise SystemExit(f"{FIGURE} is out of date; regenerate it")
        print(f"{FIGURE} matches the committed results")
        return 0
    draw(points, FIGURE, _coarse(RESULTS))
    print(f"{len(points)} points -> {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
