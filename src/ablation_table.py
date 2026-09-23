"""PROJECT_SPEC.md §6 as one table, assembled from the committed results.

Seven ablations, owned by four plans, measured on **three different query
populations with two different metrics**:

    1, 2   Recall@100   4,130 fold-0 queries, full 1.2M-product corpus
    3,4,5,7  NDCG       8,956 test queries
    6        NDCG       2,000-query frozen test sample

Printed as seven rows of one "delta" column with no scope, those read as
comparable, and the summary sentence anyone writes from the table inherits the
error. So `scope` and `n_queries` are required on every row, the renderer
prints them, and the table carries the sentence that says they are not
comparable.

Nothing here computes a number. Every value is read from a committed JSON, so
the table cannot drift from the files it claims to summarise.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path("docs/results")
DEFAULT_OUT = DEFAULT_DIR / "ablation-table.json"

REQUIRED_FILES: tuple[str, ...] = (
    "recall",
    "coarse-rank-test",
    "fine-rank-test",
    "blend-test",
)

_RECALL, _COARSE, _FINE = (
    "docs/results/recall.json",
    "docs/results/coarse-rank-test.json",
    "docs/results/fine-rank-test.json",
)

# One row per measured comparison, named "A − B" so the sign reads the same
# way on every row: negative means B won.
#
# Ablation 2 is a ladder - dense, +BM25, +image - and each rung's claim is
# about the rung below it; one fusion-vs-BM25 row would credit the image
# channel with the whole fusion gain (+0.0661) when its step is +0.0048.
# Ablations 5 and 6 each measured two comparisons as well, and showing one
# hides the other: 5a is §6's literal "pointwise classifier", 6a the
# fine-tuned cross-encoder beside 6b's LLM.
ABLATIONS: tuple[dict, ...] = (
    {"number": 1, "label": "1", "name": "LLM-rewritten − raw query", "metric": "Recall@100",
     "scope": "fold 0, full corpus", "source": _RECALL, "owner": "Plan 4"},
    {"number": 2, "label": "2a", "name": "dense+BM25 − dense-only (RRF)", "metric": "Recall@100",
     "scope": "fold 0, full corpus", "source": _RECALL, "owner": "Plan 4"},
    {"number": 2, "label": "2b", "name": "dense+BM25+image − dense+BM25 (RRF)", "metric": "Recall@100",
     "scope": "fold 0, full corpus", "source": _RECALL, "owner": "Plan 4"},
    {"number": 3, "label": "3", "name": "text+image − text-only features", "metric": "NDCG",
     "scope": "test", "source": _COARSE, "owner": "Plan 5"},
    {"number": 4, "label": "4", "name": "behavioural values − text+presence indicators", "metric": "NDCG",
     "scope": "test", "source": _COARSE, "owner": "Plan 5"},
    {"number": 5, "label": "5a", "name": "pointwise classifier − lambdarank", "metric": "NDCG",
     "scope": "test", "source": _COARSE, "owner": "Plan 5"},
    {"number": 5, "label": "5b", "name": "pointwise regression − lambdarank", "metric": "NDCG",
     "scope": "test", "source": _COARSE, "owner": "Plan 5"},
    {"number": 6, "label": "6a", "name": "coarse+cross-encoder − coarse-only", "metric": "NDCG",
     "scope": "test sample", "source": _FINE, "owner": "Plan 6"},
    {"number": 6, "label": "6b", "name": "coarse+LLM listwise − coarse-only", "metric": "NDCG",
     "scope": "test sample", "source": _FINE, "owner": "Plan 6"},
    {"number": 7, "label": "7", "name": "learned fusion − fixed global weight", "metric": "NDCG",
     "scope": "test", "source": _COARSE, "owner": "Plan 5"},
)

_SCOPE_WARNING = (
    "> These rows are **not comparable to each other**: they were measured on "
    "three different query populations with two different metrics. The scope "
    "and n columns say which."
)


@dataclass(frozen=True)
class Row:
    number: int
    label: str
    name: str
    metric: str
    scope: str
    n_queries: int
    delta: Mapping[str, float] | None
    significant: bool | None
    source: str

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "label": self.label,
            "name": self.name,
            "metric": self.metric,
            "scope": self.scope,
            "n_queries": self.n_queries,
            "delta": dict(self.delta) if self.delta else None,
            "significant": self.significant,
            "source": self.source,
        }


def load_results(directory: Path = DEFAULT_DIR) -> dict[str, dict]:
    """Every committed results file, keyed by stem."""
    directory = Path(directory)
    loaded: dict[str, dict] = {}
    for stem in REQUIRED_FILES:
        path = directory / f"{stem}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; the ablation table reads the committed "
                "results rather than recomputing them, so every plan's file "
                "must be present"
            )
        loaded[stem] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(directory.glob("*.json")):
        loaded.setdefault(path.stem, json.loads(path.read_text(encoding="utf-8")))
    return loaded


def by_arm(rows: Sequence[Mapping], arm: str) -> Mapping:
    """The comparison row for `arm`, or a KeyError naming it.

    Never by position: the plan as written read `5_objective[0]`, which is the
    regression arm, under the name of §6's pointwise *classifier*.
    """
    for row in rows:
        if row.get("arm") == arm:
            return row
    raise KeyError(
        f"no comparison for arm {arm!r} among {[r.get('arm') for r in rows]}"
    )


def build_rows(rows: Sequence[Row]) -> list[Row]:
    """Validate and order the rows. Scope and n are not optional."""
    for row in rows:
        if not row.scope:
            raise ValueError(
                f"ablation {row.number} has no scope; a table without one "
                "invites reading seven incomparable numbers as comparable"
            )
        if not row.n_queries:
            raise ValueError(
                f"ablation {row.number} has no n_queries; an interval without "
                "its sample size cannot be judged"
            )
    return sorted(rows, key=lambda row: (row.number, row.label))


def format_markdown(rows: Sequence[Row]) -> str:
    """The §6 table, with the scope columns that keep it honest."""
    lines = [
        "| # | ablation | metric | scope | n | delta | |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row.delta is None:
            cell, verdict = "—", ""
        else:
            cell = (f"{row.delta['point']:+.4f} "
                    f"[{row.delta['low']:+.4f}, {row.delta['high']:+.4f}]")
            verdict = "significant" if row.significant else "ties"
        lines.append(
            f"| {row.label} | {row.name} | {row.metric} | {row.scope} | "
            f"{row.n_queries:,} | {cell} | {verdict} |"
        )
    return "\n".join(lines) + "\n\n" + _SCOPE_WARNING


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    recall, coarse = results["recall"], results["coarse-rank-test"]
    fine, blend = results["fine-rank-test"], results["blend-test"]

    by_label = {a["label"]: dict(a) for a in ABLATIONS}

    # 1 - the rewritten query against the same fused channel. Plan 4 already
    # computed this as a *paired* bootstrap and stored it under
    # `ablation_1_rewrite`; subtracting two vs-BM25 intervals instead would be
    # the same interval arithmetic that over-corrected Plan 5's Ablation 4 by
    # 0.0143. Read the file, never re-derive.
    by_label["1"]["delta"] = recall["ablation_1_rewrite"]["delta"]
    by_label["1"]["significant"] = recall["ablation_1_rewrite"]["significant"]
    by_label["1"]["n_queries"] = recall["n_queries"]

    # 2 - the ladder, rung by rung, from `ablation_2_ladder`. Never from
    # `comparisons`, whose fusion-vs-BM25 row credits the image channel with
    # the whole fusion gain - the over-credit Plan 4 stored the ladder to avoid.
    for label, arm in [("2a", "dense+bm25"), ("2b", "dense+bm25+image")]:
        entry = by_arm(recall["ablation_2_ladder"], arm)
        by_label[label]["delta"] = entry["delta"]
        by_label[label]["significant"] = entry["significant"]
        by_label[label]["n_queries"] = recall["n_queries"]

    for label, key in [("3", "3_text_vs_image"), ("7", "7_learned_fusion")]:
        entry = coarse["ablations"][key]
        by_label[label]["delta"] = entry["delta"]
        by_label[label]["significant"] = entry["significant"]
        by_label[label]["n_queries"] = coarse["n_queries"]

    behavioural = coarse["ablations"]["4_behavioural"]
    by_label["4"]["delta"] = behavioural["values_over_indicators"]
    by_label["4"]["significant"] = behavioural["values_over_indicators_significant"]
    by_label["4"]["n_queries"] = coarse["n_queries"]

    # 5 - each pointwise baseline against lambdarank, found by name.
    for label, arm in [("5a", "full/pointwise_class"), ("5b", "full/pointwise_regression")]:
        entry = by_arm(coarse["ablations"]["5_objective"], arm)
        by_label[label]["delta"] = entry["delta"]
        by_label[label]["significant"] = entry["significant"]
        by_label[label]["n_queries"] = coarse["n_queries"]

    # 6 - each fine-tuned reranker against coarse-only, on the frozen sample.
    for label, arm in [("6a", "stage2+ce"), ("6b", "stage2+llm")]:
        entry = by_arm(fine["ablation_6"], arm)
        by_label[label]["delta"] = entry["delta"]
        by_label[label]["significant"] = entry["significant"]
        by_label[label]["n_queries"] = fine["n_queries"]

    rows = build_rows([
        Row(number=a["number"], label=a["label"], name=a["name"],
            metric=a["metric"], scope=a["scope"], n_queries=a["n_queries"],
            delta=a["delta"], significant=a["significant"], source=a["source"])
        for a in by_label.values()
    ])
    table = format_markdown(rows)
    print(table)

    payload = {
        "ablations": [row.to_dict() for row in rows],
        "markdown": table,
        "stage4": {
            "best_single_stage": blend["best_single_stage"],
            "any_blend_beats_best_single": blend["any_blend_beats_best_single"],
            "scope": blend["scope"],
            "n_queries": blend["n_queries"],
            "arms": {arm["name"]: arm["ndcg"] for arm in blend["arms"]},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
