"""The recall stage with a fine-tuned dense retriever and tuned fusion.

Plan 4's recall stage fused three untuned channels - BM25, an off-the-shelf
`all-MiniLM-L12-v2` and zero-shot CLIP - with RRF at k = 60 and equal weights:
Recall@100 0.5551 on fold 0. This report measures what replaces it:

  * a fourth channel, the retriever src.dense_finetune trains on folds 2/3/4;
  * RRF's k and per-channel weights, chosen by `--sweep` on fold 1 and never
    on the surface they are reported on.

Plan 4's own channels are kept, and the published fusion is re-scored beside
the new one on the same queries. On fold 0 it must reproduce recall.json, or
every delta here is against a baseline nobody published.

Three modes, all writing to docs/results/:

    --sweep                      fold 1   recall-fusion-sweep.json  (the choice)
    (default)                    fold 0   recall-tuned.json
    --split test --final         test     recall-tuned-test.json    (once)
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

CHANNELS: tuple[str, ...] = ("dense_ft", "dense_base", "bm25", "image")

# Plan 4's recall stage: the three untuned channels at RRF's default.
PUBLISHED_FUSION: tuple[int, tuple[float, ...]] = (60, (0.0, 1.0, 1.0, 1.0))
# The fine-tuned retriever swapped in for the off-the-shelf one, nothing else
# changed: what the fine-tune buys on its own.
SWAPPED_FUSION: tuple[int, tuple[float, ...]] = (60, (1.0, 0.0, 1.0, 1.0))

SWEEP_GRID: tuple[tuple[int, tuple[float, ...]], ...] = tuple(
    (k, (w_ft, w_base, 1.0, w_image))
    for k, w_ft, w_base, w_image in itertools.product(
        (60, 100, 200), (1.0, 1.5, 2.0), (0.0, 0.25, 0.5, 1.0), (0.0, 0.25, 0.5)
    )
)

DEPTH = 1000
KS: tuple[int, ...] = (10, 50, 100, 500, 1000)
SWEEP_FOLD = 1
REPORT_FOLD = 0

DEFAULT_FT_MODEL = Path("models/retriever/bge-base")
DEFAULT_FT_STORE = Path("data/embeddings/dense-ft")
SWEEP_OUT = Path("docs/results/recall-fusion-sweep.json")
OUT = {"train": Path("docs/results/recall-tuned.json"),
       "test": Path("docs/results/recall-tuned-test.json")}
PUBLISHED_RESULTS = Path("docs/results/recall.json")


def fuse_channels(
    hits: Mapping[str, Sequence[Sequence[str]]],
    *,
    k: int,
    weights: Sequence[float],
    limit: int = DEPTH,
) -> list[list[str]]:
    """RRF over the channels with a non-zero weight, in CHANNELS order.

    A zero-weight channel is left out rather than fused at weight 0, so the
    published fusion's channel order - and with it src.rrf's first-seen
    tie-break - is exactly Plan 4's.
    """
    from src.rrf import fuse_batches

    if len(weights) != len(CHANNELS):
        raise ValueError(f"{len(weights)} weights for {len(CHANNELS)} channels")
    used = [(name, w) for name, w in zip(CHANNELS, weights) if w > 0]
    return fuse_batches(
        [hits[name] for name, _ in used], k=k, weights=[w for _, w in used], limit=limit
    )


def best_config(rows: Sequence[Mapping]) -> Mapping:
    """The sweep row with the highest Recall@100, then Recall@1000; first listed on a tie."""
    if not rows:
        raise ValueError("an empty sweep has no best config")
    return max(
        enumerate(rows),
        key=lambda item: (item[1]["recall_at_100"], item[1]["recall_at_1000"], -item[0]),
    )[1]


def load_choice(path: Path = SWEEP_OUT) -> tuple[int, tuple[float, ...]]:
    """The fusion `--sweep` chose, refusing one chosen on any surface but fold 1."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no fusion choice at {path}; run python -m src.tuned_recall_report --sweep"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fold") != SWEEP_FOLD:
        raise ValueError(
            f"the fusion was chosen on fold {payload.get('fold')!r}; it must be chosen "
            f"on fold {SWEEP_FOLD}, never on the surface it is reported on"
        )
    chosen = payload["chosen"]
    return int(chosen["k"]), tuple(float(w) for w in chosen["weights"])


def check_published(
    measured: Mapping[int, float], published: Mapping[int, float], tol: float = 1e-4
) -> None:
    """Raise unless the re-scored published fusion matches recall.json."""
    for k, value in published.items():
        if abs(measured[k] - value) > tol:
            raise ValueError(
                f"the published fusion measures {measured[k]:.4f} at @{k} here against "
                f"{value:.4f} in {PUBLISHED_RESULTS}; the channels have drifted"
            )


def _published_recall() -> dict[int, float]:
    payload = json.loads(PUBLISHED_RESULTS.read_text(encoding="utf-8"))
    arm = next(a for a in payload["arms"] if a["name"] == "dense+bm25+image")
    return {k: arm["recall"][str(k)]["recall"] for k in (100, 1000)}


def _hits(split: str, folds, ft_model: Path, ft_store: Path):
    """Every channel's top DEPTH for the scope's queries, plus the relevant sets."""
    import pandas as pd

    from src.bm25_index import open_channel as open_bm25
    from src.dense_embed import open_channel as open_dense
    from src.image_channel import open_channel as open_image
    from src.recall import DEFAULT_JUDGEMENTS, load_queries, relevant_sets

    queries = load_queries(split, folds=folds)
    judgements = pd.read_parquet(DEFAULT_JUDGEMENTS)
    judgements = judgements.loc[judgements["query_id"].isin(set(queries["query_id"]))]
    relevant = relevant_sets(judgements, relevance="E")
    texts = queries["query"].astype(str).tolist()

    openers = {
        "dense_ft": lambda: open_dense(store_dir=ft_store, model_name=str(ft_model)),
        "dense_base": lambda: open_dense(),
        "bm25": lambda: open_bm25(),
        "image": lambda: open_image(),
    }
    hits = {}
    for name in CHANNELS:
        channel = openers[name]()
        hits[name] = channel.search(texts, DEPTH)
        del channel
        print(f"  {name}: searched {len(texts):,} queries")
    return queries["query_id"].tolist(), hits, relevant


def _main() -> int:
    from src.recall_report import compare, evaluate_arm, format_table

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--final", action="store_true", help="required with --split test")
    parser.add_argument("--sweep", action="store_true",
                        help=f"choose the fusion on fold {SWEEP_FOLD}")
    parser.add_argument("--ft-model", type=Path, default=DEFAULT_FT_MODEL)
    parser.add_argument("--ft-store", type=Path, default=DEFAULT_FT_STORE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.sweep and args.split != "train":
        raise SystemExit(f"--sweep chooses the fusion on fold {SWEEP_FOLD}; it never touches test")
    if args.split == "test" and not args.final:
        raise SystemExit(
            "refusing to touch the test split without --final. The retriever, k and "
            "the weights are all chosen on the training folds; test is measured once."
        )

    if args.sweep:
        ids, hits, relevant = _hits("train", [SWEEP_FOLD], args.ft_model, args.ft_store)
        rows = []
        for k, weights in (PUBLISHED_FUSION, *SWEEP_GRID):
            arm = evaluate_arm("fusion", dict(zip(ids, fuse_channels(hits, k=k, weights=weights))),
                               relevant, ks=(100, 1000))
            rows.append({"k": k, "weights": list(weights),
                         "recall_at_100": arm.table[100].mean,
                         "recall_at_1000": arm.table[1000].mean})
            print(f"  k={k:3d} weights {weights}: @100 {rows[-1]['recall_at_100']:.4f} "
                  f"@1000 {rows[-1]['recall_at_1000']:.4f}")
        published, grid = rows[0], rows[1:]
        chosen = best_config(grid)
        out = args.out or SWEEP_OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "fold": SWEEP_FOLD, "n_queries": len(ids), "channels": list(CHANNELS),
            "ft_model": str(args.ft_model), "published": published,
            "chosen": chosen, "grid": grid,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nchosen on fold {SWEEP_FOLD}: k={chosen['k']} weights {chosen['weights']} "
              f"(@100 {chosen['recall_at_100']:.4f}, published {published['recall_at_100']:.4f}) "
              f"-> {out}")
        return 0

    k, weights = load_choice()
    folds = None if args.split == "test" else [REPORT_FOLD]
    ids, hits, relevant = _hits(args.split, folds, args.ft_model, args.ft_store)
    as_map = lambda ranked: dict(zip(ids, ranked))  # noqa: E731
    arms = [evaluate_arm(name, as_map(hits[name]), relevant, ks=KS) for name in CHANNELS]
    fusions = {
        "published fusion": PUBLISHED_FUSION,
        "fine-tuned dense swapped in": SWAPPED_FUSION,
        "tuned fusion": (k, weights),
    }
    for name, (fk, fw) in fusions.items():
        arms.append(evaluate_arm(name, as_map(fuse_channels(hits, k=fk, weights=fw)), relevant, ks=KS))
    by_name = {arm.name: arm for arm in arms}
    print()
    print(format_table(arms))

    if args.split == "train":
        check_published({kk: by_name["published fusion"].table[kk].mean for kk in (100, 1000)},
                        _published_recall())

    pairs = [("dense_ft", "dense_base"),
             ("fine-tuned dense swapped in", "published fusion"),
             ("tuned fusion", "fine-tuned dense swapped in"),
             ("tuned fusion", "published fusion")]
    comparisons = [compare(by_name[a], by_name[b], k=kk) for a, b in pairs for kk in (100, 1000)]
    print()
    for row in comparisons:
        d = row["delta"]
        print(f"  {row['arm']:28s} - {row['baseline']:28s} @{row['k']:<4d} "
              f"{d['point']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]  "
              f"{'significant' if row['significant'] else 'ties'}")

    from src.dense_finetune import RETRIEVER_RECORD
    record_path = Path(args.ft_model) / RETRIEVER_RECORD
    payload = {
        "split": args.split,
        "scope": f"fold {REPORT_FOLD}" if args.split == "train" else "test, all queries",
        "relevance": "E",
        "n_queries": len(ids),
        "retrieval_depth": DEPTH,
        "channels": list(CHANNELS),
        "fusion": {"k": k, "weights": list(weights), "chosen_on": f"fold {SWEEP_FOLD}"},
        "arms": [arm.to_dict() for arm in arms],
        "comparisons": comparisons,
        "retriever": {
            "model": str(args.ft_model), "store": str(args.ft_store),
            "training": json.loads(record_path.read_text(encoding="utf-8"))
            if record_path.exists() else None,
        },
    }
    out = args.out or OUT[args.split]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
